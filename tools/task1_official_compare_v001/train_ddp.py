"""FP32 DDP training with an unchanged global batch and exact episode split."""
import common
import argparse
import copy
import faulthandler
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from datetime import timedelta
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from diffusion_policy.model.diffusion.ema_model import EMAModel


class LossModule(torch.nn.Module):
    def __init__(self, policy):
        super().__init__();self.policy=policy
    def forward(self,batch):
        return self.policy.compute_loss(batch)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    ap.add_argument('--stop-after',type=int);ap.add_argument('--smoke',action='store_true');ap.add_argument('--resume-check',action='store_true')
    faulthandler.register(signal.SIGUSR1,all_threads=True)
    args=ap.parse_args();cfg=json.loads(Path(args.config).read_text())
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30),device_id=torch.device('cuda',local))
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    torch.backends.cudnn.benchmark=True
    resume_checkpoint=Path(cfg['output_dir'])/'checkpoints/last.pt'
    if args.resume_check:
        assert resume_checkpoint.exists(),resume_checkpoint
        cfg['output_dir']=str(Path(cfg['output_dir'])/'resume_probe')
    if args.smoke:
        cfg['output_dir']=str(Path(cfg['output_dir'])/'ddp_smoke');cfg['initial_checkpoint']=None
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    ckpts=out/'checkpoints';ckpts.mkdir(exist_ok=True)
    limit=2 if args.smoke else min(cfg['updates'],args.stop_after or cfg['stage_updates'])
    ds=common.Sequences(cfg,'train');val=common.Sequences(cfg,'validation')
    kwargs=dict(num_workers=2,pin_memory=True,persistent_workers=False,prefetch_factor=2,multiprocessing_context='spawn')
    model=common.build(cfg);model.set_normalizer(ds.normalizer());model.cuda()
    ema_model=copy.deepcopy(model) if rank==0 else None
    ema=EMAModel(ema_model,power=.75) if rank==0 else None
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=(.95,.999),eps=1e-8,weight_decay=1e-6)
    def factor(s):
        if s<cfg['warmup_steps']:return s/max(1,cfg['warmup_steps'])
        f=(s-cfg['warmup_steps'])/(cfg['scheduler_total_steps']-cfg['warmup_steps'])
        return .5*(1+math.cos(math.pi*min(f,1)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,factor)
    step=0;epoch=0;batch_index=-1;best=float('inf');train_seconds=0.
    last=ckpts/'last.pt';resume=resume_checkpoint if args.resume_check else (None if args.smoke else (last if last.exists() else cfg.get('initial_checkpoint')))
    restored_rng=None;previous_world_size=None
    if resume:
        payload=torch.load(resume,map_location='cpu',weights_only=False)
        assert payload['config']['official_policy']==cfg['official_policy']
        assert payload['config']['batch_size']==cfg['batch_size']==64
        model.load_state_dict(payload['model']);model.cuda()
        if rank==0:
            ema_model.load_state_dict(payload['ema']);ema_model.cuda()
            ema.optimization_step=payload['ema_optimization_step']
        opt.load_state_dict(payload['optimizer']);sched.load_state_dict(payload['scheduler'])
        step=payload['step'];epoch=payload['epoch'];batch_index=payload['batch_index']
        best=payload['best_val_loss'];train_seconds=payload['train_seconds']
        if 'rank_rng' in payload:
            previous_world_size=len(payload['rank_rng'])
            if rank<previous_world_size:restored_rng=payload['rank_rng'][rank]
        elif rank==0:
            restored_rng={k:payload[k] for k in ('torch_rng','cuda_rng','numpy_rng','python_rng')}
        del payload
    # Device-tracking sentinels have no values and do not participate in the loss.
    for name, parameter in model.named_parameters():
        if name.endswith('_dummy_variable') and parameter.numel()==0:
            parameter.requires_grad_(False)
    ddp=DistributedDataParallel(LossModule(model),device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True)
    torch.manual_seed(cfg['seed']+rank);np.random.seed(cfg['seed']+rank);random.seed(cfg['seed']+rank)
    if restored_rng:
        torch.set_rng_state(restored_rng['torch_rng']);torch.cuda.set_rng_state(restored_rng['cuda_rng'])
        np.random.set_state(restored_rng['numpy_rng']);random.setstate(restored_rng['python_rng'])
    start_step=step
    if args.resume_check:limit=start_step+2
    if rank==0:
        common.write_json(out/'config.json',cfg)
        common.write_json(out/'training_start.json',dict(step=step,world_size=world,device=torch.cuda.get_device_name(),
            global_batch=64,local_batch=64//world,precision='float32',previous_world_size=previous_world_size,
            rng_streams_resharded=previous_world_size is not None and previous_world_size!=world,train_windows=len(ds),validation_windows=len(val),
            train_episodes=96,validation_episodes=11,parameters=sum(p.numel() for p in model.parameters()),
            lossless_cache_enabled=True,initial_checkpoint=str(resume) if resume else None))
    log=(out/'metrics.jsonl').open('a',buffering=1) if rank==0 else None
    def record(row):
        if rank==0:log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
    def rng():
        return dict(torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
    def save(path):
        states=[None]*world;dist.all_gather_object(states,rng());started=time.monotonic()
        if rank==0:
            payload=dict(config=cfg,model=model.state_dict(),ema=ema_model.state_dict(),optimizer=opt.state_dict(),
                scheduler=sched.state_dict(),step=step,epoch=epoch,batch_index=batch_index,best_val_loss=best,
                train_seconds=train_seconds,ema_optimization_step=ema.optimization_step,rank_rng=states)
            tmp=path.with_suffix('.partial');torch.save(payload,tmp);tmp.replace(path)
            record(dict(event='checkpoint',step=step,path=str(path),seconds=time.monotonic()-started))
        dist.barrier()
    def validate():
        nonlocal best
        dist.barrier()
        if rank==0:
            t=time.monotonic();ema_model.eval();total=0.;count=0
            loader=DataLoader(val,batch_size=64,shuffle=False,generator=torch.Generator().manual_seed(1729),**kwargs)
            with torch.random.fork_rng(devices=[local]),torch.no_grad():
                torch.manual_seed(1729)
                for batch in loader:
                    batch=common.cuda_batch(batch,cfg);loss=ema_model.compute_loss(batch)
                    n=len(batch['action']);total+=float(loss)*n;count+=n
            ema_model.train();value=total/count;improved=value<best;best=min(best,value)
            record(dict(event='validation',step=step,ema_val_loss=value,windows=count,seconds=time.monotonic()-t))
            flag=torch.tensor(int(improved),device='cuda')
        else:flag=torch.zeros((),device='cuda',dtype=torch.int64)
        dist.broadcast(flag,src=0)
        if flag.item():save(ckpts/'best.pt')
    interrupted=False;stop_requested=False
    def stop(*_):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    stable=[];window=[];resume_batch=batch_index
    while step<limit:
        ids=torch.randperm(len(ds),generator=torch.Generator().manual_seed(cfg['seed']+epoch)).tolist()
        global_batches=[ids[i:i+64] for i in range(0,len(ids),64)]
        batches=global_batches[resume_batch+1:]
        local_batches=[np.array_split(np.asarray(b),world)[rank].tolist() for b in batches]
        loader=DataLoader(ds,batch_sampler=local_batches,generator=torch.Generator().manual_seed(cfg['seed']+epoch+rank),**kwargs)
        iterator=iter(loader)
        for global_ids in batches:
            if step>=limit:break
            t=time.monotonic();batch=next(iterator);wait=time.monotonic()-t
            relative=step-start_step;measured=200<=relative<700
            events=[torch.cuda.Event(enable_timing=True) for _ in range(6)] if measured else None
            if events:events[0].record()
            batch=common.cuda_batch(batch,cfg)
            if events:events[1].record()
            loss=ddp(batch)
            weighted=loss*(len(batch['action'])*world/len(global_ids))
            if events:events[2].record()
            weighted.backward()
            if events:events[3].record()
            opt.step();opt.zero_grad(set_to_none=True);sched.step()
            if events:events[4].record()
            if rank==0:ema.step(model)
            if events:events[5].record()
            values=torch.stack((weighted.detach(),torch.tensor(float(stop_requested),device='cuda')))
            dist.all_reduce(values)
            loss_value=float(values[0]/world);interrupted=bool(values[1].item())
            if not math.isfinite(loss_value):raise RuntimeError(f'Nonfinite loss at {step+1}')
            torch.cuda.synchronize();elapsed=time.monotonic()-t
            step+=1;batch_index+=1;train_seconds+=elapsed
            window.append((elapsed,wait,loss_value))
            if rank==0 and measured:
                stable.append(dict(update_ms=elapsed*1000,data_wait_ms=wait*1000,
                    **dict(zip(('h2d_and_preprocess_ms','forward_ms','backward_and_ddp_ms','optimizer_ms','ema_ms'),
                               [events[i].elapsed_time(events[i+1]) for i in range(5)]))))
            if step%100==0:
                arr=np.asarray(window);window=[];logging=time.monotonic()
                row=dict(event='train',step=step,epoch=epoch+(batch_index+1)/cfg['steps_per_epoch'],
                    stage_updates=limit,full_budget_updates=cfg['updates'],loss=float(arr[:,2].mean()),
                    lr=sched.get_last_lr()[0],update_ms=float(arr[:,0].mean()*1000),data_wait_ms=float(arr[:,1].mean()*1000),
                    training_seconds=train_seconds,eta_stage_training_seconds=float(arr[:,0].mean()*(limit-step)),
                    world_size=world,global_batch=64,gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20)
                record(row)
                if rank==0:
                    common.write_json(out/'progress.json',row)
                    if measured:stable[-1]['logging_ms']=(time.monotonic()-logging)*1000
            if rank==0 and step-start_step==700:
                common.write_json(out/'timing.json',dict(warmup=200,stable_updates=500,start_step=start_step,
                    means={key:float(np.mean([x.get(key,0.) for x in stable])) for key in (*stable[0].keys(),'logging_ms')},
                    samples=stable,includes_validation=False,includes_checkpoint=False,world_size=world,global_batch=64))
            if not (args.smoke or args.resume_check) and (step%cfg['validate_every']==0 or step==limit):validate()
            if step%cfg['checkpoint_every']==0 or step==limit or interrupted:save(last)
            if interrupted:break
        del iterator,loader
        if step>=limit or interrupted:break
        epoch+=1;batch_index=-1;resume_batch=-1
    if rank==0:
        if args.smoke or args.resume_check:
            payload=torch.load(last,map_location='cpu',weights_only=False)
            probe=common.build(cfg);probe.load_state_dict(payload['ema']);probe.cuda().eval();del payload
            sample=ds[0];obs=common.cuda_obs({k:v[None] for k,v in sample['obs'].items()},cfg)
            with torch.no_grad():prediction=probe.predict_action(obs)['action']
            assert prediction.shape==(1,8,7) and torch.isfinite(prediction).all()
            common.write_json(out/('resume_result.json' if args.resume_check else 'smoke_result.json'),dict(passed=True,steps=step,start_step=start_step,updates=step-start_step,previous_world_size=previous_world_size,world_size=world,global_batch=64,
                prediction_shape=list(prediction.shape),gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20))
            print('TASK1_DDP_RESUME_OK' if args.resume_check else 'TASK1_DDP_SMOKE_OK',flush=True)
        marker='paused.json' if interrupted else f'segment_{step:07d}.json'
        if not interrupted:
            snapshot=ckpts/f'step_{step:07d}.pt'
            if not snapshot.exists():snapshot.hardlink_to(last)
        common.write_json(out/marker,dict(step=step,training_seconds=train_seconds,best_val_loss=best,
            checkpoint=str(last),world_size=world,full_budget_complete=step==cfg['updates']))
        log.close()
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
