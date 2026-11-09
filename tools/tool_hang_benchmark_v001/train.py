"""Train the repository DP architecture on all ToolHang PH demonstrations."""
import common
import argparse
import copy
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.common.sampler import SequenceSampler


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--stop-after',type=int)
    args=parser.parse_args()
    cfg=common.DEFAULT.copy()
    if args.config:cfg.update(json.loads(Path(args.config).read_text()))
    limit=min(cfg['updates'],args.stop_after or cfg['updates'])
    amp=cfg['precision']=='bfloat16'
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    ckptdir=out/'checkpoints';ckptdir.mkdir(exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    assert torch.cuda.is_available(), 'This experiment requires CUDA'
    data_started=time.monotonic()
    ds=common.dataset(cfg);val=ds.get_validation_dataset()
    val.sampler=SequenceSampler(val.replay_buffer,sequence_length=cfg['horizon'],
        pad_before=cfg['n_obs_steps']-1,pad_after=cfg['n_action_steps']-1,
        episode_mask=val.train_mask,key_first_k={k:cfg['n_obs_steps'] for k in common.SHAPE['obs']})
    kwargs=dict(batch_size=cfg['batch_size'],num_workers=cfg['num_workers'],
                pin_memory=True,persistent_workers=cfg.get('persistent_workers',True),prefetch_factor=2)
    val_loader=DataLoader(val,shuffle=False,**kwargs)
    model=common.build(cfg);model.set_normalizer(ds.get_normalizer());model.cuda()
    ema_model=copy.deepcopy(model)
    ema=EMAModel(ema_model,power=.75)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=(.95,.999),weight_decay=1e-6)
    def lr_factor(s):
        if s<cfg['warmup_steps']:return s/max(1,cfg['warmup_steps'])
        total=cfg.get('scheduler_total_steps',cfg['updates'])
        frac=(s-cfg['warmup_steps'])/(total-cfg['warmup_steps'])
        return .5*(1+math.cos(math.pi*min(frac,1)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_factor)
    step=0;epoch=0;batch_index=-1;best=float('inf');train_seconds=0.
    last=ckptdir/'last.pt'
    if last.exists():
        payload=torch.load(last,map_location='cpu',weights_only=False)
        assert payload['config']==cfg
        model.load_state_dict(payload['model']);ema_model.load_state_dict(payload['ema'])
        # Normalizer loading recreates its tensors on the checkpoint device.
        model.cuda();ema_model.cuda()
        opt.load_state_dict(payload['optimizer']);sched.load_state_dict(payload['scheduler'])
        step=payload['step'];epoch=payload['epoch'];batch_index=payload['batch_index']
        best=payload['best_val_loss'];train_seconds=payload['train_seconds']
        ema.optimization_step=payload['ema_optimization_step']
        torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state(payload['cuda_rng'])
        np.random.set_state(payload['numpy_rng']);random.setstate(payload['python_rng'])
        del payload
    info=dict(event='training_started',device=torch.cuda.get_device_name(),
              parameters=sum(p.numel() for p in model.parameters()),train_episodes=int(ds.train_mask.sum()),
              validation_episodes=int(val.train_mask.sum()),train_windows=len(ds),validation_windows=len(val),
              dataset_load_seconds=time.monotonic()-data_started,step=step,config=cfg)
    (out/'training_start.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
    log=(out/'metrics.jsonl').open('a',buffering=1)
    def record(row):
        log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
    def save(path):
        t=time.monotonic()
        payload=dict(config=cfg,model=model.state_dict(),ema=ema_model.state_dict(),
            optimizer=opt.state_dict(),scheduler=sched.state_dict(),step=step,epoch=epoch,
            batch_index=batch_index,best_val_loss=best,train_seconds=train_seconds,
            ema_optimization_step=ema.optimization_step,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
        tmp=path.with_suffix('.partial');torch.save(payload,tmp);tmp.replace(path)
        record(dict(event='checkpoint',step=step,path=str(path),seconds=time.monotonic()-t))
    def validate():
        t=time.monotonic();ema_model.eval();total=0.;n=0
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            torch.manual_seed(1729)
            for batch in val_loader:
                batch=common.cuda_batch(batch)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):loss=ema_model.compute_loss(batch)
                b=len(batch['action']);total+=float(loss)*b;n+=b
        ema_model.train()
        return total/n,time.monotonic()-t
    resume_batch=batch_index
    window=[];stable=[]
    while step<limit:
        generator=torch.Generator().manual_seed(cfg['seed']+epoch)
        indices=torch.randperm(len(ds),generator=generator).tolist()
        indices=indices[(resume_batch+1)*cfg['batch_size']:]
        loader=DataLoader(ds,sampler=indices,**kwargs)
        iterator=iter(loader)
        while step<limit:
            t=time.monotonic()
            try:batch=next(iterator)
            except StopIteration:break
            wait=time.monotonic()-t
            batch=common.cuda_batch(batch)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):loss=model.compute_loss(batch)
            if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at update {step+1}')
            loss.backward();opt.step();opt.zero_grad(set_to_none=True);sched.step();ema.step(model)
            loss_value=float(loss)
            torch.cuda.synchronize()
            elapsed=time.monotonic()-t
            step+=1;batch_index+=1;train_seconds+=elapsed
            window.append((elapsed,wait,loss_value))
            if 200<step<=700:stable.append((elapsed,wait,loss_value))
            if step%100==0:
                arr=np.array(window)
                row=dict(event='train',step=step,updates=cfg['updates'],epoch=epoch,
                    loss=float(arr[:,2].mean()),lr=sched.get_last_lr()[0],
                    update_ms=float(arr[:,0].mean()*1000),data_wait_ms=float(arr[:,1].mean()*1000),
                    training_seconds=train_seconds,eta_training_seconds=float(arr[:,0].mean()*(cfg['updates']-step)),
                    gpu_allocated_mib=torch.cuda.memory_allocated()/2**20,
                    gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20)
                record(row);(out/'progress.json').write_text(json.dumps(row,indent=2));window=[]
            if step==700 and len(stable)==500:
                arr=np.array(stable)
                timing=dict(warmup_updates=200,measured_updates=500,mean_update_ms=float(arr[:,0].mean()*1000),
                    p50_update_ms=float(np.median(arr[:,0])*1000),p95_update_ms=float(np.percentile(arr[:,0],95)*1000),
                    mean_data_wait_ms=float(arr[:,1].mean()*1000),
                    estimated_budget_training_hours=float(arr[:,0].mean()*cfg['updates']/3600),
                    includes_validation=False,includes_checkpoint=False,exclusive_gpu=False)
                (out/'timing.json').write_text(json.dumps(timing,indent=2));record(dict(event='timing',**timing))
            if step%cfg['validate_every']==0 or step==limit:
                val_loss,seconds=validate()
                record(dict(event='validation',step=step,ema_val_loss=val_loss,windows=len(val),seconds=seconds))
                if val_loss<best:best=val_loss;save(ckptdir/'best.pt')
            if step%cfg['checkpoint_every']==0 or step%cfg['eval_every']==0 or step==limit:save(last)
            if step%cfg['eval_every']==0 or step==limit:
                if cfg.get('preserve_eval_checkpoints',False):
                    snapshot=ckptdir/f'step_{step:06d}.pt'
                    if not snapshot.exists():snapshot.hardlink_to(last)
                evaldir=out/f'eval_step_{step:06d}'
                t_eval=time.monotonic()
                with (evaldir.with_suffix('.log')).open('a') as f:
                    result=subprocess.run([sys.executable,'-u',str(common.HERE/'evaluate.py'),
                        '--checkpoint',str(last),'--episodes',str(cfg['eval_episodes']),
                        '--max-steps',str(cfg['max_episode_steps']),'--output',str(evaldir)],stdout=f,stderr=subprocess.STDOUT)
                record(dict(event='evaluation',step=step,returncode=result.returncode,
                            seconds=time.monotonic()-t_eval,output=str(evaldir)))
                if result.returncode:raise RuntimeError(f'Evaluation failed; see {evaldir}.log')
        del iterator,loader
        epoch+=1;batch_index=-1;resume_batch=-1
    marker='completed.json' if step==cfg['updates'] else f'segment_{step:06d}.json'
    (out/marker).write_text(json.dumps(dict(step=step,training_seconds=train_seconds,
        best_val_loss=best,final_eval=str(out/f'eval_step_{step:06d}/summary.json')),indent=2))
    log.close()


if __name__=='__main__':main()
