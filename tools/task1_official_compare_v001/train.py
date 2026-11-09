"""Resumable FP32 training of the official DP-C Task1 policy."""
import common
import argparse
import copy
import json
import math
from pathlib import Path
import random
import signal
import subprocess
import threading
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusion_policy.model.diffusion.ema_model import EMAModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--stop-after', type=int)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    limit = min(cfg['updates'], args.stop_after or cfg['stage_updates'])
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    ckpts = out/'checkpoints'; ckpts.mkdir(exist_ok=True)
    common.write_json(out/'config.json',cfg)
    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed']); np.random.seed(cfg['seed']); random.seed(cfg['seed'])
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    ds = common.Sequences(cfg,'train'); val = common.Sequences(cfg,'validation')
    kwargs = dict(batch_size=cfg['batch_size'],num_workers=cfg['num_workers'],pin_memory=True,
                  persistent_workers=False,prefetch_factor=2)
    model = common.build(cfg); model.set_normalizer(ds.normalizer()); model.cuda()
    ema_model = copy.deepcopy(model)
    ema = EMAModel(ema_model,power=.75)
    opt = torch.optim.AdamW(model.parameters(),lr=cfg['lr'],betas=(.95,.999),eps=1e-8,weight_decay=1e-6)
    def lr_factor(s):
        if s<cfg['warmup_steps']: return s/max(1,cfg['warmup_steps'])
        fraction=(s-cfg['warmup_steps'])/(cfg['scheduler_total_steps']-cfg['warmup_steps'])
        return .5*(1+math.cos(math.pi*min(fraction,1)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_factor)
    step=0;epoch=0;batch_index=-1;best=float('inf');train_seconds=0.
    last=ckpts/'last.pt'
    if last.exists():
        payload=torch.load(last,map_location='cpu',weights_only=False)
        assert payload['config']==cfg
        model.load_state_dict(payload['model']);ema_model.load_state_dict(payload['ema'])
        model.cuda();ema_model.cuda()
        opt.load_state_dict(payload['optimizer']);sched.load_state_dict(payload['scheduler'])
        step=payload['step'];epoch=payload['epoch'];batch_index=payload['batch_index']
        best=payload['best_val_loss'];train_seconds=payload['train_seconds']
        ema.optimization_step=payload['ema_optimization_step']
        torch.set_rng_state(payload['torch_rng']);torch.cuda.set_rng_state(payload['cuda_rng'])
        np.random.set_state(payload['numpy_rng']);random.setstate(payload['python_rng'])
        del payload
    common.write_json(out/'training_start.json',dict(device=torch.cuda.get_device_name(),
        parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        train_windows=len(ds),validation_windows=len(val),train_episodes=96,validation_episodes=11,
        lossless_cache_enabled=True,batch_size=cfg['batch_size'],precision=cfg['precision'],step=step))
    halt=threading.Event()
    signal.signal(signal.SIGTERM,lambda *_:halt.set())
    signal.signal(signal.SIGINT,lambda *_:halt.set())
    def monitor():
        with (out/'gpu.jsonl').open('a',buffering=1) as stream:
            while not halt.is_set():
                gpu=subprocess.run(['nvidia-smi','--query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],capture_output=True,text=True)
                processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],capture_output=True,text=True)
                stream.write(json.dumps(dict(time=time.time(),gpu=gpu.stdout.strip(),processes=processes.stdout.strip()))+'\n')
                halt.wait(10)
    threading.Thread(target=monitor,daemon=True).start()
    log=(out/'metrics.jsonl').open('a',buffering=1)
    def record(row):
        log.write(json.dumps(row)+'\n');print(json.dumps(row),flush=True)
    def save(path):
        started=time.monotonic()
        payload=dict(config=cfg,model=model.state_dict(),ema=ema_model.state_dict(),optimizer=opt.state_dict(),
            scheduler=sched.state_dict(),step=step,epoch=epoch,batch_index=batch_index,best_val_loss=best,
            train_seconds=train_seconds,ema_optimization_step=ema.optimization_step,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
        tmp=path.with_suffix('.partial');torch.save(payload,tmp);tmp.replace(path)
        record(dict(event='checkpoint',step=step,path=str(path),seconds=time.monotonic()-started))
    def validate():
        started=time.monotonic();ema_model.eval();total=0.;count=0
        loader=DataLoader(val,shuffle=False,generator=torch.Generator().manual_seed(1729),**kwargs)
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            torch.manual_seed(1729)
            for batch in loader:
                batch=common.cuda_batch(batch,cfg);loss=ema_model.compute_loss(batch)
                n=len(batch['action']);total+=float(loss)*n;count+=n
        ema_model.train()
        return total/count,time.monotonic()-started
    window=[];stable=[]
    resume_batch=batch_index
    while step<limit and not halt.is_set():
        generator=torch.Generator().manual_seed(cfg['seed']+epoch)
        indices=torch.randperm(len(ds),generator=generator).tolist()[(resume_batch+1)*cfg['batch_size']:]
        loader=DataLoader(ds,sampler=indices,generator=torch.Generator().manual_seed(cfg['seed']+epoch),**kwargs)
        iterator=iter(loader)
        while step<limit and not halt.is_set():
            start=time.monotonic()
            try: batch=next(iterator)
            except StopIteration: break
            wait=time.monotonic()-start
            measured=200<=step<700
            events=[torch.cuda.Event(enable_timing=True) for _ in range(6)] if measured else None
            if events: events[0].record()
            batch=common.cuda_batch(batch,cfg)
            if events: events[1].record()
            loss=model.compute_loss(batch)
            if events: events[2].record()
            loss.backward()
            if events: events[3].record()
            opt.step();opt.zero_grad(set_to_none=True);sched.step()
            if events: events[4].record()
            ema.step(model)
            if events: events[5].record()
            loss_value=float(loss)
            if not math.isfinite(loss_value): raise RuntimeError(f'Nonfinite loss at {step+1}')
            torch.cuda.synchronize()
            elapsed=time.monotonic()-start
            step+=1;batch_index+=1;train_seconds+=elapsed
            window.append((elapsed,wait,loss_value))
            if measured:
                stable.append(dict(update_ms=elapsed*1000,data_wait_ms=wait*1000,
                    **dict(zip(('h2d_and_preprocess_ms','forward_ms','backward_ms','optimizer_ms','ema_ms'),
                               [events[i].elapsed_time(events[i+1]) for i in range(5)]))))
            if step%100==0:
                logging_start=time.monotonic();arr=np.asarray(window)
                row=dict(event='train',step=step,epoch=epoch+(batch_index+1)/cfg['steps_per_epoch'],
                    stage_updates=limit,full_budget_updates=cfg['updates'],loss=float(arr[:,2].mean()),
                    lr=sched.get_last_lr()[0],update_ms=float(arr[:,0].mean()*1000),data_wait_ms=float(arr[:,1].mean()*1000),
                    training_seconds=train_seconds,eta_stage_training_seconds=float(arr[:,0].mean()*(limit-step)),
                    gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20)
                record(row);common.write_json(out/'progress.json',row);window=[]
                if measured: stable[-1]['logging_ms']=1000*(time.monotonic()-logging_start)
            if step==700 and len(stable)==500:
                timing=dict(warmup=200,stable_updates=500,means={key:float(np.mean([x.get(key,0.) for x in stable]))
                    for key in (*stable[0].keys(),'logging_ms')},includes_validation=False,includes_checkpoint=False,
                    gpu_occupancy_log=str(out/'gpu.jsonl'),samples=stable)
                common.write_json(out/'timing.json',timing)
            if step%cfg['validate_every']==0 or step==limit:
                val_loss,seconds=validate();record(dict(event='validation',step=step,ema_val_loss=val_loss,windows=len(val),seconds=seconds))
                if val_loss<best:best=val_loss;save(ckpts/'best.pt')
            if step%cfg['checkpoint_every']==0 or step==700 or step==limit:save(last)
        del iterator,loader
        if step>=limit or halt.is_set(): break
        epoch+=1;batch_index=-1;resume_batch=-1
    if halt.is_set(): save(last)
    else:
        snapshot=ckpts/f'step_{step:07d}.pt'
        if not snapshot.exists():snapshot.hardlink_to(last)
    common.write_json(out/('paused.json' if halt.is_set() else f'segment_{step:07d}.json'),
        dict(step=step,epoch=epoch+(batch_index+1)/cfg['steps_per_epoch'],full_budget_updates=cfg['updates'],
             training_seconds=train_seconds,best_val_loss=best,checkpoint=str(last),full_budget_complete=step==cfg['updates']))
    halt.set();log.close()


if __name__=='__main__':main()
