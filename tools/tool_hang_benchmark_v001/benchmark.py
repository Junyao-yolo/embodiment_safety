"""Comparable CUDA update timings for current and published ToolHang policies."""
import common
import argparse
import contextlib
import copy
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(common.HERE / 'official_deps'))
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from diffusion_policy.model.diffusion.ema_model import EMAModel


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant',choices=['current_b16_bf16','official_b16_bf16','official_b64_fp32'],required=True)
    ap.add_argument('--output',required=True)
    ap.add_argument('--warmup',type=int,default=200)
    ap.add_argument('--updates',type=int,default=500)
    args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    official=OmegaConf.load(Path(common.DEFAULT['output_dir'])/'profile_output/official_tool_hang_config.yaml')
    native=args.variant=='official_b64_fp32'
    batch_size=64 if native else 16
    workers=8 if native else 4
    amp=not native
    t=time.monotonic();ds=common.dataset(common.DEFAULT)
    loader=DataLoader(ds,batch_size=batch_size,num_workers=workers,shuffle=True,pin_memory=True,
                      persistent_workers=False,prefetch_factor=2,generator=torch.Generator().manual_seed(42))
    model=common.build(common.DEFAULT) if args.variant.startswith('current') else hydra.utils.instantiate(official.policy)
    model.set_normalizer(ds.get_normalizer());model.cuda().train()
    ema_model=copy.deepcopy(model)
    ema=EMAModel(ema_model,power=.75)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),eps=1e-8,weight_decay=1e-6)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(s/500,1.))
    info=dict(variant=args.variant,batch_size=batch_size,num_workers=workers,precision='bfloat16' if amp else 'float32',
              model_class=type(model).__name__,parameters=sum(p.numel() for p in model.parameters()),
              observation_feature_dim=model.obs_feature_dim,train_windows=len(ds),batches_per_epoch=len(loader),
              warmup_updates=args.warmup,measured_updates=args.updates,
              setup_seconds=time.monotonic()-t,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
              tf32_matmul=torch.backends.cuda.matmul.allow_tf32,tf32_cudnn=torch.backends.cudnn.allow_tf32,
              cudnn_benchmark=torch.backends.cudnn.benchmark,includes_validation=False,
              includes_checkpoint=False,includes_rollout=False)
    (out/'config.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
    events=[torch.cuda.Event(enable_timing=True) for _ in range(6)]
    rows=[];iterator=iter(loader)
    log=(out/'steps.jsonl').open('w',buffering=1)
    for i in range(args.warmup+args.updates):
        start=time.perf_counter()
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        wait=time.perf_counter()-start
        events[0].record();batch=common.cuda_batch(batch);events[1].record()
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=amp):loss=model.compute_loss(batch)
        assert torch.isfinite(loss),f'Nonfinite loss at {i}'
        events[2].record();loss.backward();events[3].record()
        opt.step();opt.zero_grad(set_to_none=True);sched.step();events[4].record()
        ema.step(model);events[5].record()
        value=float(loss);torch.cuda.synchronize()
        stage=[events[j].elapsed_time(events[j+1]) for j in range(5)]
        row=dict(update=i+1,measured=i>=args.warmup,loss=value,data_wait_ms=wait*1000,
                 transfer_ms=stage[0],forward_ms=stage[1],backward_ms=stage[2],optimizer_ms=stage[3],ema_ms=stage[4])
        log_start=time.perf_counter();log.write(json.dumps(row)+'\n');logging_ms=(time.perf_counter()-log_start)*1000
        row['logging_ms']=logging_ms;row['update_ms']=(time.perf_counter()-start)*1000
        if i>=args.warmup:rows.append(row)
        if (i+1)%100==0:
            recent=rows[-100:]
            print(json.dumps(dict(update=i+1,total=args.warmup+args.updates,loss=value,
                                  mean_update_ms=float(np.mean([r['update_ms'] for r in recent])) if recent else None)),flush=True)
    log.close()
    keys=['update_ms','data_wait_ms','transfer_ms','forward_ms','backward_ms','optimizer_ms','ema_ms','logging_ms']
    timing={k:float(np.mean([r[k] for r in rows])) for k in keys}
    sec=timing['update_ms']/1000
    result=dict(**info,status='completed',timing_ms=timing,
                p50_update_ms=float(np.median([r['update_ms'] for r in rows])),
                p95_update_ms=float(np.percentile([r['update_ms'] for r in rows],95)),
                samples_per_second=batch_size/sec,epoch_minutes=len(loader)*sec/60,
                hours_for_100k_updates=100000*sec/3600,
                hours_for_1p6m_windows=(1600000/batch_size)*sec/3600,
                hours_for_official_3000_epochs=3000*len(loader)*sec/3600 if native else None,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,last_loss=rows[-1]['loss'])
    (out/'result.json').write_text(json.dumps(result,indent=2))
    (out/'measurements.json').write_text(json.dumps(rows))
    print('BENCHMARK_DONE',json.dumps(result),flush=True)


if __name__=='__main__':
    try:main()
    except torch.cuda.OutOfMemoryError as e:
        args=sys.argv;out=Path(args[args.index('--output')+1]);out.mkdir(parents=True,exist_ok=True)
        (out/'result.json').write_text(json.dumps(dict(status='out_of_memory',error=str(e)),indent=2))
        raise
