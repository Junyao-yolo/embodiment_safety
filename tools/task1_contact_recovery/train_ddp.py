"""Two-GPU Task1 recovery updates with official Policy and Workspace checkpoints."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import time
from datetime import timedelta

from learning.bootstrap import configure
configure()
import dill
import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("eval", eval, replace=True)
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.common.pytorch_util import dict_apply
from learning.dataset.task1_image_dataset import Task1ImageDataset

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = '/mnt/data2/wjy_SERL/Embodied_Safety'


class LossModule(torch.nn.Module):
    def __init__(self, policy):
        super().__init__(); self.policy = policy
    def forward(self, batch):
        return self.policy.compute_loss(batch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', choices=['official', 'current'], required=True)
    parser.add_argument('--config')
    parser.add_argument('--output', required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK']); world = int(os.environ['WORLD_SIZE'])
    assert world == 2
    torch.set_num_threads(2); torch.cuda.set_device(local)
    dist.init_process_group('nccl', timeout=timedelta(minutes=30), device_id=torch.device('cuda', local))
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    parent = ROOT / 'parents' / (args.variant + '.ckpt')
    payload = torch.load(parent, map_location='cpu', pickle_module=dill, weights_only=False, mmap=True)
    cfg = payload['cfg'] if args.smoke else OmegaConf.load(args.config)
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload['state_dicts']['ema_model'])
    del payload
    policy.cuda()
    if args.smoke:
        data = ROOT / 'outputs/datasets/task1/diffusion_parallel_diverse_new_wrist_v001'
        ds = Task1ImageDataset(str(data), args.variant, proprio_path=str(data/'proprio9.npy'))
    else:
        dataset_config = OmegaConf.create(OmegaConf.to_container(cfg.task.dataset, resolve=True))
        for key in ('dataset_path', 'recovery_path', 'normalizer_path'):
            dataset_config[key] = dataset_config[key].replace(SOURCE_ROOT, str(ROOT), 1)
        ds = hydra.utils.instantiate(dataset_config)
    validation = ds.get_validation_dataset()
    ema_policy = copy.deepcopy(policy) if rank == 0 else None
    ema = EMAModel(ema_policy, power=.75) if rank == 0 else None
    for name, parameter in policy.named_parameters():
        if name.endswith('_dummy_variable') and parameter.numel()==0:
            parameter.requires_grad_(False)
    ddp = DistributedDataParallel(LossModule(policy), device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-5, betas=(.95,.999), eps=1e-8, weight_decay=1e-6)
    global_batch = 32
    epochs = 1 if args.smoke else 3
    updates = 2 if args.smoke else math.ceil(len(ds)/global_batch)*epochs
    def factor(step):
        if step<100:return step/100
        return .5*(1+math.cos(math.pi*min(1.,(step-100)/max(1,updates-100))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    torch.manual_seed(42+rank); np.random.seed(42+rank); random.seed(42+rank)
    workers = dict(num_workers=2, pin_memory=True, multiprocessing_context='spawn', prefetch_factor=2)
    log = (out/'metrics.jsonl').open('w', buffering=1) if rank==0 else None
    def record(row):
        if rank==0:
            log.write(json.dumps(row)+'\n'); print(json.dumps(row),flush=True)
            (out/'progress.json').write_text(json.dumps(row,indent=2))
    record(dict(event='start', variant=args.variant, world_size=world, global_batch=32, local_batch=16,
        train_windows=len(ds), validation_windows=len(validation), target_updates=updates,
        device=torch.cuda.get_device_name(), precision='FP32, TF32 disabled', parent=str(parent)))
    step=0; best=float('inf'); start=time.monotonic(); update_seconds=0.
    def checkpoint(path, epoch):
        rng=dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
        rngs=[None]*world; dist.all_gather_object(rngs,rng)
        if rank==0:
            state=dict(cfg=cfg, state_dicts=dict(model=policy.state_dict(),ema_model=ema_policy.state_dict(),optimizer=optimizer.state_dict()),
                pickles=dict(global_step=dill.dumps(step),epoch=dill.dumps(epoch)),
                ddp_state=dict(scheduler=scheduler.state_dict(),rank_rng=rngs,world_size=world,global_batch=32,
                    step=step,epoch=epoch,best_val_loss=best,ema_optimization_step=ema.optimization_step))
            temporary=path.with_suffix('.partial');torch.save(state,temporary,pickle_module=dill);temporary.replace(path)
        dist.barrier()
    for epoch in range(epochs):
        ids=torch.randperm(len(ds),generator=torch.Generator().manual_seed(42+epoch)).tolist()
        batches=[ids[i:i+global_batch] for i in range(0,len(ids),global_batch)]
        if args.smoke:batches=batches[:2]
        local_batches=[np.array_split(np.asarray(ids),world)[rank].tolist() for ids in batches]
        loader=DataLoader(ds,batch_sampler=local_batches,**workers)
        for batch, global_ids in zip(loader,batches):
            tick=time.monotonic()
            batch=dict_apply(batch,lambda x:x.cuda(non_blocking=True))
            loss=ddp(batch)
            weighted=loss*(len(batch['action'])*world/len(global_ids))
            weighted.backward();optimizer.step();optimizer.zero_grad(set_to_none=True);scheduler.step()
            if rank==0:ema.step(policy)
            value=weighted.detach();dist.all_reduce(value);value=float(value/world)
            if not math.isfinite(value):raise RuntimeError('Nonfinite training loss')
            torch.cuda.synchronize();update_seconds+=time.monotonic()-tick;step+=1
            if step%100==0 or args.smoke or step==updates:
                record(dict(event='train',step=step,target_updates=updates,epoch=epoch,
                    loss=value,lr=scheduler.get_last_lr()[0],elapsed_s=time.monotonic()-start,
                    mean_compute_update_ms=1000*update_seconds/step,gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20))
        dist.barrier()
        result=torch.zeros(2,device='cuda')
        if rank==0:
            ema_policy.eval();total=0.;count=0
            with torch.random.fork_rng(devices=[local]),torch.no_grad():
                torch.manual_seed(1729)
                for batch in DataLoader(validation,batch_size=32,shuffle=False,**workers):
                    batch=dict_apply(batch,lambda x:x.cuda(non_blocking=True));n=len(batch['action'])
                    total+=float(ema_policy.compute_loss(batch))*n;count+=n
                    if args.smoke:break
            val_loss=total/count;result[0]=val_loss;result[1]=float(val_loss<best)
            ema_policy.train()
            record(dict(event='validation',step=step,epoch=epoch,val_loss=val_loss,windows=count))
        dist.broadcast(result,src=0);best=min(best,float(result[0]))
        checkpoint(out/'latest.ckpt',epoch)
        if bool(result[1]):checkpoint(out/'best.ckpt',epoch)
        if args.smoke:
            # A real DDP update must leave the participating replicas equal.
            probe=next(p for p in policy.parameters() if p.requires_grad and p.numel()>0).detach().flatten()[:32].clone()
            reference=probe.clone();dist.broadcast(reference,src=0)
            assert torch.equal(probe,reference)
            dist.barrier()
            if rank==0:
                loaded=torch.load(out/'latest.ckpt',map_location='cpu',pickle_module=dill,weights_only=False)
                ema_policy.load_state_dict(loaded['state_dicts']['ema_model']);del loaded
                ema_policy.cuda()
                ema_policy.eval();sample=ds[0]
                obs=dict_apply(sample['obs'],lambda x:x[None].cuda())
                with torch.no_grad():action=ema_policy.predict_action(obs)['action']
                assert list(action.shape)==[1,8,7] and torch.isfinite(action).all()
                (out/'smoke_result.json').write_text(json.dumps(dict(passed=True,world_size=2,global_batch=32,local_batch=16,
                    updates=step,prediction_shape=list(action.shape),replica_probe_equal=True),indent=2))
    if rank==0:
        (out/'completed.json').write_text(json.dumps(dict(step=step,target_updates=updates,training_seconds=time.monotonic()-start,
            best_val_loss=best,world_size=2,global_batch=32,smoke=args.smoke),indent=2));log.close()
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
