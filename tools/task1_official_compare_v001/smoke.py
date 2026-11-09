"""Seeded full-batch training, checkpoint, and DDPM inference witness."""
import common
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusion_policy.model.diffusion.ema_model import EMAModel


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);args=ap.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(cfg['output_dir'])/'smoke';out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    ds=common.Sequences(cfg,'train')
    sample=ds[0];first=ds.starts[0]
    for key in cfg['cameras']:
        original=np.load(Path(cfg['data'])/(key+'.npy'),mmap_mode='r')
        np.testing.assert_array_equal(sample['obs'][key].numpy(),np.moveaxis(original[first:first+2],-1,1))
    batch=common.cuda_batch(next(iter(DataLoader(ds,batch_size=64,num_workers=0))),cfg)
    model=common.build(cfg);model.set_normalizer(ds.normalizer());model.cuda()
    ema_model=copy.deepcopy(model);ema=EMAModel(ema_model,power=.75)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
    start=time.monotonic();losses=[]
    for _ in range(2):
        loss=model.compute_loss(batch);assert torch.isfinite(loss)
        loss.backward();opt.step();opt.zero_grad(set_to_none=True);ema.step(model);losses.append(float(loss))
    torch.cuda.synchronize()
    info=dict(variant=cfg['input_variant'],device=torch.cuda.get_device_name(),batch_size=64,precision='float32',
        parameters=sum(x.numel() for x in model.parameters()),trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
        loss=losses,two_update_seconds=time.monotonic()-start,gpu_peak_mib=torch.cuda.max_memory_allocated()/2**20,
        observations={k:list(v.shape) for k,v in batch['obs'].items()},action=list(batch['action'].shape))
    cp=out/'checkpoint.pt'
    torch.save(dict(config=cfg,model=model.state_dict(),ema=ema_model.state_dict(),optimizer=opt.state_dict(),
                    step=2,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),cp)
    del model,ema_model,ema,opt,loss
    torch.cuda.empty_cache()
    payload=torch.load(cp,map_location='cpu',weights_only=False)
    model=common.build(cfg);model.load_state_dict(payload['ema']);model.cuda().eval();del payload
    with torch.no_grad():action=model.predict_action({k:v[:1] for k,v in batch['obs'].items()})['action']
    assert action.shape==(1,8,7) and torch.isfinite(action).all()
    info.update(reloaded_action=list(action.shape),checkpoint=str(cp),cache_pixels_verified=True)
    common.write_json(out/'result.json',info);print(json.dumps(info),flush=True)
    print('TASK1_OFFICIAL_SMOKE_OK',flush=True)


if __name__=='__main__':main()
