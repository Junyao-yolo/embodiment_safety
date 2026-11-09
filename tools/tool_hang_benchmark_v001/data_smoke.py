"""Check dataset/environment agreement and train one real-data GPU batch."""
import common
import json
from pathlib import Path
import time
import subprocess
import sys
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from diffusion_policy.env_runner.robomimic_image_runner import create_env


def main():
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    start=time.monotonic();cfg=common.DEFAULT
    with h5py.File(cfg['dataset_path'],'r') as f:
        meta=json.loads(f['data'].attrs['env_args'])
        meta['env_kwargs']['controller_configs']['control_delta']=False
        env=create_env(meta,common.SHAPE,enable_render=True)
        rows=[]
        for name in ('demo_0','demo_99','demo_199'):
            demo=f['data'][name]
            state=dict(model=demo.attrs['model_file'],states=demo['states'][-1])
            obs=env.reset_to(state)
            success=bool(env.is_success()['task'])
            pos_err=float(np.max(np.abs(obs['robot0_eef_pos']-demo['obs/robot0_eef_pos'][-1])))
            row=dict(demo=name,restored_terminal_success=success,eef_pos_max_error=pos_err)
            rows.append(row)
            assert pos_err<1e-3,row
            assert success,row
        env.env.close()
    ds=common.dataset(cfg)
    p=common.build(cfg);p.set_normalizer(ds.get_normalizer());p.cuda()
    batch=common.cuda_batch(next(iter(DataLoader(ds,batch_size=16,shuffle=False,num_workers=0))))
    opt=torch.optim.AdamW(p.parameters(),lr=1e-4)
    with torch.autocast('cuda',dtype=torch.bfloat16):loss=p.compute_loss(batch)
    assert torch.isfinite(loss)
    loss.backward();opt.step();opt.zero_grad(set_to_none=True)
    p.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        action=p.predict_action({k:v[:1] for k,v in batch['obs'].items()})['action']
    assert action.shape==(1,8,10) and torch.isfinite(action).all()
    out=Path(cfg['output_dir'])
    checkpoint=out/'smoke_policy.pt'
    torch.save(dict(config=cfg,ema=p.state_dict(),step=0),checkpoint)
    subprocess.run([sys.executable,'-u',str(common.HERE/'evaluate.py'),
        '--checkpoint',str(checkpoint),'--episodes','1','--max-steps','16',
        '--output',str(out/'smoke_rollout')],check=True)
    row=dict(event='REAL_DATA_GPU_OK',loss=float(loss),action_shape=list(action.shape),
             terminal_checks=rows,elapsed_seconds=time.monotonic()-start)
    (out/'data_smoke.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True)


if __name__=='__main__':main()
