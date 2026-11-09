"""Exercise CUDA training, diffusion inference, and EGL ToolHang rendering."""
import common
import json
import time
from pathlib import Path
import numpy as np
import torch
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer


def main():
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    assert torch.cuda.is_available()
    p = common.build(common.DEFAULT)
    norm = LinearNormalizer()
    norm['action'] = SingleFieldLinearNormalizer.create_identity()
    for key in common.SHAPE['obs']:
        norm[key] = SingleFieldLinearNormalizer.create_identity()
    p.set_normalizer(norm)
    p.cuda()
    batch = {'obs':{k:torch.rand(16,2,*v['shape'],device='cuda')
                    for k,v in common.SHAPE['obs'].items()},
             'action':torch.randn(16,16,10,device='cuda')*.1}
    opt = torch.optim.AdamW(p.parameters(),lr=1e-4)
    times=[]
    for i in range(25):
        torch.cuda.synchronize();start=time.monotonic()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss=p.compute_loss(batch)
        assert torch.isfinite(loss)
        loss.backward();opt.step();opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if i>=5:times.append(time.monotonic()-start)
    p.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        action=p.predict_action({k:v[:1] for k,v in batch['obs'].items()})['action']
    assert action.shape==(1,8,10) and torch.isfinite(action).all()
    row=dict(event='CUDA_WITNESS',gpu=torch.cuda.get_device_name(),torch=torch.__version__,
             loss=float(loss),action_shape=list(action.shape),parameters=sum(x.numel() for x in p.parameters()),
             synthetic_update_ms=float(np.mean(times)*1000),
             peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    print(json.dumps(row),flush=True)
    del p,opt,batch,action
    torch.cuda.empty_cache()
    import robosuite
    from robosuite.controllers import load_controller_config
    cfg=load_controller_config(default_controller='OSC_POSE')
    env=robosuite.make('ToolHang',robots='Panda',controller_configs=cfg,
        has_renderer=False,has_offscreen_renderer=True,use_camera_obs=True,
        camera_names=['sideview','robot0_eye_in_hand'],camera_heights=240,camera_widths=240,
        control_freq=20,ignore_done=True,render_gpu_device_id=0)
    obs=env.reset()
    for _ in range(3):obs,_,_,_=env.step(np.zeros(7))
    for key in common.CAMERAS:
        assert obs[key].shape==(240,240,3) and obs[key].std()>1
    row['simulator']=dict(robosuite=robosuite.__version__,image_shapes={k:list(obs[k].shape) for k in common.CAMERAS})
    env.close()
    out=Path(common.DEFAULT['output_dir']);out.mkdir(parents=True,exist_ok=True)
    (out/'smoke.json').write_text(json.dumps(row,indent=2))
    print('TOOL_HANG_SMOKE_OK',json.dumps(row),flush=True)


if __name__=='__main__':main()
