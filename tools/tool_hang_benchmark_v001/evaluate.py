"""Independent-state ToolHang rollouts with full-task and stage success metrics."""
import common
import argparse
from collections import deque
import json
from pathlib import Path
import time
import h5py
import imageio.v2 as imageio
import numpy as np
import torch
from diffusion_policy.model.common.rotation_transformer import RotationTransformer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--episodes',type=int,default=20)
    parser.add_argument('--max-steps',type=int,default=700)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    payload=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=payload['config'];policy=common.build(cfg)
    from diffusion_policy.env_runner.robomimic_image_runner import create_env
    policy.load_state_dict(payload['ema']);step=payload['step'];del payload
    policy.cuda().eval()
    with h5py.File(cfg['dataset_path'],'r') as f:
        meta=json.loads(f['data'].attrs['env_args'])
    meta['env_kwargs']['controller_configs']['control_delta']=False
    env=create_env(meta,common.SHAPE,enable_render=True)
    env.env.hard_reset=False
    rot=RotationTransformer('axis_angle','rotation_6d')
    rows=[];started=time.monotonic()
    for i in range(args.episodes):
        seed=100000+i
        np.random.seed(seed);torch.manual_seed(seed)
        obs=env.reset();policy.reset()
        history=deque([obs]*cfg['n_obs_steps'],maxlen=cfg['n_obs_steps'])
        video=out/f'episode_{i+1:03d}.mp4'
        wall=time.monotonic();infer_seconds=sim_seconds=0.;steps=0
        assembled=on_frame=success=False
        writer=imageio.get_writer(video,fps=20,codec='libx264',macro_block_size=1)
        try:
            while steps<args.max_steps and not success:
                batch={k:torch.from_numpy(np.stack([h[k] for h in history])).unsqueeze(0).float().cuda()
                       for k in common.SHAPE['obs']}
                torch.cuda.synchronize();t=time.monotonic()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,
                                                   enabled=cfg['precision']=='bfloat16'):
                    act=policy.predict_action(batch)['action'][0].float().cpu().numpy()
                infer_seconds+=time.monotonic()-t
                physical=np.concatenate([act[:,:3],rot.inverse(act[:,3:9]),act[:,9:10]],axis=-1)
                for a in physical:
                    t=time.monotonic();obs,reward,done,info=env.step(a)
                    sim_seconds+=time.monotonic()-t;history.append(obs);steps+=1
                    assembled=assembled or bool(env.env._check_frame_assembled())
                    on_frame=on_frame or bool(env.env._check_tool_on_frame())
                    success=bool(env.is_success()['task'])
                    frames=[np.clip(np.moveaxis(obs[k],0,-1)*255,0,255).astype(np.uint8) for k in common.CAMERAS]
                    writer.append_data(np.concatenate(frames,axis=1))
                    if success or steps>=args.max_steps:break
        finally:writer.close()
        row=dict(episode=i+1,seed=seed,checkpoint_step=step,success=success,
                 frame_assembled=assembled,tool_on_frame=on_frame,steps=steps,
                 wall_seconds=time.monotonic()-wall,policy_seconds=infer_seconds,
                 environment_seconds=sim_seconds,simulated_seconds=steps/20,video=str(video))
        rows.append(row)
        report=dict(checkpoint=args.checkpoint,checkpoint_step=step,completed=len(rows),
                    model_kind=cfg.get('model_kind','current'),precision=cfg['precision'],
                    num_inference_steps=cfg['num_inference_steps'],
                    requested=args.episodes,successes=sum(x['success'] for x in rows),
                    success_rate=np.mean([x['success'] for x in rows]),
                    frame_assembled_rate=np.mean([x['frame_assembled'] for x in rows]),
                    tool_on_frame_rate=np.mean([x['tool_on_frame'] for x in rows]),
                    elapsed_seconds=time.monotonic()-started,episodes=rows)
        (out/'summary.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(row),flush=True)
    env.env.close()
    print('EVALUATION_DONE',json.dumps({k:v for k,v in report.items() if k!='episodes'}),flush=True)


if __name__=='__main__':main()
