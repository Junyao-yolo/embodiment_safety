"""Task1 IPC worker for official DDPM100 EMA inference."""
import common
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from algos.diffusion_policy.action_chunk import ActionChunk

OUT=Path(sys.argv[1])


def main():
    torch.set_num_threads(2)
    assert os.environ.get('TASK1_ONLINE_LEARN')!='1'
    seed=int(os.environ.get('TASK1_EPISODE_SEED','42'));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    checkpoint=Path(os.environ['SERL_INPUT_CHECKPOINT'])
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    cfg=payload['config'];model=common.build(cfg);model.load_state_dict(payload['ema']);model.cuda().eval()
    assert model.n_action_steps==8 and model.num_inference_steps==100
    common.write_json(OUT/'loaded_checkpoint.json',dict(path=str(checkpoint),weights='ema',step=payload['step'],
        input_variant=cfg['input_variant'],seed=seed,trained_action_steps=8,runtime_action_steps=8,
        algorithm='official_hybrid',scheduler='DDPM',inference_steps=100))
    del payload
    history=[];chunk=ActionChunk(8);n=0
    while True:
        request=OUT/f'request_{n:03d}.json';deadline=time.monotonic()+600
        while not request.exists() and not (OUT/'policy_stop.json').exists():
            if (OUT/'actor_error.json').exists():raise RuntimeError('Actor failed')
            if time.monotonic()>deadline:raise TimeoutError('Policy request')
            time.sleep(.002)
        if not request.exists():break
        start=time.perf_counter()
        with np.load(OUT/f'obs_{n:03d}.npz') as source:obs={k:source[k][0].copy() for k in source.files}
        state=common.lowdim(json.loads((OUT/f'raw_{n:03d}.json').read_text())) if cfg['input_variant']=='official' else obs['state']
        history.append((obs,state));history=history[-2:]
        pair=history if len(history)==2 else history*2
        timings=dict(preprocessing_ms=0.,inference_ms=0.,graph_capture_ms=0.)
        def predict():
            t=time.perf_counter()
            batch=common.observation(cfg,{k:np.stack([x[0][k] for x in pair]) for k in cfg['cameras']},np.stack([x[1] for x in pair]))
            batch=common.cuda_obs({k:v[None] for k,v in batch.items()},cfg)
            torch.cuda.synchronize();timings['preprocessing_ms']=(time.perf_counter()-t)*1000
            t=time.perf_counter()
            with torch.no_grad():actions=model.predict_action(batch)['action'][0].cpu().numpy()
            timings['inference_ms']=(time.perf_counter()-t)*1000
            assert actions.shape==(8,7) and np.isfinite(actions).all()
            return actions
        action,inferred,plan,offset=chunk.take(n,json.loads(request.read_text()).get('action_source','policy'),predict)
        common.write_json(OUT/f'action_{n:03d}.json',dict(index=n,proposed=action.tolist(),executed=np.clip(action,-1,1).tolist(),
            network_updates=0,sampling_mode='seeded_ddpm_100',seed=seed,history_frames=len(history),action_horizon=16,
            executed_chunk=8,inference_performed=inferred,plan_start_index=plan,chunk_offset=offset,**timings,
            request_compute_ms=(time.perf_counter()-start)*1000))
        n+=1
    stop=json.loads((OUT/'policy_stop.json').read_text())
    common.write_json(OUT/'worker_result.json',dict(passed=True,network_updates=0,
        new_transitions=stop['final_observation_index'],learning=False,algorithm='official_hybrid'))


if __name__=='__main__':
    try:main()
    except BaseException:
        common.write_json(OUT/'worker_error.json',dict(error=traceback.format_exc()));raise
