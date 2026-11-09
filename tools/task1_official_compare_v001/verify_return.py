"""Load returned raw/EMA/optimizer states and run CUDA inference before cleanup."""
import common
import argparse
import gc
import json
from pathlib import Path
import torch
from evaluate import wait_gpu


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);args=ap.parse_args()
    wait_gpu();torch.set_num_threads(2);torch.manual_seed(42)
    rows=[]
    for variant in ('official','current'):
        directory=args.run/'outputs'/variant/'checkpoints'
        for path in (directory/'best.pt',directory/'last.pt'):
            payload=torch.load(path,map_location='cpu',weights_only=False)
            assert all(k in payload for k in ('model','ema','optimizer','scheduler','rank_rng','step','epoch','batch_index'))
            model=common.build(payload['config']);model.load_state_dict(payload['model']);model.load_state_dict(payload['ema']);model.cuda().eval()
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.95,.999),weight_decay=1e-6)
            optimizer.load_state_dict(payload['optimizer'])
            cfg=dict(payload['config'],data=str(common.ROOT/'outputs/datasets/task1/diffusion_parallel_diverse_new_wrist_v001'),
                proprio_path=str(common.ROOT/'outputs/runs/task1/diffusion_policy/official_input_compare_v001/proprio9.npy'))
            sample=common.Sequences(cfg,'validation')[0]
            obs=common.cuda_obs({k:v[None] for k,v in sample['obs'].items()},cfg)
            with torch.no_grad():action=model.predict_action(obs)['action']
            assert action.shape==(1,8,7) and torch.isfinite(action).all()
            assert 0 < payload['step'] <= cfg['stage_updates']
            if path.name=='last.pt':assert payload['step']==cfg['stage_updates']
            rows.append(dict(path=str(path),step=payload['step'],raw_and_ema_loaded=True,optimizer_loaded=True,action_shape=list(action.shape)))
            del payload,model,optimizer,action,obs,sample;gc.collect();torch.cuda.empty_cache()
    common.write_json(args.run/'checkpoint_verification.json',dict(passed=True,checkpoints=rows))
    print('RETURNED_CHECKPOINTS_OK',flush=True)


if __name__=='__main__':main()
