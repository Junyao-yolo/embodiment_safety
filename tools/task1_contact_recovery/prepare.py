"""Pack accepted teacher-only windows and prepare a fresh fine-tuning optimizer."""

from pathlib import Path
import json
import zlib
import numpy as np


def pack(cases, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    accepted = []
    for case in cases:
        episode = Path(case['episode'])
        result = json.loads((episode/'teacher_result.json').read_text())
        if not result['training_eligible']:
            continue
        actions = sorted(episode.glob('action_*.json'), key=lambda p:int(p.stem.rsplit('_',1)[1]))
        teacher = [(int(p.stem.rsplit('_',1)[1]), json.loads(p.read_text())) for p in actions]
        teacher = [(i,a) for i,a in teacher if a.get('action_source')=='teacher']
        ids = [i for i,a in teacher]
        if len(ids)<16 or ids != list(range(ids[0],ids[-1]+1)):
            raise ValueError('Teacher segment is not contiguous: '+str(episode))
        accepted.append((case,teacher))
    if len(accepted)<3 or len({case['seed'] for case,_ in accepted})<2:
        raise ValueError('At least three accepted recovery episodes required for train/validation split')
    count=sum(len(actions) for _,actions in accepted)
    state=np.lib.format.open_memmap(output/'state.npy', mode='w+', dtype='float32', shape=(count,31))
    proprio=np.lib.format.open_memmap(output/'proprio9.npy', mode='w+', dtype='float32', shape=(count,9))
    action=np.lib.format.open_memmap(output/'action.npy', mode='w+', dtype='float32', shape=(count,7))
    cache=output/'lossless_rgb_cache';cache.mkdir()
    index=np.zeros((3,count,2),dtype=np.int64)
    cameras=('external','wrist_left','wrist_right')
    episodes=[];offset=0;cursor=0
    with (cache/'rgb.bin').open('wb') as blob, (output/'rows.jsonl').open('w') as rows:
        for k,(case,actions) in enumerate(accepted):
            episode=Path(case['episode'])
            split='validation' if case['seed']==accepted[-1][0]['seed'] else 'train'
            episodes.append(dict(episode=str(episode),offset=offset,rows=len(actions),split=split,seed=case['seed']))
            for step,a in actions:
                raw=json.loads((episode/f'raw_{step:03d}.json').read_text())
                with np.load(episode/f'obs_{step:03d}.npz') as obs:
                    state[offset]=obs['state'][0]
                    for camera,key in enumerate(cameras):
                        rgb=obs[key][0].astype(np.uint8)
                        residual=rgb.copy();residual[:,1:]=rgb[:,1:]-rgb[:,:-1]
                        compressed=zlib.compress(residual.tobytes(),3)
                        decoded=np.cumsum(np.frombuffer(zlib.decompress(compressed),np.uint8).reshape(rgb.shape),axis=1,dtype=np.uint8)
                        if not np.array_equal(decoded,rgb):raise ValueError('Image encoding mismatch')
                        index[camera,offset]=cursor,len(compressed);blob.write(compressed);cursor+=len(compressed)
                proprio[offset]=[*raw['ee_position_m'],*raw['ee_quaternion_xyzw'],*raw['joint_positions'][-2:]]
                action[offset]=a['executed']
                rows.write(json.dumps(dict(episode=str(episode),source_index=step,stage=a['teacher_stage'],source_kind='accepted_contact_recovery'))+'\n')
                offset+=1
    state.flush();proprio.flush();action.flush();np.save(cache/'index.npy',index)
    manifest=dict(rows=count,episodes=episodes,horizon=16,n_obs_steps=2,n_action_steps=8)
    for split in ('train','validation'):
        manifest[split]=[i for e in episodes if e['split']==split for i in range(e['offset'],e['offset']+e['rows']-15)]
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (cache/'manifest.json').write_text(json.dumps(dict(status='finished',rows=count,cameras=cameras,frames_verified=count*3,all_pixels_equal=True),indent=2))
    return manifest


def initialize(parent, recovery_path, output, epochs=3):
    from learning.bootstrap import configure
    configure()
    import dill
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    source=torch.load(parent,map_location='cpu',pickle_module=dill,weights_only=False)
    cfg=source['cfg'];weights=source['state_dicts']['ema_model']
    norm=LinearNormalizer()
    norm.load_state_dict({k[len('normalizer.'):]:v for k,v in weights.items() if k.startswith('normalizer.')})
    normalizer_path=output/'normalizer.pt';torch.save(norm.state_dict(),normalizer_path)
    original_data=cfg.task.dataset.dataset_path
    cfg.task.dataset=OmegaConf.create(dict(_target_='learning.dataset.task1_recovery_dataset.Task1RecoveryDataset',
        dataset_path=original_data,recovery_path=str(recovery_path),normalizer_path=str(normalizer_path),
        input_variant=cfg.task.input_variant,horizon=16,n_obs_steps=2))
    cfg.task.env_runner.n_test=0
    cfg.training.resume=True;cfg.training.num_epochs=epochs
    cfg.training.max_train_steps=None;cfg.training.max_val_steps=None
    cfg.training.lr_warmup_steps=100;cfg.training.checkpoint_every=1;cfg.training.val_every=1
    cfg.training.rollout_every=1;cfg.training.sample_every=10000
    cfg.dataloader.batch_size=32;cfg.dataloader.num_workers=4
    cfg.val_dataloader.batch_size=32;cfg.val_dataloader.num_workers=2
    cfg.optimizer.lr=1e-5
    cfg.checkpoint.topk.monitor_key='val_loss';cfg.checkpoint.topk.mode='min';cfg.checkpoint.topk.k=1
    cfg.checkpoint.topk.format_str='epoch={epoch:04d}-val_loss={val_loss:.6f}.ckpt'
    train=output/'train';checkpoint=train/'checkpoints/latest.ckpt';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(cfg=cfg,state_dicts=dict(model=weights,ema_model=weights),pickles=dict(global_step=dill.dumps(0),epoch=dill.dumps(0)))
    torch.save(payload,checkpoint,pickle_module=dill)
    OmegaConf.save(cfg,output/'finetune.yaml')
    plan=dict(parent=str(parent),initial_weights='ema',optimizer='fresh AdamW',normalizer='parent unchanged',
        learning_rate=1e-5,batch_size=32,epochs=epochs,original_window_fraction=.75,teacher_window_fraction=.25,
        training_output=str(train),validation='original 11 episodes plus held-out recovery episode')
    (output/'training_plan.json').write_text(json.dumps(plan,indent=2))
    return train
