"""Prepare the official replay cache with a bounded CPU worker count."""
import common
import json
from pathlib import Path
import time
import h5py
import zarr
from diffusion_policy.dataset.robomimic_replay_image_dataset import _convert_robomimic_to_replay
from diffusion_policy.model.common.rotation_transformer import RotationTransformer


def main():
    cfg=common.DEFAULT
    started=time.monotonic()
    path=Path(cfg['dataset_path'])
    cache=Path(str(path)+'.zarr.zip')
    with h5py.File(path,'r') as f:
        names=sorted(f['data'],key=lambda x:int(x.split('_')[-1]))
        assert len(names)==200, len(names)
        lengths=[len(f['data'][n]['actions']) for n in names]
        demo=f['data'][names[0]]
        for k,shape in common.SHAPE['obs'].items():
            expected=tuple(shape['shape'])
            if shape.get('type')=='rgb':expected=(expected[1],expected[2],expected[0])
            assert demo['obs'][k].shape[1:]==expected,(k,demo['obs'][k].shape)
        report=dict(episodes=len(names),steps=sum(lengths),length_min=min(lengths),length_max=max(lengths),
                    source='https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip',
                    dataset_path=str(path),dataset_bytes=path.stat().st_size,
                    env_metadata=json.loads(f['data'].attrs['env_args']),
                    observation_shapes={k:list(demo['obs'][k].shape[1:]) for k in common.SHAPE['obs']},
                    stored_action_shape=list(demo['actions'].shape[1:]),network_action_shape=[10])
    if not cache.exists():
        buf=_convert_robomimic_to_replay(store=zarr.MemoryStore(),shape_meta=common.SHAPE,
            dataset_path=str(path),abs_action=True,
            rotation_transformer=RotationTransformer('axis_angle','rotation_6d'),
            n_workers=4,max_inflight_tasks=20)
        tmp=Path(str(cache)+'.partial')
        with zarr.ZipStore(str(tmp),mode='w') as store:buf.save_to_store(store=store)
        tmp.rename(cache)
        del buf
    report.update(cache_path=str(cache),cache_bytes=cache.stat().st_size,
                  cache_format='official DP JPEG2000 level=50',prepare_seconds=time.monotonic()-started)
    out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True)
    (out/'dataset.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
