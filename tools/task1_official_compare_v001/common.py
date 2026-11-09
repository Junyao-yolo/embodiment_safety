"""Official DP-C with two Task1 observation contracts."""
import json
import os
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BENCH = ROOT / 'tools/tool_hang_benchmark_v001'
sys.path[:0] = [str(BENCH/'official_deps'), str(BENCH/'deps'),
               str(ROOT/'algos/diffusion_policy/vendor'), str(ROOT)]
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '4'
import numpy as np
import torch
from torch.utils.data import Dataset
from algos.diffusion_policy.representation import encode_state, ROT6D

CAMERAS = ('external', 'wrist_left', 'wrist_right')


def build(cfg):
    import hydra
    from omegaconf import OmegaConf
    return hydra.utils.instantiate(OmegaConf.create(cfg['official_policy']))


def lowdim(raw):
    return np.asarray([*raw['ee_position_m'], *raw['ee_quaternion_xyzw'],
                       *raw['joint_positions'][-2:]], dtype=np.float32)


def observation(cfg, images, state):
    obs = {k: torch.from_numpy(np.moveaxis(images[k], -1, -3).copy()) for k in cfg['cameras']}
    if cfg['input_variant'] == 'official':
        obs.update(robot0_eef_pos=torch.from_numpy(state[..., :3].copy()),
                   robot0_eef_quat=torch.from_numpy(state[..., 3:7].copy()),
                   robot0_gripper_qpos=torch.from_numpy(state[..., 7:9].copy()))
    else:
        obs['state'] = torch.from_numpy(encode_state(state, ROT6D))
    return obs


def cuda_obs(obs, cfg):
    result = {}
    for key, value in obs.items():
        value = value.cuda(non_blocking=True)
        if key in cfg['cameras']:
            value = value.float().div_(255)
            if cfg['input_variant'] == 'official':
                b, t = value.shape[:2]
                value = torch.nn.functional.interpolate(value.flatten(0, 1), size=(240, 240),
                    mode='bilinear', align_corners=False, antialias=True).reshape(b, t, 3, 240, 240)
        result[key] = value
    return result


def cuda_batch(batch, cfg):
    return dict(obs=cuda_obs(batch['obs'], cfg), action=batch['action'].cuda(non_blocking=True))


class Sequences(Dataset):
    def __init__(self, cfg, split):
        self.cfg = cfg
        self.split = split
        root = Path(cfg['data'])
        self.meta = json.loads((root/'manifest.json').read_text())
        self.starts = self.meta[split]
        self.action = np.load(root/'action.npy', mmap_mode='r')
        self.state = np.load(cfg['proprio_path'] if cfg['input_variant']=='official' else root/'state.npy', mmap_mode='r')
        cache = root/'lossless_rgb_cache'
        meta = json.loads((cache/'manifest.json').read_text())
        assert meta['status']=='finished' and meta['rows']==self.meta['rows']
        assert meta['cameras']==list(CAMERAS) and meta['all_pixels_equal']
        self.index = np.load(cache/'index.npy', mmap_mode='r')
        self.blob = np.memmap(cache/'rgb.bin', dtype=np.uint8, mode='r')

    def __getstate__(self):
        return self.cfg, self.split

    def __setstate__(self, state):
        self.__init__(*state)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, item):
        start = self.starts[item]
        images = {}
        for key in self.cfg['cameras']:
            frames = []
            for index in (start, start+1):
                pos, size = self.index[CAMERAS.index(key), index]
                residual = np.frombuffer(zlib.decompress(self.blob[pos:pos+size]), np.uint8).reshape(256,256,3)
                frames.append(np.cumsum(residual, axis=1, dtype=np.uint8))
            images[key] = np.stack(frames)
        return dict(obs=observation(self.cfg, images, self.state[start:start+2]),
                    action=torch.from_numpy(self.action[start:start+16].copy()))

    def normalizer(self):
        from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer as Field
        from diffusion_policy.common.normalize_util import get_image_range_normalizer
        ids = np.concatenate([np.arange(e['offset'],e['offset']+e['rows'])
                              for e in self.meta['episodes'] if e['split']=='train'])
        norm = LinearNormalizer()
        norm['action'] = Field.create_fit(np.asarray(self.action[ids]), mode='limits', range_eps=1e-8)
        state = self.state[ids]
        if self.cfg['input_variant']=='official':
            norm['robot0_eef_pos'] = Field.create_fit(state[:,:3], mode='limits')
            norm['robot0_eef_quat'] = Field.create_identity()
            norm['robot0_gripper_qpos'] = Field.create_fit(state[:,7:9], mode='limits')
        else:
            norm['state'] = Field.create_fit(encode_state(state, ROT6D), mode='limits')
            with torch.no_grad():
                norm.params_dict['state']['scale'][22:28] = 1.
                norm.params_dict['state']['offset'][22:28] = 0.
        for key in self.cfg['cameras']:
            norm[key] = get_image_range_normalizer() if self.cfg['input_variant']=='official' else Field.create_identity()
        return norm


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)
