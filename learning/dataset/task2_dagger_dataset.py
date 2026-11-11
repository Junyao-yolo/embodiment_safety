"""Balanced DAgger and demonstration windows with frozen Task2 normalization."""
import copy,json
from functools import lru_cache
from pathlib import Path
import numpy as np
import torch
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from learning.dataset.task2_image_dataset import CAMERAS,joint_delta_actions

@lru_cache(maxsize=24)
def image_array(path):
    return np.load(path,mmap_mode="r")

class EpisodeRGBArray:
    dtype=np.dtype("uint8")
    def __init__(self,paths,ends,camera):
        self.paths=paths;self.ends=np.asarray(ends);self.starts=np.r_[0,self.ends[:-1]]
        self.camera=camera;self.shape=(int(self.ends[-1]),128,128,3)
    def __getitem__(self,selection):
        ids=np.arange(*selection.indices(self.shape[0])) if isinstance(selection,slice) else np.atleast_1d(selection)
        rows=[]
        for i in ids:
            e=int(np.searchsorted(self.ends,i,side="right"))
            rows.append(image_array(str(Path(self.paths[e])/(self.camera+".npy")))[i-self.starts[e]])
        return np.stack(rows)

class Task2DaggerDataset(BaseImageDataset):
    def __init__(self,manifest,normalizer_path,horizon=16,n_obs_steps=2,pad_before=1,pad_after=3,epoch_samples=3200,validation_samples=512,seed=42):
        self.n_obs_steps=n_obs_steps;self.epoch_samples=epoch_samples
        self.validation_samples=validation_samples;self.seed=seed;self.is_validation=False
        self.episodes=json.loads(Path(manifest).read_text())["episodes"]
        states=[];actions=[];ends=[];total=0
        for row in self.episodes:
            p=Path(row["path"]);x=np.load(p/"state.npy");a=np.load(p/"action.npy")
            states.append(x);actions.append(joint_delta_actions(x,a));total+=len(x);ends.append(total)
        self.replay_buffer=ReplayBuffer(dict(data=dict(state=np.concatenate(states),action=np.concatenate(actions),
            **{k:EpisodeRGBArray([e["path"] for e in self.episodes],ends,k) for k in CAMERAS}),
            meta=dict(episode_ends=np.asarray(ends,dtype=np.int64))))
        self.train_mask=np.asarray([x["split"]=="train" for x in self.episodes])
        options=dict(sequence_length=horizon,pad_before=pad_before,pad_after=pad_after,
            key_first_k={k:n_obs_steps for k in (*CAMERAS,"state")})
        self.sampler=SequenceSampler(self.replay_buffer,episode_mask=self.train_mask,**options)
        self.val_sampler=SequenceSampler(self.replay_buffer,episode_mask=~self.train_mask,**options)
        episode_ids=np.searchsorted(ends,self.sampler.indices[:,0],side="right")
        correction=np.asarray([x["source"]=="dagger" for x in self.episodes])[episode_ids]
        self.original_indices=np.flatnonzero(~correction);self.correction_indices=np.flatnonzero(correction)
        if not len(self.correction_indices) or not len(self.original_indices):raise ValueError("Both replay and correction windows are required")
        self.val_indices=np.random.default_rng(seed).choice(len(self.val_sampler),size=min(validation_samples,len(self.val_sampler)),replace=False)
        self.normalizer=LinearNormalizer()
        self.normalizer.load_state_dict(torch.load(normalizer_path,map_location="cpu",weights_only=False))
    def get_normalizer(self,**kwargs):return self.normalizer
    def get_validation_dataset(self):
        other=copy.copy(self);other.is_validation=True;return other
    def get_all_actions(self):return torch.from_numpy(self.replay_buffer["action"])
    def __len__(self):
        return len(self.val_indices) if self.is_validation else self.epoch_samples
    def __getitem__(self,index):
        if self.is_validation:sample=self.val_sampler.sample_sequence(self.val_indices[index])
        else:
            pool=self.correction_indices if index%2 else self.original_indices
            sample=self.sampler.sample_sequence(pool[np.random.randint(len(pool))])
        obs={k:torch.from_numpy(np.moveaxis(sample[k][:self.n_obs_steps],-1,1).copy()).float()/255 for k in CAMERAS}
        obs["state"]=torch.from_numpy(sample["state"][:self.n_obs_steps].copy()).float()
        return dict(obs=obs,action=torch.from_numpy(sample["action"].copy()).float())
