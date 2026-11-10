"""Replay original demonstrations alongside complete, accepted teacher segments."""

import copy
import math
import torch
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from learning.dataset.task1_image_dataset import Task1ImageDataset


class Task1RecoveryDataset(BaseImageDataset):
    def __init__(self, dataset_path, recovery_path, normalizer_path, input_variant,
                 horizon=16, n_obs_steps=2):
        self.base = Task1ImageDataset(dataset_path, input_variant,
            proprio_path=str(dataset_path)+'/proprio9.npy', horizon=horizon, n_obs_steps=n_obs_steps)
        self.recovery = Task1ImageDataset(recovery_path, input_variant,
            proprio_path=str(recovery_path)+'/proprio9.npy', horizon=horizon, n_obs_steps=n_obs_steps)
        if len(self.recovery) == 0:
            raise ValueError('No accepted teacher training windows')
        self.normalizer_path = normalizer_path
        self.validation = False

    def __len__(self):
        return len(self.base)+len(self.recovery) if self.validation else 4*math.ceil(len(self.base)/3)

    def __getitem__(self, index):
        if self.validation:
            return self.base[index] if index < len(self.base) else self.recovery[index-len(self.base)]
        if index % 4 == 3:
            return self.recovery[(index//4) % len(self.recovery)]
        return self.base[(index//4*3+index%4) % len(self.base)]

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.base = self.base.get_validation_dataset()
        result.recovery = self.recovery.get_validation_dataset()
        result.validation = True
        return result

    def get_normalizer(self, **kwargs):
        normalizer = LinearNormalizer()
        normalizer.load_state_dict(torch.load(self.normalizer_path, map_location='cpu', weights_only=False))
        return normalizer

    def get_all_actions(self):
        return torch.cat([self.base.get_all_actions(), self.recovery.get_all_actions()])
