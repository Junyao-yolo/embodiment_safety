"""Task1 image sequences with whole-episode splits and the official sampler."""

import copy
import json
from pathlib import Path
import zlib
import numpy as np
import torch
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer as Field,
)
from learning.common.representation import encode_state, ROT6D
from learning.dataset.task1_observation import CAMERAS, observation


class Task1RGB:
    dtype = np.dtype("uint8")

    def __init__(self, index, blob, camera):
        self.index = index[camera]
        self.blob = blob
        self.shape = (len(self.index), 256, 256, 3)

    def __getitem__(self, selection):
        frames = []
        for i in range(*selection.indices(self.shape[0])):
            offset, size = self.index[i]
            residual = np.frombuffer(
                zlib.decompress(self.blob[offset : offset + size]), dtype=np.uint8
            ).reshape(256, 256, 3)
            frames.append(np.cumsum(residual, axis=1, dtype=np.uint8))
        return np.stack(frames)


class Task1ImageDataset(BaseImageDataset):
    def __init__(
        self,
        dataset_path,
        input_variant="current",
        proprio_path=None,
        horizon=16,
        n_obs_steps=2,
    ):
        self.root = Path(dataset_path)
        self.variant = input_variant
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.meta = json.loads((self.root / "manifest.json").read_text())
        cache = self.root / "lossless_rgb_cache"
        cache_meta = json.loads((cache / "manifest.json").read_text())
        if (
            cache_meta["status"] != "finished"
            or cache_meta["rows"] != self.meta["rows"]
        ):
            raise ValueError("Task1 RGB cache is incomplete")
        index = np.load(cache / "index.npy", mmap_mode="r")
        blob = np.memmap(cache / "rgb.bin", mode="r", dtype=np.uint8)
        state_path = (
            Path(proprio_path)
            if self.variant == "official"
            else self.root / "state.npy"
        )
        self.state = np.load(state_path, mmap_mode="r")
        self.action = np.load(self.root / "action.npy", mmap_mode="r")
        self.cameras = (
            ("external", "wrist_right") if self.variant == "official" else CAMERAS
        )
        ends = [row["offset"] + row["rows"] for row in self.meta["episodes"]]
        self.replay_buffer = ReplayBuffer(
            dict(
                data={
                    "state": self.state,
                    "action": self.action,
                    **{
                        key: Task1RGB(index, blob, CAMERAS.index(key))
                        for key in self.cameras
                    },
                },
                meta={"episode_ends": np.asarray(ends, dtype=np.int64)},
            )
        )
        self.train_mask = np.asarray(
            [row["split"] == "train" for row in self.meta["episodes"]]
        )
        self.sampler = self._sampler(self.train_mask)

    def _sampler(self, mask):
        return SequenceSampler(
            self.replay_buffer,
            sequence_length=self.horizon,
            episode_mask=mask,
            pad_before=0,
            pad_after=0,
            key_first_k={key: self.n_obs_steps for key in (*self.cameras, "state")},
        )

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.sampler = self._sampler(~self.train_mask)
        return result

    def get_normalizer(self, **kwargs):
        ids = np.concatenate(
            [
                np.arange(row["offset"], row["offset"] + row["rows"])
                for row in self.meta["episodes"]
                if row["split"] == "train"
            ]
        )
        norm = LinearNormalizer()
        norm["action"] = Field.create_fit(
            np.asarray(self.action[ids]), mode="limits", range_eps=1e-8
        )
        state = self.state[ids]
        if self.variant == "official":
            norm["robot0_eef_pos"] = Field.create_fit(state[:, :3], mode="limits")
            norm["robot0_eef_quat"] = Field.create_identity()
            norm["robot0_gripper_qpos"] = Field.create_fit(state[:, 7:9], mode="limits")
        else:
            norm["state"] = Field.create_fit(encode_state(state, ROT6D), mode="limits")
            with torch.no_grad():
                norm.params_dict["state"]["scale"][22:28] = 1
                norm.params_dict["state"]["offset"][22:28] = 0
        for key in self.cameras:
            norm[key] = (
                get_image_range_normalizer()
                if self.variant == "official"
                else Field.create_identity()
            )
        return norm

    def get_all_actions(self):
        return torch.from_numpy(
            np.concatenate(
                [
                    self.action[row["offset"] : row["offset"] + row["rows"]]
                    for row in self.meta["episodes"]
                    if row["split"] == "train"
                ]
            )
        )

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        sample = self.sampler.sample_sequence(index)
        obs = observation(
            {key: sample[key][: self.n_obs_steps] for key in self.cameras},
            sample["state"][: self.n_obs_steps],
            self.variant,
        )
        return {"obs": obs, "action": torch.from_numpy(sample["action"].copy())}
