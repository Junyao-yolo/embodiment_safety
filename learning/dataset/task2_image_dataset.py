"""Episode-based Task2 samples using the official Diffusion Policy sampler."""

import copy
import json
import zlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)

CAMERAS = ("external", "wrist_left", "wrist_right")


def joint_delta_actions(state, absolute):
    targets = np.asarray(absolute, dtype=np.float32).reshape(-1, 6, 8)
    preceding = np.concatenate(
        [np.asarray(state)[:, None, 15:22], targets[:, :-1, :7]], axis=1
    )
    result = targets.copy()
    result[:, :, :7] -= preceding
    return result.reshape(-1, 48)


@lru_cache(maxsize=1)
def load_prepared(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    cache = json.loads((directory / "cache.json").read_text())
    if cache["status"] != "COMPLETE":
        raise ValueError("The image cache is incomplete")
    if [row["path"] for row in manifest["episodes"]] != [
        row["path"] for row in cache["episodes"]
    ]:
        raise ValueError("Image cache and episode manifest have different orders")
    index = np.load(directory / "index.npy")
    blob = np.fromfile(directory / "rgb.bin", dtype=np.uint8)
    blob.flags.writeable = False
    return manifest, index, blob


class PackedRGBArray:
    """Expose lossless compressed RGB frames as a time-indexed array."""

    dtype = np.dtype("uint8")

    def __init__(self, index, blob, camera):
        self.index = index[:, camera]
        self.blob = blob
        self.shape = (len(index), 128, 128, 3)

    def __getitem__(self, selection):
        ids = (
            np.arange(*selection.indices(self.shape[0]))
            if isinstance(selection, slice)
            else np.atleast_1d(selection)
        )
        frames = []
        for index in ids:
            offset, size = self.index[index]
            residual = np.frombuffer(
                zlib.decompress(self.blob[offset : offset + size]), dtype=np.uint8
            ).reshape(128, 128, 3)
            frames.append(np.cumsum(residual, axis=1, dtype=np.uint8))
        return np.stack(frames)


class Task2ImageDataset(BaseImageDataset):
    def __init__(
        self, prepared_dir, horizon=16, n_obs_steps=2, pad_before=1, pad_after=7
    ):
        self.horizon = int(horizon)
        self.n_obs_steps = int(n_obs_steps)
        self.pad_before = int(pad_before)
        self.pad_after = int(pad_after)
        manifest, index, blob = load_prepared(str(Path(prepared_dir).resolve()))
        self.episodes = manifest["episodes"]
        states, actions, ends = [], [], []
        total = 0
        for row in self.episodes:
            episode = Path(row["path"])
            state = np.load(episode / "state.npy")
            absolute = np.load(episode / "action.npy")
            if state.shape != (len(state), 23) or absolute.shape != (len(state), 48):
                raise ValueError(f"Invalid observation/action shape: {episode}")
            states.append(state)
            actions.append(joint_delta_actions(state, absolute))
            total += len(state)
            ends.append(total)
        data = {
            "state": np.concatenate(states),
            "action": np.concatenate(actions),
            **{key: PackedRGBArray(index, blob, i) for i, key in enumerate(CAMERAS)},
        }
        self.replay_buffer = ReplayBuffer(
            {
                "data": data,
                "meta": {"episode_ends": np.asarray(ends, dtype=np.int64)},
            }
        )
        self.train_mask = np.asarray(
            [row["split"] == "train" for row in self.episodes], dtype=bool
        )
        self.sampler = self._make_sampler(self.train_mask)

    def _make_sampler(self, mask):
        return SequenceSampler(
            self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=mask,
            key_first_k={key: self.n_obs_steps for key in (*CAMERAS, "state")},
        )

    def get_validation_dataset(self):
        result = copy.copy(self)
        result.sampler = self._make_sampler(~self.train_mask)
        return result

    def _training_rows(self, key):
        ends = self.replay_buffer.episode_ends
        starts = np.r_[0, ends[:-1]]
        return np.concatenate(
            [
                self.replay_buffer[key][start:end]
                for start, end, keep in zip(starts, ends, self.train_mask)
                if keep
            ]
        )

    def get_normalizer(self, **kwargs):
        normalizer = LinearNormalizer()
        for key in ("state", "action"):
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self._training_rows(key), mode="limits"
            )
        for key in CAMERAS:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self):
        return torch.from_numpy(self._training_rows("action"))

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, index):
        sample = self.sampler.sample_sequence(index)
        obs = {
            key: torch.from_numpy(
                np.moveaxis(sample[key][: self.n_obs_steps], -1, 1).copy()
            ).float()
            / 255.0
            for key in CAMERAS
        }
        obs["state"] = torch.from_numpy(
            sample["state"][: self.n_obs_steps].copy()
        ).float()
        return {
            "obs": obs,
            "action": torch.from_numpy(sample["action"].copy()).float(),
        }
