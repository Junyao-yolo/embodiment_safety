"""Check real episode boundaries, image alignment and action decoding."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from learning.bootstrap import ROOT
from learning.dataset.task2_image_dataset import (
    CAMERAS,
    Task2ImageDataset,
    joint_delta_actions,
)
from tasks.task2_riser_io.sim.actions import decode_action

PREPARED = ROOT / "outputs/task2_riser_io/data/opposed_v002/prepared"


@pytest.fixture(scope="module")
def dataset():
    return Task2ImageDataset(PREPARED)


def test_group_split_and_window_counts(dataset):
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    train = {row["reset_group"] for row in dataset.episodes if row["split"] == "train"}
    validation = {
        row["reset_group"] for row in dataset.episodes if row["split"] == "validation"
    }
    assert train and validation and train.isdisjoint(validation)
    assert len(dataset) == manifest["counts"]["train"]["windows"]
    assert (
        len(dataset.get_validation_dataset())
        == manifest["counts"]["validation"]["windows"]
    )


@pytest.mark.parametrize("split", ["train", "validation"])
def test_samples_match_raw_episode_at_boundaries(dataset, split):
    selected = dataset if split == "train" else dataset.get_validation_dataset()
    for index in (0, 1, len(selected) // 2, len(selected) - 1):
        start, end, sample_start, sample_end = selected.sampler.indices[index]
        ep = int(
            np.searchsorted(dataset.replay_buffer.episode_ends, start, side="right")
        )
        base = int(dataset.replay_buffer.episode_ends[ep - 1]) if ep else 0
        path = Path(dataset.episodes[ep]["path"])
        state = np.load(path / "state.npy", mmap_mode="r")
        absolute = np.load(path / "action.npy", mmap_mode="r")
        ids = np.clip(start - base - sample_start + np.arange(16), 0, len(state) - 1)
        sample = selected[index]
        torch.testing.assert_close(
            sample["obs"]["state"], torch.from_numpy(state[ids[:2]].copy())
        )
        torch.testing.assert_close(
            sample["action"],
            torch.from_numpy(joint_delta_actions(state, absolute)[ids]),
        )
        for camera in CAMERAS:
            raw = np.load(path / f"{camera}.npy", mmap_mode="r")
            expected = (
                torch.from_numpy(raw[ids[:2]].copy()).permute(0, 3, 1, 2).float()
                / 255.0
            )
            torch.testing.assert_close(sample["obs"][camera], expected, rtol=0, atol=0)


def test_action_roundtrip_and_training_only_normalizer(dataset):
    for episode in (dataset.episodes[0], dataset.episodes[-1]):
        path = Path(episode["path"])
        state = np.load(path / "state.npy")
        action = np.load(path / "action.npy")
        row = len(state) // 2
        delta = joint_delta_actions(state[row : row + 1], action[row : row + 1])[0]
        decoded = decode_action(delta, state[row, 15:22])
        expected = action[row].reshape(6, 8)
        np.testing.assert_allclose(decoded[:, :7], expected[:, :7], rtol=0, atol=2e-6)
        np.testing.assert_allclose(
            decoded[:, 7] + decoded[:, 8], expected[:, 7], rtol=0, atol=1e-7
        )
    normalizer = dataset.get_normalizer()
    for key in ("state", "action"):
        expected = dataset._training_rows(key)
        stats = normalizer[key].get_input_stats()
        np.testing.assert_allclose(stats["min"].numpy(), expected.min(axis=0))
        np.testing.assert_allclose(stats["max"].numpy(), expected.max(axis=0))
