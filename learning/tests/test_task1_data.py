"""Verify Task1 samples, episode splits and runtime representation against recorded data."""

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from learning.bootstrap import ROOT
from learning.dataset.task1_image_dataset import Task1ImageDataset

DATA = ROOT / "outputs/datasets/task1/diffusion_parallel_diverse_new_wrist_v001"
PROPRIO = (
    ROOT
    / "outputs/runs/task1/diffusion_policy/official_input_compare_v001/proprio9.npy"
)


@pytest.fixture(scope="module", params=["current", "official"])
def dataset(request):
    return Task1ImageDataset(DATA, input_variant=request.param, proprio_path=PROPRIO)


def test_episode_split_and_samples(dataset):
    for split, selected in [
        ("train", dataset),
        ("validation", dataset.get_validation_dataset()),
    ]:
        np.testing.assert_array_equal(
            selected.sampler.indices[:, 0], dataset.meta[split]
        )
        for index in (0, len(selected) // 2, len(selected) - 1):
            start = int(selected.sampler.indices[index, 0])
            sample = selected[index]
            np.testing.assert_array_equal(
                sample["action"].numpy(), dataset.action[start : start + 16]
            )
            for camera in dataset.cameras:
                raw = np.load(DATA / f"{camera}.npy", mmap_mode="r")
                expected = (
                    torch.from_numpy(raw[start : start + 2].copy())
                    .permute(0, 3, 1, 2)
                    .float()
                    / 255
                )
                if dataset.variant == "official":
                    expected = torch.nn.functional.interpolate(
                        expected,
                        size=(240, 240),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    )
                torch.testing.assert_close(
                    sample["obs"][camera], expected, rtol=0, atol=0
                )
            if dataset.variant == "current":
                raw = dataset.state[start : start + 2]
                matrix = Rotation.from_quat(raw[:, 22:26]).as_matrix()
                expected = np.concatenate(
                    [raw[:, :22], matrix[:, :, 0], matrix[:, :, 1], raw[:, 26:]],
                    axis=-1,
                )
                np.testing.assert_allclose(
                    sample["obs"]["state"], expected, rtol=1e-6, atol=1e-6
                )
            else:
                np.testing.assert_array_equal(
                    sample["obs"]["robot0_eef_pos"],
                    dataset.state[start : start + 2, :3],
                )
    train = {
        row["episode"] for row in dataset.meta["episodes"] if row["split"] == "train"
    }
    valid = {
        row["episode"]
        for row in dataset.meta["episodes"]
        if row["split"] == "validation"
    }
    assert train.isdisjoint(valid)
    assert len(train) == 96 and len(valid) == 11


def test_normalization_uses_training_rows(dataset):
    normalizer = dataset.get_normalizer()
    ids = np.concatenate(
        [
            np.arange(row["offset"], row["offset"] + row["rows"])
            for row in dataset.meta["episodes"]
            if row["split"] == "train"
        ]
    )
    stats = normalizer["action"].get_input_stats()
    np.testing.assert_allclose(stats["min"], dataset.action[ids].min(axis=0))
    np.testing.assert_allclose(stats["max"], dataset.action[ids].max(axis=0))
    if dataset.variant == "current":
        np.testing.assert_array_equal(
            normalizer.params_dict["state"]["scale"][22:28], np.ones(6)
        )
        np.testing.assert_array_equal(
            normalizer.params_dict["state"]["offset"][22:28], np.zeros(6)
        )
