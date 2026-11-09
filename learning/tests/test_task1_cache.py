"""A prepared cache must preserve each camera's row and pixels."""

import json
import numpy as np
from learning.dataset.task1_image_dataset import Task1RGB
from tasks.task1_dimm_insert.data.cache_rgb import CAMERAS, build_cache


def test_cache_pixels_and_rows(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    np.save(source / "state.npy", np.zeros((2, 31), np.float32))
    rng = np.random.default_rng(42)
    expected = {}
    for key in CAMERAS:
        expected[key] = rng.integers(0, 256, (2, 256, 256, 3), dtype=np.uint8)
        np.save(source / f"{key}.npy", expected[key])
    cache = tmp_path / "cache"
    build_cache(source, cache)
    meta = json.loads((cache / "manifest.json").read_text())
    assert meta["status"] == "finished" and meta["rows"] == 2
    index = np.load(cache / "index.npy")
    blob = np.memmap(cache / "rgb.bin", mode="r", dtype=np.uint8)
    for i, key in enumerate(CAMERAS):
        actual = Task1RGB(index, blob, i)
        np.testing.assert_array_equal(actual[0:2], expected[key])
        np.testing.assert_array_equal(actual[1:2], expected[key][1:2])
