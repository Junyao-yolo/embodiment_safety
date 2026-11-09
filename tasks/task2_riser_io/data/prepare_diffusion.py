"""Prepare successful Task2 episodes, reset-group splits and a lossless RGB cache."""

from learning.bootstrap import configure

configure()

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import time
import zlib

import numpy as np
from learning.dataset.task2_image_dataset import (
    CAMERAS,
    joint_delta_actions,
)


def encode(frame):
    residual = np.empty_like(frame)
    residual[:, 0] = frame[:, 0]
    residual[:, 1:] = frame[:, 1:] - frame[:, :-1]
    return zlib.compress(residual.tobytes(), 3)


def prepare(source, output, seed, validation_fraction):
    output.mkdir(parents=True, exist_ok=False)
    episodes = []
    max_error = 0.0
    for path in sorted(source.glob("episode_*")):
        if not (path / "result.json").exists():
            continue
        result = json.loads((path / "result.json").read_text())
        if not result.get("accepted"):
            continue
        meta = json.loads((path / "episode.json").read_text())
        state = np.load(path / "state.npy", mmap_mode="r")
        action = np.load(path / "action.npy", mmap_mode="r")
        n = len(state)
        if state.shape != (n, 23) or action.shape != (n, 48) or n < 8:
            raise ValueError(f"Invalid episode shape or length: {path}")
        for camera in CAMERAS:
            if np.load(path / f"{camera}.npy", mmap_mode="r").shape != (n, 128, 128, 3):
                raise ValueError(f"Image/state alignment differs: {path} / {camera}")
        delta = joint_delta_actions(state, action).reshape(-1, 6, 8)
        reconstructed = state[:, None, 15:22] + np.cumsum(delta[:, :, :7], axis=1)
        max_error = max(
            max_error,
            float(np.abs(reconstructed - action.reshape(-1, 6, 8)[:, :, :7]).max()),
        )
        episodes.append(
            dict(
                path=str(path),
                samples=n,
                windows=n - 7,
                reset_group=int(meta["initial_conditions"]["reset_seed"]),
                route=meta["route_mode"],
            )
        )
    if not episodes or max_error >= 2e-6:
        raise ValueError("No successful aligned episodes, or action conversion failed")
    groups = np.array(sorted({row["reset_group"] for row in episodes}))
    if len(groups) < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Training and validation need distinct reset groups")
    count = min(len(groups) - 1, math.ceil(len(groups) * validation_fraction))
    validation_groups = set(
        np.random.default_rng(seed).permutation(groups)[:count].tolist()
    )
    for row in episodes:
        row["split"] = (
            "validation" if row["reset_group"] in validation_groups else "train"
        )
    counts = {
        split: dict(
            episodes=sum(row["split"] == split for row in episodes),
            windows=sum(row["windows"] for row in episodes if row["split"] == split),
            reset_groups=sorted(
                {row["reset_group"] for row in episodes if row["split"] == split}
            ),
            routes=dict(
                Counter(row["route"] for row in episodes if row["split"] == split)
            ),
        )
        for split in ("train", "validation")
    }
    manifest = dict(
        episodes=episodes,
        counts=counts,
        seed=seed,
        action_roundtrip_max_error_rad=max_error,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n = sum(row["samples"] for row in episodes)
    index = np.lib.format.open_memmap(
        output / "index.npy", mode="w+", dtype=np.int64, shape=(n, 3, 2)
    )
    started = time.monotonic()
    offset = 0
    cache_rows = []
    checks = 0
    with (
        (output / "rgb.bin").open("wb") as stream,
        ThreadPoolExecutor(max_workers=8) as pool,
    ):
        for episode in episodes:
            path = Path(episode["path"])
            length = episode["samples"]
            for c, camera in enumerate(CAMERAS):
                frames = np.load(path / f"{camera}.npy", mmap_mode="r")
                for left in range(0, length, 128):
                    chunk = np.asarray(frames[left : left + 128]).copy()
                    for j, blob in enumerate(pool.map(encode, chunk)):
                        frame_id = left + j
                        index[offset + frame_id, c] = [stream.tell(), len(blob)]
                        stream.write(blob)
                        if frame_id in (0, length - 1):
                            restored = np.cumsum(
                                np.frombuffer(zlib.decompress(blob), np.uint8).reshape(
                                    128, 128, 3
                                ),
                                axis=1,
                                dtype=np.uint8,
                            )
                            if not np.array_equal(restored, frames[frame_id]):
                                raise ValueError(f"RGB cache roundtrip failed: {path}")
                            checks += 1
            cache_rows.append(dict(path=str(path), offset=offset, samples=length))
            offset += length
            (output / "progress.json").write_text(
                json.dumps(
                    dict(
                        episodes=len(cache_rows),
                        total_episodes=len(episodes),
                        rows=offset,
                        wall_s=time.monotonic() - started,
                    ),
                    indent=2,
                )
            )
    index.flush()
    (output / "cache.json").write_text(
        json.dumps(
            dict(
                status="COMPLETE",
                rows=n,
                cameras=list(CAMERAS),
                episodes=cache_rows,
                encoding="horizontal uint8 delta + zlib level3",
                verified_roundtrip_frames=checks,
                cache_bytes=(output / "rgb.bin").stat().st_size,
            ),
            indent=2,
        )
    )
    print(json.dumps(counts, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()
    prepare(
        args.source.resolve(),
        args.output.resolve(),
        args.seed,
        args.validation_fraction,
    )


if __name__ == "__main__":
    main()
