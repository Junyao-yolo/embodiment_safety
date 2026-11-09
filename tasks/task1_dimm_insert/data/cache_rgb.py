"""Build the lossless RGB cache consumed by Task1ImageDataset."""

import argparse
import json
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

CAMERAS = ("external", "wrist_left", "wrist_right")


def encode(image):
    residual = image.copy()
    residual[:, 1:] -= image[:, :-1]
    return zlib.compress(residual.tobytes(), 3)


def build_cache(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    count = len(np.load(source / "state.npy", mmap_mode="r"))
    index = np.zeros((3, count, 2), np.int64)
    position = 0
    started = time.monotonic()
    with (
        (output / "rgb.bin").open("wb") as stream,
        ThreadPoolExecutor(max_workers=6) as pool,
    ):
        for camera, name in enumerate(CAMERAS):
            images = np.load(source / f"{name}.npy", mmap_mode="r")
            for begin in range(0, count, 256):
                end = min(count, begin + 256)
                blobs = pool.map(
                    encode, (np.array(images[i]) for i in range(begin, end))
                )
                for i, blob in zip(range(begin, end), blobs):
                    index[camera, i] = (position, len(blob))
                    stream.write(blob)
                    position += len(blob)
                if begin % 4096 == 0:
                    progress = dict(
                        frames_done=camera * count + end,
                        frames_total=3 * count,
                        bytes_written=position,
                        elapsed_s=time.monotonic() - started,
                    )
                    (output / "status.json").write_text(json.dumps(progress, indent=2))
                    print(json.dumps(progress), flush=True)
    np.save(output / "index.npy", index)
    meta = dict(
        status="finished",
        source=str(source.resolve()),
        rows=count,
        cameras=CAMERAS,
        shape=[256, 256, 3],
        dtype="uint8",
        codec="horizontal uint8 residual plus zlib level3",
        bytes=position,
        elapsed_s=time.monotonic() - started,
    )
    (output / "manifest.json").write_text(json.dumps(meta, indent=2))
    (output / "status.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_cache(args.source, args.output or args.source / "lossless_rgb_cache")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
