"""Download the ToolHang member of the official Diffusion Policy archive."""
import concurrent.futures
import json
import os
from pathlib import Path
import struct
import time
import zlib

import requests

ROOT = Path('/mnt/data2/wjy_SERL/Embodied_Safety')
OUT = ROOT / 'outputs/datasets/robomimic/tool_hang/ph'
URL = 'https://diffusion-policy.cs.columbia.edu/data/training/robomimic_image.zip'
OFFSET = int('00000001196ed2fd', 16)
COMPRESSED = int('000000053291b912', 16)
UNCOMPRESSED = int('0000000f7d8d39d0', 16)
BLOCK = 32 * 1024 * 1024


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / 'image_abs.hdf5'
    if target.exists() and target.stat().st_size == UNCOMPRESSED:
        return
    r = requests.get(URL, headers={'Range': f'bytes={OFFSET}-{OFFSET+1023}'}, timeout=60)
    r.raise_for_status()
    h = struct.unpack('<4s5H3I2H', r.content[:30])
    assert h[0] == b'PK\x03\x04' and h[3] == 8, h
    assert b'tool_hang/ph/image_abs.hdf5' in r.content[:30+h[-2]]
    start = OFFSET + 30 + h[-2] + h[-1]
    packed = OUT / 'image_abs.deflate.partial'
    state = OUT / 'download_state.json'
    done = set(json.loads(state.read_text()).get('blocks', [])) if state.exists() else set()
    fd = os.open(packed, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, COMPRESSED)
    begin = time.monotonic()
    initial = sum(min(BLOCK, COMPRESSED-i*BLOCK) for i in done)

    def fetch(i):
        n = min(BLOCK, COMPRESSED-i*BLOCK)
        for retry in range(5):
            try:
                rr = requests.get(URL, headers={'Range': f'bytes={start+i*BLOCK}-{start+i*BLOCK+n-1}'}, timeout=(30, 180))
                rr.raise_for_status()
                assert rr.status_code == 206 and len(rr.content) == n
                os.pwrite(fd, rr.content, i*BLOCK)
                return i
            except Exception:
                if retry == 4:
                    raise
                time.sleep(2 ** retry)

    count = (COMPRESSED+BLOCK-1)//BLOCK
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(fetch, i) for i in range(count) if i not in done]
        for fut in concurrent.futures.as_completed(futures):
            done.add(fut.result())
            size = sum(min(BLOCK, COMPRESSED-i*BLOCK) for i in done)
            elapsed = time.monotonic()-begin
            row = dict(blocks=sorted(done), completed_bytes=size, total_bytes=COMPRESSED,
                       elapsed_seconds=elapsed, mib_per_second=(size-initial)/max(elapsed,1)/2**20,
                       source=URL, member='robomimic/datasets/tool_hang/ph/image_abs.hdf5')
            state.write_text(json.dumps(row, indent=2))
            if len(done) % 10 == 0 or len(done) == count:
                print(json.dumps({k:v for k,v in row.items() if k != 'blocks'}), flush=True)
    os.close(fd)
    dec = zlib.decompressobj(-15)
    tmp = OUT / 'image_abs.hdf5.partial'
    with packed.open('rb') as src, tmp.open('wb') as dst:
        while block := src.read(1024*1024):
            dst.write(dec.decompress(block))
        dst.write(dec.flush())
    assert dec.eof and tmp.stat().st_size == UNCOMPRESSED
    tmp.rename(target)
    print(json.dumps(dict(event='dataset_downloaded', path=str(target), seconds=time.monotonic()-begin)), flush=True)


if __name__ == '__main__':
    main()
