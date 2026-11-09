"""Isolated DP timing with full-data sampling and preserved training state."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import resource
import sys
import time
from contextlib import contextmanager

ROOT = Path('/mnt/data2/wjy_SERL/Embodied_Safety')
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from algos.diffusion_policy.common import build, CAMERAS
from algos.diffusion_policy.data import SequenceData
from algos.diffusion_policy.sampling import ResumableRandomSampler
from tools.task1_train_throughput_v001.cache import decode


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


class MeasuredData(Dataset):
    def __init__(self, source, cache=None, compact=False):
        self.source = source
        self.compact = compact
        self.blob = None
        if cache:
            cache = Path(cache)
            meta = json.loads((cache/'manifest.json').read_text())
            assert meta['source'] == str(source.path.resolve())
            assert meta['rows'] == len(source.state)
            self.blob = np.memmap(cache/'rgb.bin', dtype=np.uint8, mode='r')
            self.index = np.load(cache/'index.npy', mmap_mode='r')

    def __len__(self):
        return len(self.source)

    def __getitem__(self, i):
        from algos.diffusion_policy.representation import encode_state
        k = self.source.starts[i]
        obs = {}
        read_s = decode_s = transform_s = 0.
        reads = decodes = 0
        for ci, camera in enumerate(CAMERAS):
            start = time.perf_counter()
            if self.blob is None:
                pixels = np.array(self.source.images[camera][k:k+2], copy=True)
                reads += pixels.nbytes
                read_s += time.perf_counter() - start
            else:
                frames = []
                for row in (k, k+1):
                    start = time.perf_counter()
                    offset, size = self.index[ci, row]
                    blob = self.blob[offset:offset+size].tobytes()
                    reads += int(size)
                    read_s += time.perf_counter() - start
                    start = time.perf_counter()
                    frames.append(decode(blob))
                    decode_s += time.perf_counter() - start
                    decodes += 1
                pixels = np.stack(frames)
            start = time.perf_counter()
            value = torch.from_numpy(pixels).permute(0, 3, 1, 2)
            obs[camera] = value if self.compact else value.float()/255.
            transform_s += time.perf_counter() - start
        obs['state'] = torch.from_numpy(encode_state(self.source.state[k:k+2], self.source.representation))
        return dict(obs=obs, action=torch.from_numpy(np.array(self.source.action[k:k+self.source.horizon], copy=True)),
                    measurement=np.array([i, read_s, decode_s, transform_s, reads, decodes], np.float64))


def device(batch):
    if isinstance(batch, dict):
        return {k: device(v) for k, v in batch.items()}
    value = batch.cuda(non_blocking=True)
    return value.float()/255. if value.dtype == torch.uint8 else value


def io_snapshot(pids):
    result = {}
    for pid in pids:
        try:
            values = dict(line.split(':', 1) for line in Path(f'/proc/{pid}/io').read_text().splitlines())
            stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
            result[str(pid)] = {k: int(v.strip()) for k, v in values.items()}
            result[str(pid)].update(minor_faults=int(stat[7]), major_faults=int(stat[9]))
        except FileNotFoundError:
            pass
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache')
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    cp = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = cp['config']
    source = SequenceData(cfg['data'], 'train', cfg['state_representation'], cfg['scope'])
    data = MeasuredData(source, args.cache, args.compact)
    # Compare real sampled windows before timing; no samples are removed or replaced.
    order = cp.get('sampler', {}).get('order')
    cursor = cp.get('sampler', {}).get('cursor', 0)
    checks = order[cursor:cursor+8] if order else list(range(8))
    input_error = 0.
    for i in checks:
        ref, actual = source[i], data[i]
        assert torch.equal(ref['action'], actual['action'])
        for name in ref['obs']:
            value = actual['obs'][name]
            if value.dtype == torch.uint8:
                value = (value.cuda().float()/255.).cpu()
            input_error = max(input_error, (ref['obs'][name]-value).abs().max().item())
    assert input_error <= 1e-7, input_error
    model = build(cfg)
    model.load_state_dict(cp['model'])
    model = model.cuda()
    ema = copy.deepcopy(model).eval()
    ema.load_state_dict(cp['ema'])
    ema = ema.cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    optimizer.load_state_dict(cp['optimizer'])
    sampler = ResumableRandomSampler(data, cp.get('sampler'))
    torch.set_rng_state(cp['torch_rng'])
    torch.cuda.set_rng_state_all(cp['cuda_rng'])
    loader = DataLoader(data, batch_size=cfg['batch_size'], sampler=sampler,
                        num_workers=2, pin_memory=True, persistent_workers=True)
    iterator = iter(loader)
    if 'sampler' in cp:
        torch.set_rng_state(cp['torch_rng'])
        torch.cuda.set_rng_state_all(cp['cuda_rng'])
        random.setstate(cp['python_rng'])
        np.random.set_state(cp['numpy_rng'])
    start_step = cp['step']
    best = cp['best']
    del cp
    parameter_count = sum(p.numel() for p in model.parameters())
    debug = dict(cuda_launch_blocking=os.environ.get('CUDA_LAUNCH_BLOCKING'),
                 anomaly_detection=torch.is_anomaly_enabled(), cudnn_benchmark=torch.backends.cudnn.benchmark,
                 cudnn_deterministic=torch.backends.cudnn.deterministic,
                 deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                 matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                 cudnn_tf32=torch.backends.cudnn.allow_tf32, cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 torch_version=torch.__version__, cuda_version=torch.version.cuda,
                 device=torch.cuda.get_device_name(), threads=torch.get_num_threads())
    write(out/'config.json', dict(config=cfg, checkpoint=args.checkpoint, cache=args.cache,
          compact=args.compact, train_windows=len(data), workers=2, prefetch_factor=2,
          warmup=200, stable=500, input_max_abs_error=input_error,
          parameter_count=parameter_count, runtime=debug))
    rows, events, selected = [], [], []
    current = {}

    @contextmanager
    def region(name, gpu=False):
        if gpu:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
        t = time.perf_counter()
        yield
        current[name+'_host_ms'] = (time.perf_counter()-t)*1000
        if gpu:
            end.record()
            events.append((len(rows), name, begin, end))

    worker_pids = [w.pid for w in iterator._workers]
    write(out/'progress.json', dict(phase='warmup', pid=os.getpid(), workers=worker_pids, update=0))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    run_start = started
    stable_io = None
    with (out/'metrics.jsonl').open('w') as log:
        for relative in range(1, 701):
            if relative == 201:
                torch.cuda.synchronize()
                warmup_s = time.perf_counter()-started
                stable_io = io_snapshot([os.getpid()]+worker_pids)
                stable_timestamp_start = time.time()
                started = time.perf_counter()
            update_start = time.perf_counter()
            current = dict(relative=relative, step=start_step+relative, stable=relative>200)
            with region('data_wait'):
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
            with region('instrumentation'):
                measurement = batch.pop('measurement').numpy()
                current['samples'] = len(measurement)
                selected.extend(measurement[:, 0].astype(int).tolist())
                current.update(worker_read_ms=float(measurement[:, 1].sum()*1000),
                               worker_decode_ms=float(measurement[:, 2].sum()*1000),
                               worker_transform_ms=float(measurement[:, 3].sum()*1000),
                               logical_rgb_bytes=int(measurement[:, 4].sum()),
                               image_decode_calls=int(measurement[:, 5].sum()))
            with region('h2d', gpu=True):
                batch = device(batch)
            with region('setup', gpu=True):
                model.train()
                optimizer.zero_grad(set_to_none=True)
            with region('forward', gpu=True):
                loss = model.compute_loss(batch)
            with region('finite_check', gpu=True):
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite loss')
            with region('backward', gpu=True):
                loss.backward()
            with region('gradient_clip', gpu=True):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            with region('optimizer', gpu=True):
                optimizer.step()
            with region('sampler'):
                sampler.advance(batch['action'].shape[0])
            with region('ema', gpu=True):
                decay = min(.9999, 1-(1+start_step+relative)**(-.75))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.lerp_(pm, 1-decay)
                    for be, bm in zip(ema.buffers(), model.buffers()):
                        be.copy_(bm)
            with region('loss_to_cpu', gpu=True):
                loss_value = float(loss)
            with region('log_io'):
                rec = dict(step=start_step+relative, loss=loss_value, elapsed_s=time.perf_counter()-run_start)
                log.write(json.dumps(rec)+'\n')
                log.flush()
                if relative % 100 == 0:
                    print(json.dumps(rec), flush=True)
            current['update_wall_ms'] = (time.perf_counter()-update_start)*1000
            current['loss'] = loss_value
            rows.append(current)
            if relative % 100 == 0:
                write(out/'progress.json', dict(phase='stable' if relative>200 else 'warmup',
                      pid=os.getpid(), workers=worker_pids, update=relative))
        torch.cuda.synchronize()
        stable_s = time.perf_counter()-started
        stable_timestamp_end = time.time()
        io_end = io_snapshot([os.getpid()]+worker_pids)
    for i, name, begin, end in events:
        rows[i][name+'_cuda_ms'] = begin.elapsed_time(end)
    # Timed regions are sequential on the host; CUDA durations overlap host waits.
    stable = rows[200:]
    numeric = [k for k in stable[0] if k.endswith('_ms')]
    summary = {k: dict(mean=float(np.mean([r[k] for r in stable])),
                      median=float(np.median([r[k] for r in stable])),
                      p95=float(np.percentile([r[k] for r in stable], 95))) for k in numeric}
    io_delta = {p: {k: io_end[p][k]-v for k, v in vals.items()} for p, vals in stable_io.items() if p in io_end}
    timing = dict(warmup_updates=200, stable_updates=500, warmup_s=warmup_s,
                  stable_timestamp_start=stable_timestamp_start, stable_timestamp_end=stable_timestamp_end,
                  stable_s=stable_s, update_mean_ms=stable_s/500*1000,
                  stable_windows=sum(r['samples'] for r in stable),
                  windows_per_second=sum(r['samples'] for r in stable)/stable_s, batch_size=cfg['batch_size'],
                  cache=args.cache, compact=args.compact, regions=summary, io_delta=io_delta,
                  stable_decodes=sum(r['image_decode_calls'] for r in stable),
                  stable_logical_rgb_bytes=sum(r['logical_rgb_bytes'] for r in stable),
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                  start_step=start_step, end_step=start_step+700,
                  timing_scope='Host regions partition update wall time; CUDA events overlap host. No per-region synchronization added. Validation and saving excluded.')
    write(out/'timing.json', timing)
    with (out/'update_timing.jsonl').open('w') as f:
        for r in rows:
            f.write(json.dumps(r)+'\n')
    np.save(out/'sample_indices.npy', np.array(selected, dtype=np.int64))
    write(out/'progress.json', dict(phase='validation', pid=os.getpid(), update=700))
    # Use the checkpoint's full diagnostic subsets, outside update timing.
    @torch.no_grad()
    def evaluate(split, chosen):
        dataset = SequenceData(cfg['data'], split, cfg['state_representation'], cfg['scope'])
        measured = MeasuredData(dataset, args.cache, args.compact)
        evalloader = DataLoader(Subset(measured, chosen), batch_size=cfg['batch_size'], num_workers=0)
        predictions, targets, losses = [], [], []
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            for batch in evalloader:
                batch.pop('measurement')
                batch = device(batch)
                losses.append(float(ema.compute_loss(batch)))
                predictions.append(ema.predict_action(batch['obs'])['action'][:, 0].cpu().numpy())
                targets.append(batch['action'][:, 1].cpu().numpy())
        error = np.concatenate(predictions)-np.concatenate(targets)
        return dict(samples=len(chosen), action_mse=float(np.mean(error**2)),
                    per_axis_rmse=np.sqrt(np.mean(error**2, axis=0)).tolist(), diffusion_loss=float(np.mean(losses)))
    torch.cuda.synchronize()
    t = time.perf_counter()
    validation = evaluate('validation', cfg['validation_subset'])
    fit_probe = evaluate('train', cfg['fit_probe_subset'])
    torch.cuda.synchronize()
    timing['validation_s'] = time.perf_counter()-t
    timing['validation'] = validation
    timing['fit_probe'] = fit_probe
    payload = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                   step=start_step+700, best=best, config=cfg, sampler=sampler.state_dict(),
                   torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                   python_rng=random.getstate(), numpy_rng=np.random.get_state())
    t = time.perf_counter()
    with (out/'benchmark_only.pt').open('wb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    timing['checkpoint_save_fsync_s'] = time.perf_counter()-t
    write(out/'timing.json', timing)
    write(out/'progress.json', dict(phase='complete', pid=os.getpid(), update=700,
          formal_training_advanced=False))


if __name__ == '__main__':
    main()
