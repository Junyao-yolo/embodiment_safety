"""Wait for a free compute GPU, measure DP variants, and leave formal training paused."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path('/mnt/data2/wjy_SERL/Embodied_Safety')
OUT = ROOT/'profile_output/task1_training_20261102_2310'
PYTHON = ROOT/'algos/diffusion_policy/.venv/bin/python'
CACHE = ROOT/'outputs/runs/task1/diffusion_policy/success100_throughput_v001/lossless_rgb_cache'
NVME = Path('/home/ufdwaa/.cache/task1_training_20261102_2310/rgb')


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def status(phase, **kw):
    write(OUT/'queue_status.json', dict(phase=phase, updated_at=time.time(),
          supervisor_pid=os.getpid(), formal_resume=False, **kw))


def telemetry():
    raw = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True)
    gpu = ET.fromstring(raw).find('gpu')
    processes = []
    for p in gpu.findall('processes/process_info'):
        entry = {x.tag: x.text for x in p}
        pid = int(entry['pid'])
        try:
            stat = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
            entry['state'] = stat[0]
            entry['process_group'] = int(stat[2])
        except FileNotFoundError:
            entry['state'] = 'exited'
        processes.append(entry)
    def number(path):
        value = gpu.findtext(path)
        try:
            return float(value.split()[0])
        except (TypeError, AttributeError, ValueError):
            return None
    record = dict(timestamp=time.time(), gpu_name=gpu.findtext('product_name'),
                  uuid=gpu.findtext('uuid'), utilization_pct=number('utilization/gpu_util'),
                  memory_utilization_pct=number('utilization/memory_util'),
                  memory_mib=number('fb_memory_usage/used'),
                  power_w=number('gpu_power_readings/instant_power_draw') or number('power_readings/power_draw'),
                  sm_clock_mhz=number('clocks/sm_clock'), pstate=gpu.findtext('performance_state'),
                  processes=processes)
    record['active_compute'] = [p for p in processes if 'C' in p.get('type', '') and p['state'] not in ('T', 't', 'Z', 'exited')]
    record['meminfo'] = {k: int(v.split()[0]) for line in Path('/proc/meminfo').read_text().splitlines()
                         for k, v in [line.split(':', 1)] if k in ('MemAvailable', 'Cached', 'SwapFree')}
    record['vmstat'] = {k: int(v) for line in Path('/proc/vmstat').read_text().splitlines()
                        for k, v in [line.split()] if k in ('pswpin', 'pswpout', 'pgmajfault')}
    record['diskstats'] = [line.split() for line in Path('/proc/diskstats').read_text().splitlines()
                           if line.split()[2] in ('sda', 'sda3', 'nvme0n1', 'nvme0n1p3')]
    return record


def wait_idle():
    clear_since = None
    while True:
        sample = telemetry()
        with (OUT/'waiting_telemetry.jsonl').open('a') as f:
            f.write(json.dumps(sample)+'\n')
        if sample['active_compute']:
            clear_since = None
            status('waiting_for_idle_gpu', other_compute=sample['active_compute'])
        elif clear_since is None:
            clear_since = time.monotonic()
            status('confirming_idle_gpu')
        elif time.monotonic()-clear_since >= 30:
            return
        time.sleep(15)


def stage_cache():
    NVME.mkdir(parents=True, exist_ok=False)
    total = sum((CACHE/n).stat().st_size for n in ('rgb.bin', 'index.npy', 'manifest.json'))
    if shutil.disk_usage(NVME).free < total + 15*1024**3:
        raise RuntimeError('Not enough NVMe space for the complete cache and 15 GiB free space')
    start = time.monotonic()
    status('staging_complete_lossless_cache', source=str(CACHE), destination=str(NVME), bytes=total)
    for name in ('rgb.bin', 'index.npy', 'manifest.json'):
        shutil.copyfile(CACHE/name, NVME/name)
        with (NVME/name).open('rb') as f:
            os.fsync(f.fileno())
    write(OUT/'cache_staging.json', dict(source=str(CACHE), destination=str(NVME),
          bytes=total, wall_s=time.monotonic()-start, images_removed=0, resolution_changed=False))


def stop_trial(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def evict_trial_files(data, cache):
    # Affect only this trial's image files; do not drop global page caches.
    files = [Path(cache)/'rgb.bin'] if cache else [Path(data)/(camera+'.npy') for camera in ('external', 'wrist_left', 'wrist_right')]
    for path in files:
        with path.open('rb') as f:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)


def run_trial(name, checkpoint, data, cache=None, compact=False):
    attempt = 0
    while True:
        wait_idle()
        attempt += 1
        path = OUT/(name+f'_attempt{attempt:02d}')
        evict_trial_files(data, cache)
        command = [str(PYTHON), '-u', '-B', str(OUT/'profile_train.py'), '--checkpoint', str(checkpoint), '--output', str(path)]
        if cache:
            command += ['--cache', str(cache)]
        if compact:
            command += ['--compact']
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
        with (OUT/(name+f'_attempt{attempt:02d}.log')).open('w') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        status('benchmark', trial=name, attempt=attempt, pid=proc.pid, command=command)
        contaminated = False
        try:
            with (OUT/(name+f'_attempt{attempt:02d}_gpu.jsonl')).open('w') as monitor:
                while proc.poll() is None:
                    sample = telemetry()
                    sample['trial_pid'] = proc.pid
                    monitor.write(json.dumps(sample)+'\n')
                    monitor.flush()
                    others = [p for p in sample['active_compute'] if int(p['pid']) != proc.pid]
                    if others:
                        stop_trial(proc)
                        contaminated = True
                        write(OUT/(name+f'_attempt{attempt:02d}_excluded.json'),
                              dict(reason='Another compute task started; trial excluded and stopped', other_compute=others))
                        break
                    time.sleep(1)
        except BaseException:
            stop_trial(proc)
            raise
        if contaminated:
            continue
        if proc.returncode != 0:
            raise RuntimeError(f'{name} returned {proc.returncode}; inspect its log')
        result = json.loads((path/'timing.json').read_text())
        result.update(name=name, output=str(path), telemetry=str(OUT/(name+f'_attempt{attempt:02d}_gpu.jsonl')))
        return result


def main():
    status('waiting_for_idle_gpu')
    wait_idle()
    if not (OUT/'cache_staging.json').exists():
        stage_cache()
    old = ROOT/'outputs/runs/task1/diffusion_policy/full_flow_bc_v001/train/last.pt'
    new = OUT/'batch4_step5000.pt'
    old_data = ROOT/'outputs/datasets/task1/diffusion_sequences_v001'
    new_data = ROOT/'outputs/datasets/task1/diffusion_success100_v001'
    trials = [
        ('old_batch8_raw', old, old_data, None, False),
        ('current_batch4_raw', new, new_data, None, False),
        ('current_batch4_cache_hdd', new, new_data, CACHE, False),
        ('current_batch4_cache_nvme', new, new_data, NVME, False),
        ('current_batch4_cache_nvme_uint8', new, new_data, NVME, True),
    ]
    results = []
    for trial in trials:
        results.append(run_trial(*trial))
        write(OUT/'results.json', results)
    subprocess.run([str(PYTHON), '-B', str(OUT/'report_profile.py')], cwd=ROOT, check=True)
    status('completed_awaiting_resume_decision', report=str(OUT/'REPORT.md'), formal_training_advanced=False)


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        status('error', error=repr(exc))
        raise
