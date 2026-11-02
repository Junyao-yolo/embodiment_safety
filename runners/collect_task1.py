"""Collect distinct successful Task1 demonstrations with smooth transit paths."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .common import ROOT, resolve
from .eval_diffusion import label_summary, stop_group, write_json


def smoothness(episode):
    rows = [json.loads(s) for s in (episode/'transitions.jsonl').read_text().splitlines()]
    phases = {}
    signature = []
    def high_frequency(values):
        average = np.stack([np.convolve(values[:, k], np.ones(5)/5, mode='valid') for k in range(3)], axis=1)
        residual = values[2:-2]-average
        return float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
    for name in ('LEFT_ABOVE', 'RIGHT_ABOVE', 'GRASP_ABOVE', 'TRANSFER'):
        selected = [r for r in rows if r['diagnostics'].get('expert_stage') == name]
        if len(selected) < 8:
            return dict(accepted=False, reason='missing_transit:'+name), []
        positions = np.array([r['observations']['ee_position_m'] for r in selected])
        rotations = Rotation.from_quat([r['observations']['ee_quaternion_xyzw'] for r in selected])
        times = np.array([r['observations']['sim_time'] for r in selected])
        dt = np.diff(times)
        velocity = np.diff(positions, axis=0)/dt[:, None]
        angular = (rotations[1:]*rotations[:-1].inv()).as_rotvec()/dt[:, None]
        jitter = high_frequency(velocity)
        spin_jitter = high_frequency(angular)
        distance = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        arc = np.r_[0., np.cumsum(distance)]
        sampled = np.column_stack([np.interp(np.linspace(0, arc[-1], 20), arc, positions[:, k]) for k in range(3)])
        signature.extend(sampled.tolist())
        direct = float(np.linalg.norm(positions[-1]-positions[0]))
        ratio = float(arc[-1]/max(direct, .01))
        phases[name] = dict(samples=len(selected), path_length_m=float(arc[-1]),
            path_to_chord_ratio=ratio, velocity_jitter_rms_m_s=jitter,
            angular_jitter_rms_rad_s=spin_jitter,
            accepted=bool(jitter <= .005 and spin_jitter <= .05 and ratio <= 1.5))
    contact_phases = {}
    for name in sorted({r['diagnostics'].get('expert_stage') for r in rows}-set(phases)):
        selected = [r for r in rows if r['diagnostics'].get('expert_stage') == name]
        if len(selected) < 8:
            continue
        positions = np.array([r['observations']['ee_position_m'] for r in selected])
        rotations = Rotation.from_quat([r['observations']['ee_quaternion_xyzw'] for r in selected])
        dt = np.diff([r['observations']['sim_time'] for r in selected])
        jitter = high_frequency(np.diff(positions, axis=0)/dt[:, None])
        spin_jitter = high_frequency((rotations[1:]*rotations[:-1].inv()).as_rotvec()/dt[:, None])
        contact_phases[name] = dict(samples=len(selected), velocity_jitter_rms_m_s=jitter,
            angular_jitter_rms_rad_s=spin_jitter, accepted=bool(jitter <= .005 and spin_jitter <= .05))
    return dict(accepted=all(p['accepted'] for p in [*phases.values(), *contact_phases.values()]),
                phases=phases, contact_and_other_phases=contact_phases,
                thresholds=dict(velocity_jitter_rms_m_s=.005, angular_jitter_rms_rad_s=.05,
                                path_to_chord_ratio=1.5),
                definition='RMS residual from a centered five-sample velocity average within each task phase'), signature


def assess(output, previous):
    result = json.loads((output/'result.json').read_text())
    episode = Path(result['episode'])
    validation = json.loads((episode/'validation.json').read_text())
    labels = label_summary(output)
    quality, signature = smoothness(episode)
    distances = [float(np.sqrt(np.mean(np.sum((np.array(signature)-np.array(x['path_samples_m']))**2, axis=1))))
                 for x in previous] if signature else []
    distinct = not distances or min(distances) >= .0005
    accepted = bool(result['full_task_success'] and not result['failure_reason'] and
        validation['data_valid'] and labels['reward'] >= 1.39 and quality['accepted'] and distinct)
    report = dict(accepted=accepted, full_task_success=result['full_task_success'],
        failure_reason=result['failure_reason'], data_valid=validation['data_valid'],
        reward=labels['reward'], stages=labels['stages'], smoothness=quality,
        distinct_path=distinct, nearest_path_rms_m=min(distances) if distances else None,
        path_samples_m=signature, episode=str(episode), output=str(output))
    write_json(output/'acceptance.json', report)
    meta = json.loads((episode/'metadata.json').read_text())
    meta.update(task_success=result['full_task_success'], safety_pass=True if result['full_task_success'] else meta.get('safety_pass'),
                accepted_successful_demonstration=accepted, acceptance=str(output/'acceptance.json'))
    write_json(episode/'metadata.json', meta)
    validation.update(task_success=meta['task_success'], safety_pass=meta['safety_pass'], eligible_safe_demo=accepted)
    write_json(episode/'validation.json', validation)
    result.update(training_ready=accepted, acceptance_pending=False, safety_pass=meta['safety_pass'],
                  acceptance=str(output/'acceptance.json'))
    write_json(output/'result.json', result)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--successes', type=int, default=100)
    parser.add_argument('--max-attempts', type=int, default=200)
    parser.add_argument('--seed', type=int, default=2026110201)
    parser.add_argument('--variation', type=float, default=1.)
    parser.add_argument('--wall-timeout-s', type=float, default=1800.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    out = resolve(args.output).resolve()
    if not out.is_relative_to(ROOT/'outputs') or args.successes < 1 or args.max_attempts < args.successes:
        raise ValueError('Repository output and sufficient positive collection budget required')
    if not 0 < args.variation <= 1:
        raise ValueError('Path variation must be in (0, 1]')
    if args.resume:
        summary = json.loads((out/'summary.json').read_text())
        if summary['seed'] != args.seed or summary['variation'] != args.variation:
            raise ValueError('Resume must retain the path distribution')
    else:
        out.mkdir(parents=True, exist_ok=False)
        summary = dict(seed=args.seed, variation=args.variation, attempts=[], accepted=[], started_at=time.time())
    summary.update(status='running', pid=os.getpid(), target_successes=args.successes,
                   max_attempts=args.max_attempts, wall_timeout_s=args.wall_timeout_s, workers=args.workers)
    summary.pop('error', None)
    def save():
        summary['accepted_count'] = len(summary['accepted'])
        summary['elapsed_s'] = time.time()-summary['started_at']
        write_json(out/'summary.json', summary)
        write_json(out/'manifest.json', dict(task='task1_dimm_insert', requested=args.successes,
            successful_episodes=len(summary['accepted']), episodes=summary['accepted'],
            control_hz=10, physics_hz=240, ik_solver='dogbox_trf',
            demonstration_type='feedback_expert_with_smooth_transit_variation'))
    save()
    jobs = []
    try:
        while len(summary['accepted']) < args.successes:
            while (len(jobs) < args.workers and len(summary['attempts']) < args.max_attempts
                   and len(summary['accepted'])+len(jobs) < args.successes):
                index = len(summary['attempts'])+1
                dest = out/f'attempt_{index:03d}'
                seed = args.seed+index-1
                row = dict(index=index, seed=seed, output=str(dest), status='running')
                summary['attempts'].append(row)
                env = os.environ.copy()
                env.update(TASK1_EXPERT_SEED=str(seed), TASK1_PATH_VARIATION=str(args.variation),
                    TASK1_EVAL='1', TASK1_ONLINE_LEARN='0', TASK1_SAFETY_MODE='stop',
                    TASK1_MAX_STEPS='2400', TASK1_DIAGNOSTIC_STOP_STAGE='', TASK1_SMOKE='0',
                    CUDA_VISIBLE_DEVICES='0')
                log = (out/f'attempt_{index:03d}.log').open('w')
                start = time.monotonic()
                proc = subprocess.Popen(['/mnt/data2/isaacsim-5.0.0/python.sh', '-B',
                    str(ROOT/'tasks/task1_dimm_insert/env/collect_actor.py'), str(dest)],
                    cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                row['pid'] = proc.pid
                jobs.append((proc, log, start, row, dest))
                save()
            if not jobs:
                break
            completed = False
            for job in list(jobs):
                proc, log, start, row, dest = job
                rc = proc.poll()
                if rc is None:
                    if time.monotonic()-start > args.wall_timeout_s:
                        stop_group(proc)
                        jobs.remove(job)
                        log.close()
                        row.update(status='wall_timeout', failure_reason='wall_timeout',
                                   wall_s=time.monotonic()-start, returncode=proc.poll())
                        completed = True
                        save()
                    continue
                completed = True
                jobs.remove(job)
                log.close()
                row.update(wall_s=time.monotonic()-start, returncode=rc)
                if rc or (dest/'actor_error.json').exists() or not (dest/'result.json').exists():
                    row['status'] = 'error'
                    raise RuntimeError('Collector execution error: '+str(dest))
                report = assess(dest, summary['accepted'])
                row.update(status='accepted' if report['accepted'] else 'rejected',
                           failure_reason=report['failure_reason'], reward=report['reward'],
                           smoothness_passed=report['smoothness']['accepted'])
                if report['accepted']:
                    summary['accepted'].append(dict(report, seed=row['seed'], index=row['index']))
                save()
            if not completed:
                time.sleep(5)
        summary['status'] = 'finished' if len(summary['accepted']) >= args.successes else 'attempt_budget_reached'
    except BaseException as exc:
        summary.update(status='error', error=repr(exc))
        raise
    finally:
        for proc, log, _, row, _ in jobs:
            stop_group(proc)
            log.close()
            row['status'] = 'interrupted'
        save()
    print(json.dumps(dict(status=summary['status'], accepted=len(summary['accepted']), attempts=len(summary['attempts'])), indent=2))


if __name__ == '__main__':
    main()
