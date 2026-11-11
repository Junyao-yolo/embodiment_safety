"""Paired intervention DAgger for Task1 official and current observations."""

import argparse
import json
from pathlib import Path
import time

from .queue import ROOT, BASE, PYTHON, write, wait_gpu, command, latest_summary

OUT = ROOT / 'outputs/runs/task1/diffusion_policy/dagger_input_compare_v001'
VARIANTS = ('official', 'current')


def status(phase, **details):
    record = dict(phase=phase, updated_at=time.time(), **details)
    write(OUT / 'status.json', record)
    print(json.dumps(record), flush=True)


def plan_cases(reports):
    eligible = {
        variant: {row['seed']: row for row in report['episodes']
                  if not row.get('task_completion_observed') and 'grasp_clear' in row.get('stages', [])}
        for variant, report in reports.items()
    }
    seeds = sorted(set.intersection(*(set(rows) for rows in eligible.values())))[:5]
    if len(seeds) != 5:
        raise ValueError('Five shared post-grasp source seeds are required for this paired experiment')
    planned = {variant: [] for variant in VARIANTS}
    for seed in seeds:
        for variant in VARIANTS:
            row = eligible[variant][seed]
            episode = (Path(row['episode']).parents[1] if row.get('episode') else
                       Path(reports[variant]['rollout_summary']).parent / f"trial_{row['trial']:03d}/eval/episode_01")
            stage = 'insertion_ready' if 'insertion_ready' in row['stages'] else 'grasp_clear'
            first_time = row.get('stage_first_sim_seconds', {}).get(stage)
            if first_time is not None:
                anchor = round(first_time / .1)
                paths = [episode / f'stage_transition_{i-1:03d}.json'
                         for i in range(max(1, anchor-20), anchor+21)]
                labels = [json.loads(p.read_text()) for p in paths if p.exists()]
            else:
                labels = [json.loads(p.read_text()) for p in episode.glob('stage_transition_*.json')]
            labels.sort(key=lambda d: d['index'])
            anchor = next(d['index'] + 1 for d in labels if any(e['stage'] == stage for e in d['events']))
            stable = []
            failure = None
            for d in labels:
                index = d['index'] + 1
                e = d['evidence']
                if index > anchor and failure is None and not all(e.get(k) for k in
                        ('bilateral_grasp_stable', 'orientation_ok', 'no_safety_violation')):
                    failure = index
                if anchor - 20 <= index <= anchor + 20 and all(e.get(k) for k in
                        ('bilateral_grasp_stable', 'grasp_in_allowed_band', 'no_safety_violation')):
                    stable.append(index)
            if len(stable) < 2:
                raise ValueError(f'Insufficient stable handoff observations: {variant} seed {seed}')
            first = min(stable, key=lambda i: (abs(i - anchor), i))
            target = max(anchor, (failure or anchor + 22) - 2)
            second = min((i for i in stable if i != first), key=lambda i: (abs(i - target), i))
            for label, index in zip(('stage_entry', 'near_failure'), (first, second)):
                planned[variant].append(dict(seed=seed, handoff_index=index,
                    reference_episode=str(episode), reference_raw=json.loads((episode / f'raw_{index:03d}.json').read_text()),
                    teacher_max_seconds=120, diagnostic_phase=label, source_stage=stage))
    return planned


def training_gate(collected):
    accepted = {v: [r for r in collected[v] if r['teacher'].get('training_eligible')] for v in VARIANTS}
    return all(len(rows) >= 3 and len({r['seed'] for r in rows}) >= 2 for rows in accepted.values()), accepted


def describe():
    return dict(variants=list(VARIANTS), shared_source_seeds=5, handoffs_per_seed=2,
        attempts_per_variant=10, teacher_seconds=120,
        training_gate='both arms: >=3 accepted episodes from >=2 source seeds',
        finetune=dict(batch=32, local_batch=16, gpus_per_variant=2, epochs=3, updates=10134, lr=1e-5, seed=42,
                      original_fraction=.75, teacher_fraction=.25),
        evaluation=dict(episodes_per_checkpoint=20, checkpoints=4, total_episodes=80,
                        start_seed=2026112001, action_steps=8, teacher=False),
        scope='One round of intervention DAgger; each arm uses its own accepted corrections')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.describe:
        print(json.dumps(describe(), indent=2)); return
    reports = {v: json.loads((BASE / (v + '_eval20') / 'summary.json').read_text()) for v in VARIANTS}
    if any(r['completed'] != 20 for r in reports.values()):
        raise ValueError('Complete both baseline eval20 runs before planning paired DAgger')
    planned = plan_cases(reports)
    if args.plan_only:
        print(json.dumps({v: [{k: c[k] for k in ('seed', 'handoff_index', 'source_stage')}
                             for c in planned[v]] for v in VARIANTS}, indent=2)); return
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'plan.json').exists():
        raise RuntimeError('Existing paired run requires resume inspection')
    write(OUT / 'plan.json', dict(**describe(), cases=planned,
        parents={v: str(BASE / (v + '_workspace_eval.ckpt')) for v in VARIANTS}))
    collected = {v: [] for v in VARIANTS}
    for index in range(10):
        for variant in VARIANTS:
            case = OUT / variant / f'case_{index:03d}'; case.mkdir(parents=True)
            spec = planned[variant][index]
            spec_path = case / 'recovery.json'; write(spec_path, spec)
            status('waiting_for_teacher_gpu', variant=variant, case=index, attempts_per_variant=10)
            wait_gpu()
            status('teacher_recovery', variant=variant, case=index, seed=spec['seed'])
            wall = command([PYTHON, '-B', ROOT / 'eval.py', '--checkpoint', BASE / (variant + '_workspace_eval.ckpt'),
                '--output-dir', case / 'rollout', '--device', 'cuda:0',
                'task.env_runner.n_test=1', f'task.env_runner.test_start_seed={spec["seed"]}',
                f'task.env_runner.max_steps={spec["handoff_index"] + 1250}',
                'task.env_runner.wall_seconds=3600', 'n_action_steps=8',
                f'task.env_runner.recovery_config={spec_path}'], case / 'run.log')
            summary_path = latest_summary(case / 'rollout')
            episode = summary_path.parent / 'trial_000/eval/episode_01'
            result_path = episode / 'teacher_result.json'
            teacher = json.loads(result_path.read_text()) if result_path.exists() else dict(training_eligible=False, error='no_teacher_result')
            collected[variant].append(dict(case=index, episode=str(episode), seed=spec['seed'],
                wall_seconds=wall, teacher=teacher))
            write(OUT / 'teacher_results.json', collected)
    passed, accepted = training_gate(collected)
    counts = {v: dict(attempts=len(collected[v]), accepted=len(accepted[v]),
                      source_seeds=len({r['seed'] for r in accepted[v]})) for v in VARIANTS}
    write(OUT / 'collection_summary.json', counts)
    if not passed:
        status('teacher_validation_insufficient', counts=counts, gradient_updates=0)
        return
    from .prepare import pack, initialize
    for variant in VARIANTS:
        arm = OUT / variant
        status('preparing_finetune', variant=variant)
        manifest = pack(accepted[variant], arm / 'prepared_recovery')
        parent = BASE / (variant + '_workspace_eval.ckpt')
        initialize(parent, arm / 'prepared_recovery', arm / 'finetune', save_initial=False)
        write(arm / 'data_counts.json', dict(rows=manifest['rows'], train_windows=len(manifest['train']),
            validation_windows=len(manifest['validation']), accepted_episodes=len(accepted[variant])))
    from .remote_3090 import train_and_return
    status('remote_finetuning', host='machine2', gpus={'official':[3,4], 'current':[5,6]},
           global_batch=32, local_batch=16, updates_per_arm=10134)
    checkpoints = train_and_return()
    write(OUT / 'checkpoints.json', checkpoints)
    results = {}
    for model in ('parent', 'finetuned'):
        for variant in VARIANTS:
            key = variant + '_' + model
            status('waiting_for_evaluation_gpu', model=key); wait_gpu()
            status('paired_evaluation', model=key, episodes=20)
            target = OUT / (key + '_eval20')
            wall = command([PYTHON, '-B', ROOT / 'eval.py', '--checkpoint', checkpoints[variant][model],
                '--output-dir', target, '--device', 'cuda:0', 'task.env_runner.n_test=20',
                'task.env_runner.test_start_seed=2026112001', 'task.env_runner.max_steps=3000',
                'task.env_runner.wall_seconds=2700', 'n_action_steps=8',
                'task.env_runner.recovery_config=null'], OUT / (key + '_eval20.log'))
            results[key] = json.loads(latest_summary(target).read_text())
            results[key].update(evaluation_wall_seconds=wall, checkpoint=checkpoints[variant][model])
            write(OUT / 'comparison.json', results)
    status('complete', successes={key: r['task_success_rate'] for key, r in results.items()})


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            status('failed', error=repr(error))
        raise
