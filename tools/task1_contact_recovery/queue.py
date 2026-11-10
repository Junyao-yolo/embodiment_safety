"""Wait for paired evaluations, verify teacher recovery, fine-tune and compare."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
PYTHON=ROOT/'algos/diffusion_policy/.venv/bin/python'
BASE=ROOT/'outputs/runs/task1/diffusion_policy/official_input_compare_3090_v001'
OUT=ROOT/'outputs/runs/task1/diffusion_policy/contact_recovery_v001'


def write(path,value):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def status(phase,**details):
    row=dict(phase=phase,updated_at=time.time(),**details)
    write(OUT/'status.json',row)
    print(json.dumps(row),flush=True)


def wait_gpu():
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():
        time.sleep(30)


def command(args, log_path):
    started=time.monotonic()
    with Path(log_path).open('w') as log:
        subprocess.run([str(x) for x in args],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
    return time.monotonic()-started


def select_cases(report):
    candidates=[]
    for row in report['episodes']:
        if row.get('task_completion_observed') or 'insertion_ready' not in row.get('stages',[]):
            continue
        episode=Path(row['episode']).parents[1]
        labels=sorted(episode.glob('stage_transition_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
        ready=None;loss=None
        for p in labels:
            d=json.loads(p.read_text());index=d['index']+1
            if ready is None and any(e['stage']=='insertion_ready' for e in d['events']):ready=index
            if ready is not None and index>ready:
                e=d['evidence']
                if not e.get('bilateral_grasp_stable') or not e.get('orientation_ok') or not e.get('no_safety_violation'):
                    loss=index;break
        if ready is not None:
            indices=list(dict.fromkeys([ready,max(ready,(loss or ready+20)-10),max(ready,(loss or ready+20)-2)]))
            candidates.append((row,episode,indices))
    cases=[]
    for level in range(3):
        for row,episode,indices in candidates:
            if level>=len(indices):continue
            index=indices[level]
            cases.append(dict(seed=row['seed'],handoff_index=index,
                reference_episode=str(episode),reference_raw=json.loads((episode/f'raw_{index:03d}.json').read_text()),
                teacher_max_seconds=120,diagnostic_phase=('entry','before_failure','near_failure')[level]))
    return cases[:10]


def latest_summary(directory):
    return sorted(Path(directory).glob('rollouts/round_*/summary.json'))[-1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--describe',action='store_true');args=parser.parse_args()
    if args.describe:
        print(json.dumps(dict(wait_for='both baseline eval20 summaries and completed lifecycle',
            max_teacher_attempts=10,teacher_time_budget_s=120,
            training_gate='at least 3 accepted recoveries from at least 2 source seeds',
            finetune=dict(batch=32,epochs=3,lr=1e-5,original_fraction=.75,teacher_fraction=.25),
            evaluation=dict(episodes_per_checkpoint=20,paired_start_seed=2026112001,teacher=False)),indent=2))
        return
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'selection.json').exists():raise RuntimeError('Existing recovery run requires explicit resume inspection')
    while True:
        files=[BASE/(v+'_eval20')/'summary.json' for v in ('official','current')]
        lifecycle=json.loads((BASE/'lifecycle_status.json').read_text())
        if all(p.exists() for p in files) and lifecycle.get('status')=='complete':
            reports={v:json.loads(p.read_text()) for v,p in zip(('official','current'),files)}
            if all(x['completed']==20 for x in reports.values()):break
        if lifecycle.get('status')=='failed':
            status('baseline_failed',details=lifecycle);return
        status('waiting_for_baseline_tests',lifecycle=lifecycle.get('status'))
        time.sleep(60)
    variant=max(reports,key=lambda v:(reports[v]['task_success_rate'],reports[v]['stage_counts']['insertion_ready'],reports[v]['stage_counts']['grasp_clear']))
    cases=select_cases(reports[variant])
    write(OUT/'selection.json',dict(variant=variant,reason='highest baseline completion, then insertion-ready and grasp counts',
        source_checkpoint=reports[variant]['checkpoint'],case_count=len(cases)))
    if not cases:
        status('no_contact_candidates',variant=variant);return
    parent=BASE/(variant+'_workspace_eval.ckpt');collected=[]
    for index,spec in enumerate(cases):
        case=OUT/f'case_{index:03d}';case.mkdir()
        spec_path=case/'recovery.json';write(spec_path,spec)
        status('waiting_for_teacher_gpu',case=index,total=len(cases));wait_gpu()
        status('teacher_recovery',case=index,total=len(cases),variant=variant,seed=spec['seed'],handoff_index=spec['handoff_index'])
        wall=command([PYTHON,'-B',ROOT/'eval.py','--checkpoint',parent,'--output-dir',case/'rollout','--device','cuda:0',
            'task.env_runner.n_test=1',f'task.env_runner.test_start_seed={spec["seed"]}',
            f'task.env_runner.max_steps={spec["handoff_index"]+1250}',
            'task.env_runner.wall_seconds=3600','n_action_steps=8',f'task.env_runner.recovery_config={spec_path}'],case/'run.log')
        summary_path=latest_summary(case/'rollout');summary=json.loads(summary_path.read_text())
        episode=summary_path.parent/'trial_000/eval/episode_01'
        result_path=episode/'teacher_result.json'
        result=json.loads(result_path.read_text()) if result_path.exists() else dict(training_eligible=False,error='no_teacher_result')
        row=dict(case=index,episode=str(episode),seed=spec['seed'],wall_seconds=wall,teacher=result)
        collected.append(row);write(OUT/'teacher_results.json',collected)
    accepted=[r for r in collected if r['teacher'].get('training_eligible')]
    if len(accepted)<3 or len({r['seed'] for r in accepted})<2:
        status('teacher_validation_insufficient',attempts=len(collected),accepted=len(accepted),
            interpretation='No gradient updates; inspect teacher/controller/contact evidence before attributing failure to mechanics')
        return
    status('packing_teacher_segments',accepted=len(accepted))
    from .prepare import pack,initialize
    packed=OUT/'prepared_recovery';manifest=pack(accepted,packed)
    status('preparing_finetune',rows=manifest['rows'])
    train=initialize(parent,packed,OUT/'finetune')
    wait_gpu();status('finetuning',variant=variant,original_fraction=.75,teacher_fraction=.25)
    train_seconds=command([PYTHON,'-B',ROOT/'train.py','--config-dir',OUT/'finetune','--config-name','finetune',
        f'hydra.run.dir={train}'],OUT/'finetune.log')
    candidates=list((train/'checkpoints').glob('epoch=*.ckpt'))
    if not candidates:raise RuntimeError('No validation-selected fine-tuned checkpoint')
    tuned=min(candidates,key=lambda p:float(p.stem.split('val_loss=')[-1]))
    results={}
    for name,checkpoint in (('parent',parent),('finetuned',tuned)):
        wait_gpu();status('paired_evaluation',model=name,episodes=20,seed=2026112001)
        target=OUT/(name+'_eval20')
        wall=command([PYTHON,'-B',ROOT/'eval.py','--checkpoint',checkpoint,'--output-dir',target,'--device','cuda:0',
            'task.env_runner.n_test=20','task.env_runner.test_start_seed=2026112001',
            'task.env_runner.max_steps=3000','task.env_runner.wall_seconds=2700','n_action_steps=8',
            'task.env_runner.recovery_config=null'],OUT/(name+'_eval20.log'))
        results[name]=json.loads(latest_summary(target).read_text())
        results[name].update(checkpoint=str(checkpoint),evaluation_wall_seconds=wall)
        write(OUT/'comparison.json',results)
    status('complete',training_seconds=train_seconds,accepted_recoveries=len(accepted),
        parent_success=results['parent']['task_success_rate'],finetuned_success=results['finetuned']['task_success_rate'])


if __name__=='__main__':
    try:main()
    except BaseException as error:
        if OUT.exists():status('failed',error=repr(error))
        raise
