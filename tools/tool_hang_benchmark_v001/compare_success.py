"""Run ToolHang success comparisons at matched nominal data exposure."""
import common
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

CURRENT=Path(common.DEFAULT['output_dir'])
OFFICIAL=CURRENT.parent/'tool_hang_official_success_v001'
TRAIN_PID=389161
EVAL_PID=524562


def process_state(pid):
    path=Path(f'/proc/{pid}/stat')
    return path.read_text().split(') ',1)[1][0] if path.exists() else None


def compare():
    pairs=[]
    progress=json.loads((CURRENT/'progress.json').read_text())
    for current_step in (25000,50000,75000,100000):
        official_step=current_step//4
        pair=dict(current_step=current_step,official_step=official_step,
                  nominal_windows=current_step*16)
        for name,root,step in [('current',CURRENT,current_step),('official',OFFICIAL,official_step)]:
            path=root/f'eval_step_{step:06d}/summary.json'
            if path.exists():
                summary=json.loads(path.read_text())
                pair[name]={k:v for k,v in summary.items() if k!='episodes'}
                pair[name]['seeds']=[r['seed'] for r in summary['episodes']]
                if name=='current' and progress['step']==step:
                    snapshot=CURRENT/'checkpoints'/f'step_{step:06d}.pt'
                    if not snapshot.exists():snapshot.hardlink_to(CURRENT/'checkpoints/last.pt')
        pair['complete']=all(pair.get(k,{}).get('completed')==20 for k in ('current','official'))
        if pair['complete']:
            assert pair['current']['seeds']==pair['official']['seeds']
            pair['success_rate_difference']=pair['official']['success_rate']-pair['current']['success_rate']
        pairs.append(pair)
    path=OFFICIAL/'comparison.json'
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(dict(pairs=pairs,training_seed=42,test_seeds=list(range(100000,100020)),
        max_steps=700,action_chunk=8,notes=[
            'Same dataset and split; nominal window exposure matched, not optimizer update count.',
            'Last epoch batches are shorter; exact exposure can differ by less than one batch per epoch.',
            'Native inference: current DDIM16 BF16, official DDPM100 FP32.',
            'Current scheduler spans 100000 updates; official retains published 3000-epoch cosine horizon.',
            'Current evaluation episode timing overlapping the earlier speed benchmark includes its pause.',
            'One training seed and 20 test episodes per checkpoint; this is an initial comparison.']),indent=2))
    temp.replace(path)


def status(stage,**extra):
    row=dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)
    (OFFICIAL/'comparison_status.json').write_text(json.dumps(row,indent=2))
    print(json.dumps(row),flush=True)


def gpu_processes():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True)
    return {int(s.strip()) for s in result.stdout.splitlines() if s.strip()}


def wait_gpu(allowed):
    while True:
        other=gpu_processes()-allowed
        if not other:return
        status('waiting_for_other_gpu_tasks',pids=sorted(other))
        time.sleep(15)


def run_segment(limit):
    status('official_train_and_evaluate',stop_after=limit)
    with (OFFICIAL/'train.log').open('a') as log:
        p=subprocess.Popen([sys.executable,'-u',str(common.HERE/'train.py'),
                            '--config',str(OFFICIAL/'launch_config.json'),'--stop-after',str(limit)],
                           stdout=log,stderr=subprocess.STDOUT)
        (OFFICIAL/'train.pid').write_text(str(p.pid))
        with (OFFICIAL/'gpu.jsonl').open('a',buffering=1) as telemetry:
            while p.poll() is None:
                compare()
                gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
                                    '--format=csv,noheader'],capture_output=True,text=True)
                telemetry.write(json.dumps(dict(time=time.time(),gpu=gpu.stdout.strip(),
                                               processes=sorted(gpu_processes())))+'\n')
                time.sleep(15)
        if p.returncode:raise RuntimeError(f'Official training exited {p.returncode}; see {OFFICIAL}/train.log')
    compare()


def main():
    OFFICIAL.mkdir(parents=True,exist_ok=True)
    if (CURRENT/'completed.json').exists() and (OFFICIAL/'segment_006250.json').exists():
        wait_gpu(set())
        run_segment(25000)
        status('completed',comparison=str(OFFICIAL/'comparison.json'))
        return
    suspended=False
    try:
        # Pause the waiting trainer; its active evaluator finishes normally.
        if process_state(TRAIN_PID) not in (None,'Z'):
            os.kill(TRAIN_PID,signal.SIGSTOP);suspended=True
            status('finishing_current_25000_evaluation',paused_training_pid=TRAIN_PID)
        while process_state(EVAL_PID) not in (None,'Z'):
            compare();time.sleep(10)
        summary=json.loads((CURRENT/'eval_step_025000/summary.json').read_text())
        assert summary['completed']==summary['requested']==20
        wait_gpu({TRAIN_PID})
        run_segment(6250)
    finally:
        if suspended and process_state(TRAIN_PID) not in (None,'Z'):
            os.kill(TRAIN_PID,signal.SIGCONT)
            pause_path=OFFICIAL/'current_pause.json'
            pause=json.loads(pause_path.read_text())
            pause['resume_unix']=time.time()
            pause['pause_seconds']=pause['resume_unix']-pause['pause_unix']
            pause_path.write_text(json.dumps(pause,indent=2))
            print('CURRENT_TRAINING_RESUMED',TRAIN_PID,flush=True)
    status('waiting_for_current_training_completion')
    while process_state(TRAIN_PID) not in (None,'Z'):
        compare();time.sleep(30)
    if not (CURRENT/'completed.json').exists():
        raise RuntimeError('Current training stopped before completion; comparison results retained')
    wait_gpu(set())
    run_segment(25000)
    status('completed',comparison=str(OFFICIAL/'comparison.json'))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        status('failed',error=str(exc))
        raise
