"""Matched Task1 rollouts with stage rates, timing, and saved RGB videos."""
import common
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from runners.eval_diffusion import stop_group, label_summary


def wait_gpu():
    while True:
        rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
        if not rows:return
        time.sleep(30)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--episodes',type=int,default=20)
    ap.add_argument('--max-steps',type=int,default=3000)
    ap.add_argument('--wall-seconds',type=int,default=2700)
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cp=str(args.checkpoint.resolve())
    report=dict(checkpoint=cp,requested=args.episodes,episodes=[],teacher_enabled=False,action_steps=8,
        reset_distribution='unchanged Task1 default scene; paired diffusion RNG seeds',
        max_steps=args.max_steps,max_wall_seconds=args.wall_seconds,safety_mode='warn',status='evaluating')
    summary=out/'summary.json'
    if summary.exists():
        report=json.loads(summary.read_text())
        assert report['checkpoint']==cp and report['requested']==args.episodes
    def save():
        n=len(report['episodes']);report['completed']=n
        report['stage_counts']={stage:sum(stage in row.get('stages',[]) for row in report['episodes'])
            for stage in ('both_open','grasp_clear','insertion_ready','seated_locked')}
        report['stage_rates']={key:value/n if n else None for key,value in report['stage_counts'].items()}
        for key in ('task_completion_observed','full_task_success'):
            report[key+'_count']=sum(bool(row.get(key)) for row in report['episodes'])
            report[key+'_rate']=report[key+'_count']/n if n else None
        report['wall_seconds']=sum(x['wall_s'] for x in report['episodes'])
        common.write_json(summary,report)
    save()
    for index in range(len(report['episodes'])+1,args.episodes+1):
        wait_gpu()
        base=out/f'episode_{index:02d}';dest=base/'eval/episode_01';dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise RuntimeError(f'Interrupted episode requires inspection: {dest}')
        env=os.environ.copy()
        for key in ('TASK1_DIAGNOSTIC_STOP_STAGE','TASK1_NATIVE_PROFILE','TASK1_WINDOW_GATES','TASK1_DAGGER_REFERENCES'):
            env.pop(key,None)
        env.update(CUDA_VISIBLE_DEVICES='0',TASK1_POLICY_BACKEND='diffusion_policy_official',TASK1_EVAL='1',
            TASK1_DP_ACTION_STEPS='8',TASK1_ONLINE_LEARN='0',TASK1_MAX_STEPS=str(args.max_steps),TASK1_SMOKE='0',
            TASK1_EPISODE_SEED=str(2026110300+index),TASK1_SAFETY_MODE='warn',SERL_INPUT_CHECKPOINT=cp,
            TASK1_EXECUTION_VARIANT='batched_io',TASK1_GOAL_LEAD_HORIZON_S='.25',TASK1_MAX_FREE_GOAL_LEAD_M='.03',
            TASK1_IK_ROTATION_TOLERANCE_RAD='.002')
        start=time.monotonic();report.update(current_episode=index,status='evaluating');save()
        with (base/'actor.log').open('w') as log:
            proc=subprocess.Popen(['/mnt/data2/isaacsim-5.0.0/python.sh','-B',
                str(common.ROOT/'tools/task1_execution_v004/actor.py'),str(dest)],cwd=common.ROOT,env=env,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            report['actor_pid']=proc.pid;save()
            try:proc.wait(timeout=args.wall_seconds);timed_out=False
            except subprocess.TimeoutExpired:stop_group(proc);timed_out=True
        row=dict(index=index,seed=2026110300+index,wall_s=time.monotonic()-start,wall_timeout=timed_out,
                 returncode=proc.returncode,task_completion_observed=False,full_task_success=False)
        if (dest/'result.json').exists():row.update(json.loads((dest/'result.json').read_text()))
        row.update(label_summary(dest))
        raw=sorted(dest.glob('raw_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
        if raw:
            t0=json.loads(raw[0].read_text())['sim_time']
            row['simulation_elapsed_s']=json.loads(raw[-1].read_text())['sim_time']-t0
            row['stage_first_sim_seconds']={}
            for file in sorted(dest.glob('stage_transition_*.json'),key=lambda p:int(p.stem.rsplit('_',1)[1])):
                for event in json.loads(file.read_text())['events']:
                    row['stage_first_sim_seconds'].setdefault(event['stage'],event['time']-t0)
        actions=[json.loads(p.read_text()) for p in dest.glob('action_*.json')]
        row['inference_count']=sum(bool(x.get('inference_performed')) for x in actions)
        row['inference_seconds']=sum(x.get('inference_ms',0) for x in actions)/1000
        row['policy_compute_seconds']=sum(x.get('request_compute_ms',0) for x in actions)/1000
        common.write_json(dest.parent/'summary.json',row)
        if raw:
            with (base/'render.log').open('w') as log:
                rendered=subprocess.run([str(common.ROOT/'algos/diffusion_policy/.venv/bin/python'),'-B',
                    str(common.ROOT/'tools/task1_execution_v004/render_replay.py'),str(base)],cwd=common.ROOT,stdout=log,stderr=subprocess.STDOUT)
            row.update(video=str(base/'replay/actual_policy_replay.mp4'),video_ready=rendered.returncode==0)
        error=(dest/'worker_error.json').exists() or (dest/'actor_error.json').exists() or (proc.returncode!=0 and not timed_out)
        row['infrastructure_error']=error;report['episodes'].append(row);save()
        if error:
            report['status']='infrastructure_error';save();raise RuntimeError(f'Rollout infrastructure failed: {dest}')
    report['status']='finished';save()


if __name__=='__main__':main()
