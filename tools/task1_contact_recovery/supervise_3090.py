"""Launch two disjoint DDP fine-tunes after all four assigned GPUs are idle."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
GPUS={'official':[3,4],'current':[5,6]}

def state(phase,**details):
    p=ROOT/'training_status.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(dict(phase=phase,updated_at=time.time(),**details),indent=2));t.replace(p)

def main():
    assert json.loads((ROOT/'setup_status.json').read_text())['phase']=='ready'
    while any(subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip() for pair in GPUS.values() for g in pair):
        state('waiting_for_four_gpus',gpus=GPUS);time.sleep(30)
    jobs={};logs=[]
    for v,gpus in GPUS.items():
        out=ROOT/'trained'/v;out.mkdir(parents=True,exist_ok=True)
        cfg=ROOT/'outputs/runs/task1/diffusion_policy/dagger_input_compare_v001'/v/'finetune/finetune.yaml'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        cmd=[str(ROOT/'.venv/bin/python'),'-B','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
             '-m','tools.task1_contact_recovery.train_ddp','--variant',v,'--config',str(cfg),'--output',str(out)]
        log=(ROOT/'logs'/(v+'_train.log')).open('w');logs.append(log)
        jobs[v]=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    with (ROOT/'logs/gpu.jsonl').open('a',buffering=1) as telemetry:
        while any(p.poll() is None for p in jobs.values()):
            rows={v:dict(pid=p.pid,returncode=p.poll(),gpus=GPUS[v]) for v,p in jobs.items()}
            for v in jobs:
                progress=ROOT/'trained'/v/'progress.json'
                if progress.exists():
                    try:rows[v]['progress']=json.loads(progress.read_text())
                    except json.JSONDecodeError:pass
            state('training',jobs=rows)
            telemetry.write(json.dumps(dict(time=time.time(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True)))+'\n')
            time.sleep(30)
    for f in logs:f.close()
    codes={v:p.returncode for v,p in jobs.items()}
    state('failed' if any(codes.values()) else 'complete',exit_codes=codes)

if __name__=='__main__':
    try:main()
    except Exception as e:state('failed',error=repr(e));raise
