"""Pause this experiment, compare update speeds, then resume its process state."""
import common
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def main():
    run=Path(common.DEFAULT['output_dir'])
    out=run/'profile_output/official_comparison_v001';out.mkdir(parents=True,exist_ok=True)
    train_pid=int((run/'train_and_evaluate.pid').read_text())
    pipeline_pid=int((run/'pipeline.pid').read_text())
    assert os.getpgid(train_pid)==pipeline_pid
    assert os.getpgid(os.getpid())!=pipeline_pid
    stop=threading.Event()
    def gpu_monitor():
        with (out/'gpu.jsonl').open('a',buffering=1) as f:
            while not stop.is_set():
                r=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total,power.draw',
                                  '--format=csv,noheader'],capture_output=True,text=True)
                apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                     '--format=csv,noheader'],capture_output=True,text=True)
                f.write(json.dumps(dict(time=time.time(),gpu=r.stdout.strip(),processes=apps.stdout.strip()))+'\n')
                stop.wait(2)
    paused=False
    pause_start=time.time()
    pause=dict(process_group=pipeline_pid,training_pid=train_pid,
               pause_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               pause_unix=pause_start,training_progress=json.loads((run/'progress.json').read_text()),
               checkpoint_files=[str(p) for p in (run/'checkpoints').glob('*.pt')],
               method='SIGSTOP/SIGCONT; process memory and optimizer state retained',
               gpu_memory_of_paused_processes_remains_allocated=True)
    try:
        os.killpg(pipeline_pid,signal.SIGSTOP);paused=True
        (out/'pause.json').write_text(json.dumps(pause,indent=2))
        print('TRAINING_PAUSED',json.dumps(pause),flush=True)
        threading.Thread(target=gpu_monitor,daemon=True).start()
        time.sleep(2)
        variants=['current_b16_bf16','official_b16_bf16','official_b64_fp32']
        results={}
        for variant in variants:
            target=out/variant;target.mkdir(exist_ok=True)
            print('START',variant,flush=True)
            with (target/'run.log').open('w') as f:
                p=subprocess.Popen([sys.executable,'-u',str(common.HERE/'benchmark.py'),
                                    '--variant',variant,'--output',str(target)],stdout=f,stderr=subprocess.STDOUT)
                (out/'active.json').write_text(json.dumps(dict(variant=variant,pid=p.pid)))
                code=p.wait()
            rp=target/'result.json'
            results[variant]=json.loads(rp.read_text()) if rp.exists() else dict(status='failed',returncode=code)
            (out/'results.json').write_text(json.dumps(results,indent=2))
            print('FINISH',variant,results[variant].get('status'),flush=True)
    finally:
        if paused:
            os.killpg(pipeline_pid,signal.SIGCONT)
            pause.update(resume_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                         resume_unix=time.time(),pause_seconds=time.time()-pause_start)
            (out/'pause.json').write_text(json.dumps(pause,indent=2))
            print('TRAINING_RESUMED',json.dumps(pause),flush=True)
        stop.set()


if __name__=='__main__':main()
