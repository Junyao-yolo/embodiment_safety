"""Wait for the dataset, prepare it, validate it, and train/evaluate on CUDA."""
import common
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import threading


def main():
    out=Path(common.DEFAULT['output_dir'])
    started=time.monotonic();timings={}
    stop=threading.Event()
    def monitor():
        with (out/'gpu.jsonl').open('a',buffering=1) as f:
            while not stop.is_set():
                row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                for key,query in [('gpu','--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw'),
                                  ('processes','--query-compute-apps=pid,process_name,used_memory')]:
                    r=subprocess.run(['nvidia-smi',query,'--format=csv,noheader'],capture_output=True,text=True)
                    row[key]=r.stdout.strip()
                f.write(json.dumps(row)+'\n')
                stop.wait(30)
    threading.Thread(target=monitor,daemon=True).start()
    def status(stage,**extra):
        row=dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 elapsed_seconds=time.monotonic()-started,stage_seconds=timings,**extra)
        (out/'pipeline_status.json').write_text(json.dumps(row,indent=2))
        print(json.dumps(row),flush=True)
    try:
        status('waiting_for_dataset')
        while not Path(common.DEFAULT['dataset_path']).exists():
            pid=int((out/'download.pid').read_text())
            if not Path(f'/proc/{pid}').exists():raise RuntimeError('Dataset download stopped; see download.log')
            time.sleep(15)
        timings['wait_for_download']=time.monotonic()-started
        for stage,script in [('prepare_cache','prepare.py'),('data_and_environment_check','data_smoke.py'),
                             ('train_and_evaluate','train.py')]:
            status(stage)
            t=time.monotonic()
            with (out/f'{stage}.log').open('a') as f:
                p=subprocess.Popen([sys.executable,'-u',str(common.HERE/script)],stdout=f,stderr=subprocess.STDOUT)
                (out/f'{stage}.pid').write_text(str(p.pid))
                code=p.wait()
            timings[stage]=time.monotonic()-t
            if code:raise RuntimeError(f'{stage} exited {code}; see {stage}.log')
        status('completed')
    except Exception as e:
        status('failed',error=str(e))
        raise
    finally:
        stop.set()


if __name__=='__main__':main()
