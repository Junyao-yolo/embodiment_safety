"""Execute the documented two-GPU update, checkpoint, and inference witnesses."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]

def main():
    while not all((ROOT/n).exists() for n in ('environment_installed.json','inputs_ready.json')):
        p=ROOT/'setup_status.json'
        if p.exists() and json.loads(p.read_text())['phase']=='failed':raise RuntimeError(p.read_text())
        time.sleep(30)
    results={}
    for variant,gpus in (('official',[3,4]),('current',[5,6])):
        while any(subprocess.check_output(['nvidia-smi','-i',str(g),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip() for g in gpus):time.sleep(30)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,gpus)),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        out=ROOT/'smoke'/variant
        cmd=[str(ROOT/'.venv/bin/python'),'-B','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
             '-m','tools.task1_contact_recovery.train_ddp','--variant',variant,'--output',str(out),'--smoke']
        with (ROOT/'logs'/('witness_'+variant+'.log')).open('w') as log:
            subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads((out/'smoke_result.json').read_text());assert result['passed'] and result['world_size']==2
        results[variant]=result
        for name in ('latest.ckpt','best.ckpt'):(out/name).unlink()
    (ROOT/'doc_witness_passed.json').write_text(json.dumps(results,indent=2))
    (ROOT/'setup_status.json').write_text(json.dumps(dict(phase='ready',time=time.time(),witnesses=results),indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':
    try:main()
    except Exception as e:
        (ROOT/'setup_status.json').write_text(json.dumps(dict(phase='failed',error=repr(e)),indent=2))
        raise
