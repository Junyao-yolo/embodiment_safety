"""Run the two DDP jobs on disjoint GPUs and publish their return manifest."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

root=Path(__file__).resolve().parents[2]
python=root/'.venv/bin/python'
status=dict(status='starting',pid=os.getpid(),variants={},started_at=time.time())


def save():
    p=root/'training_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(status,indent=2));tmp.replace(p)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--adopt-current',type=int)
    args=parser.parse_args();jobs={};logs=[]
    def exit_code(variant,job):
        if job['process'] is not None:return job['process'].poll()
        proc=Path('/proc')/str(job['pid'])
        if proc.exists() and proc.joinpath('stat').read_text().split()[2]!='Z':return None
        cfg=json.loads((root/'configs'/(variant+'.json')).read_text())
        done=root/'outputs'/variant/f"segment_{cfg['stage_updates']:07d}.json"
        return 0 if done.exists() or (root/'outputs'/variant/'paused.json').exists() else 1
    for variant in ('official','current'):
        cfg=json.loads((root/'configs'/(variant+'.json')).read_text())
        smoke=json.loads((Path(cfg['output_dir'])/'resume_probe/resume_result.json').read_text())
        assert smoke['passed'] and smoke['world_size']==len(cfg['gpu_ids'])
        if variant=='current' and args.adopt_current:
            cmdline=(Path('/proc')/str(args.adopt_current)/'cmdline').read_bytes()
            assert str(root).encode() in cmdline and b'configs/current.json' in cmdline
            jobs[variant]=dict(process=None,pid=args.adopt_current)
            status['variants'][variant]=dict(pid=args.adopt_current,gpus=cfg['gpu_ids'],status='running',target_updates=cfg['stage_updates'],adopted=True)
            continue
        for gpu in cfg['gpu_ids']:
            occupied=subprocess.check_output(['nvidia-smi','-i',str(gpu),'--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if occupied:raise RuntimeError(f'GPU {gpu} became occupied: {occupied}')
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,cfg['gpu_ids'])),OMP_NUM_THREADS='2',
            OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',TMPDIR=str(root/'tmp'),PYTHONDONTWRITEBYTECODE='1')
        log=(root/'logs'/(variant+'_train.log')).open('a');logs.append(log)
        command=[str(python),'-m','torch.distributed.run','--standalone','--nnodes=1',
            '--nproc_per_node='+str(len(cfg['gpu_ids'])),str(root/'tools/task1_official_compare_v001/train_ddp.py'),
            '--config',str(root/'configs'/(variant+'.json'))]
        (Path(cfg['output_dir'])/'paused.json').unlink(missing_ok=True)
        proc=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        jobs[variant]=dict(process=proc,pid=proc.pid);status['variants'][variant]=dict(pid=proc.pid,gpus=cfg['gpu_ids'],status='running',target_updates=cfg['stage_updates'])
    status['status']='training';save()
    with (root/'logs/gpu.jsonl').open('a',buffering=1) as telemetry:
        while any(exit_code(v,j) is None for v,j in jobs.items()):
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True)
            processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True)
            telemetry.write(json.dumps(dict(time=time.time(),gpu=gpu,processes=processes))+'\n')
            for variant,job in jobs.items():
                code=exit_code(variant,job)
                if code is not None:status['variants'][variant].update(status='finished' if code==0 else 'failed',returncode=code)
                progress=root/'outputs'/variant/'progress.json'
                if progress.exists() and progress.stat().st_mtime>=status['started_at']:status['variants'][variant]['progress']=json.loads(progress.read_text())
            save();time.sleep(10)
    for log in logs:log.close()
    for variant,job in jobs.items():
        code=exit_code(variant,job)
        status['variants'][variant].update(status='finished' if code==0 else 'failed',returncode=code)
    if any(exit_code(v,j) for v,j in jobs.items()):
        status['status']='failed';save();return
    if any((root/'outputs'/v/'paused.json').exists() for v in jobs):
        status['status']='paused_by_user';save();return
    for variant in jobs:
        cfg=json.loads((root/'configs'/(variant+'.json')).read_text())
        completed=root/'outputs'/variant/f"segment_{cfg['stage_updates']:07d}.json"
        assert completed.exists(),f'Missing completion marker: {completed}'
    files=[];seen={}
    for directory in ('configs','logs','tools/task1_official_compare_v001','outputs/official','outputs/current','resume_records'):
        for path in sorted((root/directory).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix=='.partial':continue
            if ('ddp_smoke' in path.parts or 'resume_probe' in path.parts) and 'checkpoints' in path.parts:continue
            stat=path.stat();name=str(path.relative_to(root));identity=(stat.st_dev,stat.st_ino)
            row=dict(path=name,bytes=stat.st_size)
            if identity in seen:row['link_to']=seen[identity]
            else:seen[identity]=name
            files.append(row)
    (root/'return_manifest.json').write_text(json.dumps(dict(files=files,stage_epochs=100,
        total_transfer_bytes=sum(x['bytes'] for x in files if 'link_to' not in x),source_inputs_already_preserved=True),indent=2))
    status.update(status='training_complete',finished_at=time.time());save()


if __name__=='__main__':
    try:main()
    except BaseException as error:
        status.update(status='failed',error=repr(error));save();raise
