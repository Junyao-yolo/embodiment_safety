"""Train, return full checkpoints, verify, and evaluate through the shared Workspace."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/runs/task1/diffusion_policy/official_input_compare_3090_v001'
REMOTE='/mnt/file4/junyao/task1_official_compare_20261108'
HOST='junyao@47.94.205.88'
KEY=OUT/'transfer_key'
SSH=['ssh','-i',str(KEY),'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=20',
     '-o','ServerAliveInterval=30','-o','ServerAliveCountMax=10','-o','UserKnownHostsFile='+str(OUT/'transfer_known_hosts'),'-p','8002']
PYTHON=ROOT/'algos/diffusion_policy/.venv/bin/python'


def state(phase,**kwargs):
    data=dict(status=phase,updated_at=time.time(),**kwargs);p=OUT/'lifecycle_status.json';tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2));tmp.replace(p);print(json.dumps(data),flush=True)


def remote_python(code):
    return subprocess.check_output([*SSH,HOST,'python3 -'],input=code,text=True)


def remote_json(name):
    result=remote_python(f"from pathlib import Path\np=Path({REMOTE!r})/{name!r}\nprint(p.read_text() if p.exists() else 'null')\n")
    return json.loads(result)


def transfer(arguments,logname):
    command=['rsync','-a','--partial','--info=progress2','-e',shlex.join(SSH),*arguments]
    with (OUT/logname).open('a') as log:
        for attempt in range(6):
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==0:return
            state('transfer_retry',log=logname,attempt=attempt+1,returncode=result.returncode);time.sleep(30)
    raise RuntimeError('File transfer failed: '+logname)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume-ready',action='store_true');parser.add_argument('--monitor-only',action='store_true')
    args=parser.parse_args()
    if not (args.resume_ready or args.monitor_only):
        state('transferring_inputs',input_gib=17.085835267789662)
        transfer(['--files-from=/tmp/task1_3090_inputs.txt',str(ROOT)+'/',HOST+':'+REMOTE+'/'],'transfer_inputs.log')
        state('preparing_remote')
        remote_python(f"""from pathlib import Path
import json,subprocess
r=Path({REMOTE!r});m=json.loads((r/'inputs.json').read_text())
assert sum((r/p).stat().st_size for p in m['paths'])==m['bytes']
subprocess.run(['python3',str(r/'tools/task1_official_compare_v001/prepare_3090.py')],check=True)
(r/'inputs_ready.json').write_text(json.dumps(dict(bytes=m['bytes'],files=len(m['paths']))))
""")
    if not args.monitor_only:
        state('waiting_for_gpu_witness')
        while True:
            try:
                if remote_json('doc_witness_passed.json'):break
                failed=remote_json('doc_witness_failed.json')
                if failed:raise RuntimeError('GPU witness failed: '+json.dumps(failed))
            except subprocess.CalledProcessError as error:
                state('waiting_for_connection',error=str(error))
            time.sleep(30)
        response=remote_python(f"""from pathlib import Path
import subprocess,json
r=Path({REMOTE!r})
with (r/'logs/supervisor.log').open('a') as log:
 p=subprocess.Popen([str(r/'.venv/bin/python'),'-u',str(r/'tools/task1_official_compare_v001/run_3090.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(pid=p.pid)))
""")
        state('training',supervisor=json.loads(response))
    while True:
        try:
            status=remote_json('training_status.json')
            state('training',remote=status)
            if status and status['status']=='training_complete':break
            if status and status['status']=='paused_by_user':
                state('paused_by_user',remote=status);return
            if status and status['status']=='failed':raise RuntimeError('Training failed: '+json.dumps(status))
        except subprocess.CalledProcessError as error:
            state('waiting_for_connection',error=str(error))
        time.sleep(60)
    manifest=remote_json('return_manifest.json');(OUT/'return_manifest.json').write_text(json.dumps(manifest,indent=2))
    listing=OUT/'return_files.txt';listing.write_text('\n'.join(x['path'] for x in manifest['files'])+'\n')
    state('returning_results',bytes=manifest['total_transfer_bytes'])
    transfer(['-H','--files-from='+str(listing),HOST+':'+REMOTE+'/',str(OUT)+'/'],'transfer_results.log')
    for item in manifest['files']:
        assert (OUT/item['path']).stat().st_size==item['bytes'],item['path']
    state('verifying_returned_checkpoints')
    with (OUT/'verify.log').open('w') as log:
        subprocess.run([str(PYTHON),str(Path(__file__).with_name('verify_return.py')),'--run',str(OUT)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert json.loads((OUT/'checkpoint_verification.json').read_text())['passed']
    state('returned_checkpoints_verified',remote_preserved=True)
    reports={}
    for variant in ('official','current'):
        if variant == 'official' and (OUT/'official_eval_launch.json').exists():
            while True:
                early=json.loads((OUT/'official_eval_launch.json').read_text())
                if early['status']=='complete':
                    reports[variant]=json.loads((OUT/'official_eval20/summary.json').read_text())
                    break
                if early['status']=='failed':
                    raise RuntimeError('Official evaluation failed: '+json.dumps(early))
                state('waiting_for_official_evaluation',evaluation=early)
                time.sleep(30)
            continue
        state('evaluating',variant=variant)
        target=OUT/(variant+'_eval20')
        checkpoint=OUT/'outputs'/variant/'checkpoints/best.pt'
        imported=OUT/(variant+'_workspace_eval.ckpt')
        task='task1_official_image' if variant=='official' else 'task1_image'
        with (OUT/(variant+'_eval20.log')).open('a') as log:
            if not imported.exists():
                subprocess.run([str(PYTHON),'-B','-m','learning.import_checkpoint',
                    '--source',str(checkpoint),'--task',task,'--output',str(imported)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                    '--format=csv,noheader,nounits'],text=True).strip():
                state('waiting_for_evaluation_gpu',variant=variant)
                time.sleep(30)
            state('evaluating',variant=variant,framework='learning_workspace',checkpoint=str(checkpoint))
            subprocess.run([str(PYTHON),'-u',str(ROOT/'eval.py'),
                '--checkpoint',str(imported),'--output-dir',str(target),'--device','cuda:0',
                'task.env_runner.n_test=20','task.env_runner.test_start_seed=2026110301',
                'task.env_runner.max_steps=3000','task.env_runner.wall_seconds=2700',
                'task.env_runner.safety_mode=warn','n_action_steps=8'],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        rollout=sorted((target/'rollouts').glob('round_*/summary.json'))[-1]
        reports[variant]=json.loads(rollout.read_text())
        reports[variant].update(checkpoint=str(checkpoint),weights='ema',
            framework='learning_workspace',inference_checkpoint=str(imported),
            rollout_summary=str(rollout),test_start_seed=2026110301)
        (target/'summary.json').write_text(json.dumps(reports[variant],indent=2))
    (OUT/'comparison.json').write_text(json.dumps(reports,indent=2))
    state('complete')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        state('failed',error=repr(error),returned_checkpoints_verified=(OUT/'checkpoint_verification.json').exists());raise

