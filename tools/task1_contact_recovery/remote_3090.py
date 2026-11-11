"""Transfer only training inputs, run disjoint two-GPU jobs, and return checkpoints."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from .queue import ROOT, BASE, PYTHON, write

REMOTE = '/mnt/file4/junyao/task1_dagger_compare_20261111'
HOST = 'junyao@47.94.205.88'
SSH = ['ssh','-i',str(BASE/'transfer_key'),'-o','IdentitiesOnly=yes','-o','BatchMode=yes',
       '-o','ConnectTimeout=20','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=10',
       '-o','UserKnownHostsFile='+str(BASE/'transfer_known_hosts'),'-p','8002']
OUT = ROOT/'outputs/runs/task1/diffusion_policy/dagger_input_compare_v001'
GPU_IDS = {'official':[3,4], 'current':[5,6]}


def remote(code):
    return subprocess.check_output([*SSH,HOST,'python3 -'],input=code,text=True)


def transfer(args, logfile):
    with Path(logfile).open('a') as log:
        for attempt in range(4):
            result=subprocess.run(['rsync','-a','--partial','--info=progress2','-e',shlex.join(SSH),*args],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode==0:return
            time.sleep(30)
    raise RuntimeError('Transfer failed: '+str(logfile))


def read_status():
    return json.loads(remote(f'from pathlib import Path\np=Path({REMOTE!r})/"setup_status.json"\nprint(p.read_text() if p.exists() else "null")\n'))


def prepare_server():
    OUT.mkdir(parents=True,exist_ok=True)
    names=[]
    for directory in ('learning','third_party/diffusion_policy/diffusion_policy',
                      'tools/tool_hang_benchmark_v001/official_deps/robomimic','tools/task1_contact_recovery'):
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.yaml','.json','.md') and '__pycache__' not in p.parts:
                names.append(str(p.relative_to(ROOT)))
    listing=OUT/'remote_code_files.txt';listing.write_text('\n'.join(names)+'\n')
    remote(f'from pathlib import Path\nr=Path({REMOTE!r});r.mkdir(parents=True,exist_ok=True)\n(r/"logs").mkdir(exist_ok=True)\n')
    transfer(['--files-from='+str(listing),str(ROOT)+'/',HOST+':'+REMOTE+'/'],OUT/'remote_code_transfer.log')
    remote(f'''from pathlib import Path
import subprocess,json,os
r=Path({REMOTE!r})
spec=json.loads((r/'tools/task1_contact_recovery/env-spec-3090.json').read_text())
(r/'env-spec.json').write_text(json.dumps(spec,indent=2))
with (r/'logs/setup.log').open('a') as log:
 p=subprocess.Popen(['python3','-u',str(r/'tools/task1_contact_recovery/setup_3090.py')],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(r/'setup.pid').write_text(str(p.pid))
print(p.pid)
''')
    data=ROOT/'outputs/datasets/task1/diffusion_parallel_diverse_new_wrist_v001'
    names=[str((data/n).relative_to(ROOT)) for n in ('manifest.json','state.npy','action.npy','proprio9.npy')]
    names += [str(p.relative_to(ROOT)) for p in (data/'lossless_rgb_cache').iterdir() if p.is_file()]
    listing=OUT/'remote_data_files.txt';listing.write_text('\n'.join(names)+'\n')
    transfer(['--files-from='+str(listing),str(ROOT)+'/',HOST+':'+REMOTE+'/'],OUT/'remote_data_transfer.log')
    remote(f'from pathlib import Path\n(Path({REMOTE!r})/"parents").mkdir(exist_ok=True)\n')
    for variant in GPU_IDS:
        transfer([str(BASE/(variant+'_workspace_eval.ckpt')),HOST+':'+REMOTE+'/parents/'+variant+'.ckpt'],OUT/'remote_parent_transfer.log')
    remote(f'from pathlib import Path\nimport json\n(Path({REMOTE!r})/"inputs_ready.json").write_text(json.dumps(dict(ready=True)))\n')
    write(OUT/'remote_inputs.json',dict(remote=REMOTE,host=HOST,gpus=GPU_IDS,transferred=True))


def train_and_return():
    while True:
        state=read_status()
        write(OUT/'remote_setup_status.json',state)
        if state and state['phase']=='ready':break
        if state and state['phase']=='failed':raise RuntimeError('3090 setup failed: '+str(state))
        time.sleep(30)
    names=[]
    for variant in GPU_IDS:
        arm=OUT/variant
        names.extend(str(p.relative_to(ROOT)) for p in (arm/'prepared_recovery').rglob('*') if p.is_file())
        names.extend(str((arm/'finetune'/n).relative_to(ROOT)) for n in ('finetune.yaml','normalizer.pt','training_plan.json'))
    listing=OUT/'remote_correction_files.txt';listing.write_text('\n'.join(names)+'\n')
    transfer(['--files-from='+str(listing),str(ROOT)+'/',HOST+':'+REMOTE+'/'],OUT/'remote_corrections_transfer.log')
    remote(f'''from pathlib import Path
import subprocess
r=Path({REMOTE!r})
with (r/'logs/supervisor.log').open('a') as log:
 p=subprocess.Popen([str(r/'.venv/bin/python'),'-u','-B','-m','tools.task1_contact_recovery.supervise_3090'],cwd=r,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(r/'supervisor.pid').write_text(str(p.pid))
print(p.pid)
''')
    while True:
        state=json.loads(remote(f'from pathlib import Path\np=Path({REMOTE!r})/"training_status.json"\nprint(p.read_text() if p.exists() else "null")\n'))
        write(OUT/'remote_training_status.json',state)
        if state and state['phase']=='complete':break
        if state and state['phase']=='failed':raise RuntimeError('3090 training failed: '+str(state))
        time.sleep(30)
    transfer([HOST+':'+REMOTE+'/trained/',str(OUT/'returned_3090')+'/'],OUT/'return_3090.log')
    transfer([HOST+':'+REMOTE+'/logs/',str(OUT/'returned_3090/logs')+'/'],OUT/'return_3090.log')
    # Validate the returned full training payload before deleting the dedicated remote copy.
    import dill
    import torch
    from learning.bootstrap import configure
    configure()
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    checkpoints={}
    for variant in GPU_IDS:
        arm=OUT/'returned_3090'/variant
        completion=json.loads((arm/'completed.json').read_text())
        assert completion['step']==completion['target_updates']==10134
        for name in ('latest','best'):
            payload=torch.load(arm/(name+'.ckpt'),map_location='cpu',pickle_module=dill,weights_only=False,mmap=True)
            assert 'optimizer' in payload['state_dicts'] and 'rank_rng' in payload['ddp_state']
            model=hydra.utils.instantiate(payload['cfg'].policy)
            model.load_state_dict(payload['state_dicts']['model'],strict=True)
            model.load_state_dict(payload['state_dicts']['ema_model'],strict=True)
            del model,payload
        checkpoints[variant]=dict(parent=str(BASE/(variant+'_workspace_eval.ckpt')),
            finetuned=str(arm/'best.ckpt'),training_seconds=completion['training_seconds'])
    write(OUT/'returned_verification.json',dict(passed=True,checkpoints=checkpoints,expected_updates=10134))
    remote(f'import shutil\nfrom pathlib import Path\nr=Path({REMOTE!r})\nshutil.rmtree(r)\nprint("dedicated_remote_run_removed",not r.exists())\n')
    write(OUT/'remote_cleanup.json',dict(removed=REMOTE,returned_checkpoints_verified=True))
    return checkpoints


if __name__=='__main__':prepare_server()
