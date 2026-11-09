"""Run failure review sequentially with Task2 collection."""
from pathlib import Path
import os,signal,subprocess,json,time
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
D=Path('/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io/failure_review')
D.mkdir(exist_ok=True)
state={'status':'STARTING','started_unix':time.time(),'paused_workers':[],'attempts':[]}
def save():
 (D/'progress.json').write_text(json.dumps(state,indent=2))
ps=subprocess.check_output(['ps','-eo','pid,args'],text=True)
for line in ps.splitlines():
 fields=line.strip().split(None,1)
 if len(fields)==2 and '/kit/python/bin/python3 -B '+str(R/'scripts/run_dp_bc.py')+' ' in fields[1]:
  state['paused_workers'].append(int(fields[0]))
if len(state['paused_workers'])>1:raise RuntimeError('Expected at most one Task2 collection worker')
paused=[]
try:
 for pid in state['paused_workers']:
  os.kill(pid,signal.SIGSTOP);paused.append(pid)
 state['paused_at_unix']=time.time();save()
 for seed in [10,7]:
  state.update(status='RECORDING',seed=seed);save()
  folder=D/f'seed_{seed:04d}';folder.mkdir(exist_ok=True)
  with (folder/'worker.log').open('w') as log:
   worker=subprocess.Popen(['/mnt/data2/isaacsim-5.0.0/python.sh','-B',str(R/'scripts/run_failure_review.py'),'--seed',str(seed),'--record'],cwd=R,stdout=log,stderr=subprocess.STDOUT)
   state['review_worker_pid']=worker.pid;save()
   code=worker.wait()
  result=folder/'result.json'
  d=json.loads(result.read_text()) if result.exists() else {'error':'No result','accepted':False}
  state['attempts'].append({'seed':seed,'exit_code':code,'accepted':d.get('accepted',False),'error':d.get('error'),'failed_checks':d.get('failed_checks'),'directory':str(folder)});save()
  if not d.get('accepted') and (folder/'final_review.mp4').exists():
   state.update(status='FAILED_TRAJECTORY_RECORDED',video=str(folder/'final_review.mp4'));save();break
 else:
  state['status']='FAILURE_NOT_REPRODUCED';save()
finally:
 for pid in paused:
  try:os.kill(pid,signal.SIGCONT)
  except ProcessLookupError:pass
 state['resumed_at_unix']=time.time()
 state['collection_pause_seconds']=state['resumed_at_unix']-state.get('paused_at_unix',state['resumed_at_unix'])
 state['collection_resumed']=True
 save()
