"""Collect exactly 100 independently executed, accepted assembly trajectories."""
from pathlib import Path
from collections import deque
import json,subprocess,sys,time,shutil
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
D=R/'scene/success_v014'
D.mkdir(exist_ok=True)
def accepted():
 result=[]
 for p in sorted(D.glob('episode_*/result.json')):
  r=json.loads(p.read_text())
  if r.get('accepted') and all((p.parent/(n+'.mp4')).exists() for n in ['external','wrist_left','wrist_right']):
   result.append({'episode':p.parent.name,'seed':r['seed'],'variation':r['variation']})
 return result
done=accepted()
seed=max(x['seed'] for x in done)+1 if done else 4
consecutive_failures=0
while len(done)<100:
 state={'status':'RUNNING' if done else 'VALIDATING_FIRST_TRAJECTORY','target_successes':100,'completed_successes':len(done),'current_seed':seed,'retention':'accepted complete episodes only'}
 (D/'progress.json').write_text(json.dumps(state,indent=2))
 command=['/mnt/data2/isaacsim-5.0.0/python.sh','-B',str(R/'scripts/run_success_v014.py'),'--seed',str(seed),'--record']
 proc=subprocess.Popen(command,cwd=str(R),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 result=None
 for line in proc.stdout:
  if line.startswith('STATE '):
   sample=json.loads(line[6:]);state['current_phase']=sample['phase'];state['simulation_time_s']=sample['time_s']
   (D/'progress.json').write_text(json.dumps(state,indent=2))
  elif line.startswith('RESULT '):result=json.loads(line[7:])
 code=proc.wait()
 if result:
  state['acceptance_details']={k:result.get(k) for k in ['wire_seated','screw_fastened','ring_retained','ring_hold_extrema','mean_clamp_N','geometry_valid','thread_engagement_verified','smoothness_pass','smoothness']}
 done=accepted()
 if result and result.get('accepted') and (D/('episode_'+str(seed).zfill(4))/'result.json').exists():
  consecutive_failures=0
 else:
  consecutive_failures+=1
  partial=D/('episode_'+str(seed).zfill(4))
  if partial.exists():shutil.rmtree(partial)
 state.update(completed_successes=len(done),last_attempt_success=bool(result and result.get('accepted')),last_error=result.get('error') if result else 'worker exited without a result')
 print(json.dumps({'seed':seed,'successes':len(done),'accepted':state['last_attempt_success'],'error':state['last_error']}),flush=True)
 (D/'progress.json').write_text(json.dumps(state,indent=2))
 if consecutive_failures>=5 or (not done and consecutive_failures>=1):
  state.update(status='NEEDS_REPAIR',reason=result.get('error') if result else 'worker exited without an acceptance result')
  (D/'progress.json').write_text(json.dumps(state,indent=2))
  raise RuntimeError(state['reason'])
 seed+=1
(D/'dataset.json').write_text(json.dumps({'count':100,'episodes':done,'scene':'riser_io_success_v014.usda','safety_calibration':'pending; task completion is independently checked'},indent=2))
(D/'progress.json').write_text(json.dumps({'status':'COMPLETE','target_successes':100,'completed_successes':100},indent=2))
for name in ['current.log','current_kit.log']:
 (D/name).unlink(missing_ok=True)
print('COMPLETE 100')
