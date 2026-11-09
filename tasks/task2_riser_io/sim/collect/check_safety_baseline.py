from pathlib import Path
import json,sys,csv,shutil
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
D=Path('/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io')
from safety_assessment import assess_measurements,LiveSafety
profile=json.loads((D/'runs/opposed100_v001/safety_thresholds.json').read_text())
def data(n=360):
 return dict(time_s=np.arange(1,n+1)/120,phase=np.full(n,'policy'),nut_contact_count=np.ones(n),
             **{k:np.zeros(n) for k in profile['rules']})
x=data();x['screw_axial_N'][120]=31
assert assess_measurements(x,profile,True)['safe_completion'] is True
x=data();x['screw_axial_N'][60:180]=24
r=assess_measurements(x,profile,True)
assert not r['safe_completion'] and r['rules']['screw_axial_N']['events'][0]['recovered']
live=LiveSafety(profile)
for i in range(len(x['time_s'])):live.update({k:v[i] for k,v in x.items()})
assert 'screw_axial_N' in live.latched and not live.current
x=data();x['screw_tilt_deg'][:]=30;x['nut_contact_count'][:]=0
assert assess_measurements(x,profile,True)['safe_completion'] is True
assert assess_measurements(data(),profile,False)['safe_completion'] is False
rows=[]
for ep in sorted((D/'data/opposed_v002/raw').glob('episode_*')):
 report=json.loads((ep/'result.json').read_text())
 if report['physics_backend']!='PhysX CPU':continue
 with np.load(report['measurements_path'],allow_pickle=False) as raw:
  result=assess_measurements(raw,profile,True)
 rows.append(dict(episode=ep.name,safe_under_empirical_screw_limits=result['safe_completion'],violations=result['candidate_envelope_violations'],rules=result['rules']))
out=D/'runs/opposed100_v001/safety_calibration'
(out/'baseline_assessment.json').write_text(json.dumps(rows,indent=2))
summary=dict(episodes=len(rows),within_empirical_boundaries=sum(r['safe_under_empirical_screw_limits'] for r in rows),
             violating=[dict(episode=r['episode'],violations=r['violations']) for r in rows if r['violations']],
             test_checks=['31 N one-sample transient tolerated','24 N sustained load detected','recovery retains prior violation','free-space screw tilt ignored','stationary unfinished episode cannot count as safe completion'])
(out/'validation.json').write_text(json.dumps(summary,indent=2))
with (out/'baseline_assessment.csv').open('w',newline='') as f:
 w=csv.writer(f);w.writerow(['episode','within_empirical_boundaries','violations',*profile['rules']])
 for r in rows:w.writerow([r['episode'],r['safe_under_empirical_screw_limits'],';'.join(r['violations']),*[r['rules'][k]['filtered_peak'] for k in profile['rules']]])
print(json.dumps(summary,indent=2))
for name in ['trial_recorder.py','safety_assessment.py','summarize_evaluation.py','analyze_safety_baseline.py']:
 shutil.copyfile(D/name,D/'runs/opposed100_v001/source'/name)
shutil.copyfile(D/'runs/opposed100_v001/safety_thresholds.json',D/'runs/opposed100_v001/source/safety_thresholds.json')
