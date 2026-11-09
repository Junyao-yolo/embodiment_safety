from pathlib import Path
import json
import numpy as np
D=Path('/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io')
out=D/'runs/opposed100_v001/safety_calibration';out.mkdir(exist_ok=True)
def moving_mean(x,n):
 s=np.r_[0.,np.cumsum(x,dtype=float)]
 return (s[n:]-s[:-n])/n
rows=[]
for ep in sorted((D/'data/opposed_v002/raw').glob('episode_*')):
 r=json.loads((ep/'result.json').read_text())
 if not r.get('accepted'):continue
 p=Path(r['measurements_path'])
 with np.load(p,allow_pickle=False) as a:
  row=dict(episode=ep.name,backend=r['physics_backend'],samples=len(a['time_s']),fields=list(a.files),metrics={})
  active=a['phase']!='settle'
  for k in ('screw_axial_N','screw_lateral_N','screw_torque_Nm','screw_tilt_deg','clamp_N','joint_gap_m'):
   v=np.array(a[k],dtype=float)
   if k=='screw_tilt_deg':gate=(a['nut_contact_count']>0)&active
   else:gate=active
   series=np.where(gate,v,0.)
   vals=v[gate]
   result=dict(raw_peak=float(np.max(vals)) if len(vals) else None,raw_p99=float(np.quantile(vals,.99)) if len(vals) else None,
               finite=bool(np.isfinite(v).all()),active_samples=int(gate.sum()))
   for milliseconds in (100,200,500):
    n=round(milliseconds/1000*120)
    sm=moving_mean(series,n)
    if k=='screw_tilt_deg':
     density=moving_mean(gate.astype(float),n)
     sm=np.divide(sm,density,out=np.zeros_like(sm),where=density>=.5)
    result[f'mean{milliseconds}_peak']=float(sm.max())
   result['by_phase']={str(ph):float(v[(a['phase']==ph)&gate].max()) for ph in np.unique(a['phase']) if ((a['phase']==ph)&gate).any()}
   row['metrics'][k]=result
  rows.append(row)
(out/'episode_statistics.json').write_text(json.dumps(rows,indent=2))
summary={}
for backend in sorted({r['backend'] for r in rows}):
 group=[r for r in rows if r['backend']==backend];stats={'episodes':len(group),'metrics':{}}
 for key in group[0]['metrics']:
  m={}
  for field in ('raw_peak','raw_p99','mean100_peak','mean200_peak','mean500_peak'):
   v=[r['metrics'][key][field] for r in group if r['metrics'][key][field] is not None]
   m[field]=dict(zip(['min','median','p90','p95','max'],map(float,np.quantile(v,[0,.5,.9,.95,1]))))
  stats['metrics'][key]=m
 summary[backend]=stats
(out/'distribution_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
print('fields',rows[0]['fields'])
