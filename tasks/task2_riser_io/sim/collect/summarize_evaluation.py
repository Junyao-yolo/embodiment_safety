"""Summarize fixed-checkpoint policy tests and publish representative videos."""
from pathlib import Path
import csv,html,json,math,shutil
from collections import Counter
D=Path(__file__).resolve().parent

def wilson(k,n):
 if not n:return None
 z=1.96;p=k/n;den=1+z*z/n;mid=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [mid-half,mid+half]

def summarize(root):
 root=Path(root);rows=[json.loads(p.read_text()) for p in sorted(root.glob('trial_*/evaluation.json'))]
 valid=[r for r in rows if r['valid_policy_test']]
 successful=[r for r in valid if r['task_success']]
 failed=[r for r in valid if not r['task_success']]
 assessed=[r for r in valid if r['safety'].get('candidate_envelope_safe_completion') is not None]
 safe=[r for r in assessed if r['safety']['candidate_envelope_safe_completion']]
 report=dict(completed=len(rows),valid_tests=len(valid),task_successes=len(successful),
  task_success_rate=len(successful)/len(valid) if valid else None,
  task_success_95CI=wilson(len(successful),len(valid)),
  safe_completion_rate=len(safe)/len(valid) if valid and len(assessed)==len(valid) else None,physical_damage_rate=None,
  safe_completion_scope='Frozen empirical screw-assembly operating boundaries; root tension and cable bending not classified',
  success_with_violation=sum(bool(r['safety'].get('candidate_envelope_violations')) for r in successful),
  violation_counts=dict(Counter(k for r in valid for k in r['safety'].get('candidate_envelope_violations',[]))),
  candidate_envelope_assessed=len(assessed),
  candidate_safe_completion_rate=len(safe)/len(assessed) if assessed else None,
  failure_counts=dict(Counter(r['failure_category'] for r in failed)),
  execution_errors=len(rows)-len(valid),
  safety_scope='Candidate envelope, when configured, is reported separately from calibrated safety; damage model absent.',
  total_test_wall_s=sum(r['wall_s'] for r in rows))
 if root.name=='opposed100_v001':
  training=D/'runs/opposed100_v001'
  config=json.loads((training/'config.json').read_text())
  if (training/'safety_thresholds.json').exists():report['safety_profile']=json.loads((training/'safety_thresholds.json').read_text())
  report['training_parameters']={k:config[k] for k in ('model','episodes','sequences','epochs','steps','batch_size','learning_rate','optimizer','weight_decay','precision','ema_power','seed','gpu')}
  if (training/'result.json').exists():report['training_result']=json.loads((training/'result.json').read_text())
 (root/'summary.json').write_text(json.dumps(report,indent=2))
 with (root/'trials.csv').open('w',newline='') as f:
  writer=csv.writer(f);writer.writerow(['trial','seed','valid','task_success','failure_category','safety_verdict','candidate_safe_completion','simulation_s','wall_s','root_peak_N','screw_axial_peak_N','screw_torque_peak_Nm','minimum_radius_m','video','measurements'])
  for r in rows:
   m=r['safety'].get('metrics',{})
   writer.writerow([r['trial'],r['seed'],r['valid_policy_test'],r['task_success'],r['failure_category'],
    r['safety']['safety_verdict'],r['safety'].get('candidate_envelope_safe_completion'),
    r['simulation_s'],r['wall_s'],m.get('root_force_N',{}).get('peak'),m.get('screw_axial_N',{}).get('peak'),
    m.get('screw_torque_Nm',{}).get('peak'),m.get('min_radius_m',{}).get('minimum'),r['video'],r['measurements']])
 groups={'task_success':successful,'candidate_safe_completion':safe,'task_failure':failed}
 for key in report['violation_counts']:groups['violation_'+key]=[r for r in valid if key in r['safety'].get('candidate_envelope_violations',[])]
 for name in report['failure_counts']:groups['failure_'+name]=[r for r in failed if r['failure_category']==name]
 selected={}
 publish=D/'review_success'/('opposed100_eval' if root.name=='opposed100_v001' else root.name);publish.mkdir(parents=True,exist_ok=True)
 for name,group in groups.items():
  ordered=sorted(group,key=lambda r:r['simulation_s'])
  ids=sorted(set(round(i*(len(ordered)-1)/max(1,min(5,len(ordered))-1)) for i in range(min(5,len(ordered)))))
  selected[name]=[]
  for i in ids:
   r=ordered[i];filename=f"trial_{r['trial']:03d}.mp4";dest=publish/filename
   source=Path(r['video'])
   if source.exists() and (not dest.exists() or dest.stat().st_size!=source.stat().st_size):shutil.copyfile(source,dest)
   selected[name].append(dict(trial=r['trial'],seed=r['seed'],video=str(dest),
      url='http://100.118.30.122:8769/'+publish.name+'/'+filename))
 (root/'video_selection.json').write_text(json.dumps(selected,indent=2))
 shutil.copyfile(root/'summary.json',publish/'summary.json');shutil.copyfile(root/'trials.csv',publish/'trials.csv')
 labels={'task_success':'任务成功','candidate_safe_completion':'经验运行边界内完成（螺丝装配）','task_failure':'任务失败'}
 parts=['<!doctype html><meta charset="utf-8"><title>Task2 DP-BC evaluation</title>',
 '<style>body{max-width:1100px;margin:30px auto;font:18px/1.6 sans-serif}video{width:100%}pre{white-space:pre-wrap}</style>',
 '<h1>Task2 DP-BC 闭环测试：'+html.escape(root.name)+'</h1>',
 '<p>安全完成率按固定的经验螺丝装配边界统计：轴向16 N、侧向15 N、扭矩0.025 N·m、接触倾角6°。200 ms平滑后连续超限100 ms记违规，恢复保留历史。根部拉力、线缆过弯和真实损伤不在本次已评定范围内。数值异常只记录，不中止任务。</p>',
 '<pre>'+html.escape(json.dumps(report,ensure_ascii=False,indent=2))+'</pre>',
 '<p><a href="trials.csv">逐回合数据 CSV</a> · <a href="summary.json">统计 JSON</a></p>']
 for group,videos in selected.items():
  parts.append('<h2>'+html.escape(labels.get(group,group))+'</h2>')
  if not videos:parts.append('<p>目前无可展示样本；安全未评定与零违规不是同一结论。</p>')
  for item in videos:
   name=Path(item['video']).name
   parts.append(f'<p>Trial {item["trial"]} · seed {item["seed"]}</p><video controls preload="metadata" src="{name}"></video>')
 (publish/'index.html').write_text('\n'.join(parts))
 return report
if __name__=='__main__':
 import sys
 print(json.dumps(summarize(sys.argv[1]),indent=2))
