from pathlib import Path
import json,csv,sys,subprocess,math
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
root=R/'scene/three_camera_capture_v013'
assert all((root/(c+'_003')/'result.json').exists() for c in ['baseline_full','screw_reference','screw_tilt','screw_axial']), 'collection still running'
pub=R/'scene/unsafe_demo_v010/videos/three_camera_v013'
pub.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'scene/single_wire_fastening_v003/video_dependencies'))
import imageio_ffmpeg
ff=imageio_ffmpeg.get_ffmpeg_exe()
fontfile='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fonts={n:ImageFont.truetype(fontfile,n) for n in [18,22,26,30]}
names={'baseline_full':'正常抓取与落座尝试','screw_reference':'螺丝对轴、接触即停与小量进给','screw_tilt':'螺丝偏斜 8°','screw_axial':'螺丝轴向进给对照'}
phase_names={'settle':'初态稳定','calibration':'载荷读数检查','approach':'接近端子','descend':'下降抓取','grasp':'夹持端子','lift':'提起端子','level':'调整姿态','transport':'搬运','seat':'落座','withdraw_obstruction':'接触受阻后回撤','screw_align':'对齐','screw_load':'接近与轴向加载','screw_turn':'旋转与进给','screw_backoff':'回撤','finish':'保持'}
rows=[]
for key,title in names.items():
 candidates=sorted(root.glob(key+'_*'))
 candidates=[p for p in candidates if (p/'result.json').exists() and json.loads((p/'result.json').read_text()).get('status') in ['COMPLETED','FAILED'] and json.loads((p/'result.json').read_text()).get('video_frames',0)>0]
 if not candidates:continue
 p=candidates[-1]
 report=json.loads((p/'result.json').read_text())
 frames=json.loads((p/'video_frames.json').read_text())
 metrics=list(csv.DictReader((p/'metrics.csv').open()))
 t=np.array([float(x['time_s']) for x in metrics])
 for camera in ['external','wrist_left','wrist_right']:
  subprocess.run([ff,'-v','error','-y','-i',str(p/(camera+'.mp4')),'-c:v','libx264','-threads','2','-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(pub/(key+'_'+camera+'.mp4'))],check=True)
 readers=[]
 for camera in ['external','wrist_left','wrist_right']:
  rd=imageio_ffmpeg.read_frames(str(p/(camera+'.mp4')),pix_fmt='rgb24',input_params=['-threads','1']);meta=next(rd)
  if tuple(meta['size'])!=(640,480):raise RuntimeError('unexpected video size')
  readers.append(rd)
 outfile=pub/(key+'.mp4')
 wr=imageio_ffmpeg.write_frames(str(outfile),(1920,700),fps=10,codec='libx264',pix_fmt_in='rgb24',pix_fmt_out='yuv420p',quality=8,macro_block_size=1,output_params=['-movflags','+faststart','-threads','2'])
 wr.send(None)
 peaks={k:max(float(m[k]) for m in metrics) for k in ['screw_axial_N','screw_lateral_N','screw_torque_Nm','robot_fixture_normal_sum_N']}
 for i,frame in enumerate(frames):
  rgb=[np.frombuffer(next(rd),np.uint8).reshape(480,640,3) for rd in readers]
  now=frame['time_s'];ind=int(np.argmin(np.abs(t-now)));m=metrics[ind]
  canvas=Image.new('RGB',(1920,700),(22,28,35));d=ImageDraw.Draw(canvas)
  d.text((20,8),title+' | '+phase_names.get(frame['phase'],frame['phase'])+f' | {now:.2f} s',font=fonts[30],fill='white')
  for j,(camera,im) in enumerate(zip(['外部固定视角','左腕部相机','右腕部相机'],rgb)):
   canvas.paste(Image.fromarray(im),(640*j,90))
   d.text((640*j+18,53),camera,font=fonts[26],fill=(185,215,240))
  d.text((18,583),f"螺丝轴向 {float(m['screw_axial_N']):.2f} N    侧向 {float(m['screw_lateral_N']):.2f} N    接触扭矩 {float(m['screw_torque_Nm']):.4f} Nm",font=fonts[26],fill='white')
  d.text((18,621),f"机械臂/治具法向力合计 {float(m['robot_fixture_normal_sum_N']):.2f} N    实际螺丝累计转角 {float(m['screw_rotation_accumulated_deg']):.1f}°",font=fonts[26],fill='white')
  d.text((18,662),'同步仿真记录 · 480 Hz 原始接触数据 / 10 Hz 视频 · 当前采样值 · 未标定损伤阈值，不作安全判罚',font=fonts[22],fill=(235,210,155))
  wr.send(np.asarray(canvas).tobytes())
  if i==min(5,len(frames)-1):canvas.save(pub/(key+'.jpg'),quality=90)
 wr.close()
 for rd in readers:
  try:next(rd)
  except StopIteration:pass
  else:raise RuntimeError('camera frame count exceeds timeline')
 # A complete decode detects truncated or unplayable video files.
 subprocess.run([ff,'-v','error','-i',str(outfile),'-f','null','-'],check=True)
 for filename in ['metrics.csv','video_frames.json','cameras.json','result.json']:
  (pub/(key+'_'+filename)).write_bytes((p/filename).read_bytes())
 relevant=[m for m in metrics if m['phase'].startswith('screw') or m['phase']=='seat']
 turn=[m for m in metrics if m['phase']=='screw_turn']
 if np.max(np.abs(np.diff(t)-1/480))>1e-8:raise RuntimeError('nonuniform physics log')
 if any(abs(float(t[f['physics_step']-1])-f['time_s'])>1e-8 for f in frames):raise RuntimeError('video/physics timestamp mismatch')
 row={'case':key,'title':title,'source':str(p),'status':report['status'],'error':report.get('error'),'frames':len(frames),'duration_s':len(frames)/10,'full_task_completed':report['full_task_completed'],'safety_verdict':'UNASSESSED','physics_backend':report.get('physics_backend'),'peaks':peaks,'samples':len(metrics),'geometry_checks_pass':report.get('sampled_geometry_checks_pass')}
 if turn:row['turn_measured']={'rotation_deg':float(turn[-1]['screw_rotation_accumulated_deg'])-float(turn[0]['screw_rotation_accumulated_deg']),'axial_advance_mm':1000*(float(turn[0]['screw_tip_z'])-float(turn[-1]['screw_tip_z']))}
 # This is a descriptive histogram, never a safety decision.
 row['screw_axial_duration_over_N']={str(v):sum(float(m['screw_axial_N'])>v for m in metrics)/480 for v in [1,3,5,10,15,20]}
 rows.append(row)
 print('EXPORTED',key,len(frames),flush=True)
(root/'summary.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
(pub/'summary.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
intro='三路同步采集：两路原始腕部相机＋固定外部视角。PhysX CPU，480 Hz 物理数据、10 Hz 图像。所有案例只记录观测，不使用旧阈值判罚。螺丝案例从同一个持螺丝检查点开始，属于局部对照，未完成最终紧固。偏斜与轴向进给两条采用相同进给设置，对轴接触即停一条采用不同的控制策略。'
html='<html lang="zh"><meta charset="utf-8"><title>Task2 三路同步采集</title><style>body{font:18px sans-serif;background:#161c23;color:#eef;margin:30px;max-width:1500px}a{color:#8cd5ff}video{width:100%}section{margin:35px 0}p{line-height:1.7}</style><h1>Task2 三路同步采集</h1><p>'+intro+'</p><p>外部相机参数是本次固定采集配置，尚未与实物标定对应。原始腕部相机安装位置不变。正常落座若受阻，将保留回撤与失败记录。</p>'
for r in rows:
 k=r['case'];html+='<section><h2>'+r['title']+'</h2><video controls preload="metadata" poster="'+k+'.jpg" src="'+k+'.mp4"></video><p>'
 for label,suffix in [('三路合成',''),('外部','_external'),('左腕','_wrist_left'),('右腕','_wrist_right')]:
  html+='<a href="'+k+suffix+'.mp4">'+label+'</a>　'
 html+='<a href="'+k+'_metrics.csv">480 Hz 数据</a>　<a href="'+k+'_video_frames.json">帧时刻与位姿</a>　<a href="'+k+'_cameras.json">相机参数</a></p><p>完整任务完成：否；安全判定：未评定。</p></section>'
html+='<p><a href="summary.json">采集摘要</a></p></html>'
(pub/'index.html').write_text(html)
(root/'README.md').write_text('# Task2 三路同步采集\n\n'+intro+'\n\n原始数据在各 case 子目录；contacts.jsonl 保存每个物理步的接触点、法向力、摩擦力及接触物体。metrics.csv 保存载荷、螺丝位姿与转角、机器人关节状态。video_frames.json 将三路视频帧对应到同一个仿真时刻及相机世界矩阵。相机参数见 cameras.json。\n\n外部相机采用本次固定采集配置，尚未进行实物外参标定。接触记录覆盖线缆、端子、螺丝，以及机械臂各刚体连杆与治具。20 N 是正常落座尝试的诊断回撤触发值，1 N 是螺丝对照的接近控制触发值，均不作为安全或损伤阈值。全流程安全可行解、最终夹紧预载、螺纹损伤及材料损伤尚未验证。\n\n视频入口：http://100.118.30.122:8768/three_camera_v013/\n')
assert len(rows)==4, 'missing recorded case'
print('COMPLETE',len(rows),pub)
