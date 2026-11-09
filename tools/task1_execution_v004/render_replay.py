from pathlib import Path
import json,csv,sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio.v2 as imageio
import imageio_ffmpeg
import subprocess
B=Path(sys.argv[1]).resolve();D=B/"eval/episode_01";O=B/"replay";O.mkdir(exist_ok=True)
files=sorted(D.glob("raw_*.json"),key=lambda p:int(p.stem.rsplit("_",1)[1]))
pairs=[(int(p.stem.rsplit("_",1)[1]),json.loads(p.read_text())) for p in files if (D/f"obs_{int(p.stem.rsplit('_',1)[1]):03d}.npz").exists()]
assert pairs,"No saved observations"
ids=[p[0] for p in pairs];raw=[p[1] for p in pairs];times=np.array([r["sim_time"] for r in raw]);rel=times-times[0]
keys=["left_press_N","right_press_N","dimm_environment_N","finger_left_N","finger_right_N"]
peaks=np.zeros((len(raw),len(keys)));counts=np.zeros(len(raw),int);footer=0;partial=0
limits=np.array([3.,3.,5.,5.,5.])
for f in (D/"episodes").glob("*/physics.jsonl"):
 for line in f.open():
  if not line.endswith("\n"):partial+=1;continue
  r=json.loads(line)
  if "sim_time" not in r:
   if r.get("reason")!="episode_end":raise ValueError("Unexpected physics record")
   footer+=1;continue
  j=np.searchsorted(times,r["sim_time"])
  if j>=len(raw):continue
  counts[j]+=1;peaks[j]=np.maximum(peaks[j],[r["metrics"][k] for k in keys])
events=[]
for f in sorted(D.glob("stage_transition_*.json")):
 r=json.loads(f.read_text())
 events.extend(dict(time=e["time"],stage=e["stage"]) for e in r["events"])
summary=json.loads((B/"eval/summary.json").read_text())
font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",16)
small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",13)
def frame(i,slow=False):
 n,r=pairs[i]
 with np.load(D/f"obs_{n:03d}.npz") as z:rgb=np.concatenate([z[k][-1] for k in ["external","wrist_left","wrist_right"]],axis=1)
 im=Image.new("RGB",(768,432),(18,22,28));im.paste(Image.fromarray(rgb),(0,176));dr=ImageDraw.Draw(im)
 m=r["task_truth"]["metrics"];bad=bool(np.any(peaks[i]>limits))
 color=(255,80,80) if bad else (125,230,160)
 dr.text((8,4),f"DP-BC | VERIFIED EXECUTION TEST | WARNING-ONLY | t={rel[i]:.3f}s | step={n}"+(" | 0.5x" if slow else ""),font=font,fill="white")
 dr.text((8,29),f"Latch L={m['left_angle_deg']:+.2f} / R={m['right_angle_deg']:+.2f} deg | Width={r['gripper_width_m']*1000:.3f} mm",font=font,fill="white")
 dr.text((8,54),f"Fingers L={peaks[i,3]:.2f} N / R={peaks[i,4]:.2f} N | DIMM env={peaks[i,2]:.2f} N",font=font,fill=color)
 dr.text((8,79),f"Buttons L={peaks[i,0]:.2f} N / R={peaks[i,1]:.2f} N | "+("LIMIT EXCEEDED: recorded, continued" if bad else "No force exceedance this interval"),font=small,fill=color)
 reached=[e["stage"] for e in events if e["time"]<=times[i]+1e-6]
 dr.text((8,101),"Verified stages: "+str(reached),font=small,fill="white")
 dr.text((8,123),"Actual saved RGB and preceding-interval force peaks. Diagnostic only; no training.",font=small,fill=(185,190,200))
 for x,t in [(8,"External"),(264,"Wrist left"),(520,"Wrist right")]:dr.text((x,153),t,font=small,fill="white")
 return im
videos=[]
for name,selection,fps in [("actual_policy_replay.mp4",list(range(len(raw))),10),("after_previous_failure_slow.mp4",[i for i,t in enumerate(rel) if t>=85],5)]:
 if not selection:continue
 p=O/name
 with imageio.get_writer(p,fps=fps,codec="libx264",quality=8,macro_block_size=16,ffmpeg_params=["-movflags","+faststart"]) as w:
  for i in selection:w.append_data(np.asarray(frame(i,fps==5)))
 r=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-v","error","-i",str(p),"-f","null","-"],capture_output=True,text=True)
 if r.returncode:raise RuntimeError(r.stderr)
 videos.append(dict(path=str(p),fps=fps,frames=len(selection),seconds=len(selection)/fps,decode_ok=True))
for i in sorted({0,len(raw)//2,len(raw)-1}):frame(i).save(O/f"frame_{ids[i]:04d}.png")
with (O/"force_timeline.csv").open("w") as f:
 w=csv.writer(f);w.writerow(["index","time_s",*keys,"width_mm","left_deg","right_deg","physics_frames"])
 for i,(n,r) in enumerate(pairs):
  m=r["task_truth"]["metrics"];w.writerow([n,rel[i],*peaks[i],r["gripper_width_m"]*1000,m["left_angle_deg"],m["right_angle_deg"],counts[i]])
report=dict(videos=videos,simulation_elapsed_s=float(rel[-1]),observations=len(raw),footer_records=footer,
 incomplete_physics_lines=partial,summary=summary,force_peak_N=dict(zip(keys,map(float,peaks.max(axis=0)))),diagnostic_only=True)
(O/"manifest.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
