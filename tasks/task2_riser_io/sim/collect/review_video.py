"""Three-view videos with task acceptance measurements."""
from pathlib import Path
import json,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
FFMPEG='/home/ufdwaa/anaconda3/envs/env_isaaclab/lib/python3.10/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
class ReviewVideo:
 def __init__(self,out,title,tile=512):
  self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
  self.title=title;self.tile=tile;self.width=tile*3;self.height=tile+240
  self.font=ImageFont.truetype(FONT,20);self.small=ImageFont.truetype(FONT,17)
  self.log=(self.out/'encoder.log').open('w');self.count=0;self.last=None;self.last_rec=None
  self.pipe=subprocess.Popen([FFMPEG,'-y','-hide_banner','-loglevel','warning','-f','rawvideo','-pix_fmt','rgb24','-s',f'{self.width}x{self.height}','-r','10','-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(self.out/'full_review.mp4')],stdin=subprocess.PIPE,stderr=self.log)
 def put(self,im):
  self.pipe.stdin.write(np.asarray(im,dtype=np.uint8).tobytes())
 def frame(self,images,rec):
  im=Image.new('RGB',(self.width,self.height),(20,24,30));d=ImageDraw.Draw(im)
  d.text((12,8),self.title,fill='white',font=self.font)
  d.text((12,36),f"Simulation {rec['time_s']:.1f} s | Phase: {rec['phase']} | Playback: real simulation time",fill='white',font=self.small)
  for k,(name,label) in enumerate([('external','EXTERNAL'),('wrist_left','WRIST LEFT'),('wrist_right','WRIST RIGHT')]):
   image=Image.fromarray(images[name]).resize((self.tile,self.tile))
   im.paste(image,(k*self.tile,78))
   d.text((k*self.tile+12,58),label,fill=(180,220,255),font=self.small)
  y=78+self.tile+6
  xy=np.hypot(rec['ring_x']+.092,rec['ring_y']+.071)*1000
  d.text((12,y),f"Ring XY error: {xy:.3f} mm | Z error: {abs(rec['ring_z']-.818)*1000:.3f} mm | Tilt: {rec['ring_tilt_deg']:.2f} deg",fill='white',font=self.font)
  d.text((12,y+28),f"Head clamp (instantaneous): {rec['clamp_N']:.3f} N | Cable joint gap: {rec['joint_gap_m']*1000:.3f} mm",fill='white',font=self.font)
  d.text((12,y+56),"Final acceptance checks are shown at the end; instantaneous force is not a per-frame failure.",fill=(210,210,210),font=self.small)
  d.text((12,y+82),"Task acceptance only. Safety/damage assessment is UNASSESSED. Numerical thresholds are not calibrated damage limits.",fill=(210,210,210),font=self.small)
  self.put(im);self.count+=1;self.last=im;self.last_rec=rec
 def finish(self,report):
  checks={k:bool(report[k]) for k in ['wire_seated','screw_fastened','ring_retained','ring_pose_stable','cable_connectivity_valid','thread_engagement_verified','smoothness_pass'] if k in report}
  if 'acceptance_checks' in report:checks=dict(report['acceptance_checks'])
  elif report.get('mean_clamp_N') is not None:checks['retained_head_clamp']=report['mean_clamp_N']>.2
  failed=[k for k,v in checks.items() if not v]
  report=dict(report);report['acceptance_checks']=checks;report['failed_checks']=failed
  (self.out/'result.json').write_text(json.dumps(report,indent=2))
  if self.last is not None:
   im=self.last.copy();d=ImageDraw.Draw(im)
   d.rectangle((0,0,self.width,78),fill=(20,24,30))
   passed=report.get('accepted',False)
   d.text((12,8),('PASSED REPRODUCTION - not a failure video' if passed else 'FAILED TASK ACCEPTANCE - not a damage verdict'),font=self.font,fill=(110,240,160) if passed else (255,130,110))
   d.text((12,36),'RESULT CARD: final frame held for 8 seconds',font=self.small,fill='white')
   y=78+self.tile
   d.rectangle((0,y,self.width,self.height),fill=(20,24,30))
   msg='Failed checks: '+(', '.join(failed) or ('none' if passed else report.get('error','stage aborted')))
   lines=[msg]
   force_label='required > 0.2 N' if 'retained_head_clamp' in checks else 'diagnostic only'
   if report.get('mean_clamp_N') is not None:lines.append(f"Final clamp mean {report['mean_clamp_N']:.4f} N ({force_label}) | Max cable gap {report.get('max_joint_gap_m',0)*1000:.4f} mm (required < 0.5 mm)")
   if report.get('measured_thread_lead_m') is not None:lines.append(f"Thread lead {report['measured_thread_lead_m']*1000:.4f} mm/rev (1.8-2.2) | Fit error {report['thread_lead_fit_residual_m']*1000:.4f} mm (< 0.3)")
   e=report.get('ring_hold_extrema',{})
   if e:lines.append(f"Final ring: XY {e['max_xy_error_m']*1000:.4f} mm (< 1.5); Z {e['max_z_error_m']*1000:.4f} mm (< 0.7); tilt {e['max_tilt_deg']:.3f} deg (< 5)")
   for j,line in enumerate(lines):d.text((12,y+6+j*28),line,font=self.small,fill='white')
   for _ in range(80):self.put(im)
  self.pipe.stdin.close();rc=self.pipe.wait();self.log.close()
  if rc:raise RuntimeError('Video encoding failed; see '+str(self.out/'encoder.log'))
  start=max(0.,self.count/10-35.)
  subprocess.run([FFMPEG,'-y','-hide_banner','-loglevel','error','-ss',str(start),'-i',str(self.out/'full_review.mp4'),'-c:v','libx264','-preset','veryfast','-crf','20','-threads','2','-an','-movflags','+faststart',str(self.out/'final_review.mp4')],check=True)
  return report
