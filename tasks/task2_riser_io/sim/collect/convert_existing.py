"""Convert accepted Task2 recordings to synchronized DP-BC episodes."""
from pathlib import Path
import sys,csv,json,subprocess
import numpy as np
from interface import save_episode,CAMERAS
R=Path(__file__).resolve().parents[2]
ARTIFACTS=Path('/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io')
sys.path.insert(0,str(R/"scene/single_wire_fastening_v003/video_dependencies"))
import imageio_ffmpeg
for seed in (4,5):
 src=R/"scene/success_v014"/f"episode_{seed:04d}";dst=ARTIFACTS/"data/raw"/src.name
 if (dst/"result.json").exists():continue
 report=json.loads((src/"result.json").read_text())
 if not report.get("accepted"):raise ValueError("Source episode failed")
 frames=json.loads((src/"frames.json").read_text());records=[]
 with (src/"metrics.csv").open() as f:
  for row in csv.DictReader(f):
   records.append({"time_s":float(row["time_s"]),"robot_joint_positions":json.loads(row["robot_joint_positions"]),
     "robot_joint_velocities":json.loads(row["robot_joint_velocities"]),"robot_joint_targets":json.loads(row["robot_joint_targets"])})
 images=[{} for _ in frames]
 for c in CAMERAS:
  proc=subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),"-v","error","-threads","1","-i",str(src/(c+".mp4")),"-vf","scale=128:128","-threads","1","-f","rawvideo","-pix_fmt","rgb24","pipe:1"],stdout=subprocess.PIPE)
  for i in range(len(frames)):
   raw=proc.stdout.read(128*128*3)
   if len(raw)!=128*128*3:raise ValueError("Video frame count shorter than timestamps")
   images[i][c]=np.frombuffer(raw,np.uint8).reshape(128,128,3)
  if proc.stdout.read(1):raise ValueError("Video longer than timestamps")
  if proc.wait()!=0:raise ValueError("Decoder failed")
 cameras=json.loads((src/"cameras.json").read_text())
 for c in CAMERAS:cameras[c]["resolution"]=[128,128]
 report["source_episode"]=str(src);report["collection_contract"]="task2_dp_bc_v001"
 save_episode(dst,records,frames,images,report,cameras)
 print(json.dumps({"converted":str(dst),"samples":len(frames)-1}),flush=True)
 del images,records,frames
