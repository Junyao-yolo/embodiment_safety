"""Task2 synchronized observations and six-command action blocks."""
from pathlib import Path
import json
import numpy as np

CAMERAS=("external","wrist_left","wrist_right")
STATE_DIM=23
ACTION_DIM=48
IMAGE_SIZE=128
CONTRACT="task2_joint_chunk_rgb_v001"
JOINT_MIN=np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-.0175,-2.8973],np.float32)
JOINT_MAX=np.array([2.8973,1.7628,2.8973,-.0698,2.8973,3.7525,2.8973],np.float32)
def observation_state(joints,velocities,targets):
 q=np.asarray(joints);v=np.asarray(velocities);u=np.asarray(targets)
 return np.r_[q[:7],v[:7],q[7:9].sum(),u[:7],u[7:9].sum()].astype(np.float32)
def action_block_to_targets(action):
 a=np.asarray(action,np.float32).reshape(6,8)
 if not np.isfinite(a).all():raise ValueError("Nonfinite action")
 if np.any(a[:,:7]<JOINT_MIN-1e-4) or np.any(a[:,:7]>JOINT_MAX+1e-4) or np.any(a[:,7]<-1e-4) or np.any(a[:,7]>.0801):
  raise ValueError("Command outside joint or gripper range")
 return np.concatenate([a[:,:7],a[:,7:8]/2,a[:,7:8]/2],axis=1)
def validate_episode(path):
 p=Path(path);meta=json.loads((p/"episode.json").read_text());n=meta["samples"]
 x=np.load(p/"state.npy",mmap_mode="r");u=np.load(p/"action.npy",mmap_mode="r");t=np.load(p/"time.npy")
 if x.shape!=(n,23) or u.shape!=(n,48):raise ValueError("State/action shape")
 if not np.isfinite(x).all() or not np.isfinite(u).all():raise ValueError("Nonfinite data")
 if n<16 or not np.allclose(np.diff(t),.1,atol=1e-6):raise ValueError("Noncontiguous timestamps")
 for c in CAMERAS:
  z=np.load(p/(c+".npy"),mmap_mode="r")
  if z.shape!=(n,128,128,3) or z.dtype!=np.uint8:raise ValueError("Camera shape")
 for a in u:action_block_to_targets(a)
 if meta.get("collection_method") in ("dart","dagger"):
  execution=np.load(p/"executed_action.npy",mmap_mode="r")
  sources=np.load(p/"action_source.npy",mmap_mode="r")
  if execution.shape!=(n,48) or sources.shape!=(n,6):raise ValueError("Execution provenance shape")
  for a in execution:action_block_to_targets(a)
 return {"samples":n,"state_dim":23,"action_dim":48,"cameras":list(CAMERAS),"passed":True}
def save_episode(out,records,frames,image_frames,report,cameras):
 out=Path(out)
 if not report.get("accepted"):raise ValueError("Only successful episodes")
 if len(image_frames)!=len(frames):raise ValueError("Image timestamps missing")
 hz=report["physics_rate_Hz"]
 if hz!=120:raise ValueError("Task2 collection uses 120 Hz physics")
 # Observation at t; first target acts on (t,t+1/60], never the command before t.
 valid=[i for i,f in enumerate(frames) if round(f["time_s"]*hz)+12<=len(records)]
 ts=[];xs=[];us=[];executed=[];sources=[];phases=[]
 for i in valid:
  f=frames[i];k=round(f["time_s"]*hz);o=records[k-1]
  if abs(o["time_s"]-f["time_s"])>1e-6:raise ValueError("Image/state timestamp")
  if not np.allclose(o["robot_joint_positions"],f["robot_joints"],atol=1e-6):raise ValueError("Image/state pose")
  block=[];execution=[];source=[]
  for j in range(6):
   row=records[k+2*j]
   q=np.asarray(row["robot_joint_targets"])
   expert=np.asarray(row.get("expert_joint_targets",q))
   if report.get("collection_method") in ("dart","dagger") and "expert_joint_targets" not in row:
    raise ValueError("Recovery data requires queried expert labels")
   block.append(np.r_[expert[:7],expert[7:9].sum()])
   execution.append(np.r_[q[:7],q[7:9].sum()])
   source.append(row.get("action_source","expert"))
  action=np.asarray(block,np.float32).reshape(48);action_block_to_targets(action)
  ts.append(f["time_s"]);xs.append(observation_state(o["robot_joint_positions"],o["robot_joint_velocities"],o["robot_joint_targets"]))
  us.append(action);executed.append(np.asarray(execution).reshape(48));sources.append(source);phases.append(f["phase"])
 out.mkdir(parents=True,exist_ok=False)
 np.save(out/"time.npy",np.asarray(ts,np.float64));np.save(out/"state.npy",np.asarray(xs,np.float32));np.save(out/"action.npy",np.asarray(us,np.float32))
 np.save(out/"executed_action.npy",np.asarray(executed,np.float32))
 np.save(out/"action_source.npy",np.asarray(sources,dtype="<U16"))
 for c in CAMERAS:
  z=np.lib.format.open_memmap(out/(c+".npy"),mode="w+",dtype=np.uint8,shape=(len(valid),128,128,3))
  for j,i in enumerate(valid):z[j]=image_frames[i][c]
  z.flush();del z
 meta={"interface":CONTRACT,"samples":len(valid),"state_dim":23,"action_dim":48,
       "state_fields":["joint_position[7]","joint_velocity[7]","gripper_width[1]","previous_joint_target[7]","previous_gripper_target[1]"],
       "action":"6 expert-or-hold joint_position[7]+gripper_width[1] commands after each observation",
       "executed_action":"actual 60Hz commands; previous-target state always uses these",
       "label_semantics":"active motion: expert queries along the visited rollout; physical waits: hold commands; not a counterfactual expert rollout",
       "collection_method":report.get("collection_method","expert"),
       "initial_conditions":report.get("initial_conditions"),
       "route_mode":report.get("route_mode"),
       "action_source_values":["expert","dart","policy","recovery","hold"],
       "observation_hz":10,"command_hz":60,"physics_hz":120,"image_shape":[128,128,3],"cameras":cameras,"phases":phases}
 (out/"episode.json").write_text(json.dumps(meta,indent=2))
 result=validate_episode(out);(out/"validation.json").write_text(json.dumps(result,indent=2))
 (out/"result.json").write_text(json.dumps(report,indent=2))
 diagnostics={k:report[k] for k in ("numerical_quality","diagnostic_checks","smoothness","command_limit_events","max_joint_gap_m","measured_thread_lead_m","thread_lead_fit_residual_m","largest_ik_residual","hand_tracking_errors_m","incomplete_reference_tracking_phases","measurements_path","mean_clamp_N") if k in report}
 diagnostics["role"]="Recorded numerical quality; not task acceptance or calibrated damage limits"
 (out/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2))

