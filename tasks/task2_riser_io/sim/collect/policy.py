"""Task2 observation history and executable DP action blocks."""
import numpy as np
import torch
from interface import CAMERAS,observation_state,action_block_to_targets,JOINT_MIN,JOINT_MAX
from model import build
class Task2Policy:
 def __init__(self,checkpoint,device="cpu",n_action_steps=1,fixed_noise=False,seed=42):
  self.device=device;payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
  self.action_representation=payload["config"].get("action_representation","absolute_joint_targets")
  self.model=build(payload["config"]);self.model.load_state_dict(payload["model"]);self.model=self.model.to(device).eval()
  if not 1<=n_action_steps<=self.model.horizon-self.model.n_obs_steps+1:raise ValueError("Action horizon exceeds prediction")
  self.model.n_action_steps=n_action_steps
  self.fixed_noise=fixed_noise;self.seed=int(seed)
  self.generator=torch.Generator(device=device);self.model.kwargs["generator"]=self.generator
  self.reset()
 def clear_history(self):
  self.history=[];self.pending=[]
 def clear_actions(self):
  self.pending=[]
 def reset(self):
  self.clear_history();self.inference_calls=0;self.generator.manual_seed(self.seed)
 @torch.no_grad()
 def act(self,images,joints,velocities,previous_targets):
  state=observation_state(joints,velocities,previous_targets)
  sample={**{c:np.asarray(images[c]).copy() for c in CAMERAS},"state":state}
  if any(sample[c].shape!=(128,128,3) for c in CAMERAS):raise ValueError("Expected synchronized Task2 128x128 RGB")
  self.history=(self.history+[sample])[-2:]
  if self.pending:return self.decode(self.pending.pop(0),previous_targets)
  pair=self.history if len(self.history)==2 else [sample,sample]
  batch={c:torch.from_numpy(np.stack([o[c] for o in pair])).permute(0,3,1,2)[None].float().to(self.device)/255 for c in CAMERAS}
  batch["state"]=torch.from_numpy(np.stack([o["state"] for o in pair]))[None].to(self.device)
  if self.fixed_noise:self.generator.manual_seed(self.seed)
  actions=self.model.predict_action(batch)["action"][0].cpu().numpy()
  self.inference_calls+=1
  self.pending=list(actions)
  return self.decode(self.pending.pop(0),previous_targets)
 def decode(self,action,previous_targets):
  values=np.asarray(action,np.float32).reshape(6,8).copy()
  if self.action_representation in ("joint_delta_width_v001","expert_joint_delta_width_v002"):
   values[:,:7]=np.asarray(previous_targets)[:7]+np.cumsum(values[:,:7],axis=0)
   values[:,:7]=np.clip(values[:,:7],JOINT_MIN,JOINT_MAX)
  return action_block_to_targets(values.reshape(48))
def execute_block(targets,robot,indices,physics_step,physics_hz=120):
 """Apply six 60 Hz commands; physics_step advances exactly one physics step."""
 targets=np.asarray(targets,np.float32)
 if targets.shape!=(6,9) or physics_hz%60:raise ValueError("Control block shape or physics frequency")
 for q in targets:
  robot.set_dof_position_targets(q[None],indices)
  for _ in range(physics_hz//60):physics_step()
