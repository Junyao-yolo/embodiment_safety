"""Episode-bounded image, state and action sequences for Task2."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from interface import CAMERAS,validate_episode
class Task2Data(Dataset):
 def __init__(self,root,horizon=16,stationary_stride=10):
  self.episodes=[];self.starts=[];self.horizon=horizon
  for p in sorted(Path(root).glob("episode_*")):
   if not (p/"result.json").exists():continue
   if not json.loads((p/"result.json").read_text()).get("accepted"):continue
   validate_episode(p)
   e={k:np.load(p/(k+".npy"),mmap_mode="r") for k in ["state","action",*CAMERAS]}
   e["path"]=str(p)
   meta=json.loads((p/"episode.json").read_text())
   phases=meta.get("phases",["unknown"]*len(e["state"]))
   holds={"settle","wire_verify","preload_verify","full_verify"}
   last_kept=-stationary_stride
   for i in range(len(e["state"])-horizon+1):
    x=np.asarray(e["state"][i:i+horizon]);a=np.asarray(e["action"][i:i+horizon]).reshape(-1,6,8)
    stationary=(len(set(phases[i:i+horizon]))==1 and phases[i] in holds
      and np.max(np.abs(x[:,7:14]))<.01
      and np.max(np.ptp(a[:,:,:7].reshape(-1,7),axis=0))<1e-4
      and np.ptp(a[:,:,7])<1e-4)
    if not stationary or i-last_kept>=stationary_stride:
     self.starts.append((len(self.episodes),i));last_kept=i
   self.episodes.append(e)
  if not self.starts:raise ValueError("No accepted Task2 sequences")
 def __len__(self):return len(self.starts)
 def __getitem__(self,index):
  ep,k=self.starts[index];e=self.episodes[ep]
  obs={c:torch.from_numpy(np.array(e[c][k:k+2])).permute(0,3,1,2).float()/255 for c in CAMERAS}
  obs["state"]=torch.from_numpy(np.array(e["state"][k:k+2]))
  return {"obs":obs,"action":torch.from_numpy(np.array(e["action"][k:k+self.horizon]))}
