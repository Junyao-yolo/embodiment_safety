"""CPU inference cost for one versus eight executed 0.1-second actions."""
from pathlib import Path
import json,time,gc
import numpy as np
import torch
from policy import Task2Policy
from interface import CAMERAS
D=Path(__file__).resolve().parent
torch.set_num_threads(2)
ep=D/"data/raw/episode_0020"
states=np.load(ep/"state.npy");images={c:np.load(ep/(c+".npy"),mmap_mode="r") for c in CAMERAS}
results=[]
for count in (1,8):
 p=Task2Policy(D/"runs/pilot5/policy.pt",n_action_steps=count,fixed_noise=True,seed=103)
 row=states[0];q=np.r_[row[:7],row[14]/2,row[14]/2];v=np.r_[row[7:14],0.,0.];prev=np.r_[row[15:22],row[22]/2,row[22]/2]
 sample={c:np.asarray(images[c][0]) for c in CAMERAS}
 first=p.act(sample,q,v,prev);p.reset();again=p.act(sample,q,v,prev)
 assert np.array_equal(first,again),"Episode noise reset did not reproduce inference"
 p.reset();started=time.perf_counter()
 for i in range(16):
  row=states[i];q=np.r_[row[:7],row[14]/2,row[14]/2];v=np.r_[row[7:14],0.,0.];prev=np.r_[row[15:22],row[22]/2,row[22]/2]
  action=p.act({c:np.asarray(images[c][i]) for c in CAMERAS},q,v,prev)
  assert action.shape==(6,9)
 elapsed=time.perf_counter()-started
 assert p.inference_calls==16//count
 results.append(dict(n_action_steps=count,seconds_open_loop=count*.1,observations=16,inference_calls=p.inference_calls,wall_s=elapsed,fixed_noise_reproducible=True))
 p.act(sample,q,v,prev)
 before=p.inference_calls
 p.clear_history()
 p.act(sample,q,v,prev)
 assert p.inference_calls==before+1 and len(p.history)==1
 results[-1]["hold_history_flush_passed"]=True
 del p;gc.collect()
result=dict(passed=True,results=results,scope="CPU inference only; physical success and end-to-end speed unmeasured")
(D/"implement-stage/execution_benchmark.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
