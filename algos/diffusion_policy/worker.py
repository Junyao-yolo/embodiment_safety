"""Read-only policy worker: actual observations in, official DP action out."""
import os,sys,time,json,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from algos.diffusion_policy.inference import GraphPolicy
from algos.diffusion_policy.action_chunk import ActionChunk
import numpy as np
import torch
from algos.diffusion_policy.common import CAMERAS,build,digest
from algos.diffusion_policy.representation import encode_state,validate_config
OUT=Path(sys.argv[1])
def write(name,data):
 p=OUT/name;tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(data,indent=2));tmp.replace(p)
try:
 torch.set_num_threads(2)
 if os.environ.get("TASK1_ONLINE_LEARN")=="1":raise ValueError("DP evaluation cannot learn")
 cp=Path(os.environ["SERL_INPUT_CHECKPOINT"]);meta=json.loads(cp.with_suffix(".json").read_text())
 if digest(cp)!=meta["sha256"]:raise ValueError("Checkpoint changed")
 seed=int(os.environ.get("TASK1_EPISODE_SEED","42"));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
 state=torch.load(cp,map_location="cpu",weights_only=False);model=build(state["config"])
 contract=validate_config(state["config"])
 if meta.get("interface_version")!=contract["interface_version"]:raise ValueError("Checkpoint metadata interface mismatch")
 trained_action_steps=model.n_action_steps
 action_steps=int(os.environ.get("TASK1_DP_ACTION_STEPS",str(trained_action_steps)))
 if not 1 <= action_steps <= model.horizon-model.n_obs_steps+1:raise ValueError("Invalid execution chunk")
 model.n_action_steps=action_steps
 chunk=ActionChunk(action_steps)
 model.load_state_dict(state["ema"]);model=model.cuda().eval()
 write("loaded_checkpoint.json",dict(path=str(cp),sha256=meta["sha256"],algorithm="diffusion_policy",weights="ema",seed=seed,trained_action_steps=trained_action_steps,runtime_action_steps=action_steps,**contract))
 history=[];n=0;graph=None
 while True:
  request=OUT/f"request_{n:03d}.json";deadline=time.monotonic()+600
  while not request.exists() and not (OUT/"policy_stop.json").exists():
   if (OUT/"actor_error.json").exists():raise RuntimeError("Actor failed")
   if time.monotonic()>deadline:raise TimeoutError("Policy request")
   time.sleep(.002)
  if not request.exists():break
  request_start=time.perf_counter()
  with np.load(OUT/f"obs_{n:03d}.npz",allow_pickle=False) as f:obs={k:f[k][0].copy() for k in f.files}
  history.append(obs);history=history[-2:]
  pair=history if len(history)==2 else [history[0],history[0]]
  control=json.loads(request.read_text())
  timings=dict(preprocessing_ms=0.,inference_ms=0.,graph_capture_ms=0.)
  def predict():
   global graph
   pre_start=time.perf_counter()
   batch={k:torch.from_numpy(np.stack([o[k] for o in pair])).permute(0,3,1,2)[None].float().cuda()/255 for k in CAMERAS}
   batch["state"]=torch.from_numpy(encode_state(np.stack([o["state"] for o in pair]),contract["state_representation"]))[None].float().cuda()
   torch.cuda.synchronize()
   timings["preprocessing_ms"]=(time.perf_counter()-pre_start)*1000
   if graph is None:
    graph=GraphPolicy(model,batch);timings["graph_capture_ms"]=graph.capture_s*1000
   infer_start=time.perf_counter()
   actions=graph.predict_action(batch)["action"][0].cpu().numpy().copy()
   timings["inference_ms"]=(time.perf_counter()-infer_start)*1000
   if actions.shape!=(action_steps,7) or not np.isfinite(actions).all():raise ValueError("Invalid action chunk")
   return actions
  action,inferred,plan_start,chunk_offset=chunk.take(n,control.get("action_source","policy"),predict)
  if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError("Invalid action")
  executed=np.clip(action,-1,1)
  write(f"action_{n:03d}.json",dict(index=n,proposed=action.tolist(),executed=executed.tolist(),network_updates=0,
   sampling_mode="seeded_ddim_eta0",seed=seed,history_frames=len(history),action_horizon=model.horizon,executed_chunk=action_steps,
   inference_performed=inferred,plan_start_index=plan_start,chunk_offset=chunk_offset,
   **timings,
   request_compute_ms=(time.perf_counter()-request_start)*1000))
  n+=1
 stop=json.loads((OUT/"policy_stop.json").read_text())
 write("worker_result.json",dict(passed=True,network_updates=0,new_transitions=stop["final_observation_index"],learning=False,algorithm="diffusion_policy"))
except BaseException:
 write("worker_error.json",dict(error=traceback.format_exc()));raise
