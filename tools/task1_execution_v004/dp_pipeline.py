from pathlib import Path
import os,sys,json,time,subprocess,datetime,statistics,argparse
args=argparse.ArgumentParser();args.add_argument("--name",default="dp_episode_001");args.add_argument("--variant",default="cached_fd",choices=["cached_fd","batched_fast","batched_io"]);args=args.parse_args()
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety");sys.path.insert(0,str(ROOT))
from runners.eval_diffusion import stop_group,label_summary
HERE=Path(__file__).resolve().parent
B=ROOT/"outputs/runs/task1/diffusion_policy/performance_retest_v004"/args.name;B.mkdir(exist_ok=False);D=B/"eval/episode_01"
CP=ROOT/"outputs/runs/task1/diffusion_policy/full_flow_bc_v001/train/last.pt"
report=dict(status="starting",role="fixed_replay_verified_execution",seed=20261035,checkpoint=str(CP),wall_timeout_s=2700,started_at_unix=time.time())
def save():
 p=B/"status.json";tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(report,indent=2));tmp.replace(p)
save()
env=os.environ.copy()
env.update(TASK1_POLICY_BACKEND="diffusion_policy",TASK1_EVAL="1",TASK1_ONLINE_LEARN="0",TASK1_MAX_STEPS="0",TASK1_SMOKE="0",TASK1_EPISODE_SEED="20261035",TASK1_SAFETY_MODE="warn",SERL_INPUT_CHECKPOINT=str(CP))
env.pop("TASK1_DIAGNOSTIC_STOP_STAGE",None)
env.pop("TASK1_WINDOW_GATES",None)
env["TASK1_EXECUTION_VARIANT"]=args.variant
cmd=["/mnt/data2/isaacsim-5.0.0/python.sh","-B",str(HERE/"actor.py"),str(D)]
start=time.monotonic()
with (B/"actor.log").open("w") as f:
 proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 report.update(status="running",pid=proc.pid,command=cmd,started_at_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat());save()
 try:rc=proc.wait(timeout=2700)
 except subprocess.TimeoutExpired:
  stop_group(proc);rc=proc.returncode;report["wall_timeout"]=True
report.update(wall_s=time.monotonic()-start,returncode=rc,finished_at_beijing=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
result=D/"result.json"
report["result"]=json.loads(result.read_text()) if result.exists() else dict(full_task_success=False,task_completion_observed=False,failure_reason="wall_time_limit" if report.get("wall_timeout") else "actor_failed")
report.update(label_summary(D))
if (B/"manual_interruption.json").exists():
 report["manual_interruption"]=json.loads((B/"manual_interruption.json").read_text());report["exclude_from_policy_success_statistics"]=True
raw=sorted(D.glob("raw_*.json"),key=lambda p:int(p.stem.rsplit("_",1)[1]))
sim_s=json.loads(raw[-1].read_text())["sim_time"]-json.loads(raw[0].read_text())["sim_time"] if raw else 0.
report["label_rows"]=report.get("full_transitions",0)
report["full_transitions"]=report["result"].get("full_transitions",int((sim_s+1e-6)/.1))
report.update(status="rendering",simulation_elapsed_s=sim_s,end_to_end_realtime_factor=sim_s/report["wall_s"])
actions=[json.loads(p.read_text()) for p in sorted(D.glob("action_*.json"))][1:]
report["inference_mean_ms"]=statistics.mean([a["inference_ms"] for a in actions]) if actions else None
report["preprocess_plus_inference_mean_ms"]=statistics.mean([a["request_compute_ms"] for a in actions]) if actions else None
report["controller_diagnostics"]=json.loads((D/"controller_diagnostics.json").read_text()) if (D/"controller_diagnostics.json").exists() else None
save()
(B/"eval/summary.json").write_text(json.dumps(report,indent=2))
render_start=time.monotonic()
with (B/"render.log").open("w") as f:
 video_rc=subprocess.call([str(ROOT/"algos/diffusion_policy/.venv/bin/python"),"-B",str(HERE/"render_replay.py"),str(B)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
report.update(status="finished",video_returncode=video_rc,video_ready=video_rc==0,video=str(B/"replay/actual_policy_replay.mp4"),render_wall_s=time.monotonic()-render_start)
save()
