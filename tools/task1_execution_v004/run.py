"""Run one isolated recorded-action benchmark."""
import os,sys,json,subprocess,time,hashlib,argparse
from pathlib import Path
P=argparse.ArgumentParser();P.add_argument("--variant",choices=["baseline","analytic","fast","cached_fd","batched_fd","batched_fast","batched_io"],required=True);P.add_argument("--steps",type=int,required=True);P.add_argument("--name",required=True)
a=P.parse_args();ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety");HERE=Path(__file__).resolve().parent
R=ROOT/"outputs/runs/task1/diffusion_policy/performance_retest_v004";R.mkdir(exist_ok=True)
out=R/a.name
if out.exists():raise FileExistsError(out)
contract=json.loads((HERE/"contract.json").read_text())
source=Path(contract["source"])
env=os.environ.copy()
env.update(PYTHONDONTWRITEBYTECODE="1",CUDA_VISIBLE_DEVICES="0",TASK1_EVAL="1",TASK1_ONLINE_LEARN="0",TASK1_POLICY_BACKEND="recorded_action",TASK1_SAFETY_MODE=contract["safety_mode"],TASK1_DIAGNOSTIC_STOP_STAGE="",TASK1_SMOKE="0",TASK1_RECORDED_SOURCE=str(source),TASK1_RECORDED_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),TASK1_MAX_STEPS=str(a.steps),TASK1_EXECUTION_VARIANT=a.variant)
cmd=["/mnt/data2/isaacsim-5.0.0/python.sh","-B",str(HERE/"actor.py"),str(out)]
start=time.time()
status=R/(a.name+"_status.json")
def write(data):status.write_text(json.dumps(data,indent=2))
with (R/(a.name+".log")).open("x") as f:
 p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
 write(dict(status="running",variant=a.variant,steps=a.steps,pid=p.pid,started_unix=start,command=cmd))
 rc=p.wait()
result=json.loads((out/"result.json").read_text()) if (out/"result.json").exists() else {}
write(dict(status="finished" if rc==0 and result else "error",variant=a.variant,steps=a.steps,returncode=rc,wall_s=time.time()-start,result=result))
raise SystemExit(rc if rc else (0 if result else 1))
