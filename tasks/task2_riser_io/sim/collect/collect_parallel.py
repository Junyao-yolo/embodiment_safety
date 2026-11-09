"""Collect a fixed quota of new-camera demonstrations in persistent workers."""
from pathlib import Path
import argparse,json,subprocess,time
D=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument("--output-root",type=Path,required=True)
parser.add_argument("--workers",type=int,default=2)
parser.add_argument("--successes",type=int,default=100)
parser.add_argument("--seed",type=int,default=100)
parser.add_argument("--collection-config",type=Path,default=D/"feedback_dart.json")
parser.add_argument("--max-attempts",type=int,default=150)
opts=parser.parse_args()
if opts.workers not in (1,2):parser.error("Use one or two Task2 workers")
root=opts.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
started=time.time();processes={};handles={};quotas={i:opts.successes//opts.workers+(i<opts.successes%opts.workers) for i in range(opts.workers)}
def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return {}
def launch(i):
    seed=opts.seed+(i-opts.seed)%opts.workers
    handles[i]=(root/f"worker_{i}.log").open("a")
    cmd=["/mnt/data2/isaacsim-5.0.0/python.sh","-B",str(D/"collect_persistent.py"),
         "--output-root",str(root),"--worker-id",str(i),"--num-workers",str(opts.workers),
         "--successes",str(quotas[i]),"--seed",str(seed),"--max-attempts",str(opts.max_attempts),
         "--collection-config",str(opts.collection_config.resolve())]
    processes[i]=subprocess.Popen(cmd,cwd=D,stdout=handles[i],stderr=subprocess.STDOUT)
    print(json.dumps({"event":"worker_started","worker":i,"pid":processes[i].pid,"target":quotas[i],"seed":seed}),flush=True)
launch(0)
while True:
    states={i:read(root/(f"progress_worker_{i}.json" if opts.workers>1 else "progress.json")) for i in processes}
    # Verify a completed episode and an in-place reset before adding another simulator.
    if opts.workers==2 and 1 not in processes and states.get(0,{}).get("completed_successes",0)>=min(2,quotas[0]):
        launch(1)
    count=sum(s.get("completed_successes",0) for s in states.values())
    failures={i:p.returncode for i,p in processes.items() if p.poll() is not None and p.returncode!=0}
    all_finished=len(processes)==opts.workers and all(p.poll() is not None for p in processes.values())
    status="COMPLETE" if all_finished and count>=opts.successes else ("NEEDS_REPAIR" if failures else ("VALIDATING_FIRST_WORKER" if len(processes)<opts.workers else "RUNNING"))
    result={"status":status,"target_successes":opts.successes,"completed_successes":count,"camera_layout":"opposed_v002",
            "started_unix":started,"wall_s":time.time()-started,"workers":states,
            "worker_pids":{i:p.pid for i,p in processes.items()},"worker_exit_errors":failures,
            "data_root":str(root/"raw")}
    (root/"progress.json" if opts.workers>1 else root/"batch_progress.json").write_text(json.dumps(result,indent=2))
    (D/"progress.json").write_text(json.dumps(result,indent=2))
    if all_finished or (failures and all(p.poll() is not None for p in processes.values())):break
    time.sleep(5)
for handle in handles.values():handle.close()
print(json.dumps({"status":status,"completed_successes":count,"target_successes":opts.successes}),flush=True)
raise SystemExit(0 if status=="COMPLETE" else 2)
