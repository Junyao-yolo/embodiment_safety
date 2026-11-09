"""Resume Task2 after the selected formal Task1 run has finished."""
from pathlib import Path
import json,subprocess,time,os
D=Path(__file__).resolve().parent;R=D.parents[1]
config=json.loads((D/"wait_for_task1.json").read_text())
result_path=Path(config["task1_result"])
state={"status":"WAITING_FOR_TASK1","watcher_pid":os.getpid(),"required_steps":config["required_steps"]}
def live_compute():
    output=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits"],text=True)
    return [int(line.strip()) for line in output.splitlines() if line.strip().isdigit() and Path("/proc",line.strip()).exists()]
while True:
    result=json.loads(result_path.read_text()) if result_path.exists() else {}
    finished=(result.get("status")=="finished" and result.get("steps",0)>=config["required_steps"]
              and result_path.stat().st_mtime>=config["task1_started_unix"])
    active=live_compute() if finished else []
    state.update(checked_at=time.time(),task1_status=result.get("status"),task1_steps=result.get("steps"),other_compute_pids=active)
    (D/"wait_status.json").write_text(json.dumps(state,indent=2)+"\n")
    if finished and not active:
        collectors=[]
        for p in Path("/proc").iterdir():
            if not p.name.isdigit():continue
            try:cmd=(p/"cmdline").read_bytes().split(b"\0")
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            if str(D/"collect.py").encode() in cmd or str(D/"collect_persistent.py").encode() in cmd:collectors.append(int(p.name))
        if not collectors:
            with (D/"launcher.log").open("a") as log:
                proc=subprocess.Popen(["/usr/bin/python3","-B",str(D/"collect.py")],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                collectors=[proc.pid]
        state.update(status="TASK2_RESUMED",collector_pids=collectors,resumed_at=time.time())
        (D/"wait_status.json").write_text(json.dumps(state,indent=2)+"\n")
        break
    time.sleep(60)
