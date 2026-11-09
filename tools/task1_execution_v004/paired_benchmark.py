"""Adjacent lightweight baseline/candidate timing under the current shared load."""
import os,sys,signal,time,json,subprocess
from pathlib import Path
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
HERE=Path(__file__).resolve().parent
OUT=ROOT/"outputs/runs/task1/diffusion_policy/performance_retest_v004"
PY=ROOT/"algos/diffusion_policy/.venv/bin/python"
pid=int(sys.argv[1])
def status(phase,**more):
    (OUT/"paired_benchmark_status.json").write_text(json.dumps(dict(phase=phase,**more),indent=2))
try:
    status("waiting_for_full_replay")
    while True:
        s=json.loads((OUT/"batched_io_full_status.json").read_text())
        if s["status"]!="running":break
        time.sleep(5)
    if s["status"]!="finished":raise RuntimeError("Complete replay failed")
    for variant,name in [("baseline","paired_baseline_short"),("batched_io","paired_optimized_short")]:
        status("running_"+name)
        subprocess.run(["python3","-B",str(HERE/"run.py"),"--variant",variant,"--steps","60","--name",name],cwd=ROOT,check=True)
        subprocess.run([str(PY),"-B",str(HERE/"summarize.py"),str(OUT/name)],cwd=ROOT,check=True)
    rc=subprocess.run([str(PY),"-B",str(HERE/"compare.py"),str(OUT/"paired_baseline_short"),str(OUT/"paired_optimized_short"),str(OUT/"paired_comparison.json")],cwd=ROOT).returncode
    a=json.loads((OUT/"paired_baseline_short/timing_summary.json").read_text())
    b=json.loads((OUT/"paired_optimized_short/timing_summary.json").read_text())
    result=dict(baseline=a,candidate=b,trajectory_passed=rc==0,
        active_wall_reduction=1-b["active_wall_s"]/a["active_wall_s"],
        ik_reduction=1-b["ik"]["seconds"]/a["ik"]["seconds"],
        scope="Adjacent 60-action runs with lightweight timers. GPU remains shared; this is a short-window estimate, not a replacement for complete-replay duration.")
    (OUT/"paired_benchmark.json").write_text(json.dumps(result,indent=2))
    status("finished",trajectory_passed=rc==0,active_wall_reduction=result["active_wall_reduction"])
except Exception as e:
    status("error",error=repr(e))
    raise
finally:
    os.kill(pid,signal.SIGCONT)
