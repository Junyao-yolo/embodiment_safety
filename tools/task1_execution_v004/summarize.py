"""Summarize exclusive cycle timers and compare fixed recorded-action runs."""
import sys,json,statistics
from pathlib import Path
import numpy as np
def summarize(p):
    rows=[json.loads(s) for s in (p/"cycle_timing.jsonl").read_text().splitlines()]
    cycles=[x for x in rows if x["index"]>0]
    keys=sorted({k for x in cycles for k in x["exclusive_s"]})
    summary=dict(cycles=len(cycles),sim_s=.1*len(cycles),active_wall_s=sum(x["wall_s"] for x in cycles),
        cycle_median_s=statistics.median(x["wall_s"] for x in cycles),
        cycle_p95_s=float(np.percentile([x["wall_s"] for x in cycles],95)),
        modules={k:dict(total_s=sum(x["exclusive_s"].get(k,0.) for x in cycles),
           median_s=statistics.median(x["exclusive_s"].get(k,0.) for x in cycles),
           p95_s=float(np.percentile([x["exclusive_s"].get(k,0.) for x in cycles],95)),
           calls=sum(x["counts"].get(k,0) for x in cycles)) for k in keys})
    summary["wall_over_sim"]=summary["active_wall_s"]/summary["sim_s"]
    ik=[json.loads(s) for s in (p/"ik_calls.jsonl").read_text().splitlines()]
    summary["ik"]=dict(calls=len(ik),seconds=sum(x["tick_wall_s"] for x in ik),
       residual_calls=sum(x.get("residual_calls",0) for x in ik),
       jacobian_evaluations=sum(x.get("jacobian_calls",0) for x in ik),
       iterations=sum(x.get("iterations",0) for x in ik) if all("iterations" in x for x in ik) else None,solver_calls=sum(x.get("solver_calls",0) for x in ik),retries=sum(x.get("retries",0) for x in ik),
       max_position_error_m=max((x["position_error_m"] for x in ik),default=0),
       max_rotation_error_rad=max((x["orientation_error_rad"] for x in ik),default=0))
    summary["exclusive_sum_error_s"]=abs(summary["active_wall_s"]-sum(x["total_s"] for x in summary["modules"].values()))
    summary["unresolved_gpu_split"]="Native PhysX and graphics-render execution cannot be separated by these CPU wall timers."
    return summary
if __name__=="__main__":
    for arg in sys.argv[1:]:
        p=Path(arg);s=summarize(p);(p/"timing_summary.json").write_text(json.dumps(s,indent=2))
        print(str(p));print(json.dumps(s,indent=2))
