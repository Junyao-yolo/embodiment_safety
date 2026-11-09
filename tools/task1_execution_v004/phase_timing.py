"""Per-phase wall-time comparison on identical recorded-action boundaries."""
from pathlib import Path
import json,csv,statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
R=ROOT/"outputs/runs/task1/diffusion_policy/performance_retest_v004"
BASE=ROOT/"outputs/runs/task1/diffusion_policy/insertion_diagnosis_v001/recorded_full/eval/episode_01"
C=R/"batched_io_full"
with (R/"success_reference_progress.csv").open() as f:progress=list(csv.DictReader(f))
tbase=[(BASE/f"raw_{i:03d}.json").stat().st_mtime for i in range(1350)]
tnew=[(C/f"raw_{i:03d}.json").stat().st_mtime for i in range(1350)]
b=np.diff(tbase);n=np.diff(tnew)
ph=np.array([int(float(x["stage_progress"])) for x in progress[:-1]])
names={0:"Opening",1:"Grasp / rack clearance",2:"Transport / align",3:"Insert / lock / release"}
rows=[]
for k in sorted(set(ph)):
    bm=b[ph==k];nm=n[ph==k]
    rows.append(dict(phase=names.get(k,str(k)),cycles=len(bm),baseline_wall_s=float(bm.sum()),optimized_wall_s=float(nm.sum()),
        reduction=1-float(nm.sum()/bm.sum()),baseline_median_s=float(np.median(bm)),optimized_median_s=float(np.median(nm)),
        baseline_p95_s=float(np.percentile(bm,95)),optimized_p95_s=float(np.percentile(nm,95))))
(R/"controlled_phase_timing.json").write_text(json.dumps(dict(rows=rows,scope="Both runs use filesystem raw-observation boundary timestamps. Historical reference and current shared GPU; startup excluded."),indent=2))
with (R/"controlled_phase_timing.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
x=np.arange(len(rows));w=.36
axes[0].bar(x-w/2,[z["baseline_wall_s"]/60 for z in rows],w,label="Successful baseline")
axes[0].bar(x+w/2,[z["optimized_wall_s"]/60 for z in rows],w,label="Batched finite differences")
axes[0].set_xticks(x,[z["phase"] for z in rows],rotation=15,ha="right")
axes[0].set_ylabel("Wall minutes");axes[0].legend()
axes[1].plot(np.arange(1,1350)*.1,np.cumsum(b)/60,label="Baseline")
axes[1].plot(np.arange(1,1350)*.1,np.cumsum(n)/60,label="Optimized")
axes[1].set_xlabel("Simulation seconds");axes[1].set_ylabel("Cumulative wall minutes");axes[1].legend();axes[1].grid(alpha=.2)
fig.suptitle("Identical full successful action replay; Original rendering; shared-load historical comparison")
fig.tight_layout();fig.savefig(R/"controlled_full_replay_timing.png",dpi=160)
print(json.dumps(rows,indent=2))
