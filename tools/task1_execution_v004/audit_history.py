"""Inventory existing DP evaluation records without inferring missing timings."""
import os,json,csv
from pathlib import Path
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
R=ROOT/"outputs/runs/task1/diffusion_policy"
OUT=R/"performance_retest_v004"
records={}
def add(item,source,output):
    if not output or "wall_s" not in item or not item.get("full_transitions"):return
    p=Path(output)
    if not p.is_absolute():p=ROOT/p
    key=str(p)
    row=dict(output=key,metadata=str(source),wall_s=item["wall_s"],steps=item["full_transitions"],
        sim_s=item.get("simulation_elapsed_s",item.get("simulation_transition_s",item["full_transitions"]*.1)),
        stages=item.get("stages"),reward=item.get("reward"),failure_reason=item.get("failure_reason",item.get("stage_failure")),
        inference_mean_ms=item.get("inference_mean_ms"),preprocess_inference_mean_ms=item.get("preprocess_plus_inference_mean_ms"),
        timeout=bool(item.get("wall_timeout")),wall_is_lower_bound=bool(item.get("wall_s_is_lower_bound")),
        excluded=bool(item.get("excluded_from_analysis") or item.get("interrupted_by_user")))
    if (p.parent.parent/"manual_interruption.json").exists():row["excluded"]=True;row["failure_reason"]="manual_performance_diagnostic_interruption"
    old=records.get(key)
    if old is None or sum(v is not None for v in row.values())>sum(v is not None for v in old.values()):records[key]=row
for dp,ds,fs in os.walk(R):
    depth=len(Path(dp).relative_to(R).parts)
    ds[:]=[d for d in ds if d not in ["images","train","episodes","replay","checkpoints","__pycache__"] and depth<4]
    for name in ["summary.json","status.json"]:
        if name not in fs:continue
        p=Path(dp)/name
        if p.stat().st_size>100000:continue
        data=json.loads(p.read_text())
        if not isinstance(data,dict):continue
        for e in data.get("episodes",[]):
            if isinstance(e,dict):add(e,p,e.get("output"))
        if "wall_s" in data and "full_transitions" in data:
            cmd=data.get("command",[])
            output=cmd[-1] if cmd and str(cmd[-1]).startswith(str(ROOT)) and "/episode_" in str(cmd[-1]) else None
            if output:add(data,p,output)
rows=list(records.values())
for x in rows:
    x["wall_per_transition_s"]=x["wall_s"]/x["steps"]
    # Mean may exclude model warmup; this is an estimate, not an exclusive profiler total.
    x["inference_s_estimate"]=None if x["inference_mean_ms"] is None else x["inference_mean_ms"]*x["steps"]/1000
    x["inference_wall_fraction_estimate"]=None if x["inference_s_estimate"] is None else x["inference_s_estimate"]/x["wall_s"]
rows.sort(key=lambda x:x["output"])
(OUT/"historical_tests.json").write_text(json.dumps(dict(rows=rows,notes="Unique output directories; missing component timers remain null. Different weights/controllers/safety modes are not pooled as a controlled comparison. Inference estimates use recorded means and completed transition count; no GPU/CPU overlap inference."),indent=2))
with (OUT/"historical_tests.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print("Historical tests with wall time",len(rows))
for x in rows:print(Path(x["output"]).relative_to(R),round(x["wall_s"]/60,2),x["steps"],x["reward"],x["inference_s_estimate"],x["excluded"])
