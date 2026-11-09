"""Validate the fixed replay before allowing one unchanged DP evaluation."""
import sys,time,json,subprocess
from pathlib import Path
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety");HERE=Path(__file__).resolve().parent
OUT=ROOT/"outputs/runs/task1/diffusion_policy/performance_retest_v004"
PY=ROOT/"algos/diffusion_policy/.venv/bin/python"
def phase(name,**more):
    (OUT/"controlled_workflow_status.json").write_text(json.dumps(dict(phase=name,**more),indent=2))
phase("waiting_for_previous_dp")
import os
while json.loads((OUT/"optimized_workflow_status.json").read_text())["phase"] not in ["finished","dp_pipeline_error"]:
    time.sleep(10)
phase("controlled_short_replay")
subprocess.run(["python3","-B",str(HERE/"run.py"),"--variant","batched_io","--steps","60","--name","batched_io_short"],cwd=ROOT,check=True)
ref=ROOT/"outputs/runs/task1/diffusion_policy/insertion_diagnosis_v001/recorded_full/eval/episode_01"
for args in [
    [str(HERE/"compare.py"),str(ref),str(OUT/"batched_io_short"),str(OUT/"controlled_short_comparison.json")],
    [str(HERE/"check_cameras.py"),str(OUT/"batched_io_short"),str(OUT/"controlled_short_cameras.json")],
    [str(HERE/"summarize.py"),str(OUT/"batched_io_short")]]:
    rc=subprocess.run([str(PY),"-B"]+args,cwd=ROOT).returncode
    if rc:phase("controlled_short_failed",command=args);raise SystemExit(1)
phase("controlled_full_replay")
env=os.environ.copy();env["TASK1_WINDOW_GATES"]="1";env.pop("TASK1_NATIVE_PROFILE",None)
subprocess.run(["python3","-B",str(HERE/"run.py"),"--variant","batched_io","--steps","1349","--name","batched_io_full"],cwd=ROOT,env=env)

while True:
    s=json.loads((OUT/"batched_io_full_status.json").read_text())
    if s["status"]!="running":break
    time.sleep(10)
if s["status"]!="finished":
    phase("replay_gate_failed",result=s);raise SystemExit(1)
phase("auditing_complete_replay")
C=OUT/"batched_io_full"
subprocess.run([str(PY),"-B",str(HERE/"summarize.py"),str(C)],check=True,cwd=ROOT)
subprocess.run([str(PY),"-B",str(HERE/"audit_episode.py"),str(C),str(OUT/"controlled_contact_audit.json")],check=True,cwd=ROOT)
ref=ROOT/"outputs/runs/task1/diffusion_policy/insertion_diagnosis_v001/recorded_full/eval/episode_01"
rc=subprocess.run([str(PY),"-B",str(HERE/"compare.py"),str(ref),str(C),str(OUT/"controlled_full_comparison.json")],cwd=ROOT).returncode
a=json.loads((OUT/"controlled_contact_audit.json").read_text())
camera_rc=subprocess.run([str(PY),"-B",str(HERE/"check_cameras.py"),str(C),str(OUT/"controlled_full_cameras.json")],cwd=ROOT).returncode
result=s["result"]
limits=dict(left_press_N=3.,right_press_N=3.,finger_left_N=5.,finger_right_N=5.,dimm_environment_N=5.)
passed=bool(rc==0 and camera_rc==0 and result.get("full_task_success") and not result.get("failure_reason")
    and result.get("source_unchanged") and result["full_transitions"]==1349
    and result["safety_warnings"]["violating_physics_frames"]==0 and result["evidence_warning_intervals"]==0
    and a["transfer_alignment_passed"] and a["maximum_recorded_dimm_penetration_m"]<=.0001
    and all(a["force_peaks_N"][k]<=v for k,v in limits.items()))
baseline_wall=(ref/"raw_1349.json").stat().st_mtime-(ref/"raw_000.json").stat().st_mtime
candidate_wall=(C/"raw_1349.json").stat().st_mtime-(C/"raw_000.json").stat().st_mtime
timing=json.loads((C/"timing_summary.json").read_text())
validation=timing["modules"].get("validation_gate",{}).get("total_s",0.)
acceptance=dict(execution_equivalence_ready=passed,baseline_active_wall_s=baseline_wall,
    candidate_active_wall_s=candidate_wall,validation_gate_s=validation,
    active_reduction_fraction=1-candidate_wall/baseline_wall,
    engineering_30_percent_goal_met=candidate_wall<=baseline_wall*.7,
    timing_comparison="Same raw_000 to raw_1349 boundary; historical reference and current shared-GPU run, not an exclusive-GPU benchmark.",
    remaining_costs=sorted([(k,v["total_s"]) for k,v in timing["modules"].items()],key=lambda x:x[1],reverse=True),
    policy_success=False)
(OUT/"controlled_execution_acceptance.json").write_text(json.dumps(acceptance,indent=2))
if not passed:phase("complete_replay_failed",acceptance=acceptance);raise SystemExit(1)
phase("dp_single_episode",acceptance=acceptance)
rc=subprocess.run(["/usr/bin/python3","-B",str(HERE/"dp_pipeline.py"),"--variant","batched_io","--name","dp_episode_003"],cwd=ROOT).returncode
D=OUT/"dp_episode_003/eval/episode_01"
if (D/"cycle_timing.jsonl").exists():
    subprocess.run([str(PY),"-B",str(HERE/"summarize.py"),str(D)],check=True,cwd=ROOT)
phase("finished" if rc==0 else "dp_pipeline_error",acceptance=acceptance,dp_returncode=rc)
