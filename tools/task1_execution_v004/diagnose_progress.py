"""Join stalled-phase observations, predicted actions, execution and contacts."""
import json,csv,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
OUT=ROOT/"outputs/runs/task1/diffusion_policy/performance_retest_v004"
base=json.loads((OUT/"success_reference_progress.json").read_text())
durations={p["progress"]:p["duration_s"] for p in base["phases"]}
cases=[]
for arg in sys.argv[1:]:
    path=Path(arg);name=path.parents[1].name
    s=json.loads((OUT/(name+"_progress.json")).read_text())
    with (OUT/(name+"_progress.csv")).open() as f:rows=list(csv.DictReader(f))
    selected={}
    for phase in s["phases"]:
        reference=durations.get(phase["progress"])
        if reference is None or phase["duration_s"]<=1.5*reference:continue
        start=phase["start_s"]+1.5*reference
        choices=[x for x in rows if int(x["stage_progress"])==phase["progress"] and float(x["time_s"])>=start and x["actual_translation_mm_s"]]
        if not choices:continue
        for x in [choices[0],choices[len(choices)//2],choices[-1]]:
            selected[int(x["index"])]=(phase,x)
    ep=next(path.glob("episodes/*/transitions.jsonl"))
    contacts={}
    if selected:
        with ep.open() as f:
            for line in f:
                row=json.loads(line)
                if row["index"] in selected:contacts[row["index"]]=row["diagnostics"]
    for i,(phase,metrics) in sorted(selected.items()):
        o=json.loads((path/f"raw_{i:03d}.json").read_text())
        nxt=json.loads((path/f"raw_{i+1:03d}.json").read_text())
        a=json.loads((path/f"action_{i:03d}.json").read_text())
        d=contacts[i];commands=d["controller_commands"]
        dr=(Rotation.from_quat(nxt["ee_quaternion_xyzw"])*Rotation.from_quat(o["ee_quaternion_xyzw"]).inv()).as_rotvec()
        item=dict(run=name,index=i,time_s=float(metrics["time_s"]),stage_progress=phase["progress"],
            phase_elapsed_s=float(metrics["time_s"])-phase["start_s"],reference_phase_duration_s=durations[phase["progress"]],
            raw_observation=str(path/f"raw_{i:03d}.json"),policy_input_files=[str(p) for p in path.glob(f"*_{i:03d}.npz")],
            observation=dict(ee_position_m=o["ee_position_m"],ee_quaternion_xyzw=o["ee_quaternion_xyzw"],
                joint_positions=o["joint_positions"],gripper_width_m=o["gripper_width_m"],
                dimm_pose=o["task_truth"]["poses"]["dimm"],socket_metrics=o["task_truth"]["metrics"],
                controller_goal=o["controller_state"]),
            predicted_action=a["proposed"],executed_action=a["executed"],
            actual_ee_translation_m=(np.array(nxt["ee_position_m"])-o["ee_position_m"]).tolist(),
            actual_ee_rotation_vector_rad=dr.tolist(),
            actual_dimm_translation_m=(np.array(nxt["task_truth"]["poses"]["dimm"][:3])-o["task_truth"]["poses"]["dimm"][:3]).tolist(),
            contact_peaks=d.get("contact_peaks"),controller_error=d.get("controller_error"),
            goal_lead_limited_physics_steps=sum(c["goal_lead_limited"] for c in commands),
            physics_steps=len(commands),progress_metrics=metrics,
            interpretation="A diagnostic candidate based on phase duration and continuous progress; no action rescaling or automatic termination.")
        cases.append(item)
(OUT/"stagnation_evidence.json").write_text(json.dumps(cases,indent=2))
print("Evidence snapshots",len(cases))
