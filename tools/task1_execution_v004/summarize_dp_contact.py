"""Join the first grasp-force violation to recorded actions and contacts."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
D=Path(sys.argv[1])
out=D.parents[1]/"first_grasp_violation.json"
transition_path=next(D.glob("episodes/*/transitions.jsonl"))
first=None
for line in transition_path.open():
    try:row=json.loads(line)
    except json.JSONDecodeError:break
    peaks=row["diagnostics"]["contact_peaks"]
    if any(peaks.get(k,0)>5 for k in ("finger_left_N","finger_right_N","dimm_environment_N")):
        first=row
        break
if first is None:
    print("No recorded grasp-force violation.")
    raise SystemExit(0)
i=first["index"]
o=json.loads((D/f"raw_{i:03d}.json").read_text())
n=json.loads((D/f"raw_{i+1:03d}.json").read_text())
a=json.loads((D/f"action_{i:03d}.json").read_text())
initial=json.loads((D/"raw_000.json").read_text())
contacts=[]
for line in next(D.glob("episodes/*/physics.jsonl")).open():
    try:p=json.loads(line)
    except json.JSONDecodeError:break
    if p["sim_time"]<o["sim_time"]-1e-7:continue
    if p["sim_time"]>n["sim_time"]+1e-7:break
    contacts.append(p)
base=ROOT/"outputs/runs/task1/diffusion_policy/insertion_diagnosis_v001/recorded_full/eval/episode_01"
choices=[]
for j in range(600,751):
    b=json.loads((base/f"raw_{j:03d}.json").read_text())
    choices.append((abs(b["ee_position_m"][2]-o["ee_position_m"][2]),j,b))
_,j,b=min(choices,key=lambda x:x[0])
dt=n["sim_time"]-o["sim_time"]
summary=dict(index=i,relative_time_s=o["sim_time"]-initial["sim_time"],
    gripper_width_mm=o["gripper_width_m"]*1000,
    command_translation_mm_s=(np.array(a["executed"][:3])*320).tolist(),
    actual_translation_mm_s=((np.array(n["ee_position_m"])-o["ee_position_m"])/dt*1000).tolist(),
    force_peaks_N=first["diagnostics"]["contact_peaks"],
    controller_error=first["diagnostics"]["controller_error"],
    goal_lead_limited_steps=sum(c["goal_lead_limited"] for c in first["diagnostics"]["controller_commands"]),
    matched_success_descent=dict(index=j,gripper_width_mm=b["gripper_width_m"]*1000,
        ee_position_m=b["ee_position_m"],position_difference_mm=((np.array(o["ee_position_m"])-b["ee_position_m"])*1000).tolist(),
        matching_scope="Closest EE height in recorded successful descending segment 600..750, not the same complete pose."),
    raw_observation_path=str(D/f"raw_{i:03d}.json"),action_path=str(D/f"action_{i:03d}.json"),
    transition=first,physics_records=contacts)
out.write_text(json.dumps(summary,indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ("transition","physics_records")},indent=2))
