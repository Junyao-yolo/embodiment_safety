"""Compare aligned observations and diagnostics against the fixed success replay."""
import json,math,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
HERE=Path(__file__).resolve().parent
contract=json.loads((HERE/"contract.json").read_text())
def compare(base,candidate,end=None):
    def raw(p):return {int(f.stem.split("_")[-1]):f for f in p.glob("raw_*.json")}
    b=raw(base);c=raw(candidate);ids=sorted(c)
    if end is not None:ids=[i for i in ids if i<=end]
    peaks=dict(ee_position_m=0.,ee_rotation_rad=0.,joints_rad=0.,dimm_position_m=0.,latch_angle_deg=0.)
    age=0.;dt=0.
    for i in ids:
        x=json.loads(b[i].read_text());y=json.loads(c[i].read_text())
        dt=max(dt,abs(x["sim_time"]-y["sim_time"]))
        peaks["ee_position_m"]=max(peaks["ee_position_m"],math.dist(x["ee_position_m"],y["ee_position_m"]))
        peaks["ee_rotation_rad"]=max(peaks["ee_rotation_rad"],float((Rotation.from_quat(x["ee_quaternion_xyzw"])*Rotation.from_quat(y["ee_quaternion_xyzw"]).inv()).magnitude()))
        peaks["joints_rad"]=max(peaks["joints_rad"],max(abs(a-b) for a,b in zip(x["joint_positions"],y["joint_positions"])))
        peaks["dimm_position_m"]=max(peaks["dimm_position_m"],math.dist(x["task_truth"]["poses"]["dimm"][:3],y["task_truth"]["poses"]["dimm"][:3]))
        for side in ["left","right"]:
            k=side+"_angle_deg"
            peaks["latch_angle_deg"]=max(peaks["latch_angle_deg"],abs(x["task_truth"]["metrics"][k]-y["task_truth"]["metrics"][k]))
        for k in ["external","wrist_left","wrist_right"]:
            age=max(age,y["sim_time"]-y["images"][k]["sim_time"])
            if not y["images"][k]["quality"]["valid"]:raise ValueError("Invalid camera")
    tolerances=contract["comparison_tolerances"]
    return dict(observations=len(ids),max_errors=peaks,max_image_age_s=age,max_timestamp_difference_s=dt,
        passed=bool(ids) and all(peaks[k]<=v for k,v in tolerances.items()) and age<=.1 and dt<=1e-6)
if __name__=="__main__":
    result=compare(Path(sys.argv[1]),Path(sys.argv[2]))
    print(json.dumps(result,indent=2))
    if len(sys.argv)>3:Path(sys.argv[3]).write_text(json.dumps(result,indent=2))
    raise SystemExit(0 if result["passed"] else 1)
