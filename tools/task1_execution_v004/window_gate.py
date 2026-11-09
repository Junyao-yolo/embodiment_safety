"""Compare a completed prefix to the fixed success replay before continuing."""
import json
from pathlib import Path
from compare import compare
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
REFERENCE=ROOT/"outputs/runs/task1/diffusion_policy/insertion_diagnosis_v001/recorded_full/eval/episode_01"
def check(out,end):
    result=compare(REFERENCE,out,end)
    warnings=json.loads((out/"safety_warning_summary.json").read_text())
    result["warning_frames"]=warnings["violating_physics_frames"]
    result["evidence_warning_intervals"]=warnings.get("evidence_warning_intervals",0)
    def events(p):
        got=[]
        for i in range(end):
            f=p/f"stage_transition_{i:03d}.json"
            if f.exists():
                for e in json.loads(f.read_text()).get("events",[]):got.append(e["stage"])
        return got
    result["reference_events"]=events(REFERENCE);result["candidate_events"]=events(out)
    result["passed"]=result["passed"] and result["warning_frames"]==0 and result["evidence_warning_intervals"]==0 and result["reference_events"]==result["candidate_events"]
    (out/f"window_gate_{end:04d}.json").write_text(json.dumps(result,indent=2))
    return result
