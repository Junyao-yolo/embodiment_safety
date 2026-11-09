"""Read-only progress reconstruction; no policy action or reward changes."""
import sys,json,csv,math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
BASE=ROOT/"outputs/runs/task1/diffusion_policy/insertion_diagnosis_v001/recorded_full/eval/episode_01"
OUT=ROOT/"outputs/runs/task1/diffusion_policy/performance_retest_v004"
def ordered(p,prefix):return sorted(p.glob(prefix+"*.json"),key=lambda x:int(x.stem.split("_")[-1]))
ref=json.loads(ordered(BASE,"raw_")[-1].read_text())
target_R=Rotation.from_quat(ref["task_truth"]["poses"]["dimm"][3:])
target_local=np.array(ref["task_truth"]["metrics"]["dimm_position_in_socket_m"])
def analyze(path,name):
    raws=[json.loads(f.read_text()) for f in ordered(path,"raw_")]
    act={int(f.stem.split("_")[-1]):json.loads(f.read_text()) for f in ordered(path,"action_")}
    labels={int(f.stem.split("_")[-1]):json.loads(f.read_text()) for f in ordered(path,"stage_transition_")}
    peaks={}
    ep=list(path.glob("episodes/*/transitions.jsonl"))
    if ep:
        with ep[0].open() as f:
            for line in f:
                row=json.loads(line)
                peaks[row["index"]]=row.get("diagnostics",{}).get("contact_peaks",{})
    start=raws[0]["sim_time"];z0=raws[0]["task_truth"]["poses"]["dimm"][2]
    progress=[0,0,0,0];rows=[];previous_rel=None;events=[]
    for i,o in enumerate(raws):
        d=o["task_truth"]["poses"]["dimm"];m=o["task_truth"]["metrics"]
        ee=np.array(o["ee_position_m"]);ee_R=Rotation.from_quat(o["ee_quaternion_xyzw"])
        rel=ee_R.inv().apply(np.array(d[:3])-ee)
        label=labels.get(i-1,{})
        progress=label.get("progress_after",progress)
        for e in label.get("events",[]):events.append(dict(e,relative_time=e["time"]-start))
        evidence=label.get("evidence",{})
        a=act.get(i,{}).get("executed")
        dt=(raws[i+1]["sim_time"]-o["sim_time"]) if i+1<len(raws) else None
        delta=(np.array(raws[i+1]["ee_position_m"])-ee) if dt else None
        p=np.array(m["dimm_position_in_socket_m"])
        row=dict(index=i,time_s=o["sim_time"]-start,stage_progress=sum(progress),
            left_angle_deg=m["left_angle_deg"],right_angle_deg=m["right_angle_deg"],
            opening_valid=evidence.get("insertion_path_clear"),
            grasp_stable=evidence.get("bilateral_grasp_stable"),
            grasp_allowed=evidence.get("grasp_in_allowed_band"),
            rack_clear=evidence.get("rack_clear"),rise_mm=(d[2]-z0)*1000,
            slip_step_mm=float(np.linalg.norm(rel-previous_rel))*1000 if previous_rel is not None else None,
            target_distance_mm=float(np.linalg.norm(p-target_local))*1000,
            lateral_mm=float(np.linalg.norm(p[:2]-target_local[:2]))*1000,
            orientation_deg=float((Rotation.from_quat(d[3:])*target_R.inv()).magnitude()*180/np.pi),
            seated_z_error_mm=float(p[2]-target_local[2])*1000,
            left_locked=evidence.get("left_hook_engaged"),right_locked=evidence.get("right_hook_engaged"),
            retention_stable=evidence.get("retention_stable"),
            commanded_translation_mm_s=float(np.linalg.norm(a[:3]))*320 if a else None,
            actual_translation_mm_s=float(np.linalg.norm(delta))/dt*1000 if dt else None,
            controller_goal_error_mm=float(np.linalg.norm(np.array(o["controller_state"]["goal_position"])-ee))*1000,
            action=a,contact_peaks=peaks.get(i,{}),contact_metrics_present=bool(peaks.get(i)),
            gripper_width_mm=o["gripper_width_m"]*1000,
            commanded_rotation_deg_s=float(np.linalg.norm(a[3:6]))*90 if a else None)
        rows.append(row);previous_rel=rel
    fields=list(rows[0])
    with (OUT/(name+"_progress.csv")).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
    phases=[]
    for phase in sorted(set(x["stage_progress"] for x in rows)):
        subset=[x for x in rows if x["stage_progress"]==phase]
        fields=["commanded_translation_mm_s","actual_translation_mm_s","controller_goal_error_mm","slip_step_mm","target_distance_mm","lateral_mm","orientation_deg","seated_z_error_mm"]
        phases.append(dict(progress=phase,start_s=subset[0]["time_s"],end_s=subset[-1]["time_s"],duration_s=subset[-1]["time_s"]-subset[0]["time_s"],
            statistics={k:dict(p10=float(np.percentile([x[k] for x in subset if x[k] is not None],10)),median=float(np.median([x[k] for x in subset if x[k] is not None])),p95=float(np.percentile([x[k] for x in subset if x[k] is not None],95))) for k in fields if any(x[k] is not None for x in subset)}))
    tails=[]
    for duration in [5,10,30]:
        subset=[x for x in rows if x["time_s"]>=rows[-1]["time_s"]-duration]
        tails.append(dict(window_s=duration,stage_delta=subset[-1]["stage_progress"]-subset[0]["stage_progress"],
            target_distance_reduction_mm=subset[0]["target_distance_mm"]-subset[-1]["target_distance_mm"],
            lateral_reduction_mm=subset[0]["lateral_mm"]-subset[-1]["lateral_mm"],
            orientation_reduction_deg=subset[0]["orientation_deg"]-subset[-1]["orientation_deg"],
            rise_change_mm=subset[-1]["rise_mm"]-subset[0]["rise_mm"],
            command_median_mm_s=float(np.median([x["commanded_translation_mm_s"] for x in subset if x["commanded_translation_mm_s"] is not None])),
            actual_median_mm_s=float(np.median([x["actual_translation_mm_s"] for x in subset if x["actual_translation_mm_s"] is not None])),
            controller_goal_error_p95_mm=float(np.percentile([x["controller_goal_error_mm"] for x in subset],95)),
            force_peaks={k:max((float(x["contact_peaks"][k]) for x in subset if x["contact_peaks"].get(k) is not None),default=None) for k in ["left_press_N","right_press_N","finger_left_N","finger_right_N","dimm_environment_N"]}))
    result=dict(source=str(path),events=events,phases=phases,tail_windows=tails,
        references="Position and orientation errors are relative to the measured final successful replay pose; slip is DIMM motion relative to the hand. No new success gate or policy input.",
        maximum_rise_mm=max(x["rise_mm"] for x in rows))
    (OUT/(name+"_progress.json")).write_text(json.dumps(result,indent=2))
    return result
if __name__=="__main__":
    sources=[(BASE,"success_reference")]
    if len(sys.argv)>1:sources += [(Path(x),Path(x).parents[1].name) for x in sys.argv[1:]]
    summaries={name:analyze(p,name) for p,name in sources}
    base={x["progress"]:x for x in summaries["success_reference"]["phases"]}
    candidates=[]
    for name,s in summaries.items():
        if name=="success_reference":continue
        for phase in s["phases"]:
            b=base.get(phase["progress"])
            if b and phase["duration_s"]>1.5*b["duration_s"]:
                candidates.append(dict(run=name,progress=phase["progress"],duration_s=phase["duration_s"],reference_duration_s=b["duration_s"],reason="Phase duration exceeds 1.5 times the successful reference; review recorded progress, contacts and action tracking. Diagnostic marker only."))
    (OUT/"stagnation_candidates.json").write_text(json.dumps(candidates,indent=2))
    print(json.dumps(candidates,indent=2))
