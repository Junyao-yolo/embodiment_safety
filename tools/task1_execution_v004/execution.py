"""Isolated policy execution candidate. No Isaac imports at module load."""
import copy
import os
import math

CONTROL = dict(ik_period_s=.05, ik_position_tolerance_m=.00005,
               ik_rotation_tolerance_rad=.0005, ik_max_evaluations=400)
ACTION = dict(max_goal_lead_m=.005)
class StartupRelease:
    """One-way startup-assist release; reset only with a fresh episode instance."""
    def __init__(self):
        self.released={"left":False,"right":False}
    def update(self,side,angle,press,band,contact_threshold):
        if side not in self.released:raise ValueError(side)
        vals=[press,band,contact_threshold]+([] if angle is None else [angle])
        if not all(math.isfinite(float(x)) for x in vals):
            raise ValueError("Non-finite latch measurement")
        if press>=contact_threshold or (angle is not None and abs(angle)>band):
            self.released[side]=True
        return self.released[side]

def speed_scaled_lead():
    """Optional free-space goal lead; collection and evaluation of one policy must use the same values."""
    horizon=os.environ.get("TASK1_GOAL_LEAD_HORIZON_S");cap=os.environ.get("TASK1_MAX_FREE_GOAL_LEAD_M")
    if (horizon is None)!=(cap is None):raise ValueError("Set both TASK1_GOAL_LEAD_HORIZON_S and TASK1_MAX_FREE_GOAL_LEAD_M")
    if horizon is None:return {}
    horizon,cap=float(horizon),float(cap)
    if not (0<horizon<=1 and ACTION["max_goal_lead_m"]<=cap<=.05):raise ValueError("Invalid speed-scaled goal lead")
    return dict(goal_lead_horizon_s=horizon,max_free_goal_lead_m=cap)

def control_overrides():
    """Optional IK rotation residual; collection and evaluation of one policy must use the same value."""
    raw=os.environ.get("TASK1_IK_ROTATION_TOLERANCE_RAD")
    if raw is None:return {}
    value=float(raw)
    if not (CONTROL["ik_rotation_tolerance_rad"]<=value<=.01):raise ValueError("Invalid IK rotation tolerance")
    return dict(ik_rotation_tolerance_rad=value)

def install(panel,emit):
    panel.controller.cfg["control"].update(CONTROL,**control_overrides())
    for key in ("goal_lead_horizon_s","max_free_goal_lead_m"):panel.cfg["action"].pop(key,None)
    panel.cfg["action"].update(ACTION,**speed_scaled_lead())
    frozen=copy.deepcopy(dict(control=panel.controller.cfg["control"],
                              action=panel.cfg["action"]))
    release=StartupRelease()
    original=panel.assistance.update_latch_soft_hold
    def update(stage,safety,metrics,calib):
        out=original(stage,safety,metrics,calib)
        for side,item in out["sides"].items():
            if not item.get("ok"):raise RuntimeError("Missing latch joint")
            before=release.released[side]
            released=release.update(side,item["angle_abs_deg"],item["press_N"],
                                    out["band_deg"],out["contact_N"])
            if released:
                joint=stage.GetPrimAtPath(safety["paths"]["socket"]+"/Joints/"+side.title()+"Hinge")
                attr=joint.GetAttribute("drive:angular:physics:stiffness")
                if attr.Get()!=0.:attr.Set(0.)
                item.update(stiffness=0.,mode="startup_assist_permanently_released")
            if released and not before:
                emit(dict(event="startup_assist_released",time=panel.t,side=side,
                          angle_abs_deg=item["angle_abs_deg"],press_N=item["press_N"]))
        return out
    panel.assistance.update_latch_soft_hold=update
    def check():
        current=dict(control=panel.controller.cfg["control"],action=panel.cfg["action"])
        if current!=frozen:raise RuntimeError("Uniform execution configuration changed")
    return frozen,check
