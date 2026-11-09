"""7D world velocity command. Shared by teleop and future env.step."""
import numpy as np
from scipy.spatial.transform import Rotation

def keyboard_action(keys,cfg):
    a=np.zeros(7)
    rotate=bool(keys & {"LEFT_SHIFT","RIGHT_SHIFT"})
    for pos,neg,i in [("UP","DOWN",0),("LEFT","RIGHT",1),("PAGE_UP","PAGE_DOWN",2)]:
        a[i+(3 if rotate else 0)]=float(pos in keys)-float(neg in keys)
    a[6]=float("LEFT_BRACKET" in keys)-float("RIGHT_BRACKET" in keys)
    if keys & {"LEFT_CONTROL","RIGHT_CONTROL"}:
        a[:3]*=cfg["precision_translation_m_s"]/cfg["translation_m_s"]
        a[3:6]*=cfg["precision_rotation_rad_s"]/cfg["rotation_rad_s"]
    return a

def apply_velocity(controller,action,dt,cfg):
    a=np.asarray(action,dtype=float)
    if a.shape!=(7,) or not np.isfinite(a).all() or np.max(np.abs(a))>1.00001:
        raise ValueError("Bad action")
    p,R=controller.goal
    goal=p+a[:3]*cfg["translation_m_s"]*dt
    actual,_=controller.pose()
    lead=goal-actual;distance=np.linalg.norm(lead)
    max_lead=cfg["max_goal_lead_m"]
    if cfg.get("goal_lead_horizon_s"):
        # Slow (contact-scale) commands keep the base lead; only fast free-space commands get more.
        speed=np.linalg.norm(a[:3])*cfg["translation_m_s"]
        max_lead=float(np.clip(speed*cfg["goal_lead_horizon_s"],max_lead,cfg["max_free_goal_lead_m"]))
    limited=distance>max_lead
    if limited:goal=actual+lead*(max_lead/distance)
    controller.goal=(goal,Rotation.from_rotvec(a[3:6]*cfg["rotation_rad_s"]*dt)*R)
    controller.width_goal=float(np.clip(controller.width_goal+a[6]*cfg["gripper_m_s"]*dt,0,.08))
    return {"goal_lead_limited":bool(limited),"target_position_m":goal.tolist(),
            "target_quaternion_xyzw":controller.goal[1].as_quat().tolist(),
            "target_gripper_width_m":controller.width_goal}

def apply_soft_xy_attract(controller, stage, calib, assist_cfg, dt, metrics):
    """Gently pull EE goal toward socket-entry XY when DIMM tip is near the mouth.

    This is teleop ASSIST only: it does not seat the DIMM, does not flip latches,
    and must never be treated as mechanical / SERL success.
    """
    cfg = assist_cfg or {}
    if not cfg.get("enabled", False):
        return {"active": False, "reason": "disabled"}
    local = None if metrics is None else metrics.get("dimm_position_in_socket_m")
    if local is None:
        return {"active": False, "reason": "no_pose"}
    entry = calib.get("socket_local", {}).get("entry_point", [0.0, 0.0, 0.0068])
    ex, ey, ez = float(entry[0]), float(entry[1]), float(entry[2])
    lx, ly, lz = float(local[0]), float(local[1]), float(local[2])
    dxy = float(((lx - ex) ** 2 + (ly - ey) ** 2) ** 0.5)
    dz = float(lz - ez)
    radius = float(cfg.get("xy_capture_radius_m", 0.012))
    z_above = float(cfg.get("z_window_above_entry_m", 0.025))
    z_below = float(cfg.get("z_window_below_entry_m", 0.008))
    if dxy > radius or dz > z_above or dz < -z_below:
        return {"active": False, "align_xy_m": dxy, "dz_from_entry_m": dz, "reason": "out_of_window"}
    from pxr import UsdGeom, Gf
    socket = stage.GetPrimAtPath("/World/MatchedSocket/Asset")
    if not socket:
        return {"active": False, "align_xy_m": dxy, "reason": "no_socket"}
    sm = UsdGeom.XformCache().GetLocalToWorldTransform(socket)
    err_world = sm.TransformDir(Gf.Vec3d(ex - lx, ey - ly, 0.0))
    err_w = np.array([float(err_world[0]), float(err_world[1]), float(err_world[2])], dtype=float)
    gain = float(cfg.get("xy_attract_gain", 0.45))
    max_v = float(cfg.get("max_attract_m_s", 0.08))
    v = err_w * gain
    speed = float(np.linalg.norm(v))
    if speed > max_v and speed > 1e-12:
        v = v * (max_v / speed)
        speed = max_v
    p, R = controller.goal
    controller.goal = (p + v * float(dt), R)
    return {
        "active": True,
        "assist": "soft_xy_near_entrance",
        "align_xy_m": dxy,
        "dz_from_entry_m": dz,
        "attract_m_s": speed,
        "note": "assist_only_not_success",
    }
