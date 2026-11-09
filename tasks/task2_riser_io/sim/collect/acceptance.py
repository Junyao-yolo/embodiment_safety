"""Released assembly checks shared by collection and policy evaluation."""
import numpy as np

ACCEPTANCE_VERSION = "task2_released_assembly_v004"

def assembly_hold(rows, physics_hz, target=(-.092,-.071,.818)):
    """Require a three-second stable assembly and screw-head contact evidence."""
    target=np.asarray(target)
    n = int(3 * physics_hz)
    rows = list(rows[-n:])
    result = dict(screw_fastened=False, ring_retained=False, ring_pose_stable=False,
                  head_seated=False, mean_clamp_N=None)
    if len(rows) < n:
        return result
    xyz = np.asarray([[r["ring_x"], r["ring_y"], r["ring_z"]] for r in rows])
    xy_error = np.linalg.norm(xyz[:, :2] - target[:2], axis=1)
    z_error = np.abs(xyz[:, 2] - target[2])
    tilt = np.asarray([r["ring_tilt_deg"] for r in rows])
    span = np.ptp(xyz, axis=0)
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=1) * physics_hz
    stable = bool(np.max(speed) < .01 and np.linalg.norm(span) < .0002)
    ring_good = bool(np.all(xy_error < .0015) and np.all(z_error < .0007)
                     and np.all(tilt < 5))
    screw_good = all(abs(r["screw_tip_z"] - (target[2]-.015)) < .0007
                     and np.hypot(r["screw_x_m"] - target[0], r["screw_y_m"] - target[1]) < .001
                     and r["screw_tilt_deg"] < 5 for r in rows)
    # Use contact points reported on the underside of the screw head.
    head_seated = any(r["head_contact_count"] > 0 for r in rows[-int(physics_hz):])
    result.update(
        screw_fastened=bool(screw_good), ring_retained=bool(ring_good and stable),
        ring_pose_stable=stable, head_seated=bool(head_seated),
        head_contact_samples=sum(r["head_contact_count"] > 0 for r in rows[-int(physics_hz):]),
        head_contact_signed_force_range_N=[float(min(r["head_contact_signed_sum_N"] for r in rows[-int(physics_hz):])), float(max(r["head_contact_signed_sum_N"] for r in rows[-int(physics_hz):]))],
        mean_clamp_N=float(np.mean([r["clamp_N"] for r in rows])),
        ring_hold_extrema=dict(max_xy_error_m=float(np.max(xy_error)),
                              max_z_error_m=float(np.max(z_error)),
                              max_tilt_deg=float(np.max(tilt)),
                              max_native_speed_m_s=float(max(r["ring_speed_m_s"] for r in rows)),
                              position_span_m=span.tolist(),
                              max_pose_speed_m_s=float(np.max(speed))))
    return result

def hold_passed(result):
    return all(result[k] for k in ("screw_fastened", "ring_retained", "head_seated"))


def authored_assembly_geometry(stage):
    """Read dimensions of the seating fixture from the unrandomized USD stage."""
    from pxr import UsdGeom
    cache=UsdGeom.BBoxCache(0,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render,UsdGeom.Tokens.proxy])
    def bounds(path):
        b=cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
        return np.asarray(b.GetMin()),np.asarray(b.GetMax())
    lo,hi=bounds("/PCB_4_Terminal/SingleSeat")
    ring_lo,ring_hi=bounds("/wire1/_0347300_ring_terminal")
    nut_lo,nut_hi=bounds("/NutPlate_Coarse/Nut_1/IndependentNut")
    screw_lo,screw_hi=bounds("/ScrewTray/Nest_1/Screw_1")
    return dict(seat_radius_m=float(min((hi-lo)[:2])/2),
                ring_radius_m=float((ring_hi-ring_lo)[0]/2),
                seat_z_m=[float(lo[2]),float(hi[2])],
                nut_z_m=[float(nut_lo[2]),float(nut_hi[2])],
                screw_head_diameter_m=float(max((screw_hi-screw_lo)[:2])),
                screw_under_head_z_m=.018,
                source="USD fixture bounds and authored screw underside at local z=0.018 m")

def ring_on_seat(row,target,geometry):
    # Ring centre must fit inside the seat's physical footprint.
    clearance=geometry["seat_radius_m"]-geometry["ring_radius_m"]
    return bool(np.hypot(row["ring_x"]-target[0],row["ring_y"]-target[1])<=clearance
                and geometry["seat_z_m"][0]<=row["ring_z"]<=geometry["seat_z_m"][1])

def task_assembly_hold(rows,physics_hz,target,geometry):
    """Classify completed released assembly; numerical quality remains diagnostic."""
    rows=list(rows[-int(3*physics_hz):])
    diagnostics=assembly_hold(rows,physics_hz,target)
    result=dict(diagnostics,numerical_quality=diagnostics.copy())
    if len(rows)<3*physics_hz:return result
    last=rows[-1]
    ring_good=ring_on_seat(last,target,geometry)
    head_seated=any(r["head_contact_count"]>0 for r in rows[-int(physics_hz):])
    tip=last["screw_tip_z"]
    nut_contact=any(r.get("nut_contact_count",0)>0 for r in rows[-int(physics_hz):])
    screw_in_nut=(tip<geometry["nut_z_m"][1]
                  and tip+geometry["screw_under_head_z_m"]>geometry["nut_z_m"][0])
    # Contact and released retention verify assembly; force/pitch/velocity peaks do not reject it.
    screw_above_ring=np.hypot(last["screw_x_m"]-last["ring_x"],last["screw_y_m"]-last["ring_y"])<geometry["ring_radius_m"]
    released=sum(last["robot_joint_positions"][7:9])>geometry["screw_head_diameter_m"]
    result.update(ring_retained=ring_good,screw_fastened=bool(screw_in_nut and screw_above_ring and head_seated and nut_contact),
                  head_seated=bool(head_seated),nut_contact=bool(nut_contact),gripper_released=bool(released))
    return result
