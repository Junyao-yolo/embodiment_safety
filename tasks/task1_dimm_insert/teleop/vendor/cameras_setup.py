"""Session-layer cameras for franka_manual_safety_v002: dual wrist + external overview."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PACKAGE = Path("/mnt/data2/wjy_SERL/Embodied_Safety/outputs/T1_rl/franka_manual_safety_v002")


def _load_json(name: str) -> Dict[str, Any]:
    return json.loads((PACKAGE / name).read_text(encoding="utf-8"))


def _find_prim_by_name(stage, root_path: str, name: str):
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return None
    if root.GetName() == name:
        return root
    for prim in stage.Traverse():
        path = prim.GetPath().pathString
        if not path.startswith(root_path.rstrip("/") + "/") and path != root_path:
            continue
        if prim.GetName() == name:
            return prim
    return None


def _look_at_quat(eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0.0, 0.0, 1.0)):
    """USD Camera looks along local -Z; build orient quat (Gf.Quatf) for that convention."""
    from pxr import Gf

    e = Gf.Vec3d(float(eye[0]), float(eye[1]), float(eye[2]))
    t = Gf.Vec3d(float(target[0]), float(target[1]), float(target[2]))
    u = Gf.Vec3d(float(up[0]), float(up[1]), float(up[2]))
    forward = (t - e)
    if forward.GetLength() < 1e-9:
        forward = Gf.Vec3d(0.0, 0.0, -1.0)
    else:
        forward.Normalize()
    # Camera -Z should align with forward (from eye to target)
    z_axis = -forward
    x_axis = Gf.Cross(u, z_axis)
    if x_axis.GetLength() < 1e-9:
        # up parallel to look; pick alternate up
        u = Gf.Vec3d(0.0, 1.0, 0.0)
        x_axis = Gf.Cross(u, z_axis)
    x_axis.Normalize()
    y_axis = Gf.Cross(z_axis, x_axis)
    y_axis.Normalize()
    m = Gf.Matrix4d(
        x_axis[0], x_axis[1], x_axis[2], 0.0,
        y_axis[0], y_axis[1], y_axis[2], 0.0,
        z_axis[0], z_axis[1], z_axis[2], 0.0,
        0.0, 0.0, 0.0, 1.0,
    )
    return Gf.Quatf(m.ExtractRotationQuat())


def _ensure_camera(stage, path: str, translate, orient_quat, focal_length: float, h_aperture: float, clip: Tuple[float, float]):
    from pxr import UsdGeom, Sdf, Gf

    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        prim = stage.DefinePrim(path, "Camera")
    cam = UsdGeom.Camera(prim)
    # Clear conflicting ops then author translate+orient
    xform = UsdGeom.Xformable(prim)
    try:
        xform.ClearXformOpOrder()
    except Exception:
        pass
    t_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    t_op.Set(Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2])))
    o_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    o_op.Set(orient_quat)
    cam.CreateFocalLengthAttr(float(focal_length))
    cam.CreateHorizontalApertureAttr(float(h_aperture))
    cam.CreateClippingRangeAttr(Gf.Vec2f(float(clip[0]), float(clip[1])))
    # Hide camera gizmo clutter slightly — still selectable in stage
    try:
        img = cam.CreateProjectionAttr()
        img.Set(UsdGeom.Tokens.perspective)
    except Exception:
        pass
    return path


def apply_session_cameras(stage, safety: Optional[Dict[str, Any]] = None, calib: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create / refresh dual wrist + external overview cameras on the session layer."""
    from pxr import UsdGeom, Gf, Sdf

    if safety is None:
        safety = _load_json("safety_config.json")
    if calib is None:
        try:
            calib = _load_json("geometry_calib.json")
        except Exception:
            calib = {}

    cfg = dict(safety.get("cameras") or {})
    wrist_path = Path(__file__).resolve().parents[2] / "configs/wrist_cameras.json"
    wrist_cfg = json.loads(wrist_path.read_text(encoding="utf-8"))
    for key in ("wrist_left", "wrist_right"):
        cfg[key] = dict(cfg.get(key) or {}, **wrist_cfg[key])
    if not bool(cfg.get("enabled", True)):
        return {"ok": False, "skipped": True, "reason": "cameras.enabled=false"}

    stage.SetEditTarget(stage.GetSessionLayer())
    report: Dict[str, Any] = {"created": [], "errors": [], "paths": {},
                              "wrist_config": wrist_cfg, "wrist_config_path": str(wrist_path)}

    robot_path = (safety.get("paths") or {}).get("robot", "/World/Robot")
    hand = _find_prim_by_name(stage, robot_path, cfg.get("wrist_parent_name", "panda_hand"))
    if hand is None:
        report["errors"].append("missing panda_hand under " + robot_path)
        report["ok"] = False
        return report
    hand_path = hand.GetPath().pathString
    report["wrist_parent"] = hand_path

    # Grasp look point in hand local: slightly forward of palm between fingers.
    # Isaac Franka panda_hand: fingers open along ±Y, approach roughly +Z (flange→tips).
    grasp_local = cfg.get("wrist_look_local", [0.0, 0.0, 0.12])
    wl = cfg.get("wrist_left", {})
    wr = cfg.get("wrist_right", {})
    ext = cfg.get("external", {})

    # Wrist cameras parented under hand (follow EE)
    for side, side_cfg, default_eye in (
        ("left", wl, [0.048, -0.055, 0.0]),
        ("right", wr, [0.048, 0.055, 0.0]),
    ):
        name = side_cfg.get("name", f"WristCam_{'L' if side == 'left' else 'R'}")
        path = f"{hand_path}/{name}"
        eye = side_cfg.get("translate_local", default_eye)
        look = side_cfg.get("look_local", grasp_local)
        # up roughly hand +Y so stereo stays upright relative to gripper
        quat = _look_at_quat(eye, look, up=side_cfg.get("up_local", [0.0, 0.0, 1.0]))
        try:
            _ensure_camera(
                stage, path, eye, quat,
                focal_length=float(side_cfg.get("focal_length", 11.0)),
                h_aperture=float(side_cfg.get("horizontal_aperture", 24.0)),
                clip=(float(side_cfg.get("clip_near", 0.008)), float(side_cfg.get("clip_far", 1.5))),
            )
            report["created"].append(path)
            report["paths"][f"wrist_{side}"] = path
        except Exception as e:
            report["errors"].append(f"{name}: {e!r}")

    # External overview under /World/Cameras
    root = cfg.get("external_root", "/World/Cameras")
    if not stage.GetPrimAtPath(root):
        stage.DefinePrim(root, "Xform")
    ext_name = ext.get("name", "ExternalOverview")
    ext_path = f"{root}/{ext_name}"
    # Default eye: table-side 45° view of socket (~-0.01, 0.03, 0.77)
    eye_w = ext.get("translate_world", [0.32, 0.55, 1.20])
    look_w = ext.get("look_world", [-0.01, 0.03, 0.77])
    quat_w = _look_at_quat(eye_w, look_w, up=ext.get("up_world", [0.0, 0.0, 1.0]))
    try:
        _ensure_camera(
            stage, ext_path, eye_w, quat_w,
            focal_length=float(ext.get("focal_length", 14.0)),
            h_aperture=float(ext.get("horizontal_aperture", 20.955)),
            clip=(float(ext.get("clip_near", 0.05)), float(ext.get("clip_far", 8.0))),
        )
        report["created"].append(ext_path)
        report["paths"]["external"] = ext_path
    except Exception as e:
        report["errors"].append(f"external: {e!r}")

    report["ok"] = not report["errors"]
    report["note"] = (
        "Session-only cameras. Viewport: Cameras menu → WristCam_L / WristCam_R / ExternalOverview. "
        "Not baked into USDA unless you Save with session merged."
    )
    return report
