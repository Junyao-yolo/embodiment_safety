"""Coherent planar fixture resets applied before physics initialization."""

import math
import numpy as np

CABLE_ROOTS = (
    "/wire1",
    "/TE_L26620011_4_connector_v2",
    "/World/CableSensor",
    "/World/RootStrainRelief/Boot",
    "/World/PickupNest",
    "/World/TerminalMountBoltL",
    "/World/TerminalMountBoltR",
)
BOARD_ROOTS = (
    "/PCB_holder",
    "/PCB_4_Terminal",
    "/NutPlate_Coarse",
    "/World/TargetCableLanding",
)
MODES = ("direct", "left", "right", "high")
TARGET = np.array([-0.092, -0.071, 0.818])


def sample_reset(seed, mode=None):
    # Four adjacent collection seeds share a reset, with distinct route families.
    rng = np.random.default_rng(int(seed) // 4)

    def pose(xy, yaw, pivot):
        return dict(
            xy_m=rng.uniform(-xy, xy, 2).tolist(),
            yaw_deg=float(rng.uniform(-yaw, yaw)),
            pivot=list(pivot),
        )

    return dict(
        reset_seed=int(seed) // 4,
        route_mode=mode or MODES[int(seed) % 4],
        cable=pose(0.015, 6.0, [-0.045275535, 0.31645606, 0.0]),
        board=pose(0.010, 8.0, TARGET),
        tray=pose(0.020, 15.0, [-0.14, -0.22, 0.75]),
        screw=pose(0.00015, 180.0, [-0.16, -0.24, 0.7525214455]),
        robot_xyz_m=rng.uniform(-0.01, 0.01, 3).tolist(),
        robot_rpy_deg=rng.uniform([-2.0, -2.0, -5.0], [2.0, 2.0, 5.0]).tolist(),
        ring_grasp_offset_m=[
            float(rng.uniform(-0.0005, 0.0005)),
            float(rng.uniform(-0.001, 0.001)),
            0.0,
        ],
        screw_grasp_offset_m=[
            float(rng.uniform(-0.00015, 0.00015)),
            float(rng.uniform(-0.00015, 0.00015)),
            float(rng.uniform(-0.001, 0.001)),
        ],
    )


def rotation(spec):
    a = math.radians(spec["yaw_deg"])
    c = math.cos(a)
    s = math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def transform_point(point, spec):
    pivot = np.asarray(spec["pivot"])
    return (
        rotation(spec) @ (np.asarray(point) - pivot) + pivot + np.r_[spec["xy_m"], 0.0]
    )


def apply_reset(stage, spec):
    from pxr import UsdGeom, UsdPhysics, Gf

    def transform_roots(paths, pose):
        rot = rotation(pose)
        translation = transform_point(np.zeros(3), pose)
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), pose["yaw_deg"]))
        matrix.SetTranslateOnly(Gf.Vec3d(*translation))
        cache = UsdGeom.XformCache()
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not UsdGeom.Xformable(prim):
                raise ValueError("Missing transformable randomized component " + path)
            world = cache.GetLocalToWorldTransform(prim)
            parent = cache.GetLocalToWorldTransform(prim.GetParent())
            UsdGeom.Xformable(prim).MakeMatrixXform().Set(
                world * matrix * parent.GetInverse()
            )
        # Joint world endpoints are expressed in world coordinates, not their parent Xform.
        for prim in stage.Traverse():
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            bodies = [
                joint.GetBody0Rel().GetTargets(),
                joint.GetBody1Rel().GetTargets(),
            ]
            for side in (0, 1):
                other = bodies[1 - side]
                if (
                    not bodies[side]
                    and other
                    and any(
                        str(other[0]) == p or str(other[0]).startswith(p + "/")
                        for p in paths
                    )
                ):
                    pos = prim.GetAttribute("physics:localPos" + str(side))
                    quat = prim.GetAttribute("physics:localRot" + str(side))
                    pos.Set(Gf.Vec3f(*transform_point(pos.Get(), pose)))
                    angle = math.radians(pose["yaw_deg"]) / 2
                    quat.Set(
                        Gf.Quatf(math.cos(angle), Gf.Vec3f(0, 0, math.sin(angle)))
                        * quat.Get()
                    )

    transform_roots(CABLE_ROOTS, spec["cable"])
    transform_roots(BOARD_ROOTS, spec["board"])
    # Screw perturbation is relative to its nest; tray carries both afterwards.
    transform_roots(("/ScrewTray/Nest_1/Screw_1",), spec["screw"])
    transform_roots(("/ScrewTray",), spec["tray"])
    return transform_point(TARGET, spec["board"]), rotation(spec["board"])


def route_waypoint(start, target, spec, base_height=0.875):
    mid = (np.asarray(start) + target) / 2
    d = target[:2] - np.asarray(start)[:2]
    lateral = np.array([-d[1], d[0]]) / max(np.linalg.norm(d), 1e-9)
    mode = spec["route_mode"]
    mid[:2] += (
        lateral * {"direct": 0.0, "left": 0.020, "right": -0.020, "high": 0.0}[mode]
    )
    mid[2] = base_height + (0.025 if mode == "high" else 0.0)
    return mid


def capture_initial_transforms(stage):
    from pxr import UsdGeom, UsdPhysics

    paths = set(CABLE_ROOTS + BOARD_ROOTS + ("/ScrewTray",))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and not str(prim.GetPath()).startswith(
            "/World/Robot/"
        ):
            paths.add(str(prim.GetPath()))
    transforms = {
        p: UsdGeom.Xformable(stage.GetPrimAtPath(p)).GetLocalTransformation()
        for p in paths
    }
    anchors = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint) and not str(prim.GetPath()).startswith(
            "/World/Robot/"
        ):
            for side in (0, 1):
                if not prim.GetRelationship("physics:body" + str(side)).GetTargets():
                    for field in ("localPos", "localRot"):
                        attr = prim.GetAttribute("physics:" + field + str(side))
                        anchors[(str(prim.GetPath()), attr.GetName())] = attr.Get()
    return dict(transforms=transforms, anchors=anchors)


def restore_initial_transforms(stage, initial):
    from pxr import UsdGeom

    for path, matrix in initial["transforms"].items():
        UsdGeom.Xformable(stage.GetPrimAtPath(path)).MakeMatrixXform().Set(matrix)
    for (path, name), value in initial["anchors"].items():
        stage.GetPrimAtPath(path).GetAttribute(name).Set(value)
