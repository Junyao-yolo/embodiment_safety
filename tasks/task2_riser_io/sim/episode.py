"""Execute one Task2 rollout with measured physics and a remote learned policy."""

from pathlib import Path
from types import SimpleNamespace
import json
import sys
import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation
from tasks.task2_riser_io.sim.acceptance import (
    ACCEPTANCE_VERSION,
    task_assembly_hold,
    hold_passed,
    authored_assembly_geometry,
    ring_on_seat,
)
from tasks.task2_riser_io.sim.kinematics import pose_residual_jacobian


def run_episode(
    config, out, seed, app, scene_session, recorder, controller, progress_callback
):
    args = SimpleNamespace(
        seed=int(seed), record=True, physics_hz=int(config["physics_hz"])
    )
    R = Path(config["task_root"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runtime_config = config["runtime"]
    motion_config = config["motion"]
    physics_device = runtime_config["physics_device"]
    physics_hz = args.physics_hz
    reset_spec = None
    target = np.array([-0.092, -0.071, 0.818])
    board_rotation = np.eye(3)
    records = []
    frames = []
    events = []
    image_frames = []
    phase = "initializing"
    stepnum = 0
    report = {
        "scene": Path(config["scene"]).name,
        "full_task_completed": False,
        "safety_verdict": "UNASSESSED",
        "ring_geometry_unchanged": True,
        "task_body_pose_writes_after_reset": False,
        "seed": int(seed),
        "acceptance_version": ACCEPTANCE_VERSION,
        "collection_method": "learned_policy",
        "reset_configuration": {"randomize": True},
        "motion_config": motion_config,
        "runtime_config": runtime_config,
        "runtime_metadata": {"runner": "Task2ImageRunner"},
        "clamp_force_role": "diagnostic only; no minimum force magnitude",
    }
    performance_totals = {}
    wall_started = time.perf_counter()
    POLICY_CHECKPOINT = config.get("checkpoint_path")
    MAX_SECONDS = config["max_seconds"]
    GRASP_TIMEOUT = None
    EXECUTION_OPTIONS = {
        "n_action_steps": config["n_action_steps"],
        "fixed_noise": config["fixed_noise"],
        "seed": int(seed),
    }

    def assembly_hold(rows, hz, target):
        return task_assembly_hold(rows, hz, target, assembly_geometry)

    def measured(name):

        def decorate(fn):

            def call(*a, **kw):
                start = time.perf_counter()
                try:
                    return fn(*a, **kw)
                finally:
                    entry = performance_totals.setdefault(
                        name, {"seconds": 0.0, "calls": 0}
                    )
                    entry["seconds"] += time.perf_counter() - start
                    entry["calls"] += 1

            return call

        return decorate

    def performance_snapshot():
        return {
            "wall_seconds": time.perf_counter() - wall_started,
            "simulation_seconds": stepnum / args.physics_hz,
            "components": performance_totals,
        }

    try:
        sys.argv = [sys.argv[0]]
        import omni.usd, omni.physics.tensors as tensors, torch

        torch.set_num_threads(1)
        from pxr import UsdGeom, UsdPhysics, UsdLux, Gf
        from isaacsim.core.api import SimulationContext
        from isaacsim.robot_motion.motion_generation import (
            LulaKinematicsSolver,
            interface_config_loader,
        )

        if not scene_session:
            omni.usd.get_context().open_stage(str(config["scene"]))
            for _ in range(8):
                app.update()
        stage = omni.usd.get_context().get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        from tasks.task2_riser_io.sim.randomization import (
            capture_initial_transforms,
            restore_initial_transforms,
        )

        if scene_session:
            restore_initial_transforms(stage, scene_session["initial_transforms"])
            initial_transforms = scene_session["initial_transforms"]
        else:
            initial_transforms = capture_initial_transforms(stage)
        assembly_geometry = scene_session.get(
            "assembly_geometry"
        ) or authored_assembly_geometry(stage)
        report["assembly_geometry"] = assembly_geometry
        report["acceptance_policy"] = (
            "released functional assembly; numerical limits recorded only"
        )
        report["scene_reused"] = bool(scene_session)
        from tasks.task2_riser_io.sim.randomization import (
            sample_reset,
            apply_reset,
        )

        reset_spec = config.get("initial_conditions") or sample_reset(args.seed)
        (target, board_rotation) = apply_reset(stage, reset_spec)
        report["initial_conditions"] = reset_spec
        report["route_mode"] = reset_spec["route_mode"]
        report["target_position_m"] = target.tolist()
        UsdLux.DomeLight.Define(stage, "/World/DemoLight").CreateIntensityAttr(1500.0)
        from pxr import PhysxSchema

        for p in stage.Traverse():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                if str(p.GetPath()) == "/ScrewTray/Nest_1/Screw_1":
                    PhysxSchema.PhysxContactReportAPI.Apply(p).CreateThresholdAttr(0.0)
                elif p.HasAPI(PhysxSchema.PhysxContactReportAPI):
                    p.RemoveAPI(PhysxSchema.PhysxContactReportAPI)
                body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(p)
                body_api.CreateSolverPositionIterationCountAttr(
                    runtime_config["solver_position_iterations"]
                )
                body_api.CreateSolverVelocityIterationCountAttr(
                    runtime_config["solver_velocity_iterations"]
                )
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(p)
                articulation_api.CreateSolverPositionIterationCountAttr(
                    runtime_config["articulation_position_iterations"]
                )
                articulation_api.CreateSolverVelocityIterationCountAttr(
                    runtime_config["solver_velocity_iterations"]
                )
        for p in stage.Traverse():
            if p.IsA(UsdPhysics.Scene):
                physics = PhysxSchema.PhysxSceneAPI.Apply(p)
                physics.CreateEnableGPUDynamicsAttr(physics_device != "cpu")
                physics.CreateBroadphaseTypeAttr(
                    "MBP" if physics_device == "cpu" else "GPU"
                )
        report["physics_backend"] = (
            "PhysX CPU" if physics_device == "cpu" else "PhysX GPU"
        )
        report["contact_reporting"] = (
            "screw contact pairs used by assembly controller and clamp verification"
        )
        for path in ["/World/Robot", "/World/CableSensor"]:
            PhysxSchema.PhysxArticulationAPI.Apply(
                stage.GetPrimAtPath(path)
            ).CreateSolverPositionIterationCountAttr(
                runtime_config["articulation_position_iterations"]
            )
        report["solver_position_iterations"] = runtime_config[
            "solver_position_iterations"
        ]
        report["solver_velocity_iterations"] = runtime_config[
            "solver_velocity_iterations"
        ]
        physics_hz = args.physics_hz
        report["physics_rate_Hz"] = physics_hz
        report["measurement_rate_Hz"] = 120 if args.record else 60
        dt = 1 / physics_hz
        if scene_session and str(scene_session["sim"].device) == physics_device:
            sim = scene_session["sim"]
            sim.reset()
        else:
            if scene_session:
                SimulationContext.clear_instance()
            sim = SimulationContext(
                physics_dt=dt,
                rendering_dt=1 / 60,
                backend="torch",
                device=physics_device,
            )
            sim.get_physics_context().enable_fabric(runtime_config["fabric"])
            sim.initialize_physics()
            sim.play()
        report["fabric_enabled"] = runtime_config["fabric"]
        for _ in range(4):
            sim.step(render=True)
        view = tensors.create_simulation_view("torch")
        view.set_subspace_roots("/")

        class HostObservations:
            """Expose GPU physics state as NumPy observations to the controller."""

            def __init__(self, native):
                self.native = native

            def __getattr__(self, name):
                member = getattr(self.native, name)
                if not callable(member):
                    return member

                def call(*args, **kwargs):

                    def upload(x):
                        if isinstance(x, np.ndarray):
                            if x.dtype == np.uint32:
                                x = x.astype(np.int32)
                            return torch.as_tensor(
                                x,
                                device="cpu"
                                if name
                                in [
                                    "set_dof_max_forces",
                                    "set_dof_stiffnesses",
                                    "set_dof_dampings",
                                ]
                                else physics_device,
                            ).contiguous()
                        return x

                    def download(x):
                        if isinstance(x, torch.Tensor):
                            return x.detach().cpu().numpy()
                        if isinstance(x, (tuple, list)):
                            return type(x)((download(v) for v in x))
                        return x

                    value = member(*(upload(v) for v in args), **kwargs)
                    if name in ["get_contact_data", "get_friction_data"]:
                        (counts, starts) = value[-2:]
                        used = int(
                            torch.where(counts > 0, starts + counts, 0).max().item()
                        )
                        value = tuple((v[:used] for v in value[:-2])) + tuple(
                            value[-2:]
                        )
                    return download(value)

                return call

        robot = HostObservations(view.create_articulation_view("/World/Robot"))
        sensor = HostObservations(view.create_articulation_view("/World/CableSensor"))
        seg = HostObservations(view.create_rigid_body_view("/wire1/Segment_*"))
        rv = HostObservations(
            view.create_rigid_body_view("/wire1/_0347300_ring_terminal")
        )
        sv = HostObservations(view.create_rigid_body_view("/ScrewTray/Nest_1/Screw_1"))
        assert (
            robot.count == 1
            and sensor.count == 1
            and (seg.count == 24)
            and (rv.count == 1)
        )
        names = list(robot.shared_metatype.dof_names)
        ln = list(sensor.shared_metatype.link_names)
        si = ln.index("Segment_00")
        ci = next((i for (i, n) in enumerate(ln) if "TE_L26620011" in n))
        report["sensor_links"] = ln
        report["robot_dof_names"] = names
        report["action_representation"] = (
            "joint position targets at 60 Hz; radians for arm joints and meters for fingers"
        )
        report["clamp_measurement"] = (
            "normal force under the screw head, evaluated after releasing the gripper"
        )
        solver = LulaKinematicsSolver(
            **interface_config_loader.load_supported_lula_kinematics_solver_config(
                "Franka"
            )
        )
        ai = np.array([names.index(n) for n in solver.get_joint_names()])
        fi = np.array(
            [names.index("panda_finger_joint1"), names.index("panda_finger_joint2")]
        )
        ix = np.array([0], np.uint32)
        base = robot.get_root_transforms()[0]
        solver.set_robot_base_pose(base[:3], np.r_[base[6], base[3:6]])

        @measured("inverse_kinematics")
        def solve(p, rr, seed):
            (joints, ok) = (
                solver.compute_inverse_kinematics(
                    "panda_hand",
                    p,
                    np.r_[rr[3], rr[:3]],
                    warm_start=seed,
                    position_tolerance=2e-05,
                    orientation_tolerance=0.0002,
                )
                if phase == "initializing"
                else (None, False)
            )
            if ok and (
                phase == "initializing"
                or np.max(np.abs(np.asarray(joints) - seed)) < 0.03
            ):
                return np.asarray(joints)
            from scipy.optimize import least_squares

            goal = Rotation.from_quat(rr)

            def residual(qs):
                (pp, rot) = solver.compute_forward_kinematics("panda_hand", qs)
                return np.r_[
                    (np.asarray(pp) - p) * 10,
                    (Rotation.from_matrix(rot) * goal.inv()).as_rotvec(),
                ]

            lo = np.array(
                [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
            )
            hi = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
            if phase != "initializing":
                bounds = (
                    np.maximum(lo + 1e-07, seed - 0.03),
                    np.minimum(hi - 1e-07, seed + 0.03),
                )

                def local_residual(x):
                    return np.r_[residual(x), 0.01 * (x - seed)]

                def local_jacobian(x):
                    return pose_residual_jacobian(
                        solver._kinematics,
                        x,
                        solver._robot_rot,
                        solver._meters_per_unit,
                        residual(x)[3:],
                    )

                r = least_squares(
                    local_residual,
                    np.clip(seed, *bounds),
                    jac=local_jacobian,
                    bounds=bounds,
                    max_nfev=60,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                )
                err = residual(r.x)
                if (
                    np.linalg.norm(err[:3]) < 0.0002
                    and np.linalg.norm(err[3:]) < 0.0002
                ):
                    return r.x
                report["largest_ik_residual"] = max(
                    report.get("largest_ik_residual", 0.0), float(np.linalg.norm(err))
                )
                if np.isfinite(r.x).all():
                    return r.x
                raise RuntimeError("IK produced nonfinite command")
            best = None
            for start in [
                seed,
                [0, -0.8, 0, -2.6, 0, 1.8, 0.7854],
                [0.5, 0.2, -0.5, -2.0, 0, 2.3, 0],
                [-0.5, 0.2, 0.5, -2.0, 0, 2.3, 1.5],
            ]:
                r = least_squares(
                    residual,
                    np.clip(start, lo + 1e-06, hi - 1e-06),
                    bounds=(lo, hi),
                    max_nfev=160,
                    ftol=1e-10,
                    xtol=1e-10,
                    gtol=1e-10,
                )
                if best is None or np.linalg.norm(r.fun) < np.linalg.norm(best.fun):
                    best = r
                if (
                    np.linalg.norm(r.fun[:3]) < 0.005
                    and np.linalg.norm(r.fun[3:]) < 0.01
                ):
                    return r.x
            raise RuntimeError(
                "IK unreachable: " + str(p) + " residual " + str(best.fun)
            )

        Rdown = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
        q = robot.get_dof_positions()[0].copy()
        q[ai] = [0, -0.5, 0, -2.2, 0, 1.7, 0.7854]
        q[fi] = 0.04
        initial_hand = np.array([-0.285, 0.02, 1.0434])
        initial_rotation = Rdown
        if reset_spec is not None:
            initial_hand += reset_spec["robot_xyz_m"]
            initial_rotation = (
                Rotation.from_euler(
                    "xyz", reset_spec["robot_rpy_deg"], degrees=True
                ).as_matrix()
                @ Rdown
            )
        q[ai] = solve(
            initial_hand, Rotation.from_matrix(initial_rotation).as_quat(), q[ai]
        )
        robot.set_dof_positions(q[None].astype(np.float32), ix)
        robot.set_dof_velocities(np.zeros_like(q)[None].astype(np.float32), ix)
        robot.set_dof_position_targets(q[None].astype(np.float32), ix)
        forces = robot.get_dof_max_forces().copy()
        forces[0, fi] = 15.0
        robot.set_dof_max_forces(forces, ix)
        stiffness = robot.get_dof_stiffnesses().copy()
        stiffness[0, fi] = 400.0
        robot.set_dof_stiffnesses(stiffness, ix)
        damping = robot.get_dof_dampings().copy()
        damping[0, fi] = 15.0
        robot.set_dof_dampings(damping, ix)
        report["finger_control"] = {
            "stiffness_N_m": 400.0,
            "damping_Ns_m": 15.0,
            "effort_cap_N": 15.0,
        }
        jp0 = np.array(
            [
                stage.GetPrimAtPath(f"/wire1/Bend_{i:02d}")
                .GetAttribute("physics:localPos0")
                .Get()
                for i in range(23)
            ]
        )
        jp1 = np.array(
            [
                stage.GetPrimAtPath(f"/wire1/Bend_{i:02d}")
                .GetAttribute("physics:localPos1")
                .Get()
                for i in range(23)
            ]
        )
        screw_contacts = HostObservations(
            view.create_rigid_contact_view(
                "/ScrewTray/Nest_1/Screw_1",
                [
                    "/NutPlate_Coarse/Nut_1/IndependentNut",
                    "/wire1/_0347300_ring_terminal",
                    "/PCB_4_Terminal/SingleSeat",
                ],
                2048,
            )
        )
        step_physics = measured("physics_step")(sim.step)
        latched = False
        latest = {}

        @measured("state_contact_readback")
        def capture_states():
            normal_data = screw_contacts.native.get_contact_data(dt)
            normal_data = tuple(normal_data[:-2]) + tuple(
                (x.clone() for x in normal_data[-2:])
            )
            requests = {
                "segments": seg.native.get_transforms(),
                "ring": rv.native.get_transforms(),
                "screw": sv.native.get_transforms(),
                "ring_velocity": rv.native.get_velocities(),
                "joints": robot.native.get_dof_positions(),
                "joint_velocities": robot.native.get_dof_velocities(),
                "screw_contacts": normal_data,
                "screw_friction": screw_contacts.native.get_friction_data(dt),
            }
            chunks = []
            layout = []
            offset = 0
            for name, value in requests.items():
                values = value if isinstance(value, (tuple, list)) else [value]
                entries = []
                for v in values:
                    size = v.numel()
                    entries.append((offset, size, tuple(v.shape)))
                    offset += size
                    chunks.append(v.flatten().to(torch.float32))
                layout.append((name, entries, isinstance(value, (tuple, list))))
            packed = torch.cat(chunks).detach().cpu().numpy()
            latest.clear()
            for name, entries, islist in layout:
                values = [packed[o : o + n].reshape(shape) for (o, n, shape) in entries]
                latest[name] = values if islist else values[0]

        def screw_measure():
            sp = latest["screw"][0].copy()
            force = np.zeros(3)
            moment = np.zeros(3)
            (sf, cp, cn, dep, counts, starts) = latest["screw_contacts"]
            (ff, fp, fc, fs) = latest["screw_friction"]
            for nn, ss in zip(counts.ravel(), starts.ravel()):
                sl = slice(int(ss), int(ss + nn))
                if nn:
                    v = sf[sl] * cn[sl]
                    force += v.sum(axis=0)
                    moment += np.cross(cp[sl] - sp[:3], v).sum(axis=0)
            for nn, ss in zip(fc.ravel(), fs.ravel()):
                sl = slice(int(ss), int(ss + nn))
                if nn:
                    force += ff[sl].sum(axis=0)
                    moment += np.cross(fp[sl] - sp[:3], ff[sl]).sum(axis=0)
            axis = Rotation.from_quat(sp[3:]).apply([0, 0, 1])
            axial = float(np.dot(force, axis))
            return (
                sp,
                abs(axial),
                float(np.linalg.norm(force - axis * axial)),
                abs(float(np.dot(moment, axis))),
            )

        def handpose():
            return solver.compute_forward_kinematics(
                "panda_hand",
                latest["joints"][0, ai] if latest else robot.get_dof_positions()[0, ai],
            )

        def tick(n=1):
            nonlocal stepnum, latched
            for _ in range(n):
                step_physics(render=False)
                stepnum += 1
                capture_states()
                analysis_started = time.perf_counter()
                tr = latest["segments"]
                xyz = tr[:, :3]
                ring = latest["ring"][0]
                gaps = np.linalg.norm(
                    xyz[:-1]
                    + Rotation.from_quat(tr[:-1, 3:]).apply(jp0)
                    - xyz[1:]
                    - Rotation.from_quat(tr[1:, 3:]).apply(jp1),
                    axis=1,
                )
                (sp, sax, slat, storque) = screw_measure()
                sq = Rotation.from_quat(sp[3:])
                if not hasattr(tick, "last_screw_rotation"):
                    tick.last_screw_rotation = sq
                    tick.screw_angle = 0.0
                tick.screw_angle += float(
                    np.dot(
                        (sq * tick.last_screw_rotation.inv()).as_rotvec(),
                        sq.apply([0, 0, 1]),
                    )
                )
                tick.last_screw_rotation = sq
                (cf, cp, cn, cd, cc, cs) = latest["screw_contacts"]
                start = int(cs.ravel()[1])
                count = int(cc.ravel()[1])
                part = slice(start, start + count)
                local = sq.inv().apply(cp[part] - sp[:3])
                under_head = (np.linalg.norm(local[:, :2], axis=1) > 0.0055) & (
                    np.abs(local[:, 2] - 0.018) < 0.0005
                )
                head_force = (cf[part] * cn[part])[under_head].sum(axis=0)
                rec = {
                    "time_s": stepnum * dt,
                    "phase": phase,
                    "screw_axial_N": sax,
                    "screw_lateral_N": slat,
                    "screw_torque_Nm": storque,
                    "screw_tip_z": float(sp[2]),
                    "screw_x_m": float(sp[0]),
                    "screw_y_m": float(sp[1]),
                    "screw_tilt_deg": float(
                        np.degrees(np.arccos(np.clip(sq.apply([0, 0, 1])[2], -1, 1)))
                    ),
                    "screw_rotation_accumulated_deg": float(
                        np.degrees(tick.screw_angle)
                    ),
                    "clamp_N": float(abs(np.dot(head_force, sq.apply([0, 0, 1])))),
                    "head_contact_count": int(np.count_nonzero(under_head)),
                    "nut_contact_count": int(cc.ravel()[0]),
                    "head_contact_signed_sum_N": float(
                        cf[part].reshape(-1)[under_head].sum()
                    ),
                    "ring_x": float(ring[0]),
                    "ring_y": float(ring[1]),
                    "ring_z": float(ring[2]),
                    "ring_speed_m_s": float(
                        np.linalg.norm(latest["ring_velocity"][0, :3])
                    ),
                    "ring_tilt_deg": float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    Rotation.from_quat(ring[3:]).as_matrix()[2, 2],
                                    -1,
                                    1,
                                )
                            )
                        )
                    ),
                    "robot_joint_positions": latest["joints"][0].tolist(),
                    "robot_joint_velocities": latest["joint_velocities"][0].tolist(),
                    "robot_joint_targets": q.tolist(),
                    "joint_gap_m": float(gaps.max()),
                    "contact_separation_m": 0.0,
                    "action_source": "policy" if phase == "policy" else "hold",
                }
                recorder.measure(rec, latest, sensor, ci, si, tr, target)
                records.append(rec)
                active = phase not in ["settle", "calibration", "initializing"]
                if stepnum % (physics_hz // 60) == 0:
                    commanded = q[ai].copy()
                    if hasattr(tick, "previous_command"):
                        command_v = (commanded - tick.previous_command) * 60.0
                        if hasattr(tick, "previous_command_v") and active:
                            command_a = (command_v - tick.previous_command_v) * 60.0
                            if (
                                np.max(np.abs(command_v)) > 2.0
                                or np.max(np.abs(command_a)) > 12.0
                            ):
                                report["command_limit_events"] = (
                                    report.get("command_limit_events", 0) + 1
                                )
                        tick.previous_command_v = command_v
                    tick.previous_command = commanded
                if not np.isfinite(xyz).all():
                    raise RuntimeError("physics produced nonfinite body positions")
                entry = performance_totals.setdefault(
                    "state_analysis", {"seconds": 0.0, "calls": 0}
                )
                entry["seconds"] += time.perf_counter() - analysis_started
                entry["calls"] += 1
                if args.record and stepnum % (physics_hz // 10) == 0:
                    write_frame(rec, tr, ring)
                if stepnum % (physics_hz * 5) == 0:
                    print("PERFORMANCE", json.dumps(performance_snapshot()), flush=True)
                if stepnum % physics_hz == 0:
                    reset_request = Path(out / "reset_requested")
                    if reset_request.exists():
                        reason = reset_request.read_text().strip()
                        reset_request.unlink()
                        raise RuntimeError("Assembly attempt reset: " + reason)
                    print(
                        "STATE",
                        json.dumps(
                            {
                                k: rec[k]
                                for k in [
                                    "time_s",
                                    "phase",
                                    "ring_x",
                                    "ring_y",
                                    "ring_z",
                                    "clamp_N",
                                ]
                            }
                        ),
                        flush=True,
                    )
                    if progress_callback is not None:
                        progress_callback(
                            {
                                k: rec[k]
                                for k in [
                                    "time_s",
                                    "phase",
                                    "ring_x",
                                    "ring_y",
                                    "ring_z",
                                    "clamp_N",
                                ]
                            },
                            performance_snapshot(),
                        )

        def setup_render():
            import omni.replicator.core as rep

            if hasattr(setup_render, "anns"):
                return
            if scene_session.get("render"):
                for name, value in scene_session["render"].items():
                    setattr(setup_render, name, value)
                return
            external = UsdGeom.Camera.Define(stage, "/World/ExternalCamera")
            external.CreateFocalLengthAttr(23.0)
            external.CreateClippingRangeAttr(Gf.Vec2f(0.005, 10.0))
            mat = (
                Gf.Matrix4d()
                .SetLookAt(
                    Gf.Vec3d(0.13, -0.57, 1.35),
                    Gf.Vec3d(-0.15, 0.055, 0.86),
                    Gf.Vec3d(0, 0, 1),
                )
                .GetInverse()
            )
            UsdGeom.Xformable(external).AddTransformOp().Set(mat)
            paths = {
                "external": "/World/ExternalCamera",
                "wrist_left": "/World/Robot/panda_hand/WristCam_L",
                "wrist_right": "/World/Robot/panda_hand/WristCam_R",
            }
            setup_render.anns = {}
            setup_render.products = []
            setup_render.paths = paths
            manifest = {}
            for name, path in paths.items():
                p = stage.GetPrimAtPath(path)
                if not p or p.GetTypeName() != "Camera":
                    raise RuntimeError("missing camera " + path)
                product = rep.create.render_product(path, (128, 128))
                setup_render.products.append(product)
                ann = rep.AnnotatorRegistry.get_annotator("rgb")
                ann.attach([product])
                setup_render.anns[name] = ann
                c = UsdGeom.Camera(p)
                manifest[name] = {
                    "path": path,
                    "resolution": [128, 128],
                    "focal_length": c.GetFocalLengthAttr().Get(),
                    "horizontal_aperture": c.GetHorizontalApertureAttr().Get(),
                    "vertical_aperture": c.GetVerticalApertureAttr().Get(),
                    "local_matrix": np.asarray(
                        UsdGeom.Xformable(p).GetLocalTransformation()
                    ).tolist(),
                }
            manifest["external_configuration"] = (
                "fixed collection camera; task2 has no saved external camera calibration"
            )
            manifest["timestamp"] = (
                "simulation time; all cameras rendered without intervening physics steps"
            )
            setup_render.manifest = manifest
            for attempt in range(60):
                sim.render()
                if all(
                    (
                        a.get_data() is not None and a.get_data().size
                        for a in setup_render.anns.values()
                    )
                ):
                    break
            else:
                raise RuntimeError("three-camera render initialization failed")

        @measured("camera_render_readback")
        def write_frame(rec, tr, ring):
            setup_render()
            sim.render()
            images = {}
            for name, ann in setup_render.anns.items():
                rgb = ann.get_data()
                if rgb is None or not rgb.size:
                    raise RuntimeError("empty camera " + name)
                images[name] = np.ascontiguousarray(rgb[:, :, :3]).copy()
            recorder.frame(images, rec)
            image_frames.append(images)
            if len(image_frames) > 2:
                image_frames.pop(0)
            frames.append(
                {
                    "frame": len(frames),
                    "time_s": rec["time_s"],
                    "phase": phase,
                    "ring_pose": ring.tolist(),
                    "screw_pose": latest["screw"][0].tolist(),
                    "robot_joints": latest["joints"][0].tolist(),
                    "physics_step": stepnum,
                }
            )

        intended_grip_width = float(q[fi].sum())
        from tasks.task2_riser_io.sim.actions import (
            observation_state,
        )

        report.update(
            test_kind="learned_policy_closed_loop",
            checkpoint=POLICY_CHECKPOINT,
            demonstration_actions_used=False,
            policy_updates=0,
            command_velocity_cap_rad_s=1.8,
            command_acceleration_cap_rad_s2=10.0,
            max_simulation_seconds=MAX_SECONDS,
            initial_grasp_timeout_s=GRASP_TIMEOUT,
        )
        phase = "settle"
        tick(physics_hz * 2)
        initial_ring = latest["ring"][0].copy()
        initial_screw = latest["screw"][0].copy()
        policy_client = controller
        report["execution_options"] = EXECUTION_OPTIONS
        phase = "policy"
        from tasks.task2_riser_io.sim.actions import JointCommandLimits

        command_limiter = JointCommandLimits(q)
        limited = 0
        issued = 0
        proposal_jump = 0.0
        seat_count = 0
        report.update(
            wire_lifted=False,
            wire_seated=False,
            screw_lifted=False,
            ring_retained=False,
            screw_fastened=False,
            thread_engagement_verified=False,
            ring_pose_stable=False,
            head_seated=False,
            gripper_released=False,
            mean_clamp_N=None,
        )
        for observation in range(int(MAX_SECONDS * 10)):
            state = observation_state(
                latest["joints"][0], latest["joint_velocities"][0], q
            )
            proposal = policy_client.act(image_frames[-1], state)
            for proposed in proposal:
                proposal_jump = max(
                    proposal_jump, float(np.max(np.abs(proposed[:7] - q[:7])))
                )
                next_command = command_limiter.commit(proposed)
                limited += int(
                    np.max(np.abs(next_command[:7] - proposed[:7])) * 60 > 1e-05
                )
                issued += 1
                q[:] = next_command
                robot.set_dof_position_targets(q[None].astype(np.float32), ix)
                tick(physics_hz // 60)
            ring = latest["ring"][0]
            screw = latest["screw"][0]
            report["wire_lifted"] = bool(
                report["wire_lifted"] or ring[2] - initial_ring[2] > 0.045
            )
            report["screw_lifted"] = bool(
                report["screw_lifted"] or screw[2] - initial_screw[2] > 0.05
            )
            seated = ring_on_seat(records[-1], target, assembly_geometry)
            seat_count = seat_count + 1 if seated else 0
            report["wire_seated"] = bool(report["wire_seated"] or seat_count >= 10)
            report["command_blocks"] = observation + 1
            report["limited_command_fraction"] = limited / max(issued, 1)
            report["max_proposed_joint_jump_rad"] = proposal_jump
            report["max_ring_lift_m"] = max(
                report.get("max_ring_lift_m", 0.0), float(ring[2] - initial_ring[2])
            )
            report["simulation_seconds"] = stepnum / physics_hz
            window = records[-physics_hz * 3 :]
            if (
                observation % 10 == 0
                and len(window) == physics_hz * 3
                and (
                    float(latest["joints"][0, 7:9].sum())
                    > assembly_geometry["screw_head_diameter_m"]
                )
            ):
                hold_check = assembly_hold(window, physics_hz, target)
                report.update(hold_check)
                report["wire_seated"] = bool(
                    report["wire_seated"] or hold_check["ring_retained"]
                )
                (hp, hr) = handpose()
                tcp = np.asarray(hp) + np.asarray(hr) @ np.array([0, 0, 0.1034])
                hand_clear = np.linalg.norm(tcp - (screw[:3] + [0, 0, 0.033])) > 0.04
                fit_rows = [
                    r
                    for r in records
                    if 0.804 < r["screw_tip_z"] < 0.811
                    and np.linalg.norm(
                        np.array([r["screw_x_m"], r["screw_y_m"]]) - target[:2]
                    )
                    < 0.001
                    and (r["screw_tilt_deg"] < 5)
                ]
                thread_ok = False
                if len(fit_rows) > 20:
                    aa = np.array(
                        [r["screw_rotation_accumulated_deg"] / 360 for r in fit_rows]
                    )
                    zz = np.array([r["screw_tip_z"] for r in fit_rows])
                    if np.ptp(aa) > 1.0:
                        (pitch, intercept) = np.polyfit(aa, zz, 1)
                        residual = float(
                            np.quantile(np.abs(zz - (pitch * aa + intercept)), 0.95)
                        )
                        report["measured_thread_lead_m"] = float(abs(pitch))
                        report["thread_lead_residual_p95_m"] = residual
                        thread_ok = 0.0018 < abs(pitch) < 0.0022 and residual < 0.0003
                report.update(
                    thread_engagement_verified=bool(thread_ok),
                    hand_clear=bool(hand_clear),
                )
                if (
                    hold_passed(hold_check)
                    and hold_check["gripper_released"]
                    and thread_ok
                ):
                    break
            if (observation + 1) % 10 == 0:
                progress = {
                    k: report[k]
                    for k in [
                        "wire_lifted",
                        "wire_seated",
                        "screw_lifted",
                        "command_blocks",
                        "limited_command_fraction",
                        "max_proposed_joint_jump_rad",
                        "max_ring_lift_m",
                        "simulation_seconds",
                    ]
                }
                (out / "progress.json").write_text(json.dumps(progress, indent=2))
            if (
                GRASP_TIMEOUT is not None
                and (observation + 1) / 10 >= GRASP_TIMEOUT
                and (not report["wire_lifted"])
            ):
                report["failure_stage"] = "initial_grasp"
                report["stop_reason"] = "initial_grasp_timeout"
                break
        report["cable_connectivity_valid"] = all(
            (r["joint_gap_m"] < 0.0005 for r in records)
        )
        report["full_task_completed"] = bool(
            report["ring_retained"]
            and report["screw_fastened"]
            and report["head_seated"]
            and report.get("gripper_released")
            and report.get("thread_engagement_verified")
        )
        if not report["full_task_completed"]:
            report.setdefault("stop_reason", "episode_budget_exhausted")
        report["learned_policy_task_success_verified"] = report["full_task_completed"]
        active_records = [r for r in records if r["phase"] != "settle"]
        sample_hz = 120 if args.record else 60
        target_q = (
            np.array([r["robot_joint_targets"][:7] for r in active_records])[1::2]
            if args.record
            else np.array([r["robot_joint_targets"][:7] for r in active_records])
        )
        velocity = np.diff(target_q, axis=0) * 60
        acceleration = np.diff(velocity, axis=0) * 60
        actual_v = np.array([r["robot_joint_velocities"][:7] for r in active_records])
        rapid_reversals = False
        for j in range(7):
            indices = np.flatnonzero(np.abs(actual_v[:, j]) > 0.1)
            signs = np.sign(actual_v[indices, j])
            times = indices[1:][signs[1:] != signs[:-1]] / sample_hz
            if len(times) >= 3 and np.any(times[2:] - times[:-2] < 0.3):
                rapid_reversals = True
        report["smoothness"] = {
            "command_peak_velocity_rad_s": float(np.abs(velocity).max()),
            "command_peak_acceleration_rad_s2": float(np.abs(acceleration).max()),
            "rapid_joint_reversals": bool(rapid_reversals),
            "method": "60 Hz commanded joints; record three substantial velocity sign reversals within 0.3 s",
        }
        (acc_row, acc_joint) = np.unravel_index(
            np.abs(acceleration).argmax(), acceleration.shape
        )
        sampled = active_records[1::2] if args.record else active_records
        report["smoothness"]["peak_acceleration_phase"] = sampled[int(acc_row) + 2][
            "phase"
        ]
        report["smoothness"]["peak_acceleration_joint"] = int(acc_joint) + 1
        report["smoothness_pass"] = (
            report["smoothness"]["command_peak_velocity_rad_s"] <= 2.0
            and report["smoothness"]["command_peak_acceleration_rad_s2"] <= 12.0
            and (not rapid_reversals)
        )
        report["diagnostic_checks"] = {
            k: bool(report[k])
            for k in [
                "ring_pose_stable",
                "cable_connectivity_valid",
                "thread_engagement_verified",
                "smoothness_pass",
            ]
        }
        report["acceptance_checks"] = {
            k: bool(report[k])
            for k in [
                "wire_seated",
                "screw_fastened",
                "ring_retained",
                "head_seated",
                "gripper_released",
            ]
        }
        report["failed_checks"] = [
            k for (k, v) in report["acceptance_checks"].items() if not v
        ]
        report["accepted"] = report["full_task_completed"]
        if not report["accepted"]:
            raise RuntimeError("released assembly incomplete")
        report["status"] = "SUCCESS"
    except Exception as e:
        report["status"] = "FAILED"
        report["error"] = str(e)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        report["measurements_path"] = str(out / "measurements.npz")
        report["phase_simulation_seconds"] = {
            name: sum(row["phase"] == name for row in records) / physics_hz
            for name in dict.fromkeys(row["phase"] for row in records)
        }
        if records:
            report["max_joint_gap_m"] = max(row["joint_gap_m"] for row in records)
        report["performance"] = performance_snapshot()
        print("PERFORMANCE", json.dumps(report["performance"]), flush=True)
    render = {}
    if "setup_render" in locals() and hasattr(setup_render, "anns"):
        render = {
            name: getattr(setup_render, name)
            for name in ("anns", "products", "paths", "manifest")
        }
    if "sim" in locals():
        sim.stop()
        scene_session = dict(
            sim=sim,
            initial_transforms=initial_transforms,
            assembly_geometry=assembly_geometry,
            render=render,
        )
    if "view" in locals():
        view.invalidate()
    return (report, scene_session, stepnum / physics_hz)
