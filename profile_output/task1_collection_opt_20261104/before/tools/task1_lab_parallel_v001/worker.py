"""Persistent Task1 collection worker: one Kit process, clean stage reopen between episodes.

Recording, reward labels, safety stop and validation follow tasks/task1_dimm_insert/env/collect_actor.py.
Reset reopens collection.usda (fresh session layer and physics state), exactly like the panel RESET.
Started by tools/task1_lab_parallel_v001/collect.py; exit code 3 asks the supervisor for a fresh process.
"""
import argparse, hashlib, inspect, json, os, sys, textwrap, time, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"tools/task1_execution_v004")]
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--output", required=True)
ap.add_argument("--worker", type=int, required=True)
ap.add_argument("--successes", type=int, required=True)
ap.add_argument("--max-attempts", type=int, required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--episodes-per-process", type=int, default=20)
ap.add_argument("--wall-timeout-s", type=float, default=1800)
ap.add_argument("--gui", action="store_true")
ARGS = ap.parse_args()
OUT = Path(ARGS.output).resolve()
assert OUT.is_relative_to(ROOT/"outputs")
RECYCLE = 3
_PROFILE = json.loads((ROOT/"tools/task1_lab_parallel_v001/profile.json").read_text())
PROFILE_ACTION = _PROFILE["action_interface"]
os.environ.update(TASK1_GOAL_LEAD_HORIZON_S=str(PROFILE_ACTION["goal_lead_horizon_s"]),
                  TASK1_MAX_FREE_GOAL_LEAD_M=str(PROFILE_ACTION["max_free_goal_lead_m"]),
                  TASK1_IK_ROTATION_TOLERANCE_RAD=str(_PROFILE["control"]["ik_rotation_tolerance_rad"]))

from isaacsim import SimulationApp
app = SimulationApp({"headless": not ARGS.gui, "renderer": "RayTracedLighting", "width": 640, "height": 480, "sync_loads": True})

import numpy as np
import omni.usd
from pxr import UsdLux
from scipy.spatial.transform import Rotation
from tasks.task1_dimm_insert.env.episode import Episode as EpisodeCoordinator
from tasks.task1_dimm_insert.env.evidence import Evidence
from tasks.task1_dimm_insert.teleop import runtime
from tasks.task1_dimm_insert.env.data import Episode, clean
from tasks.task1_dimm_insert.env.boundary import Boundary
from tasks.task1_dimm_insert.env.trace import traced_apply_velocity
from tasks.task1_dimm_insert.env.transition import labels, validate_action, completed_partial
from execution import install
from tools.task1_lab_parallel_v001.expert import Task1Expert
from tools.task1_lab_parallel_v001.plan import PROFILE, sample
from tools.task1_lab_parallel_v001.collect import assess, accepted_count, release_claim
from tools.task1_diverse_collect_v001.setup import place

SCENE = ROOT/"tasks/task1_dimm_insert/scene/collection.usda"
PROTECTED = [SCENE, Path(runtime.__file__), ROOT/"tasks/task1_dimm_insert/env/actions.py"]
LIMITS = dict(left_press_N=3., right_press_N=3., dimm_environment_N=5., finger_left_N=5., finger_right_N=5.)
MAX_STEPS = int(PROFILE["run"]["max_steps"])
CLAIMS = OUT/"claims"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Run:
    """Per-episode mutable state shared by the recorder callbacks."""
    def __init__(self, dest, seed):
        self.out = dest; self.seed = seed
        self.failure = None; self.stopped = False; self.executed_steps = 0; self.calls = 0
        self.initial = None; self.final = None; self.terminal_label = None
        self.live_frames = []; self.evidence = None; self.evaluator = None; self.last_obs = None
        self.expert = None; self.panel = None; self.ik_log = None; self.check = None
        self.force_peaks = {}

    def write(self, name, value):
        p = self.out/name; q = p.with_suffix(p.suffix+".tmp")
        q.write_text(json.dumps(value, indent=2)); q.replace(p)


CURRENT = None

# One-time patches, identical to collect_actor: the 100 ms boundary drives the expert.
_callback = textwrap.dedent(inspect.getsource(runtime.CollectionPanel.physics_step))
_start = _callback.index("        if self.active:\n            if self.pending_obs is None:")
_end = _callback.index("        if not self.enabled:return", _start)
_callback = _callback[:_start]+"""        if self.active:
            self.native_boundary.step()
        else:self.action=np.zeros(7)
"""+_callback[_end:]
_scope = {}
exec(compile(_callback, "<native actor callback>", "exec"), runtime.__dict__, _scope)
runtime.CollectionPanel.physics_step = _scope["physics_step"]
_original_load = runtime.load


def _candidate_load(name, path):
    if Path(path).name == "franka_teleop.py":
        path = ROOT/"tools/task1_execution_v004/controller_batched_fd.py"
    return _original_load(name, path)


runtime.load = _candidate_load
_apply = traced_apply_velocity(runtime.apply_velocity)


def _counted(c, a, dt, cfg):
    x = _apply(c, a, dt, cfg)
    CURRENT.executed_steps += 1
    return x


runtime.apply_velocity = _counted
runtime.Episode = lambda ignored, meta: Episode(CURRENT.out/"episodes", meta)


def updates(n):
    for _ in range(n):
        app.update()


def capture(run, n, raw):
    p = run.panel
    if run.evaluator is None:
        from pxr import UsdGeom
        q = UsdGeom.XformCache().GetLocalToWorldTransform(p.stage.GetPrimAtPath("/World/MatchedSocket/Asset")).ExtractRotationQuat().GetNormalized()
        run.evaluator = EpisodeCoordinator([*q.GetImaginary(), q.GetReal()], timeout_s=MAX_STEPS*.1, safety_mode="warn")
        run.evaluator.reset(raw); run.evidence = Evidence(raw)
    elif n > 0:
        frames = list(run.live_frames)
        measured = run.evidence.update(raw, run.last_obs["sim_time"], frames)
        if not measured["valid"]:
            raise RuntimeError("Invalid live reward measurements")
        peaks = {k: max(f[1]["metrics"][k] for f in frames) for k in ("left_press_N", "right_press_N")}
        if run.failure:
            label = run.evaluator.evaluator.reward.update(raw["sim_time"], dict(valid=measured["valid"], sim_time=raw["sim_time"],
                                                          source="measured_short_terminal"), terminated=True, failure=run.failure)
            label.update(dt_s=raw["sim_time"]-run.last_obs["sim_time"], physics_steps=len(frames), injected=False)
        else:
            label = run.evaluator.step(raw, measured, peaks)
        run.terminal_label = label
        run.write(f"stage_transition_{n-1:03d}.json", dict(index=n-1, **label))
    run.live_frames.clear(); run.last_obs = clean(raw)
    return raw


def prepare_arm(run, plan):
    p = run.panel; c = p.controller
    pos, m = c.pose(); target = np.array(pos)+plan["initial_hand_delta_m"]
    c.arm(); p.enabled = True
    deadline = p.t+15.; stable = None
    while p.t < deadline:
        c.goal = (target, Rotation.from_matrix(m))
        app.update()
        if c.error:
            raise RuntimeError("Initial arm positioning: "+c.error)
        actual, _ = c.pose()
        if np.linalg.norm(actual-target) < .0005 and np.max(np.abs(c.view.get_dof_velocities()[0, c.arm_indices])) < .01:
            stable = p.t if stable is None else stable
            if p.t-stable > .3:
                break
        else:
            stable = None
    else:
        raise RuntimeError("Initial arm positioning did not settle")
    p.enabled = False; c.armed = False; c.sync_goal()
    return dict(target_m=target.tolist(), actual_m=np.asarray(c.pose()[0]).tolist())


def reset_check(run, metrics):
    """Every reopened scene must match the first recorded initial robot/latch state."""
    p = run.panel
    joints = np.asarray(p.controller.get_joint_positions(), dtype=float)
    latches = [float(metrics[k+"_angle_deg"]) for k in ("left", "right")]
    dimm_z = float(p.bodies["dimm"].get_transforms()[0, 2])
    state = dict(joint_positions=joints.tolist(), latch_angles_deg=latches, dimm_z_m=dimm_z)
    reference = OUT/"reset_reference.json"
    candidate = run.out/"reset_reference_candidate.json"
    candidate.write_text(json.dumps(state, indent=2))
    try:
        os.link(candidate, reference)
    except FileExistsError:
        pass
    REFERENCE = json.loads(reference.read_text())
    deviation = dict(joint_rad=float(np.max(np.abs(joints-np.asarray(REFERENCE["joint_positions"])))),
                     latch_deg=float(max(abs(a) for a in latches)),
                     dimm_z_m=abs(dimm_z-REFERENCE["dimm_z_m"]))
    ok = deviation["joint_rad"] < 1e-3 and deviation["latch_deg"] < 1. and deviation["dimm_z_m"] < 1e-3
    run.write("reset_check.json", dict(passed=ok, state=state, reference=REFERENCE, deviation=deviation))
    if not ok:
        raise RuntimeError("In-process reset differs from the first initial state: "+json.dumps(deviation))


def open_scene():
    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(SCENE)):
        raise RuntimeError("Could not open collection scene")
    updates(40)
    stage = ctx.get_stage(); stage.SetEditTarget(stage.GetSessionLayer())
    if not any(x.HasAPI(UsdLux.LightAPI) for x in stage.Traverse()):
        UsdLux.DomeLight.Define(stage, "/World/DiagnosticDome").CreateIntensityAttr(1000.)
    return stage


def build_panel(run):
    p = runtime.CollectionPanel(show_ui=False); run.panel = p
    p.cfg["control_hz"] = 10
    for key in ("translation_m_s", "rotation_rad_s", "gripper_m_s"):
        if abs(float(p.cfg["action"][key])-PROFILE["action_interface"][key]) > 1e-9:
            raise RuntimeError("Action interface differs from evaluation: "+key)
    p.cfg["teleop_assist"]["enabled"] = False
    p.cfg["max_image_age_s"] = .1
    p.cfg["demonstration_limits"] = dict(LIMITS)

    def partial(row):
        run.write("partial_transition.json", clean(row))

    def decide():
        run.check()
        n = p.native_boundary.total; raw = p.pending_obs
        capture(run, n, raw)
        if run.initial is None:
            run.initial = clean(raw)
        t = run.terminal_label
        if t and (t["terminated"] or t["truncated"]):
            run.final = clean(raw); run.stopped = True
            run.write("time_limit_event.json", dict(time=p.t, full_transitions=n))
            p.finish("success_request" if t["success"] else "interrupted", t.get("failure") or "episode_end")
            p.disable()
            run.write("policy_stop.json", dict(final_observation_index=n, reason="episode_end"))
            return np.zeros(7)
        a = validate_action(run.expert.action(p.cfg["action"]))
        if run.expert.error:
            run.failure = "expert:"+run.expert.error
            return np.zeros(7)
        run.calls += 1
        return a
    p.native_boundary = Boundary(p, decide, partial)
    finish = p.finish

    def finish_record(outcome, reason=""):
        if reason.startswith("controller_disarmed:"):
            run.failure = reason; run.stopped = True
            run.write("controller_diagnostics.json", clean(dict(time=p.t, error=p.controller.error,
                      ik=p.controller.ik_diagnostics, control=p.controller.cfg["control"])))
        if p.active:
            p.native_boundary.flush(outcome, reason)
            run.write("aligned_decisions.json", clean(p.native_boundary.decisions))
        return finish(outcome, reason)
    p.finish = finish_record
    measured = p.measure_contacts

    def measure(dt):
        contacts = measured(dt)
        if p.active:
            run.live_frames.append((run.executed_steps, dict(sim_time=p.t, contacts=contacts, metrics=p.contact_metrics,
                                                              controller_error=p.controller.error)))
        for k, limit in LIMITS.items():
            value = float(p.contact_metrics.get(k, 0) or 0)
            run.force_peaks[k] = max(run.force_peaks.get(k, 0.), value)
            if value > limit and run.expert is not None:
                run.expert.note("force_limit:"+k, value_N=run.force_peaks[k], limit_N=limit)
        return contacts
    p.measure_contacts = measure
    return p


def close_panel(run):
    p = run.panel
    if p is None:
        return
    try:
        if p.active:
            p.finish("interrupted", "worker_shutdown")
        p.disable(); p.close()
    except Exception:
        traceback.print_exc()
    for cam in getattr(p, "cameras", {}).values():
        try:
            cam.destroy()
        except Exception:
            traceback.print_exc()
    p.sv = None; p.bodies = {}
    if run.ik_log:
        run.ik_log.close(); run.ik_log = None
    p.timeline.stop()
    updates(5)
    run.panel = None


def episode(dest, index, seed):
    global CURRENT
    run = Run(dest, seed); CURRENT = run
    times = dict(reset_start=time.monotonic())
    protected_before = {str(x): sha(x) for x in PROTECTED}
    try:
        stage = open_scene()
        plan = sample(seed)
        run.write("randomization.json", dict(plan=plan, placement=place(stage, plan), source_scene_saved=False))
        p = build_panel(run)
        p.timeline.play(); updates(180)
        p.initialize(); updates(60)
        limits = p.controller.view.get_dof_max_forces().copy(); limits[0, p.controller.fingers] = 5.
        p.controller.view.set_dof_max_forces(limits, np.array([0], dtype=np.uint32))
        frozen, run.check = install(p, lambda row: run.write("unexpected_startup_release.json", row))
        if frozen["action"] != {**frozen["action"], **PROFILE["action_interface"]}:
            raise RuntimeError("Installed action interface differs from the collection profile")
        if abs(frozen["control"]["ik_rotation_tolerance_rad"]-PROFILE["control"]["ik_rotation_tolerance_rad"]) > 1e-12:
            raise RuntimeError("Installed IK rotation tolerance differs from the collection profile")
        run.ik_log = (dest/"ik_trace.jsonl").open("x", buffering=1)
        tick = p.controller.tick

        def timed_tick(dt):
            previous = p.controller.ik_diagnostics
            start = time.perf_counter(); tick(dt); elapsed = time.perf_counter()-start
            if p.controller.ik_diagnostics is not previous:
                run.ik_log.write(json.dumps(clean(dict(p.controller.ik_diagnostics, sim_time=p.t, tick_wall_s=elapsed)))+"\n")
        p.controller.tick = timed_tick
        run.write("execution_configuration.json", clean(dict(frozen, collection_profile=PROFILE,
                  controller="controller_batched_fd", persistent_worker=ARGS.worker, reset="reopen_collection_usda")))
        poses = {k: v.get_transforms()[0].copy() for k, v in p.bodies.items()}
        metrics = p.task_metrics(poses)
        if max(abs(metrics[k+"_angle_deg"]) for k in ("left", "right")) > 1:
            raise RuntimeError("Reset latches open")
        if np.linalg.norm(metrics["dimm_position_in_socket_m"][:2]) < .05:
            raise RuntimeError("Not free-space start")
        reset_check(run, metrics)
        run.write("reset.json", clean(dict(metrics=metrics, poses=poses, ready=True)))
        run.write("initial_arm.json", prepare_arm(run, plan))
        run.executed_steps = 0
        run.expert = Task1Expert(p, seed, 1.)
        p.start()
        append = p.active.append

        def append_expert(obs, action, nxt, diag):
            e = run.expert
            return append(obs, action, nxt, dict(diag, operator="feedback_expert", action_source="feedback_expert",
                          expert_stage=e.state, expert_clean_action=e.clean_action.tolist(),
                          perturbation=e.applied_perturbation, grasp_relative_slip_m=e.slip_m))
        p.active.append = append_expert
        p.active.meta.update(demonstration_type="feedback_expert", expert_seed=seed, expert_profile=run.expert.PROFILE,
                             transit_speed_limits_m_s=run.expert.TRANSIT, path_variation=1., diversity_plan=plan,
                             collection_profile_version=PROFILE["version"], bc_label_field="diagnostics.expert_clean_action",
                             policy_observations="three RGB views and proprioception",
                             teacher_observations="measured robot and task geometry/contact state",
                             safety_mode="warn", diagnostic_only=False, eligible_for_training=True)
        p.enable()
        run.write("safety_mode.json", dict(mode="warn", limits=LIMITS, source_scene_unchanged=True,
                                           network_updates_allowed=False, force_actuation_unchanged=True))
        times["recording_start"] = time.monotonic()
        last = time.monotonic()
        while app.is_running():
            app.update()
            now = time.monotonic()
            if now-last > 15:
                run.write("heartbeat.json", dict(wall_s=now-times["recording_start"], sim_time=p.t, commands=run.executed_steps,
                                                 policy_actions=run.calls, failure=run.failure, worker=ARGS.worker))
                run.write("expert_status.json", clean(dict(stage=run.expert.state, error=run.expert.error, stages=run.expert.trace)))
                last = now
            if p.sensor_errors:
                raise RuntimeError(str(p.sensor_errors))
            if run.stopped and not p.active:
                break
            if p.controller.error and run.calls > 0:
                raise RuntimeError("controller_fault:"+str(p.controller.error))
            if now-times["recording_start"] > ARGS.wall_timeout_s and not run.failure:
                run.failure = "wall_timeout"
                p.enabled = False; p.controller.stop(run.failure); run.stopped = True
        times["recording_end"] = time.monotonic()
        p.timeline.pause()
        saved = Path(p.last_saved_path)
        tail = None
        if run.failure and (dest/"partial_transition.json").exists():
            tail = completed_partial(json.loads((dest/"partial_transition.json").read_text()))
            run.final = tail["next_observations"]; capture(run, p.native_boundary.total+1, run.final)
        if run.failure:
            run.write("policy_stop.json", dict(final_observation_index=p.native_boundary.total+1, reason=run.failure))
        e = run.expert
        run.write("expert_result.json", clean(dict(stage=e.state, error=e.error, stages=e.trace, paths=e.plans,
                                                   seed=seed, perturbations=e.perturb.log,
                                                   safety_constraints=e.safety_constraints,
                                                   grasp_relative_slip_peak_m=e.slip_peak_m,
                                                   insert_force_peak_N=e.insert_force_peak_N,
                                                   force_peaks=run.force_peaks)))
        rows = [json.loads(s) for s in (saved/"transitions.jsonl").read_text().splitlines()]
        run.write("replay_labels.json", [dict(json.loads((dest/f"stage_transition_{i:03d}.json").read_text()), index=i)
                                         for i in range(len(rows))])
        if tail is not None:
            label = labels(tail["dt_s"], len(tail["controller_commands"]), terminated=True, reason=run.failure)
            run.write("terminal_transition.json", dict(tail, **label, diagnostic_only=False, training_eligible=True))
        unchanged = {k: sha(k) == v for k, v in protected_before.items()}
        run.write("source_integrity.json", dict(all_unchanged=all(unchanged.values()), files=unchanged, hashes=protected_before))
        final = run.final or run.last_obs
        t = run.terminal_label
        run.write("result.json", dict(runtime_wall_s=times["recording_end"]-times["recording_start"], sim_time_s=p.t,
            execution_completed=True, full_transitions=len(rows), physics_commands=run.executed_steps,
            observed_physics_steps=len(rows)*24+(tail["physics_steps"] if tail else 0), policy_actions=run.calls,
            network_updates=0, injected_stop=False, failure_reason=run.failure,
            ee_displacement_m=float(np.linalg.norm(np.array(final["ee_position_m"])-run.initial["ee_position_m"])),
            episode=str(saved), source_unchanged=all(unchanged.values()), acceptance_pending=True,
            full_task_success=bool(t and t["success"]), task_completion_observed=bool(t and t["success"]),
            safety_mode="warn", diagnostic_only=False, training_ready=False, native_app_update=True,
            persistent_worker=ARGS.worker))
    finally:
        close_panel(run)
        times["closed"] = time.monotonic()
        run.write("timing.json", dict(reset_and_setup_s=times.get("recording_start", times["closed"])-times["reset_start"],
                                      recording_wall_s=(times["recording_end"]-times["recording_start"]) if "recording_end" in times else None,
                                      finalize_s=times["closed"]-times.get("recording_end", times["closed"]),
                                      worker=ARGS.worker, index=index))
        CURRENT = None


def claim():
    """Atomically reserve the next attempt index while the target is not yet covered."""
    CLAIMS.mkdir(parents=True, exist_ok=True)
    pending = [c for c in CLAIMS.iterdir() if not (c/"done.json").exists()]
    if accepted_count(OUT) + len(pending) >= ARGS.successes:
        return None
    for i in range(1, ARGS.max_attempts+1):
        try:
            (CLAIMS/f"{i:03d}").mkdir()
        except FileExistsError:
            continue
        (CLAIMS/f"{i:03d}"/"owner.json").write_text(json.dumps(dict(worker=ARGS.worker, pid=os.getpid())))
        return i
    return None


def main():
    episodes = 0; errors = 0
    while episodes < ARGS.episodes_per_process:
        index = claim()
        if index is None:
            return 0
        seed = ARGS.seed+index-1
        dest = OUT/f"attempt_{index:03d}"
        dest.mkdir(parents=True, exist_ok=False)
        started = time.monotonic(); row = dict(index=index, seed=seed, worker=ARGS.worker)
        status = "error"
        try:
            episode(dest, index, seed)
            row["wall_s"] = time.monotonic()-started
            report = assess(dest, row)
            status = "accepted" if report["accepted"] else "rejected"
            errors = 0
            print(json.dumps(dict(worker=ARGS.worker, index=index, status=status, failure=report["failure_reason"],
                                  sim_s=report["sim_s"], wall_min=round(row["wall_s"]/60, 1))), flush=True)
        except Exception:
            (dest/"actor_error.json").write_text(json.dumps(dict(error=traceback.format_exc()), indent=2))
            traceback.print_exc()
            errors += 1
            if errors >= 2:
                release_claim(OUT, index, "error")
                return 1
        release_claim(OUT, index, status)
        episodes += 1
    return RECYCLE


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    finally:
        app.close()
        os._exit(code)
