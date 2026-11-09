"""Collect randomized Task1 demonstrations with persistent parallel Isaac workers.

Each worker (worker.py) keeps one Kit process alive and resets between episodes by reopening
collection.usda. Workers are recycled after --episodes-per-process episodes or restarted after a crash.
Run from the repository root:
    algos/diffusion_policy/.venv/bin/python -B -m tools.task1_lab_parallel_v001.collect \
        --output outputs/datasets/task1/parallel_diverse_v001 --successes 100
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation

from runners.common import ROOT, resolve
from runners.eval_diffusion import label_summary, stop_group, write_json
from tasks.task1_dimm_insert.env.smooth_path import SmoothPath
from .plan import PROFILE, TRANSIT_STAGES

ISAAC = "/mnt/data2/isaacsim-5.0.0/python.sh"
WORKER = ROOT/"tools/task1_lab_parallel_v001/worker.py"
ACCEPT = PROFILE["acceptance"]
RECYCLE = 3


def free_gpu_mib():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return int(out.splitlines()[0])


def free_ram_mib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])//1024
    raise RuntimeError("MemAvailable missing")


def high_frequency(values):
    average = np.stack([np.convolve(values[:, k], np.ones(5)/5, mode="valid") for k in range(3)], axis=1)
    return float(np.sqrt(np.mean(np.sum((values[2:-2]-average)**2, axis=1))))


def planned_lengths(expert):
    """Planned transit arc length plus the out-and-back travel of that stage's perturbation."""
    lengths = {}
    for plan in expert.get("paths", []):
        path = SmoothPath(plan["start_m"], plan["end_m"], plan["offset_m"], 1.)
        points = np.array([path.sample(u*path.duration)[0] for u in np.linspace(0, 1, 201)])
        lengths[plan["stage"]] = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    for push in expert.get("perturbations", []):
        if push["state"] in lengths:
            lengths[push["state"]] += push["speed_m_s"]*push["duration_s"]
    return lengths


def motion_quality(rows, expert):
    planned = planned_lengths(expert)
    phases = {}
    for name in sorted({r["diagnostics"].get("expert_stage") for r in rows}):
        selected = [r for r in rows if r["diagnostics"].get("expert_stage") == name]
        if len(selected) < 8:
            continue
        positions = np.array([r["observations"]["ee_position_m"] for r in selected])
        rotations = Rotation.from_quat([r["observations"]["ee_quaternion_xyzw"] for r in selected])
        dt = np.diff([r["observations"]["sim_time"] for r in selected])
        jitter = high_frequency(np.diff(positions, axis=0)/dt[:, None])
        spin = high_frequency((rotations[1:]*rotations[:-1].inv()).as_rotvec()/dt[:, None])
        row = dict(samples=len(selected), velocity_jitter_rms_m_s=jitter, angular_jitter_rms_rad_s=spin)
        ok = jitter <= ACCEPT["velocity_jitter_rms_m_s"] and spin <= ACCEPT["angular_jitter_rms_rad_s"]
        if name in TRANSIT_STAGES:
            arc = float(np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1)))
            chord = float(np.linalg.norm(positions[-1]-positions[0]))
            row["path_to_chord_ratio"] = arc/max(chord, .01)
            row["path_to_plan_ratio"] = arc/max(planned.get(name, chord), .01)
            ok = ok and row["path_to_plan_ratio"] <= ACCEPT["path_to_chord_ratio"]
        row["accepted"] = bool(ok)
        phases[name] = row
    return dict(accepted=all(p["accepted"] for p in phases.values()), phases=phases, thresholds=ACCEPT)


def assess(dest, row):
    result = json.loads((dest/"result.json").read_text())
    episode = Path(result["episode"])
    validation = json.loads((episode/"validation.json").read_text())
    rows = [json.loads(s) for s in (episode/"transitions.jsonl").read_text().splitlines()]
    labels = label_summary(dest)
    expert = json.loads((dest/"expert_result.json").read_text())
    quality = motion_quality(rows, expert)
    stage_s = {}
    for r in rows:
        stage = r["diagnostics"].get("expert_stage")
        stage_s[stage] = stage_s.get(stage, 0.)+.1
    labelled = all("expert_clean_action" in r["diagnostics"] for r in rows)
    constraints = expert.get("safety_constraints", [])
    accepted = bool(result["full_task_success"] and not result["failure_reason"] and validation["data_valid"]
                    and result["source_unchanged"] and labels["reward"] >= ACCEPT["min_reward"]
                    and quality["accepted"] and labelled)
    report = dict(accepted=accepted, full_task_success=result["full_task_success"],
                  failure_reason=result["failure_reason"], data_valid=validation["data_valid"],
                  reward=labels["reward"], stages=labels["stages"], motion_quality=quality,
                  clean_labels_present=labelled, transitions=len(rows), sim_s=round(len(rows)*.1, 1),
                  wall_s=row.get("wall_s"), stage_sim_s={k: round(v, 1) for k, v in stage_s.items()},
                  perturbations=expert.get("perturbations", []),
                  safety_constraints=constraints, safety_pass=not constraints,
                  grasp_relative_slip_peak_m=expert.get("grasp_relative_slip_peak_m", 0.),
                  insert_force_peak_N=expert.get("insert_force_peak_N", 0.),
                  force_peaks=expert.get("force_peaks", {}),
                  randomization=json.loads((dest/"randomization.json").read_text())["plan"],
                  episode=str(episode), output=str(dest), seed=row["seed"], index=row["index"],
                  worker=row.get("worker"))
    write_json(dest/"acceptance.json", report)
    meta = json.loads((episode/"metadata.json").read_text())
    meta.update(task_success=result["full_task_success"], safety_pass=not constraints,
                accepted_successful_demonstration=accepted, acceptance=str(dest/"acceptance.json"))
    write_json(episode/"metadata.json", meta)
    result.update(training_ready=accepted, acceptance_pending=False, acceptance=str(dest/"acceptance.json"))
    write_json(dest/"result.json", result)
    return report


def accepted_count(out):
    return sum(json.loads(p.read_text())["accepted"] for p in Path(out).glob("attempt_*/acceptance.json"))


def release_claim(out, index, status):
    write_json(Path(out)/"claims"/f"{index:03d}"/"done.json", dict(status=status, time=time.time()))


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def release_stale_claims(out, running_pids):
    for claim in sorted((Path(out)/"claims").glob("*")):
        if (claim/"done.json").exists() or not (claim/"owner.json").exists():
            continue
        pid = json.loads((claim/"owner.json").read_text())["pid"]
        if pid not in running_pids and not alive(pid):
            release_claim(out, int(claim.name), "abandoned_by_crashed_worker")


def aggregate(out, target, started_at):
    attempts, accepted = [], []
    for dest in sorted(out.glob("attempt_*")):
        index = int(dest.name.split("_")[1])
        row = dict(index=index, output=str(dest))
        done = out/"claims"/f"{index:03d}"/"done.json"
        row["status"] = json.loads(done.read_text())["status"] if done.exists() else "running"
        if (dest/"acceptance.json").exists():
            a = json.loads((dest/"acceptance.json").read_text())
            row.update(failure_reason=a["failure_reason"] or (None if a["accepted"] else "acceptance_checks"),
                       sim_s=a["sim_s"], wall_s=a["wall_s"], worker=a.get("worker"))
            if a["accepted"]:
                accepted.append({k: v for k, v in a.items() if k != "motion_quality"})
        elif (dest/"actor_error.json").exists():
            row["failure_reason"] = "execution_error"
        if (dest/"timing.json").exists():
            row["timing"] = json.loads((dest/"timing.json").read_text())
        attempts.append(row)
    accepted = sorted(accepted, key=lambda a: a["index"])[:target]
    finished = [a for a in attempts if a.get("wall_s")]
    resets = [a["timing"]["reset_and_setup_s"] for a in attempts if a.get("timing")]
    elapsed = time.time()-started_at
    failures = {}
    for a in attempts:
        if a.get("failure_reason"):
            failures[a["failure_reason"]] = failures.get(a["failure_reason"], 0)+1
    summary = dict(target_successes=target, accepted_count=len(accepted), attempts=attempts,
                   failure_counts=failures, elapsed_s=elapsed,
                   mean_episode_wall_s=float(np.mean([a["wall_s"] for a in finished])) if finished else None,
                   mean_episode_sim_s=float(np.mean([a["sim_s"] for a in finished])) if finished else None,
                   mean_reset_and_setup_s=float(np.mean(resets)) if resets else None,
                   accepted_per_hour=len(accepted)/max(elapsed/3600, 1e-9))
    write_json(out/"manifest.json", dict(task="task1_dimm_insert", requested=target,
        successful_episodes=len(accepted), episodes=accepted, control_hz=10, physics_hz=240,
        controller="controller_batched_fd", collection_profile=PROFILE["version"],
        action_interface=PROFILE["action_interface"], bc_label_field="diagnostics.expert_clean_action",
        demonstration_type="feedback_expert_randomized_scene_with_recoverable_perturbations"))
    return summary


def interrupted(signum, frame):
    raise KeyboardInterrupt(signal.Signals(signum).name)


def main():
    run = PROFILE["run"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True)
    ap.add_argument("--successes", type=int, default=100)
    ap.add_argument("--max-attempts", type=int)
    ap.add_argument("--workers", type=int, default=run["workers"])
    ap.add_argument("--seed", type=int, default=PROFILE["seed"])
    ap.add_argument("--episodes-per-process", type=int, default=20)
    ap.add_argument("--gui", action="store_true", help="show the Isaac window (use with --workers 1)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    max_attempts = args.max_attempts or int(args.successes*1.6)+4
    out = resolve(args.output).resolve()
    if not out.is_relative_to(ROOT/"outputs") or args.successes < 1 or args.workers < 1 or max_attempts < args.successes:
        raise ValueError("Repository output, positive counts and sufficient attempt budget required")
    config = dict(seed=args.seed, profile=PROFILE, successes=args.successes, max_attempts=max_attempts)
    if args.resume:
        previous = json.loads((out/"run_config.json").read_text())
        if previous["seed"] != args.seed or previous["profile"] != PROFILE:
            raise ValueError("Resume must keep the seed and collection profile")
        started_at = previous["started_at"]
    else:
        out.mkdir(parents=True, exist_ok=False)
        started_at = time.time()
    write_json(out/"run_config.json", dict(config, started_at=started_at, workers=args.workers, pid=os.getpid()))
    (out/"claims").mkdir(exist_ok=True)
    release_stale_claims(out, set())
    signal.signal(signal.SIGTERM, interrupted)
    slots = [None]*args.workers
    crashes = 0
    last_launch = 0.
    status = "running"
    try:
        while True:
            summary = aggregate(out, args.successes, started_at)
            running = {s[0].pid for s in slots if s and s[0].poll() is None}
            for k, slot in enumerate(slots):
                if slot is None or slot[0].poll() is None:
                    continue
                proc, log, _ = slot
                log.close(); slots[k] = None
                if proc.returncode not in (0, RECYCLE):
                    crashes += 1
                    print(json.dumps(dict(worker=k, exited=proc.returncode, crashes=crashes)), flush=True)
                release_stale_claims(out, running)
            if crashes > run["max_consecutive_errors"]+args.workers:
                raise RuntimeError("Collection workers keep crashing; see worker_*.log and attempt_*/actor_error.json")
            claims = list((out/"claims").glob("*"))
            pending = sum(not (c/"done.json").exists() for c in claims)
            work_left = summary["accepted_count"]+pending < args.successes and len(claims) < max_attempts
            if summary["accepted_count"] >= args.successes and not running:
                status = "finished"; break
            if not work_left and not running and pending == 0:
                status = "attempt_budget_reached"; break
            for k in range(args.workers):
                if (slots[k] is None and work_left and time.monotonic()-last_launch >= 45
                        and free_gpu_mib() >= run["min_free_gpu_mib"] and free_ram_mib() >= run["min_free_ram_mib"]):
                    threads = str(run["threads_per_worker"])
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES="0", OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                               OPENBLAS_NUM_THREADS=threads, PYTHONDONTWRITEBYTECODE="1")
                    log = (out/f"worker_{k}.log").open("a")
                    cmd = [ISAAC, "-B", str(WORKER), "--output", str(out), "--worker", str(k),
                           "--successes", str(args.successes), "--max-attempts", str(max_attempts),
                           "--seed", str(args.seed), "--episodes-per-process", str(args.episodes_per_process),
                           "--wall-timeout-s", str(run["wall_timeout_s"])]+(["--gui"] if args.gui else [])
                    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log,
                                            stderr=subprocess.STDOUT, start_new_session=True)
                    slots[k] = (proc, log, time.monotonic()); last_launch = time.monotonic()
                    print(json.dumps(dict(worker=k, launched_pid=proc.pid, accepted=summary["accepted_count"])), flush=True)
                    break
            summary.update(status=status, workers=[dict(slot=k, pid=s[0].pid) for k, s in enumerate(slots) if s],
                           crashes=crashes, pid=os.getpid())
            write_json(out/"summary.json", summary)
            time.sleep(10)
    except BaseException as exc:
        status = "error: "+repr(exc)
        raise
    finally:
        for slot in slots:
            if slot:
                stop_group(slot[0]); slot[1].close()
        summary = aggregate(out, args.successes, started_at)
        summary.update(status=status, crashes=crashes)
        write_json(out/"summary.json", summary)
        print(json.dumps({k: summary[k] for k in ("status", "accepted_count", "mean_episode_wall_s",
                          "mean_episode_sim_s", "mean_reset_and_setup_s", "accepted_per_hour")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
