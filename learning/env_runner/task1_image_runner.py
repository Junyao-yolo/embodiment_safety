"""Task1's BaseImageRunner using the existing native 100 ms simulator."""

import json
import os
import subprocess
import time
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from learning.bootstrap import ROOT
from learning.common.policy_server import next_rollout_dir, run_policy_process
from learning.controller.task1_controller import Task1Controller
from tasks.task1_dimm_insert.env.rollout_summary import label_summary


class Task1ImageRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        input_variant="current",
        n_test=3,
        test_start_seed=2026110401,
        max_steps=3000,
        wall_seconds=2700,
        metric_prefix="test",
        safety_mode="warn",
        recovery_config=None,
        isaac_python="/mnt/data2/isaacsim-5.0.0/python.sh",
    ):
        super().__init__(output_dir)
        self.variant = input_variant
        self.n_test = int(n_test)
        self.seed = int(test_start_seed)
        self.max_steps = int(max_steps)
        self.wall_seconds = float(wall_seconds)
        self.prefix = metric_prefix
        self.safety_mode = safety_mode
        self.recovery_config = recovery_config
        self.isaac_python = isaac_python

    def run(self, policy):
        if self.n_test == 0:
            return {}
        out = next_rollout_dir(self.output_dir)
        config = dict(
            input_variant=self.variant,
            recovery_config=self.recovery_config,
            n_test=self.n_test,
            test_start_seed=self.seed,
            max_steps=self.max_steps,
            wall_seconds=self.wall_seconds,
            n_action_steps=policy.n_action_steps,
            safety_mode=self.safety_mode,
            reset_distribution="Task1 default scene; seeds vary diffusion sampling",
        )
        (out / "config.json").write_text(json.dumps(config, indent=2))
        original_generator = policy.kwargs.get("generator")
        controller = Task1Controller(policy, self.variant)
        results = []
        try:
            for index in range(self.n_test):
                base = out / f"trial_{index:03d}"
                dest = base / "eval/episode_01"
                dest.parent.mkdir(parents=True)
                env = os.environ.copy()
                for name in (
                    "TASK1_DIAGNOSTIC_STOP_STAGE",
                    "TASK1_NATIVE_PROFILE",
                    "TASK1_WINDOW_GATES",
                    "TASK1_DAGGER_REFERENCES",
                ):
                    env.pop(name, None)
                env.update(
                    TASK1_POLICY_BACKEND="diffusion_policy_workspace",
                    TASK1_EVAL="1",
                    TASK1_DP_ACTION_STEPS=str(policy.n_action_steps),
                    TASK1_ONLINE_LEARN="0",
                    TASK1_MAX_STEPS=str(self.max_steps),
                    TASK1_SMOKE="0",
                    TASK1_EPISODE_SEED=str(self.seed + index),
                    TASK1_SAFETY_MODE=self.safety_mode,
                    TASK1_DP_INPUT_VARIANT=self.variant,
                    TASK1_EXECUTION_VARIANT="batched_io",
                    TASK1_GOAL_LEAD_HORIZON_S=".25",
                    TASK1_MAX_FREE_GOAL_LEAD_M=".03",
                    TASK1_IK_ROTATION_TOLERANCE_RAD=".002",
                )
                env.pop("TASK1_CONTACT_RECOVERY", None)
                if self.recovery_config:
                    env["TASK1_CONTACT_RECOVERY"] = str(self.recovery_config)
                started = time.monotonic()
                timed_out = False
                try:
                    run_policy_process(
                        [
                            self.isaac_python,
                            "-B",
                            str(ROOT / "tools/task1_execution_v004/actor.py"),
                            str(dest),
                        ],
                        controller,
                        base / "actor.log",
                        env=env,
                        wall_seconds=self.wall_seconds,
                    )
                except TimeoutError:
                    timed_out = True
                report = (
                    json.loads((dest / "result.json").read_text())
                    if (dest / "result.json").exists()
                    else {}
                )
                row = {
                    "trial": index,
                    "seed": self.seed + index,
                    "wall_s": time.monotonic() - started,
                    "wall_timeout": timed_out,
                    "task_completion_observed": False,
                    "full_task_success": False,
                    **report,
                }
                row.update(label_summary(dest))
                if (dest / "teacher_result.json").exists():
                    row["teacher_recovery"] = json.loads((dest / "teacher_result.json").read_text())
                raw = sorted(dest.glob("raw_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
                if raw:
                    t0 = json.loads(raw[0].read_text())["sim_time"]
                    row["simulation_elapsed_s"] = json.loads(raw[-1].read_text())["sim_time"] - t0
                    row["stage_first_sim_seconds"] = {}
                    for path in sorted(dest.glob("stage_transition_*.json"), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
                        for event in json.loads(path.read_text())["events"]:
                            row["stage_first_sim_seconds"].setdefault(event["stage"], event["time"] - t0)
                actions = [json.loads(p.read_text()) for p in dest.glob("action_*.json")]
                row["inference_count"] = sum(bool(x.get("inference_performed")) for x in actions)
                row["inference_seconds"] = sum(x.get("inference_ms", 0) for x in actions) / 1000
                row["preprocessing_seconds"] = sum(x.get("preprocessing_ms", 0) for x in actions) / 1000
                row["policy_compute_seconds"] = sum(x.get("request_compute_ms", 0) for x in actions) / 1000
                valid = (dest / "result.json").exists() or (
                    timed_out and row["full_transitions"] > 0
                )
                row["valid_policy_test"] = valid
                (dest.parent / "summary.json").write_text(json.dumps(row, indent=2))
                if list(dest.glob("raw_*.json")):
                    with (base / "render.log").open("w") as log:
                        subprocess.run(
                            [
                                str(ROOT / "algos/diffusion_policy/.venv/bin/python"),
                                "-B",
                                str(
                                    ROOT / "tools/task1_execution_v004/render_replay.py"
                                ),
                                str(base),
                            ],
                            cwd=ROOT,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    row["video"] = str(base / "replay/actual_policy_replay.mp4")
                results.append(row)
                successes = sum(bool(x["task_completion_observed"]) for x in results)
                safe = sum(bool(x["full_task_success"]) for x in results)
                summary = dict(
                    requested=self.n_test,
                    teacher_enabled=bool(self.recovery_config),
                    action_steps=policy.n_action_steps,
                    stage_counts={stage: sum(stage in x.get("stages", []) for x in results)
                                  for stage in ("both_open", "grasp_clear", "insertion_ready", "seated_locked")},
                    wall_seconds=sum(x["wall_s"] for x in results),
                    completed=len(results),
                    task_successes=successes,
                    task_success_rate=successes / len(results),
                    safe_completion_rate=safe / len(results),
                    episodes=results,
                    damage_rate=None,
                    safety_scope="Existing Task1 measured task and safety gates",
                )
                summary["stage_rates"] = {key: value / len(results) for key, value in summary["stage_counts"].items()}
                (out / "summary.json").write_text(json.dumps(summary, indent=2))
                if not valid:
                    raise RuntimeError(f"Invalid Task1 policy test: {dest}")
        finally:
            if original_generator is None:
                policy.kwargs.pop("generator", None)
            else:
                policy.kwargs["generator"] = original_generator
        metrics = {
            f"{self.prefix}/mean_score": summary["task_success_rate"],
            f"{self.prefix}/safe_completion_rate": summary["safe_completion_rate"],
        }
        print(json.dumps({"rollout": str(out), **metrics}), flush=True)
        return metrics


