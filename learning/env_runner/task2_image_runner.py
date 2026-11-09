"""Task2's BaseImageRunner: persistent scene, measured assembly and safety."""

import json
from pathlib import Path
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from learning.bootstrap import ROOT
from learning.common.policy_server import next_rollout_dir, run_policy_process
from learning.controller.task2_controller import ActionController


class Task2ImageRunner(BaseImageRunner):
    def __init__(
        self,
        output_dir,
        scene,
        runtime,
        motion,
        thresholds,
        n_test=3,
        test_start_seed=980000,
        max_seconds=360,
        fixed_noise=False,
        metric_prefix="test",
        isaac_python="/mnt/data2/isaacsim-5.0.0/python.sh",
    ):
        super().__init__(output_dir)
        self.config = dict(
            task_root=str(ROOT / "tasks/task2_riser_io"),
            scene=scene,
            runtime=json.loads(Path(runtime).read_text()),
            motion=json.loads(Path(motion).read_text()),
            thresholds=json.loads(Path(thresholds).read_text()),
            seeds=[test_start_seed + 4 * i for i in range(n_test)],
            max_seconds=max_seconds,
            fixed_noise=fixed_noise,
            physics_hz=120,
            metric_prefix=metric_prefix,
        )
        self.isaac_python = isaac_python

    def run(self, policy):
        out = next_rollout_dir(self.output_dir)
        config = dict(
            self.config,
            output_dir=str(out),
            kit_log=str(out / "kit.log"),
            n_action_steps=policy.n_action_steps,
        )
        (out / "config.json").write_text(json.dumps(config, indent=2))
        original_generator = policy.kwargs.get("generator")
        controller = ActionController(policy, fixed_noise=config["fixed_noise"])
        try:
            summary = run_policy_process(
                [
                    self.isaac_python,
                    "-B",
                    "-u",
                    str(ROOT / "tasks/task2_riser_io/sim/run_sim.py"),
                    "--config",
                    str(out / "config.json"),
                ],
                controller,
                out / "sim.log",
            )
        finally:
            if original_generator is None:
                policy.kwargs.pop("generator", None)
            else:
                policy.kwargs["generator"] = original_generator
        prefix = config["metric_prefix"]
        metrics = {
            f"{prefix}/mean_score": summary["task_success_rate"],
            f"{prefix}/safe_completion_rate": summary["safe_completion_rate"],
            f"{prefix}/boundary_violation_rate": summary["boundary_violation_rate"],
        }
        print(json.dumps({"rollout": str(out), **metrics}), flush=True)
        return metrics
