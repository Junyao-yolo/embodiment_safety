"""Isaac process: reuse one scene and renderer across policy trials."""

import argparse
import gc
import json
from pathlib import Path
import socket
import os
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from learning.common.protocol import RemoteController, send
from tasks.task2_riser_io.sim.summary import classify_trial, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)

    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    out = Path(config["output_dir"])
    sys.argv = [sys.argv[0]]
    with socket.create_connection(
        ("127.0.0.1", int(os.environ["DP_POLICY_PORT"]))
    ) as connection:
        from isaacsim import SimulationApp

        app = SimulationApp(
            {
                "headless": True,
                "limit_cpu_threads": 4,
                "width": 128,
                "height": 128,
                "sync_loads": True,
                "extra_args": ["--/log/file=" + config["kit_log"]],
            }
        )
        from tasks.task2_riser_io.sim.episode import run_episode
        from tasks.task2_riser_io.sim.recorder import TrialRecorder

        controller = RemoteController(connection)
        session = {}
        results = []
        started = time.perf_counter()
        try:
            for i, seed in enumerate(config["seeds"]):
                trial = out / f"trial_{i:03d}"
                trial.mkdir()
                before = time.perf_counter()
                controller.reset(seed)
                recorder = TrialRecorder(
                    trial, seed, config["thresholds"], label="Task2 Diffusion Policy"
                )

                def progress(row, performance):
                    (out / "progress.json").write_text(
                        json.dumps(
                            dict(
                                status="TESTING",
                                completed=len(results),
                                total=len(config["seeds"]),
                                active_trial=i,
                                seed=seed,
                                trial_wall_s=time.perf_counter() - before,
                                measurements=row,
                                performance=performance,
                            ),
                            indent=2,
                        )
                    )

                report, session, simulation_s = run_episode(
                    config, trial, seed, app, session, recorder, controller, progress
                )
                report["policy_updates"] = (
                    report.get("command_blocks", 0) + config["n_action_steps"] - 1
                ) // config["n_action_steps"]
                (trial / "result.json").write_text(json.dumps(report, indent=2))
                safety = recorder.finish(report)
                result = classify_trial(
                    report,
                    safety,
                    i,
                    seed,
                    time.perf_counter() - before,
                    simulation_s,
                    trial,
                )
                (trial / "evaluation.json").write_text(json.dumps(result, indent=2))
                results.append(result)
                summary = summarize(results, out)
                del recorder
                gc.collect()
                if not result["valid_policy_test"]:
                    raise RuntimeError(f"Execution error: {trial / 'result.json'}")
            (out / "progress.json").write_text(
                json.dumps(
                    dict(
                        status="COMPLETE",
                        completed=len(results),
                        total=len(config["seeds"]),
                        wall_s=time.perf_counter() - started,
                    ),
                    indent=2,
                )
            )
            send(connection, {"kind": "complete", "summary": summary})
        except Exception:
            (out / "error.txt").write_text(traceback.format_exc())
            raise
        finally:
            for annotator in session.get("render", {}).get("anns", {}).values():
                annotator.detach()
            for product in session.get("render", {}).get("products", []):
                product.destroy()
            app.close()


if __name__ == "__main__":
    main()
