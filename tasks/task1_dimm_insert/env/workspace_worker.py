"""Bridge Task1's observation boundary to the shared in-memory policy server."""

from pathlib import Path
import json
import os
import socket
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from learning.common.protocol import receive, send


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main(out):
    seed = int(os.environ["TASK1_EPISODE_SEED"])
    variant = os.environ["TASK1_DP_INPUT_VARIANT"]
    with socket.create_connection(
        ("127.0.0.1", int(os.environ["DP_POLICY_PORT"]))
    ) as connection:
        send(connection, {"kind": "reset", "seed": seed})
        receive(connection)
        index = 0
        while True:
            request = out / f"request_{index:03d}.json"
            deadline = time.monotonic() + 600
            while not request.exists() and not (out / "policy_stop.json").exists():
                if (out / "actor_error.json").exists():
                    raise RuntimeError("Actor failed")
                if time.monotonic() > deadline:
                    raise TimeoutError("Policy request")
                time.sleep(0.002)
            if not request.exists():
                break
            started = time.perf_counter()
            with np.load(out / f"obs_{index:03d}.npz") as source:
                obs = {key: source[key][0].copy() for key in source.files}
            state = obs["state"]
            if variant == "official":
                raw = json.loads((out / f"raw_{index:03d}.json").read_text())
                state = np.asarray(
                    [
                        *raw["ee_position_m"],
                        *raw["ee_quaternion_xyzw"],
                        *raw["joint_positions"][-2:],
                    ],
                    dtype=np.float32,
                )
            images = {
                key: obs[key] for key in ("external", "wrist_left", "wrist_right")
            }
            requested = json.loads(request.read_text())
            if "teacher_action" in requested:
                response = dict(action=np.asarray(requested["teacher_action"], dtype=np.float32),
                    inference_performed=False, **requested["teacher_info"])
            else:
                send(connection, {"kind": "action", "images": images, "state": state})
                response = receive(connection)
                response["action_source"] = "student"
            if response.get("error"):
                raise ValueError(response["error"])
            action = response.pop("action")
            write(
                out / f"action_{index:03d}.json",
                dict(
                    index=index,
                    executed=action.tolist(),
                    network_updates=0,
                    request_compute_ms=(time.perf_counter() - started) * 1000,
                    **response,
                ),
            )
            index += 1
        stop = json.loads((out / "policy_stop.json").read_text())
        result = dict(
            passed=True,
            network_updates=0,
            new_transitions=stop["final_observation_index"],
            learning=False,
        )
        write(out / "worker_result.json", result)
        send(connection, {"kind": "complete", "summary": result})


if __name__ == "__main__":
    output = Path(sys.argv[1])
    try:
        main(output)
    except BaseException:
        write(output / "worker_error.json", {"error": traceback.format_exc()})
        raise

