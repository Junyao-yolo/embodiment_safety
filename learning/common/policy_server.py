"""Serve an in-memory policy to a local simulator process."""

import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from learning.bootstrap import ROOT
from learning.common.protocol import receive, send


def next_rollout_dir(output_dir):
    root = Path(output_dir) / "rollouts"
    root.mkdir(parents=True, exist_ok=True)
    index = 0
    while (root / f"round_{index:03d}").exists():
        index += 1
    output = root / f"round_{index:03d}"
    output.mkdir()
    return output


def run_policy_process(command, controller, log_path, env=None, wall_seconds=None):
    deadline = None if wall_seconds is None else time.monotonic() + wall_seconds

    def remaining():
        if deadline is None:
            return None
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("Simulator wall-time budget exhausted")
        return value

    with socket.socket() as listener, Path(log_path).open("w") as log:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(1)
        environment = dict(os.environ if env is None else env)
        environment["DP_POLICY_PORT"] = str(listener.getsockname()[1])
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        completed = False
        try:
            while True:
                remaining()
                try:
                    connection, _ = listener.accept()
                    break
                except socket.timeout:
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Simulator failed before connecting: {log_path}"
                        )
            with connection:
                while True:
                    connection.settimeout(remaining())
                    try:
                        message = receive(connection)
                    except EOFError:
                        raise RuntimeError(
                            f"Simulator disconnected: {log_path}"
                        ) from None
                    kind = message["kind"]
                    if kind == "reset":
                        controller.reset(message["seed"])
                        send(connection, {"ok": True})
                    elif kind == "action":
                        try:
                            action = controller.act(message["images"], message["state"])
                            response = {
                                "action": action,
                                **getattr(controller, "last_info", {}),
                            }
                        except ValueError as error:
                            response = {"error": str(error)}
                        send(connection, response)
                    elif kind == "complete":
                        summary = message["summary"]
                        break
                    else:
                        raise ValueError(f"Unknown simulator request: {kind}")
            try:
                code = process.wait(timeout=remaining())
            except subprocess.TimeoutExpired:
                raise TimeoutError("Simulator wall-time budget exhausted") from None
            if code != 0:
                raise RuntimeError(f"Simulator failed: {log_path}")
            completed = True
            return summary
        finally:
            if not completed:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
