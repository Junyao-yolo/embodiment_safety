"""Versioned DP-only state adapter; raw Task1 state31 and stored demos stay intact."""

import numpy as np

LEGACY = "quaternion31_v001"
ROT6D = "rotation6d33_v002"


def state_config(name):
    if name == LEGACY:
        return dict(
            state_representation=name,
            state_dim=31,
            interface_version="task1_rgb_state31_reward_history_v001",
        )
    if name == ROT6D:
        return dict(
            state_representation=name,
            state_dim=33,
            interface_version="task1_rgb_state33_rot6d_reward_history_v002",
        )
    raise ValueError("Unknown state representation: " + str(name))


def validate_config(cfg):
    expected = state_config(cfg.get("state_representation", LEGACY))
    if cfg.get("state_dim", 31) != expected["state_dim"]:
        raise ValueError("State dimension / representation mismatch")
    if (
        cfg.get("interface_version", expected["interface_version"])
        != expected["interface_version"]
    ):
        raise ValueError("State interface version mismatch")
    return expected


def encode_state(raw, representation=LEGACY):
    """Quaternion XYZW -> concatenated first/second rotation-matrix columns.
    Input [...,31]: joints/velocities/width/goalXYZ, q[22:26], target/history.
    Output [...,33]: unchanged [:22], R[:,0], R[:,1], unchanged [26:].
    q and -q map to exactly the same features; no history-dependent sign rule.
    """
    state_config(representation)
    a = np.asarray(raw, dtype=np.float32)
    if a.shape[-1] != 31 or not np.isfinite(a).all():
        raise ValueError("Expected finite raw Task1 state31")
    if representation == LEGACY:
        return a.copy()
    q = a[..., 22:26].astype(np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Invalid zero quaternion")
    q = q / norm
    x, y, z, w = np.moveaxis(q, -1, 0)
    c0 = np.stack(
        (1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)), axis=-1
    )
    c1 = np.stack(
        (2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)), axis=-1
    )
    return np.concatenate((a[..., :22], c0, c1, a[..., 26:]), axis=-1).astype(
        np.float32
    )
