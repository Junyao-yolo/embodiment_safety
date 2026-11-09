"""Task1's measured state and image representations shared by training and inference."""

import numpy as np
import torch
from learning.common.representation import encode_state, ROT6D

CAMERAS = ("external", "wrist_left", "wrist_right")


def official_lowdim(raw):
    return np.asarray(
        [
            *raw["ee_position_m"],
            *raw["ee_quaternion_xyzw"],
            *raw["joint_positions"][-2:],
        ],
        dtype=np.float32,
    )


def observation(images, state, variant):
    cameras = ("external", "wrist_right") if variant == "official" else CAMERAS
    result = {}
    for key in cameras:
        rgb = torch.from_numpy(np.moveaxis(images[key], -1, -3).copy()).float() / 255
        if variant == "official":
            rgb = torch.nn.functional.interpolate(
                rgb,
                size=(240, 240),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        result[key] = rgb
    if variant == "official":
        result.update(
            robot0_eef_pos=torch.from_numpy(state[..., :3].copy()),
            robot0_eef_quat=torch.from_numpy(state[..., 3:7].copy()),
            robot0_gripper_qpos=torch.from_numpy(state[..., 7:9].copy()),
        )
    elif variant == "current":
        result["state"] = torch.from_numpy(encode_state(state, ROT6D))
    else:
        raise ValueError(f"Unknown Task1 observation variant: {variant}")
    return result
