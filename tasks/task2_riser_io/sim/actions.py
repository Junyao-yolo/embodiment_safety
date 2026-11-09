"""Task2 physical action units, decoding and joint command limits."""

import numpy as np

CAMERAS = ("external", "wrist_left", "wrist_right")

STATE_DIM = 23

ACTION_DIM = 48

IMAGE_SIZE = 128

JOINT_MIN = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], np.float32
)

JOINT_MAX = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], np.float32
)


def observation_state(joints, velocities, targets):
    q = np.asarray(joints)
    v = np.asarray(velocities)
    u = np.asarray(targets)
    return np.r_[q[:7], v[:7], q[7:9].sum(), u[:7], u[7:9].sum()].astype(np.float32)


def action_block_to_targets(action):
    a = np.asarray(action, np.float32).reshape(6, 8)
    if not np.isfinite(a).all():
        raise ValueError("Nonfinite action")
    if (
        np.any(a[:, :7] < JOINT_MIN - 0.0001)
        or np.any(a[:, :7] > JOINT_MAX + 0.0001)
        or np.any(a[:, 7] < -0.0001)
        or np.any(a[:, 7] > 0.0801)
    ):
        raise ValueError("Command outside joint or gripper range")
    return np.concatenate([a[:, :7], a[:, 7:8] / 2, a[:, 7:8] / 2], axis=1)


class JointCommandLimits:
    def __init__(self, initial, hz=60.0, velocity=1.8, acceleration=10.0):
        self.previous = np.asarray(initial).copy()
        self.velocity = np.zeros(7)
        self.hz = hz
        self.vmax = velocity
        self.amax = acceleration

    def preview(self, desired):
        desired = np.asarray(desired)
        v = (desired[:7] - self.previous[:7]) * self.hz
        step = self.amax / self.hz
        distance_low = np.maximum(0.0, self.previous[:7] - JOINT_MIN)
        distance_high = np.maximum(0.0, JOINT_MAX - self.previous[:7])
        brake_low = np.sqrt(step * step + 2 * self.amax * distance_low) - step
        brake_high = np.sqrt(step * step + 2 * self.amax * distance_high) - step
        lower = np.maximum(np.maximum(-self.vmax, self.velocity - step), -brake_low)
        upper = np.minimum(np.minimum(self.vmax, self.velocity + step), brake_high)
        v = np.clip(v, lower, upper)
        q = self.previous.copy()
        q[:7] = np.clip(q[:7] + v / self.hz, JOINT_MIN, JOINT_MAX)
        width = np.clip(desired[7:9].sum(), 0.0, 0.08)
        old = q[7:9].sum()
        q[7:9] = (old + np.clip(width - old, -0.15 / self.hz, 0.15 / self.hz)) / 2
        return q

    def commit(self, desired):
        q = self.preview(desired)
        self.velocity = (q[:7] - self.previous[:7]) * self.hz
        self.previous = q.copy()
        return q


def decode_action(action, previous_arm_target):
    values = np.asarray(action, np.float32).reshape(6, 8).copy()
    values[:, :7] = np.asarray(previous_arm_target) + np.cumsum(values[:, :7], axis=0)
    values[:, :7] = np.clip(values[:, :7], JOINT_MIN, JOINT_MAX)
    return action_block_to_targets(values.reshape(48))
