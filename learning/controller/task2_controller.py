"""Translate the official image-policy interface to Task2 command blocks."""

import numpy as np
import torch
from tasks.task2_riser_io.sim.actions import (
    CAMERAS,
    decode_action,
)


class ActionController:
    def __init__(self, policy, fixed_noise=False):
        self.policy = policy
        self.fixed_noise = bool(fixed_noise)
        self.generator = torch.Generator(device=policy.device)
        self.policy.kwargs["generator"] = self.generator
        self.reset(42)

    def reset(self, seed):
        self.seed = int(seed)
        self.generator.manual_seed(self.seed)
        self.policy.reset()
        self.history = []
        self.pending = []
        self.inference_calls = 0

    @torch.no_grad()
    def act(self, images, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (23,):
            raise ValueError("Expected Task2 state with 23 values")
        sample = {"state": state.copy()}
        for key in CAMERAS:
            sample[key] = np.asarray(images[key]).copy()
            if sample[key].shape != (128, 128, 3):
                raise ValueError(f"Invalid RGB shape for {key}")
        self.history = (self.history + [sample])[-self.policy.n_obs_steps :]
        if not self.pending:
            pair = [self.history[0]] * (
                self.policy.n_obs_steps - len(self.history)
            ) + self.history
            obs = {
                key: torch.from_numpy(np.stack([row[key] for row in pair]))
                .permute(0, 3, 1, 2)[None]
                .float()
                .to(self.policy.device)
                / 255.0
                for key in CAMERAS
            }
            obs["state"] = torch.from_numpy(np.stack([row["state"] for row in pair]))[
                None
            ].to(self.policy.device)
            if self.fixed_noise:
                self.generator.manual_seed(self.seed)
            self.pending = list(
                self.policy.predict_action(obs)["action"][0].cpu().numpy()
            )
            self.inference_calls += 1
        return decode_action(self.pending.pop(0), state[15:22])
