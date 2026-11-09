"""Execute Task1 velocity commands from the official image policy."""

import time
import numpy as np
import torch
from learning.dataset.task1_observation import observation
from learning.controller.action_chunk import ActionChunk


class Task1Controller:
    def __init__(self, policy, input_variant):
        self.policy = policy
        self.variant = input_variant
        self.generator = torch.Generator(device=policy.device)
        self.policy.kwargs["generator"] = self.generator
        self.reset(42)

    def reset(self, seed):
        self.seed = int(seed)
        self.generator.manual_seed(self.seed)
        self.policy.reset()
        self.history = []
        self.chunk = ActionChunk(self.policy.n_action_steps)
        self.index = 0
        self.last_info = {}

    @torch.no_grad()
    def act(self, images, state):
        self.history.append((images, np.asarray(state, dtype=np.float32)))
        self.history = self.history[-self.policy.n_obs_steps :]
        pair = [self.history[0]] * (
            self.policy.n_obs_steps - len(self.history)
        ) + self.history

        timings = dict(preprocessing_ms=0.0, inference_ms=0.0)

        def predict():
            started = time.perf_counter()
            obs = observation(
                {key: np.stack([item[0][key] for item in pair]) for key in images},
                np.stack([item[1] for item in pair]),
                self.variant,
            )
            batch = {
                key: value[None].to(self.policy.device) for key, value in obs.items()
            }
            timings["preprocessing_ms"] = (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            actions = self.policy.predict_action(batch)["action"][0].cpu().numpy()
            timings["inference_ms"] = (time.perf_counter() - started) * 1000
            return actions

        action, inferred, plan, offset = self.chunk.take(self.index, "policy", predict)
        if not np.isfinite(action).all():
            raise ValueError("Nonfinite action")
        self.last_info = dict(
            **timings,
            proposed=action.tolist(),
            inference_performed=inferred,
            plan_start_index=plan,
            chunk_offset=offset,
            executed_chunk=self.policy.n_action_steps,
            seed=self.seed,
            history_frames=len(self.history),
            action_horizon=self.policy.horizon,
        )
        self.index += 1
        return np.clip(action, -1, 1)

