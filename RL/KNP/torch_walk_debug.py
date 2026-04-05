from __future__ import annotations

import argparse

import numpy as np
import torch
from torch import nn

from desktop_rl_env import DesktopRobotEnv


class TorchWalkPolicy(nn.Module):
    def __init__(self, obs_size: int, hidden_size: int = 128, action_scale_deg: float = 180.0) -> None:
        super().__init__()
        self.action_scale_deg = action_scale_deg
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 4),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs) * self.action_scale_deg


def run_debug_rollout(steps: int, repeat_steps: int) -> None:
    env = DesktopRobotEnv(repeat_steps=repeat_steps)
    reset = env.reset()
    policy = TorchWalkPolicy(reset.observation.shape[0])
    obs = torch.from_numpy(reset.observation).float()
    total_reward = 0.0

    for step_idx in range(steps):
        with torch.no_grad():
            action_deg = policy(obs).cpu().numpy()
        result = env.step(action_deg)
        total_reward += result.reward
        obs = torch.from_numpy(result.observation).float()
        print(
            f"step={step_idx:04d} reward={result.reward:+.4f} "
            f"done={result.done} truncated={result.truncated} "
            f"torso_y={result.raw['observation']['torso_height']:.3f} "
            f"torso_angle={np.degrees(result.raw['observation']['torso_angle']):+.1f}°"
        )
        if result.done or result.truncated:
            break

    print(f"total_reward={total_reward:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Torch debug rollout against the desktop robot simulator.")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--repeat-steps", type=int, default=4)
    args = parser.parse_args()
    run_debug_rollout(args.steps, args.repeat_steps)


if __name__ == "__main__":
    main()
