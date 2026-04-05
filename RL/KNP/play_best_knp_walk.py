from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from desktop_rl_env import DesktopRobotEnv


class ReplayAgent:
    def __init__(self, policy_path: Path) -> None:
        data = np.load(policy_path)
        self.weights_ih = data["weights_ih"].astype(np.float32)
        self.weights_ho = data["weights_ho"].astype(np.float32)
        self.gait_amp = data["gait_amp"].astype(np.float32)
        self.gait_bias = data["gait_bias"].astype(np.float32)
        self.gait_speed = float(data["gait_speed"])
        self.gait_phase = 0.0

    def reset(self) -> None:
        self.gait_phase = 0.0

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float32)
        obs = obs / (np.linalg.norm(obs) + 1e-6)
        hidden_activity = np.dot(obs, self.weights_ih)
        hidden_spikes = (hidden_activity > np.mean(hidden_activity)).astype(np.float32)
        output_activity = np.dot(hidden_spikes, self.weights_ho)
        output_drive = np.tanh(output_activity) * 12.0

        phase_sin = np.sin(self.gait_phase)
        self.gait_phase += self.gait_speed
        gait = np.array(
            [
                self.gait_bias[0] + self.gait_amp[0] * phase_sin,
                self.gait_bias[1] + self.gait_amp[1] * max(0.0, -phase_sin),
                self.gait_bias[2] - self.gait_amp[2] * phase_sin,
                self.gait_bias[3] + self.gait_amp[3] * max(0.0, phase_sin),
            ],
            dtype=np.float32,
        )
        return gait + output_drive


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the best MNIST-style KNP walking policy.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("C:\\Users\\root\\Documents\\New project\\RL\\KNP\\mnist_style_walk\\walk_forward_best_mnist_style.npz"),
    )
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--repeat-steps", type=int, default=2)
    args = parser.parse_args()

    env = DesktopRobotEnv(repeat_steps=args.repeat_steps)
    agent = ReplayAgent(args.policy)
    step_result = env.reset()
    start_robot_x = float(step_result.raw["observation"]["base_x"])
    start_ball_x = start_robot_x + float(step_result.raw["observation"]["values"][9])
    agent.reset()

    for _ in range(args.steps):
        action_deg = agent.act(step_result.observation)
        step_result = env.step(action_deg)
        if step_result.done or step_result.truncated:
            break

    robot_x = float(step_result.raw["observation"]["base_x"])
    ball_x = robot_x + float(step_result.raw["observation"]["values"][9])
    print(f"robot_dx={robot_x - start_robot_x:+.3f} m")
    print(f"ball_dx_world={ball_x - start_ball_x:+.3f} m")


if __name__ == "__main__":
    main()
