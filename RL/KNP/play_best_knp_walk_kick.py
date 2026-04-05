from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from knp.neuron_traits import BLIFATNeuronParameters

from desktop_rl_env import DesktopRobotEnv


class ReplayKnpStylePolicy:
    def __init__(self, policy_path: Path, action_scale_deg: float = 20.0) -> None:
        data = np.load(policy_path)
        self.w_in = data["w_in"].astype(np.float32)
        self.w_out = data["w_out"].astype(np.float32)
        self.bias_hidden = data["bias_hidden"].astype(np.float32)
        self.bias_out = data["bias_out"].astype(np.float32)
        self.action_scale_deg = action_scale_deg

        self.hidden_params = BLIFATNeuronParameters()
        self.output_params = BLIFATNeuronParameters()
        self.hidden_threshold = float(self.hidden_params.activation_threshold) * 0.85
        self.output_threshold = float(self.output_params.activation_threshold) * 0.80
        self.hidden_decay = 0.92
        self.output_decay = 0.94
        self.reset_value = float(self.hidden_params.potential_reset_value)

        self.hidden_potential = np.zeros(self.w_in.shape[1], dtype=np.float32)
        self.output_potential = np.zeros(self.w_out.shape[1], dtype=np.float32)
        self.phase = 0.0
        self.phase_speed = float(data["phase_speed"][0]) if "phase_speed" in data else 0.22
        self.gait_amplitude = data["gait_amplitude"].astype(np.float32) if "gait_amplitude" in data else np.array([22.0, 18.0, 22.0, 18.0], dtype=np.float32)
        self.gait_offset = data["gait_offset"].astype(np.float32) if "gait_offset" in data else np.array([6.0, 18.0, -6.0, 18.0], dtype=np.float32)

    def reset(self) -> None:
        self.hidden_potential.fill(0.0)
        self.output_potential.fill(0.0)
        self.phase = 0.0

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = np.asarray(obs, dtype=np.float32)
        obs_norm = obs / (np.linalg.norm(obs) + 1e-6)

        hidden_current = obs_norm @ self.w_in + self.bias_hidden
        self.hidden_potential = self.hidden_potential * self.hidden_decay + hidden_current
        hidden_spikes = (self.hidden_potential >= self.hidden_threshold).astype(np.float32)
        self.hidden_potential[hidden_spikes > 0] = self.reset_value

        output_current = hidden_spikes @ self.w_out + self.bias_out
        self.output_potential = self.output_potential * self.output_decay + output_current
        output_spikes = (self.output_potential >= self.output_threshold).astype(np.float32)
        self.output_potential[output_spikes > 0] = self.reset_value

        motor_drive = np.tanh(self.output_potential * 0.35 + output_spikes * 0.65)
        phase_sin = np.sin(self.phase)
        self.phase += self.phase_speed
        gait = np.array(
            [
                self.gait_offset[0] + self.gait_amplitude[0] * phase_sin,
                self.gait_offset[1] + self.gait_amplitude[1] * max(0.0, -phase_sin),
                self.gait_offset[2] - self.gait_amplitude[2] * phase_sin,
                self.gait_offset[3] + self.gait_amplitude[3] * max(0.0, phase_sin),
            ],
            dtype=np.float32,
        )
        correction = motor_drive * self.action_scale_deg
        return gait + correction


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the best KNP-style walk-and-kick policy.")
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("C:\\Users\\root\\Documents\\New project\\RL\\KNP\\knp_walk_kick_best.npz"),
    )
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--repeat-steps", type=int, default=2)
    parser.add_argument("--action-scale-deg", type=float, default=20.0)
    args = parser.parse_args()

    env = DesktopRobotEnv(repeat_steps=args.repeat_steps)
    agent = ReplayKnpStylePolicy(args.policy, action_scale_deg=args.action_scale_deg)
    step_result = env.reset()
    start_robot_x = float(step_result.raw["observation"]["base_x"])
    start_ball_x = start_robot_x + float(step_result.raw["observation"]["values"][9])
    agent.reset()

    for _ in range(args.steps):
        step_result = env.step(agent.act(step_result.observation))
        if step_result.done or step_result.truncated:
            break

    robot_x = float(step_result.raw["observation"]["base_x"])
    ball_x = robot_x + float(step_result.raw["observation"]["values"][9])
    print(f"robot_dx={robot_x - start_robot_x:+.3f} m")
    print(f"ball_dx_world={ball_x - start_ball_x:+.3f} m")


if __name__ == "__main__":
    main()
