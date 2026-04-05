from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from knp.neuron_traits import BLIFATNeuronParameters

from desktop_rl_env import DesktopRobotEnv


@dataclass
class TrainingStats:
    episode: int
    reward: float
    robot_dx: float
    ball_dx_world: float
    robot_to_ball_dx: float
    steps: int


class KnpStyleSNNPolicy:
    """
    Practical fallback controller:
    uses KNP neuron parameter classes, but integrates membrane dynamics in Python,
    because the installed wheel backend runtime cannot currently load a backend.
    """

    def __init__(self, obs_size: int, hidden_size: int = 64, action_scale_deg: float = 35.0) -> None:
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.action_size = 4
        self.action_scale_deg = action_scale_deg

        self.hidden_params = BLIFATNeuronParameters()
        self.output_params = BLIFATNeuronParameters()
        self.hidden_threshold = float(self.hidden_params.activation_threshold) * 0.85
        self.output_threshold = float(self.output_params.activation_threshold) * 0.80
        self.hidden_decay = 0.92
        self.output_decay = 0.94
        self.reset_value = float(self.hidden_params.potential_reset_value)

        rng = np.random.default_rng(42)
        self.w_in = rng.normal(0.0, 0.18, size=(obs_size, hidden_size)).astype(np.float32)
        self.w_out = rng.normal(0.0, 0.16, size=(hidden_size, self.action_size)).astype(np.float32)
        self.bias_hidden = np.zeros(hidden_size, dtype=np.float32)
        self.bias_out = np.zeros(self.action_size, dtype=np.float32)

        self.hidden_potential = np.zeros(hidden_size, dtype=np.float32)
        self.output_potential = np.zeros(self.action_size, dtype=np.float32)
        self.last_hidden_spikes = np.zeros(hidden_size, dtype=np.float32)
        self.last_actions = np.zeros(self.action_size, dtype=np.float32)
        self.phase = 0.0
        self.phase_speed = 0.22
        self.gait_amplitude = np.array([22.0, 18.0, 22.0, 18.0], dtype=np.float32)
        self.gait_offset = np.array([6.0, 18.0, -6.0, 18.0], dtype=np.float32)

    def reset_state(self) -> None:
        self.hidden_potential.fill(0.0)
        self.output_potential.fill(0.0)
        self.last_hidden_spikes.fill(0.0)
        self.last_actions.fill(0.0)
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

        # Mix membrane and spikes to get smoother motor control than pure binary spikes.
        motor_drive = np.tanh(self.output_potential * 0.35 + output_spikes * 0.65)
        phase_sin = np.sin(self.phase)
        phase_cos = np.cos(self.phase)
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
        actions = gait + correction

        self.last_hidden_spikes = hidden_spikes
        self.last_actions = actions
        return actions.astype(np.float32)

    def update(self, obs: np.ndarray, reward: float, info: dict) -> None:
        obs = np.asarray(obs, dtype=np.float32)
        obs_norm = obs / (np.linalg.norm(obs) + 1e-6)

        robot_x = float(info["observation"]["base_x"])
        ball_center = info["observation"]["center_of_mass"]
        _ = ball_center
        ball_dx = float(info["observation"]["values"][9])
        forward_term = reward
        kick_bonus = max(0.0, info["breakdown"].get("ball_progress", 0.0))
        forward_bonus = max(0.0, info["breakdown"].get("forward_progress", 0.0))

        reinforce = 0.0015 * forward_term + 0.006 * kick_bonus
        anti = 0.0008 * max(0.0, -reward)

        self.w_in += reinforce * np.outer(obs_norm, np.where(self.last_hidden_spikes > 0, 1.0, 0.2))
        self.w_out += reinforce * np.outer(np.where(self.last_hidden_spikes > 0, 1.0, 0.1), self.last_actions / self.action_scale_deg)

        # Encourage rightward motion and ball contact when the ball is in front.
        direction_hint = np.array([0.4, -0.2, -0.4, 0.2], dtype=np.float32)
        if ball_dx > 0.0:
            self.bias_out += (0.0007 + 0.002 * kick_bonus) * direction_hint
            self.gait_offset += (0.002 + 0.003 * forward_bonus) * np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
        if forward_bonus > 0.0:
            self.gait_amplitude += 0.01 * forward_bonus * np.array([1.0, 0.8, 1.0, 0.8], dtype=np.float32)
            self.phase_speed = float(np.clip(self.phase_speed + 0.0005 * forward_bonus, 0.16, 0.34))

        if anti > 0.0:
            self.w_in -= anti * np.outer(obs_norm, np.ones(self.hidden_size, dtype=np.float32))
            self.w_out -= anti * np.outer(np.ones(self.hidden_size, dtype=np.float32), self.last_actions / self.action_scale_deg)
            self.gait_amplitude -= 0.01 * anti * np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        self.w_in = np.clip(self.w_in, -2.5, 2.5)
        self.w_out = np.clip(self.w_out, -2.5, 2.5)
        self.bias_out = np.clip(self.bias_out, -1.5, 1.5)
        self.gait_amplitude = np.clip(self.gait_amplitude, 8.0, 40.0)
        self.gait_offset = np.clip(self.gait_offset, np.array([-15.0, 0.0, -15.0, 0.0]), np.array([20.0, 35.0, 20.0, 35.0]))


def train_visible_knp(
    episodes: int,
    max_steps: int,
    repeat_steps: int,
    action_scale_deg: float,
    hidden_size: int,
    log_path: Path,
) -> None:
    env = DesktopRobotEnv(repeat_steps=repeat_steps)
    first = env.reset()
    policy = KnpStyleSNNPolicy(
        obs_size=first.observation.shape[0],
        hidden_size=hidden_size,
        action_scale_deg=action_scale_deg,
    )

    history: list[dict] = []
    best_score = -float("inf")

    print("KNP walking+kicking training started")
    print(f"obs_size={first.observation.shape[0]} action_size=4 hidden_size={hidden_size}")

    def evaluate_current_policy(policy: KnpStyleSNNPolicy, steps: int = 180) -> tuple[float, float]:
        eval_env = DesktopRobotEnv(repeat_steps=repeat_steps)
        eval_result = eval_env.reset()
        start_robot_x = float(eval_result.raw["observation"]["base_x"])
        start_ball_x = start_robot_x + float(eval_result.raw["observation"]["values"][9])
        policy.reset_state()
        last_result = eval_result
        for _ in range(steps):
            last_result = eval_env.step(policy.act(last_result.observation))
            if last_result.done or last_result.truncated:
                break
        robot_x = float(last_result.raw["observation"]["base_x"])
        ball_x = robot_x + float(last_result.raw["observation"]["values"][9])
        return robot_x - start_robot_x, ball_x - start_ball_x

    for episode in range(1, episodes + 1):
        policy.reset_state()
        step_result = env.reset()
        start_robot_x = float(step_result.raw["observation"]["base_x"])
        start_ball_dx = float(step_result.raw["observation"]["values"][9])
        start_ball_x = start_robot_x + start_ball_dx
        episode_reward = 0.0
        final_raw = step_result.raw
        steps_taken = 0

        for step_idx in range(max_steps):
            action_deg = policy.act(step_result.observation)
            step_result = env.step(action_deg)
            policy.update(step_result.observation, step_result.reward, step_result.raw)
            episode_reward += step_result.reward
            final_raw = step_result.raw
            steps_taken = step_idx + 1

            if step_result.done or step_result.truncated:
                break

        robot_x = float(final_raw["observation"]["base_x"])
        ball_dx = float(final_raw["observation"]["values"][9])
        ball_x = robot_x + ball_dx
        stats = TrainingStats(
            episode=episode,
            reward=episode_reward,
            robot_dx=robot_x - start_robot_x,
            ball_dx_world=ball_x - start_ball_x,
            robot_to_ball_dx=ball_dx,
            steps=steps_taken,
        )
        history.append(stats.__dict__)

        replay_robot_dx, replay_ball_dx = evaluate_current_policy(policy, steps=min(max_steps, 180))
        score = replay_robot_dx + 1.5 * replay_ball_dx + 0.02 * episode_reward
        if score > best_score:
            best_score = score
            np.savez(
                "C:\\Users\\root\\Documents\\New project\\RL\\KNP\\knp_walk_kick_best.npz",
                w_in=policy.w_in,
                w_out=policy.w_out,
                bias_hidden=policy.bias_hidden,
                bias_out=policy.bias_out,
                gait_amplitude=policy.gait_amplitude,
                gait_offset=policy.gait_offset,
                phase_speed=np.array([policy.phase_speed], dtype=np.float32),
                score=np.array([score], dtype=np.float32),
                reward=np.array([episode_reward], dtype=np.float32),
                robot_dx=np.array([stats.robot_dx], dtype=np.float32),
                ball_dx_world=np.array([stats.ball_dx_world], dtype=np.float32),
                replay_robot_dx=np.array([replay_robot_dx], dtype=np.float32),
                replay_ball_dx=np.array([replay_ball_dx], dtype=np.float32),
            )

        if episode % 5 == 0 or episode == 1:
            print(
                f"episode={episode:04d} reward={episode_reward:+.3f} "
                f"robot_dx={robot_x - start_robot_x:+.3f} "
                f"ball_dx_world={ball_x - start_ball_x:+.3f} "
                f"replay_dx={replay_robot_dx:+.3f} replay_ball={replay_ball_dx:+.3f} "
                f"ball_gap={ball_dx:.3f} steps={steps_taken}"
            )
            log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    log_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"saved history to {log_path}")
    print("saved best policy to RL/KNP/knp_walk_kick_best.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visible KNP-style spiking training for walking forward and kicking the ball.")
    parser.add_argument("--episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--repeat-steps", type=int, default=2)
    parser.add_argument("--action-scale-deg", type=float, default=28.0)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("C:\\Users\\root\\Documents\\New project\\RL\\KNP\\knp_walk_kick_training.json"),
    )
    args = parser.parse_args()
    train_visible_knp(
        episodes=args.episodes,
        max_steps=args.max_steps,
        repeat_steps=args.repeat_steps,
        action_scale_deg=args.action_scale_deg,
        hidden_size=args.hidden_size,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
