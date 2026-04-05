#!/usr/bin/env python3
"""
Walking training in the same practical style as neurochip/train_mnist_knp.py:
- simple NumPy forward pass
- simple STDP-like weight updates
- real desktop robot simulator as the environment
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from desktop_rl_env import DesktopRobotEnv


class MnistStyleWalkingAgent:
    def __init__(self, obs_size: int, hidden_size: int = 96, action_size: int = 4) -> None:
        self.obs_size = obs_size
        self.hidden_size = hidden_size
        self.action_size = action_size

        rng = np.random.default_rng(7)
        self.weights_ih = rng.normal(0.0, 0.08, size=(obs_size, hidden_size)).astype(np.float32)
        self.weights_ho = rng.normal(0.0, 0.08, size=(hidden_size, action_size)).astype(np.float32)

        self.gait_phase = 0.0
        self.gait_speed = 0.22
        self.gait_amp = np.array([22.0, 18.0, 22.0, 18.0], dtype=np.float32)
        self.gait_bias = np.array([6.0, 18.0, -6.0, 18.0], dtype=np.float32)

    def reset(self) -> None:
        self.gait_phase = 0.0

    def forward_pass(self, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        obs = np.asarray(observation, dtype=np.float32)
        obs = obs / (np.linalg.norm(obs) + 1e-6)

        hidden_activity = np.dot(obs, self.weights_ih)
        hidden_spikes = (hidden_activity > np.mean(hidden_activity)).astype(np.float32)

        output_activity = np.dot(hidden_spikes, self.weights_ho)
        output_drive = np.tanh(output_activity) * 8.0

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
        action_deg = gait + output_drive
        return action_deg.astype(np.float32), hidden_spikes, obs

    def update_weights(
        self,
        obs_norm: np.ndarray,
        hidden_spikes: np.ndarray,
        action_deg: np.ndarray,
        forward_progress: float,
        ball_progress: float,
        total_reward: float,
    ) -> None:
        if forward_progress > 0.0:
            self.weights_ih += 0.0022 * forward_progress * np.outer(obs_norm, np.where(hidden_spikes > 0, 1.0, 0.2))
            self.weights_ho += 0.0020 * forward_progress * np.outer(hidden_spikes, action_deg / 30.0)
            self.gait_bias += 0.010 * forward_progress * np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
            self.gait_amp += 0.030 * forward_progress * np.array([1.0, 0.7, 1.0, 0.7], dtype=np.float32)
            self.gait_speed = float(np.clip(self.gait_speed + 0.0008 * forward_progress, 0.18, 0.34))

        if ball_progress > 0.0:
            self.weights_ho += 0.0040 * ball_progress * np.outer(np.where(hidden_spikes > 0, 1.0, 0.4), np.array([0.8, 0.2, -0.8, 0.2], dtype=np.float32))
            self.gait_amp += 0.040 * ball_progress * np.array([1.0, 0.7, 1.0, 0.7], dtype=np.float32)

        if forward_progress < 0.0:
            back = abs(forward_progress)
            self.weights_ih -= 0.0015 * back * np.outer(obs_norm, np.ones(self.hidden_size, dtype=np.float32))
            self.weights_ho -= 0.0012 * back * np.outer(np.where(hidden_spikes > 0, 1.0, 0.3), action_deg / 30.0)
            self.gait_bias -= 0.010 * back * np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
            self.gait_amp -= 0.020 * back * np.array([1.0, 0.7, 1.0, 0.7], dtype=np.float32)

        if total_reward < 0.0:
            self.weights_ih *= 0.999
            self.weights_ho *= 0.9985

        self.weights_ih = np.clip(self.weights_ih, -2.0, 2.0)
        self.weights_ho = np.clip(self.weights_ho, -2.0, 2.0)
        self.gait_amp = np.clip(self.gait_amp, 8.0, 35.0)
        self.gait_bias = np.clip(self.gait_bias, np.array([-15.0, 0.0, -15.0, 0.0]), np.array([20.0, 35.0, 20.0, 35.0]))


def train_walking(
    episodes: int,
    max_steps: int,
    repeat_steps: int,
    hidden_size: int,
    output_dir: Path,
) -> tuple[dict, list[dict]]:
    print("=" * 70)
    print("WALKING TRAINING WITH KNP (MNIST-STYLE SIMPLIFIED)")
    print("=" * 70)

    env = DesktopRobotEnv(repeat_steps=repeat_steps)
    first = env.reset()
    agent = MnistStyleWalkingAgent(first.observation.shape[0], hidden_size=hidden_size)

    history: list[dict] = []
    best_episode: dict | None = None
    best_score = -float("inf")

    print(f"Observation size: {first.observation.shape[0]}")
    print(f"Hidden size: {hidden_size}")
    print(f"Episodes: {episodes}")

    for episode in range(1, episodes + 1):
        agent.reset()
        step_result = env.reset()
        start_robot_x = float(step_result.raw["observation"]["base_x"])
        start_ball_dx = float(step_result.raw["observation"]["values"][9])
        start_ball_x = start_robot_x + start_ball_dx

        episode_reward = 0.0
        steps_taken = 0
        last_robot_x = start_robot_x
        last_ball_x = start_ball_x
        best_step_action = np.zeros(4, dtype=np.float32)

        for step_idx in range(max_steps):
            action_deg, hidden_spikes, obs_norm = agent.forward_pass(step_result.observation)
            step_result = env.step(action_deg)

            robot_x = float(step_result.raw["observation"]["base_x"])
            ball_x = robot_x + float(step_result.raw["observation"]["values"][9])
            forward_progress = robot_x - last_robot_x
            ball_progress = max(0.0, ball_x - last_ball_x)
            last_robot_x = robot_x
            last_ball_x = ball_x

            agent.update_weights(
                obs_norm=obs_norm,
                hidden_spikes=hidden_spikes,
                action_deg=action_deg,
                forward_progress=forward_progress,
                ball_progress=ball_progress,
                total_reward=step_result.reward,
            )
            episode_reward += step_result.reward
            steps_taken = step_idx + 1
            best_step_action = action_deg

            if step_result.done or step_result.truncated:
                break

        robot_dx = last_robot_x - start_robot_x
        ball_dx_world = last_ball_x - start_ball_x
        score = 4.0 * robot_dx + 1.5 * ball_dx_world + 0.01 * episode_reward
        episode_info = {
            "episode": episode,
            "reward": float(episode_reward),
            "robot_dx": float(robot_dx),
            "ball_dx_world": float(ball_dx_world),
            "steps": steps_taken,
            "score": float(score),
        }
        history.append(episode_info)

        if score > best_score:
            best_score = score
            best_episode = {
                "score": float(score),
                "episode": episode,
                "weights_ih": agent.weights_ih.copy(),
                "weights_ho": agent.weights_ho.copy(),
                "gait_amp": agent.gait_amp.copy(),
                "gait_bias": agent.gait_bias.copy(),
                "gait_speed": float(agent.gait_speed),
                "last_action_deg": best_step_action.copy(),
            }

        if episode == 1 or episode % 5 == 0:
            print(
                f"Episode {episode:03d}: reward={episode_reward:+8.2f}, "
                f"robot_dx={robot_dx:+6.2f}m, ball_dx={ball_dx_world:+6.2f}m, score={score:+7.2f}"
            )

    assert best_episode is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "walk_forward_best_mnist_style.npz",
        weights_ih=best_episode["weights_ih"],
        weights_ho=best_episode["weights_ho"],
        gait_amp=best_episode["gait_amp"],
        gait_bias=best_episode["gait_bias"],
        gait_speed=best_episode["gait_speed"],
        last_action_deg=best_episode["last_action_deg"],
        episode=best_episode["episode"],
        score=best_episode["score"],
    )
    (output_dir / "walk_forward_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_episode, history


def save_training_plot(history: list[dict], output_dir: Path) -> None:
    episodes = [item["episode"] for item in history]
    rewards = [item["reward"] for item in history]
    robot_dx = [item["robot_dx"] for item in history]
    ball_dx = [item["ball_dx_world"] for item in history]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(episodes, rewards, "b-", alpha=0.8)
    axes[0].set_title("Reward per Episode")
    axes[0].set_ylabel("Reward")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episodes, robot_dx, "g-", alpha=0.8, label="robot_dx")
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Forward Distance per Episode")
    axes[1].set_ylabel("Robot dx (m)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(episodes, ball_dx, "r-", alpha=0.8, label="ball_dx_world")
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_title("Ball Progress per Episode")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Ball dx (m)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "walk_forward_training_results.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train forward walking in the same simplified style as neurochip MNIST.")
    parser.add_argument("--episodes", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--repeat-steps", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("C:\\Users\\root\\Documents\\New project\\RL\\KNP\\mnist_style_walk"),
    )
    args = parser.parse_args()

    best_episode, history = train_walking(
        episodes=args.episodes,
        max_steps=args.max_steps,
        repeat_steps=args.repeat_steps,
        hidden_size=args.hidden_size,
        output_dir=args.output_dir,
    )
    save_training_plot(history, args.output_dir)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Best episode: {best_episode['episode']}")
    print(f"Best score: {best_episode['score']:.3f}")
    print(f"Saved policy: {args.output_dir / 'walk_forward_best_mnist_style.npz'}")
    print(f"Saved history: {args.output_dir / 'walk_forward_history.json'}")
    print(f"Saved plot: {args.output_dir / 'walk_forward_training_results.png'}")


if __name__ == "__main__":
    main()
