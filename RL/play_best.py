"""Play a trained SAC policy on the SimRoki simulator.

Loads a model saved by train_sac.py (or any SB3 SAC checkpoint) and runs
episodes, printing statistics.

Usage::

    # Play best model from default run directory
    python play_best.py

    # Play a specific checkpoint
    python play_best.py --model runs/sac_simroki/checkpoints/sac_simroki_100000_steps.zip

    # Record observations to CSV
    python play_best.py --csv episode_log.csv

    # Run multiple episodes
    python play_best.py --episodes 5
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gym_env import SimRokiEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Play trained SAC policy on SimRoki")
    p.add_argument(
        "--model",
        type=str,
        default="runs/sac_simroki/best_model/best_model.zip",
        help="Path to saved SB3 SAC model (.zip)",
    )
    p.add_argument("--port", type=int, default=8080, help="Simulator port (default: 8080)")
    p.add_argument("--repeat-steps", type=int, default=4, help="Physics sub-steps per action")
    p.add_argument("--episodes", type=int, default=1, help="Number of episodes to run")
    p.add_argument("--max-steps", type=int, default=2000, help="Max steps per episode")
    p.add_argument("--deterministic", action="store_true", default=True,
                   help="Use deterministic actions (default: True)")
    p.add_argument("--stochastic", action="store_true",
                   help="Use stochastic actions (overrides --deterministic)")
    p.add_argument("--csv", type=str, default=None,
                   help="Optional: save step-by-step log to CSV file")
    p.add_argument("--delay", type=float, default=0.0,
                   help="Delay between steps in seconds (for visual inspection)")
    p.add_argument("--device", type=str, default="auto", help="Torch device")
    return p.parse_args()


def run_episode(
    model: SAC,
    env: SimRokiEnv,
    max_steps: int,
    deterministic: bool,
    csv_writer: csv.writer | None = None,
    step_delay: float = 0.0,
) -> dict:
    """Run one episode and return stats."""
    obs, info = env.reset()
    obs_names = info.get("observation_names", [])

    total_reward = 0.0
    step_count = 0
    breakdown_totals: dict[str, float] = {}

    for step in range(max_steps):
        action, _states = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        # Accumulate breakdown
        bd = info.get("breakdown", {})
        for k, v in bd.items():
            breakdown_totals[k] = breakdown_totals.get(k, 0.0) + v

        # Write CSV row
        if csv_writer is not None:
            row = [step, reward, total_reward, terminated, truncated]
            row.extend(obs.tolist())
            row.extend(action.tolist())
            for k in sorted(bd.keys()):
                row.append(bd[k])
            csv_writer.writerow(row)

        if step_delay > 0:
            time.sleep(step_delay)

        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "steps": step_count,
        "episode_time": info.get("episode_time", 0.0),
        "breakdown_totals": breakdown_totals,
        "terminated": terminated,
        "truncated": truncated,
    }


def main() -> None:
    args = parse_args()
    deterministic = not args.stochastic

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[play] Model not found: {model_path}")
        print("[play] Available models:")
        run_dir = Path("runs/sac_simroki")
        if run_dir.exists():
            for p in sorted(run_dir.rglob("*.zip")):
                print(f"  {p}")
        sys.exit(1)

    print(f"[play] Loading model from {model_path}")
    model = SAC.load(str(model_path), device=args.device)

    env = SimRokiEnv(
        base_url=f"http://127.0.0.1:{args.port}",
        repeat_steps=args.repeat_steps,
    )

    # Prepare CSV writer
    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        # Header will be written after first obs (we need obs names)

    print(f"[play] Running {args.episodes} episode(s), deterministic={deterministic}")
    print(f"[play] Simulator at port {args.port}")
    print()

    all_rewards = []
    all_steps = []

    for ep in range(args.episodes):
        stats = run_episode(
            model, env, args.max_steps, deterministic, csv_writer, args.delay
        )

        all_rewards.append(stats["total_reward"])
        all_steps.append(stats["steps"])

        print(f"Episode {ep + 1}/{args.episodes}:")
        print(f"  Total reward:  {stats['total_reward']:+.3f}")
        print(f"  Steps:         {stats['steps']}")
        print(f"  Episode time:  {stats['episode_time']:.2f}s")
        print(f"  Terminated:    {stats['terminated']}")
        print(f"  Truncated:     {stats['truncated']}")

        if stats["breakdown_totals"]:
            print("  Reward breakdown (episode totals):")
            for k, v in sorted(stats["breakdown_totals"].items()):
                print(f"    {k:>25s}: {v:+.3f}")
        print()

    # Summary
    if args.episodes > 1:
        print("--- Summary ---")
        print(f"  Mean reward: {np.mean(all_rewards):+.3f} +/- {np.std(all_rewards):.3f}")
        print(f"  Mean steps:  {np.mean(all_steps):.0f}")
        print(f"  Min reward:  {np.min(all_rewards):+.3f}")
        print(f"  Max reward:  {np.max(all_rewards):+.3f}")

    if csv_file is not None:
        csv_file.close()
        print(f"[play] CSV log saved to {args.csv}")

    env.close()
    print("[play] Done.")


if __name__ == "__main__":
    main()
