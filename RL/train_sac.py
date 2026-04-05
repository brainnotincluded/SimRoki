"""SAC training for the SimRoki 2D biped robot.

Uses stable-baselines3 SAC with DummyVecEnv (the heavy simulation already
runs out-of-process, so the Python envs are just HTTP clients -- no need for
SubprocVecEnv).

Usage::

    # Single env (one sim on default port)
    python train_sac.py

    # 10 parallel envs on ports 8080-8089
    python train_sac.py --num-envs 10 --base-port 8080

    # Resume from checkpoint
    python train_sac.py --resume runs/sac_simroki/checkpoint_100000.zip
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Local imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gym_env import SimRokiEnv, make_env


# ------------------------------------------------------------------
# Custom callback: episode logging
# ------------------------------------------------------------------


class EpisodeStatsCallback(BaseCallback):
    """Log per-episode stats: reward, length, breakdown components."""

    def __init__(self, log_path: str | None = None, verbose: int = 1):
        super().__init__(verbose)
        self.log_path = log_path
        self._episode_rewards: list[float] = []
        self._episode_lengths: list[int] = []
        self._log_entries: list[dict] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                r = ep_info["r"]
                l = ep_info["l"]
                self._episode_rewards.append(r)
                self._episode_lengths.append(l)

                entry = {
                    "timestep": self.num_timesteps,
                    "reward": r,
                    "length": l,
                }
                # Add breakdown if available
                if "breakdown" in info:
                    entry["breakdown"] = info["breakdown"]

                self._log_entries.append(entry)

                if self.verbose >= 1 and len(self._episode_rewards) % 10 == 0:
                    recent = self._episode_rewards[-50:]
                    print(
                        f"[step {self.num_timesteps:>8d}] "
                        f"episodes={len(self._episode_rewards)}  "
                        f"mean_reward={np.mean(recent):+.2f}  "
                        f"mean_len={np.mean(self._episode_lengths[-50:]):.0f}"
                    )

        return True

    def _on_training_end(self) -> None:
        if self.log_path and self._log_entries:
            Path(self.log_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "w") as f:
                json.dump(self._log_entries, f, indent=2)
            print(f"[stats] Wrote {len(self._log_entries)} episode records to {self.log_path}")


# ------------------------------------------------------------------
# Env creation helpers
# ------------------------------------------------------------------


def make_training_envs(
    num_envs: int, base_port: int, repeat_steps: int
) -> DummyVecEnv:
    """Create a DummyVecEnv with *num_envs* SimRoki HTTP clients."""
    env_fns = []
    for i in range(num_envs):
        port = base_port + i
        env_fns.append(lambda p=port: Monitor(SimRokiEnv(
            base_url=f"http://127.0.0.1:{p}",
            repeat_steps=repeat_steps,
        )))
    return DummyVecEnv(env_fns)


def make_eval_env(port: int, repeat_steps: int) -> DummyVecEnv:
    """Single-env DummyVecEnv for evaluation."""
    return DummyVecEnv([
        lambda: Monitor(SimRokiEnv(
            base_url=f"http://127.0.0.1:{port}",
            repeat_steps=repeat_steps,
        ))
    ])


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SAC training for SimRoki biped")
    p.add_argument("--num-envs", type=int, default=1,
                   help="Number of parallel env instances (default: 1)")
    p.add_argument("--base-port", type=int, default=8080,
                   help="Base HTTP port for simulators (default: 8080)")
    p.add_argument("--repeat-steps", type=int, default=4,
                   help="Physics sub-steps per action (default: 4)")
    p.add_argument("--total-timesteps", type=int, default=500_000,
                   help="Total training timesteps (default: 500000)")
    p.add_argument("--learning-rate", type=float, default=3e-4,
                   help="Learning rate (default: 3e-4)")
    p.add_argument("--buffer-size", type=int, default=100_000,
                   help="Replay buffer size (default: 100000)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Mini-batch size (default: 256)")
    p.add_argument("--tau", type=float, default=0.005,
                   help="Soft target update tau (default: 0.005)")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor (default: 0.99)")
    p.add_argument("--train-freq", type=int, default=1,
                   help="Update model every N steps (default: 1)")
    p.add_argument("--gradient-steps", type=int, default=1,
                   help="Gradient steps per update (default: 1)")
    p.add_argument("--ent-coef", type=str, default="auto",
                   help="Entropy coefficient ('auto' or float, default: auto)")
    p.add_argument("--net-arch", type=int, nargs="+", default=[256, 256],
                   help="MLP hidden layer sizes (default: 256 256)")
    p.add_argument("--eval-freq", type=int, default=10_000,
                   help="Evaluate every N steps (default: 10000)")
    p.add_argument("--checkpoint-freq", type=int, default=50_000,
                   help="Save checkpoint every N steps (default: 50000)")
    p.add_argument("--eval-episodes", type=int, default=5,
                   help="Episodes per evaluation (default: 5)")
    p.add_argument("--run-dir", type=str, default="runs/sac_simroki",
                   help="Directory for logs & models")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a saved model zip to resume from")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed")
    p.add_argument("--device", type=str, default="auto",
                   help="Torch device: auto, cpu, cuda, mps (default: auto)")
    p.add_argument("--normalize-obs", action="store_true",
                   help="Wrap envs in VecNormalize for observation normalization")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save args for reproducibility
    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[train] Run directory: {run_dir.resolve()}")
    print(f"[train] Envs: {args.num_envs} on ports "
          f"{args.base_port}..{args.base_port + args.num_envs - 1}")
    print(f"[train] Total timesteps: {args.total_timesteps:,}")

    # --- Create environments ---
    train_envs = make_training_envs(args.num_envs, args.base_port, args.repeat_steps)
    if args.normalize_obs:
        train_envs = VecNormalize(train_envs, norm_obs=True, norm_reward=False)

    # Use the last port for evaluation so it doesn't collide with training
    eval_port = args.base_port + args.num_envs - 1
    if args.num_envs > 1:
        # Dedicate the last port to eval if we have more than one env
        eval_port = args.base_port + args.num_envs - 1
    eval_env = make_eval_env(eval_port, args.repeat_steps)
    if args.normalize_obs:
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)

    # --- Parse ent_coef ---
    ent_coef = args.ent_coef
    if ent_coef != "auto":
        ent_coef = float(ent_coef)

    # --- Create or load SAC model ---
    policy_kwargs = dict(net_arch=args.net_arch)

    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        model = SAC.load(
            args.resume,
            env=train_envs,
            device=args.device,
            print_system_info=True,
        )
        model.learning_rate = args.learning_rate
    else:
        model = SAC(
            "MlpPolicy",
            train_envs,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            ent_coef=ent_coef,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device=args.device,
            tensorboard_log=str(run_dir / "tb_logs"),
        )

    print(f"[train] Policy network:\n{model.policy}")

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.num_envs, 1),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="sac_simroki",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=max(args.eval_freq // args.num_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        verbose=1,
    )

    stats_cb = EpisodeStatsCallback(
        log_path=str(run_dir / "episode_stats.json"),
        verbose=1,
    )

    # --- Train ---
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_cb, eval_cb, stats_cb],
            log_interval=10,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[train] Interrupted by user. Saving current model ...")

    elapsed = time.time() - t0
    print(f"[train] Training finished in {elapsed:.0f}s ({elapsed / 3600:.1f}h)")

    # --- Save final model ---
    final_path = str(run_dir / "final_model")
    model.save(final_path)
    print(f"[train] Final model saved to {final_path}.zip")

    if args.normalize_obs and isinstance(train_envs, VecNormalize):
        stats_path = str(run_dir / "vec_normalize.pkl")
        train_envs.save(stats_path)
        print(f"[train] VecNormalize stats saved to {stats_path}")

    # --- Cleanup ---
    train_envs.close()
    eval_env.close()
    print("[train] Done.")


if __name__ == "__main__":
    main()
