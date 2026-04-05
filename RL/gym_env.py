"""Gymnasium-compliant wrapper for the SimRoki 2D biped robot simulator.

Communicates with a running native_app instance over HTTP.
Each env instance can target a different port, enabling parallel training.

Usage::

    env = SimRokiEnv(base_url="http://127.0.0.1:8080")
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
import requests
from gymnasium import spaces


class SimRokiEnv(gym.Env):
    """Gymnasium wrapper around the SimRoki HTTP RL API."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        repeat_steps: int = 4,
        action_limit_deg: float = 35.0,
        timeout_s: float = 5.0,
    ) -> None:
        super().__init__()

        self.base_url = base_url.rstrip("/")
        self.repeat_steps = repeat_steps
        self.action_limit_deg = action_limit_deg
        self.timeout_s = timeout_s
        self.session = requests.Session()

        # Action space: normalized to [-1, 1], scaled to degrees internally
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32,
        )

        # Observation space: probe the sim to get the dimension
        self._observation_names: list[str] = []
        self._action_names: list[str] = []

        # Get obs dimension from the simulator
        try:
            data = self._post("/rl/reset", {})
            obs = self._extract_obs(data)
            obs_dim = obs.shape[0]
        except Exception:
            obs_dim = 24  # fallback

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self._obs_space_ready = True

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)

        payload: dict[str, Any] = {}
        if options and "direction" in options:
            payload["direction"] = float(options["direction"])

        data = self._post("/rl/reset", payload)
        obs = self._extract_obs(data)
        self._ensure_obs_space(obs)

        info = self._build_info(data)
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).flatten()[:4]
        action = np.clip(action, -1.0, 1.0) * self.action_limit_deg  # scale to degrees

        data = self._post(
            "/rl/step",
            {
                "action_deg": [float(v) for v in action],
                "repeat_steps": self.repeat_steps,
            },
        )

        obs = self._extract_obs(data)
        reward = float(data["reward"])
        terminated = bool(data["done"])
        truncated = bool(data["truncated"])
        info = self._build_info(data)

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.session.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _ensure_obs_space(self, obs: np.ndarray) -> None:
        if self._obs_space_ready:
            return
        dim = obs.shape[0]
        self._obs_dim = dim
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
        )
        self._obs_space_ready = True

    def _extract_obs(self, data: dict[str, Any]) -> np.ndarray:
        obs_payload = data["observation"]
        self._observation_names = obs_payload.get("names", [])
        self._action_names = obs_payload.get("action_order", [])
        return np.asarray(obs_payload["values"], dtype=np.float32)

    def _build_info(self, data: dict[str, Any]) -> dict[str, Any]:
        info: dict[str, Any] = {
            "episode_time": data.get("episode_time", 0.0),
            "observation_names": self._observation_names,
            "action_names": self._action_names,
        }
        if "breakdown" in data:
            info["breakdown"] = data["breakdown"]
        if "observation" in data:
            info["raw_observation"] = data["observation"]
        return info

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        for attempt in range(3):
            try:
                resp = self.session.post(url, json=payload, timeout=self.timeout_s)
                resp.raise_for_status()
                return resp.json()
            except Exception:
                if attempt == 2:
                    raise
                import time
                time.sleep(0.5)
                self.session = requests.Session()

    def _get(self, path: str) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()


# ------------------------------------------------------------------
# Factory for vectorised envs (used by train_sac.py)
# ------------------------------------------------------------------


def make_env(port: int, repeat_steps: int = 4) -> callable:
    """Return a thunk that creates a SimRokiEnv bound to the given port."""

    def _init() -> SimRokiEnv:
        return SimRokiEnv(
            base_url=f"http://127.0.0.1:{port}",
            repeat_steps=repeat_steps,
        )

    return _init


# ------------------------------------------------------------------
# Quick self-test
# ------------------------------------------------------------------

if __name__ == "__main__":
    from stable_baselines3.common.env_checker import check_env

    env = SimRokiEnv()
    print("Running SB3 env checker ...")
    check_env(env, warn=True, skip_render_check=True)
    print("Env check passed.")
    obs, info = env.reset()
    print(f"obs shape: {obs.shape}")
    print(f"obs names: {info['observation_names']}")
    for _ in range(5):
        act = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(act)
        print(f"  reward={rew:+.3f}  terminated={term}  truncated={trunc}")
        if term or trunc:
            obs, info = env.reset()
    env.close()
    print("Self-test done.")
