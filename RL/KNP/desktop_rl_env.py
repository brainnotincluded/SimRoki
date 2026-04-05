from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import requests


@dataclass
class StepResult:
    observation: np.ndarray
    reward: float
    done: bool
    truncated: bool
    episode_time: float
    raw: dict[str, Any]


class DesktopRobotEnv:
    """Visual RL wrapper over the running desktop simulator."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        repeat_steps: int = 4,
        timeout_s: float = 5.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.repeat_steps = repeat_steps
        self.session = requests.Session()
        self.timeout_s = timeout_s
        self.observation_names: list[str] = []
        self.action_names: list[str] = []

    def reset(self) -> StepResult:
        payload = self._post("/rl/reset", {})
        return self._decode_step(payload)

    def reset_with_direction(self, direction: float) -> StepResult:
        payload = self._post("/rl/reset", {"direction": float(direction)})
        return self._decode_step(payload)

    def observation(self) -> np.ndarray:
        payload = self._get("/rl/observation")
        self.observation_names = payload["names"]
        self.action_names = payload["action_order"]
        return np.asarray(payload["values"], dtype=np.float32)

    def step(self, action_deg: np.ndarray | list[float], direction: float | None = None) -> StepResult:
        action = np.asarray(action_deg, dtype=np.float32).reshape(4)
        payload: dict[str, Any] = {
            "action_deg": [float(v) for v in action],
            "repeat_steps": self.repeat_steps,
        }
        if direction is not None:
            payload["direction"] = float(direction)
        payload = self._post(
            "/rl/step",
            payload,
        )
        return self._decode_step(payload)

    def set_walk_direction(self, direction: float, enabled: bool = True) -> dict[str, Any]:
        payload = self._post(
            "/walk/direction",
            {
                "direction": float(direction),
                "enabled": bool(enabled),
            },
        )
        return payload

    def _decode_step(self, payload: dict[str, Any]) -> StepResult:
        observation_payload = payload["observation"]
        self.observation_names = observation_payload["names"]
        self.action_names = observation_payload["action_order"]
        observation = np.asarray(observation_payload["values"], dtype=np.float32)
        return StepResult(
            observation=observation,
            reward=float(payload["reward"]),
            done=bool(payload["done"]),
            truncated=bool(payload["truncated"]),
            episode_time=float(payload["episode_time"]),
            raw=payload,
        )

    def _get(self, path: str) -> dict[str, Any]:
        response = self.session.get(f"{self.base_url}{path}", timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.session.post(f"{self.base_url}{path}", json=payload, timeout=self.timeout_s)
        response.raise_for_status()
        return response.json()
