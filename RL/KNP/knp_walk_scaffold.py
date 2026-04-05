from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from desktop_rl_env import DesktopRobotEnv
from knp.base_framework import Model, Network
from knp.core import BLIFATNeuronPopulation, DeltaSynapseProjection


@dataclass
class KnpRuntimeInfo:
    model_type: str
    network_type: str
    neuron_population_type: str
    synapse_type: str


def describe_knp_runtime() -> KnpRuntimeInfo:
    return KnpRuntimeInfo(
        model_type=Model.__name__,
        network_type=Network.__name__,
        neuron_population_type=BLIFATNeuronPopulation.__name__,
        synapse_type=DeltaSynapseProjection.__name__,
    )


def zero_action_rollout(steps: int, repeat_steps: int) -> None:
    env = DesktopRobotEnv(repeat_steps=repeat_steps)
    runtime = describe_knp_runtime()
    print("KNP runtime:")
    print(f"  model={runtime.model_type}")
    print(f"  network={runtime.network_type}")
    print(f"  neurons={runtime.neuron_population_type}")
    print(f"  synapses={runtime.synapse_type}")

    result = env.reset()
    print(f"obs_size={result.observation.shape[0]} action_size=4")
    total_reward = 0.0
    action = np.zeros(4, dtype=np.float32)
    for step_idx in range(steps):
        result = env.step(action)
        total_reward += result.reward
        print(
            f"step={step_idx:04d} reward={result.reward:+.4f} "
            f"done={result.done} truncated={result.truncated}"
        )
        if result.done or result.truncated:
            break
    print(f"total_reward={total_reward:.4f}")
    print("Scaffold ready: next step is wiring KNP spikes -> 4 motor outputs on top of this env.")


def main() -> None:
    parser = argparse.ArgumentParser(description="KNP scaffold check against the desktop robot simulator.")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--repeat-steps", type=int, default=4)
    args = parser.parse_args()
    zero_action_rollout(args.steps, args.repeat_steps)


if __name__ == "__main__":
    main()
