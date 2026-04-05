#!/usr/bin/env python3
"""
Feedback-based walking controller for SimRoki.
Uses /rl/step for fast physics stepping with sensor feedback.

Key difference from sinusoidal CPG:
- Active torso balance correction (PD on torso angle)
- Asymmetric knee trajectory (stance straight, swing bent)
- Contact-aware phase progression
- Velocity regulation

Based on analysis of the built-in directional walk in sim_core/src/lib.rs:1381
"""
import argparse
import csv
import json
import math
import sys
import time
from urllib import request, error

BASE_URL = "http://127.0.0.1:8080"


def api_post(path, body=None):
    data = json.dumps(body or {}).encode()
    req = request.Request(f"{BASE_URL}{path}", data=data,
                          headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=5) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw else {}


def api_get(path):
    req = request.Request(f"{BASE_URL}{path}", method="GET")
    with request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode())


class WalkController:
    """
    Phase-based walking controller with torso feedback.

    The gait cycle has a continuous phase from 0 to 2π:
    - phase 0→π: right leg stance, left leg swing
    - phase π→2π: left leg stance, right leg swing

    Hip targets follow a sinusoid with torso correction.
    Knee targets are asymmetric: straighter during stance, bent during swing.
    """

    def __init__(self, config=None):
        cfg = config or {}

        # Gait frequency
        self.omega = cfg.get("omega", 4.0)  # rad/s (slightly less than built-in 4.2)

        # Hip parameters (in radians, relative to zero)
        self.hip_amplitude = cfg.get("hip_amplitude", 0.15)  # conservative start
        self.hip_bias = cfg.get("hip_bias", 0.03)  # slight forward lean

        # Knee parameters (relative to zero)
        self.stance_knee = cfg.get("stance_knee", 0.15)  # almost straight during stance
        self.swing_knee = cfg.get("swing_knee", 0.50)   # moderate bend during swing
        self.knee_blend = cfg.get("knee_blend", 0.10)    # transition smoothing

        # Torso feedback gains
        self.torso_kp = cfg.get("torso_kp", 0.20)
        self.torso_kd = cfg.get("torso_kd", 0.08)
        self.torso_max = cfg.get("torso_max", 0.15)  # clamp correction

        # Velocity feedback
        self.target_velocity = cfg.get("target_velocity", 0.3)  # m/s
        self.velocity_gain = cfg.get("velocity_gain", 0.05)
        self.velocity_max = cfg.get("velocity_max", 0.10)

        # Phase state — start at π/2 so right leg is in full stance, left in full swing
        self.phase = math.pi / 2

    def compute_action(self, obs, state=None):
        """
        Compute joint targets given observation.

        Returns action_deg: [right_hip, right_knee, left_hip, left_knee]
        relative to zero offsets, in degrees.
        """
        # Extract feedback signals
        torso_angle = obs.get("torso_angle", 0.0)
        torso_height = obs.get("torso_height", 1.1)

        # Get velocity from state if available
        if state and state.get("base"):
            vx = state["base"]["vx"]
            omega_torso = state["base"]["omega"]
        else:
            vx = 0.0
            omega_torso = 0.0

        # Phase sinusoids
        phase_sin = math.sin(self.phase)       # right leg phase
        anti_sin = -phase_sin                   # left leg phase (180° offset)

        # --- Torso feedback ---
        torso_correction = (
            -self.torso_kp * torso_angle
            - self.torso_kd * omega_torso
        )
        torso_correction = max(-self.torso_max, min(self.torso_max, torso_correction))

        # --- Velocity feedback ---
        vel_correction = self.velocity_gain * (self.target_velocity - vx)
        vel_correction = max(-self.velocity_max, min(self.velocity_max, vel_correction))

        # --- Hip targets (relative to zero, in radians) ---
        right_hip_rel = self.hip_bias + self.hip_amplitude * phase_sin + torso_correction + vel_correction
        left_hip_rel = -self.hip_bias + self.hip_amplitude * anti_sin + torso_correction + vel_correction

        # --- Knee targets (asymmetric stance/swing) ---
        # Right knee: when phase_sin > 0, right leg is in stance (straighter)
        if phase_sin > 0:
            right_knee_rel = self.stance_knee + self.knee_blend * (1.0 - phase_sin)
        else:
            right_knee_rel = self.swing_knee + self.knee_blend * (-phase_sin)

        # Left knee: anti-phase
        if anti_sin > 0:
            left_knee_rel = self.stance_knee + self.knee_blend * (1.0 - anti_sin)
        else:
            left_knee_rel = self.swing_knee + self.knee_blend * (-anti_sin)

        # Convert to degrees for /rl/step
        action_deg = [
            math.degrees(right_hip_rel),
            math.degrees(right_knee_rel),
            math.degrees(left_hip_rel),
            math.degrees(left_knee_rel),
        ]

        return action_deg

    def advance_phase(self, dt):
        self.phase = (self.phase + self.omega * dt) % (2 * math.pi)


def run_walk_test(config, num_steps, substeps, out_file, verbose=True):
    controller = WalkController(config)

    # Disable directional walk, reset properly
    api_post("/walk/direction", {"direction": 1.0, "enabled": False})
    api_post("/resume")
    time.sleep(0.1)

    # Use RL reset for proper episode initialization
    result = api_post("/rl/reset", {"direction": 1.0})
    if verbose:
        obs = result["observation"]
        print(f"After rl/reset: h={obs['torso_height']:.3f} θ={math.degrees(obs['torso_angle']):.1f}°")

    # Settle: hold standing pose for 60 steps (~2s) to let robot reach ground
    if verbose:
        print("Settling robot (holding standing pose)...")
    for i in range(60):
        result = api_post("/rl/step", {"action_deg": [0, 0, 0, 0], "repeat_steps": substeps})
        if result["done"]:
            if verbose:
                print(f"  Robot fell during settling at step {i}")
            break
    obs = result["observation"]
    if verbose:
        print(f"After settling: h={obs['torso_height']:.3f} θ={math.degrees(obs['torso_angle']):.1f}° "
              f"contacts: L={obs['contacts']['left_foot']} R={obs['contacts']['right_foot']}")
    if result["done"]:
        if verbose:
            print("Robot cannot even stand. Aborting.")
        return []

    # Reset RL episode again after settling
    result = api_post("/rl/reset", {"direction": 1.0})
    for i in range(30):
        result = api_post("/rl/step", {"action_deg": [0, 0, 0, 0], "repeat_steps": substeps})
        if result["done"]:
            break
    if verbose:
        obs = result["observation"]
        print(f"Ready: h={obs['torso_height']:.3f} θ={math.degrees(obs['torso_angle']):.1f}° "
              f"contacts: L={obs['contacts']['left_foot']} R={obs['contacts']['right_foot']}")

    dt_per_step = substeps / 120.0  # seconds per control step

    fieldnames = [
        "step", "sim_time", "phase",
        "base_x", "base_y", "base_angle", "base_vx", "base_omega",
        "torso_height", "torso_angle",
        "left_contact", "right_contact",
        "action_rh", "action_rk", "action_lh", "action_lk",
        "torso_correction", "vel_correction",
        "reward", "done",
    ]

    rows = []
    start_wall = time.time()

    if verbose:
        print(f"Walk controller: ω={controller.omega:.1f} hip_amp={math.degrees(controller.hip_amplitude):.1f}° "
              f"stance_knee={math.degrees(controller.stance_knee):.1f}° swing_knee={math.degrees(controller.swing_knee):.1f}°")
        print(f"Torso feedback: Kp={controller.torso_kp} Kd={controller.torso_kd}")
        print(f"Running {num_steps} steps, substeps={substeps}, dt={dt_per_step:.4f}s")

    for step in range(num_steps):
        # Get full state for feedback
        if step % 5 == 0:
            state = api_get("/state")
        else:
            state = None

        # Build observation from last rl_step result or initial state
        if step == 0:
            st = api_get("/state")
            obs = {
                "torso_angle": st["base"]["angle"] if st.get("base") else 0,
                "torso_height": st["base"]["y"] if st.get("base") else 1.1,
                "base_x": st["base"]["x"] if st.get("base") else 0,
            }
            state = st
        else:
            obs = {
                "torso_angle": result["observation"]["torso_angle"],
                "torso_height": result["observation"]["torso_height"],
                "base_x": result["observation"]["base_x"],
            }

        # Compute action
        action_deg = controller.compute_action(obs, state)

        # Step simulation
        result = api_post("/rl/step", {
            "action_deg": action_deg,
            "repeat_steps": substeps,
        })

        # Advance phase
        controller.advance_phase(dt_per_step)

        # Log
        r_obs = result["observation"]
        row = {
            "step": step,
            "sim_time": round(result["episode_time"], 4),
            "phase": round(controller.phase, 4),
            "base_x": round(r_obs["base_x"], 4),
            "torso_height": round(r_obs["torso_height"], 4),
            "torso_angle": round(r_obs["torso_angle"], 4),
            "left_contact": int(r_obs["contacts"]["left_foot"]),
            "right_contact": int(r_obs["contacts"]["right_foot"]),
            "action_rh": round(action_deg[0], 3),
            "action_rk": round(action_deg[1], 3),
            "action_lh": round(action_deg[2], 3),
            "action_lk": round(action_deg[3], 3),
            "reward": round(result["reward"], 5),
            "done": int(result["done"]),
        }

        if state and state.get("base"):
            row["base_y"] = round(state["base"]["y"], 4)
            row["base_vx"] = round(state["base"]["vx"], 4)
            row["base_omega"] = round(state["base"]["omega"], 4)

        rows.append(row)

        if verbose and step % 50 == 0:
            elapsed = time.time() - start_wall
            speedup = result["episode_time"] / elapsed if elapsed > 0 else 0
            print(f"  step {step:5d} | t={result['episode_time']:7.3f}s | "
                  f"h={r_obs['torso_height']:.3f}m | θ={math.degrees(r_obs['torso_angle']):+6.1f}° | "
                  f"x={r_obs['base_x']:+.3f}m | "
                  f"L={r_obs['contacts']['left_foot']} R={r_obs['contacts']['right_foot']} | "
                  f"phase={controller.phase:.2f} | "
                  f"{speedup:.1f}x")

        if result["done"]:
            if verbose:
                print(f"  DONE at step {step}: h={r_obs['torso_height']:.3f}m θ={math.degrees(r_obs['torso_angle']):.1f}°")
            break

    elapsed = time.time() - start_wall

    # Write CSV
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    heights = [r["torso_height"] for r in rows]
    xs = [r["base_x"] for r in rows]
    final_time = rows[-1]["sim_time"]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Steps: {len(rows)}, Sim time: {final_time:.2f}s, Wall: {elapsed:.2f}s ({final_time/elapsed:.1f}x)")
        print(f"Height: min={min(heights):.3f} max={max(heights):.3f} mean={sum(heights)/len(heights):.3f}")
        print(f"Forward: {xs[-1] - xs[0]:+.3f}m")
        print(f"Reward: {sum(r['reward'] for r in rows):+.3f}")
        print(f"Log: {out_file}")

    return rows


# --- Parameter sweep ---
CONFIGS = {
    "conservative": {
        "omega": 3.0,
        "hip_amplitude": 0.06,
        "hip_bias": 0.05,            # more forward lean
        "stance_knee": -0.03,
        "swing_knee": 0.15,
        "knee_blend": 0.03,
        "torso_kp": 0.80,            # much stronger
        "torso_kd": 0.30,
        "torso_max": 0.50,           # allow big corrections (28°)
        "target_velocity": 0.10,
        "velocity_gain": 0.03,
        "velocity_max": 0.08,
    },
    "moderate": {
        "omega": 3.5,
        "hip_amplitude": 0.10,
        "hip_bias": 0.06,
        "stance_knee": -0.05,
        "swing_knee": 0.22,
        "knee_blend": 0.04,
        "torso_kp": 1.00,
        "torso_kd": 0.35,
        "torso_max": 0.50,
        "target_velocity": 0.20,
        "velocity_gain": 0.04,
        "velocity_max": 0.10,
    },
    "aggressive": {
        "omega": 4.0,
        "hip_amplitude": 0.15,
        "hip_bias": 0.08,
        "stance_knee": -0.08,
        "swing_knee": 0.30,
        "knee_blend": 0.06,
        "torso_kp": 1.20,
        "torso_kd": 0.40,
        "torso_max": 0.60,
        "target_velocity": 0.35,
        "velocity_gain": 0.05,
        "velocity_max": 0.12,
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()) + ["all", "custom"], default="moderate")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--out", default=None)
    # Custom config overrides
    parser.add_argument("--omega", type=float)
    parser.add_argument("--hip-amp", type=float)
    parser.add_argument("--stance-knee", type=float)
    parser.add_argument("--swing-knee", type=float)
    parser.add_argument("--torso-kp", type=float)
    parser.add_argument("--torso-kd", type=float)
    args = parser.parse_args()

    if args.config == "all":
        # Run all configs
        for name, cfg in CONFIGS.items():
            print(f"\n{'='*60}")
            print(f"CONFIG: {name}")
            print(f"{'='*60}")
            run_walk_test(cfg, args.steps, args.substeps, f"log_feedback_{name}.csv")
    else:
        cfg = CONFIGS.get(args.config, {})
        # Apply overrides
        if args.omega: cfg["omega"] = args.omega
        if args.hip_amp: cfg["hip_amplitude"] = args.hip_amp
        if args.stance_knee: cfg["stance_knee"] = args.stance_knee
        if args.swing_knee: cfg["swing_knee"] = args.swing_knee
        if args.torso_kp: cfg["torso_kp"] = args.torso_kp
        if args.torso_kd: cfg["torso_kd"] = args.torso_kd

        out = args.out or f"log_feedback_{args.config}.csv"
        run_walk_test(cfg, args.steps, args.substeps, out)
