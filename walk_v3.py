#!/usr/bin/env python3
"""
Walking controller v3 for SimRoki.
Based on the RL baseline CPG pattern from RL/KNP/knp_walk_kick_train.py
with torso feedback and parameter sweep.

Key insight from the RL code: knee uses half-wave rectification max(0, -sin(phase))
— only bends during swing phase, stays at offset during stance.
"""
import argparse
import csv
import json
import math
import sys
import time
from urllib import request


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


class WalkV3:
    """
    RL-baseline-inspired CPG with torso feedback.

    From RL/KNP/knp_walk_kick_train.py:
      right_hip  = hip_offset + hip_amp * sin(phase)
      right_knee = knee_offset + knee_amp * max(0, -sin(phase))
      left_hip   = -hip_offset - hip_amp * sin(phase)
      left_knee  = knee_offset + knee_amp * max(0, sin(phase))

    Key: knee only activates during swing (half-wave). During stance, knee = offset.
    """

    def __init__(self, cfg):
        # CPG parameters (all in degrees, as used by /rl/step)
        self.omega = cfg.get("omega", 3.2)         # rad/s phase speed
        self.hip_amp = cfg.get("hip_amp", 15.0)     # degrees
        self.hip_offset = cfg.get("hip_offset", 4.0)  # forward lean bias, degrees
        self.knee_amp = cfg.get("knee_amp", 12.0)    # degrees
        self.knee_offset = cfg.get("knee_offset", 10.0)  # baseline knee bend, degrees

        # Torso feedback (applied to hip targets, in degrees)
        self.torso_kp = cfg.get("torso_kp", 15.0)   # deg per rad of torso tilt
        self.torso_kd = cfg.get("torso_kd", 5.0)     # deg per rad/s of torso angular velocity
        self.torso_max = cfg.get("torso_max", 20.0)   # max correction in degrees

        # Velocity feedback
        self.vel_target = cfg.get("vel_target", 0.2)  # m/s
        self.vel_gain = cfg.get("vel_gain", 3.0)       # deg per m/s error
        self.vel_max = cfg.get("vel_max", 8.0)         # max correction in degrees

        self.phase = cfg.get("initial_phase", math.pi / 2)  # start with one leg in stance
        self.ramp_steps = cfg.get("ramp_steps", 30)  # steps to ramp up amplitude
        self.step_count = 0

    def compute(self, torso_angle, torso_omega, vx):
        """Returns action_deg [rh, rk, lh, lk] relative to zeros, in degrees."""
        s = math.sin(self.phase)

        # Ramp up amplitude gradually
        ramp = min(1.0, self.step_count / max(1, self.ramp_steps))
        self.step_count += 1

        # CPG base pattern (from RL baseline) with ramp
        rh = self.hip_offset * ramp + self.hip_amp * ramp * s
        rk = self.knee_offset * ramp + self.knee_amp * ramp * max(0.0, -s)
        lh = -self.hip_offset * ramp - self.hip_amp * ramp * s
        lk = self.knee_offset * ramp + self.knee_amp * ramp * max(0.0, s)

        # Torso feedback (stabilization)
        # Reversed sign: when torso leans backward (negative angle), PULL hips backward
        # This creates a counter-torque via Newton's third law that pushes torso forward
        fb = self.torso_kp * torso_angle + self.torso_kd * torso_omega
        fb = max(-self.torso_max, min(self.torso_max, fb))

        # Velocity feedback
        vfb = self.vel_gain * (self.vel_target - vx)
        vfb = max(-self.vel_max, min(self.vel_max, vfb))

        # Apply feedback to both hips equally
        rh += fb + vfb
        lh += fb + vfb

        return [rh, rk, lh, lk]

    def advance(self, dt):
        self.phase = (self.phase + self.omega * dt) % (2 * math.pi)


def settle(substeps=4, steps=60):
    """Reset and let robot settle to standing pose."""
    api_post("/walk/direction", {"direction": 1.0, "enabled": False})
    api_post("/resume")
    time.sleep(0.05)
    result = api_post("/rl/reset", {"direction": 1.0})
    for _ in range(steps):
        result = api_post("/rl/step", {"action_deg": [0, 0, 0, 0], "repeat_steps": substeps})
        if result["done"]:
            return result, False
    return result, True


def run(cfg, num_steps=600, substeps=4, out_file="log.csv", verbose=True):
    ctrl = WalkV3(cfg)

    result, ok = settle(substeps)
    if not ok:
        if verbose:
            print("Robot fell during settling!")
        return []

    obs = result["observation"]
    if verbose:
        print(f"Ready: h={obs['torso_height']:.3f} θ={math.degrees(obs['torso_angle']):.1f}° "
              f"L={obs['contacts']['left_foot']} R={obs['contacts']['right_foot']}")
        print(f"CPG: ω={ctrl.omega:.1f} hip={ctrl.hip_amp:.0f}°±{ctrl.hip_offset:.0f}° "
              f"knee={ctrl.knee_amp:.0f}°+{ctrl.knee_offset:.0f}° "
              f"torso_fb=Kp{ctrl.torso_kp}/Kd{ctrl.torso_kd}")

    # Reset episode for clean timing
    result = api_post("/rl/reset", {"direction": 1.0})
    for _ in range(30):
        result = api_post("/rl/step", {"action_deg": [0, 0, 0, 0], "repeat_steps": substeps})
        if result["done"]:
            break

    dt = substeps / 120.0
    rows = []
    start = time.time()

    for step in range(num_steps):
        obs = result["observation"]
        torso_angle = obs["torso_angle"]
        torso_height = obs["torso_height"]

        # Get velocity from state every few steps
        if step % 5 == 0:
            st = api_get("/state")
            vx = st["base"]["vx"] if st.get("base") else 0
            omega = st["base"]["omega"] if st.get("base") else 0
        else:
            omega = 0  # approximate

        action = ctrl.compute(torso_angle, omega, vx)
        result = api_post("/rl/step", {"action_deg": action, "repeat_steps": substeps})
        ctrl.advance(dt)

        rows.append({
            "step": step,
            "time": round(result["episode_time"], 4),
            "height": round(obs["torso_height"], 4),
            "angle": round(math.degrees(obs["torso_angle"]), 2),
            "x": round(obs["base_x"], 4),
            "lc": int(obs["contacts"]["left_foot"]),
            "rc": int(obs["contacts"]["right_foot"]),
            "rh": round(action[0], 1),
            "rk": round(action[1], 1),
            "lh": round(action[2], 1),
            "lk": round(action[3], 1),
            "reward": round(result["reward"], 4),
            "done": int(result["done"]),
        })

        if verbose and step % 50 == 0:
            wall = time.time() - start
            spd = result["episode_time"] / wall if wall > 0 else 0
            print(f"  {step:4d} t={result['episode_time']:6.2f}s h={torso_height:.3f} "
                  f"θ={math.degrees(torso_angle):+5.1f}° x={obs['base_x']:+.3f} "
                  f"L={obs['contacts']['left_foot']} R={obs['contacts']['right_foot']} "
                  f"act=[{action[0]:+5.1f} {action[1]:+5.1f} {action[2]:+5.1f} {action[3]:+5.1f}] "
                  f"{spd:.0f}x")

        if result["done"]:
            if verbose:
                print(f"  DONE step {step}: h={obs['torso_height']:.3f} θ={math.degrees(obs['torso_angle']):.1f}°")
            break

    wall = time.time() - start
    if verbose:
        hs = [r["height"] for r in rows]
        xs = [r["x"] for r in rows]
        t = rows[-1]["time"]
        print(f"\n{len(rows)} steps, {t:.2f}s sim, {wall:.1f}s wall ({t/wall:.0f}x)")
        print(f"Height: {min(hs):.3f}-{max(hs):.3f}, Forward: {xs[-1]-xs[0]:+.3f}m")
        print(f"Reward: {sum(r['reward'] for r in rows):+.1f}")

    if out_file:
        with open(out_file, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    return rows


# Configs based on RL baseline + research findings
CONFIGS = {
    # Exact RL baseline values
    "rl_baseline": {
        "omega": 3.2,
        "hip_amp": 22.0, "hip_offset": 6.0,
        "knee_amp": 18.0, "knee_offset": 18.0,
        "torso_kp": 15.0, "torso_kd": 5.0, "torso_max": 25.0,
        "vel_target": 0.3, "vel_gain": 3.0,
    },
    # Conservative: smaller amplitudes
    "small": {
        "omega": 3.0,
        "hip_amp": 8.0, "hip_offset": 3.0,
        "knee_amp": 8.0, "knee_offset": 5.0,
        "torso_kp": 18.0, "torso_kd": 6.0, "torso_max": 25.0,
        "vel_target": 0.15, "vel_gain": 3.0,
    },
    # Medium
    "medium": {
        "omega": 3.2,
        "hip_amp": 14.0, "hip_offset": 5.0,
        "knee_amp": 12.0, "knee_offset": 10.0,
        "torso_kp": 16.0, "torso_kd": 5.5, "torso_max": 25.0,
        "vel_target": 0.2, "vel_gain": 3.0,
    },
    # Near natural frequency
    "natural": {
        "omega": 2.8,
        "hip_amp": 10.0, "hip_offset": 4.0,
        "knee_amp": 10.0, "knee_offset": 8.0,
        "torso_kp": 20.0, "torso_kd": 7.0, "torso_max": 30.0,
        "vel_target": 0.15, "vel_gain": 2.5,
    },
    # Forward lean: walking as controlled falling
    "lean": {
        "omega": 3.5,
        "hip_amp": 12.0, "hip_offset": 12.0,  # strong forward lean
        "knee_amp": 10.0, "knee_offset": 8.0,
        "torso_kp": 25.0, "torso_kd": 8.0, "torso_max": 35.0,
        "vel_target": 0.4, "vel_gain": 5.0, "vel_max": 12.0,
        "ramp_steps": 45,
    },
    # Very strong feedback, tiny walk
    "stable": {
        "omega": 2.5,
        "hip_amp": 5.0, "hip_offset": 2.0,
        "knee_amp": 5.0, "knee_offset": 3.0,
        "torso_kp": 40.0, "torso_kd": 15.0, "torso_max": 45.0,
        "vel_target": 0.1, "vel_gain": 4.0, "vel_max": 10.0,
        "ramp_steps": 50,
    },
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()) + ["all"], default="all")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--substeps", type=int, default=4)
    args = parser.parse_args()

    if args.config == "all":
        for name, cfg in CONFIGS.items():
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}")
            run(cfg, args.steps, args.substeps, f"log_v3_{name}.csv")
    else:
        run(CONFIGS[args.config], args.steps, args.substeps, f"log_v3_{args.config}.csv")
