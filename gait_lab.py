#!/usr/bin/env python3
"""
Headless gait testing lab for SimRoki.
Uses /rl/step for fast physics stepping + full state logging.
Outputs CSV for analysis.

Usage:
  python3 gait_lab.py --gait corrected_gait.json --steps 1000 --out log.csv
  python3 gait_lab.py --gait corrected_gait_v2.json --steps 2000 --out log_v2.csv --substeps 8
"""
import argparse
import csv
import json
import math
import sys
import time
from urllib import request, error


BASE_URL = "http://127.0.0.1:8080"
TIMEOUT = 5.0


def api_get(path):
    req = request.Request(f"{BASE_URL}{path}", method="GET")
    with request.urlopen(req, timeout=TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def api_post(path, body=None):
    data = json.dumps(body or {}).encode()
    req = request.Request(f"{BASE_URL}{path}", data=data,
                          headers={"Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=TIMEOUT) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw else {}


def rl_step(action_deg, repeat_steps=1):
    """Advance simulation by repeat_steps physics steps. Returns full observation."""
    return api_post("/rl/step", {
        "action_deg": action_deg,
        "repeat_steps": repeat_steps,
    })


def get_state():
    return api_get("/state")


def compute_gait_action_deg(gait, sim_time, zeros):
    """Compute target joint angles from gait at given time, return as action_deg (relative to zeros, in degrees)."""
    cycle_s = gait["cycle_s"]
    t = sim_time % cycle_s

    phases = gait["phases"]
    elapsed = 0.0
    phase_idx = 0
    phase_start = 0.0

    for idx, phase in enumerate(phases):
        dur = phase["duration"]
        if t <= elapsed + dur:
            phase_idx = idx
            phase_start = elapsed
            break
        elapsed += dur
    else:
        phase_idx = len(phases) - 1
        phase_start = elapsed - phases[-1]["duration"]

    phase = phases[phase_idx]
    dur = phase["duration"]
    alpha = min(max((t - phase_start) / dur, 0.0), 1.0)

    if phase_idx == 0:
        prev_joints = phases[-1]["joints"]  # wrap around for cycling
    else:
        prev_joints = phases[phase_idx - 1]["joints"]

    cur_joints = phase["joints"]

    # Interpolate to get absolute target radians
    targets_rad = {}
    for jname in ["right_hip", "right_knee", "left_hip", "left_knee"]:
        a = prev_joints.get(jname, 0.0)
        b = cur_joints.get(jname, 0.0)
        targets_rad[jname] = a + alpha * (b - a)

    # Convert to action_deg: offset from zero, in degrees
    joint_order = ["right_hip", "right_knee", "left_hip", "left_knee"]
    action_deg = []
    for jname in joint_order:
        abs_rad = targets_rad[jname]
        zero_rad = zeros[jname]
        offset_rad = abs_rad - zero_rad
        action_deg.append(math.degrees(offset_rad))

    return action_deg, targets_rad


def run_gait_test(gait_file, num_steps, substeps, out_file):
    # Load gait
    with open(gait_file) as f:
        gait = json.load(f)

    print(f"Gait: {gait.get('name', 'unnamed')}, cycle={gait['cycle_s']}s, {len(gait['phases'])} phases")

    # Get zero offsets from sim state
    state = get_state()
    zeros = state["servo_zeros"]
    print(f"Servo zeros: rh={zeros['right_hip']:.4f} rk={zeros['right_knee']:.4f} "
          f"lh={zeros['left_hip']:.4f} lk={zeros['left_knee']:.4f}")

    # Reset
    api_post("/reset")
    time.sleep(0.2)
    api_post("/resume")
    time.sleep(0.1)

    # CSV header
    fieldnames = [
        "step", "sim_time",
        "base_x", "base_y", "base_angle", "base_vx", "base_vy", "base_omega",
        "rh_angle", "rh_target", "rh_torque",
        "rk_angle", "rk_target", "rk_torque",
        "lh_angle", "lh_target", "lh_torque",
        "lk_angle", "lk_target", "lk_torque",
        "left_foot_contact", "right_foot_contact",
        "reward", "done",
        "torso_height", "torso_angle",
        "action_rh_deg", "action_rk_deg", "action_lh_deg", "action_lk_deg",
    ]

    rows = []
    start_wall = time.time()

    print(f"Running {num_steps} steps (substeps={substeps}, dt={1/120:.5f}s)...")
    print(f"Simulated time per step: {substeps/120:.4f}s, total: {num_steps*substeps/120:.2f}s")

    for step in range(num_steps):
        # Get current state for timing
        if step == 0:
            st = get_state()
            sim_time = st["time"]
        else:
            sim_time = result["episode_time"]

        # Compute gait targets
        action_deg, targets_rad = compute_gait_action_deg(gait, sim_time, zeros)

        # Step simulation
        result = rl_step(action_deg, repeat_steps=substeps)
        obs = result["observation"]

        # Get detailed state for joint info
        if step % 10 == 0 or result["done"]:
            st = get_state()
            joints = st.get("joints", {})
        else:
            joints = {}

        row = {
            "step": step,
            "sim_time": round(result["episode_time"], 5),
            "base_x": round(obs.get("base_x", 0), 5),
            "torso_height": round(obs.get("torso_height", 0), 5),
            "torso_angle": round(obs.get("torso_angle", 0), 5),
            "left_foot_contact": int(obs["contacts"]["left_foot"]),
            "right_foot_contact": int(obs["contacts"]["right_foot"]),
            "reward": round(result["reward"], 6),
            "done": int(result["done"]),
            "action_rh_deg": round(action_deg[0], 3),
            "action_rk_deg": round(action_deg[1], 3),
            "action_lh_deg": round(action_deg[2], 3),
            "action_lk_deg": round(action_deg[3], 3),
        }

        # Add base state if available
        if step % 10 == 0 and "base" in st and st["base"]:
            base = st["base"]
            row.update({
                "base_y": round(base["y"], 5),
                "base_angle": round(base["angle"], 5),
                "base_vx": round(base["vx"], 5),
                "base_vy": round(base["vy"], 5),
                "base_omega": round(base["omega"], 5),
            })

        # Add joint details
        for jname, short in [("right_hip", "rh"), ("right_knee", "rk"),
                             ("left_hip", "lh"), ("left_knee", "lk")]:
            if jname in joints:
                j = joints[jname]
                row[f"{short}_angle"] = round(j["angle"], 5)
                row[f"{short}_target"] = round(j["target"], 5)
                row[f"{short}_torque"] = round(j["torque"], 5)

        rows.append(row)

        # Progress
        if step % 100 == 0:
            elapsed_wall = time.time() - start_wall
            speedup = result["episode_time"] / elapsed_wall if elapsed_wall > 0 else 0
            print(f"  step {step:5d} | sim_t={result['episode_time']:7.3f}s | "
                  f"h={obs['torso_height']:.3f}m | θ={math.degrees(obs['torso_angle']):+.1f}° | "
                  f"x={obs['base_x']:+.3f}m | "
                  f"contacts: L={obs['contacts']['left_foot']} R={obs['contacts']['right_foot']} | "
                  f"reward={result['reward']:+.4f} | "
                  f"speedup={speedup:.1f}x")

        if result["done"]:
            print(f"  EPISODE DONE at step {step}, sim_time={result['episode_time']:.3f}s")
            print(f"  Reason: height={obs['torso_height']:.3f}m, angle={math.degrees(obs['torso_angle']):.1f}°")
            break

    elapsed_wall = time.time() - start_wall
    final_sim_time = rows[-1]["sim_time"]
    print(f"\nDone: {len(rows)} steps in {elapsed_wall:.2f}s wall time")
    print(f"Simulated {final_sim_time:.2f}s at {final_sim_time/elapsed_wall:.1f}x realtime")

    # Write CSV
    with open(out_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Log saved to {out_file}")

    # Summary stats
    heights = [r["torso_height"] for r in rows if r.get("torso_height")]
    xs = [r["base_x"] for r in rows if r.get("base_x")]
    if heights:
        print(f"\nTorso height: min={min(heights):.3f} max={max(heights):.3f} mean={sum(heights)/len(heights):.3f}")
    if len(xs) > 1:
        print(f"Forward progress: {xs[-1] - xs[0]:+.3f}m")
    total_reward = sum(r.get("reward", 0) for r in rows)
    print(f"Total reward: {total_reward:+.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless gait testing for SimRoki")
    parser.add_argument("--gait", required=True, help="Path to gait JSON file")
    parser.add_argument("--steps", type=int, default=500, help="Number of control steps")
    parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per control step (speedup)")
    parser.add_argument("--out", default="gait_log.csv", help="Output CSV file")
    args = parser.parse_args()

    run_gait_test(args.gait, args.steps, args.substeps, args.out)
