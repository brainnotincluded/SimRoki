#!/usr/bin/env python3
"""
Record servo dynamics from a trained SAC model, find the periodic gait cycle,
and split into 10 consistent motion sequences in SimRoki format.
"""
import csv
import json
import math
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, "RL")
from gym_env import SimRokiEnv
from stable_baselines3 import SAC


def record_episode(model, env, max_steps=3000):
    """Record full servo state for one episode."""
    obs, info = env.reset()
    records = []

    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)

        # Get detailed state
        import requests
        resp = requests.get(f"{env.base_url}/state", timeout=5)
        state = resp.json()

        joints = state.get("joints", {})
        base = state.get("base", {})

        rec = {
            "step": step,
            "time": state.get("time", 0),
            "base_x": base.get("x", 0),
            "base_y": base.get("y", 0),
            "base_angle": base.get("angle", 0),
            "base_vx": base.get("vx", 0),
        }

        for jname in ["right_hip", "right_knee", "left_hip", "left_knee"]:
            j = joints.get(jname, {})
            rec[f"{jname}_angle"] = j.get("angle", 0)
            rec[f"{jname}_target"] = j.get("target", 0)
            rec[f"{jname}_torque"] = j.get("torque", 0)

        # Store the action that was sent (in degrees, relative to zero)
        rec["action_rh"] = float(action[0])
        rec["action_rk"] = float(action[1])
        rec["action_lh"] = float(action[2])
        rec["action_lk"] = float(action[3])

        records.append(rec)

        if term or trunc:
            break

    return records


def find_gait_cycle(records, min_steps=20):
    """Find the dominant gait cycle by autocorrelation on right_hip_target."""
    targets = np.array([r["right_hip_target"] for r in records])
    if len(targets) < min_steps * 2:
        return len(targets) // 2  # fallback

    # Remove trend
    targets = targets - np.linspace(targets[0], targets[-1], len(targets))

    # Autocorrelation
    n = len(targets)
    corr = np.correlate(targets, targets, mode='full')
    corr = corr[n-1:]  # positive lags only
    corr = corr / corr[0]  # normalize

    # Find first peak after the initial decay (skip first min_steps)
    for i in range(min_steps, len(corr) - 1):
        if corr[i] > corr[i-1] and corr[i] > corr[i+1] and corr[i] > 0.3:
            return i

    return min_steps * 2  # fallback


def extract_stable_cycles(records, cycle_len, num_cycles=10, skip_start=20):
    """Extract num_cycles consistent gait cycles from the middle of the recording."""
    available = len(records) - skip_start
    total_needed = num_cycles * cycle_len

    if available < total_needed:
        # Use what we have, reduce num_cycles
        num_cycles = max(1, available // cycle_len)
        total_needed = num_cycles * cycle_len

    # Start from after initial transient
    start = skip_start

    cycles = []
    for i in range(num_cycles):
        cycle_start = start + i * cycle_len
        cycle_end = cycle_start + cycle_len
        if cycle_end > len(records):
            break
        cycles.append(records[cycle_start:cycle_end])

    return cycles


def cycle_to_gait_json(cycle_records, n_phases=12, name="learned_gait"):
    """Convert a cycle of records into SimRoki /gait JSON format.

    Resamples the cycle into n_phases evenly-spaced waypoints.
    Joint values are ABSOLUTE radians (as the /gait endpoint expects).
    """
    n = len(cycle_records)
    dt_total = cycle_records[-1]["time"] - cycle_records[0]["time"]
    if dt_total <= 0:
        dt_total = n * (4 / 120.0)  # fallback: 4 substeps at 120Hz

    phase_duration = dt_total / n_phases

    phases = []
    for p in range(n_phases):
        # Sample point in the cycle
        idx = int((p + 0.5) * n / n_phases)
        idx = min(idx, n - 1)
        rec = cycle_records[idx]

        phases.append({
            "duration": round(phase_duration, 4),
            "joints": {
                "right_hip": round(rec["right_hip_target"], 6),
                "right_knee": round(rec["right_knee_target"], 6),
                "left_hip": round(rec["left_hip_target"], 6),
                "left_knee": round(rec["left_knee_target"], 6),
            }
        })

    return {
        "name": name,
        "cycle_s": round(dt_total, 4),
        "phases": phases,
    }


def cycle_to_motion_sequence(cycle_records):
    """Convert to /motion/sequence_deg format: [[ms, rh_deg, rk_deg, lh_deg, lk_deg], ...]

    Values are ABSOLUTE degrees.
    """
    frames = []
    dt_step = 4 / 120.0  # seconds per step
    ms_per_step = dt_step * 1000

    for rec in cycle_records:
        frames.append([
            round(ms_per_step, 1),
            round(math.degrees(rec["right_hip_target"]), 3),
            round(math.degrees(rec["right_knee_target"]), 3),
            round(math.degrees(rec["left_hip_target"]), 3),
            round(math.degrees(rec["left_knee_target"]), 3),
        ])

    return frames


def main():
    model_path = "runs/sac_150m_v3/best_model/best_model.zip"
    if not Path(model_path).exists():
        # fallback to other models
        for p in ["runs/sac_speed/best_model/best_model.zip",
                   "runs/sac_consistent/best_model/best_model.zip",
                   "runs/sac_simroki/best_model/best_model.zip"]:
            if Path(p).exists():
                model_path = p
                break

    print(f"Loading model: {model_path}")
    model = SAC.load(model_path)
    env = SimRokiEnv(base_url="http://127.0.0.1:9090", repeat_steps=4)

    # Record a long episode
    print("Recording episode...")
    records = record_episode(model, env, max_steps=3000)
    print(f"Recorded {len(records)} steps ({records[-1]['time'] - records[0]['time']:.1f}s)")

    # Save raw recording
    raw_csv = "runs/servo_dynamics.csv"
    Path("runs").mkdir(exist_ok=True)
    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Raw dynamics saved to {raw_csv}")

    # Find gait cycle
    cycle_len = find_gait_cycle(records)
    cycle_time = cycle_len * 4 / 120.0
    print(f"Detected gait cycle: {cycle_len} steps = {cycle_time:.3f}s")

    # Extract 10 consistent cycles
    cycles = extract_stable_cycles(records, cycle_len, num_cycles=10)
    print(f"Extracted {len(cycles)} cycles of {cycle_len} steps each")

    # Convert each to both formats
    output_dir = Path("runs/motion_sequences")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, cycle in enumerate(cycles):
        # Gait format (for /gait endpoint)
        gait = cycle_to_gait_json(cycle, n_phases=12, name=f"learned_cycle_{i+1}")
        gait_path = output_dir / f"gait_cycle_{i+1:02d}.json"
        with open(gait_path, "w") as f:
            json.dump(gait, f, indent=2)

        # Motion sequence format (for /motion/sequence_deg)
        motion = cycle_to_motion_sequence(cycle)
        motion_path = output_dir / f"motion_cycle_{i+1:02d}.json"
        with open(motion_path, "w") as f:
            json.dump(motion, f, indent=2)

        # Print summary
        rh_targets = [r["right_hip_target"] for r in cycle]
        rk_targets = [r["right_knee_target"] for r in cycle]
        print(f"  Cycle {i+1}: {len(cycle)} steps, "
              f"rh=[{min(rh_targets):.3f}, {max(rh_targets):.3f}] "
              f"rk=[{min(rk_targets):.3f}, {max(rk_targets):.3f}]")

    # Also create an averaged cycle (mean of all cycles)
    if len(cycles) > 1:
        min_len = min(len(c) for c in cycles)
        avg_cycle = []
        for step_idx in range(min_len):
            avg_rec = {}
            for key in cycles[0][0]:
                vals = [cycles[ci][step_idx].get(key, 0) for ci in range(len(cycles))]
                if isinstance(vals[0], (int, float)):
                    avg_rec[key] = sum(vals) / len(vals)
                else:
                    avg_rec[key] = vals[0]
            avg_cycle.append(avg_rec)

        avg_gait = cycle_to_gait_json(avg_cycle, n_phases=12, name="learned_avg_cycle")
        avg_path = output_dir / "gait_averaged.json"
        with open(avg_path, "w") as f:
            json.dump(avg_gait, f, indent=2)

        avg_motion = cycle_to_motion_sequence(avg_cycle)
        avg_motion_path = output_dir / "motion_averaged.json"
        with open(avg_motion_path, "w") as f:
            json.dump(avg_motion, f, indent=2)

        print(f"\n  Averaged cycle saved to {avg_path}")

    print(f"\nAll files in {output_dir}/")
    print(f"  gait_cycle_XX.json   → POST to /gait")
    print(f"  motion_cycle_XX.json → POST to /motion/sequence_deg")
    print(f"  gait_averaged.json   → best for consistent replay")

    env.close()


if __name__ == "__main__":
    main()
