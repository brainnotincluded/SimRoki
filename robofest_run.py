#!/usr/bin/env python3
"""
Robofest runner — watches the sim, waits for ROBOFEST button press,
then runs the best SAC model. Repeatable — press the button again to restart.

Usage: python3 robofest_run.py [--port 9090]
"""
import sys, time, json, argparse
from urllib import request as req
from pathlib import Path

sys.path.insert(0, "RL")
from stable_baselines3 import SAC

BASE = "http://127.0.0.1:9090"


def api_get(path):
    with req.urlopen(f"{BASE}{path}", timeout=5) as r:
        return json.loads(r.read())


def api_post(path, body=None):
    data = json.dumps(body or {}).encode()
    r = req.Request(f"{BASE}{path}", data=data,
                    headers={"Content-Type": "application/json"}, method="POST")
    with req.urlopen(r, timeout=5) as resp:
        raw = resp.read().decode()
        return json.loads(raw) if raw else {}


def rl_step(action_deg):
    return api_post("/rl/step", {
        "action_deg": [float(v) for v in action_deg],
        "repeat_steps": 4,
    })


def get_obs_from_result(result):
    import numpy as np
    return np.asarray(result["observation"]["values"], dtype=np.float32)


def run_once(model):
    """Wait for robofest button, then run the model."""
    print("\nWaiting for ROBOFEST button press (press Start ROBOFEST 2026)...")

    # Wait until robot is far from origin (previous run ended)
    # or already at origin (fresh start)
    was_away = False
    while True:
        try:
            s = api_get("/state")
            base = s.get("base", {})
            x = base.get("x", 999)
            y = base.get("y", 0)

            if abs(x) > 5.0:
                was_away = True

            # Detect reset: robot snapped back to origin
            if abs(x) < 1.5 and y > 0.8 and not s.get("paused", True):
                if was_away:
                    print("ROBOFEST reset detected! GO!")
                    break
                else:
                    # First run — robot starts at origin
                    print("Robot at start. GO!")
                    break
        except:
            pass
        time.sleep(0.1)

    # Run the model — DON'T call /rl/reset, use the robofest-reset state
    # Just get initial observation
    result = api_post("/rl/step", {"action_deg": [0, 0, 0, 0], "repeat_steps": 1})
    obs = get_obs_from_result(result)

    import numpy as np
    start = time.time()
    step = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = np.clip(action, -1.0, 1.0) * 35.0  # scale to degrees

        result = rl_step(action)
        obs = get_obs_from_result(result)
        step += 1

        if step % 25 == 0:
            try:
                s = api_get("/state")
                rx = s["base"]["x"] if s.get("base") else 0
                bx = s["ball"]["x"] if s.get("ball") else 0
                t = time.time() - start
                print(f"  t={t:5.1f}s  robot={rx:6.1f}m  ball={bx:6.1f}m")
                if rx >= 100 and bx >= 100:
                    print(f"\n  === 100m DONE in {t:.2f}s! ===\n")
                    return
            except:
                pass

        if result.get("done") or result.get("truncated"):
            t = time.time() - start
            print(f"  Episode ended at {t:.1f}s")
            return

        if time.time() - start > 120:
            print("  Timeout 120s")
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--model", default="runs/sac_sustained/best_model/best_model.zip")
    args = parser.parse_args()

    global BASE
    BASE = f"http://127.0.0.1:{args.port}"

    print(f"Loading model: {args.model}")
    model = SAC.load(args.model)
    print(f"Sim: {BASE}")
    print("Press START ROBOFEST 2026 in the sim window. Repeatable.\n")

    while True:
        try:
            run_once(model)
        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)


if __name__ == "__main__":
    main()
