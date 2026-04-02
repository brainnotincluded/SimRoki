# Robot Simulator Python SDK

This folder contains the Python-side tools for controlling the desktop Rust simulator over the local HTTP server.

The simulator itself is desktop-first now. Python control is optional and talks to the running desktop app at `http://127.0.0.1:8080`.

## What is here

- `robot_sim/client.py`: small Python client for the local simulator API
- `robot_sim/cli.py`: command-line tool `simctl`
- `models.py`: pose and gait payload models
- `servo_sliders.py`: simple external slider GUI kept as an optional tool

## Default server

- `http://127.0.0.1:8080`

## Supported CLI commands

- `simctl state`
- `simctl reset`
- `simctl pause`
- `simctl resume`
- `simctl joint set --name right_hip --angle 0.1`
- `simctl pose set --file pose.json`
- `simctl gait send --file gait.json`

The available joint names are:

- `right_hip`
- `right_knee`
- `left_hip`
- `left_knee`

All joint angles sent from Python are in radians.

## Usage

If you want to run without installation:

```powershell
$env:PYTHONPATH="C:\Users\root\Documents\New project\python-sdk"
python -m robot_sim.cli state
```

Examples:

```powershell
$env:PYTHONPATH="C:\Users\root\Documents\New project\python-sdk"
python -m robot_sim.cli joint set --name right_hip --angle -0.2
python -m robot_sim.cli pose set --file pose.json
python -m robot_sim.cli gait send --file gait.json
```

## Python API

```python
from robot_sim import SimulatorClient, Pose, Gait, GaitPhase

client = SimulatorClient()

state = client.get_state()
client.set_joint("right_knee", 1.1)
client.set_pose(
    Pose(
        base_x=0.0,
        base_y=1.0,
        base_yaw=0.0,
        joints={
            "right_hip": -0.15,
            "right_knee": 1.15,
            "left_hip": -0.15,
            "left_knee": 1.15,
        },
    )
)
```

## Payload examples

### Pose

```json
{
  "base": { "x": 0.0, "y": 1.0, "yaw": 0.0 },
  "joints": {
    "right_hip": -0.15,
    "right_knee": 1.15,
    "left_hip": -0.15,
    "left_knee": 1.15
  }
}
```

### Gait

```json
{
  "name": "walk",
  "cycle_s": 0.8,
  "phases": [
    {
      "duration": 0.2,
      "joints": {
        "right_hip": 0.2,
        "right_knee": 1.0,
        "left_hip": -0.1,
        "left_knee": 1.2
      }
    }
  ]
}
```

## Notes

- The desktop app already contains built-in sliders and PID controls, so Python is no longer required for normal manual use.
- External Python commands temporarily take control priority when they are sent to the simulator.
- The simulator owns the actual servo dynamics, joint limits, torque clamping, masses, and link geometry.
- `/state` can be used for logging, analysis, or external control loops.
