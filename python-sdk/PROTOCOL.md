# Control Protocol v0.1

This document defines the minimal control contract between Python and the Rust simulator.

## Transport

- Commands: HTTP JSON requests.
- State updates: `GET /state` polling is mandatory for the first version.
- Optional live stream can be added later if needed.

## Endpoints

### `GET /state`

Returns the current simulator snapshot.

Example response:

```json
{
  "time": 1.24,
  "mode": "running",
  "base": { "x": 0.0, "y": 1.0, "yaw": 0.0 },
  "joints": {
    "right_hip": 0.0,
    "right_knee": -1.2,
    "left_hip": 0.0,
    "left_knee": -1.2
  }
}
```

### `POST /reset`

Resets the simulation to the initial robot pose.

### `POST /pause`

Pauses physics stepping.

### `POST /resume`

Resumes physics stepping.

### `POST /joint/angle`

Sets a single servo target.

Request:

```json
{ "joint": "right_knee", "angle": -1.2 }
```

### `POST /pose`

Sets the full robot pose in one message.

Request:

```json
{
  "base": { "x": 0.0, "y": 1.0, "yaw": 0.0 },
  "joints": {
    "right_hip": 0.0,
    "right_knee": -1.2,
    "left_hip": 0.0,
    "left_knee": -1.2
  }
}
```

### `POST /gait`

Uploads a gait program. The Rust side should interpret it as a time-ordered sequence of joint targets.

Request:

```json
{
  "name": "walk",
  "cycle_s": 0.8,
  "phases": [
    {
      "duration": 0.2,
      "joints": {
        "right_hip": 0.2,
        "right_knee": -1.4,
        "left_hip": -0.1,
        "left_knee": -1.0
      }
    }
  ]
}
```

## Pose format

`pose` is an instantaneous robot target.

- `base.x`, `base.y`, `base.yaw` describe the floating body target.
- `joints` is a map from joint name to angle in radians.
- Joint names for the minimal robot are `right_hip`, `right_knee`, `left_hip`, `left_knee`.

Recommended defaults:

- `base.y = 1.0`
- knees start near `-1.2` rad
- all angles are in radians

## Gait format

`gait` is a sequence of phases.

- `name` is a free-form label.
- `cycle_s` is the nominal cycle time in seconds.
- Each phase has `duration` and `joints`.
- The Rust simulator should interpolate linearly between phases unless a later version adds splines.

## Servo semantics

- One command equals one target angle.
- The simulator currently uses one simple built-in default servo model for all joints.
- The simulator owns interpolation, joint limits, and motor dynamics.
- Python should stay dumb and deterministic: it sends targets, not low-level physics.
