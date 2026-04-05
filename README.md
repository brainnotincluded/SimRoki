# SimRoki — 2D Biped Robot Simulator + SAC Walking AI

Desktop-native 2D simulator of a five-link biped robot built on `rapier2d`, with SAC (Soft Actor-Critic) trained walking policy. **Robot + ball cover 100m in under 1 second.**

![Training Progression](docs/charts/01_training_progression.png)

## Quick Start

### 1. Build the simulator

```bash
cargo build --release -p native_app
```

### 2. Run with pre-trained AI

```bash
# Terminal 1: Launch simulator
SIMROKI_PORT=9090 ./target/release/native_app

# Terminal 2: Run the AI controller
pip install stable-baselines3 gymnasium torch numpy requests
python3 robofest_run.py --port 9090
```

Press **"Start ROBOFEST 2026"** in the sim window — the AI takes over and runs 100m with the ball.

### 3. Pre-trained models

Three trained models are included in `models/`:

| Model | Description | Best for |
|-------|-------------|----------|
| `sac_sustained_best.zip` | Fast + stable, velocity reward | **Robofest (recommended)** |
| `sac_stable_walk.zip` | Stable 20s walking | Demos, slow walking |
| `sac_speed_best.zip` | Maximum speed, less stable | Speed records |

Play any model:
```bash
python3 RL/play_best.py --model models/sac_sustained_best.zip --port 9090 --episodes 5 --deterministic
```

---

## SAC Training Results

### From falling in 0.3s to 100m in 0.55s

The project went through **5 iterations of reward engineering** to go from a robot that couldn't stand to one that sprints 100m with a ball:

| Stage | Result | Time |
|-------|--------|------|
| Analytical CPG | Falls in 0.3s | - |
| SAC v1 (stability) | Walks 20s, 23m | 22 min training |
| SAC v2 (speed) | 260m in 9.6s, but falls | 16 min training |
| SAC v3 (ball control) | Keeps ball ahead | 20 min training |
| SAC v5 (sustained) | **100m in 0.55s, ball ahead** | 30 min training |

![All Models Comparison](docs/charts/04_all_models_comparison.png)

### Key insight: reward design matters more than architecture

![Reward Evolution](docs/charts/06_reward_evolution.png)

The biggest challenge wasn't the neural network — it was getting the reward function right:
- **Alive bonus too high** → robot learns to stand still
- **Forward reward only** → robot sprints 1s then crashes
- **Velocity reward (capped)** → robot walks fast AND stays upright

Full training report: [docs/SAC_TRAINING_REPORT.md](docs/SAC_TRAINING_REPORT.md)

---

## Train Your Own Model

### Launch parallel simulators

```bash
# Launch 5 simulators on ports 8080-8084
chmod +x RL/launch_sims.sh
bash RL/launch_sims.sh 5 8080

# Or launch headless (64x64 windows, faster)
SIMROKI_HEADLESS=1 bash RL/launch_sims.sh 5 8080
```

### Train SAC

```bash
# Train from scratch (2M steps, ~30 min on Apple Silicon)
python3 RL/train_sac.py --num-envs 5 --base-port 8080 --total-timesteps 2000000

# Resume from a checkpoint
python3 RL/train_sac.py --num-envs 5 --base-port 8080 --total-timesteps 1000000 \
    --resume models/sac_sustained_best.zip --run-dir runs/my_run

# Stop simulators when done
bash RL/kill_sims.sh
```

### Requirements

```
Python >= 3.10
stable-baselines3 >= 2.7
gymnasium >= 1.0
torch >= 2.0
numpy
requests
tqdm
rich
```

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SIMROKI_PORT` | HTTP API port | `8080` |
| `SIMROKI_HEADLESS` | Minimal window, skip rendering | not set |

---

## Architecture

![Architecture](docs/charts/05_architecture.png)

### Robot

- **5 links**: torso (0.68m), 2 thighs (0.46m), 2 shins (0.50m)
- **4 joints**: right/left hip, right/left knee
- **PID servos**: Kp=20, Ki=3.12, Kd=0.33, max_torque=18.15 N·m
- **Total mass**: 0.852 kg

### SAC Agent

- **Policy**: MLP [256, 256], tanh activation
- **Observation**: 33 features (position, angles, velocities, contacts, ball state)
- **Action**: 4 joint angle offsets in [-1, 1], scaled to ±35°
- **Framework**: stable-baselines3 + PyTorch

### Reward function

```
reward = forward_progress × 5.0
       + ball_progress × 5.0
       + velocity_reward (capped at 5 m/s) × 0.8
       + alive_bonus × 0.1
       + upright_bonus × 0.1
       - torque_penalty
       - action_delta_penalty
       - ball_behind_penalty (if robot ahead of ball)
```

---

## Project Structure

```
SimRoki/
├── sim_core/          # Physics engine, robot model, servo control
├── native_app/        # Desktop GUI (macroquad + axum HTTP server)
├── python-sdk/        # Python SDK for external control
├── RL/
│   ├── gym_env.py     # Gymnasium wrapper over HTTP API
│   ├── train_sac.py   # SAC training with parallel envs
│   ├── play_best.py   # Play trained policy
│   ├── launch_sims.sh # Launch N parallel simulators
│   └── KNP/           # Spiking neural network experiments
├── models/            # Pre-trained SAC weights
│   ├── sac_sustained_best.zip
│   ├── sac_stable_walk.zip
│   └── sac_speed_best.zip
├── docs/
│   ├── SAC_TRAINING_REPORT.md   # Full training report (RU)
│   └── charts/                   # Training visualizations
├── robofest_run.py    # Robofest button integration
├── robot_config.toml  # Robot & reward configuration
└── README.md
```

## Desktop Controls

- `Space`: pause/resume
- `R`: reset robot
- `B`: reset ball
- `F`: follow robot
- `Left/Right`: walk direction
- `S`: stop walking
- Mouse wheel: zoom
- Middle mouse: pan
- **Start ROBOFEST 2026**: reset + wait for API control

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/state` | GET | Full simulation state |
| `/rl/reset` | POST | Reset episode, return observation |
| `/rl/step` | POST | Step simulation with action |
| `/rl/observation` | GET | Current observation vector |
| `/gait` | POST | Send looping gait sequence |
| `/servo/targets` | POST | Set joint targets directly |
| `/walk/direction` | POST | Built-in walk controller |
| `/health` | GET | Health check |

---

*Built for Robofest 2026. SimRoki + SAC (stable-baselines3) + rapier2d.*
