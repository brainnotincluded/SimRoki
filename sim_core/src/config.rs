use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub physics: PhysicsConfig,
    pub robot: RobotConfig,
    pub servo: ServoConfig,
    #[serde(default)]
    pub rl: RlConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    pub gravity_y: f32,
    pub dt: f32,
    pub ground_half_width: f32,
    pub ground_friction: f32,
    pub ground_restitution: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotConfig {
    pub torso: LinkConfig,
    pub thigh: LinkConfig,
    pub shin: LinkConfig,
    pub body_dynamics: BodyDynamicsConfig,
    pub suspend_clearance: f32,
    pub initial_pose: InitialPoseConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkConfig {
    pub length: f32,
    pub width: f32,
    pub mass: f32,
    pub friction: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyDynamicsConfig {
    pub angular_damping: f32,
    pub linear_damping: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialPoseConfig {
    pub torso: BodyPoseConfig,
    pub left_thigh: BodyPoseConfig,
    pub left_shin: BodyPoseConfig,
    pub right_thigh: BodyPoseConfig,
    pub right_shin: BodyPoseConfig,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BodyPoseConfig {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct JointAnglesConfig {
    pub right_hip: f32,
    pub right_knee: f32,
    pub left_hip: f32,
    pub left_knee: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServoConfig {
    pub kp: f32,
    pub ki: f32,
    pub kd: f32,
    pub max_torque: f32,
    pub integral_limit: f32,
    pub zero_offsets: JointAnglesConfig,
    pub initial_targets: JointAnglesConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlConfig {
    pub control_substeps: u32,
    pub episode_timeout_s: f32,
    pub torso_min_height: f32,
    pub torso_max_tilt_rad: f32,
    pub action_limit_deg: f32,
    pub reward_forward_weight: f32,
    pub reward_alive_bonus: f32,
    pub reward_upright_weight: f32,
    pub reward_height_weight: f32,
    pub reward_contact_weight: f32,
    pub reward_ball_forward_weight: f32,
    pub penalty_torque_weight: f32,
    pub penalty_action_delta_weight: f32,
}

impl Default for RlConfig {
    fn default() -> Self {
        Self {
            control_substeps: 4,
            episode_timeout_s: 20.0,
            torso_min_height: 0.30,
            torso_max_tilt_rad: 1.25,
            action_limit_deg: 180.0,
            reward_forward_weight: 2.0,
            reward_alive_bonus: 0.02,
            reward_upright_weight: 0.15,
            reward_height_weight: 0.10,
            reward_contact_weight: 0.03,
            reward_ball_forward_weight: 3.0,
            penalty_torque_weight: 0.0025,
            penalty_action_delta_weight: 0.0015,
        }
    }
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            physics: PhysicsConfig {
                gravity_y: -9.81,
                dt: 1.0 / 120.0,
                ground_half_width: 5_000.0,
                ground_friction: 1.35,
                ground_restitution: 0.0,
            },
            robot: RobotConfig {
                torso: LinkConfig {
                    length: 0.68,
                    width: 0.09,
                    mass: 0.318_240_02,
                    friction: 1.0,
                },
                thigh: LinkConfig {
                    length: 0.46,
                    width: 0.09,
                    mass: 0.140_760_02,
                    friction: 1.0,
                },
                shin: LinkConfig {
                    length: 0.50,
                    width: 0.09,
                    mass: 0.126,
                    friction: 1.2,
                },
                body_dynamics: BodyDynamicsConfig {
                    angular_damping: 0.85,
                    linear_damping: 0.08,
                },
                suspend_clearance: 0.8,
                initial_pose: InitialPoseConfig {
                    torso: BodyPoseConfig {
                        x: 0.0,
                        y: 1.12,
                        angle: 0.0,
                    },
                    left_thigh: BodyPoseConfig {
                        x: 0.086_619_23,
                        y: 0.566_934,
                        angle: 0.386_127_5,
                    },
                    left_shin: BodyPoseConfig {
                        x: -0.003_380_775,
                        y: 0.176_934_03,
                        angle: -0.784_512_16,
                    },
                    right_thigh: BodyPoseConfig {
                        x: -0.086_619_23,
                        y: 0.566_934,
                        angle: -0.386_127_5,
                    },
                    right_shin: BodyPoseConfig {
                        x: 0.003_380_775,
                        y: 0.176_934_03,
                        angle: 0.784_512_16,
                    },
                },
            },
            servo: ServoConfig {
                kp: 20.0,
                ki: 0.0,
                kd: 0.0,
                max_torque: 10.0,
                integral_limit: 10.0,
                zero_offsets: JointAnglesConfig {
                    right_hip: -0.386_127_5,
                    right_knee: 1.170_639_6,
                    left_hip: 0.386_127_5,
                    left_knee: -1.170_639_6,
                },
                initial_targets: JointAnglesConfig {
                    right_hip: -0.386_127_5,
                    right_knee: 1.170_639_6,
                    left_hip: 0.386_127_5,
                    left_knee: -1.170_639_6,
                },
            },
            rl: RlConfig::default(),
        }
    }
}

impl Default for JointAnglesConfig {
    fn default() -> Self {
        SimulationConfig::default().servo.zero_offsets
    }
}

impl SimulationConfig {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self, String> {
        let path = path.as_ref();
        let text = fs::read_to_string(path)
            .map_err(|err| format!("failed to read config '{}': {err}", path.display()))?;
        toml::from_str(&text)
            .map_err(|err| format!("failed to parse config '{}': {err}", path.display()))
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<(), String> {
        let path = path.as_ref();
        let text = toml::to_string_pretty(self)
            .map_err(|err| format!("failed to serialize config '{}': {err}", path.display()))?;
        fs::write(path, text).map_err(|err| format!("failed to write config '{}': {err}", path.display()))
    }
}
