use nalgebra::{point, vector};
use rapier2d::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const GRAVITY_Y: f32 = -9.81;
const DT: f32 = 1.0 / 120.0;
const SERVO_KP_DEFAULT: f32 = 20.0;
const SERVO_KI_DEFAULT: f32 = 0.0;
const SERVO_KD_DEFAULT: f32 = 0.0;
const SERVO_MAX_TORQUE_DEFAULT: f32 = 10.0;
const TORSO_UPRIGHT_KP: f32 = 0.0;
const TORSO_UPRIGHT_KD: f32 = 0.0;
const GROUND_HALF_WIDTH: f32 = 5_000.0;
const LEG_HALF_WIDTH: f32 = 0.045;
const THIGH_HALF_HEIGHT: f32 = 0.23;
const SHIN_HALF_HEIGHT: f32 = 0.25;
const TORSO_HALF_WIDTH: f32 = 0.045;
const TORSO_HALF_HEIGHT: f32 = 0.34;
const SUSPEND_CLEARANCE: f32 = 0.45;
const TORSO_DENSITY: f32 = 5.2;
const THIGH_DENSITY: f32 = 3.4;
const SHIN_DENSITY: f32 = 2.8;
const GROUP_GROUND: Group = Group::GROUP_1;
const GROUP_ROBOT: Group = Group::GROUP_2;
const GROUP_BALL: Group = Group::GROUP_3;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SceneKind {
    Ball,
    Robot,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash, Ord, PartialOrd)]
#[serde(rename_all = "snake_case")]
pub enum JointName {
    RightHip,
    RightKnee,
    LeftHip,
    LeftKnee,
}

impl JointName {
    pub const ALL: [JointName; 4] = [
        JointName::RightHip,
        JointName::RightKnee,
        JointName::LeftHip,
        JointName::LeftKnee,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            JointName::RightHip => "right_hip",
            JointName::RightKnee => "right_knee",
            JointName::LeftHip => "left_hip",
            JointName::LeftKnee => "left_knee",
        }
    }
}

impl std::str::FromStr for JointName {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "right_hip" => Ok(Self::RightHip),
            "right_knee" => Ok(Self::RightKnee),
            "left_hip" => Ok(Self::LeftHip),
            "left_knee" => Ok(Self::LeftKnee),
            _ => Err(format!("unknown joint '{value}'")),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BodyState {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub vx: f32,
    pub vy: f32,
    pub omega: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointState {
    pub angle: f32,
    pub velocity: f32,
    pub target: f32,
    pub torque: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContactState {
    pub left_foot: bool,
    pub right_foot: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationState {
    pub time: f32,
    pub mode: String,
    pub scene: SceneKind,
    pub paused: bool,
    pub base: Option<BodyState>,
    pub ball: Option<BodyState>,
    pub joints: BTreeMap<String, JointState>,
    pub link_masses: BTreeMap<String, f32>,
    pub link_lengths: BTreeMap<String, f32>,
    pub contacts: ContactState,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServoTargets {
    pub right_hip: Option<f32>,
    pub right_knee: Option<f32>,
    pub left_hip: Option<f32>,
    pub left_knee: Option<f32>,
}

impl ServoTargets {
    pub fn get(&self, joint: JointName) -> Option<f32> {
        match joint {
            JointName::RightHip => self.right_hip,
            JointName::RightKnee => self.right_knee,
            JointName::LeftHip => self.left_hip,
            JointName::LeftKnee => self.left_knee,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointCommand {
    pub joint: JointName,
    pub angle: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoseCommand {
    pub base: Option<PoseBase>,
    pub joints: BTreeMap<String, f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PoseBase {
    pub x: f32,
    pub y: f32,
    pub yaw: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaitCommand {
    pub name: String,
    pub cycle_s: f32,
    pub phases: Vec<GaitPhase>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaitPhase {
    pub duration: f32,
    pub joints: BTreeMap<String, f32>,
}

#[derive(Debug, Clone)]
struct JointServo {
    parent: RigidBodyHandle,
    child: RigidBodyHandle,
    target: f32,
    last_torque: f32,
    integral_error: f32,
}

#[derive(Debug, Clone)]
struct RobotHandles {
    torso: RigidBodyHandle,
    left_thigh: RigidBodyHandle,
    left_shin: RigidBodyHandle,
    left_shin_collider: ColliderHandle,
    right_thigh: RigidBodyHandle,
    right_shin: RigidBodyHandle,
    right_shin_collider: ColliderHandle,
    servos: BTreeMap<JointName, JointServo>,
}

#[derive(Debug, Clone)]
struct GaitRuntime {
    phases: Vec<GaitPhase>,
    cycle_s: f32,
    elapsed: f32,
}

pub struct Simulation {
    pipeline: PhysicsPipeline,
    gravity: Vector<Real>,
    integration_parameters: IntegrationParameters,
    island_manager: IslandManager,
    broad_phase: BroadPhaseMultiSap,
    narrow_phase: NarrowPhase,
    bodies: RigidBodySet,
    colliders: ColliderSet,
    impulse_joints: ImpulseJointSet,
    multibody_joints: MultibodyJointSet,
    ccd_solver: CCDSolver,
    query_pipeline: QueryPipeline,
    ball: Option<RigidBodyHandle>,
    robot: Option<RobotHandles>,
    scene: SceneKind,
    time: f32,
    paused: bool,
    accumulator: f32,
    gait: Option<GaitRuntime>,
    robot_suspended: bool,
    servo_kp: f32,
    servo_ki: f32,
    servo_kd: f32,
    servo_max_torque: f32,
}

impl Default for Simulation {
    fn default() -> Self {
        Self::new_robot()
    }
}

impl Simulation {
    pub fn new_ball() -> Self {
        let mut sim = Self::empty(SceneKind::Ball);
        sim.spawn_ground();

        let body = RigidBodyBuilder::dynamic()
            .translation(vector![0.0, 5.0])
            .additional_mass(1.0)
            .build();
        let handle = sim.bodies.insert(body);
        let collider = ColliderBuilder::ball(0.5)
            .collision_groups(InteractionGroups::new(
                GROUP_BALL,
                GROUP_GROUND | GROUP_BALL,
            ))
            .friction(0.85)
            .restitution(0.1)
            .build();
        sim.colliders
            .insert_with_parent(collider, handle, &mut sim.bodies);
        sim.ball = Some(handle);
        sim
    }

    pub fn new_robot() -> Self {
        Self::new_robot_with_suspension(false)
    }

    pub fn new_robot_with_suspension(suspended: bool) -> Self {
        let mut sim = Self::empty(SceneKind::Robot);
        sim.robot_suspended = suspended;
        sim.spawn_ground();
        sim.spawn_robot();
        sim
    }

    fn empty(scene: SceneKind) -> Self {
        let mut integration_parameters = IntegrationParameters::default();
        integration_parameters.dt = DT;

        Self {
            pipeline: PhysicsPipeline::new(),
            gravity: vector![0.0, GRAVITY_Y],
            integration_parameters,
            island_manager: IslandManager::new(),
            broad_phase: BroadPhaseMultiSap::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            query_pipeline: QueryPipeline::new(),
            ball: None,
            robot: None,
            scene,
            time: 0.0,
            paused: false,
            accumulator: 0.0,
            gait: None,
            robot_suspended: false,
            servo_kp: SERVO_KP_DEFAULT,
            servo_ki: SERVO_KI_DEFAULT,
            servo_kd: SERVO_KD_DEFAULT,
            servo_max_torque: SERVO_MAX_TORQUE_DEFAULT,
        }
    }

    fn spawn_ground(&mut self) {
        let ground = self.bodies.insert(
            RigidBodyBuilder::fixed()
                .translation(vector![0.0, -0.1])
                .build(),
        );
        self.colliders.insert_with_parent(
            ColliderBuilder::cuboid(GROUND_HALF_WIDTH, 0.1)
                .collision_groups(InteractionGroups::new(
                    GROUP_GROUND,
                    GROUP_GROUND | GROUP_ROBOT | GROUP_BALL,
                ))
                .friction(1.35)
                .restitution(0.0)
                .build(),
            ground,
            &mut self.bodies,
        );
    }

    fn spawn_robot(&mut self) {
        let (torso, _) = self.spawn_box(
            vector![0.0, 1.08],
            TORSO_HALF_WIDTH,
            TORSO_HALF_HEIGHT,
            TORSO_DENSITY,
            1.0,
        );
        let (left_thigh, _) = self.spawn_box(
            vector![-0.08, 0.63],
            LEG_HALF_WIDTH,
            THIGH_HALF_HEIGHT,
            THIGH_DENSITY,
            1.0,
        );
        let (left_shin, left_shin_collider) = self.spawn_box(
            vector![-0.17, 0.14],
            LEG_HALF_WIDTH,
            SHIN_HALF_HEIGHT,
            SHIN_DENSITY,
            1.2,
        );
        let (right_thigh, _) = self.spawn_box(
            vector![0.08, 0.63],
            LEG_HALF_WIDTH,
            THIGH_HALF_HEIGHT,
            THIGH_DENSITY,
            1.0,
        );
        let (right_shin, right_shin_collider) = self.spawn_box(
            vector![0.17, 0.14],
            LEG_HALF_WIDTH,
            SHIN_HALF_HEIGHT,
            SHIN_DENSITY,
            1.2,
        );

        self.insert_revolute(
            torso,
            left_thigh,
            point![0.0, -TORSO_HALF_HEIGHT],
            point![0.0, THIGH_HALF_HEIGHT],
        );
        self.insert_revolute(left_thigh, left_shin, point![0.0, -THIGH_HALF_HEIGHT], point![0.0, SHIN_HALF_HEIGHT]);
        self.insert_revolute(
            torso,
            right_thigh,
            point![0.0, -TORSO_HALF_HEIGHT],
            point![0.0, THIGH_HALF_HEIGHT],
        );
        self.insert_revolute(right_thigh, right_shin, point![0.0, -THIGH_HALF_HEIGHT], point![0.0, SHIN_HALF_HEIGHT]);

        if self.robot_suspended {
            let anchor = self.bodies.insert(
                RigidBodyBuilder::fixed()
                    .translation(vector![0.0, 1.08 + TORSO_HALF_HEIGHT + SUSPEND_CLEARANCE])
                    .build(),
            );
            let hang_joint = RevoluteJointBuilder::new()
                .local_anchor1(point![0.0, 0.0])
                .local_anchor2(point![0.0, TORSO_HALF_HEIGHT])
                .contacts_enabled(false)
                .build();
            self.impulse_joints.insert(anchor, torso, hang_joint, true);
        }

        let mut servos = BTreeMap::new();
        servos.insert(JointName::RightHip, JointServo::new(torso, right_thigh, -0.15));
        servos.insert(JointName::RightKnee, JointServo::new(right_thigh, right_shin, 1.15));
        servos.insert(JointName::LeftHip, JointServo::new(torso, left_thigh, -0.15));
        servos.insert(JointName::LeftKnee, JointServo::new(left_thigh, left_shin, 1.15));

        self.robot = Some(RobotHandles {
            torso,
            left_thigh,
            left_shin,
            left_shin_collider,
            right_thigh,
            right_shin,
            right_shin_collider,
            servos,
        });
    }

    fn spawn_box(
        &mut self,
        translation: Vector<Real>,
        half_x: f32,
        half_y: f32,
        density: f32,
        friction: f32,
    ) -> (RigidBodyHandle, ColliderHandle) {
        let handle = self.bodies.insert(
            RigidBodyBuilder::dynamic()
                .translation(translation)
                .angular_damping(0.85)
                .linear_damping(0.08)
                .build(),
        );
        let collider = self.colliders.insert_with_parent(
            ColliderBuilder::cuboid(half_x, half_y)
                .collision_groups(InteractionGroups::new(
                    GROUP_ROBOT,
                    GROUP_GROUND | GROUP_ROBOT,
                ))
                .density(density)
                .friction(friction)
                .restitution(0.02)
                .build(),
            handle,
            &mut self.bodies,
        );
        (handle, collider)
    }

    fn insert_revolute(
        &mut self,
        parent: RigidBodyHandle,
        child: RigidBodyHandle,
        parent_anchor: Point<Real>,
        child_anchor: Point<Real>,
    ) {
        let joint = RevoluteJointBuilder::new()
            .local_anchor1(parent_anchor)
            .local_anchor2(child_anchor)
            .contacts_enabled(false)
            .build();
        self.impulse_joints.insert(parent, child, joint, true);
    }

    pub fn reset_ball(&mut self) {
        *self = Self::new_ball();
    }

    pub fn reset_robot(&mut self) {
        let targets = self.current_targets();
        let gains = self.servo_gains();
        *self = Self::new_robot_with_suspension(self.robot_suspended);
        self.set_servo_gains(gains.0, gains.1, gains.2, gains.3);
        self.apply_targets(targets);
    }

    pub fn pause(&mut self) {
        self.paused = true;
    }

    pub fn resume(&mut self) {
        self.paused = false;
    }

    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
    }

    pub fn set_scene(&mut self, scene: SceneKind) {
        *self = match scene {
            SceneKind::Ball => Self::new_ball(),
            SceneKind::Robot => Self::new_robot_with_suspension(self.robot_suspended),
        };
    }

    pub fn set_robot_suspended(&mut self, suspended: bool) {
        let targets = self.current_targets();
        let gains = self.servo_gains();
        *self = Self::new_robot_with_suspension(suspended);
        self.set_servo_gains(gains.0, gains.1, gains.2, gains.3);
        self.apply_targets(targets);
    }

    pub fn robot_suspended(&self) -> bool {
        self.robot_suspended
    }

    pub fn set_servo_gains(&mut self, kp: f32, ki: f32, kd: f32, max_torque: f32) {
        self.servo_kp = kp.clamp(-20.0, 20.0);
        self.servo_ki = ki.clamp(-5.0, 5.0);
        self.servo_kd = kd.clamp(-1.0, 1.0);
        self.servo_max_torque = max_torque.clamp(0.5, 100.0);
    }

    pub fn servo_gains(&self) -> (f32, f32, f32, f32) {
        (self.servo_kp, self.servo_ki, self.servo_kd, self.servo_max_torque)
    }

    pub fn set_joint_target(&mut self, joint: JointName, angle: f32) {
        if let Some(robot) = &mut self.robot {
            if let Some(servo) = robot.servos.get_mut(&joint) {
                servo.target = clamp_joint_target(joint, angle);
            }
        }
    }

    pub fn apply_targets(&mut self, targets: ServoTargets) {
        for joint in JointName::ALL {
            if let Some(target) = targets.get(joint) {
                self.set_joint_target(joint, target);
            }
        }
    }

    pub fn apply_pose(&mut self, pose: PoseCommand) {
        let mut targets = ServoTargets::default();
        targets.right_hip = pose.joints.get("right_hip").copied();
        targets.right_knee = pose.joints.get("right_knee").copied();
        targets.left_hip = pose.joints.get("left_hip").copied();
        targets.left_knee = pose.joints.get("left_knee").copied();
        self.apply_targets(targets);
    }

    pub fn set_gait(&mut self, gait: GaitCommand) {
        self.gait = Some(GaitRuntime {
            phases: gait.phases,
            cycle_s: gait.cycle_s.max(DT),
            elapsed: 0.0,
        });
    }

    pub fn clear_gait(&mut self) {
        self.gait = None;
    }

    pub fn step_for_seconds(&mut self, frame_dt: f32) {
        self.accumulator += frame_dt.max(0.0);
        while self.accumulator >= DT {
            self.step_fixed();
            self.accumulator -= DT;
        }
    }

    pub fn step_fixed(&mut self) {
        if self.paused {
            return;
        }

        if self.scene == SceneKind::Robot {
            self.apply_gait_targets();
            self.apply_servo_forces();
        }

        self.pipeline.step(
            &self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            Some(&mut self.query_pipeline),
            &(),
            &(),
        );

        self.time += DT;
    }

    fn apply_gait_targets(&mut self) {
        let Some(gait) = &mut self.gait else {
            return;
        };
        if gait.phases.is_empty() {
            return;
        }

        gait.elapsed = (gait.elapsed + DT) % gait.cycle_s;
        let mut elapsed = 0.0;
        let mut phase_joints = gait.phases[0].joints.clone();
        for candidate in &gait.phases {
            elapsed += candidate.duration.max(DT);
            phase_joints = candidate.joints.clone();
            if gait.elapsed <= elapsed {
                break;
            }
        }
        let _ = gait;
        self.apply_pose(PoseCommand {
            base: None,
            joints: phase_joints,
        });
    }

    fn apply_servo_forces(&mut self) {
        let Some(robot) = &mut self.robot else {
            return;
        };

        for servo in robot.servos.values_mut() {
            let (rel_angle, rel_velocity) = match (
                self.bodies.get(servo.parent),
                self.bodies.get(servo.child),
            ) {
                (Some(parent), Some(child)) => (
                    normalize_angle(child.rotation().angle() - parent.rotation().angle()),
                    child.angvel() - parent.angvel(),
                ),
                _ => continue,
            };

            let error = normalize_angle(servo.target - rel_angle);
            servo.integral_error = (servo.integral_error + error * DT).clamp(-10.0, 10.0);
            let torque = (self.servo_kp * error
                + self.servo_ki * servo.integral_error
                - self.servo_kd * rel_velocity)
                .clamp(-self.servo_max_torque, self.servo_max_torque);
            servo.last_torque = torque;
            let impulse = torque * DT;

            if let Some(parent) = self.bodies.get_mut(servo.parent) {
                parent.apply_torque_impulse(-impulse, true);
            }
            if let Some(child) = self.bodies.get_mut(servo.child) {
                child.apply_torque_impulse(impulse, true);
            }
        }

        if let Some(torso) = self.bodies.get(robot.torso) {
            let upright_torque =
                (-TORSO_UPRIGHT_KP * torso.rotation().angle() - TORSO_UPRIGHT_KD * torso.angvel())
                    .clamp(-self.servo_max_torque, self.servo_max_torque);
            let impulse = upright_torque * DT;
            if let Some(torso) = self.bodies.get_mut(robot.torso) {
                torso.apply_torque_impulse(impulse, true);
            }
        }
    }

    pub fn state(&self) -> SimulationState {
        let base = self.robot.as_ref().and_then(|robot| self.body_state(robot.torso));
        let ball = self.ball.and_then(|handle| self.body_state(handle));
        let mut joints = BTreeMap::new();
        let mut link_masses = BTreeMap::new();
        let mut link_lengths = BTreeMap::new();

        if let Some(robot) = &self.robot {
            for (name, servo) in &robot.servos {
                joints.insert(
                    name.as_str().to_owned(),
                    JointState {
                        angle: self.relative_angle(servo.parent, servo.child).unwrap_or_default(),
                        velocity: self.relative_velocity(servo.parent, servo.child).unwrap_or_default(),
                        target: servo.target,
                        torque: servo.last_torque,
                    },
                );
            }

            for (name, handle) in [
                ("torso", robot.torso),
                ("left_thigh", robot.left_thigh),
                ("left_shin", robot.left_shin),
                ("right_thigh", robot.right_thigh),
                ("right_shin", robot.right_shin),
            ] {
                if let Some(body) = self.bodies.get(handle) {
                    link_masses.insert(name.to_owned(), body.mass());
                }
            }

            link_lengths.insert("torso".to_owned(), TORSO_HALF_HEIGHT * 2.0);
            link_lengths.insert("left_thigh".to_owned(), THIGH_HALF_HEIGHT * 2.0);
            link_lengths.insert("left_shin".to_owned(), SHIN_HALF_HEIGHT * 2.0);
            link_lengths.insert("right_thigh".to_owned(), THIGH_HALF_HEIGHT * 2.0);
            link_lengths.insert("right_shin".to_owned(), SHIN_HALF_HEIGHT * 2.0);
        }

        SimulationState {
            time: self.time,
            mode: if self.paused { "paused" } else { "running" }.to_owned(),
            scene: self.scene,
            paused: self.paused,
            base,
            ball,
            joints,
            link_masses,
            link_lengths,
            contacts: ContactState {
                left_foot: self.foot_contact(true),
                right_foot: self.foot_contact(false),
            },
        }
    }

    pub fn robot_segments(&self) -> Vec<([f32; 2], [f32; 2])> {
        let Some(robot) = &self.robot else {
            return Vec::new();
        };
        let torso_top = self.body_point(robot.torso, point![0.0, TORSO_HALF_HEIGHT]);
        let pelvis = self.body_point(robot.torso, point![0.0, -TORSO_HALF_HEIGHT]);
        let left_knee = self.body_point(robot.left_thigh, point![0.0, -THIGH_HALF_HEIGHT]);
        let left_foot = self.body_point(robot.left_shin, point![0.0, -SHIN_HALF_HEIGHT]);
        let right_knee = self.body_point(robot.right_thigh, point![0.0, -THIGH_HALF_HEIGHT]);
        let right_foot = self.body_point(robot.right_shin, point![0.0, -SHIN_HALF_HEIGHT]);

        let points = [torso_top, pelvis, left_knee, left_foot, right_knee, right_foot];
        if points.iter().any(Option::is_none) {
            return Vec::new();
        }
        let points: Vec<[f32; 2]> = points.into_iter().flatten().collect();
        vec![
            (points[1], points[0]),
            (points[1], points[2]),
            (points[2], points[3]),
            (points[1], points[4]),
            (points[4], points[5]),
        ]
    }

    pub fn joint_markers(&self) -> BTreeMap<String, [f32; 2]> {
        let Some(robot) = &self.robot else {
            return BTreeMap::new();
        };
        let mut markers = BTreeMap::new();
        if let Some(pelvis) = self.body_point(robot.torso, point![0.0, -TORSO_HALF_HEIGHT]) {
            markers.insert("left_hip".to_owned(), [pelvis[0] - 0.05, pelvis[1]]);
            markers.insert("right_hip".to_owned(), [pelvis[0] + 0.05, pelvis[1]]);
        }
        if let Some(left_knee) = self.body_point(robot.left_thigh, point![0.0, -THIGH_HALF_HEIGHT]) {
            markers.insert("left_knee".to_owned(), left_knee);
        }
        if let Some(right_knee) = self.body_point(robot.right_thigh, point![0.0, -THIGH_HALF_HEIGHT]) {
            markers.insert("right_knee".to_owned(), right_knee);
        }
        markers
    }

    fn body_point(&self, handle: RigidBodyHandle, local: Point<Real>) -> Option<[f32; 2]> {
        self.bodies.get(handle).map(|body| {
            let point = body.position() * local;
            [point.x, point.y]
        })
    }

    fn body_state(&self, handle: RigidBodyHandle) -> Option<BodyState> {
        self.bodies.get(handle).map(|body| BodyState {
            x: body.translation().x,
            y: body.translation().y,
            angle: body.rotation().angle(),
            vx: body.linvel().x,
            vy: body.linvel().y,
            omega: body.angvel(),
        })
    }

    fn relative_angle(&self, parent: RigidBodyHandle, child: RigidBodyHandle) -> Option<f32> {
        let parent = self.bodies.get(parent)?;
        let child = self.bodies.get(child)?;
        Some(normalize_angle(child.rotation().angle() - parent.rotation().angle()))
    }

    fn relative_velocity(&self, parent: RigidBodyHandle, child: RigidBodyHandle) -> Option<f32> {
        let parent = self.bodies.get(parent)?;
        let child = self.bodies.get(child)?;
        Some(child.angvel() - parent.angvel())
    }

    fn foot_contact(&self, left: bool) -> bool {
        let Some(robot) = &self.robot else {
            return false;
        };
        let collider = if left {
            robot.left_shin_collider
        } else {
            robot.right_shin_collider
        };
        self.narrow_phase
            .contact_pairs_with(collider)
            .any(|pair| pair.has_any_active_contact)
    }

    fn current_targets(&self) -> ServoTargets {
        let mut targets = ServoTargets::default();
        if let Some(robot) = &self.robot {
            for (joint, servo) in &robot.servos {
                match joint {
                    JointName::RightHip => targets.right_hip = Some(servo.target),
                    JointName::RightKnee => targets.right_knee = Some(servo.target),
                    JointName::LeftHip => targets.left_hip = Some(servo.target),
                    JointName::LeftKnee => targets.left_knee = Some(servo.target),
                }
            }
        }
        targets
    }
}

impl JointServo {
    fn new(parent: RigidBodyHandle, child: RigidBodyHandle, target: f32) -> Self {
        Self {
            parent,
            child,
            target,
            last_torque: 0.0,
            integral_error: 0.0,
        }
    }
}

fn normalize_angle(angle: f32) -> f32 {
    let mut wrapped = angle;
    while wrapped > std::f32::consts::PI {
        wrapped -= std::f32::consts::TAU;
    }
    while wrapped < -std::f32::consts::PI {
        wrapped += std::f32::consts::TAU;
    }
    wrapped
}

fn clamp_joint_target(joint: JointName, angle: f32) -> f32 {
    match joint {
        JointName::RightHip | JointName::LeftHip => angle.clamp(-1.0, 1.0),
        JointName::RightKnee | JointName::LeftKnee => angle.clamp(0.05, 1.9),
    }
}
