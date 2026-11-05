"""Dual FR3 (UMI) cabinet book insert: the left hand already holds the cabinet handle, the right hand holds a book.

Goal: open the door with the left hand and push the book into the cabinet (no release). Plan and decisions:
``docs/dual_fr3_cabinet_book_insert_plan.md`` (RL design: section 7).

Frames: ``base`` = robot root (arm mounting plate, midpoint of the two arm bases), spawned at world z ``ROBOT_BASE_Z``.
Quaternions are (x, y, z, w). Placement constants are in the base frame and meant to be tuned in the rule-based step.

v1 contact model (plan 5-1, item 4/6):
- left hand: fixed joint between the UMI TCP and the handle bar (``LEFT_GRASP_*``), fingers held at the bar width
- right hand: fixed joint between the UMI TCP and the book (``RIGHT_GRASP_*``), fingers held at the book thickness
- actions: 14 arm joints only, same contract as lift_plate Sim2Real-100Hz (20 Hz, q_meas + clip(0.1 a, ±0.1 rad))

RL (plan 7): the book starts standing on the table at one of the start-table rows (x/y over BOOK_START_REGION_*),
the insert goal is fixed. Policy obs = what the real robot gets from the FK of the attached hands (no camera);
the critic adds contact/handle forces, door velocity and the DR values. DR: book mass, door hinge damping, arm PD
gains, obs noise, book start.
"""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, JointWrenchSensorCfg
from isaaclab.sim.spawners.from_files import MjcfFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg as Gnoise
from isaaclab_physx.physics import PhysxCfg

from dual_fr3_lab.robots import DUAL_FR3_UMI_SIM2REAL_100HZ_CFG, UMI_TCP_OFFSET_POS, UMI_TCP_OFFSET_ROT
from dual_fr3_lab.tasks.manager_based.lift_plate.joint_pos_env_cfg import (
    FR3_JOINT_LIMIT_INSET,
    fr3_joint_limit_margin,
    fr3_joint_limits_inset,
    set_fr3_datasheet_joint_limits,
)
from dual_fr3_lab.tasks.manager_based.reach.joint_pos_env_cfg import DUAL_ARM_JOINT_NAMES
from dual_fr3_lab.tasks.manager_based.reach.mdp.actions import ReachRelativeJointPositionActionCfg

from . import mdp

ASSET_DIR = Path(__file__).resolve().parents[3] / "assets/cabinet_book"
# copy of codex content/cabinet/cabinet_bimanual_handle.xml with the handle moved onto the door face
CABINET_ASSET = str(ASSET_DIR / "cabinet_book_insert.xml")
# reset table: one row per book start on the table (solver grid), joints + book pose + solver joint paths
START_TABLE_PATH = str(ASSET_DIR / "start_table.npz")

# ---------- placement (base frame) ----------
ROBOT_BASE_Z = 0.405  # world z of the base frame (husky on the floor), same as DUAL_FR3_CFG
TABLE_TOP_Z_B = 0.045  # table height is adjustable on the real setup
TABLE_X_RANGE_B = (0.40, 1.25)  # front edge pulled in so the whole book start region sits on it
TABLE_WIDTH = 1.2  # y +-0.6 (the book start region reaches y -0.505)
# Cabinet model origin (bottom centre) on the table, yawed 180 deg so the door faces the robot.
# Model (codex MJCF copy, handle moved onto the door face): outer 0.39 x 0.39 x 0.39 m, inside about 0.356 m cube,
# door hinge at robot-left, vertical bar handle on the door face near its free (robot-right) edge.
# The UMI TCP is 0.284 m past link7 (plus 5.5 cm to the book centre), so the arms fold near the robot and run out of
# reach deep in the cabinet. Placement x grasp pitch sweep (scripts/tools/solve_cabinet_book_start.py, 2026-11-05):
# horizontal grasps fail everywhere (left door <= 25-75 deg or right slide); (0.95, 0.10) with 45 deg pitch follows
# door 0-90 deg and the book to the cabinet centre with path joint margins 0.40 (left) / 0.51 (right) rad.
CABINET_XY_B = (0.95, 0.10)
CABINET_ROT = (0.0, 0.0, 1.0, 0.0)
CABINET_CENTER_Z_ABOVE_TABLE = 0.195
CABINET_INNER_HALF = 0.178  # inner half extent along every axis (walls 17 mm)
CABINET_OUTER_HALF = 0.195
# inner box centre (base frame); the open side faces the robot (-x), so the opening is at x = centre - CABINET_INNER_HALF
CABINET_INNER_CENTER_B = (CABINET_XY_B[0], CABINET_XY_B[1], TABLE_TOP_Z_B + CABINET_CENTER_Z_ABOVE_TABLE)
# MJCF frames (model frame = cabinet root, x toward the door; the handle body has the root's orientation)
DOOR_IN_CABINET = (0.1865, 0.0, CABINET_CENTER_Z_ABOVE_TABLE)
HANDLE_IN_DOOR = (0.0085, 0.135, 0.0)
DOOR_HINGE_IN_DOOR = (0.0, -0.1555, 0.0)  # drawer_joint position (axis = door z, opening is negative)

# ---------- book (right hand) ----------
BOOK_SIZE = (0.15, 0.03, 0.22)  # (depth = insertion direction, thickness = grip direction, height)
BOOK_MASS = 0.3
# Rule-based path (base frame, book axes parallel to the base axes): the book waits right of the door sweep at the
# pre-insert x, slides left in front of the opening once the door is open, then goes straight in along +x.
CABINET_OPENING_X_B = CABINET_XY_B[0] - CABINET_INNER_HALF
BOOK_PRE_GAP = 0.05  # book front this far in front of the cabinet front face while it slides in front of it
BOOK_INSERT_DY = -0.04  # book y - cabinet centre y: keeps the right hand (wide along y) off the open door
# Book centre this far past the opening at the end (back face 3.5 cm inside). Deeper, the 45 deg right hand hits the
# cabinet's top front edge (2026-11-05 rule-based: 92 N at 14.5 cm); the cabinet centre is 17.8 cm in.
BOOK_INSERT_DEPTH = 0.12
_BOOK_Z_B = CABINET_CENTER_Z_ABOVE_TABLE + TABLE_TOP_Z_B
BOOK_PRE_POS_B = (CABINET_XY_B[0] - CABINET_OUTER_HALF - BOOK_PRE_GAP - 0.5 * BOOK_SIZE[0], CABINET_XY_B[1] + BOOK_INSERT_DY, _BOOK_Z_B)
BOOK_INSERT_POS_B = (CABINET_OPENING_X_B + BOOK_INSERT_DEPTH, BOOK_PRE_POS_B[1], _BOOK_Z_B)
# Book start (RL, 2026-11-05 decision): standing on the table right of the door sweep and the left hand, x/y drawn per
# episode from START_TABLE_PATH (region below, ~20 cm each). Path: lift to the lane height, x to the pre-insert x,
# slide to the lane once the door is open, insert.
BOOK_ON_TABLE_Z_B = TABLE_TOP_Z_B + 0.5 * BOOK_SIZE[2] + 0.001
# 20 x 20 cm around the rule_v4 start (0.63, -0.39). Solver map (2026-11-05, grid 13 over x 0.43-0.79, y -0.60--0.24):
# every book start with x >= 0.52 follows the whole path (joint margin 0.1-0.4 rad, lowest at small x / large y);
# x <= 0.49 with y >= -0.39 fails (the right arm meets the left one).
BOOK_START_REGION_X = (0.53, 0.73)
BOOK_START_REGION_Y = (-0.49, -0.29)
BOOK_START_POS_B = (0.5 * sum(BOOK_START_REGION_X), 0.5 * sum(BOOK_START_REGION_Y), BOOK_ON_TABLE_Z_B)


def book_path_points(start):
    """Rule-based book centre path (base frame): start -> lifted -> pre-insert x -> pre-insert (lane) -> inserted."""
    lane_z = BOOK_PRE_POS_B[2]
    return [
        tuple(start),
        (start[0], start[1], lane_z),
        (BOOK_PRE_POS_B[0], start[1], lane_z),
        BOOK_PRE_POS_B,
        BOOK_INSERT_POS_B,
    ]


# waypoints per segment of book_path_points (solver path check, ~2 cm apart)
BOOK_PATH_SEGMENT_STEPS = (6, 6, 24, 14)

# ---------- grasps (TCP = UMI fingertip centre, z = approach, y = finger travel) ----------
FINGER_CLEARANCE = 0.001  # per finger, so the pads touch without squeezing the attached object
HANDLE_BAR_RADIUS = 0.015
HANDLE_BAR_CENTER_IN_HANDLE = (0.045, 0.0, 0.0)  # MJCF handle body frame (x = door outward normal, z = bar axis)
HANDLE_TIP_DEPTH = 0.02  # fingertips pass the bar axis by this much toward the door
# Approach pitch: the approach is tilted down from horizontal by this angle (rotation about the finger axis, so the
# fingers stay across the vertical bar / the book thickness). 0 = horizontal.
LEFT_GRASP_PITCH = math.radians(45.0)
RIGHT_GRASP_PITCH = math.radians(45.0)
# True: rotate the grasp 180 deg about the approach axis (wrist j7 + pi). Left False: with True every left path passes
# a near-singular arm at door 20-25 deg (TCP Jacobian sigma_min 0.02), where the welded handle took 500-1200 N;
# False + the solver's sigma_min ascent keeps sigma_min >= 0.15 over the whole door path.
LEFT_GRASP_FLIP = False
RIGHT_GRASP_FLIP = True
BOOK_GRIP_DEPTH = 0.02  # fingertips this far past the book's rear face (mid-height), along the approach


def left_grasp_pos(pitch: float):
    """TCP in the handle frame: approach toward the door (-x_handle, tilted down), fingertips past the bar axis."""
    return (HANDLE_BAR_CENTER_IN_HANDLE[0] - HANDLE_TIP_DEPTH * math.cos(pitch), 0.0, -HANDLE_TIP_DEPTH * math.sin(pitch))


def right_grasp_pos(pitch: float):
    """TCP in the book frame: approach along +x_book (tilted down), fingers along y_book (thickness)."""
    return (-0.5 * BOOK_SIZE[0] + BOOK_GRIP_DEPTH * math.cos(pitch), 0.0, -BOOK_GRIP_DEPTH * math.sin(pitch))


LEFT_GRASP_POS = left_grasp_pos(LEFT_GRASP_PITCH)
RIGHT_GRASP_POS = right_grasp_pos(RIGHT_GRASP_PITCH)


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _qy(angle):
    return (0.0, math.sin(0.5 * angle), 0.0, math.cos(0.5 * angle))


_QZ180 = (0.0, 0.0, 1.0, 0.0)


def grasp_rot(approach_sign: float, flip: bool, pitch: float = 0.0):
    """TCP orientation in the object frame: TCP z = approach_sign * x_obj tilted down (-z_obj) by pitch, TCP y = +-y_obj."""
    q = _qy(approach_sign * (0.5 * math.pi + pitch))
    return _qmul(q, _QZ180) if flip else q


LEFT_GRASP_ROT = grasp_rot(-1.0, LEFT_GRASP_FLIP, LEFT_GRASP_PITCH)
RIGHT_GRASP_ROT = grasp_rot(+1.0, RIGHT_GRASP_FLIP, RIGHT_GRASP_PITCH)
# finger joint targets (each finger's travel from the centre) while holding
LEFT_FINGER_POS = HANDLE_BAR_RADIUS + FINGER_CLEARANCE
RIGHT_FINGER_POS = 0.5 * BOOK_SIZE[1] + FINGER_CLEARANCE

EPISODE_LENGTH_S = 12.0  # rule-based book-on-table run: HOLD at 7.95-8.90 s over all start rows

# ---------- RL (plan 7) ----------
# success (2026-11-05 decision): book centre >= 9 cm past the opening (back face 1.5 cm inside), on the lane within
# 3 cm in y and z, tilt < 10 deg
SUCCESS_PARAMS = {"goal_pos_b": BOOK_INSERT_POS_B, "opening_x_b": CABINET_OPENING_X_B, "min_depth": 0.09,
                  "lane_tol": 0.03, "max_tilt_deg": 10.0}
HANDLE_FORCE_LIMIT = 30.0  # N (2026-11-05 decision): above this the real UMI grip on the handle may slip
CONTACT_FORCE_LIMIT = 5.0  # N per sensor
DOOR_GATE_DEG = (60.0, 80.0)  # book rewards ramp in while the door opens through this range
BOOK_MASS_RANGE = (0.2, 0.5)  # kg
DOOR_DAMPING_RANGE = (0.5, 3.0)  # N m s / rad (MJCF 1.0)
ARM_CONTACT_LINKS = ("link4", "link5", "link6", "link7", "leftfinger", "rightfinger")
ARM_CONTACT_SENSORS = [f"contact_{s}_{lk}" for s in ("left", "right") for lk in ARM_CONTACT_LINKS]
CONTACT_SENSORS = ARM_CONTACT_SENSORS + ["contact_book"]
ARM_JOINTS = SceneEntityCfg("robot", joint_names=DUAL_ARM_JOINT_NAMES, preserve_order=True)


def arm_link_path(side: str, link: str) -> str:
    """Prim path of an arm body (the converted USD nests every link under its parent)."""
    chain = "/".join(f"{side}_fr3_link{k}" for k in range(1, 8 if link.endswith("finger") else int(link[-1]) + 1))
    path = "{ENV_REGEX_NS}/Robot/Geometry/base/" + chain
    return path + f"/{side}_fr3_{link}" if link.endswith("finger") else path


def _b2w(p):
    return (p[0], p[1], p[2] + ROBOT_BASE_Z)


def _box(name, size, pos_b, color):
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/" + name,
        init_state=AssetBaseCfg.InitialStateCfg(pos=_b2w(pos_b)),
        spawn=sim_utils.MeshCuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


_TABLE_CX = 0.5 * (TABLE_X_RANGE_B[0] + TABLE_X_RANGE_B[1])
_TABLE_DX = TABLE_X_RANGE_B[1] - TABLE_X_RANGE_B[0]
_TABLE_H = TABLE_TOP_Z_B + ROBOT_BASE_Z  # top height above the floor


@configclass
class CabinetBookInsertSceneCfg(InteractiveSceneCfg):
    robot = DUAL_FR3_UMI_SIM2REAL_100HZ_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=MjcfFileCfg(
            asset_path=CABINET_ASSET,
            fix_base=True,
            collision_from_visuals=False,
            import_physics_scene=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=_b2w((CABINET_XY_B[0], CABINET_XY_B[1], TABLE_TOP_Z_B)), rot=CABINET_ROT, joint_pos={"drawer_joint": 0.0}
        ),
        actuators={
            "door_hinge": ImplicitActuatorCfg(
                joint_names_expr=["drawer_joint"], effort_limit_sim=20.0, stiffness=0.2, damping=1.0
            )
        },
    )
    book = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Book",
        init_state=RigidObjectCfg.InitialStateCfg(pos=_b2w(BOOK_START_POS_B)),
        spawn=sim_utils.MeshCuboidCfg(
            size=BOOK_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_depenetration_velocity=1.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=BOOK_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            activate_contact_sensors=True,
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.6, dynamic_friction=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.75)),
        ),
    )
    table = _box("Table", (_TABLE_DX, TABLE_WIDTH, 0.03), (_TABLE_CX, 0.0, TABLE_TOP_Z_B - 0.015), (0.8, 0.6, 0.4))
    table_leg_l = _box("TableLegL", (0.04, 0.04, _TABLE_H - 0.03), (_TABLE_CX, 0.5 * TABLE_WIDTH - 0.05, TABLE_TOP_Z_B - 0.03 - 0.5 * (_TABLE_H - 0.03)), (0.8, 0.8, 0.8))
    table_leg_r = _box("TableLegR", (0.04, 0.04, _TABLE_H - 0.03), (_TABLE_CX, -0.5 * TABLE_WIDTH + 0.05, TABLE_TOP_Z_B - 0.03 - 0.5 * (_TABLE_H - 0.03)), (0.8, 0.8, 0.8))
    # net contact force per body (critic, contact penalty); the arm link sensors are added in the env cfg
    contact_book = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Book")
    # incoming joint wrench per cabinet link: the handle's = the force the left hand applies to the handle
    cabinet_wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Cabinet")
    plane = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(color=(0.8, 0.8, 0.8), intensity=3000.0))


@configclass
class ActionsCfg:
    arm_action = ReachRelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=DUAL_ARM_JOINT_NAMES,
        preserve_order=True,
        scale=0.1,
        clip={".*": (-0.1, 0.1)},
        reference_mode="policy_step",
        target_limits=fr3_joint_limits_inset(FR3_JOINT_LIMIT_INSET) * 2,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """50 dims, all available on the real robot (encoders + FK of the attached hands)."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": ARM_JOINTS}, noise=Gnoise(std=0.005))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": ARM_JOINTS}, noise=Gnoise(std=0.05))
        door_angle = ObsTerm(func=mdp.door_angle, noise=Gnoise(std=0.02))  # left-hand FK + cabinet calibration
        book_pos = ObsTerm(  # right-hand FK + grasp offset, relative to the insert goal
            func=mdp.book_pos_rel, params={"target_pos_b": BOOK_INSERT_POS_B}, noise=Gnoise(std=0.005)
        )
        book_quat = ObsTerm(func=mdp.book_quat, noise=Gnoise(std=0.01))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        """Policy terms without noise + privileged terms."""

        door_velocity = ObsTerm(func=mdp.door_velocity)
        contact_forces = ObsTerm(func=mdp.contact_forces, params={"sensor_names": CONTACT_SENSORS, "scale": 0.01})
        handle_force = ObsTerm(func=mdp.handle_force, params={"scale": 0.01})
        book_mass = ObsTerm(func=mdp.book_mass)
        door_damping = ObsTerm(func=mdp.door_damping)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    fr3_joint_limits = EventTerm(
        func=set_fr3_datasheet_joint_limits,
        mode="startup",
        params={
            "inset": FR3_JOINT_LIMIT_INSET,
            "asset_cfg": SceneEntityCfg("robot", joint_names=DUAL_ARM_JOINT_NAMES, preserve_order=True),
        },
    )
    attach_left_handle = EventTerm(
        func=mdp.attach_fixed_joint,
        mode="prestartup",
        params={
            "joint_name": "LeftHandleGrasp",
            "body0": ("Robot", "left_fr3_link7"),
            "body1": ("Cabinet", "handle"),
            "local_pos0": UMI_TCP_OFFSET_POS,
            "local_rot0": UMI_TCP_OFFSET_ROT,
            "local_pos1": LEFT_GRASP_POS,
            "local_rot1": LEFT_GRASP_ROT,
        },
    )
    attach_right_book = EventTerm(
        func=mdp.attach_fixed_joint,
        mode="prestartup",
        params={
            "joint_name": "RightBookGrasp",
            "body0": ("Robot", "right_fr3_link7"),
            "body1": ("Book", "Book"),
            "local_pos0": UMI_TCP_OFFSET_POS,
            "local_rot0": UMI_TCP_OFFSET_ROT,
            "local_pos1": RIGHT_GRASP_POS,
            "local_rot1": RIGHT_GRASP_ROT,
        },
    )
    # DR (plan 7-5): book mass, door hinge damping, arm PD gains (+-15 %, plate), book start (reset table row)
    randomize_book_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("book"),
            "mass_distribution_params": BOOK_MASS_RANGE,
            "operation": "abs",
            "recompute_inertia": True,
        },
    )
    randomize_door_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cabinet"),
            "damping_distribution_params": DOOR_DAMPING_RANGE,
            "operation": "abs",
        },
    )
    randomize_arm_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_fr3_joint[1-7]"]),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )
    reset_start = EventTerm(func=mdp.reset_to_start_table, mode="reset", params={"table_path": START_TABLE_PATH})


@configclass
class RewardsCfg:
    """Plan 7-3. Positive terms follow the rule-based order: door, then (gated by the door) book to the lane, in."""

    door_open = RewTerm(func=mdp.door_open, params={"goal_deg": 90.0}, weight=1.0)
    book_approach = RewTerm(
        func=mdp.book_approach, params={"pre_pos_b": BOOK_PRE_POS_B, "std": 0.3, "gate_deg": DOOR_GATE_DEG}, weight=1.0
    )
    book_lane = RewTerm(
        func=mdp.book_lane, params={"goal_pos_b": BOOK_INSERT_POS_B, "std": 0.03, "gate_deg": DOOR_GATE_DEG}, weight=1.0
    )
    book_depth = RewTerm(
        func=mdp.book_depth,
        params={"goal_pos_b": BOOK_INSERT_POS_B, "opening_x_b": CABINET_OPENING_X_B, "gate_deg": DOOR_GATE_DEG},
        weight=2.0,
    )
    insert_success = RewTerm(func=mdp.insert_success, params=SUCCESS_PARAMS, weight=5.0)
    contact = RewTerm(
        func=mdp.contact_excess, params={"sensor_names": CONTACT_SENSORS, "threshold": CONTACT_FORCE_LIMIT}, weight=-0.01
    )
    handle_force = RewTerm(func=mdp.handle_force_excess, params={"threshold": HANDLE_FORCE_LIMIT}, weight=-0.02)
    joint_limit = RewTerm(func=fr3_joint_limit_margin, params={"asset_cfg": ARM_JOINTS}, weight=-1.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, params={"asset_cfg": ARM_JOINTS}, weight=-0.001)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class DualFR3CabinetBookInsertEnvCfg(ManagerBasedRLEnvCfg):
    # replicate_physics=False: per-env book mass / gains (DR) and the per-env attachment joints
    scene: CabinetBookInsertSceneCfg = CabinetBookInsertSceneCfg(num_envs=1024, env_spacing=3.0, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 5  # 100 Hz physics, 20 Hz policy (lift_plate Sim2Real-100Hz)
        self.episode_length_s = EPISODE_LENGTH_S
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024,
            gpu_total_aggregate_pairs_capacity=64 * 1024,
            friction_correlation_distance=0.00625,
        )
        self.viewer.eye = (2.0, -1.4, 1.6)
        self.viewer.lookat = (0.6, 0.0, 0.6)
        for side in ("left", "right"):
            for link in ARM_CONTACT_LINKS:
                setattr(self.scene, f"contact_{side}_{link}", ContactSensorCfg(prim_path=arm_link_path(side, link)))


@configclass
class DualFR3CabinetBookInsertEnvCfg_PLAY(DualFR3CabinetBookInsertEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.observations.policy.enable_corruption = False
