"""G1 SONIC with BrainCo Revo 2 hands (2026-10-23): the rot0 teleop robot (data/robots/g1_rot0_teleop_revo2, made by
scripts/robots/make_g1_revo2.py) whose Dex3 hands are replaced by the Revo 2. 11 joints per hand in MJCF tree order
(thumb metacarpal / proximal / distal, then index, middle, ring, pinky proximal / distal); the distal joints are coupled to
their proximal joint by equality constraints as on the real hand (thumb 1.0 x, fingers 1.155 x). Hand PD torques are
written per joint like the Dex3 (the engine indexes torques by joint). The real hand is commanded in grip mode as a
fraction of each motor's travel: open = thumb proximal at 0.8 of its range, everything else 0; closed = thumb metacarpal at
0.55, the rest fully closed (the deploy profile's BRAINCO_THUMB_CLOSE_SCALE=0.55 and the 2026-09-08 BrainCo teleop commands).
Registered as "g1_sonic_revo2"; its uid stays "g1_sonic" so recorded scene states (spatial key g1_sonic) still apply."""
from __future__ import annotations
import json, os
from pathlib import Path
import numpy as np
from simple.robots.registry import RobotRegistry
from simple.robots.g1_sonic import G1Sonic, LEFT_LEG_JOINTS, RIGHT_LEFT_JOINTS, WAIST_JOINTS, LEFT_ARM_JOINTS, RIGHT_ARM_JOINTS
from simple.robots.controllers.combo import WholeBodyEEFControllerCfg
from simple.robots.controllers.eef import DexHandEEFControllerCfg
from simple.robots.controllers.qpos import PDJointPosControllerCfg

FINGERS = ["thumb_metacarpal", "thumb_proximal", "thumb_distal", "index_proximal", "index_distal", "middle_proximal", "middle_distal",
           "ring_proximal", "ring_distal", "pinky_proximal", "pinky_distal"]
LEFT_REVO2_JOINTS = [f"left_hand_{f}_joint" for f in FINGERS]
RIGHT_REVO2_JOINTS = [f"right_hand_{f}_joint" for f in FINGERS]
REVO2_UPPER = np.array([1.5184, 1.0472, 1.0472, 1.4661, 1.693, 1.4661, 1.693, 1.4661, 1.693, 1.4661, 1.693])   # joint ranges (rad), lower = 0
COUPLING = np.array([1.0, 1.0, 1.0, 1.0, 1.155, 1.0, 1.155, 1.0, 1.155, 1.0, 1.155])                         # distal = k x proximal
OPEN_FRACTION = np.array([0.0, 0.8, 0.8, 0, 0, 0, 0, 0, 0, 0, 0])
CLOSED_FRACTION = np.array([0.55, 1.0, 1.0, 1.0, 1.155, 1.0, 1.155, 1.0, 1.155, 1.0, 1.155])
OPEN_Q = np.minimum(OPEN_FRACTION * np.array([REVO2_UPPER[0], REVO2_UPPER[1], REVO2_UPPER[1], *([REVO2_UPPER[3]] * 8)]), REVO2_UPPER)
CLOSED_Q = np.minimum(CLOSED_FRACTION * np.array([REVO2_UPPER[0], REVO2_UPPER[1], REVO2_UPPER[1], *([REVO2_UPPER[3]] * 8)]), REVO2_UPPER)
# second closing pose (PICO trigger, 2026-10-26, user): a thumb-index pinch; the other fingers stay straight
CLOSE2_Q = np.array([1.23, 0.67, 0.67, 0.705, 0.8143, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
DEFAULT_OPEN_Q, DEFAULT_CLOSED_Q, DEFAULT_CLOSE2_Q = OPEN_Q.copy(), CLOSED_Q.copy(), CLOSE2_Q.copy()
# the grip poses actually used come from this file when it exists (written from the pose tuner page docs/revo2_pose_tuner,
# or by hand): {"open": [11 rad, MJCF order], "closed": [11], "close2": [11] (the trigger's pinch), "threshold": 0.5}; absent = the module defaults above
GRIP_POSES_PATH = Path(os.environ.get("SIMPLE_REVO2_GRIP_POSES") or Path(__file__).resolve().parents[3] / "data/robots/g1_rot0_teleop_revo2/grip_poses.json")
GRIP_THRESHOLD = 0.5


def set_grip_poses(open_q, closed_q, threshold: float | None = None, close2_q=None) -> None:
    """Replace the grip-mode targets in place (11 values each, MJCF order, clipped to the joint ranges), so every holder of
    OPEN_Q / CLOSED_Q / CLOSE2_Q (revo2_targets, the teleop agent, the replay scripts) sees the new values."""
    global GRIP_THRESHOLD
    OPEN_Q[:] = np.clip(np.asarray(open_q, dtype=float)[:11], 0.0, REVO2_UPPER)
    CLOSED_Q[:] = np.clip(np.asarray(closed_q, dtype=float)[:11], 0.0, REVO2_UPPER)
    if close2_q is not None: CLOSE2_Q[:] = np.clip(np.asarray(close2_q, dtype=float)[:11], 0.0, REVO2_UPPER)
    if threshold is not None: GRIP_THRESHOLD = float(threshold)


def load_grip_poses(path=GRIP_POSES_PATH) -> dict | None:
    p = Path(path)
    if not p.is_file(): return None
    cfg = json.loads(p.read_text()); set_grip_poses(cfg["open"], cfg["closed"], cfg.get("threshold"), cfg.get("close2")); cfg["path"] = str(p); return cfg


GRIP_POSES = load_grip_poses()

# ---- hand variants (2026-11-04, user: "a new version of the robot with the official hand ... switch between the old revo2 hand
# and the new hand"). legacy = the 10-23 hand (HoloMotion module kinematics, Unitree meshes; the v3 file every recording so
# far used). official = BrainCo's own revo2_description v1.1.1 hand on the same robot (scripts/robots/make_g1_revo2.py --hand
# official): per-finger couplings, 80.8 deg finger limits, the thumb zero offsets, BrainCo's pad links as the tactile pads.
# select_hand_variant() sets the class file, REVO2_UPPER / COUPLING in place from the compiled model (joint ranges, equality
# polycoef) and reloads the grip poses from that robot's folder, so every importer of these arrays follows; the controller
# config is rebuilt with the new open / closed poses. Recordings pin their file (run_meta robot_mjcf, revo2_hand); the replay
# selects by it (hand_variant_of).
HAND_VARIANTS = {"legacy": "robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2_v3.xml",
                 "official": "robots/g1_rot0_teleop_revo2_official/g1_rot0_teleop_revo2_official.xml"}
HAND_VARIANT = "legacy"


def hand_variant_of(mjcf_path) -> str:
    """The variant a robot file belongs to (a recording's run_meta robot_mjcf)."""
    return "official" if "g1_rot0_teleop_revo2_official" in str(mjcf_path or "") else "legacy"


def select_hand_variant(name: str, mjcf_path: str | None = None) -> dict:
    """Switch the Revo 2 robot to `name` ("legacy" / "official"), optionally a specific file of that variant (under data/).
    Call before the env is made (the class attribute is what the registry's robot reads)."""
    global HAND_VARIANT
    if name not in HAND_VARIANTS: raise ValueError(f"Revo 2 hand variant {name!r}: one of {sorted(HAND_VARIANTS)}")
    import mujoco
    from simple.utils import get_data_dir
    rel = mjcf_path or HAND_VARIANTS[name]; path = Path(get_data_dir()) / rel
    m = mujoco.MjModel.from_xml_path(str(path))
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in RIGHT_REVO2_JOINTS]
    upper = np.array([m.jnt_range[j][1] for j in jid]); coupling = np.ones(11)
    for e in range(m.neq):
        if m.eq_type[e] == mujoco.mjtEq.mjEQ_JOINT and int(m.eq_obj1id[e]) in jid: coupling[jid.index(int(m.eq_obj1id[e]))] = float(m.eq_data[e][1])
    REVO2_UPPER[:] = upper; COUPLING[:] = coupling
    G1SonicRevo2.mjcf_path = rel; HAND_VARIANT = name
    cfg = load_grip_poses(path.parent / "grip_poses.json")
    if cfg is None: set_grip_poses(DEFAULT_OPEN_Q, DEFAULT_CLOSED_Q, 0.5, DEFAULT_CLOSE2_Q)
    G1SonicRevo2.controller_cfg = _controller_cfg()
    return dict(variant=name, mjcf=rel, upper=upper.tolist(), coupling=coupling.tolist(), grip_poses=cfg["path"] if cfg else None)


def revo2_targets(grip: float, threshold: float | None = None, trigger: float = 0.0) -> np.ndarray:
    """Grip-mode hand target (11, MJCF order) from the PICO grip and trigger values in 0..1: grip pressed -> CLOSED_Q (the
    power grasp, as the Dex3 gripper's target_q_for_grip), else trigger pressed -> CLOSE2_Q (the thumb-index pinch), else OPEN_Q."""
    thr = GRIP_THRESHOLD if threshold is None else float(threshold)
    q = CLOSED_Q if float(grip) > thr else CLOSE2_Q if float(trigger) > thr else OPEN_Q
    return q.astype(np.float32).copy()


BODY_JOINTS = LEFT_LEG_JOINTS + RIGHT_LEFT_JOINTS + WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
WHOLE_BODY_REVO2_JOINTS = BODY_JOINTS + LEFT_REVO2_JOINTS + RIGHT_REVO2_JOINTS


def _controller_cfg() -> WholeBodyEEFControllerCfg:
    """The whole-body controller config with the CURRENT grip poses (rebuilt by select_hand_variant)."""
    return WholeBodyEEFControllerCfg(
        left_leg=PDJointPosControllerCfg(joint_names=LEFT_LEG_JOINTS, init_qpos=[0.0] * 6),
        right_leg=PDJointPosControllerCfg(joint_names=RIGHT_LEFT_JOINTS, init_qpos=[0.0] * 6),
        waist=PDJointPosControllerCfg(joint_names=WAIST_JOINTS, init_qpos=[0.0] * 3),
        left_arm=PDJointPosControllerCfg(joint_names=LEFT_ARM_JOINTS, init_qpos=[0.0] * 7),
        right_arm=PDJointPosControllerCfg(joint_names=RIGHT_ARM_JOINTS, init_qpos=[0.0] * 7),
        left_eef=DexHandEEFControllerCfg(joint_names=LEFT_REVO2_JOINTS, init_qpos=OPEN_Q.tolist(), close_qpos=CLOSED_Q.tolist()),
        right_eef=DexHandEEFControllerCfg(joint_names=RIGHT_REVO2_JOINTS, init_qpos=OPEN_Q.tolist(), close_qpos=CLOSED_Q.tolist()),
    )


@RobotRegistry.register("g1_sonic_revo2")
class G1SonicRevo2(G1Sonic):
    uid: str = "g1_sonic"
    # v2 (2026-10-27): + low-friction hand self-contact, so released fingers cannot stay locked on the thumb
    # (scripts/robots/add_revo2_hand_selfcontact.py). v3 (2026-10-28): the left thumb tactile pad moved to the fingertip (it sat
    # 42.5 mm behind it; scripts/robots/add_revo2_touch.py). Recordings keep the file they were made with (run_meta robot_mjcf).
    # 2026-11-04: select_hand_variant("official") swaps this for the BrainCo v1.1.1 hand robot (HAND_VARIANTS).
    mjcf_path: str = HAND_VARIANTS["legacy"]
    hand_uid: str = "revo2_right"
    hand_dof: int = 11
    joint_names = WHOLE_BODY_REVO2_JOINTS
    hand_names = LEFT_REVO2_JOINTS + RIGHT_REVO2_JOINTS
    init_joint_states: dict[str, float] = dict(zip(WHOLE_BODY_REVO2_JOINTS, [0.0] * len(WHOLE_BODY_REVO2_JOINTS)))
    controller_cfg = _controller_cfg()
    hand_prim_path = "right_hand_base_link"
    eef_prim_path: str = "right_hand_base_link"
    LEFT_ARM_EE_LINK: str = "left_hand_base_link"
    RIGHT_ARM_EE_LINK: str = "right_hand_base_link"
    HAND_KP = np.array([2.0, 3.0, 1.5, 3.0, 1.5, 3.0, 1.5, 3.0, 1.5, 3.0, 1.5])
    HAND_KD = 0.3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sonic_robot.NUM_HAND_JOINTS = 11                      # the controller package assumes the Dex3's 7
        self.num_hand_dof = 11
        self.torques = np.zeros(self.num_body_dof + self.num_hand_dof * 2)

    def setup_control(self, mjData, mjModel, **kwargs):
        out = super().setup_control(mjData, mjModel, **kwargs)
        # the torque clip list comes from the SONIC yaml (29 body + 2 x 7 Dex3 entries in MJCF joint order); with 2 x 11 hand
        # joints every actuator is clipped at its own ctrlrange instead (51 actuators in joint order; the body values are the same)
        self.torque_limit = np.asarray(mjModel.actuator_ctrlrange[:, 1], dtype=float).copy()
        return out

    def _apply_hand_pd(self, left_hand_q, right_hand_q) -> None:
        if left_hand_q is not None:
            q = self.mjData.qpos[self.left_hand_index + self.qpos_offset - 1]; dq = self.mjData.qvel[self.left_hand_index + self.qvel_offset - 1]
            self.torques[self.left_hand_index - 1] = self.HAND_KP * (np.asarray(left_hand_q)[:11] - q) - self.HAND_KD * dq
        if right_hand_q is not None:
            q = self.mjData.qpos[self.right_hand_index + self.qpos_offset - 1]; dq = self.mjData.qvel[self.right_hand_index + self.qvel_offset - 1]
            self.torques[self.right_hand_index - 1] = self.HAND_KP * (np.asarray(right_hand_q)[:11] - q) - self.HAND_KD * dq
