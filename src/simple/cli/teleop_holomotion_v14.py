"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Teleoperate a G1 scene with the HoloMotion v1.4 controller (teleop-collection branch) and optionally record
LeRobot episodes.

  Terminal 1 - reference publisher (holomotion_teleop conda env: SMPL/GMR + XRoboToolkit SDK):
      ~/miniconda3/envs/holomotion_teleop/bin/python src/simple/teleop/holomotion_v14/publisher.py --source pico
  Terminal 2 (SIMPLE venv):
      python -m simple.cli.teleop_holomotion_v14 --scene bottle_bin --record

PICO buttons (--button-map sim, the default; --button-map robot = the robot's v1.4.1 map, B motion / Y back to walking):
  A                  walking mode (policy on; also lowers the sim safety band)
  Y                  motion tracking (follows your body); A: back to walking
  B                  nothing (free for the PICO app's view toggle)
  left stick / right stick x   walk / turn
  grips              close the Dex3 hands
  X                  zero torque        right stick click   emergency stop
  left stick click   stand up (only needed with --no-quick-start, or after X / emergency stop)
Sim only (hold the left menu button; nothing reaches the robot controller while it is held):
  left menu + A      start recording / save the episode
  left menu + B      abandon the episode
  left menu + Y      skip to a new random setup
  both triggers      reset the scene (same setup: a retry)

Every episode starts from a new random setup (robot start, target, bin / cart, distractors; --no-randomize for the
nominal scene), drawn from its own seed. With --record each saved episode also gets a bit-exact log
(replay/episode_XXXXXX.npz next to the dataset): the start state, every physics input and the setup, so
`python -m simple.cli.replay_holomotion_v14` replays it to the last bit.

--quick-start (the default; --no-quick-start for the leash): every scene -- the first one at launch, then after each
save, left menu + Y or both triggers -- starts with the robot standing on the floor, no leash, in walking mode (the
walking policy balances it even before the PICO is connected). Recording starts once the PICO controllers are there;
press Y for motion tracking when ready (--auto-motion presses it for you).

--random-init-pose: every scene init also draws a small random robot pose (base x/y/yaw, joint angles; seeded from
--seed), so no two episodes start from the same robot state. With --quick-start the robot is placed and held in that
pose and recording starts on the first step the controllers are connected, so frame 0 is the randomized state (the
episode then includes the ~1 s automatic hand-over to motion tracking). --init-pose-scale scales the ranges.

Headset view: --stream-view single (default) sends the left head camera once, centred, with a status overlay: one
picture in the XRoboToolkit app's default view (the app's right-B view toggle would show only its left half -- press B
again). mono puts it in both halves (for the app's single view), stereo sends left | right.

Robot: --robot stereo (default) is the G1 with the HBVCAM stereo head camera (the v1.4 collection robot's camera,
third_party/hbvcam_stereo); the headset shows its rectified left eye as a pinhole image (--stream-camera pinhole,
1280 x 720, 104.7 x 72.2 deg). --robot stock / --stream-camera head: SIMPLE's G1 / the scene's D455 head camera.

G1 with a backpack: --backpack-kg 3.2 (the v1.4.1 backpack policy's mass) adds HoloMotion's backpack body to the torso;
--motion-model-bundle DIR swaps in the backpack motion model (model_22000: config.yaml + model_22000.onnx, hashes
checked against the upstream lock).
"""

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path
from typing import Annotated

import gymnasium as gym
import numpy as np
import typer
import tyro
from tqdm import tqdm

from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from simple.agents.holomotion_v14_agent import HoloMotionV14Agent
from simple.cli._decoupled_wbc_recording import set_ego_view_feature_shape, validate_existing_ego_view_feature_shape
from simple.cli.teleop_decoupled_wbc import RecordingState, _save_episode_env_config
from simple.envs.sonic_loco_manip import SonicLocoManipEnv
from simple.robots.g1_sonic import G1Sonic
from simple.teleop.holomotion_v14 import DEFAULT_REFERENCE_URI
from simple.teleop.holomotion_v14.exact_log import ExactLog
from simple.teleop.holomotion_v14.scene_setup import SceneSetups

SCENES = {"bottle_bin": "bottle_bin_task", "bowl_sink": "bowl_sink_task", "coffee_cart": "coffee_cart_task"}
# Where the robot starts, as the scene's own knob (front of the feet -> first obstacle; everything but the robot moves).
# The scenes' defaults were fitted to decoupled-WBC replays that start with the hands raised above the table. The v1.4
# controller stands up with its forearms at table height, so at bottle_bin's 3 cm the hands land on the table and the
# robot leans on it; like the real v1.4 collection, the operator starts clear and walks in.
START_KNOB = {"bottle_bin": "BOTTLE_BIN_ROBOT_TO_EDGE", "bowl_sink": "BOWL_SINK_ROBOT_TO_EDGE",
              "coffee_cart": "COFFEE_CART_ROBOT_TO_CART"}
START_DEFAULT = {"bottle_bin": 0.35}           # m; bowl_sink (0.38) and coffee_cart (0.20) keep their own
LIVE_SCENE_ROOT = Path.home() / "wrk/robot_orchard_deploy/holobrain_g1_deploy/sim"      # the kits' working copies
REPO_SCENE_ROOT = Path(__file__).resolve().parents[3] / "scenes"                          # the copies in this repo


def scene_root() -> Path:
    """HOLOBRAIN_SIM_DIR if set, else the live kit folder on the workstation, else the repo's scenes/."""
    if os.environ.get("HOLOBRAIN_SIM_DIR"):
        return Path(os.environ["HOLOBRAIN_SIM_DIR"])
    return LIVE_SCENE_ROOT if LIVE_SCENE_ROOT.is_dir() else REPO_SCENE_ROOT


SCENE_ENV_PREFIX = {"bottle_bin": "BOTTLE_BIN_", "bowl_sink": "BOWL_SINK_", "coffee_cart": "COFFEE_CART_"}


def load_scene(name: str, start_distance: float | None = None):
    """Import a real-to-sim scene module (registers its env) and return it."""
    if name not in SCENES:
        raise typer.BadParameter(f"unknown scene {name!r}; one of {sorted(SCENES)}")
    dist = start_distance if start_distance is not None else START_DEFAULT.get(name)
    if dist is not None and START_KNOB[name] not in os.environ:
        os.environ[START_KNOB[name]] = f"{dist:.4f}"
    if START_KNOB[name] in os.environ:
        print(f"[HoloMotion v1.4] {name}: robot start {START_KNOB[name]}={os.environ[START_KNOB[name]]} m")
    scene_dir = scene_root() / name
    if not scene_dir.is_dir():
        raise typer.BadParameter(f"scene folder not found: {scene_dir} (set HOLOBRAIN_SIM_DIR)")
    sys.path.insert(0, str(scene_dir))
    return importlib.import_module(SCENES[name])


class SimControls:
    """Edge-detected sim-only combos (left menu held) + the reset gesture (both triggers)."""

    def __init__(self, agent: HoloMotionV14Agent) -> None:
        self.agent = agent
        self._last = {"save": False, "abort": False, "skip": False, "reset": False, "motion": False}

    def poll(self) -> dict:
        s = self.agent.pico() or {}
        menu = bool(s.get("left_menu"))
        now = {"save": menu and bool(s.get("A")), "abort": menu and bool(s.get("B")), "skip": menu and bool(s.get("Y")),
               "reset": s.get("left_trigger", 0.0) > 0.9 and s.get("right_trigger", 0.0) > 0.9,
               # the motion-tracking button on its own (not the left menu + Y combo): --record-on-motion
               "motion": not menu and bool(s.get(self.agent.motion_button))}
        edges = {k: v and not self._last[k] for k, v in now.items()}
        self._last = now
        return edges


def _load_sonic_config() -> dict:
    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config


def _check_dataset_dir(save_dir: str) -> None:
    """The LeRobot exporter counts EVERY .mp4 / .parquet under the dataset folder when it saves an episode and aborts on
    a mismatch -- after the episode is written, losing its replay log (2026-09-29: renders put in level-0/renders/).
    Refuse to start while the folder holds such files outside videos/ and data/."""
    root = Path(save_dir)
    if not root.is_dir():
        return
    # a run that ended before its first save leaves the exporter's skeleton (meta/info.json + modality.json, empty
    # folders): no tasks.jsonl yet, so the next start takes it for a corrupted dataset and stops (2026-09-30). With
    # no episode in it there is nothing to keep: start the dataset afresh.
    files = [f for f in root.rglob("*") if f.is_file()]
    if files and not (root / "meta" / "tasks.jsonl").exists() and \
            {f.relative_to(root).as_posix() for f in files} <= {"meta/info.json", "meta/modality.json"}:
        import shutil
        shutil.rmtree(root)
        print(f"[Record] {root}: an earlier run ended before saving an episode; its empty dataset skeleton was removed")
        return
    stray = [f for f in list(root.rglob("*.mp4")) + list(root.rglob("*.parquet"))
             if f.relative_to(root).parts[0] not in ("videos", "data")]
    if stray:
        listing = "\n  ".join(str(f) for f in stray[:10])
        raise typer.BadParameter(f"the dataset folder holds videos/parquet files the recorder would count; move them "
                                 f"out of {root} first:\n  {listing}")


def _init_exporter(save_dir, task_prompt, robot_model, obj_names, joint_names, ego_view_shape, realformat: bool = True):
    """The decoupled-WBC LeRobot schema (as the psi0 datasets) plus the v1.4 controller's own signals, plus (realformat)
    the real G1 recorder's extra fields (teleop/holomotion_v14/realformat.py)."""
    import json
    from decoupled_wbc.data.exporter import Gr00tDataExporter
    from decoupled_wbc.data.utils import get_dataset_features, get_modality_config
    from simple.teleop.holomotion_v14 import realformat as RF

    features = get_dataset_features(robot_model)
    set_ego_view_feature_shape(features, ego_view_shape)
    validate_existing_ego_view_feature_shape(save_dir, ego_view_shape)
    features["observation.state"]["names"] = joint_names
    if obj_names:
        names = [f"{n}.{s}" for n in obj_names for s in ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z")]
        features["observation.object_poses"] = {"dtype": "float64", "shape": (len(obj_names) * 7,), "names": names}
    features["teleop.latest_obs"] = {"dtype": "float32", "shape": (65,), "names": None}
    features["policy.mode"] = {"dtype": "int64", "shape": (1,), "names": ["0=velocity,1=motion"]}
    features["policy.raw_action"] = {"dtype": "float32", "shape": (29,), "names": None}
    features["policy.target_real"] = {"dtype": "float32", "shape": (29,), "names": None}
    if realformat:
        RF.add_features(features)
    exporter = Gr00tDataExporter.create(save_root=save_dir, fps=50, features=features,
                                        modality_config=get_modality_config(robot_model), task=task_prompt)
    spec = Path(save_dir) / "meta" / "realformat.json"
    if realformat and not spec.exists():
        spec.write_text(json.dumps({**RF.REALFORMAT, "added_features": list(RF.FEATURES)}, indent=2))
    return exporter


def _env_config_json(task) -> str:
    import json
    from simple.utils import NumpyArrayEncoder
    return json.dumps(task.state_dict(), cls=NumpyArrayEncoder)


def _build_frame(robot_model, obj_names, observation, privileged_info, action, phys=None):
    proprio = privileged_info["proprio"]
    from simple.teleop.holomotion.protocol import DEX3_NATURAL_TO_MJCF
    inv = np.argsort(DEX3_NATURAL_TO_MJCF)                     # MJCF -> natural (thumb, middle, index)
    lh = np.asarray(action["left_hand_q"], dtype=np.float64)[inv]
    rh = np.asarray(action["right_hand_q"], dtype=np.float64)[inv]
    action_q = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=np.asarray(action["target_q"], dtype=np.float64),
        left_hand_actuated_joint_values=lh, right_hand_actuated_joint_values=rh)
    frame = {
        "observation.images.ego_view": observation["head_stereo_left"],
        "observation.state": np.asarray(observation["joint_qpos"], dtype=np.float64),
        "observation.eef_state": np.zeros(14, dtype=np.float64),
        "action": np.asarray(action_q, dtype=np.float64),
        "action.eef": np.zeros(14, dtype=np.float64),
        "observation.img_state_delta": np.array([0.0], dtype=np.float32),
        "teleop.navigate_command": np.asarray(action["navigate_cmd"], dtype=np.float64),
        "teleop.base_height_command": np.zeros(1, dtype=np.float64),
        "observation.base_pose": np.asarray(proprio["floating_base_pose"], dtype=np.float64),
        "observation.base_vel": np.asarray(proprio["floating_base_vel"], dtype=np.float64),
        "observation.torso_rpy_command": np.zeros(3, dtype=np.float64),
        "teleop.latest_obs": np.asarray(action["latest_obs"], dtype=np.float32),
        "policy.mode": np.array([1 if action["policy_mode"] == "motion" else 0], dtype=np.int64),
        "policy.raw_action": np.asarray(action["raw_action"], dtype=np.float32),
        "policy.target_real": np.asarray(action["target_real"], dtype=np.float32),
    }
    if phys:                                                   # the real recorder's extra fields (realformat)
        frame.update(phys)
    if obj_names:                                              # fixed slots; an absent distractor is NaN
        nan = np.full(7, np.nan)
        frame["observation.object_poses"] = np.concatenate(
            [np.asarray(privileged_info[n], dtype=np.float64) if n in privileged_info else nan for n in obj_names])
    return frame


# --random-init-pose ranges (uniform +-, at --init-pose-scale 1): joint offsets from the start pose by body part (rad),
# and the base (m, deg) on top of the setup's robot start
INIT_POSE_NOISE = dict(leg=0.03, waist=0.05, arm=0.15, base_xy=0.02, base_yaw_deg=3.0)


def _draw_init_pose(rng, robot, scale: float) -> dict:
    """One random start-pose offset: body joints (MJCF order, as robot.body_joint_index) and base (dx, dy, dyaw_deg)."""
    names = list(robot.joint_names[:len(robot.body_joint_index)])

    def part(n: str) -> str:
        return "leg" if any(k in n for k in ("hip", "knee", "ankle")) else "waist" if "waist" in n else "arm"

    half = np.array([INIT_POSE_NOISE[part(n)] for n in names]) * scale
    dq = np.round(rng.uniform(-half, half), 4)
    b = INIT_POSE_NOISE
    base = [round(float(rng.uniform(-1, 1) * b["base_xy"] * scale), 4), round(float(rng.uniform(-1, 1) * b["base_xy"] * scale), 4),
            round(float(rng.uniform(-1, 1) * b["base_yaw_deg"] * scale), 3)]
    return dict(scale=float(scale), joint_names=names, dq=dq.tolist(), base=base)


def _clip_to_range(m, joint_ids, q):
    lo = np.where(m.jnt_limited[joint_ids] > 0, m.jnt_range[joint_ids, 0], -np.inf)
    hi = np.where(m.jnt_limited[joint_ids] > 0, m.jnt_range[joint_ids, 1], np.inf)
    return np.clip(q, lo, hi)


def _perturb_hanging(sonic_env, init_pose: dict) -> None:
    """--random-init-pose without --quick-start: offset the joints of the robot hanging on the leash, at rest."""
    import mujoco
    m, d, robot = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData, sonic_env.task.robot
    adr = m.jnt_qposadr[robot.body_joint_index]
    d.qpos[adr] = _clip_to_range(m, robot.body_joint_index, d.qpos[adr] + np.asarray(init_pose["dq"]))
    d.qvel[m.jnt_dofadr[robot.body_joint_index]] = 0.0
    mujoco.mj_forward(m, d)


def _stand_on_floor(sonic_env, agent, init_pose: dict | None = None) -> np.ndarray:
    """Quick start: the robot at the policy's default pose (plus the --random-init-pose offset), feet on the floor, at
    rest, no leash (x, y and yaw as the setup spawned it, plus the offset). Returns the body joints placed (MJCF order)."""
    import mujoco
    m, d, robot = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData, sonic_env.task.robot
    if getattr(robot, "elastic_band", None) is not None:
        robot.elastic_band.enable = False
        d.xfrc_applied[robot.band_attached_link] = 0.0
    root = int(m.body_rootid[m.body("pelvis").id])
    joints = [j for j in range(m.njnt) if int(m.body_rootid[m.jnt_bodyid[j]]) == root]
    free = next(j for j in joints if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
    a = int(m.jnt_qposadr[free])
    w, x, y, z = d.qpos[a + 3:a + 7]
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    q = agent.default_real[agent.real_to_mjcf].astype(np.float64)
    if init_pose is not None:
        d.qpos[a] += init_pose["base"][0]
        d.qpos[a + 1] += init_pose["base"][1]
        yaw += np.radians(init_pose["base"][2])
        q = _clip_to_range(m, robot.body_joint_index, q + np.asarray(init_pose["dq"]))
    d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    d.qpos[m.jnt_qposadr[robot.body_joint_index]] = q
    ndof = {int(mujoco.mjtJoint.mjJNT_FREE): 6, int(mujoco.mjtJoint.mjJNT_BALL): 3}
    for j in joints:
        d.qvel[m.jnt_dofadr[j]:m.jnt_dofadr[j] + ndof.get(int(m.jnt_type[j]), 1)] = 0.0
    d.qpos[a + 2] = 1.0
    mujoco.mj_forward(m, d)
    feet = [g for g in range(m.ngeom) if int(m.body_rootid[m.geom_bodyid[g]]) == root
            and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and (m.geom_contype[g] or m.geom_conaffinity[g])]
    low = min(float(d.geom_xpos[g][2] - m.geom_size[g][0]) for g in feet)
    d.qpos[a + 2] += 0.001 - low                          # the lowest foot sphere 1 mm above the floor
    mujoco.mj_forward(m, d)
    return q


def _stream_status(agent, rec_state, rec_frames: int, episodes_saved: int, num_episodes: int, setup: dict,
                   privileged_info, quick: bool, record_on_motion: bool = False) -> list:
    """Overlay lines for the headset stream: (text, BGR)."""
    red, white, grey, yellow = (60, 60, 255), (255, 255, 255), (190, 190, 190), (60, 220, 255)
    if rec_state == RecordingState.RECORDING:
        top = (f"REC {rec_frames / 50.0:5.1f} s", red)
    elif agent.quick_active:
        top = ("starting ...", yellow)
    elif record_on_motion:
        top = (f"not recording  ({agent.motion_button}: motion + record)", yellow)
    else:
        top = ("not recording  (L-menu + A: record)", grey)
    mode = {"WAIT": "WAIT (L3: stand)", "MOVE_TO_DEFAULT": "STAND (A: walk)", "ZERO_TORQUE": "ZERO TORQUE",
            "EMERGENCY_STOP": "E-STOP"}.get(agent.main_state)
    if mode is None:
        mode = "MOTION" if agent.node.current_policy_mode == "motion" else f"WALK ({agent.motion_button}: motion)"
    lines = [top, (f"{mode}   saved {episodes_saved}/{num_episodes}   setup {setup.get('index', '-')}", white)]
    prog = (privileged_info.get("task_progress") or {}) if isinstance(privileged_info, dict) else {}
    gates = [k for k in ("at_bowl", "grasped", "at_bin", "at_basin", "at_cart", "placed") if k in prog]
    if gates:
        lines.append(("  ".join(f"{k} [{'x' if prog.get(k) else ' '}]" for k in gates), white))
    lines.append(("L-menu + A save  + B drop  + Y new setup" if rec_state == RecordingState.RECORDING
                  else "L-menu + A record  + Y new setup", grey))
    return lines


def _save_episode_meta(exporter, episode_index: int, extra: dict):
    """Add fields to the episode's line in meta/episodes.jsonl."""
    import json
    from simple.utils import NumpyArrayEncoder
    meta_file = exporter.root / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return
    lines = [json.loads(line) for line in meta_file.read_text().splitlines() if line.strip()]
    for k, v in extra.items():
        lines[episode_index][k] = v if isinstance(v, str) else json.dumps(v, cls=NumpyArrayEncoder)
    meta_file.write_text("".join(json.dumps(e) + "\n" for e in lines))


def main(
    scene: Annotated[str, typer.Option(help=f"real-to-sim scene: {', '.join(SCENES)}")] = "bottle_bin",
    env_id: Annotated[str, typer.Option(help="any SIMPLE G1Sonic env id instead of --scene")] = "",
    start_distance: Annotated[float, typer.Option(help="robot start: feet to the table/counter/cart (m); -1 = scene default for v1.4 teleop")] = -1.0,
    reference_uri: Annotated[str, typer.Option(help="ZMQ endpoint of the v1.4 reference publisher")] = DEFAULT_REFERENCE_URI,
    headless: Annotated[bool, typer.Option()] = False,
    record: Annotated[bool, typer.Option()] = False,
    save_dir: Annotated[str, typer.Option()] = "data/teleop_holomotion_v14",
    num_episodes: Annotated[int, typer.Option()] = 100,
    max_episode_steps: Annotated[int, typer.Option()] = 30000,
    dr_level: Annotated[int, typer.Option()] = 0,
    auto_stand: Annotated[bool, typer.Option(help="stand up (MOVE_TO_DEFAULT) automatically after each reset")] = False,
    randomize: Annotated[bool, typer.Option(help="a new random setup (robot, target, bin/cart, distractors) every episode")] = True,
    seed: Annotated[int, typer.Option(help="first setup's seed (the n-th setup uses seed + n); -1 = pick one")] = -1,
    max_distractors: Annotated[int, typer.Option(help="distractors per setup: 0..this (lv3: exactly this)")] = 3,
    setup_ranges: Annotated[str, typer.Option(help="random setup ranges: teleop (robot +-8/+-10 cm +-10 deg, target, bin/cart, "
                                                   "0..3 distractors, fixed table) or lv3 (the kits' level-3 eval sets: robot "
                                                   "+-10 cm back/forward +-5 cm sideways, the level-3 target region, table / "
                                                   "counter / cart-box height +-4 cm, 3 distractors; records into level-3)")] = "lv3",
    exact_log: Annotated[bool, typer.Option(help="with --record: save the bit-exact replay log of every episode")] = True,
    realformat: Annotated[bool, typer.Option(help="with --record: also record the real G1 recorder's fields (joint velocities and efforts, root pose and velocity, pelvis IMU, hand velocities and efforts, PD gains, the reference queue and the HoloMotion observation terms), as scripts/holomotion_sim_realformat.py derives them")] = True,
    save_model: Annotated[bool, typer.Option(help="with the exact log: also save the compiled MuJoCo model (.mjb, ~125 MB: meshes + collision trees)")] = False,
    backpack_kg: Annotated[float, typer.Option(help="HoloMotion's G1 backpack body on the robot, this mass (kg; 3.2 = what the backpack model was trained for, 0 = none)")] = 3.2,
    motion_model: Annotated[str, typer.Option(help="motion-tracking policy: backpack (v1.4.1 BrainCo 3.2 kg backpack, model_22000) or public (v1.4.1, model_16200)")] = "backpack",
    robot_kind: Annotated[str, typer.Option("--robot", help="stereo: the G1 with the HBVCAM stereo head camera (third_party/hbvcam_stereo); stock: SIMPLE's G1")] = "stereo",
    stream_camera: Annotated[str, typer.Option(help="headset picture: pinhole (the stereo camera's rectified left eye, 1280x720, 104.7 x 72.2 deg), fisheye (both eyes' raw fisheye images side by side, 2560x720, like the real camera) or head (the scene's D455 head camera)")] = "pinhole",
    head_tilt_deg: Annotated[float, typer.Option(help="pitch the head cameras (the HBVCAM stereo camera the headset shows and SIMPLE's head camera) this many degrees further down (+) or up (-); 0 = the calibrated mount")] = 10.0,
    motion_model_bundle: Annotated[str, typer.Option(help="another backpack model bundle dir (model_22000.onnx [+ config.yaml]); overrides --motion-model")] = "",
    quick_start: Annotated[bool, typer.Option(help="every scene: standing, no leash, walking policy on, recording starts by itself (--no-quick-start: hang on the leash, stand up by hand)")] = True,
    record_camera: Annotated[str, typer.Option(help="image saved as ego_view: auto (the headset's camera: pinhole with --stream-camera pinhole, fisheye with fisheye, head with head), head (SIMPLE's head camera, 640x360), pinhole (1280x720) or fisheye (both eyes, 2560x720)")] = "auto",
    fisheye_face_px: Annotated[int, typer.Option(help="--stream-camera fisheye: size of the 90-degree face renders (960 ~ the lens's own sharpness; lower = cheaper, softer)")] = 960,
    record_on_motion: Annotated[bool, typer.Option(help="start each recording (the first and every later one) when you press the motion-tracking button (Y), not at the start of the scene")] = False,
    auto_motion: Annotated[bool, typer.Option(help="quick start also switches to motion tracking (presses Y) instead of leaving it to the operator")] = False,
    stream_view: Annotated[str, typer.Option(help="headset view: single (one picture), mono (left camera in both halves) or stereo")] = "single",
    button_map: Annotated[str, typer.Option(help="sim: A walk, Y motion tracking, B free; robot: the v1.4.1 map (A walk, B motion, Y back to walking)")] = "sim",
    random_init_pose: Annotated[bool, typer.Option(help="every scene init: a small random robot pose (base, joints), so each episode starts from a different state (--no-random-init-pose: the policy's default pose)")] = True,
    init_pose_scale: Annotated[float, typer.Option(help="scale of the --random-init-pose ranges (1 = arms 0.15 rad, waist 0.05, legs 0.03, base 2 cm / 3 deg)")] = 1.0,
    pico_stream: Annotated[bool, typer.Option(help="stream the head camera to the headset")] = True,
    stream_port: Annotated[int, typer.Option(help="port the XRoboToolkit app connects to for the video (the app uses 13579)")] = 13579,
    max_seconds: Annotated[float, typer.Option(help="stop after this long (0 = run until Ctrl-C); for tests")] = 0.0,
    log_every: Annotated[float, typer.Option(help="seconds between status lines")] = 2.0,
):
    if setup_ranges not in ("teleop", "lv3"):
        raise typer.BadParameter("--setup-ranges is teleop or lv3")
    if setup_ranges == "lv3" and randomize and dr_level == 0:
        dr_level = 3                                     # the dataset folder (level-3) and the metadata; sampling is ours
    target, mod = None, None
    if not env_id:
        mod = load_scene(scene, None if start_distance < 0 else start_distance)
        env_id, target = mod.ENV_ID, getattr(mod, "TARGET", None)
    else:
        scene = ""
    from simple.teleop.holomotion_v14 import BACKPACK_MODELS_DIR, MODELS_DIR
    if motion_model_bundle:
        from simple.teleop.holomotion_v14.robot_variants import backpack_models_dir
        models_dir = backpack_models_dir(motion_model_bundle)
    elif motion_model in ("backpack", "public"):
        models_dir = BACKPACK_MODELS_DIR if motion_model == "backpack" else MODELS_DIR
        if not (models_dir / "motion_tracking_model").is_dir():
            raise typer.BadParameter(f"--motion-model {motion_model}: {models_dir} not set up (see docs/holomotion_v14_teleop.md)")
    else:
        raise typer.BadParameter("--motion-model is backpack or public")
    os.environ["HOLOMOTION_UNITREE_POLICY_MODELS_DIR"] = str(models_dir)
    print(f"[HoloMotion v1.4] policies from {models_dir} (motion model: {'bundle' if motion_model_bundle else motion_model}, "
          f"robot backpack {backpack_kg:g} kg)")
    if (motion_model == "backpack" or motion_model_bundle) != (backpack_kg > 0):
        print("[HoloMotion v1.4] note: the backpack model is trained for a 3.2 kg backpack and the public one for none; "
              "this run mixes them")
    sonic_config = _load_sonic_config()
    render_hz = 50
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id, sim_mode="mujoco", render_hz=render_hz, physics_dt=sonic_config["SIMULATE_DT"],
                   headless=headless, max_episode_steps=max_episode_steps, sonic_config=sonic_config, target=target,
                   dr_level=dr_level, success_criteria=0.9)
    sonic_env: SonicLocoManipEnv = env.unwrapped  # type: ignore
    task, robot = sonic_env.task, sonic_env.task.robot
    assert isinstance(robot, G1Sonic)
    from simple.teleop.holomotion_v14.robot_variants import teleop_mjcf, tilt_head_sensor
    robot.mjcf_path = teleop_mjcf(robot_kind, backpack_kg, tilt_deg=head_tilt_deg)
    if head_tilt_deg:
        if not scene:
            raise typer.BadParameter("--head-tilt-deg works with --scene (the scene kits' head-camera pose patch)")
        tilt_head_sensor(task, head_tilt_deg)
        print(f"[HoloMotion v1.4] head cameras pitched {head_tilt_deg:+g} deg ({'down' if head_tilt_deg > 0 else 'up'})")
    print(f"[HoloMotion v1.4] robot: {robot.mjcf_path} ({'G1 + HBVCAM stereo head camera' if robot_kind == 'stereo' else 'stock G1'}, "
          f"{backpack_kg:g} kg backpack)")
    if stream_camera not in ("pinhole", "fisheye", "head"):
        raise typer.BadParameter("--stream-camera is pinhole, fisheye or head")
    if stream_camera in ("pinhole", "fisheye") and robot_kind != "stereo":
        print(f"[HoloMotion v1.4] --stream-camera {stream_camera} needs --robot stereo: streaming the scene's head camera instead")
        stream_camera = "head"
    if record_camera == "auto":
        record_camera = stream_camera                   # record what the headset shows
    if record_camera not in ("head", "pinhole", "fisheye"):
        raise typer.BadParameter("--record-camera is auto, head, pinhole or fisheye")
    if record_camera != "head" and robot_kind != "stereo":
        print(f"[HoloMotion v1.4] --record-camera {record_camera} needs --robot stereo: recording the scene's head camera instead")
        record_camera = "head"
    from simple.teleop.holomotion_v14.hbvcam_views import FisheyeView, PinholeView

    def make_view(kind: str):
        return FisheyeView(sonic_env, face_px=fisheye_face_px) if kind == "fisheye" else PinholeView(sonic_env)

    rec_view = make_view(record_camera) if (record and record_camera != "head") else None
    hb_view = None
    if pico_stream and stream_camera != "head":
        hb_view = rec_view if stream_camera == record_camera and rec_view is not None else make_view(stream_camera)
    next_stream_t = 0.0

    def recorded_view():
        """The recorded camera's image of the current (pre-step) state: (image to save, (left, right) for the stream)."""
        if rec_view is None:
            return None, None
        if record_camera == "fisheye":                          # both eyes fresh every step: a consistent pair
            left, right = rec_view.render()
            return np.hstack([left, right]), (left, right)
        img = rec_view.render()
        return img, (img, None)
    control_decimal = int(round(1 / sonic_config["SIMULATE_DT"] / render_hz))
    assert control_decimal == 4, "the v1.4 policy runs at 50 Hz (4x decimation at 200 Hz physics)"
    control_dt = control_decimal * robot.sim_dt

    if stream_view not in ("single", "mono", "stereo"):
        raise typer.BadParameter("--stream-view is single, mono or stereo")
    if stream_camera == "fisheye":
        stream_view = "stereo"                              # the two fisheye eyes are the frame's two halves
    agent = HoloMotionV14Agent(robot, reference_uri=reference_uri, auto_stand=auto_stand, enable_pico_stream=pico_stream,
                               stream_view=stream_view, button_map=button_map, stream_port=stream_port)
    agent.num_episodes = num_episodes
    controls = SimControls(agent)

    setups = (SceneSetups(scene, mod, task, max_distractors=max_distractors, ranges=setup_ranges)
              if (mod is not None and randomize) else None)
    if setups is not None:
        print(f"[HoloMotion v1.4] random setups: {setup_ranges} ranges {setups.ranges}")
    base_seed = seed if seed >= 0 else int.from_bytes(os.urandom(4), "little") % 1_000_000
    n_setups = 0
    setup = setups.nominal() if setups is not None else (SceneSetups(scene, mod, task).nominal() if mod is not None else {})

    def next_setup():
        nonlocal setup, n_setups
        if setups is not None:
            setup = setups.sample(base_seed + n_setups)
            setup["index"] = n_setups
            n_setups += 1
            setups.apply(setup)
            print(f"[HoloMotion v1.4] setup {setup['index']} ({SceneSetups.describe(setup)})")

    def same_setup():
        if setups is not None:
            setups.apply(setup)

    init_rng = np.random.RandomState((base_seed * 7919 + 17) % 2 ** 32)      # the start-pose offsets, reproducible per --seed
    init_pose = None

    def init_robot(quick: bool) -> None:
        """After env.reset: the start pose (quick start: standing, else hanging on the leash), randomized if asked."""
        nonlocal observation, privileged_info, init_pose
        init_pose = _draw_init_pose(init_rng, robot, init_pose_scale) if random_init_pose else None
        if quick:
            placed = _stand_on_floor(sonic_env, agent, init_pose)
            agent.quick_start(hold_real=placed[agent.mjcf_to_real], auto_motion=auto_motion)
        elif init_pose is not None:
            _perturb_hanging(sonic_env, init_pose)
        if quick or init_pose is not None:
            observation, privileged_info = sonic_env._get_obs(), sonic_env._get_info()
        if init_pose is not None:
            b = init_pose["base"]
            print(f"[HoloMotion v1.4] random start pose: base ({b[0]:+.3f}, {b[1]:+.3f}) m {b[2]:+.1f} deg, "
                  f"joints max |dq| {max(abs(v) for v in init_pose['dq']):.3f} rad")

    next_setup()
    observation, privileged_info = env.reset()
    agent.reset_policy()
    init_robot(quick=quick_start)                 # quick start: standing, no leash, from the first scene on
    obj_names = list(sonic_env.mujoco.mj_objects.keys())
    if setups is not None:
        # the LeRobot schema is fixed, the distractor count is not: slots for the task objects + max_distractors
        obj_names = [n for n in obj_names if not n.startswith("distractor_")] + [f"distractor_{i}" for i in range(max_distractors)]
    exporter = robot_model = None
    xlog = None
    rf_sampler = None
    if record:
        from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
        robot_model = instantiate_g1_robot_model(waist_location="lower_body")
        # a dataset holds one image size: recordings of the stereo camera get their own folder
        run_save_dir = f"{os.path.abspath(save_dir)}/{env_id}/level-{dr_level}" + ("" if record_camera == "head" else f"_{record_camera}")
        _check_dataset_dir(run_save_dir)
        ego_shape = {"fisheye": (720, 2560, 3), "pinhole": (720, 1280, 3)}.get(record_camera, observation["head_stereo_left"].shape)
        exporter = _init_exporter(run_save_dir, task.instruction, robot_model, obj_names, robot.joint_names, ego_shape, realformat)
        if realformat:
            from simple.teleop.holomotion_v14 import realformat as RF
            rf_sampler = RF.LiveSampler(sonic_env)
            print(f"[Record] real-format fields on ({len(RF.FEATURES)} extra columns)")
        print(f"[Record] ego_view = {record_camera} camera, {ego_shape[1]} x {ego_shape[0]}")
        print(f"[Record] saving to {run_save_dir}")
        if exact_log:
            xlog = ExactLog(sonic_env)
            prefix = SCENE_ENV_PREFIX.get(scene, "\0")
            knobs = {k: v for k, v in os.environ.items() if k.startswith(prefix) or k.startswith("SIMPLE_")}
            run_meta = dict(scene=scene, scene_root=str(scene_root()), env_id=env_id, dr_level=dr_level, env_knobs=knobs,
                            setup_ranges=setup_ranges if setups is not None else None,
                            backpack_kg=backpack_kg, robot=robot_kind, head_tilt_deg=head_tilt_deg,
                            stream_camera=stream_camera, record_camera=record_camera,
                            robot_mjcf=robot.mjcf_path, physics_dt=sonic_config["SIMULATE_DT"], render_hz=render_hz,
                            policy_models_dir=os.environ.get("HOLOMOTION_UNITREE_POLICY_MODELS_DIR", ""),
                            velocity_onnx=str(agent.node.velocity_onnx_path), motion_onnx=str(agent.node.motion_onnx_path))

    rec_state = RecordingState.WAITING_FOR_LANDING
    episode_init_pose = None
    episodes_saved, sim_cnt, pbar = 0, 0, None
    t_start = time.monotonic()
    t_log = t_start
    print(f"[HoloMotion v1.4] waiting for the reference publisher at {reference_uri}")
    print(f"  A = walk, {agent.motion_button} = motion tracking; left menu + A = record/save")

    def reset_scene(reason: str, new_setup: bool):
        nonlocal observation, privileged_info, rec_state, sim_cnt, pbar
        if exporter is not None and rec_state == RecordingState.RECORDING:
            exporter.skip_and_start_new_episode()
            print(f"[Record] {reason}: in-progress episode discarded")
        if rf_sampler is not None:
            rf_sampler.discard()
        if xlog is not None:
            xlog.discard()
        if pbar is not None:
            pbar.close()
            pbar = None
        next_setup() if new_setup else same_setup()
        observation, privileged_info = env.reset()
        agent.reset_policy()
        init_robot(quick=quick_start)
        rec_state, sim_cnt = RecordingState.WAITING_FOR_LANDING, 0
        print(f"[HoloMotion v1.4] scene reset ({reason}{'; quick start' if quick_start else ''})")

    try:
        while not max_seconds or time.monotonic() - t_start < max_seconds:
            step_start = time.monotonic()
            ego_img, rec_pair = recorded_view()
            action = agent.get_action(observation, instruction=task.instruction, privileged_info=privileged_info)
            frame_obs = observation if ego_img is None else {**observation, "head_stereo_left": ego_img}
            frame_inputs = {"observation": frame_obs, "action": action, "privileged_info": privileged_info}
            if rf_sampler is not None:                                   # every step: the first recorded frame is decided after it
                frame_inputs["phys"] = rf_sampler.sample(action)          # the pre-step state, as observation.state
            if xlog is not None:
                xlog.before_step()
            observation, reward, terminated, truncated, privileged_info = env.step(action)
            if not headless:
                sonic_env.update_viewer()
            ev = controls.poll()
            quick_ready, quick_engaged = agent.quick_ready, agent.quick_engaged
            if (quick_engaged and random_init_pose and not record_on_motion and agent.quick_engaged_at > 1
                    and rec_state == RecordingState.WAITING_FOR_LANDING):
                # the PICO connected after this scene started (the launch scene): start over from a fresh random pose
                reset_scene("PICO controllers connected", new_setup=False)
                continue
            # quick start records from the first step the controllers are there (frame 0 = the start pose, randomized
            # with --random-init-pose); with --auto-motion and a fixed start pose, from motion-on as before
            auto_record = quick_start and (quick_ready if (auto_motion and not random_init_pose) else quick_engaged)
            if record_on_motion:                                # --record-on-motion: the operator's Y starts it
                auto_record = ev["motion"] or (auto_motion and quick_ready)   # (--auto-motion presses it internally)

            if ev["reset"]:
                reset_scene("both triggers", new_setup=False)
                continue
            if ev["skip"]:
                reset_scene("left menu + Y", new_setup=True)
                continue

            if exporter is not None:
                if ev["abort"] and rec_state == RecordingState.RECORDING:
                    exporter.skip_and_start_new_episode()
                    if rf_sampler is not None:
                        rf_sampler.discard()
                    if xlog is not None:
                        xlog.discard()
                    rec_state = RecordingState.WAITING_FOR_LANDING
                    if pbar is not None:
                        pbar.close()
                        pbar = None
                    print("[Record] episode abandoned (left menu + B)")
                elif (ev["save"] or auto_record) and rec_state == RecordingState.WAITING_FOR_LANDING:
                    rec_state = RecordingState.RECORDING
                    episode_init_pose = init_pose
                    if xlog is not None:
                        xlog.start()
                    pbar = tqdm(desc=f"Recording episode {episodes_saved + 1}", unit="frame", leave=False)
                    why = ("" if not auto_record else f" ({agent.motion_button} pressed)" if record_on_motion
                           else " (auto, from the random start pose)" if random_init_pose else " (auto, quick start)")
                    print(f"[Record] recording{why} "
                          "(left menu + A to save, left menu + B to abandon)")
                elif ev["save"] and rec_state == RecordingState.RECORDING:
                    rec_state = RecordingState.EPISODE_DONE
                if rec_state == RecordingState.RECORDING:
                    exporter.add_frame(_build_frame(robot_model, obj_names, **frame_inputs))
                    if rf_sampler is not None:
                        rf_sampler.frame_recorded()
                    if xlog is not None:
                        xlog.add_frame(frame_inputs["action"])
                    if pbar is not None:
                        pbar.update(1)
                    if terminated or truncated:
                        rec_state = RecordingState.EPISODE_DONE
                        print(f"[Record] task {'succeeded' if terminated else 'timed out'}; saving")
                if rec_state == RecordingState.EPISODE_DONE:
                    ep_idx = exporter.episode_buffer["episode_index"]
                    extra = {"scene_setup": setup, "init_pose": episode_init_pose}
                    if rf_sampler is not None:
                        RF.finish_episode(exporter.episode_buffer, rf_sampler)
                        extra["realformat"] = RF.episode_meta(dict(backpack_kg=backpack_kg, head_tilt_deg=head_tilt_deg), setup,
                                                              xlog.model_sha if xlog is not None else "")
                    exporter.save_episode()
                    _save_episode_env_config(exporter, task, ep_idx)
                    if xlog is not None:
                        log_path = Path(run_save_dir) / "replay" / f"episode_{ep_idx:06d}.npz"
                        xlog.save(log_path, setup, dict(run_meta, episode_index=int(ep_idx), init_pose=episode_init_pose,
                                                        environment_config=_env_config_json(task)), save_model=save_model)
                        extra["exact_log"] = str(log_path.relative_to(run_save_dir))
                    _save_episode_meta(exporter, ep_idx, extra)
                    episodes_saved += 1
                    agent.episodes_saved = episodes_saved
                    print(f"[Record] episode {episodes_saved} saved")
                    if episodes_saved >= num_episodes:
                        break
                    reset_scene("episode saved", new_setup=True)
                    continue

            if pico_stream:
                rec_frames = int(exporter.episode_buffer["size"]) if exporter is not None and exporter.episode_buffer else 0
                agent.stream_status = _stream_status(agent, rec_state, rec_frames, episodes_saved, num_episodes, setup,
                                                     privileged_info, quick_start, record_on_motion)
            if hb_view is not None and agent.streaming:
                pair = None
                if hb_view is rec_view and rec_pair is not None:   # the recorded view, already rendered this step
                    pair = rec_pair
                elif stream_camera == "fisheye":                # one eye per step, the newest pair every step
                    pair = hb_view.step()
                elif time.monotonic() >= next_stream_t:
                    pair = (hb_view.render(), None)
                now_s = time.monotonic()
                if pair is not None and now_s >= next_stream_t:  # headset frames at the stream rate: 30 fps on average
                    next_stream_t = max(next_stream_t + 1.0 / 30.0, now_s - 1.0 / 30.0)   # (steps are 20 ms apart)
                    agent.update_render_caches({"head_stereo_left": pair[0], "head_stereo_right": pair[1]})
            else:
                agent.update_render_caches(observation)

            now = time.monotonic()
            if log_every and now >= t_log:
                prog = (privileged_info.get("task_progress") or {}) if isinstance(privileged_info, dict) else {}
                gates = " ".join(k for k in ("at_bowl", "grasped", "at_bin", "at_basin", "at_cart", "placed") if prog.get(k))
                base = privileged_info["proprio"]["floating_base_pose"]
                w, x, y, z = base[3:7]
                yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                st = agent.node.runtime.state
                print(f"[v1.4 t={now - t_start:6.1f}s] main={agent.main_state:15s} policy={agent.policy_mode:15s} "
                      f"cmd=({st.vx:+.2f},{st.vy:+.2f},{st.vyaw:+.2f}) ref={agent.sub.n_obs:6d} "
                      f"pelvis=({base[0]:+.2f},{base[1]:+.2f},{base[2]:.2f}) yaw={yaw:+6.1f} "
                      f"run={agent._last.get('run_ms', 0):5.1f}ms gates=[{gates}]", flush=True)
                t_log = now + log_every

            sleep = control_dt - (time.monotonic() - step_start)
            if sleep > 0:
                time.sleep(sleep)
            sim_cnt += 1
    except KeyboardInterrupt:
        print("interrupted")
    except Exception:                                          # os._exit below would swallow the traceback
        import traceback
        traceback.print_exc()
    finally:
        if exporter is not None and rec_state == RecordingState.RECORDING:
            try:
                ep_idx = exporter.episode_buffer["episode_index"]
                extra = {"scene_setup": setup, "init_pose": episode_init_pose}
                if rf_sampler is not None:
                    RF.finish_episode(exporter.episode_buffer, rf_sampler)
                    extra["realformat"] = RF.episode_meta(dict(backpack_kg=backpack_kg, head_tilt_deg=head_tilt_deg), setup,
                                                          xlog.model_sha if xlog is not None and xlog.recording else "")
                exporter.save_episode()
                if xlog is not None and xlog.recording:
                    log_path = Path(run_save_dir) / "replay" / f"episode_{ep_idx:06d}.npz"
                    xlog.save(log_path, setup, dict(run_meta, episode_index=int(ep_idx), init_pose=episode_init_pose,
                                                    environment_config=_env_config_json(task)), save_model=save_model)
                    extra["exact_log"] = str(log_path.relative_to(run_save_dir))
                _save_episode_meta(exporter, ep_idx, extra)
                print("[Record] in-progress episode saved on exit")
            except Exception as exc:  # noqa: BLE001
                print(f"[Record] could not save the in-progress episode: {exc}")
        agent.close()
        env.close()
        os._exit(0)


def cli():
    typer.run(main)


if __name__ == "__main__":
    cli()
