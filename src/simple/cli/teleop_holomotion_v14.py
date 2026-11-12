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
  left menu + both triggers   reset the scene (same setup: a retry); a trigger alone = the Revo 2 pinch pose

Every episode starts from a new random setup (robot start, target, bin / cart, distractors; --no-randomize for the
nominal scene), drawn from its own seed. With --record each saved episode also gets a bit-exact log
(replay/episode_XXXXXX.npz next to the dataset): the start state, every physics input and the setup, so
`python -m simple.cli.replay_holomotion_v14` replays it to the last bit.

--quick-start (the default; --no-quick-start for the leash): every scene -- the first one at launch, then after each
save, left menu + Y or left menu + both triggers -- starts with the robot standing on the floor, no leash, in walking mode (the
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
from simple.teleop.holomotion_v14 import node_state as NS
from simple.teleop.holomotion_v14.exact_log import ExactLog
from simple.teleop.holomotion_v14.scene_setup import SceneSetups, spawn_shift
from simple.teleop.holomotion_v14.slices import SLICES, SliceCut
from simple.teleop.holomotion_v14.grasp_start import (NUDGE_WALK_CLEAR_M, GripLatch, apply_pose, arms_to_walk_default, body_joint_qpos, foot_on_pedal,
                                                      hold_plate, nudge_back_from_edge, nudge_clear)
from simple.teleop.holomotion_v14.hand_skeleton import HANDS_COLUMN

SCENES = {"bottle_bin": "bottle_bin_task", "bowl_sink": "bowl_sink_task", "coffee_cart": "coffee_cart_task", "kitchen": "kitchen_task"}
# Where the robot starts, as the scene's own knob (front of the feet -> first obstacle; everything but the robot moves).
# The scenes' defaults were fitted to decoupled-WBC replays that start with the hands raised above the table. The v1.4
# controller stands up with its forearms at table height, so at bottle_bin's 3 cm the hands land on the table and the
# robot leans on it; like the real v1.4 collection, the operator starts clear and walks in.
START_KNOB = {"bottle_bin": "BOTTLE_BIN_ROBOT_TO_EDGE", "bowl_sink": "BOWL_SINK_ROBOT_TO_EDGE",
              "coffee_cart": "COFFEE_CART_ROBOT_TO_CART", "kitchen": "KITCHEN_ROBOT_TO_EDGE"}
START_DEFAULT = {"bottle_bin": 0.35, "kitchen": 0.35}           # m; bowl_sink (0.38) and coffee_cart (0.20) keep their own
LIVE_SCENE_ROOT = Path.home() / "wrk/robot_orchard_deploy/holobrain_g1_deploy/sim"      # the kits' working copies
REPO_SCENE_ROOT = Path(__file__).resolve().parents[3] / "scenes"                          # the copies in this repo


def scene_root() -> Path:
    """HOLOBRAIN_SIM_DIR if set, else the live kit folder on the workstation, else the repo's scenes/."""
    if os.environ.get("HOLOBRAIN_SIM_DIR"):
        return Path(os.environ["HOLOBRAIN_SIM_DIR"])
    return LIVE_SCENE_ROOT if LIVE_SCENE_ROOT.is_dir() else REPO_SCENE_ROOT


SCENE_ENV_PREFIX = {"bottle_bin": "BOTTLE_BIN_", "bowl_sink": "BOWL_SINK_", "coffee_cart": "COFFEE_CART_", "kitchen": "KITCHEN_"}


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
    if not scene_dir.is_dir() and (REPO_SCENE_ROOT / name).is_dir():     # no such kit in the chosen folder: this repo's copy (scenes/<name>)
        scene_dir = REPO_SCENE_ROOT / name
    if not scene_dir.is_dir():
        raise typer.BadParameter(f"scene folder not found: {scene_dir} (set HOLOBRAIN_SIM_DIR)")
    sys.path.insert(0, str(scene_dir))
    return importlib.import_module(SCENES[name])


class SimControls:
    """Edge-detected sim-only combos (left menu held), the reset gesture included (left menu + both triggers)."""

    def __init__(self, agent: HoloMotionV14Agent) -> None:
        self.agent = agent
        self._last = {"save": False, "abort": False, "skip": False, "reset": False, "motion": False}

    def poll(self) -> dict:
        s = self.agent.pico() or {}
        menu = bool(s.get("left_menu"))
        now = {"save": menu and bool(s.get("A")), "abort": menu and bool(s.get("B")), "skip": menu and bool(s.get("Y")),
               "reset": menu and s.get("left_trigger", 0.0) > 0.9 and s.get("right_trigger", 0.0) > 0.9,   # 2026-10-26: with the left menu (the Revo 2 pinch is on the triggers)
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
    # a run that ended before its first save leaves the exporter's skeleton: meta/ files written at startup (info.json,
    # modality.json, realformat.json, holomotion_raw.json, ...) and empty folders, no tasks.jsonl yet. The next start
    # takes it for a corrupted dataset and stops (2026-09-30; again 2026-10-18, when the startup meta files had grown
    # past the list this check used to accept). The rule now: nothing outside meta/ and no episode file in it -> no
    # episode was ever saved, nothing to keep: start the dataset afresh.
    files = [f for f in root.rglob("*") if f.is_file()]
    rel = [f.relative_to(root) for f in files]
    # 2026-10-26: a save cut short (Ctrl-C while a long episode was being written) leaves meta/tasks.jsonl too -- LeRobot
    # writes it when a save starts, meta/episodes.jsonl only when one completes -- and the next start failed with "Failed to
    # resume from corrupted dataset". So: no episodes.jsonl and nothing outside meta/ -> no episode was ever saved.
    episode_meta = {"episodes.jsonl", "episodes_stats.jsonl", "stats.json"}
    if not (root / "meta" / "episodes.jsonl").exists():
        if all(r.parts[0] == "meta" and len(r.parts) == 2 and r.name not in episode_meta for r in rel):
            import shutil
            shutil.rmtree(root)
            print(f"[Record] {root}: an earlier run ended before an episode was saved; its empty dataset skeleton "
                  f"({', '.join(sorted(r.name for r in rel)) or 'empty folders'}) was removed")
            return
        listing = "\n  ".join(str(r) for r in rel[:10])
        raise typer.BadParameter(f"{root} has no saved episode (no meta/episodes.jsonl) but holds other files, so LeRobot cannot "
                                 f"resume it and the recorder will not delete it. Move it away or use another --save-dir:\n  {listing}")
    stray = [f for f in list(root.rglob("*.mp4")) + list(root.rglob("*.parquet"))
             if f.relative_to(root).parts[0] not in ("videos", "data")]
    if stray:
        listing = "\n  ".join(str(f) for f in stray[:10])
        raise typer.BadParameter(f"the dataset folder holds videos/parquet files the recorder would count; move them "
                                 f"out of {root} first:\n  {listing}")


RAW_MAX_FRAMES = 16                # reference frames kept per 50 Hz tick, the NEWEST ones (the publisher runs at <= 70 Hz: 1-2 per tick; the first tick
                                   # after a start can carry a backlog; the node's queue only needs the last 11 frames)
LAYOUT_FILE = Path(__file__).resolve().parents[1] / "teleop" / "holomotion_v14" / "holomotion_obs_layout.json"   # baked term layouts (probe run)
QUAT = ["w", "x", "y", "z"]; XYZ3 = ["x", "y", "z"]
LATEST_OBS_LAYOUT = "dof_pos[0:29] dof_vel[29:58] root_pos[58:61] root_rot_wxyz[61:65], joints in the robot's complete_dof_order"


def _real_dof_names() -> list[str]:
    from simple.teleop.holomotion_v14.sim_node import robot_yaml
    return list(robot_yaml()["complete_dof_order"])


def _obs_layout() -> dict:
    """The per-term layout of each model's observation vector: {mode: {terms: [[name, start, end]], dof_names_onnx, dim}}.
    Baked by a probe recording (HOLOMOTION_DUMP_OBS_LAYOUT=<path> while recording with motion tracking on); the live builder
    layout is checked against it at run time."""
    import json
    return json.load(open(LAYOUT_FILE)) if LAYOUT_FILE.exists() else {}


def _raw_features(policy_obs_dim: int) -> dict:
    """The HoloMotion v1.4 interface, recorded per 50 Hz control tick with HoloMotion's own names (2026-10-16):
    MotionReference (what the operator / a VLA provides), LowState and VRState (what the robot provides), the observation
    terms the policy consumed, and the node's outputs (/humanoid/action, /humanoid/kps, /humanoid/kds)."""
    from simple.agents.holomotion_v14_agent import RAW_PICO_NAMES, VRSTATE_FIELDS
    real = _real_dof_names(); layout = _obs_layout()
    f = {
        # ---- MotionReference (holomotion_interfaces/msg/MotionReference): the newest frame fed this tick
        "holomotion.MotionReference.dof_pos": {"dtype": "float32", "shape": (29,), "names": real},
        "holomotion.MotionReference.dof_vel": {"dtype": "float32", "shape": (29,), "names": real},
        "holomotion.MotionReference.root_pos": {"dtype": "float32", "shape": (3,), "names": XYZ3},
        "holomotion.MotionReference.root_rot_wxyz": {"dtype": "float32", "shape": (4,), "names": QUAT},
        "holomotion.MotionReference.frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "holomotion.MotionReference.header.stamp": {"dtype": "float64", "shape": (1,), "names": ["sec"]},
        "holomotion.MotionReference.frames_this_tick": {"dtype": "int64", "shape": (1,), "names": None},
        "holomotion.MotionReference.frames": {"dtype": "float32", "shape": (RAW_MAX_FRAMES * 65,), "names": None},
        "holomotion.MotionReference.frames_meta": {"dtype": "float64", "shape": (RAW_MAX_FRAMES * 2,),
                                                   "names": [f"f{i}_{k}" for i in range(RAW_MAX_FRAMES) for k in ("frame_index", "stamp_sec")]},
        # ---- LowState (unitree_hg/msg/LowState) as fed to the node
        "holomotion.LowState.motor_state.q": {"dtype": "float32", "shape": (29,), "names": real},
        "holomotion.LowState.motor_state.dq": {"dtype": "float32", "shape": (29,), "names": real},
        "holomotion.LowState.imu_state.quaternion": {"dtype": "float32", "shape": (4,), "names": QUAT},
        "holomotion.LowState.imu_state.gyroscope": {"dtype": "float32", "shape": (3,), "names": XYZ3},
        # ---- VRState (robo_orchard_pico_msg_ros2/msg/VRState) as fed to the node
        "holomotion.VRState.left_controller": {"dtype": "float32", "shape": (len(VRSTATE_FIELDS),), "names": list(VRSTATE_FIELDS)},
        "holomotion.VRState.right_controller": {"dtype": "float32", "shape": (len(VRSTATE_FIELDS),), "names": list(VRSTATE_FIELDS)},
        "holomotion.pico": {"dtype": "float32", "shape": (len(RAW_PICO_NAMES),), "names": list(RAW_PICO_NAMES)},
        **({"holomotion.pico_hands": {"dtype": "float32", "shape": (HANDS_COLUMN,), "names": None}} if RAW_HANDS else {}),
        # ---- the policy's exact I/O
        "holomotion.policy_obs": {"dtype": "float32", "shape": (policy_obs_dim,), "names": None},
        "holomotion.policy_obs_dim": {"dtype": "int64", "shape": (1,), "names": None},
        "holomotion.policy_ran": {"dtype": "int64", "shape": (1,), "names": ["0=no policy step this tick,1=velocity,2=motion"]},
        "holomotion.actions_onnx": {"dtype": "float32", "shape": (29,), "names": layout.get("motion", {}).get("dof_names_onnx")},
        # ---- the node's outputs (topics /humanoid/action, /humanoid/kps, /humanoid/kds; robot joint order)
        "holomotion.action": {"dtype": "float32", "shape": (29,), "names": real},
        "holomotion.kps": {"dtype": "float32", "shape": (29,), "names": real},
        "holomotion.kds": {"dtype": "float32", "shape": (29,), "names": real},
        # ---- the node's clock at the start of the tick (time.time() as the node reads it): the replayer runs the node on it
        "holomotion.tick_time": {"dtype": "float64", "shape": (1,), "names": ["sec"]},
        # ---- the node's 11-frame reference window after this step, under the v1.3 hand-off's name (2026-10-19):
        # 11 x 65, block 0 = current, blocks 1..10 = future (v1.3 had 11 x 79: its blocks carried 14 hand values too)
        "reference_actions": {"dtype": "float32", "shape": (11 * 65,), "names": None},
        "reference_actions_frame_index": {"dtype": "int64", "shape": (11,), "names": ["current"] + [f"future_{k}" for k in range(1, 11)]},
    }
    for mode, prefix in (("motion", "holomotion.obs."), ("velocity", "holomotion.obs_velocity.")):
        for name, st, en in layout.get(mode, {}).get("terms", []):
            f[prefix + name] = {"dtype": "float32", "shape": (en - st,), "names": None}
    return f


RAW_SPEC = {
    "holomotion.MotionReference.*": "the newest reference frame received before this control tick, in the message's own fields (dof_pos, dof_vel in the robot's complete_dof_order; root_pos; root_rot_wxyz; frame_index; header.stamp in s). This is what the PICO operator provides and what a VLA must provide.",
    "holomotion.MotionReference.frames": f"every frame received since the previous tick (frames_this_tick of them; the newest {RAW_MAX_FRAMES} stored, arrival order, zero padded), each 65 = {LATEST_OBS_LAYOUT}; frames_meta = (frame_index, stamp_sec) per stored frame, -1 where unused",
    "holomotion.LowState.*": "what feed_lowstate received: motor_state q / dq for the 29 body joints (complete_dof_order), imu_state quaternion (w x y z) and gyroscope (body angular velocity)",
    "holomotion.VRState.*": "the controller sample the node received this tick, as the VRState message fields of each controller (status, a/b/x/y buttons, axis_x, axis_y, trigger, grip, axis_click, menu_button); NaN when none was fed",
    "holomotion.pico": "the raw PICO sample before the button map (sticks, grips, triggers, button bits, hand active flags, left menu)",
    "holomotion.pico_hands": "--hand-source skeleton only: the headset's hand tracking as the SDK gave it this tick, left then right hand, 26 OpenXR joints x (x y z qx qy qz qw) flat (364) + the two isActive flags; NaN when no hands sample had arrived; the agent's retarget (hand_skeleton.py) turns it into the Revo 2 targets in action.revo2.<side>.cmd_position",
    "holomotion.obs.<term>": "the motion-tracking policy's observation terms, HoloMotion's names and order (obs.obs_groups.unified.atomic_obs_list, actor terms), the exact slices of the ONNX input; zeros on ticks the motion model did not run",
    "holomotion.obs_velocity.<term>": "the same for the velocity (walking) policy",
    "holomotion.policy_obs": "the exact observation vector handed to the ONNX session this tick (zero padded to the larger model); policy_obs_dim = its valid length; policy_ran = which model",
    "holomotion.actions_onnx": "the raw ONNX output (29, the model's joint_names order from its metadata), before scaling / EMA",
    "holomotion.action / kps / kds": "the joint targets and gains the node publishes on /humanoid/action, /humanoid/kps, /humanoid/kds (robot joint order)",
    "holomotion.tick_time": "the node's clock (time.time()) at the start of the tick, before its inputs were fed; the reference data-age and controller-timeout checks replay on it",
    "reference_actions": "the node's 11-frame reference window after this step's frames were fed, 11 x 65 (the v1.3 hand-off's name; v1.3 blocks were 79 = 65 + 14 hand values): block 0 = the current frame (10 arrivals behind the newest), blocks 1..10 = future 1..10 (block 10 = the newest); each block = 65 (dof_pos dof_vel root_pos root_rot_wxyz); zeros until 11 frames have arrived. Fed whole every 20 ms it reproduces the model input exactly; reference_actions_frame_index = the frame numbers of the 11 blocks",
    "replay/episode_XXXXXX.npz: holomotion_start.*": "the node's internal state at the start of frame 0 (node_state.snapshot: reference queue, runtime state, velocity obs history, KV cache, yaw alignment, stick filter); node_state.restore puts a fresh node there, so the recording replays bit-exactly from frame 0 (episodes.jsonl holomotion_start: summary, cold_start)",
}


def _pinhole_image_spec(shape) -> dict:
    """meta/holomotion_raw.json "images": what each recorded eye is, with its intrinsics at the recorded size."""
    from simple.teleop.holomotion_v14.robot_variants import PINHOLE_H, PINHOLE_W, rectified
    h, w = int(shape[0]), int(shape[1])
    sx, sy = w / PINHOLE_W, h / PINHOLE_H
    out = {}
    for eye, key, rp in (("left", "observation.images.ego_view_left", "R1/P1"), ("right", "observation.images.ego_view_right", "R2/P2")):
        P = rectified(eye)[1]
        out[key] = dict(
            what=f"rectified {eye} eye of the HBVCAM stereo camera ({rp}), rendered at {PINHOLE_W}x{PINHOLE_H} and "
                 + (f"area-downscaled to {w}x{h}" if (w, h) != (PINHOLE_W, PINHOLE_H) else "saved as rendered"),
            width=w, height=h,
            # pixel-centre convention: x' = (x + 0.5) * s - 0.5
            fx=round(float(P[0, 0]) * sx, 4), fy=round(float(P[1, 1]) * sy, 4),
            cx=round((float(P[0, 2]) + 0.5) * sx - 0.5, 4), cy=round((float(P[1, 2]) + 0.5) * sy - 0.5, 4))
        if eye == "right":
            out[key]["baseline_mm"] = round(-float(P[0, 3]) / float(P[0, 0]), 3)     # the calibration is in mm
    return out


_LEROBOT_INDEX_KEYS = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


def _check_resume_columns(save_dir, features: dict) -> None:
    """Recording into an existing dataset resumes it with ITS column list, so a frame of another format fails LeRobot's
    validate_frame on the first recorded tick (e.g. the September level-3_pinhole sets: one ego_view, no holomotion.*
    columns). Stop at startup instead, before anything is written (2026-10-18)."""
    import json
    info = Path(save_dir) / "meta" / "info.json"
    if not info.exists():
        return
    old = json.loads(info.read_text()).get("features", {})
    have = set(old) - _LEROBOT_INDEX_KEYS
    want = set(features) - _LEROBOT_INDEX_KEYS
    resized = sorted(k for k in have & want if [int(v) for v in old[k]["shape"]] != [int(v) for v in features[k]["shape"]])
    if have == want and not resized:
        return
    new, gone = sorted(want - have), sorted(have - want)
    why = []
    if new or gone:
        why.append(f"{len(new)} columns this recorder writes are not in it{' (e.g. ' + ', '.join(new[:3]) + ')' if new else ''}, "
                   f"{len(gone)} of its columns are no longer written{' (e.g. ' + ', '.join(gone[:3]) + ')' if gone else ''}")
    if resized:
        why.append("other shapes: " + ", ".join(f"{k} {list(old[k]['shape'])} -> {list(features[k]['shape'])}" for k in resized[:3]))
    print(f"\n[Record] {save_dir} already holds a dataset in another format: {'; '.join(why)}.\n[Record] Episodes of two "
          f"formats cannot share a dataset folder. Record into a new one, e.g. --save-dir data/teleop_holomotion_v14_rot0. "
          f"Nothing was written.", file=sys.stderr, flush=True)
    os._exit(2)                                     # before the loop: no recording, no exporter; threads would block exit


def _init_exporter(save_dir, task_prompt, robot_model, obj_names, joint_names, ego_view_shape, realformat: bool = True,
                   right_eye: bool = False, policy_obs_dim: int = 0, hand: str = "dex3", wrist: bool = False, wrist_fov: float = 150.0,
                   images: bool = True, tactile: bool = True, extra_features: dict | None = None, probe: bool = False):
    """The decoupled-WBC LeRobot schema (as the psi0 datasets) plus the v1.4 controller's own signals, plus (realformat)
    the real G1 recorder's extra fields (teleop/holomotion_v14/realformat.py). images=False / tactile=False: --lean-record
    (2026-10-26), no image columns / no tactile pads; the replay's --fill adds them."""
    import json
    from decoupled_wbc.data.exporter import Gr00tDataExporter
    from decoupled_wbc.data.utils import get_dataset_features, get_modality_config
    from simple.teleop.holomotion_v14 import realformat as RF

    features = get_dataset_features(robot_model)
    if images:
        set_ego_view_feature_shape(features, ego_view_shape)
        if not right_eye:
            validate_existing_ego_view_feature_shape(save_dir, ego_view_shape)
    else:
        features.pop("observation.images.ego_view", None)
    features["observation.state"]["names"] = joint_names
    if obj_names:
        names = [f"{n}.{s}" for n in obj_names for s in ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z")]
        features["observation.object_poses"] = {"dtype": "float64", "shape": (len(obj_names) * 7,), "names": names}
    features["teleop.latest_obs"] = {"dtype": "float32", "shape": (65,), "names": None}
    features["policy.mode"] = {"dtype": "int64", "shape": (1,), "names": ["0=velocity,1=motion"]}
    features["policy.raw_action"] = {"dtype": "float32", "shape": (29,), "names": None}
    features["policy.target_real"] = {"dtype": "float32", "shape": (29,), "names": None}
    modality = get_modality_config(robot_model)
    if not images:
        modality["video"] = {}
    if hand == "revo2":                                        # the Revo 2 robot (2026-10-26): 11-joint hands, tactile pads, motor columns
        from simple.teleop.holomotion_v14 import revo2_record as R2
        R2.patch_schema(features, modality)
        features.update(R2.features(tactile=tactile))
    if right_eye and images:                                   # both rectified eyes (2026-10-16): ego_view_left + ego_view_right replace ego_view
        left = features.pop("observation.images.ego_view")
        features["observation.images.ego_view_left"] = dict(left)
        features["observation.images.ego_view_right"] = dict(left)
        modality["video"] = {"ego_view_left": {"original_key": "observation.images.ego_view_left"},
                             "ego_view_right": {"original_key": "observation.images.ego_view_right"}}
    if wrist:                                                  # the two wrist cameras (2026-10-26): observation.images.wrist_left / wrist_right, 640 x 360
        from simple.teleop.holomotion_v14 import wrist_cams as WC
        features.update(WC.features())
        modality.setdefault("video", {}).update({f"wrist_{s}": {"original_key": f"observation.images.wrist_{s}"} for s in ("left", "right")})
    if policy_obs_dim:
        features.update(_raw_features(policy_obs_dim))
    if realformat:
        RF.add_features(features, hand=hand)
    if extra_features:                                         # the replay's --fill: plain source columns this version does not write itself, copied through (2026-11-05)
        features.update({k: dict(v) for k, v in extra_features.items()})
    if probe:                                                  # the feature table alone (the fill compares it with the source before it creates the copy)
        return features
    _check_resume_columns(save_dir, features)
    exporter = Gr00tDataExporter.create(save_root=save_dir, fps=50, features=features, modality_config=modality, task=task_prompt)
    spec = Path(save_dir) / "meta" / "realformat.json"
    if realformat and not spec.exists():
        spec.write_text(json.dumps(RF.spec(hand), indent=2))
    raw_spec = Path(save_dir) / "meta" / "holomotion_raw.json"
    if policy_obs_dim and not raw_spec.exists():
        img_spec = ({"deferred": "lean recording: no images; python -m simple.cli.replay_holomotion_v14 <dataset> --all --fill renders them (meta/lean_record.json)"} if not images
                    else _pinhole_image_spec(ego_view_shape) if right_eye else {"observation.images.ego_view": "the recorded camera"})
        if wrist:
            from simple.teleop.holomotion_v14 import wrist_cams as WC
            img_spec.update(WC.image_spec(wrist_fov))
        raw_spec.write_text(json.dumps({"interface": {"MotionReference": "holomotion_interfaces/msg/MotionReference, the reference stream (PICO publisher or a VLA) -> holomotion.MotionReference.*",
                                                      "LowState": "unitree_hg/msg/LowState (/lowstate) -> holomotion.LowState.*", "VRState": "robo_orchard_pico_msg_ros2/msg/VRState (/pico_bridge/vr_state) -> holomotion.VRState.*",
                                                      "outputs": "/humanoid/action, /humanoid/kps, /humanoid/kds -> holomotion.action, holomotion.kps, holomotion.kds"},
                                        "columns": RAW_SPEC, "max_frames_per_tick": RAW_MAX_FRAMES, "policy_obs_dim_max": policy_obs_dim, "latest_obs_layout": LATEST_OBS_LAYOUT,
                                        "obs_layout": _obs_layout(), "real_dof_order": _real_dof_names(), "images": img_spec}, indent=2))
    return exporter


def lean_spec(record_camera: str, record_size, wrist: bool, wrist_hz: float, wrist_fov: float, tactile: bool, hand: str) -> dict:
    """meta/lean_record.json of a --lean-record dataset: what the recording left out and how the replay's --fill puts it back."""
    images = (["observation.images.ego_view_left", "observation.images.ego_view_right"] if record_camera == "pinhole"
              else ["observation.images.ego_view"])
    if wrist:
        images += ["observation.images.wrist_left", "observation.images.wrist_right", "wrist_cams.fresh"]
    tactile_cols = []
    if tactile and hand == "revo2":
        from simple.teleop.holomotion_v14 import revo2_record as R2
        tactile_cols = sorted(R2.tactile_features())
    return dict(lean_record=True, deferred=dict(images=images, tactile=tactile_cols), record_camera=record_camera,
                record_size=[int(record_size[0]), int(record_size[1])], wrist_cams=bool(wrist), wrist_hz=float(wrist_hz), wrist_fov=float(wrist_fov),
                tactile=bool(tactile and hand == "revo2"), hand=hand,
                fill="python -m simple.cli.replay_holomotion_v14 <this dataset dir> --all --fill   (writes <dir>_filled with the images and pads rendered on the bit-exact replay)")


_LAYOUT_CHECKED: set = set()


def _raw_columns(action, policy_obs_dim: int) -> dict:
    """The holomotion.* columns of a frame from the agent's action extras (ActionCmd: its parameters dict)."""
    action = getattr(action, "parameters", action)
    frames = action.get("raw_ref_frames") or []
    ref = np.zeros((RAW_MAX_FRAMES, 65), dtype=np.float32); meta = np.full((RAW_MAX_FRAMES, 2), -1.0, dtype=np.float64)
    for i, (obs65, fi, ts) in enumerate(frames[-RAW_MAX_FRAMES:]):          # the newest RAW_MAX_FRAMES, arrival order
        ref[i] = obs65; meta[i] = (fi, ts / 1e9)
    newest = frames[-1] if frames else None
    obs = action.get("raw_policy_obs"); out_ = action.get("raw_policy_out"); mode = action.get("raw_policy_mode")
    pobs = np.zeros(policy_obs_dim, dtype=np.float32)
    if obs is not None:
        pobs[:min(len(obs), policy_obs_dim)] = obs[:policy_obs_dim]
    low = np.asarray(action["raw_lowstate"], dtype=np.float32); vr = action["raw_vrstate"]
    cols = {
        "holomotion.MotionReference.dof_pos": newest[0][0:29] if newest else np.zeros(29, np.float32),
        "holomotion.MotionReference.dof_vel": newest[0][29:58] if newest else np.zeros(29, np.float32),
        "holomotion.MotionReference.root_pos": newest[0][58:61] if newest else np.zeros(3, np.float32),
        "holomotion.MotionReference.root_rot_wxyz": newest[0][61:65] if newest else np.zeros(4, np.float32),
        "holomotion.MotionReference.frame_index": np.array([newest[1] if newest else -1], dtype=np.int64),
        "holomotion.MotionReference.header.stamp": np.array([newest[2] / 1e9 if newest else -1.0], dtype=np.float64),
        "holomotion.MotionReference.frames_this_tick": np.array([len(frames)], dtype=np.int64),
        "holomotion.MotionReference.frames": ref.reshape(-1), "holomotion.MotionReference.frames_meta": meta.reshape(-1),
        "holomotion.LowState.motor_state.q": low[0:29], "holomotion.LowState.motor_state.dq": low[29:58],
        "holomotion.LowState.imu_state.quaternion": low[58:62], "holomotion.LowState.imu_state.gyroscope": low[62:65],
        "holomotion.VRState.left_controller": np.asarray(vr["left_controller"], dtype=np.float32),
        "holomotion.VRState.right_controller": np.asarray(vr["right_controller"], dtype=np.float32),
        "holomotion.pico": np.asarray(action["raw_pico"], dtype=np.float32),
        **({"holomotion.pico_hands": np.asarray(action["raw_pico_hands"], dtype=np.float32)} if RAW_HANDS and action.get("raw_pico_hands") is not None else {}),
        "holomotion.policy_obs": pobs, "holomotion.policy_obs_dim": np.array([0 if obs is None else len(obs)], dtype=np.int64),
        "holomotion.policy_ran": np.array([0 if obs is None else (2 if mode == "motion" else 1)], dtype=np.int64),
        "holomotion.actions_onnx": np.zeros(29, dtype=np.float32) if out_ is None else np.asarray(out_, dtype=np.float32)[:29],
        "holomotion.action": np.zeros(29, dtype=np.float32) if action.get("raw_action_real") is None else np.asarray(action["raw_action_real"], dtype=np.float32),
        "holomotion.kps": np.asarray(action["raw_kps"], dtype=np.float32), "holomotion.kds": np.asarray(action["raw_kds"], dtype=np.float32),
        "holomotion.tick_time": np.array([float(action.get("raw_tick_time", np.nan))], dtype=np.float64),
        "reference_actions": np.asarray(action.get("raw_ref_window", np.zeros((11, 65), np.float32)), dtype=np.float32).reshape(-1),
        "reference_actions_frame_index": np.asarray(action.get("raw_ref_window_index", np.full(11, -1)), dtype=np.int64).reshape(-1),
    }
    layout = _obs_layout(); live = action.get("raw_obs_layouts") or {}
    for m_, lay in live.items():                                    # the live builder layout must equal the baked one
        if m_ in layout and m_ not in _LAYOUT_CHECKED:
            if [list(t) for t in lay["terms"]] != [list(t) for t in layout[m_]["terms"]]:
                raise RuntimeError(f"HoloMotion {m_} observation layout differs from {LAYOUT_FILE}: re-bake it (HOLOMOTION_DUMP_OBS_LAYOUT)")
            _LAYOUT_CHECKED.add(m_)
    for m_, prefix in (("motion", "holomotion.obs."), ("velocity", "holomotion.obs_velocity.")):
        ran = obs is not None and ((mode == "motion") == (m_ == "motion"))
        for name, st, en in layout.get(m_, {}).get("terms", []):
            cols[prefix + name] = np.asarray(obs[st:en], dtype=np.float32).copy() if ran else np.zeros(en - st, dtype=np.float32)
    return cols


def _dump_obs_layouts(agent, save_dir) -> None:
    """Write the observation-term layouts the builders reported into meta/holomotion_raw.json; with
    HOLOMOTION_DUMP_OBS_LAYOUT=<path> also bake them for the recorder (needs an episode with motion tracking on)."""
    import json
    lay = getattr(agent.node, "obs_layouts", {})
    if not lay:
        return
    spec = Path(save_dir) / "meta" / "holomotion_raw.json"
    if spec.exists():
        d = json.loads(spec.read_text()); d["obs_layout_live"] = lay; spec.write_text(json.dumps(d, indent=2))
    dump = os.environ.get("HOLOMOTION_DUMP_OBS_LAYOUT")
    if dump:
        Path(dump).write_text(json.dumps(lay, indent=1)); print(f"[Record] observation layouts ({', '.join(lay)}) -> {dump}")


def _env_config_json(task) -> str:
    import json
    from simple.utils import NumpyArrayEncoder
    return json.dumps(task.state_dict(), cls=NumpyArrayEncoder)


def _build_frame(robot_model, obj_names, observation, privileged_info, action, phys=None, policy_obs_dim: int = 0, revo2=None, wrist=None,
                 images: bool = True):
    proprio = privileged_info["proprio"]
    if revo2 is not None:                                      # the Revo 2 robot: 51-wide action (revo2_record.action_names order) + its extra columns
        revo2 = dict(revo2); action_q = revo2.pop("_action")
    else:
        from simple.teleop.holomotion.protocol import DEX3_NATURAL_TO_MJCF
        inv = np.argsort(DEX3_NATURAL_TO_MJCF)                 # MJCF -> natural (thumb, middle, index)
        lh = np.asarray(action["left_hand_q"], dtype=np.float64)[inv]
        rh = np.asarray(action["right_hand_q"], dtype=np.float64)[inv]
        action_q = robot_model.get_configuration_from_actuated_joints(
            body_actuated_joint_values=np.asarray(action["target_q"], dtype=np.float64),
            left_hand_actuated_joint_values=lh, right_hand_actuated_joint_values=rh)
    # the decoupled-WBC wrist / height columns (2026-10-28, simple.teleop.holomotion_v14.eef): measured wrists from the 29 body
    # joints (observation.state starts with them in G1_BODY_JOINTS order on every robot); commanded = the tracked reference frame
    from simple.teleop.holomotion_v14.eef import wrist_fk
    params = getattr(action, "parameters", action)                         # ActionCmd: its parameters dict
    eef_cmd, height_cmd = wrist_fk().command(np.asarray(params.get("raw_ref_window", np.zeros((11, 65), np.float32))).reshape(11, 65)[0])
    frame = {
        "observation.state": np.asarray(observation["joint_qpos"], dtype=np.float64),
        "observation.eef_state": wrist_fk().poses(np.asarray(observation["joint_qpos"])[:29]),
        "action": np.asarray(action_q, dtype=np.float64),
        "action.eef": eef_cmd,
        "observation.img_state_delta": np.array([0.0], dtype=np.float32),       # 0 s: the images are rendered on the same state
        "teleop.navigate_command": np.asarray(action["navigate_cmd"], dtype=np.float64),
        "teleop.base_height_command": np.array([height_cmd], dtype=np.float64),
        "observation.base_pose": np.asarray(proprio["floating_base_pose"], dtype=np.float64),
        "observation.base_vel": np.asarray(proprio["floating_base_vel"], dtype=np.float64),
        "observation.torso_rpy_command": np.zeros(3, dtype=np.float64),
        "teleop.latest_obs": np.asarray(action["latest_obs"], dtype=np.float32),
        "policy.mode": np.array([1 if action["policy_mode"] == "motion" else 0], dtype=np.int64),
        "policy.raw_action": np.asarray(action["raw_action"], dtype=np.float32),
        "policy.target_real": np.asarray(action["target_real"], dtype=np.float32),
    }
    if images:                                                 # --lean-record (2026-10-26) saves no images: the replay's --fill renders them
        if "head_stereo_right_rect" in observation:           # both rectified eyes recorded (2026-10-16): ego_view_left / ego_view_right
            frame["observation.images.ego_view_left"] = observation["head_stereo_left"]
            frame["observation.images.ego_view_right"] = observation["head_stereo_right_rect"]
        else:
            frame["observation.images.ego_view"] = observation["head_stereo_left"]
    if wrist:                                                  # the wrist cameras (2026-10-26), rendered on the same pre-step state
        frame["observation.images.wrist_left"] = wrist["left"]
        frame["observation.images.wrist_right"] = wrist["right"]
        frame["wrist_cams.fresh"] = np.array([1 if wrist.get("fresh", True) else 0], dtype=np.int8)
    if policy_obs_dim:
        frame.update(_raw_columns(action, policy_obs_dim))
    if phys:                                                   # the real recorder's extra fields (realformat)
        frame.update(phys)
    if revo2:                                                  # Revo 2 joints, motors, grip and the tactile pads
        frame.update(revo2)
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


def _stand_on_floor(sonic_env, agent, init_pose: dict | None = None, shift: tuple[float, float] | None = None) -> np.ndarray:
    """Quick start: the robot at the policy's default pose (plus the --random-init-pose offset), feet on the floor, at
    rest, no leash (x, y and yaw as the setup spawned it, plus the offset, plus `shift`: a spawn-back slice start spawned
    the robot behind its start for SIMPLE's reset steps). Returns the body joints placed (MJCF order)."""
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
    if shift is not None:
        d.qpos[a] += shift[0]
        d.qpos[a + 1] += shift[1]
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


def _shift_root(sonic_env, shift: tuple[float, float]) -> None:
    """Move the robot's root by (dx, dy) in the world (a spawn-back slice start, the robot hanging on the leash)."""
    import mujoco
    m, d = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData
    root = int(m.body_rootid[m.body("pelvis").id])
    a = int(m.jnt_qposadr[m.body_jntadr[root]])
    d.qpos[a] += shift[0]; d.qpos[a + 1] += shift[1]
    mujoco.mj_forward(m, d)


RAW_HANDS = False                                                        # --hand-source skeleton: the recording carries holomotion.pico_hands
REF_LOST_S = 1.5
START_SETTLE_S = 1.5                                                     # (unused since 2026-11-04: the loop no longer watches a start for contacts)


def _start_touches(sonic_env, skip_body: str | None = None, touch_mm: float = 0.0, forward: bool = True) -> list:
    """Contacts between the robot and anything but the floor (user 2026-11-03: "the robot should not touch anything at the
    beginning"): (robot part, what it touches, penetration mm); a contact counts from touch_mm of separation (0 = touching or
    inside). Not counted: the feet (ankle_roll links: floor, mat, pedal), the held plate of a plate-in-hand start (skip_body),
    the floor plane. forward: recompute the contacts first (a just-placed robot); in the loop the last step's contacts are current."""
    import mujoco
    m, d = sonic_env.mujoco.mjModel, sonic_env.mujoco.mjData
    if forward:
        mujoco.mj_forward(m, d)
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis"); root = int(m.body_rootid[pel])
    bname = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"   # noqa: E731
    if skip_body:                                                          # a plate_in_hand name ("plate_a") or the body itself ("k_plate_a")
        from simple.teleop.holomotion_v14.grasp_start import PLATE_BODIES
        skip_body = PLATE_BODIES.get(skip_body, skip_body)
    skip_b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, skip_body) if skip_body else -1
    out = []
    for c in d.contact[:d.ncon]:
        b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
        r1, r2 = int(m.body_rootid[b1]) == root, int(m.body_rootid[b2]) == root
        if r1 == r2 or c.dist > touch_mm / 1000:
            continue
        rb, ob, og = (b1, b2, int(c.geom2)) if r1 else (b2, b1, int(c.geom1))
        if m.geom_type[og] == mujoco.mjtGeom.mjGEOM_PLANE or ob == skip_b or "ankle_roll" in bname(rb):
            continue
        out.append((bname(rb).replace("_link", ""), bname(ob).replace("k_", ""), round(-float(c.dist) * 1000, 1)))
    return out                                                         # no reference frame for this long = the headset's body tracking is gone (headset view + terminal say so)


def _stream_status(agent, rec_state, rec_frames: int, episodes_saved: int, num_episodes: int, setup: dict,
                   privileged_info, quick: bool, record_on_motion: bool = False, slice_cut=None, slice_auto_cut: bool = False) -> list:
    """Overlay lines for the headset stream: (text, BGR)."""
    red, white, grey, yellow = (60, 60, 255), (255, 255, 255), (190, 190, 190), (60, 220, 255)
    if rec_state == RecordingState.RECORDING:
        top = (f"REC {rec_frames / 50.0:5.1f} s", red)
    elif getattr(agent, "quick_holding", False):
        if ((setup or {}).get("preset") or {}).get("plate_in_hand"):   # A3: the plate held level until Y; the operator must be holding it too when the robot follows them
            top = ("standing, plate held level  (match the robot's arm, close your hand on the plate; SPACE: countdown, Y: take over)", white)
        else:
            top = ("standing, start pose held  (SPACE: countdown, Y: take over)", white)
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
    sub = getattr(agent, "sub", None)
    ref_age = sub.obs_age() if sub is not None and getattr(sub, "n_obs", 0) else 0.0
    if ref_age > REF_LOST_S:                                             # 2026-11-03: a silent freeze was the headset's body tracking stopping
        lines.insert(0, (f"NO BODY TRACKING from the headset for {ref_age:.0f} s: robot holds", red))
        lines.insert(1, ("headset on the head? controllers awake? app in front? Wi-Fi?", red))
    prog = (privileged_info.get("task_progress") or {}) if isinstance(privileged_info, dict) else {}
    gates = [k for k in ("door_open", "at_bowl", "grasped", "at_bin", "at_basin", "at_cart", "at_sink", "placed", "rack_in", "door_closed") if k in prog]
    if gates:
        lines.append(("  ".join(f"{k} [{'x' if prog.get(k) else ' '}]" for k in gates), white))
    if slice_cut is not None and rec_state == RecordingState.RECORDING:    # --slice: the end event, then "save now" (or the auto-cut countdown)
        text, past_event = slice_cut.status(rec_frames - 1, auto=slice_auto_cut)
        lines.append((text, yellow if past_event else white))
    elif rec_state == RecordingState.RECORDING and prog.get("door_closed" if "door_closed" in prog else "placed"):
        lines.append(("task done: L-menu + A to save", yellow))          # 2026-10-30: success no longer stops the recording
    lines.append(("L-menu + A save  + B drop  + Y new setup" if rec_state == RecordingState.RECORDING
                  else "L-menu + A record  + Y new setup", grey))
    return lines


def _task_success(privileged_info, episode_buffer) -> dict:
    """The task's check at save time, for the episode's `realformat` record: success = the last gate latched (bottle_bin:
    the whole bottle in the bin for 0.5 s after the hand let go); success_frame = the recorded frame on which it latched
    (the gate's t / t_placed run at the env's 20 ms step, so the frame is counted back from the last one)."""
    prog = (privileged_info.get("task_progress") or {}) if isinstance(privileged_info, dict) else {}
    final = "door_closed" if "door_closed" in prog else "placed"    # load_dishwasher ends with the door shut (2026-10-30), the others at placed
    ok = bool(prog.get(final))
    n = len(episode_buffer.get("timestamp") or []) if isinstance(episode_buffer, dict) else 0
    frame = None
    if ok and n and prog.get("t") is not None and prog.get(f"t_{final}") is not None:
        frame = max(0, n - 1 - int(round((float(prog["t"]) - float(prog[f"t_{final}"])) / 0.02)))
    return dict(success=ok, success_frame=frame)


class _SaveGuard:
    """Finish an episode save even if Ctrl-C is pressed during it (2026-10-26: a 4-minute kitchen recording was lost with
    meta/tasks.jsonl written and no parquet / exact log -- the save of a long episode takes seconds while the sim stands
    still, and a Ctrl-C then cut it in half). Ctrl-C during the save is noted and raised as KeyboardInterrupt after it
    (reraise=True) or dropped (the save on exit). Prints how long the save took."""

    def __init__(self, what: str, frames: int, reraise: bool = True):
        self.what, self.frames, self.reraise, self.pressed, self.old = what, int(frames), reraise, False, None

    def _handler(self, signum, frame):
        self.pressed = True
        print(f"[Record] Ctrl-C while saving: finishing the save of {self.what} first", flush=True)

    def __enter__(self):
        import signal
        try:
            self.old = signal.signal(signal.SIGINT, self._handler)
        except ValueError:                                     # not the main thread: nothing to guard
            self.old = None
        self.t0 = time.monotonic()
        print(f"[Record] saving {self.what} ({self.frames} frames, {self.frames / 50:.0f} s); do not stop the program", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        import signal
        if self.old is not None:
            signal.signal(signal.SIGINT, self.old)
        if exc_type is None:
            print(f"[Record] {self.what} saved in {time.monotonic() - self.t0:.1f} s", flush=True)
        if self.pressed and self.reraise and exc_type is None:
            raise KeyboardInterrupt
        return False


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
    max_distractors: Annotated[int, typer.Option(help="distractors per setup: 0..this (lv3: exactly this); a scene whose range is (0, 0), the kitchen, keeps none")] = 3,
    setup_ranges: Annotated[str, typer.Option(help="random setup ranges: teleop (robot +-8/+-10 cm +-10 deg, target, bin/cart, "
                                                   "0..3 distractors, fixed table) or lv3 (the kits' level-3 eval sets: robot "
                                                   "+-10 cm back/forward +-5 cm sideways, the level-3 target region, table / "
                                                   "counter / cart-box height +-4 cm, 3 distractors; records into level-3)")] = "lv3",
    exact_log: Annotated[bool, typer.Option(help="with --record: save the bit-exact replay log of every episode")] = True,
    realformat: Annotated[bool, typer.Option(help="with --record: also record the real G1 recorder's fields (joint velocities and efforts, root pose and velocity, pelvis IMU, hand velocities and efforts, PD gains, the reference queue and the HoloMotion observation terms), as scripts/holomotion_sim_realformat.py derives them")] = True,
    save_model: Annotated[bool, typer.Option(help="with the exact log: also save the compiled MuJoCo model (.mjb, ~125 MB: meshes + collision trees)")] = False,
    backpack_kg: Annotated[float, typer.Option(help="HoloMotion's G1 backpack body on the robot, this mass (kg; 3.2 = what the backpack model was trained for, 0 = none)")] = 3.2,
    motion_model: Annotated[str, typer.Option(help="motion-tracking policy: backpack (v1.4.1 BrainCo 3.2 kg backpack, model_22000) or public (v1.4.1, model_16200)")] = "backpack",
    robot_kind: Annotated[str, typer.Option("--robot", help="stereo: the G1 with the HBVCAM stereo head camera (third_party/hbvcam_stereo); stock: SIMPLE's G1; revo2: the rot0 stereo G1 with BrainCo Revo 2 hands and their tactile pads (data/robots/g1_rot0_teleop_revo2)")] = "stereo",
    stream_camera: Annotated[str, typer.Option(help="headset picture: pinhole (the stereo camera's rectified left eye, 1280x720, 104.7 x 72.2 deg), fisheye (both eyes' raw fisheye images side by side, 2560x720, like the real camera) or head (the scene's D455 head camera)")] = "pinhole",
    head_tilt_deg: Annotated[float, typer.Option(help="pitch the head cameras (the HBVCAM stereo camera the headset shows and SIMPLE's head camera) this many degrees further down (+) or up (-) than the robot file; 0 = the rot0 bracket (49.3 deg, the stereo URDF since 2026-10-16; the September recordings were the rot15 URDF + 10 = 44.3 deg)")] = 0.0,
    motion_model_bundle: Annotated[str, typer.Option(help="another backpack model bundle dir (model_22000.onnx [+ config.yaml]); overrides --motion-model")] = "",
    quick_start: Annotated[bool, typer.Option(help="every scene: standing, no leash, walking policy on, recording starts by itself (--no-quick-start: hang on the leash, stand up by hand)")] = True,
    record_camera: Annotated[str, typer.Option(help="image saved as ego_view: auto (the headset's camera: pinhole with --stream-camera pinhole, fisheye with fisheye, head with head), head (SIMPLE's head camera, 640x360), pinhole (both rectified eyes, --record-size each) or fisheye (both eyes side by side, --record-size each)")] = "auto",
    fisheye_face_px: Annotated[int, typer.Option(help="--stream-camera fisheye: size of the 90-degree face renders (960 ~ the lens's own sharpness; lower = cheaper, softer)")] = 960,
    black_hands: Annotated[bool, typer.Option(help="draw the Dex3 hands (palm + fingers) black, as the scene kits do (visual only; --no-black-hands: the robot file's light grey)")] = True,
    record_size: Annotated[str, typer.Option(help="recorded image size per eye, WxH, for pinhole and fisheye recordings: rendered at the full 1280x720 and downscaled with area averaging (the headset stream stays full size); 1280x720 = no downscale")] = "640x360",
    record_on_motion: Annotated[bool, typer.Option(help="start each recording (the first and every later one) when you press the motion-tracking button (Y); default since 2026-10-27 (user: \"when I hit Y the program starts recording\"). --no-record-on-motion: record from the first tick the controllers are connected (frame 0 = the start pose)")] = True,
    auto_motion: Annotated[bool, typer.Option(help="quick start also switches to motion tracking (presses Y) instead of leaving it to the operator")] = False,
    stream_view: Annotated[str, typer.Option(help="headset view: single (one picture), mono (left camera in both halves) or stereo")] = "single",
    button_map: Annotated[str, typer.Option(help="sim: A walk, Y motion tracking, B free; robot: the v1.4.1 map (A walk, B motion, Y back to walking)")] = "sim",
    random_init_pose: Annotated[bool, typer.Option(help="every scene init: a small random robot pose (base, joints), so each episode starts from a different state (--no-random-init-pose: the policy's default pose)")] = True,
    init_pose_scale: Annotated[float, typer.Option(help="scale of the --random-init-pose ranges (1 = arms 0.15 rad, waist 0.05, legs 0.03, base 2 cm / 3 deg)")] = 1.0,
    pico_stream: Annotated[bool, typer.Option(help="stream the head camera to the headset")] = True,
    hand_source: Annotated[str, typer.Option(help="grip (default): the Revo 2 fingers follow the controllers' grip / trigger poses. skeleton: the headset's own hand tracking drives the fingers (26 OpenXR joints per hand -> 6 motors, docs/HANDOVER_PICO_SKELETON_REVO2_TELEOP.md); the publisher must run with --hands (scripts/teleop_holomotion_v14.sh adds it); the controllers can be put down (PICO tracks hands only while they are idle), the keyboard then stands in for their buttons (pico_keys: r save, b abandon, n new setup, y motion ...) and --auto-cycle runs the session; the recording gets holomotion.pico_hands (the raw hand arrays) so a re-record reproduces the fingers")] = "grip",
    hand_sides: Annotated[str, typer.Option(help="--hand-source skeleton: which hands, left,right (default both)")] = "left,right",
    hand_calib_dir: Annotated[str, typer.Option(help="--hand-source skeleton: folder with calib_<side>_pico.json (default data/wuji_glove_pilot; missing = built-in defaults)")] = "",
    hand_min_conf: Annotated[float, typer.Option(help="--hand-source skeleton: frames the SDK marks not active count 0.25; below this they are held, not applied (0 = take them)")] = 0.0,
    hand_smooth_hz: Annotated[float, typer.Option(help="--hand-source skeleton: low-pass on the 6 motor targets (Hz)")] = 10.0,
    auto_cycle: Annotated[bool, typer.Option(help="hands-free session (2026-11-03): SPACE on the keyboard arms it, a countdown on the headset view (--countdown-s), then the motion-tracking button is pressed for you (motion + recording with --record-on-motion); the slice's end event + overlap save the episode by itself (--slice-auto-cut, forced on) and the scene resets; when the robot stands again the next countdown runs, until --num-episodes. SPACE pauses after the current episode. Needs --slice. Default off: the operator presses Y and left menu + A")] = False,
    countdown_s: Annotated[float, typer.Option(help="--auto-cycle: seconds counted down on the headset view before each episode")] = 5.0,
    auto_cycle_max_s: Annotated[float, typer.Option(help="--auto-cycle: abandon an episode (left menu + B, new setup) when its end event has not come after this many seconds; 0 = never")] = 0.0,
    auto_cycle_start: Annotated[bool, typer.Option(help="--auto-cycle: start armed, without the SPACE press (headless tests)")] = False,
    viewer_hz: Annotated[int, typer.Option(help="refresh the desktop MuJoCo viewer this many times per second (50 = every 20 ms tick; 10 renders the kitchen every fifth tick and gives the control loop ~4 ms back; measured 2026-11-03: live lean sessions ticked at 27 ms, headless at 20); off when --headless")] = 50,
    wrist_cams: Annotated[bool, typer.Option(help="record the robot's wrist cameras (wrist_left / wrist_right, 640 x 360) when the robot file has them; never streamed to the headset")] = True,
    wrist_hz: Annotated[float, typer.Option(help="render rate of the wrist cameras; frames are repeated to the dataset's 50 fps in between (0 = every tick)")] = 30.0,
    fpv_window: Annotated[bool, typer.Option(help="desktop window with the first-person view, the same frame the headset gets (status lines included); off when --headless")] = True,
    wrist_fov: Annotated[float, typer.Option(help="diagonal field of view of the saved wrist view (equidistant fisheye from three 90-deg faces)")] = 150.0,
    tactile: Annotated[bool, typer.Option(help="with --robot revo2 --record: the fingertip tactile pad columns (tactile.*)")] = True,
    lean_record: Annotated[bool, typer.Option("--lean-record/--full-record", help="with --record: save states, actions and the controller's inputs only -- no images "
                                                   "(ego_view, wrist cameras) and no tactile pads. Those are most of a 20 ms tick in a big scene (the kitchen: "
                                                   "about 15 of 20 ms with the Revo 2 robot) and a loop slower than real time makes the robot feel unstable. "
                                                   "The bit-exact replay puts them back: python -m simple.cli.replay_holomotion_v14 <dataset> --all --fill")] = False,
    stream_port: Annotated[int, typer.Option(help="port the XRoboToolkit app connects to for the video (the app uses 13579)")] = 13579,
    max_seconds: Annotated[float, typer.Option(help="stop after this long (0 = run until Ctrl-C); for tests")] = 0.0,
    log_every: Annotated[float, typer.Option(help="seconds between status lines")] = 2.0,
    slice_id: Annotated[str, typer.Option("--slice", help="record one slice of the long-horizon kitchen task as its own episodes (docs/long_horizon_slicing_scenes, simple.teleop.holomotion_v14.slices): S1 pick up a plate from the counter, S2 carry it to the pedal bin and wipe it, S3 carry it to the dishwasher and open the door with the plate in hand, S4 pull the rack out and load the plate, S5 fetch the second plate, S6 carry it to the dishwasher and load it next to the first, S7 push the rack in and close the door (kitchen7 ep1's order; each slice starts where the previous one ends, the plate already in the hand where ep1 had it, since 2026-11-02). Atomic slices A1-A6 (docs/atomic_slicing_scenes) are S1 + S2 as RoboAtom skills with heavily randomized starts (upper body, hands, location, plate spots and types): A1 Move to the counter, A2 Pick the plate up (start at the counter stance or with the hands at the rim), A3 Move the plate to the bin, A4 Press the pedal, A5 Wipe the plate over the open bin with a foot on the pedal (A3-A5 run with KITCHEN_PEDAL_RETURN=1: the lid is up only while a foot presses the pedal). Every setup starts in the slice's start state drawn from kitchen7 ep1's own states (robot stance or walk corridor, door angle, rack slide, plates in the rack or in the hand, plate spots, bin lid: scene_setup.SLICE_STARTS; a held plate's grip stays closed until the operator presses the grip once); the slice is complete 2 s after its end event (plate lifted, plate over the open bin, door open with a plate held, a plate newly in the rack, racks in and door shut) -- the headset then says 'save now' and left menu + A saves (the episode line records the gate / event / complete frames); left menu + B abandons, left menu + both triggers restarts the same setup. The instruction becomes the slice's (KITCHEN_INSTRUCTION overrides). --scene kitchen only")] = "",
    slice_auto_cut: Annotated[bool, typer.Option(help="with --slice: save the episode by itself when the slice is complete and go on to the next setup (off since 2026-10-30: the operator saves with left menu + A)")] = False,
    start_hold: Annotated[bool | None, typer.Option("--start-hold/--no-start-hold", help="A2: --start-hold keeps the recorded A1 end pose (nudged a little for 4 cm of clearance) until the motion-tracking button -- motion tracking holds the joints, the robot stands still and does not follow the operator; SPACE, the countdown and Y as usual, then a 0.5 s hand-over to the operator's pose. --no-start-hold: walking mode with the walking arm pose, the robot moved back until every part is 10 cm behind the worktop edge (22-36 cm further from the counter: at the recorded distance the walking arm pose is inside the drawer fronts and the robot topples). Default: the slice's start_mode (A2: hold, since 2026-11-04 13:30 -- arms close to the A1 end, standing, as close to the counter as possible)")] = None,
    live_gates: Annotated[bool, typer.Option(help="evaluate the task's gates inside the control loop (gate ticks on the headset, realformat.success at save time). Off since 2026-10-30 (user: the kitchen5 pipeline without the gate checks): the kitchen gates cost 0.25 ms per 20 ms tick with 10 ms spikes; the labels are computed on the bit-exact replay instead (replay_holomotion_v14, --gates is its default)")] = False,
    plate_balls: Annotated[int, typer.Option("--plate-balls", help="--scene kitchen: this many loose 10 mm balls (glass-marble mass, red / blue) resting at the well centre of every plate lying on the worktop (2026-11-04, user: a rolling ball shows whether the lifted plate is horizontal). Measured in the kitchen scene with the plate held still: they stay within 2 mm at 1 deg of tilt and roll to the wall at 2 deg. Sets KITCHEN_PLATE_BALLS (radius KITCHEN_PLATE_BALL_MM, rolling friction KITCHEN_PLATE_BALL_ROLL), recorded with the episode's scene knobs so the bit-exact replay rebuilds them. Default 0: no balls")] = 0,
    plate_set: Annotated[str, typer.Option("--plate-set", help="--scene kitchen: which procedural plates the episodes draw from (setup plate = seed % N, N = the set's size). v1 (default): data/kitchen_plates as recorded so far (50; rim underside 16-30 mm above the counter at the edge: with the plate flat on the counter the 15 mm thick Revo 2 fingertips get 15 mm under the rim of only 14 of the 50, the 19 mm thick thumb under 36 -- scripts/kitchen_plates/check_under_rim.py; the rest get shoved away). lift: data/kitchen_plates_lift (50; 2026-11-04, user: the fingers must reach under the plate to lift it flat): rim underside >= 24 mm, lip >= 18 mm, 50 of 70 candidates that pass the under-rim check (fingertips and thumb >= 15 mm under all 50), the rim pinch lift and the rack fit. v2: data/kitchen_plates_v2 (2026-11-06): the lift set's 50 plates (same indices) plus 20 wide-rim pasta plates (deep well, 43-65 mm rim, 224-265 mm, 400 g; make_plates.py --family pasta, the same three checks) as types 50-69, so the A2 bank starts keep their recorded types and a seed draws a pasta plate 2 times in 7. Sets KITCHEN_PLATES (an absolute path, recorded with the scene knobs; a directory also works)")] = "v1",
    revo2_hand: Annotated[str, typer.Option("--revo2-hand", help="--robot revo2: which Revo 2 hand model (2026-11-04). legacy (default): the hand every recording so far used (HoloMotion module kinematics + Unitree meshes, data/robots/g1_rot0_teleop_revo2 v3). official: BrainCo's own revo2_description v1.1.1 hand on the same robot (data/robots/g1_rot0_teleop_revo2_official, scripts/robots/make_g1_revo2.py --hand official): per-finger distal couplings 1.131 / 1.267 / 1.252 / 1.271 (thumb 1.0945), 80.8 deg finger limits, the thumb zero offsets (+3 deg abduction, +9 / +6 deg MCP / PIP: at equal joint values the thumb tip sits 12 mm from the legacy one), BrainCo's pad links as the tactile pads, its own grip_poses.json (the legacy poses converted by the thumb offsets) and pinch knots. Recorded in run_meta (revo2_hand, robot_mjcf); the replay selects the same hand")] = "legacy",
):
    if plate_balls:
        if scene != "kitchen" or env_id:
            raise typer.BadParameter("--plate-balls is for --scene kitchen")
        if not 1 <= plate_balls <= 4:
            raise typer.BadParameter("--plate-balls: 1..4 per plate")
        os.environ["KITCHEN_PLATE_BALLS"] = str(plate_balls)                                # before the scene module is imported
        os.environ.setdefault("KITCHEN_PLATE_FEET", "4")                                    # balls need still plates: sphere feet under the plates (plate_swap), recorded with the knobs
    if plate_set != "v1":
        if scene != "kitchen" or env_id:
            raise typer.BadParameter("--plate-set is for --scene kitchen")
        plates_dir = {"lift": Path(__file__).resolve().parents[3] / "data/kitchen_plates_lift", "v2": Path(__file__).resolve().parents[3] / "data/kitchen_plates_v2"}.get(plate_set, Path(plate_set))
        if not (plates_dir / "manifest.json").is_file():
            raise typer.BadParameter(f"--plate-set {plate_set}: no manifest.json in {plates_dir} (v1, lift, v2, or a plate directory)")
        os.environ["KITCHEN_PLATES"] = str(plates_dir.resolve())                              # the kit reads it at import; recorded with the KITCHEN_* knobs
        if plate_set == "v2":                                                                  # the A2 bank starts draw their plate types from v2 (else they keep the A1 episodes'
            os.environ.setdefault("KITCHEN_BANK_PLATES", "draw")                              # recorded lift types and never see a pasta plate); KITCHEN_BANK_PLATES=recorded overrides
    if setup_ranges not in ("teleop", "lv3"):
        raise typer.BadParameter("--setup-ranges is teleop or lv3")
    if slice_id:
        if slice_id not in SLICES:
            raise typer.BadParameter(f"--slice {slice_id}: built so far: {', '.join(SLICES)}")
        if scene != "kitchen" or env_id:
            raise typer.BadParameter("--slice is for --scene kitchen")
        if not randomize:
            raise typer.BadParameter("--slice draws the slice's start state for every episode: it needs --randomize")
        os.environ.setdefault("KITCHEN_INSTRUCTION", SLICES[slice_id]["instruction"])   # before the scene module is imported
        for k, v in (SLICES[slice_id].get("env") or {}).items():                       # the slice's scene knobs (the bin scenes: a pedal that springs back)
            os.environ.setdefault(k, v)
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
    if robot_kind not in ("stereo", "stock", "revo2"):
        raise typer.BadParameter("--robot is stereo, stock or revo2")
    robot_uid = "g1_sonic_revo2" if robot_kind == "revo2" else "g1_sonic"
    if revo2_hand not in ("legacy", "official"):
        raise typer.BadParameter("--revo2-hand is legacy or official")
    if robot_kind == "revo2":                                 # 2026-11-04: legacy or official hand model (class file, joint limits, couplings, grip poses, pinch knots)
        from simple.robots.g1_sonic_revo2 import select_hand_variant
        _hv = select_hand_variant(revo2_hand)
        print(f"[HoloMotion v1.4] Revo 2 hand: {revo2_hand} -> {_hv['mjcf']} (grip poses {_hv['grip_poses']}; finger limit {_hv['upper'][3] * 57.2958:.1f} deg, "
              f"couplings {_hv['coupling'][4]:g} / {_hv['coupling'][6]:g} / {_hv['coupling'][8]:g} / {_hv['coupling'][10]:g}, thumb {_hv['coupling'][2]:g})")
    elif revo2_hand != "legacy":
        print("[HoloMotion v1.4] --revo2-hand is for --robot revo2; ignored")
    env = gym.make(env_id, sim_mode="mujoco", render_hz=render_hz, physics_dt=sonic_config["SIMULATE_DT"],
                   headless=headless, max_episode_steps=max_episode_steps, sonic_config=sonic_config, target=target,
                   dr_level=dr_level, success_criteria=0.9, robot_uid=robot_uid)
    sonic_env: SonicLocoManipEnv = env.unwrapped  # type: ignore
    key_sink: list = [None]                                                    # the keyboard stand-in, once it exists: the sim window's keys go to it too
    sonic_env.viewer_key_hook = lambda code: key_sink[0].press_code(code) if key_sink[0] is not None else None
    task, robot = sonic_env.task, sonic_env.task.robot
    assert isinstance(robot, G1Sonic)
    if not live_gates:                                        # the env calls both every step; with the stubs no gate code runs in the loop
        task.compute_reward = lambda info, *a, **k: 0.0
        task.check_success = lambda info, *a, **k: False
        print("[HoloMotion v1.4] task gates off in the control loop (labels come from the replay: replay_holomotion_v14 --gates); "
              "--live-gates shows them live")
    from simple.teleop.holomotion_v14.robot_variants import teleop_mjcf, tilt_head_sensor
    if robot_kind == "revo2":                                 # the rot0 stereo robot file with the Revo 2 hands: cameras, backpack and black hands are built in
        if head_tilt_deg or abs(backpack_kg - 3.2) > 1e-6:
            print(f"[HoloMotion v1.4] --robot revo2: the robot file has the 3.2 kg backpack and a 0 deg head; --backpack-kg {backpack_kg:g} / --head-tilt-deg {head_tilt_deg:g} are not applied to it")
            head_tilt_deg = 0.0
    else:
        robot.mjcf_path = teleop_mjcf(robot_kind, backpack_kg, tilt_deg=head_tilt_deg, black_hands=black_hands)
    stereo_like = robot_kind in ("stereo", "revo2")
    if head_tilt_deg:
        if not scene:
            raise typer.BadParameter("--head-tilt-deg works with --scene (the scene kits' head-camera pose patch)")
        tilt_head_sensor(task, head_tilt_deg)
        print(f"[HoloMotion v1.4] head cameras pitched {head_tilt_deg:+g} deg ({'down' if head_tilt_deg > 0 else 'up'})")
    print(f"[HoloMotion v1.4] robot: {robot.mjcf_path} ({ {'stereo': 'G1 + HBVCAM stereo head camera', 'revo2': 'rot0 G1 + HBVCAM stereo head camera + BrainCo Revo 2 hands'}.get(robot_kind, 'stock G1') }, "
          f"{backpack_kg:g} kg backpack)")
    if stream_camera not in ("pinhole", "fisheye", "head"):
        raise typer.BadParameter("--stream-camera is pinhole, fisheye or head")
    if stream_camera in ("pinhole", "fisheye") and not stereo_like:
        print(f"[HoloMotion v1.4] --stream-camera {stream_camera} needs --robot stereo: streaming the scene's head camera instead")
        stream_camera = "head"
    if record_camera == "auto":
        record_camera = stream_camera                   # record what the headset shows
    if record_camera not in ("head", "pinhole", "fisheye"):
        raise typer.BadParameter("--record-camera is auto, head, pinhole or fisheye")
    if record_camera != "head" and not stereo_like:
        print(f"[HoloMotion v1.4] --record-camera {record_camera} needs --robot stereo: recording the scene's head camera instead")
        record_camera = "head"
    from simple.teleop.holomotion_v14.hbvcam_views import FisheyeView, PinholeView

    def make_view(kind: str):
        return FisheyeView(sonic_env, face_px=fisheye_face_px) if kind == "fisheye" else PinholeView(sonic_env)

    if lean_record and not record:
        print("[HoloMotion v1.4] --lean-record does nothing without --record")
        lean_record = False
    rec_view = make_view(record_camera) if (record and record_camera != "head" and not lean_record) else None
    hb_view = None
    if pico_stream and stream_camera != "head":
        hb_view = rec_view if stream_camera == record_camera and rec_view is not None else make_view(stream_camera)
    next_stream_t = 0.0
    # The env renders head_stereo_left/right (640 x 360) on every step for `observation`; only the scene-camera stream /
    # recording and the desktop window without a stereo view show them. The stereo views render their own pictures, so
    # skip the env's (the kitchen: 5 ms of the 20 ms tick, 2026-10-26).
    need_env_images = ((record and record_camera == "head") or stream_camera == "head"
                       or (hb_view is None and (pico_stream or (fpv_window and not headless))))
    sonic_env.mujoco.render_observation_cameras = need_env_images
    if not need_env_images:
        print("[HoloMotion v1.4] the scene's per-step head renders are off (the stereo view renders its own pictures)")

    try:
        rec_w, rec_h = (int(v) for v in record_size.lower().split("x"))
    except ValueError:
        raise typer.BadParameter(f"--record-size is WxH, e.g. 640x360 (got {record_size!r})")
    import cv2

    def small(img):
        """A rendered eye at the recorded size (2026-10-18: 640x360 by default, area-averaged from the full render)."""
        if img is None or (img.shape[1], img.shape[0]) == (rec_w, rec_h):
            return img
        return cv2.resize(img, (rec_w, rec_h), interpolation=cv2.INTER_AREA)

    def recorded_view():
        """The recorded camera's images of the current (pre-step) state: (left image to save, (left, right) at full size
        for the stream, right eye to save or None)."""
        if rec_view is None:
            return None, None, None
        if record_camera == "fisheye":                          # both eyes fresh every step: a consistent pair
            left, right = rec_view.render()
            return np.hstack([small(left), small(right)]), (left, right), None
        if record_camera == "pinhole":                          # both rectified eyes of the same state (2026-10-16)
            left, right = rec_view.render_pair()
            return small(left), (left, right), small(right)
        img = rec_view.render()
        return img, (img, None), None
    control_decimal = int(round(1 / sonic_config["SIMULATE_DT"] / render_hz))
    assert control_decimal == 4, "the v1.4 policy runs at 50 Hz (4x decimation at 200 Hz physics)"
    control_dt = control_decimal * robot.sim_dt

    if stream_view not in ("single", "mono", "stereo"):
        raise typer.BadParameter("--stream-view is single, mono or stereo")
    if stream_camera == "fisheye":
        stream_view = "stereo"                              # the two fisheye eyes are the frame's two halves
    agent = HoloMotionV14Agent(robot, reference_uri=reference_uri, auto_stand=auto_stand, enable_pico_stream=pico_stream,
                               stream_view=stream_view, button_map=button_map, stream_port=stream_port,
                               hand_source=hand_source, hand_sides=tuple(x.strip() for x in hand_sides.split(",") if x.strip()),
                               hand_calib_dir=hand_calib_dir, hand_min_conf=hand_min_conf, hand_smooth_hz=hand_smooth_hz)
    agent.num_episodes = num_episodes
    controls = SimControls(agent)
    global RAW_HANDS
    RAW_HANDS = hand_source == "skeleton"
    keys = cycle = None
    if hand_source == "skeleton" or auto_cycle:                              # the keyboard stands in for the controllers' buttons (and SPACE arms the cycle)
        from simple.teleop.holomotion_v14.pico_keys import PicoKeys
        keys = PicoKeys.start()
        key_sink[0] = keys
        if keys is None and auto_cycle and not auto_cycle_start and not os.environ.get("KITCHEN_AUTO_CYCLE_SPACE_S"):
            raise typer.BadParameter("--auto-cycle needs the keyboard (stdin is not a terminal): use --auto-cycle-start")
        if keys is None:
            keys = PicoKeys()                                                # no terminal: programmatic presses only
        agent.pico_overlay = keys
    if auto_cycle:
        if not slice_id:
            raise typer.BadParameter("--auto-cycle needs --slice (its end event is what saves the episode)")
        from simple.teleop.holomotion_v14.auto_cycle import AutoCycle
        slice_auto_cut = True
        cycle = AutoCycle(keys, countdown_s=countdown_s, motion_button=agent.motion_button, start_armed=auto_cycle_start, max_record_s=auto_cycle_max_s)
        print(f"[AutoCycle] {'armed' if auto_cycle_start else 'SPACE on the keyboard arms it'}: countdown {countdown_s:g} s -> {agent.motion_button} (motion tracking + recording) -> "
              f"the slice's end event + overlap save the episode -> new setup -> next countdown; SPACE pauses"
              + (f"; an episode without its end event after {auto_cycle_max_s:g} s is abandoned" if auto_cycle_max_s else ""), flush=True)

    setups = (SceneSetups(scene, mod, task, max_distractors=max_distractors, ranges=setup_ranges, slice=slice_id)
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

    grip_latch = GripLatch()                                                   # --slice with plate_in_hand: the hand stays closed until the operator squeezes
    start_touch: list = []                                                     # what the robot touched at its start (slice scenes; empty = clean)
    start_t = [0.0]                                                            # when the current start was placed (the settle watch runs START_SETTLE_S from it)
    ready_latch = [False]                                                      # the quick start has reported the robot standing (level; the one-shot quick_ready is
                                                                               # consumed long before the operator presses SPACE -- 2026-11-04: no countdown)
    start_held = [None]                                                        # the plate body in the hand at the start (its contacts are the grasp, not a touch)
    ref_lost_at = 0.0                                                          # when the headset's body tracking last went missing (0 = not missing)
    viewer_every = max(1, int(round(50 / max(1, viewer_hz)))); viewer_tick = 0  # --viewer-hz: the desktop viewer every N ticks

    def init_robot(quick: bool) -> None:
        """After env.reset: the start pose (quick start: standing, else hanging on the leash), randomized if asked; a slice
        preset's plate put into the hand (grasp_start.hold_plate) and that grip latched."""
        nonlocal observation, privileged_info, init_pose
        recorded = bool(setups is not None and (setup.get("start") or {}).get("bank"))   # a recorded end as the start (A2 from the A1 ends): no random offset on it (user 2026-11-04)
        init_pose = _draw_init_pose(init_rng, robot, init_pose_scale) if (random_init_pose and not recorded) else None
        preset = setup.get("preset") if setups is not None else None
        shift = spawn_shift(setup) if setups is not None and (setup.get("start") or {}).get("spawn_back_m") else None
        held = posed = footed = nudged = None
        if quick:
            placed = _stand_on_floor(sonic_env, agent, init_pose, shift)
            posed = apply_pose(sonic_env, preset)                          # an atomic slice: ep1's whole-body pose at the boundary
            footed = foot_on_pedal(sonic_env, preset)                      # the Wipe: one foot onto the pressed pedal
            held = hold_plate(sonic_env, preset)
            if clean_start and posed and start_hold:
                nudged = nudge_clear(sonic_env, held["plate"] if held else None)    # user 2026-11-04: the A1 end adjusted a little so nothing is near the scene
            if clean_start and posed and not start_hold:
                arms_to_walk_default(sonic_env)                            # A2 in walking mode: the recorded arms would be swept down through the plates anyway
                nudged = nudge_back_from_edge(sonic_env, NUDGE_WALK_CLEAR_M)   # back until every part is 10 cm behind the worktop edge
            if held or posed or footed:
                placed = body_joint_qpos(sonic_env)                        # the quick start holds that pose, not the default one
            pin = agent.arm_real_idx(held["side"]) if (held and start_hold and slice_id and SLICES[slice_id].get("hold_pin") == "held_arm") else None
            agent.quick_start(hold_real=placed[agent.mjcf_to_real], auto_motion=auto_motion,
                              hold_pose=start_hold and bool(posed or held or footed),   # --start-hold / start_mode "hold": the placed pose kept until the operator takes over (A2's recorded end, A3's level hold)
                              pin_real_idx=pin)                                         # A3: the holding arm pinned through the hold (the motion policy lets a loaded arm sag 5-10 deg)
        else:
            if shift is not None:
                _shift_root(sonic_env, shift)
            if init_pose is not None:
                _perturb_hanging(sonic_env, init_pose)
            posed = apply_pose(sonic_env, preset)
            footed = foot_on_pedal(sonic_env, preset)
            held = hold_plate(sonic_env, preset)
        grip_latch.arm(preset)
        start_held[0] = held["plate"] if held else None
        start_touch[:] = _start_touches(sonic_env, start_held[0]) if (quick and setups is not None and clean_start) else []
        if footed:
            print(f"[Slice] {footed['foot']} foot on the pedal tread (IK error {footed['err_mm']} mm, {footed['err_deg']} deg)")
        if shift is not None:
            print(f"[Slice] spawned {np.hypot(*shift):.2f} m behind the start for the reset steps, moved forward to it")
        if posed:
            print(f"[Slice] start pose: " + ", ".join(f"{k} {v}" for k, v in posed.items() if k != "pelvis_z") + f" (pelvis {posed['pelvis_z']:.3f} m)")
        if clean_start and posed and not start_hold:
            print("[Slice] walking-mode start: the arms at the walking policy's default" + ("" if nudged is None else f", the robot {nudged['back_m'] * 100:.0f} cm further back than the recorded end"
                  if nudged["ok"] else ", no step back cleared 10 cm (kept as recorded, skipped if it touches)") + " (--start-hold keeps the recorded pose there)")
        elif nudged:
            near = ", ".join(f"{a_} {b_:.0f} mm from {c_}" for a_, c_, b_ in [(t[0], t[1], t[2]) for t in nudged["before"][:3]])
            if nudged["ok"]:
                print(f"[Slice] start nudged for clearance: {nudged['back_m'] * 100:.0f} cm back, shoulders {nudged['arms_up_rad']:.2f} rad up (as recorded: {near})")
            else:
                print(f"[Slice] start within 4 cm of the scene and no small nudge clears it ({near}): kept as recorded, skipped if it touches")
        if nudged and nudged["ok"]:
            setup.setdefault("start", {})["nudge"] = dict(back_m=nudged["back_m"], arms_up_rad=nudged["arms_up_rad"], mode="walk" if not start_hold else "hold")
        if grip_latch.active():
            print(f"[Slice] hand(s) held at their start pose until the operator presses the grip once: {', '.join(grip_latch.active())}")
        if held:
            print(f"[Slice] {held['plate']} {'pinched by' if held['grasp'] == 'pinch' else 'in'} the {held['side']} hand ({held['radius_m'] * 200:.0f} cm across) at {held['plate_xyz']}; "
                  f"the {held['side']} grip is held closed until the operator presses the grip once ({held['source']})")
        if quick or init_pose is not None or held or posed:
            observation, privileged_info = sonic_env._get_obs(), sonic_env._get_info()
        if init_pose is not None:
            b = init_pose["base"]
            print(f"[HoloMotion v1.4] random start pose: base ({b[0]:+.3f}, {b[1]:+.3f}) m {b[2]:+.1f} deg, "
                  f"joints max |dq| {max(abs(v) for v in init_pose['dq']):.3f} rad")

    def start_robot() -> None:
        """init_robot, redrawn with a new setup while the robot touches anything as placed (clean-start slices, i.e. A2; user
        2026-11-03: the robot must not touch anything at the beginning -- A2 only). Checked once here, never under the operator."""
        nonlocal observation, privileged_info
        for attempt in range(8):
            init_robot(quick=quick_start)
            start_t[0] = time.monotonic(); ready_latch[0] = False
            if not start_touch:
                return
            print(f"[HoloMotion v1.4] start touches " + ", ".join(f"{a} -> {b} ({mm} mm)" for a, b, mm in start_touch[:4]) + f": new setup ({attempt + 1})", flush=True)
            next_setup()
            observation, privileged_info = env.reset()
            if slice_cut is not None:
                slice_cut.bind(sonic_env.mujoco.mjModel)
            agent.reset_policy()
        print("[HoloMotion v1.4] 8 starts in a row touched something: starting anyway", flush=True)

    slice_cut = SliceCut(slice_id) if slice_id else None
    clean_start = bool(slice_id and SLICES[slice_id].get("clean_start"))   # A2: no contact at the start (placement nudge / redraw)
    if start_hold is None:
        start_hold = bool(slice_id and SLICES[slice_id].get("start_mode") == "hold")
    if clean_start:
        print(f"[Slice {slice_id}] start: " + ("the recorded A1 end pose held by motion tracking until the operator's motion button (nudged for 4 cm of clearance; --no-start-hold: walking mode, moved back from the counter)"
                                           if start_hold else "walking mode with the walking arm pose, moved back until 10 cm clear of the worktop edge (--start-hold: the recorded pose held at the counter)"))
    elif start_hold:
        from simple.teleop.holomotion_v14.scene_setup import A3_START, A4_START, A5_START
        what = ("a recorded A2 end: the robot, its pose and the plate in the right hand as that episode had them when the A2 gate completed"
                if slice_id == "A3" and A3_START in ("a2_press", "a2_final") else
                "a recorded A3 end: the robot at the bin with its pose and the plate in the right hand as that episode finished"
                if slice_id == "A4" and A4_START == "a3_end" else
                "a recorded A4 end: the right foot on the pedal holding the lid open, the plate in the right hand, as that episode finished"
                if slice_id == "A5" and A5_START == "a4_end" else "the plate held level, the arm raised as in A2 ep41")
        print(f"[Slice {slice_id}] start: the placed pose ({what}) kept by motion tracking until the operator's motion button; "
              "SPACE, the countdown and Y as usual, then a 0.5 s hand-over to the operator's pose (--no-start-hold: walking mode, which swings the holding arm down)")
    if slice_cut is not None:
        if os.environ.get("KITCHEN_SLICE_GATE_TEST_S"):
            print(f"[Slice {slice_id}] TEST KNOB KITCHEN_SLICE_GATE_TEST_S={os.environ['KITCHEN_SLICE_GATE_TEST_S']}: the end gate counts as met from that "
                  f"recording time on, whatever the scene does -- headless tests only, never for data", flush=True)
        print(f"[Slice {slice_id}] {slice_cut.cfg['name']}: complete {slice_cut.cfg['overlap_s']:g} s after the end event "
              f"({slice_cut.cfg['end_text']}), then "
              + ("the episode is saved by itself (--slice-auto-cut)" if slice_auto_cut else "the headset says 'save now' and left menu + A saves")
              + f"; instruction: {os.environ.get('KITCHEN_INSTRUCTION', '')!r}")
    next_setup()
    observation, privileged_info = env.reset()
    if slice_cut is not None:
        slice_cut.bind(sonic_env.mujoco.mjModel)
    agent.reset_policy()
    start_robot()                                 # quick start: standing, no leash, from the first scene on; redrawn while it touches anything
    obj_names = list(sonic_env.mujoco.mj_objects.keys())
    if setups is not None:
        # the LeRobot schema is fixed, the distractor count is not: slots for the task objects + max_distractors
        obj_names = [n for n in obj_names if not n.startswith("distractor_")] + [f"distractor_{i}" for i in range(max_distractors)]
    exporter = robot_model = None
    xlog = None
    rf_sampler = None
    revo2_rec = None
    wrist_rec = None
    use_images = bool(record and not lean_record)
    if record:
        from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
        robot_model = instantiate_g1_robot_model(waist_location="lower_body")
        # a dataset holds one image size: recordings of the stereo camera get their own folder
        run_save_dir = f"{os.path.abspath(save_dir)}/{env_id}/level-{dr_level}" + ("" if record_camera == "head" else f"_{record_camera}")
        _check_dataset_dir(run_save_dir)
        ego_shape = ({"fisheye": (rec_h, 2 * rec_w, 3), "pinhole": (rec_h, rec_w, 3)}[record_camera] if record_camera in ("fisheye", "pinhole")
                     else observation["head_stereo_left"].shape)
        policy_obs_dim = max(int(sess.get_inputs()[0].shape[-1]) for sess in (agent.node.motion_policy_session, agent.node.velocity_policy_session))
        hand_kind = "revo2" if robot_kind == "revo2" else "dex3"
        from simple.teleop.holomotion_v14 import wrist_cams as WC
        has_wrist = bool(wrist_cams and WC.available(sonic_env.mujoco.mjModel))
        use_wrist = has_wrist and not lean_record
        use_tactile = bool(tactile and not lean_record)
        exporter = _init_exporter(run_save_dir, task.instruction, robot_model, obj_names, robot.joint_names, ego_shape, realformat,
                                  right_eye=(record_camera == "pinhole"), policy_obs_dim=policy_obs_dim, hand=hand_kind, wrist=use_wrist, wrist_fov=wrist_fov,
                                  images=use_images, tactile=use_tactile)
        if lean_record:
            import json as _json
            spec_path = Path(run_save_dir) / "meta" / "lean_record.json"
            if not spec_path.exists():
                spec_path.write_text(_json.dumps(lean_spec(record_camera, (rec_w, rec_h), has_wrist, wrist_hz, wrist_fov, tactile, hand_kind), indent=2))
            print(f"[Record] LEAN recording: no images{' (wrist cameras included)' if has_wrist else ''}"
                  f"{' and no tactile pads' if hand_kind == 'revo2' and tactile else ''} -- the headset still gets its picture. "
                  f"Put them back with: python -m simple.cli.replay_holomotion_v14 {run_save_dir} --all --fill")
        if use_wrist:
            wrist_rec = WC.WristCams(sonic_env, hz=wrist_hz, fov_deg=wrist_fov)
            print(f"[Record] wrist cameras: {WC.NAMES['left']} + {WC.NAMES['right']}, {wrist_fov:g} deg diagonal fisheye view at {WC.W} x {WC.H} from three 90-deg faces, "
                  f"new renders at {wrist_hz:g} Hz repeated to 50 fps (observation.images.wrist_left / wrist_right, wrist_cams.fresh; not streamed to the PICO)")
        elif wrist_cams and not lean_record:
            print("[Record] wrist cameras: none in this robot file (scripts/robots/add_cameras_mounts.py adds them to the Revo 2 robot)")
        print(f"[Record] raw HoloMotion inputs on (policy obs {policy_obs_dim}); " + ("both rectified eyes recorded" if record_camera == "pinhole" else "one image stream"))
        if hand_kind == "revo2":
            from simple.teleop.holomotion_v14.revo2_record import Revo2Recorder, features as _r2f
            revo2_rec = Revo2Recorder(sonic_env, tactile=use_tactile)
            print(f"[Record] Revo 2 hands: 51-wide state/action, {len(_r2f(tactile=use_tactile))} joint/motor{'/tactile' if use_tactile else ''} columns")
        if realformat:
            from simple.teleop.holomotion_v14 import realformat as RF
            rf_sampler = RF.LiveSampler(sonic_env, hand=hand_kind)
            print(f"[Record] real-format fields on ({len(RF.all_features(hand_kind))} extra columns)")
        print(f"[Record] ego_view = {record_camera} camera, {ego_shape[1]} x {ego_shape[0]}")
        print(f"[Record] saving to {run_save_dir}")
        if exact_log:
            xlog = ExactLog(sonic_env)
            prefix = SCENE_ENV_PREFIX.get(scene, "\0")
            knobs = {k: v for k, v in os.environ.items() if k.startswith(prefix) or k.startswith("SIMPLE_")}
            run_meta = dict(scene=scene, scene_root=str(scene_root()), env_id=env_id, dr_level=dr_level, env_knobs=knobs,
                            setup_ranges=setup_ranges if setups is not None else None, slice=slice_id or None,
                            backpack_kg=backpack_kg, robot=robot_kind, head_tilt_deg=head_tilt_deg, black_hands=bool(black_hands),
                            stream_camera=stream_camera, record_camera=record_camera, lean_record=bool(lean_record),
                            record_size=[rec_w, rec_h], wrist_hz=wrist_hz, wrist_fov=wrist_fov,
                            robot_mjcf=robot.mjcf_path, revo2_hand=revo2_hand if robot_kind == "revo2" else None, physics_dt=sonic_config["SIMULATE_DT"], render_hz=render_hz,
                            policy_models_dir=os.environ.get("HOLOMOTION_UNITREE_POLICY_MODELS_DIR", ""),
                            velocity_onnx=str(agent.node.velocity_onnx_path), motion_onnx=str(agent.node.motion_onnx_path),
                            hand_source=hand_source, hand_skeleton=getattr(agent, "hand_meta", None))

    rec_state = RecordingState.WAITING_FOR_LANDING
    episode_init_pose = None
    start_state, start_summary = None, None             # the controller's state at the start of the recording's frame 0
    episodes_saved, sim_cnt, pbar = 0, 0, None
    t_start = time.monotonic()
    space_knob_done = [False]
    if os.environ.get("KITCHEN_AUTO_CYCLE_SPACE_S"):
        print(f"[AutoCycle] TEST KNOB KITCHEN_AUTO_CYCLE_SPACE_S={os.environ['KITCHEN_AUTO_CYCLE_SPACE_S']}: SPACE is pressed for you that many seconds in -- headless tests only", flush=True)
    t_log = t_start
    prev_start, tick_ms, slow_warned, pace_n = None, [], False, 0   # loop pacing: wall time per 20 ms control tick (2026-10-26)
    print(f"[HoloMotion v1.4] waiting for the reference publisher at {reference_uri}")
    print(f"  A = walk, {agent.motion_button} = motion tracking; left menu + A = record/save")

    def reset_scene(reason: str, new_setup: bool):
        nonlocal observation, privileged_info, rec_state, sim_cnt, pbar, prev_start, pace_n
        prev_start, pace_n = None, 0
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
        if slice_cut is not None:
            slice_cut.bind(sonic_env.mujoco.mjModel)
        agent.reset_policy()
        start_robot()
        rec_state, sim_cnt = RecordingState.WAITING_FOR_LANDING, 0
        if cycle is not None:
            cycle.scene_restarted()
        print(f"[HoloMotion v1.4] scene reset ({reason}{'; quick start' if quick_start else ''})")

    success_told = False                           # the one-time "task success latched" line of the current recording

    def episode_success() -> dict:                 # realformat.success: the task's last gate, or the slice's end event (--slice)
        if slice_cut is not None:
            return slice_cut.success(int(exporter.episode_buffer["size"]))
        return _task_success(privileged_info, exporter.episode_buffer)

    fpv = None
    if fpv_window and not headless:
        from simple.teleop.holomotion_v14.fpv_window import FpvWindow
        fpv = FpvWindow(on_key=(lambda code: keys.press_code(code, from_cv=True)) if keys is not None else None)
        print("[HoloMotion v1.4] first-person window on (the headset's frame; --no-fpv-window to hide it)" + ("; its keys reach the keyboard stand-in" if keys is not None else ""))
    try:
        while not max_seconds or time.monotonic() - t_start < max_seconds:
            step_start = time.monotonic()
            if prev_start is not None:
                tick_ms.append((step_start - prev_start) * 1e3)
            prev_start = step_start
            ego_img, rec_pair, rec_right = recorded_view()
            action = agent.get_action(observation, instruction=task.instruction, privileged_info=privileged_info)
            if grip_latch.active():                                      # plate-in-hand start: keep that hand closed until the operator squeezes
                before = grip_latch.active(); grip_latch.apply(action)
                if not grip_latch.active():
                    print(f"[Slice] the operator has the {before[0]} hand now ({'hand closed on the plate' if grip_latch.kind.get(before[0]) == 'plate' else 'grip pressed / hand tracked'}): latch released")
            frame_obs = observation if ego_img is None else {**observation, "head_stereo_left": ego_img}
            if record_camera == "pinhole" and rec_right is not None:
                frame_obs = {**frame_obs, "head_stereo_right_rect": rec_right}
            frame_inputs = {"observation": frame_obs, "action": action, "privileged_info": privileged_info}
            if rf_sampler is not None:                                   # every step: the first recorded frame is decided after it
                frame_inputs["phys"] = rf_sampler.sample(action)          # the pre-step state, as observation.state
            if revo2_rec is not None:
                frame_inputs["revo2"] = revo2_rec.sample(action)          # Revo 2 joints, motors, grip and tactile pads, same instant
            if wrist_rec is not None:
                frame_inputs["wrist"] = wrist_rec.render()                # the two wrist cameras on the same pre-step state
            if xlog is not None:
                xlog.before_step()
            observation, reward, terminated, truncated, privileged_info = env.step(action)
            viewer_tick += 1
            if not headless and viewer_tick % viewer_every == 0:
                sonic_env.update_viewer()
            ev = controls.poll()
            # the headset's body tracking stopping is a silent freeze otherwise (2026-11-03: "device missing" in the publisher's
            # SDK, controller samples still coming, the robot holding on a stale reference): say so once, and when it is back
            ref_age = agent.sub.obs_age() if getattr(agent, "sub", None) is not None and agent.sub.n_obs else 0.0
            if ref_age > REF_LOST_S and not ref_lost_at:
                ref_lost_at = time.monotonic() - ref_age
                print(f"[HoloMotion v1.4] {time.strftime('%H:%M:%S')} NO BODY TRACKING from the headset for {ref_age:.1f} s: the robot holds on the "
                      f"last reference (headset off the head or asleep, the app in the background, a Wi-Fi stall, or another XRoboToolkit client "
                      f"attached to the SDK); the recording and the episode keep running; see the publisher log", flush=True)
            elif ref_lost_at and ref_age < 0.5:
                print(f"[HoloMotion v1.4] {time.strftime('%H:%M:%S')} body tracking resumed after {time.monotonic() - ref_lost_at:.0f} s", flush=True)
                ref_lost_at = 0.0
            quick_ready, quick_engaged, handover = agent.quick_ready, agent.quick_engaged, agent.handover_started   # one-shots: read once per tick
            if quick_ready:
                ready_latch[0] = True
            if os.environ.get("KITCHEN_AUTO_CYCLE_SPACE_S") and keys is not None and not space_knob_done[0] \
                    and time.monotonic() - t_start >= float(os.environ["KITCHEN_AUTO_CYCLE_SPACE_S"]):
                space_knob_done[0] = True; keys.press(" "); print("[AutoCycle] TEST KNOB: SPACE pressed for you (KITCHEN_AUTO_CYCLE_SPACE_S)", flush=True)
            # (no contact watch in the loop: a start that comes to touch something is not reset under the operator -- 2026-11-04, after a
            # session where the watch rebuilt the scene start after start; the placed start is checked once, in start_robot)
            if cycle is not None:
                cycle.step(time.monotonic(), rec_state.name, ready_latch[0], episodes_saved, num_episodes, blocked=agent.handover_blocked)
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
                if start_hold and (agent.quick_holding or handover):
                    auto_record = handover                      # a held start (A2, A3): the recording starts when the hand-over begins, not on a button the hold refused

            if ev["reset"]:
                reset_scene("left menu + both triggers", new_setup=False)
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
                    if slice_cut is not None:
                        slice_cut.reset()
                elif (ev["save"] or auto_record) and rec_state == RecordingState.WAITING_FOR_LANDING:
                    rec_state = RecordingState.RECORDING
                    episode_init_pose = init_pose
                    # the controller's state at the start of this tick = what frame 0 starts from: cold when motion
                    # tracking engaged on this very tick (--record-on-motion), warm otherwise; kept with the exact log
                    # so the node can be restored and the recording replayed from frame 0 (node_state.py)
                    start_state = getattr(agent, "tick_state", None)
                    if start_state is not None and getattr(agent, "tick_agent_state", None) is not None:
                        import json as _json                    # the agent's own state too (rerecord_holomotion_v14)
                        start_state = dict(start_state, agent_json=np.array(_json.dumps(agent.tick_agent_state)))
                    start_summary = None
                    if start_state is not None:
                        a0 = getattr(frame_inputs["action"], "parameters", frame_inputs["action"])
                        start_summary = NS.summary(start_state, ran_motion_on_frame0=a0.get("raw_policy_mode") == "motion")
                    if xlog is not None:
                        xlog.start()
                    if slice_cut is not None:
                        slice_cut.reset()
                    success_told = False
                    pbar = tqdm(desc=f"Recording episode {episodes_saved + 1}", unit="frame", leave=False)
                    why = ("" if not auto_record else f" ({agent.motion_button} pressed)" if record_on_motion
                           else " (auto, from the random start pose)" if random_init_pose else " (auto, quick start)")
                    print(f"[Record] recording{why} "
                          "(left menu + A to save, left menu + B to abandon)")
                    if start_summary is not None:
                        print("[Record] HoloMotion " + ("COLD start: motion tracking engaged on frame 0"
                                                       if start_summary["cold_start"] else
                                                       f"WARM start: frame 0 in {start_summary['policy_mode']} mode, "
                                                       f"motion step {start_summary['motion_step_idx']}")
                              + "; controller state snapshot kept for the replay")
                elif ev["save"] and rec_state == RecordingState.RECORDING:
                    rec_state = RecordingState.EPISODE_DONE
                if rec_state == RecordingState.RECORDING:
                    exporter.add_frame(_build_frame(robot_model, obj_names, **frame_inputs, policy_obs_dim=policy_obs_dim, images=use_images))
                    if rf_sampler is not None:
                        rf_sampler.frame_recorded()
                    if xlog is not None:
                        xlog.add_frame(frame_inputs["action"])
                    if pbar is not None:
                        pbar.update(1)
                    # 2026-10-30, user: no automatic stop or reset -- task success only tells; left menu + A saves. A recording longer
                    # than --max-episode-steps (30000 = 10 min) still saves, so a forgotten recording is not lost.
                    # 2026-11-03: the RECORDING's own length, not the env's `truncated` (steps since its last reset): the sim keeps
                    # stepping while the operator stands by, so after 10 min of standing by the first recorded frame was already past
                    # the env's limit and was saved as a 1-frame episode (kitchen_a1 ep0: record pressed 16 min after the start).
                    if int(exporter.episode_buffer["size"]) >= max_episode_steps:
                        rec_state = RecordingState.EPISODE_DONE
                        print("[Record] episode step limit reached; saving")
                    elif terminated and not success_told:
                        success_told = True
                        print("[Record] task success latched; recording continues until left menu + A")
                    if slice_cut is not None and rec_state == RecordingState.RECORDING:   # the slice's end event: told, or (--slice-auto-cut) the cut
                        prog_now = (privileged_info.get("task_progress") or {}) if isinstance(privileged_info, dict) else {}
                        what = slice_cut.update(prog_now, sonic_env.mujoco.mjData, int(exporter.episode_buffer["size"]) - 1)
                        if what == "gate":
                            print(f"[Slice {slice_id}] {slice_cut.cfg['end_gate']} at frame {slice_cut.gate_frame}; {slice_cut.after_gate_text()}")
                        elif what == "event":
                            print(f"[Slice {slice_id}] {slice_cut.event_text()} at frame {slice_cut.event_frame}: "
                                  + (f"saving in {slice_cut.cfg['overlap_s']:g} s" if slice_auto_cut else f"the slice is complete in {slice_cut.cfg['overlap_s']:g} s, then save with left menu + A"))
                        elif what == "cut":
                            if slice_auto_cut:
                                rec_state = RecordingState.EPISODE_DONE
                                print(f"[Slice {slice_id}] cut at frame {slice_cut.cut_frame}; saving")
                            else:
                                print(f"[Slice {slice_id}] complete at frame {slice_cut.cut_frame}: save with left menu + A (recording continues)")
                if rec_state == RecordingState.EPISODE_DONE:
                    ep_idx = exporter.episode_buffer["episode_index"]
                    extra = {"scene_setup": setup, "init_pose": episode_init_pose}
                    if slice_cut is not None:
                        extra["slice"] = slice_cut.meta(int(exporter.episode_buffer["size"]))
                    with _SaveGuard(f"episode {episodes_saved + 1}", exporter.episode_buffer["size"]):
                        if rf_sampler is not None:
                            RF.finish_episode(exporter.episode_buffer, rf_sampler)
                            extra["realformat"] = RF.episode_meta(dict(backpack_kg=backpack_kg, head_tilt_deg=head_tilt_deg, **episode_success()), setup,
                                                                  xlog.model_sha if xlog is not None else "")
                        exporter.save_episode()
                        _dump_obs_layouts(agent, run_save_dir)
                        _save_episode_env_config(exporter, task, ep_idx)
                        if xlog is not None:
                            log_path = Path(run_save_dir) / "replay" / f"episode_{ep_idx:06d}.npz"
                            xlog.save(log_path, setup, dict(run_meta, episode_index=int(ep_idx), init_pose=episode_init_pose,
                                                            environment_config=_env_config_json(task), holomotion_start=start_summary),
                                      save_model=save_model, extra=NS.pack(start_state))
                            extra["exact_log"] = str(log_path.relative_to(run_save_dir))
                        if start_summary is not None:
                            extra["holomotion_start"] = start_summary
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
                                                     privileged_info, quick_start, record_on_motion, slice_cut, slice_auto_cut)
                if cycle is not None:
                    agent.stream_status = cycle.lines(time.monotonic()) + agent.stream_status
            if hb_view is not None and (agent.streaming or fpv is not None):
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
                    if agent.streaming:
                        agent.update_render_caches({"head_stereo_left": pair[0], "head_stereo_right": pair[1]})
                    if fpv is not None:                         # the same frame on the desktop (2026-10-26)
                        fpv.show(pair[0], pair[1], getattr(agent, "stream_status", None), stream_view)
            else:
                agent.update_render_caches(observation)
                if fpv is not None and time.monotonic() >= next_stream_t:
                    next_stream_t = time.monotonic() + 1.0 / 30.0
                    fpv.show(observation.get("head_stereo_left"), observation.get("head_stereo_right"), getattr(agent, "stream_status", None), stream_view)

            now = time.monotonic()
            if log_every and now >= t_log:
                prog = (privileged_info.get("task_progress") or {}) if isinstance(privileged_info, dict) else {}
                gates = " ".join(k for k in ("door_open", "at_bowl", "grasped", "at_bin", "at_basin", "at_cart", "at_sink", "placed", "rack_in", "door_closed") if prog.get(k))
                base = privileged_info["proprio"]["floating_base_pose"]
                w, x, y, z = base[3:7]
                yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                st = agent.node.runtime.state
                pace = ""
                if tick_ms:
                    arr = np.asarray(tick_ms); mean_ms = float(arr.mean())
                    pace = f" tick={mean_ms:4.1f}ms(max {arr.max():3.0f}, late {100 * float((arr > 25).mean()):2.0f}%)"
                    pace_n += 1                                 # the first interval after a reset holds one-time costs (renderers, writers)
                    if mean_ms > 24 and not slow_warned and len(arr) >= 25 and pace_n >= 2:
                        slow_warned = True
                        print(f"[HoloMotion v1.4] the loop takes {mean_ms:.0f} ms per 20 ms control tick: the robot runs at "
                              f"{20 / mean_ms:.2f}x real time and feels sluggish. Lighter: --lean-record (images and tactile "
                              f"pads via the replay's --fill), --no-fpv-window, --wrist-hz 15, or --no-wrist-cams.", flush=True)
                    tick_ms.clear()
                print(f"[v1.4 t={now - t_start:6.1f}s] main={agent.main_state:15s} policy={agent.policy_mode:15s} "
                      f"cmd=({st.vx:+.2f},{st.vy:+.2f},{st.vyaw:+.2f}) ref={agent.sub.n_obs:6d} "
                      f"pelvis=({base[0]:+.2f},{base[1]:+.2f},{base[2]:.2f}) yaw={yaw:+6.1f} "
                      f"run={agent._last.get('run_ms', 0):5.1f}ms{pace} gates=[{gates}]", flush=True)
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
        print("[HoloMotion v1.4] control loop stopped by an exception:\n" + traceback.format_exc(), flush=True)   # stderr is not always visible in the logs (2026-10-26)
    finally:
        if exporter is not None and rec_state == RecordingState.RECORDING:
            try:
                ep_idx = exporter.episode_buffer["episode_index"]
                extra = {"scene_setup": setup, "init_pose": episode_init_pose}
                if slice_cut is not None:
                    extra["slice"] = slice_cut.meta(int(exporter.episode_buffer["size"]))
                with _SaveGuard("the in-progress episode", exporter.episode_buffer["size"], reraise=False):
                    if rf_sampler is not None:
                        RF.finish_episode(exporter.episode_buffer, rf_sampler)
                        extra["realformat"] = RF.episode_meta(dict(backpack_kg=backpack_kg, head_tilt_deg=head_tilt_deg, **episode_success()), setup,
                                                              xlog.model_sha if xlog is not None and xlog.recording else "")
                    exporter.save_episode()
                    if xlog is not None and xlog.recording:
                        log_path = Path(run_save_dir) / "replay" / f"episode_{ep_idx:06d}.npz"
                        xlog.save(log_path, setup, dict(run_meta, episode_index=int(ep_idx), init_pose=episode_init_pose,
                                                        environment_config=_env_config_json(task), holomotion_start=start_summary),
                                  save_model=save_model, extra=NS.pack(start_state))
                        extra["exact_log"] = str(log_path.relative_to(run_save_dir))
                    if start_summary is not None:
                        extra["holomotion_start"] = start_summary
                    _save_episode_meta(exporter, ep_idx, extra)
                print("[Record] in-progress episode saved on exit")
            except Exception as exc:  # noqa: BLE001
                import traceback
                print(f"[Record] could not save the in-progress episode: {exc}\n{traceback.format_exc()}", flush=True)
        if fpv is not None:
            fpv.close()
        agent.close()
        env.close()
        os._exit(0)


def cli():
    # an exception before the control loop (which has its own try / os._exit) used to end in a segfault at interpreter
    # exit (renderer / ORT threads torn down out of order), hiding the real error under it (2026-10-18): print it, exit
    # cleanly with status 1
    try:
        typer.run(main)
    except SystemExit as e:                         # click's usage / BadParameter errors and --help end here too
        sys.stderr.flush(); sys.stdout.flush()
        os._exit(e.code if isinstance(e.code, int) else (0 if e.code is None else 1))
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush(); sys.stdout.flush()
        os._exit(1)


if __name__ == "__main__":
    cli()
