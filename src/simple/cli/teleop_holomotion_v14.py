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

PICO buttons (the robot's v1.4.1 map):
  left stick click   stand up (MOVE_TO_DEFAULT, 3 s to the default pose)
  right A            policy on, walking mode (also lowers the sim safety band)
  left stick / right stick x   walk / turn
  right B            motion tracking (follows your body); Y or A: back to walking
  grips              close the Dex3 hands
  X                  zero torque        right stick click   emergency stop
Sim only (hold the left menu button; nothing reaches the robot controller while it is held):
  left menu + A      start recording / save the episode
  left menu + B      abandon the episode
  left menu + Y      skip to a new random setup
  both triggers      reset the scene (same setup: a retry)

Every episode starts from a new random setup (robot start, target, bin / cart, distractors; --no-randomize for the
nominal scene), drawn from its own seed. With --record each saved episode also gets a bit-exact log
(replay/episode_XXXXXX.npz next to the dataset): the start state, every physics input and the setup, so
`python -m simple.cli.replay_holomotion_v14` replays it to the last bit.

--quick-start (faster collection): after a save (and after left menu + Y / both triggers) the new scene starts with
the robot standing on the floor, no leash, walking policy then motion tracking switched on for you, and recording
started as soon as motion tracking is on. The first scene at launch is as usual (or press left menu + Y).

Headset view: --stream-view single (default) sends the left head camera once, centred, with a status overlay: one
picture in the XRoboToolkit app's default view (the app's right-B view toggle would show only its left half -- press B
again). mono puts it in both halves (for the app's single view), stereo sends left | right.

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
        self._last = {"save": False, "abort": False, "skip": False, "reset": False}

    def poll(self) -> dict:
        s = self.agent.pico() or {}
        menu = bool(s.get("left_menu"))
        now = {"save": menu and bool(s.get("A")), "abort": menu and bool(s.get("B")), "skip": menu and bool(s.get("Y")),
               "reset": s.get("left_trigger", 0.0) > 0.9 and s.get("right_trigger", 0.0) > 0.9}
        edges = {k: v and not self._last[k] for k, v in now.items()}
        self._last = now
        return edges


def _load_sonic_config() -> dict:
    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config


def _init_exporter(save_dir, task_prompt, robot_model, obj_names, joint_names, ego_view_shape):
    """The decoupled-WBC LeRobot schema (as the psi0 datasets) plus the v1.4 controller's own signals."""
    from decoupled_wbc.data.exporter import Gr00tDataExporter
    from decoupled_wbc.data.utils import get_dataset_features, get_modality_config

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
    return Gr00tDataExporter.create(save_root=save_dir, fps=50, features=features,
                                    modality_config=get_modality_config(robot_model), task=task_prompt)


def _env_config_json(task) -> str:
    import json
    from simple.utils import NumpyArrayEncoder
    return json.dumps(task.state_dict(), cls=NumpyArrayEncoder)


def _build_frame(robot_model, obj_names, observation, privileged_info, action):
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
    if obj_names:                                              # fixed slots; an absent distractor is NaN
        nan = np.full(7, np.nan)
        frame["observation.object_poses"] = np.concatenate(
            [np.asarray(privileged_info[n], dtype=np.float64) if n in privileged_info else nan for n in obj_names])
    return frame


def _stand_on_floor(sonic_env, agent) -> None:
    """Quick start: the robot at the policy's default pose, feet on the floor, at rest, no leash (x, y and yaw as the
    setup spawned it)."""
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
    d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    d.qpos[m.jnt_qposadr[robot.body_joint_index]] = agent.default_real[agent.real_to_mjcf]
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


def _stream_status(agent, rec_state, rec_frames: int, episodes_saved: int, num_episodes: int, setup: dict,
                   privileged_info, quick: bool) -> list:
    """Overlay lines for the headset stream: (text, BGR)."""
    red, white, grey, yellow = (60, 60, 255), (255, 255, 255), (190, 190, 190), (60, 220, 255)
    if rec_state == RecordingState.RECORDING:
        top = (f"REC {rec_frames / 50.0:5.1f} s", red)
    elif agent.quick_active:
        top = ("starting ...", yellow)
    else:
        top = ("not recording  (L-menu + A: record)", grey)
    mode = {"WAIT": "WAIT (L3: stand)", "MOVE_TO_DEFAULT": "STAND (A: walk)", "ZERO_TORQUE": "ZERO TORQUE",
            "EMERGENCY_STOP": "E-STOP"}.get(agent.main_state)
    if mode is None:
        mode = "MOTION" if agent.node.current_policy_mode == "motion" else "WALK (B: motion)"
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
    max_distractors: Annotated[int, typer.Option(help="distractors per setup: 0..this")] = 3,
    exact_log: Annotated[bool, typer.Option(help="with --record: save the bit-exact replay log of every episode")] = True,
    save_model: Annotated[bool, typer.Option(help="with the exact log: also save the compiled MuJoCo model (.mjb, ~125 MB: meshes + collision trees)")] = False,
    backpack_kg: Annotated[float, typer.Option(help="add HoloMotion's G1 backpack body with this mass (kg; 0 = none, 3.2 = the v1.4.1 backpack policy)")] = 0.0,
    motion_model_bundle: Annotated[str, typer.Option(help="backpack motion model bundle dir (config.yaml + model_22000.onnx)")] = "",
    quick_start: Annotated[bool, typer.Option(help="after a save / new setup / retry: no leash, standing, motion tracking and recording start by themselves")] = False,
    stream_view: Annotated[str, typer.Option(help="headset view: single (one picture), mono (left camera in both halves) or stereo")] = "single",
    pico_stream: Annotated[bool, typer.Option(help="stream the ego camera to the headset (port 13579)")] = True,
    max_seconds: Annotated[float, typer.Option(help="stop after this long (0 = run until Ctrl-C); for tests")] = 0.0,
    log_every: Annotated[float, typer.Option(help="seconds between status lines")] = 2.0,
):
    target, mod = None, None
    if not env_id:
        mod = load_scene(scene, None if start_distance < 0 else start_distance)
        env_id, target = mod.ENV_ID, getattr(mod, "TARGET", None)
    else:
        scene = ""
    if motion_model_bundle:
        from simple.teleop.holomotion_v14.robot_variants import backpack_models_dir
        os.environ["HOLOMOTION_UNITREE_POLICY_MODELS_DIR"] = str(backpack_models_dir(motion_model_bundle))
        print(f"[HoloMotion v1.4] backpack motion model: {os.environ['HOLOMOTION_UNITREE_POLICY_MODELS_DIR']}")
    sonic_config = _load_sonic_config()
    render_hz = 50
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id, sim_mode="mujoco", render_hz=render_hz, physics_dt=sonic_config["SIMULATE_DT"],
                   headless=headless, max_episode_steps=max_episode_steps, sonic_config=sonic_config, target=target,
                   dr_level=dr_level, success_criteria=0.9)
    sonic_env: SonicLocoManipEnv = env.unwrapped  # type: ignore
    task, robot = sonic_env.task, sonic_env.task.robot
    assert isinstance(robot, G1Sonic)
    if backpack_kg > 0:
        from simple.teleop.holomotion_v14.robot_variants import backpack_mjcf
        robot.mjcf_path = backpack_mjcf(backpack_kg)
        print(f"[HoloMotion v1.4] G1 with a {backpack_kg:g} kg backpack: {robot.mjcf_path}")
    control_decimal = int(round(1 / sonic_config["SIMULATE_DT"] / render_hz))
    assert control_decimal == 4, "the v1.4 policy runs at 50 Hz (4x decimation at 200 Hz physics)"
    control_dt = control_decimal * robot.sim_dt

    if stream_view not in ("single", "mono", "stereo"):
        raise typer.BadParameter("--stream-view is single, mono or stereo")
    agent = HoloMotionV14Agent(robot, reference_uri=reference_uri, auto_stand=auto_stand, enable_pico_stream=pico_stream,
                               stream_view=stream_view)
    agent.num_episodes = num_episodes
    controls = SimControls(agent)

    setups = SceneSetups(scene, mod, task, max_distractors=max_distractors) if (mod is not None and randomize) else None
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

    next_setup()
    observation, privileged_info = env.reset()
    agent.reset_policy()
    obj_names = list(sonic_env.mujoco.mj_objects.keys())
    if setups is not None:
        # the LeRobot schema is fixed, the distractor count is not: slots for the task objects + max_distractors
        obj_names = [n for n in obj_names if not n.startswith("distractor_")] + [f"distractor_{i}" for i in range(max_distractors)]
    exporter = robot_model = None
    xlog = None
    if record:
        from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
        robot_model = instantiate_g1_robot_model(waist_location="lower_body")
        run_save_dir = f"{os.path.abspath(save_dir)}/{env_id}/level-{dr_level}"
        exporter = _init_exporter(run_save_dir, task.instruction, robot_model, obj_names, robot.joint_names,
                                  observation["head_stereo_left"].shape)
        print(f"[Record] saving to {run_save_dir}")
        if exact_log:
            xlog = ExactLog(sonic_env)
            prefix = SCENE_ENV_PREFIX.get(scene, "\0")
            knobs = {k: v for k, v in os.environ.items() if k.startswith(prefix) or k.startswith("SIMPLE_")}
            run_meta = dict(scene=scene, scene_root=str(scene_root()), env_id=env_id, dr_level=dr_level, env_knobs=knobs,
                            backpack_kg=backpack_kg,
                            robot_mjcf=robot.mjcf_path, physics_dt=sonic_config["SIMULATE_DT"], render_hz=render_hz,
                            policy_models_dir=os.environ.get("HOLOMOTION_UNITREE_POLICY_MODELS_DIR", ""),
                            velocity_onnx=str(agent.node.velocity_onnx_path), motion_onnx=str(agent.node.motion_onnx_path))

    rec_state = RecordingState.WAITING_FOR_LANDING
    episodes_saved, sim_cnt, pbar = 0, 0, None
    t_start = time.monotonic()
    t_log = t_start
    print(f"[HoloMotion v1.4] waiting for the reference publisher at {reference_uri}")
    print("  left stick click = stand, right A = walk, right B = motion tracking; left menu + A = record/save")

    def reset_scene(reason: str, new_setup: bool):
        nonlocal observation, privileged_info, rec_state, sim_cnt, pbar
        if exporter is not None and rec_state == RecordingState.RECORDING:
            exporter.skip_and_start_new_episode()
            print(f"[Record] {reason}: in-progress episode discarded")
        if xlog is not None:
            xlog.discard()
        if pbar is not None:
            pbar.close()
            pbar = None
        next_setup() if new_setup else same_setup()
        observation, privileged_info = env.reset()
        agent.reset_policy()
        if quick_start:
            _stand_on_floor(sonic_env, agent)
            observation, privileged_info = sonic_env._get_obs(), sonic_env._get_info()
            agent.quick_start()
        rec_state, sim_cnt = RecordingState.WAITING_FOR_LANDING, 0
        print(f"[HoloMotion v1.4] scene reset ({reason}{'; quick start' if quick_start else ''})")

    try:
        while not max_seconds or time.monotonic() - t_start < max_seconds:
            step_start = time.monotonic()
            action = agent.get_action(observation, instruction=task.instruction, privileged_info=privileged_info)
            frame_inputs = {"observation": observation, "action": action, "privileged_info": privileged_info}
            if xlog is not None:
                xlog.before_step()
            observation, reward, terminated, truncated, privileged_info = env.step(action)
            if not headless:
                sonic_env.update_viewer()
            ev = controls.poll()
            auto_record = agent.quick_ready and quick_start

            if ev["reset"]:
                reset_scene("both triggers", new_setup=False)
                continue
            if ev["skip"]:
                reset_scene("left menu + Y", new_setup=True)
                continue

            if exporter is not None:
                if ev["abort"] and rec_state == RecordingState.RECORDING:
                    exporter.skip_and_start_new_episode()
                    if xlog is not None:
                        xlog.discard()
                    rec_state = RecordingState.WAITING_FOR_LANDING
                    if pbar is not None:
                        pbar.close()
                        pbar = None
                    print("[Record] episode abandoned (left menu + B)")
                elif (ev["save"] or auto_record) and rec_state == RecordingState.WAITING_FOR_LANDING:
                    rec_state = RecordingState.RECORDING
                    if xlog is not None:
                        xlog.start()
                    pbar = tqdm(desc=f"Recording episode {episodes_saved + 1}", unit="frame", leave=False)
                    print(f"[Record] recording{' (auto, quick start)' if auto_record else ''} "
                          "(left menu + A to save, left menu + B to abandon)")
                elif ev["save"] and rec_state == RecordingState.RECORDING:
                    rec_state = RecordingState.EPISODE_DONE
                if rec_state == RecordingState.RECORDING:
                    exporter.add_frame(_build_frame(robot_model, obj_names, **frame_inputs))
                    if xlog is not None:
                        xlog.add_frame(frame_inputs["action"])
                    if pbar is not None:
                        pbar.update(1)
                    if terminated or truncated:
                        rec_state = RecordingState.EPISODE_DONE
                        print(f"[Record] task {'succeeded' if terminated else 'timed out'}; saving")
                if rec_state == RecordingState.EPISODE_DONE:
                    ep_idx = exporter.episode_buffer["episode_index"]
                    exporter.save_episode()
                    _save_episode_env_config(exporter, task, ep_idx)
                    extra = {"scene_setup": setup}
                    if xlog is not None:
                        log_path = Path(run_save_dir) / "replay" / f"episode_{ep_idx:06d}.npz"
                        xlog.save(log_path, setup, dict(run_meta, episode_index=int(ep_idx),
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
                                                     privileged_info, quick_start)
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
                exporter.save_episode()
                extra = {"scene_setup": setup}
                if xlog is not None and xlog.recording:
                    log_path = Path(run_save_dir) / "replay" / f"episode_{ep_idx:06d}.npz"
                    xlog.save(log_path, setup, dict(run_meta, episode_index=int(ep_idx),
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
