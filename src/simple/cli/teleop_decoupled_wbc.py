"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Teleoperation using the decoupled whole-body control policy from decoupled_wbc.

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.
"""

from __future__ import annotations

import enum
import typer
from typing_extensions import Annotated, TYPE_CHECKING
import gymnasium as gym
import numpy as np
import simple.envs as _  # import all envs

if TYPE_CHECKING:
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv

from simple.agents.pico_decoupled_agent import PicoDecoupledAgent
from simple.cli._decoupled_wbc_recording import (
    set_ego_view_feature_shape,
    validate_existing_ego_view_feature_shape,
)
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from simple.robots.g1_sonic import G1Sonic

import json
import os
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

import time
import tyro
from datetime import datetime
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Episode recording state machine
# ---------------------------------------------------------------------------

class RecordingState(enum.Enum):
    WAITING_FOR_LANDING = "waiting_for_landing"
    RECORDING = "recording"
    EPISODE_DONE = "episode_done"


LIFT_THRESHOLD = 0.10  # metres above initial target z to end episode


class _ManualEpisodeControl:
    """Edge-detected PICO combos for manual episode control.

    left_menu + A  -> start recording, or save the episode already recording
    left_menu + X  -> abandon the episode already recording

    Bare A/X are left alone: A is the streamer's own toggle_data_collection and
    X drives base height, so both combos are gated on left_menu to disambiguate.
    Manual control runs *alongside* the automatic start/stop, it does not replace
    it -- whichever fires first wins.
    """

    def __init__(self, agent):
        self._xr = getattr(getattr(agent, "_pico_streamer", None), "xr_client", None)
        self._save_last = False
        self._abort_last = False

    def poll(self) -> tuple[bool, bool]:
        """Return (save_pressed, abort_pressed) as one-shot edges."""
        if self._xr is None:
            return False, False
        try:
            menu = bool(self._xr.get_button_state_by_name("left_menu_button"))
            save_now = menu and bool(self._xr.get_button_state_by_name("A"))
            abort_now = menu and bool(self._xr.get_button_state_by_name("X"))
        except Exception:
            return False, False
        save_edge = save_now and not self._save_last
        abort_edge = abort_now and not self._abort_last
        self._save_last, self._abort_last = save_now, abort_now
        return save_edge, abort_edge


def _save_episode_env_config(exporter, task, episode_index: int):
    """Write the task state_dict as environment_config into episodes.jsonl."""
    from simple.utils import NumpyArrayEncoder
    meta_file = exporter.root / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        return
    with open(meta_file, "r") as f:
        lines = [json.loads(line) for line in f]
    env_conf = task.state_dict()
    lines[episode_index]["environment_config"] = json.dumps(env_conf, cls=NumpyArrayEncoder)
    with open(meta_file, "w") as f:
        for entry in lines:
            f.write(json.dumps(entry) + "\n")


def _init_exporter(
    save_dir: str,
    task_prompt: str,
    robot_model,
    obj_names: list[str],
    joint_names: list[str],
    ego_view_shape=None,
):
    """Create a Gr00tDataExporter for LeRobot-format recording."""
    from decoupled_wbc.data.exporter import Gr00tDataExporter
    from decoupled_wbc.data.utils import get_dataset_features, get_modality_config

    features = get_dataset_features(robot_model)
    set_ego_view_feature_shape(features, ego_view_shape)
    validate_existing_ego_view_feature_shape(save_dir, ego_view_shape)
    features["observation.state"]["names"] = joint_names # state joint names
    modality_config = get_modality_config(robot_model)

    # Add object poses feature: each object has 7D (pos xyz + quat wxyz)
    num_objects = len(obj_names)
    if num_objects > 0:
        obj_names_flat = []
        for name in obj_names:
            for suffix in ["pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z"]:
                obj_names_flat.append(f"{name}.{suffix}")
        features["observation.object_poses"] = {
            "dtype": "float64",
            "shape": (num_objects * 7,),
            "names": obj_names_flat,
        }

    exporter = Gr00tDataExporter.create(
        save_root=save_dir,
        fps=50,
        features=features,
        modality_config=modality_config,
        task=task_prompt,
    )
    return exporter


def _build_frame(agent, obj_names: list[str], observation, privileged_info, action, target_waist):
    """Assemble one recording frame from current sim state and agent caches."""
    proprio = privileged_info["proprio"]
    rm = agent._dwbc_robot_model

    # action: 43-DOF target q in robot_model joint order
    # NOTE: joint matches with rm.joint_names
    action_q = rm.get_configuration_from_actuated_joints(
        body_actuated_joint_values=action["target_q"],
        left_hand_actuated_joint_values=action["left_hand_q"], # natural order: thumb/index/middle
        right_hand_actuated_joint_values=action["right_hand_q"],
    )

    mjcf_to_natural_order = lambda q: np.concatenate([q[:3], q[5:7], q[3:5]])
    proprio_q = rm.get_configuration_from_actuated_joints(
        body_actuated_joint_values=proprio["body_q"],
        left_hand_actuated_joint_values=mjcf_to_natural_order(proprio["left_hand_q"]),
        right_hand_actuated_joint_values=mjcf_to_natural_order(proprio["right_hand_q"]),
    )
    proprio_joints = dict(zip(rm.joint_names, proprio_q))
    from simple.robots.g1_sonic import WHOLE_BODY_JOINTS
    assert np.allclose(observation["joint_qpos"], np.array([proprio_joints[joint] for joint in WHOLE_BODY_JOINTS], dtype=np.float32))

    # override waist joints in action_q with target_waist from IK, to reflect the actual input to lower-body policy
    action_q[12:15] = target_waist  # waist joints are indices 12,13,14 in robot_model order
    frame = {
        "observation.images.ego_view": observation["head_stereo_left"],
        "observation.state": np.asarray(observation["joint_qpos"], dtype=np.float64), # obs_state
        "observation.eef_state": np.asarray(action["action_eef"], dtype=np.float64), # FIXME
        "action": np.asarray(action_q, dtype=np.float64),
        "action.eef": np.asarray(action["action_eef"], dtype=np.float64), # 1-cycle delayed teleop eef
        "observation.img_state_delta": np.array([0.0], dtype=np.float32), # FIXME
        "teleop.navigate_command": np.asarray(
            action["navigate_cmd"],
            dtype=np.float64
        ),
        "teleop.base_height_command": np.asarray(
            action["base_height_command"],
            dtype=np.float64
        ),
        "observation.base_pose": np.asarray(
            proprio["floating_base_pose"], 
            dtype=np.float64
        ),
        "observation.base_vel": np.asarray(
            proprio["floating_base_vel"],
            dtype=np.float64
        ),
        # Commanded torso orientation fed to the lower-body policy, synced with
        # navigate_cmd/base_height_command. Not derivable from action_q: its
        # waist slots are overridden with target_waist (IK input) just above.
        "observation.torso_rpy_command": np.asarray(
            action["torso_rpy_cmd"],
            dtype=np.float64
        ),
    }

    # Object poses: concatenate all object (pos + quat) in order
    if obj_names:
        obj_poses = []
        for name in obj_names:
            obj_poses.append(privileged_info[name])
        frame["observation.object_poses"] = np.concatenate(obj_poses).astype(np.float64)

    return frame


def _load_sonic_config() -> dict:
    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    # sonic_config["high_elbow_pose"] = True
    return sonic_config

def main(
    env_id: Annotated[str, typer.Argument()] = "simple/G1WholebodyBendPick-v1",
    target: str | None = None,
    sim_mode: Annotated[str, typer.Option()] = "mujoco",
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 30000,
    render_hz: Annotated[int, typer.Option()] = 50,
    data_format: Annotated[str, typer.Option()] = "lerobot",
    save_dir: Annotated[str, typer.Option()] = "data/teleop_decoupled_wbc",
    num_episodes: Annotated[int, typer.Option()] = 100,
    shard_size: Annotated[int, typer.Option()] = 100,
    dr_level: Annotated[int, typer.Option()] = 0,
    record: Annotated[bool, typer.Option()] = False,
    success_criteria: Annotated[float, typer.Option()] = 3,
):
    assert sim_mode in ["mujoco"], f"Invalid sim_mode {sim_mode} for teleop."
    sim_cnt = 0

    sonic_config = _load_sonic_config()

    print(f"Creating environment: {env_id}")
    env = gym.make(
        env_id,
        sim_mode=sim_mode,
        render_hz=render_hz,
        physics_dt=sonic_config["SIMULATE_DT"],
        headless=headless,
        max_episode_steps=max_episode_steps,
        sonic_config=sonic_config,
        target=target,
        dr_level=dr_level,
        success_criteria=success_criteria
    )
    sonic_env: SonicLocoManipEnv = env.unwrapped  # type: ignore
    task = sonic_env.task
    robot = task.robot
    assert sonic_env.spec is not None
    assert isinstance(robot, G1Sonic)

    agent = PicoDecoupledAgent(robot)
    agent.num_episodes = num_episodes

    # --- Recording setup ---
    exporter = None
    manual_ctl = None
    rec_state = RecordingState.WAITING_FOR_LANDING
    initial_target_z = None
    episodes_saved = 0
    control_decimal =  int(1/ sonic_config["SIMULATE_DT"]/render_hz)
    control_dt = control_decimal * robot.sim_dt  # = 0.02 s (50 Hz)

    def _on_episode_reset():
        """In recording mode, reset the WBC pipeline to a consistent initial pose,
        skip elastic band drop, and engage the RL policy immediately."""
        if not record:
            return
        if robot.elastic_band is not None:
            robot.elastic_band.enable = False
        agent._dropping = False
        # Reset the entire WBC pipeline (upper-body interpolation, teleop policy,
        # lower-body RL history) so the robot starts from the default pose.
        # The teleop policy is deactivated — upper body holds default pose
        # until the operator presses the Pico activation button.
        agent.reset_policy()
        # Engage the RL lower-body policy so the robot actively stabilizes
        agent._wbc_policy.lower_body_policy.use_policy_action = True
        # Print the spatialDR initial pose
        """ obs = robot.prepare_obs()
        print("\n=" * 60)
        print("[SpatialDR] Robot initial pose after reset", datetime.now().strftime("%-H:%M:%S"), sim_cnt)
        print("-" * 60)
        print(f"  base_pose (qpos[:7]): {obs['floating_base_pose']}")
        print("=" * 60)
        print("[Record] Episode reset: elastic band skipped, policy reset to initial pose")
        print("[Record] Upper body tracking PAUSED — align arms then press activation button") """

    # stabilized_printed = False
    step_pbar = None  # Progress bar for current recording episode

    # Initialize telemetry
    from decoupled_wbc.control.utils.telemetry import Telemetry
    telemetry = Telemetry(window_size=100)

    # FIXME DEBUG
    # env.unwrapped._telemetry = telemetry
    # env.unwrapped.mujoco._telemetry = telemetry

    observation, privileged_info = env.reset()
    _on_episode_reset()

    # Read obj_names after first reset so layout is populated by domain randomization.
    obj_names = list(sonic_env.mujoco.mj_objects.keys())

    if record:
        # timestamp = datetime.now().strftime("%m%d%H%M%S")
        run_save_dir = (
            f"{os.path.abspath(save_dir)}/{sonic_env.spec.id}/level-{dr_level}" #_{timestamp}
        )
        manual_ctl = _ManualEpisodeControl(agent)
        exporter = _init_exporter(
            run_save_dir,
            task.instruction, 
            agent._dwbc_robot_model, 
            obj_names,
            robot.joint_names,
            observation["head_stereo_left"].shape,
        )
        print(f"\n[Record] Exporter initialized, saving to {run_save_dir}")
        print(f"[Record] Ego view shape: {observation['head_stereo_left'].shape}")
        print(f"[Record] Recording {len(obj_names)} objects: {obj_names}")

    try:
        while True:
            step_start = time.monotonic()

            data_frame = {}

            with telemetry.timer("agent.get_action"):
                action = agent.get_action(observation, instruction=task.instruction, privileged_info=privileged_info)
            
            data_frame["target_waist"] = action["target_waist"] # for recording the actual input waist to lower-body policy
            data_frame["observation"] = observation.copy()
            data_frame["action"] = action.copy()
            data_frame["privileged_info"] = privileged_info.copy()

            with telemetry.timer("env.step"):
                # synced_proprio = info.copy()  # capture sim state before action for recording
                # for _ in range(int(control_dt / robot.sim_dt)):
                observation, reward, terminated, truncated, privileged_info = env.step(action)

            # if "proprio" in info:
            #     agent.publish_low_state(info["proprio"])

            if agent.reset_requested:
                # Discard any in-progress recording
                if exporter is not None and rec_state == RecordingState.RECORDING:
                    print("[Record] Reset requested, discarding in-progress episode")
                    exporter.skip_and_start_new_episode()
                    # Close progress bar for discarded episode
                    if step_pbar is not None:
                        step_pbar.close()
                        step_pbar = None

                observation, privileged_info = env.reset()
                # synced_proprio = info.copy()  # capture sim state after reset for recording
                _on_episode_reset()
                obj_names = list(sonic_env.mujoco.mj_objects.keys())
                sim_cnt = 0
                # stabilized_printed = False
                rec_state = RecordingState.WAITING_FOR_LANDING
                initial_target_z = None
                print("[TeleopDecoupled] Environment reset complete")

            with telemetry.timer("update_viewer"):
                if not headless:
                    sonic_env.update_viewer()

            with telemetry.timer("update_reward"):
                sonic_env.update_reward()

            with telemetry.timer("update_render"):
                agent.update_render_caches(observation)

            """ # --- Print once when robot first stabilizes ---
            if robot.stabilized and not stabilized_printed:
                stabilized_printed = True
                obs = robot.prepare_obs()
                print("=" * 60)
                print("[Stabilized] Robot pose (vel converged)", datetime.now().strftime("%-H:%M:%S"), sim_cnt)
                print("-" * 60)
                print(f"  base_pose (qpos[:7]): {obs['floating_base_pose']}")
                print(f"  max |qvel[0:6]|:      {np.max(np.abs(robot.mjData.qvel[0:6])):.4f}")
                print("=" * 60) """

            with telemetry.timer("recode_data"):
                # --- Recording logic (runs at 50Hz) ---
                if exporter is not None:
                    # Manual overrides sit alongside the automatic transitions
                    # below: left_menu+A starts/saves, left_menu+X abandons.
                    manual_save, manual_abort = (
                        manual_ctl.poll() if manual_ctl is not None else (False, False)
                    )

                    if manual_abort and rec_state == RecordingState.RECORDING:
                        if step_pbar is not None:
                            step_pbar.close()
                            step_pbar = None
                        exporter.skip_and_start_new_episode()
                        rec_state = RecordingState.WAITING_FOR_LANDING
                        initial_target_z = None
                        print("[Record] Episode abandoned (left_menu+X)")
                        manual_save = False

                    elif manual_save and rec_state == RecordingState.RECORDING:
                        # second press: close out and save this episode
                        rec_state = RecordingState.EPISODE_DONE
                        print("[Record] Manual save requested (left_menu+A)")

                    if rec_state == RecordingState.WAITING_FOR_LANDING:
                        # Check if robot has landed (elastic band done) AND
                        # teleop policy is active (operator has re-aligned and
                        # pressed the activation button)
                        elastic_done = (
                            not agent._dropping
                            and (robot.elastic_band is None or not robot.elastic_band.enable)
                        )
                        teleop_active = agent._teleop_policy.is_active
                        auto_ready = (
                            elastic_done
                            and teleop_active
                            and robot.stabilized
                            and agent._cached_target_q is not None
                        )
                        # manual start only needs a usable WBC target
                        manual_ready = manual_save and agent._cached_target_q is not None
                        if auto_ready or manual_ready:
                            rec_state = RecordingState.RECORDING
                            initial_target_z = None
                            sim_cnt = 0  # Reset sim counter for new episode
                            # Create progress bar for this recording episode
                            step_pbar = tqdm(desc=f"Recording episode {episodes_saved + 1}", unit="frame",
                                            leave=False, position=1, bar_format="{desc} {n_fmt} {rate_fmt}")
                            print(
                                "[Record] Manual start (left_menu+A)" if manual_ready and not auto_ready
                                else "[Record] Teleop active, starting episode recording"
                            )

                    if rec_state == RecordingState.RECORDING:
                        frame = _build_frame(agent, obj_names, **data_frame)
                        exporter.add_frame(frame)
                        if step_pbar is not None:
                            step_pbar.update(1)

                        # Track initial target height
                        if initial_target_z is None and "target" in privileged_info:
                            initial_target_z = privileged_info["target"][2]

                        # Check episode end condition: target lifted >= 10cm
                        # if initial_target_z is not None and "target" in privileged_info:
                        #     current_z = privileged_info["target"][2]
                        #     if current_z - initial_target_z >= LIFT_THRESHOLD:
                        #         rec_state = RecordingState.EPISODE_DONE

                        if terminated or truncated :
                            rec_state = RecordingState.EPISODE_DONE

                    if rec_state == RecordingState.EPISODE_DONE:
                        # Close progress bar for this episode
                        if step_pbar is not None:
                            step_pbar.close()
                            step_pbar = None

                        ep_idx = exporter.episode_buffer["episode_index"]
                        exporter.save_episode()
                        _save_episode_env_config(exporter, task, ep_idx)
                        episodes_saved += 1
                        agent.episodes_saved = episodes_saved
                        print(
                            f"[Record] Episode {episodes_saved} saved "
                            f"(target lifted {LIFT_THRESHOLD}m)"
                        )
                        if episodes_saved >= num_episodes:
                            print(f"[Record] Reached {num_episodes} episodes, stopping")
                            break
                        # Auto-reset for next episode
                        observation, privileged_info = env.reset()
                        _on_episode_reset()
                        obj_names = list(sonic_env.mujoco.mj_objects.keys())
                        sim_cnt = 0
                        # stabilized_printed = False
                        rec_state = RecordingState.WAITING_FOR_LANDING
                        initial_target_z = None
                        continue  # skip sleep / increment for this iteration

            elapsed = time.monotonic() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # Only show timing when loop is slow and verbose_timing is disabled
                telemetry.log_timing_info(context=f"{sim_cnt: >3}: Control loop overran by {-sleep_time:.3f} seconds", threshold=0.001)
                # ...

            sim_cnt += 1
    except KeyboardInterrupt:
        print("Simulator interrupted by user.")
        # Close progress bar if active
        if step_pbar is not None:
            step_pbar.close()
        if exporter is not None and rec_state == RecordingState.RECORDING:
            print("[Record] Saving in-progress episode before exit...")
            try:
                ep_idx = exporter.episode_buffer["episode_index"]
                exporter.save_episode()
                _save_episode_env_config(exporter, task, ep_idx)
                episodes_saved += 1
                agent.episodes_saved = episodes_saved
                print(f"[Record] Episode {episodes_saved} saved (interrupted)")
            except Exception as e:
                print(f"[Record] Failed to save interrupted episode: {e}")
    finally:
        # Ensure progress bar is closed
        if step_pbar is not None:
            step_pbar.close()
        if exporter is not None:
            exporter.stop_video_writers()
            print(f"[Record] Done. {episodes_saved} episodes saved to {run_save_dir}")
        env.close()
        agent.close()


def typer_main():
    typer.run(main)


if __name__ == "__main__":
    typer.run(main)
