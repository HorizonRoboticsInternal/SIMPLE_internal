"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Teleoperate a G1 SIMPLE task with the HoloMotion v1.4 motion-tracking policy
and (optionally) record LeRobot-format episodes.

Two processes are involved:

  Terminal 1 - reference publisher (vendored; needs its own Newton/Warp env):
      third_party/holomotion/run_publisher.sh
  or, without a headset, replay a recording from the SIMPLE venv:
      holomotion-replay --hand-demo

  Terminal 2 (SIMPLE venv):
      teleop-holomotion simple/G1WholebodyLocomotionPickBetweenTablesHoloMotionTeleop-v0 --record

PICO controls:
  right axis click           drop robot / release elastic band
  left grip + right grip     reset environment
  left/right trigger         close Dex3 hands
  left menu + A              start recording / save the current episode
  left menu + X              abandon the current episode
"""

from __future__ import annotations

import os
import time
from typing import Annotated

import gymnasium as gym
import numpy as np
import typer
import tyro
from tqdm import tqdm

from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
from simple.agents.holomotion_pico_agent import HoloMotionPicoAgent
from simple.teleop.holomotion import DEFAULT_REFERENCE_URI, REF_QPOS_DIM
from simple.cli._decoupled_wbc_recording import (
    set_ego_view_feature_shape,
    validate_existing_ego_view_feature_shape,
)
from simple.cli.teleop_decoupled_wbc import RecordingState, _save_episode_env_config
from simple.envs.sonic_loco_manip import SonicLocoManipEnv
from simple.robots.g1_sonic import G1Sonic


class _ManualEpisodeControl:
    """Edge-detected PICO combos read from the HoloMotion reference stream.

    left_menu + A -> start recording, or save the episode already recording
    left_menu + X -> abandon the episode already recording
    """

    def __init__(self, agent: HoloMotionPicoAgent):
        self._agent = agent
        self._save_last = False
        self._abort_last = False

    def poll(self) -> tuple[bool, bool]:
        menu = self._agent.button("left_menu_button") > 0.5
        save_now = menu and self._agent.button("a_button") > 0.5
        abort_now = menu and self._agent.button("x_button") > 0.5
        save_edge = save_now and not self._save_last
        abort_edge = abort_now and not self._abort_last
        self._save_last, self._abort_last = save_now, abort_now
        return save_edge, abort_edge


def _load_sonic_config() -> dict:
    config = tyro.cli(SimLoopConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=[])
    sonic_config = config.load_wbc_yaml()
    sonic_config["ENV_NAME"] = "simple"
    return sonic_config


def _init_exporter(save_dir, task_prompt, robot_model, obj_names, joint_names, ego_view_shape, agent):
    """Gr00tDataExporter with the decoupled-WBC schema plus HoloMotion fields.

    Decoupled-only fields (eef state, navigate/base-height/torso commands)
    are kept zero-filled so existing post-processing keeps working.
    """
    from decoupled_wbc.data.exporter import Gr00tDataExporter
    from decoupled_wbc.data.utils import get_dataset_features, get_modality_config

    features = get_dataset_features(robot_model)
    set_ego_view_feature_shape(features, ego_view_shape)
    validate_existing_ego_view_feature_shape(save_dir, ego_view_shape)
    features["observation.state"]["names"] = joint_names
    modality_config = get_modality_config(robot_model)

    if obj_names:
        names = [f"{n}.{s}" for n in obj_names for s in ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z")]
        features["observation.object_poses"] = {"dtype": "float64", "shape": (len(obj_names) * 7,), "names": names}

    n_window = 1 + agent.n_fut
    features["teleop.reference_qpos"] = {"dtype": "float64", "shape": (REF_QPOS_DIM,), "names": None}
    features["teleop.reference_window"] = {"dtype": "float32", "shape": (n_window * REF_QPOS_DIM,), "names": None}
    features["teleop.hand_joints"] = {"dtype": "float64", "shape": (14,), "names": None}
    features["policy.raw_action"] = {"dtype": "float32", "shape": (29,), "names": list(agent.policy.joint_names)}
    features["policy.obs"] = {"dtype": "float32", "shape": (agent.policy.obs_dim,), "names": None}

    return Gr00tDataExporter.create(
        save_root=save_dir, fps=50, features=features, modality_config=modality_config, task=task_prompt,
    )


def _build_frame(agent: HoloMotionPicoAgent, robot_model, obj_names, observation, privileged_info, action):
    proprio = privileged_info["proprio"]
    hand = np.asarray(action["hand_joints"], dtype=np.float64)  # natural order (thumb/index/middle)
    # 43-DOF target in robot_model joint order: body 29 + left hand 7 + right hand 7
    action_q = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=np.asarray(action["target_q"], dtype=np.float64),
        left_hand_actuated_joint_values=hand[:7],
        right_hand_actuated_joint_values=hand[7:],
    )
    n_window = 1 + agent.n_fut
    policy_obs = action["policy_obs"]
    frame = {
        "observation.images.ego_view": observation["head_stereo_left"],
        "observation.state": np.asarray(observation["joint_qpos"], dtype=np.float64),
        "observation.eef_state": np.zeros(14, dtype=np.float64),
        "action": np.asarray(action_q, dtype=np.float64),
        "action.eef": np.zeros(14, dtype=np.float64),
        "observation.img_state_delta": np.array([0.0], dtype=np.float32),
        "teleop.navigate_command": np.zeros(4, dtype=np.float64),
        "teleop.base_height_command": np.zeros(1, dtype=np.float64),
        "observation.base_pose": np.asarray(proprio["floating_base_pose"], dtype=np.float64),
        "observation.base_vel": np.asarray(proprio["floating_base_vel"], dtype=np.float64),
        "observation.torso_rpy_command": np.zeros(3, dtype=np.float64),
        "teleop.reference_qpos": np.asarray(action["reference_qpos"], dtype=np.float64),
        "teleop.reference_window": np.asarray(action["reference_window"], dtype=np.float32).reshape(n_window * REF_QPOS_DIM),
        "teleop.hand_joints": hand,
        "policy.raw_action": np.asarray(action["raw_action"], dtype=np.float32),
        "policy.obs": (
            np.zeros(agent.policy.obs_dim, dtype=np.float32) if policy_obs is None
            else np.asarray(policy_obs, dtype=np.float32)
        ),
    }
    if obj_names:
        frame["observation.object_poses"] = np.concatenate(
            [privileged_info[name] for name in obj_names]
        ).astype(np.float64)
    return frame


def main(
    env_id: Annotated[str, typer.Argument()] = "simple/G1WholebodyLocomotionPickBetweenTablesHoloMotionTeleop-v0",
    target: str | None = None,
    onnx_path: Annotated[str, typer.Option(help="Motion-tracking ONNX; defaults to data/holomotion/models")] = "",
    reference_uri: Annotated[str, typer.Option(help="ZMQ endpoint of the reference publisher")] = DEFAULT_REFERENCE_URI,
    use_gpu: Annotated[bool, typer.Option(help="Use onnxruntime CUDA provider when available")] = True,
    sim_mode: Annotated[str, typer.Option()] = "mujoco",
    headless: Annotated[bool, typer.Option()] = False,
    max_episode_steps: Annotated[int, typer.Option()] = 30000,
    render_hz: Annotated[int, typer.Option()] = 50,
    save_dir: Annotated[str, typer.Option()] = "data/teleop_holomotion",
    num_episodes: Annotated[int, typer.Option()] = 100,
    dr_level: Annotated[int, typer.Option()] = 0,
    record: Annotated[bool, typer.Option()] = False,
    success_criteria: Annotated[float, typer.Option()] = 3,
    pico_stream: Annotated[bool, typer.Option(help="Stream the ego camera to the headset (port 13579)")] = True,
):
    assert sim_mode in ["mujoco"], f"Invalid sim_mode {sim_mode} for teleop."
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
        success_criteria=success_criteria,
    )
    sonic_env: SonicLocoManipEnv = env.unwrapped  # type: ignore
    task = sonic_env.task
    robot = task.robot
    assert sonic_env.spec is not None
    assert isinstance(robot, G1Sonic)
    assert int(round(1.0 / sonic_config["SIMULATE_DT"] / render_hz)) == 4, "HoloMotion policy expects 50 Hz control (4x decimation at 200 Hz)"

    agent = HoloMotionPicoAgent(
        robot, onnx_path=onnx_path, reference_uri=reference_uri, use_gpu=use_gpu, enable_pico_stream=pico_stream,
    )
    agent.num_episodes = num_episodes

    exporter = None
    manual_ctl = None
    robot_model = None
    rec_state = RecordingState.WAITING_FOR_LANDING
    episodes_saved = 0
    control_decimal = int(1 / sonic_config["SIMULATE_DT"] / render_hz)
    control_dt = control_decimal * robot.sim_dt
    run_save_dir = None

    def _on_episode_reset():
        # The reference queue / KV cache / yaw alignment must restart from a
        # clean state after every reset, recording or not.
        agent.reset_policy()
        if record and robot.elastic_band is not None:
            # Skip the hanging phase: PD-hold the default pose, then the policy
            # engages as soon as the reference window is filled.
            robot.elastic_band.enable = False

    step_pbar = None
    observation, privileged_info = env.reset()
    _on_episode_reset()
    obj_names = list(sonic_env.mujoco.mj_objects.keys())

    if record:
        from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

        robot_model = instantiate_g1_robot_model(waist_location="lower_body")
        run_save_dir = f"{os.path.abspath(save_dir)}/{sonic_env.spec.id}/level-{dr_level}"
        manual_ctl = _ManualEpisodeControl(agent)
        exporter = _init_exporter(
            run_save_dir, task.instruction, robot_model, obj_names, robot.joint_names,
            observation["head_stereo_left"].shape, agent,
        )
        print(f"\n[Record] Exporter initialized, saving to {run_save_dir}")
        print(f"[Record] Recording {len(obj_names)} objects: {obj_names}")

    print("[HoloMotion] Waiting for reference stream at", reference_uri)
    sim_cnt = 0
    try:
        while True:
            step_start = time.monotonic()
            action = agent.get_action(observation, instruction=task.instruction, privileged_info=privileged_info)
            data_frame = {
                "observation": observation,
                "action": action,
                "privileged_info": privileged_info,
            }
            observation, reward, terminated, truncated, privileged_info = env.step(action)

            if agent.reset_requested:
                if exporter is not None and rec_state == RecordingState.RECORDING:
                    print("[Record] Reset requested, discarding in-progress episode")
                    exporter.skip_and_start_new_episode()
                    if step_pbar is not None:
                        step_pbar.close()
                        step_pbar = None
                observation, privileged_info = env.reset()
                _on_episode_reset()
                obj_names = list(sonic_env.mujoco.mj_objects.keys())
                sim_cnt = 0
                rec_state = RecordingState.WAITING_FOR_LANDING
                print("[HoloMotion] Environment reset complete")

            if not headless:
                sonic_env.update_viewer()
            sonic_env.update_reward()
            agent.update_render_caches(observation)

            if exporter is not None:
                manual_save, manual_abort = manual_ctl.poll() if manual_ctl is not None else (False, False)

                if manual_abort and rec_state == RecordingState.RECORDING:
                    if step_pbar is not None:
                        step_pbar.close()
                        step_pbar = None
                    exporter.skip_and_start_new_episode()
                    rec_state = RecordingState.WAITING_FOR_LANDING
                    print("[Record] Episode abandoned (left_menu+X)")
                    manual_save = False
                elif manual_save and rec_state == RecordingState.RECORDING:
                    rec_state = RecordingState.EPISODE_DONE
                    print("[Record] Manual save requested (left_menu+A)")

                if rec_state == RecordingState.WAITING_FOR_LANDING:
                    band_done = not agent._dropping and (robot.elastic_band is None or not robot.elastic_band.enable)
                    auto_ready = band_done and agent.tracking_active
                    manual_ready = manual_save and agent.tracking_active
                    if auto_ready or manual_ready:
                        rec_state = RecordingState.RECORDING
                        sim_cnt = 0
                        step_pbar = tqdm(desc=f"Recording episode {episodes_saved + 1}", unit="frame",
                                         leave=False, position=1, bar_format="{desc} {n_fmt} {rate_fmt}")
                        print("[Record] Manual start (left_menu+A)" if manual_ready and not auto_ready
                              else "[Record] Motion tracking active, starting episode recording")

                if rec_state == RecordingState.RECORDING:
                    exporter.add_frame(_build_frame(agent, robot_model, obj_names, **data_frame))
                    if step_pbar is not None:
                        step_pbar.update(1)
                    if terminated or truncated:
                        rec_state = RecordingState.EPISODE_DONE

                if rec_state == RecordingState.EPISODE_DONE:
                    if step_pbar is not None:
                        step_pbar.close()
                        step_pbar = None
                    ep_idx = exporter.episode_buffer["episode_index"]
                    exporter.save_episode()
                    _save_episode_env_config(exporter, task, ep_idx)
                    episodes_saved += 1
                    agent.episodes_saved = episodes_saved
                    print(f"[Record] Episode {episodes_saved} saved")
                    if episodes_saved >= num_episodes:
                        print(f"[Record] Reached {num_episodes} episodes, stopping")
                        break
                    observation, privileged_info = env.reset()
                    _on_episode_reset()
                    obj_names = list(sonic_env.mujoco.mj_objects.keys())
                    sim_cnt = 0
                    rec_state = RecordingState.WAITING_FOR_LANDING
                    continue

            elapsed = time.monotonic() - step_start
            sleep_time = control_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sim_cnt % 50 == 0:
                print(f"[HoloMotion] control loop overran by {-sleep_time * 1000:.1f} ms")
            sim_cnt += 1
    except KeyboardInterrupt:
        print("Simulator interrupted by user.")
        if step_pbar is not None:
            step_pbar.close()
        if exporter is not None and rec_state == RecordingState.RECORDING:
            print("[Record] Saving in-progress episode before exit...")
            try:
                ep_idx = exporter.episode_buffer["episode_index"]
                exporter.save_episode()
                _save_episode_env_config(exporter, task, ep_idx)
                episodes_saved += 1
                print(f"[Record] Episode {episodes_saved} saved (interrupted)")
            except Exception as e:
                print(f"[Record] Failed to save interrupted episode: {e}")
    finally:
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
