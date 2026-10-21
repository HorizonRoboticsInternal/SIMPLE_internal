"""
SIMPLE: SIMulation-based Policy Learning and Evaluation

Copyright (c) 2025 Songlin Wei and Contributors
Licensed under the terms in LICENSE file.

Re-record a HoloMotion v1.4 teleop dataset by replaying its recorded HoloMotion inputs closed loop (2026-10-18).

For every episode of the source dataset (recorded with the frame-0 controller snapshot, i.e. 2026-10-16 or later):

    scene     rebuilt through SIMPLE from the episode's setup + environment_config (the replay tool's _build_env /
              _reset_to), the robot drawn as asked (default: black Dex3 hands; visual only, the compiled model differs
              from the recorded one in geom_rgba alone)
    start     the live scene start re-done (quick-start placement with the recorded start pose, so the task gates see
              the same start position), then the exact physics state of frame 0 (state0 + the rest state + the band)
    control   a fresh HoloMotion node restored to the frame-0 snapshot (node_state.restore); the agent in POLICY, fed
              per tick the recorded controller sample (holomotion.pico) and reference frames
              (holomotion.MotionReference.frames), on the recorded node clock (holomotion.tick_time)
    record    the teleop CLI's own recorder (same exporter, frame builder, raw HoloMotion columns, real-format columns,
              exact log), both eyes re-rendered at the source's image size

Frame 0's observation.object_poses is copied from the source: live it came from the env.step before frame 0 (its last
substep's kinematics, before integration), which the exact log does not hold. Every frame is checked against the source: physics state bit for bit (exact log), the policy's ONNX input/output
(holomotion.policy_obs / actions_onnx) bit for bit, and the task outcome (terminated on the last frame when the source
episode ended by success). Episodes that do not reproduce are reported (and still written, flagged in the summary).

    .venv/bin/python -m simple.cli.rerecord_holomotion_v14 <source level-3_pinhole dir> --save-dir <new root> [--episodes 0,3]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Annotated

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import typer


def _episodes(spec: str, n: int) -> list[int]:
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


class _ReplaySubscriber:
    """Stands in for the ZMQ ReferenceSubscriber: this tick's recorded reference frames and controller sample."""

    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.last_pico: dict | None = None
        self.n_obs = 0
        self.last_obs_time = None

    def set_tick(self, frames: list[dict], pico: dict | None) -> None:
        self.frames, self.last_pico = frames, pico
        self.n_obs += len(frames)

    def poll(self, max_msgs: int = 512) -> list[dict]:
        f, self.frames = self.frames, []
        return f

    def pico_age(self) -> float:
        return 0.0 if self.last_pico is not None else float("inf")

    def close(self) -> None:
        pass


def _pico_sample(v: np.ndarray) -> dict | None:
    """holomotion.pico (12 float32: sticks, grips, triggers, button bits, active flags, left menu) -> the sample dict the
    agent read live (wire.pico_sample's fields; the wire carries float32, so the stored values are exact)."""
    from simple.teleop.holomotion_v14.wire import BUTTONS
    if not np.isfinite(v[0]):
        return None
    bits = int(v[8])
    s = {name: int(bool(bits & bit)) for name, bit in BUTTONS.items()}
    f = lambda x: float(np.float32(x))  # noqa: E731
    s.update(button_bits=bits, left_axis=(f(v[0]), f(v[1])), right_axis=(f(v[2]), f(v[3])), left_grip=f(v[4]),
             right_grip=f(v[5]), left_trigger=f(v[6]), right_trigger=f(v[7]), l_active=int(v[9]), r_active=int(v[10]),
             received_monotonic=0.0, timestamp_ns=0)
    return s


def main(
    source: Annotated[Path, typer.Argument(help="the source LeRobot dataset dir (…/simple/<env>/level-3_pinhole) with replay/")],
    save_dir: Annotated[str, typer.Option(help="root of the new dataset (the same simple/<env>/<level> folders are made under it)")] = "",
    episodes: Annotated[str, typer.Option(help="episode indices, e.g. 0,3,5-9 (default: all)")] = "",
    black_hands: Annotated[bool, typer.Option(help="draw the Dex3 hands black in the new recording")] = True,
    stop_on_mismatch: Annotated[bool, typer.Option(help="stop at the first episode that does not reproduce")] = False,
    reference_source: Annotated[str, typer.Option(help="recorded | action | state: the joint channels (dof_pos, dof_vel) of every reference "
        "frame fed to the node. recorded = the operator's frames (bit-exact replay); action = the model's own joint targets (holomotion.action) "
        "interpolated at each frame's stamp, dof_vel by finite difference; state = the measured joints / velocities (holomotion.LowState.motor_state.q, dq). "
        "action_upper / state_upper: the same for waist + arms only (joints 12..28), legs stay the operator's. Root position / orientation and the frame timing stay the operator's (experiment 2026-10-19: what a VLA trained on actions or states would feed)")] = "recorded",
):
    import cv2
    import mujoco
    import pyarrow.parquet as pq
    from simple.cli import teleop_holomotion_v14 as T
    from simple.cli.replay_holomotion_v14 import _build_env, _reset_to
    from simple.cli.teleop_decoupled_wbc import _save_episode_env_config
    from simple.teleop.holomotion_v14 import exact_log as XL
    from simple.teleop.holomotion_v14 import node_state as NS
    from simple.teleop.holomotion_v14 import realformat as RF
    from simple.teleop.holomotion_v14.exact_log import ExactLog
    from simple.teleop.holomotion_v14.hbvcam_views import PinholeView

    src = source.resolve()
    rel = Path(*src.parts[-3:])                                       # simple/<env>/<level folder>
    if not save_dir:
        raise typer.BadParameter("--save-dir is required (a new folder; the source is never written)")
    out_dir = Path(save_dir).resolve() / rel
    if out_dir == src:
        raise typer.BadParameter("--save-dir must differ from the source dataset")
    src_info = json.loads((src / "meta" / "info.json").read_text())
    feats = src_info["features"]
    for k in ("holomotion.pico", "holomotion.MotionReference.frames", "holomotion.tick_time"):
        if k not in feats:
            raise typer.BadParameter(f"{src} has no {k} column: recorded before 2026-10-16/18, cannot be replayed from its inputs")
    ego_shape = tuple(feats["observation.images.ego_view_left"]["shape"])          # (h, w, 3)
    obj_names = sorted({n.rsplit(".", 1)[0] for n in feats["observation.object_poses"]["names"]},
                       key=lambda n: feats["observation.object_poses"]["names"].index(n + ".pos_x"))
    meta_lines = [json.loads(l) for l in (src / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    eps = _episodes(episodes, len(meta_lines))

    env = sonic_env = mod = agent = exporter = xlog = rf = view = clock = None
    robot_model = None
    summary = []
    t_start = time.monotonic()
    for k, ep in enumerate(eps):
        meta = meta_lines[ep]
        log = XL.load(src / "replay" / f"episode_{ep:06d}.npz")
        info, setup = dict(log["info"]), log["setup"]
        snap = NS.unpack(log)
        if snap is None:
            raise typer.BadParameter(f"episode {ep}: its exact log has no frame-0 controller snapshot (holomotion_start.*)")
        t = pq.read_table(str(src / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"))
        n = t.num_rows
        col = lambda c: np.stack(t[c].to_pandas().values).reshape(n, -1)  # noqa: E731
        pico, tick_time = col("holomotion.pico"), col("holomotion.tick_time")[:, 0]
        K = col("holomotion.MotionReference.frames").shape[1] // 65
        frames = col("holomotion.MotionReference.frames").reshape(n, K, 65)
        fmeta = col("holomotion.MotionReference.frames_meta").reshape(n, K, 2)
        cnt = col("holomotion.MotionReference.frames_this_tick")[:, 0].astype(int)
        if reference_source != "recorded":                 # swap the joint channels of every frame for the action / state sequence
            base_src = reference_source.replace("_upper", "")
            if base_src == "action":
                src_q = col("holomotion.action").astype(np.float64); src_dq = np.gradient(src_q, tick_time, axis=0)
            elif base_src == "state":
                src_q = col("holomotion.LowState.motor_state.q").astype(np.float64); src_dq = col("holomotion.LowState.motor_state.dq").astype(np.float64)
            else:
                raise typer.BadParameter(f"--reference-source {reference_source!r}: recorded | action | state | action_upper | state_upper")
            J = list(range(12, 29)) if reference_source.endswith("_upper") else list(range(29))   # _upper: waist + arms only, legs stay the operator's
            frames = frames.copy()
            for i in range(n):
                for j in range(min(int(cnt[i]), K)):
                    s_ = float(fmeta[i, j, 1])                  # the frame's stamp, on the tick_time clock
                    for k in J:
                        frames[i, j, k] = np.interp(s_, tick_time, src_q[:, k]); frames[i, j, 29 + k] = np.interp(s_, tick_time, src_dq[:, k])
            ref_q0, ref_dq0 = frames[0, 0, 0:29].astype(np.float32), frames[0, 0, 29:58].astype(np.float32)   # the first fed frame (legs = operator's for _upper)
        rec_obs, rec_dim = col("holomotion.policy_obs"), col("holomotion.policy_obs_dim")[:, 0].astype(int)
        rec_ran, rec_out = col("holomotion.policy_ran")[:, 0].astype(int), col("holomotion.actions_onnx")
        assert len(log["physics"]) == n, f"episode {ep}: {len(log['physics'])} logged frames, {n} rows"
        init_pose = meta.get("init_pose")
        init_pose = json.loads(init_pose) if isinstance(init_pose, str) else init_pose

        # ---------------------------------------------------------------- scene
        info_new = dict(info, black_hands=bool(black_hands))
        info_new.pop("robot_mjcf", None)          # else _build_env reloads the recorded file (the source's hand colour)
        if env is None:
            if info.get("policy_models_dir"):
                os.environ["HOLOMOTION_UNITREE_POLICY_MODELS_DIR"] = info["policy_models_dir"]
            env, sonic_env, mod = _build_env(info_new, setup, check_legacy=False)
            robot = sonic_env.task.robot
            from simple.agents.holomotion_v14_agent import HoloMotionV14Agent
            agent = HoloMotionV14Agent(robot, enable_pico_stream=False, button_map="sim", quiet=True)
            agent.sub.close()
            agent.sub = _ReplaySubscriber()
            from simple.teleop.holomotion_v14.sim_clock import ControllerClock
            clock = ControllerClock(t0=float(tick_time[0]))
            clock.install()
            from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
            robot_model = instantiate_g1_robot_model(waist_location="lower_body")
            T._check_dataset_dir(str(out_dir))
            pdim = max(int(s.get_inputs()[0].shape[-1]) for s in (agent.node.motion_policy_session, agent.node.velocity_policy_session))
            exporter = T._init_exporter(str(out_dir), sonic_env.task.instruction, robot_model, obj_names, robot.joint_names,
                                        ego_shape, True, right_eye=True, policy_obs_dim=pdim)
            rf = RF.LiveSampler(sonic_env)
            xlog = ExactLog(sonic_env)
            view = PinholeView(sonic_env)
            print(f"[rerecord] {src} -> {out_dir}: {len(eps)} episodes, black hands {black_hands}, reference source {reference_source}, robot {robot.mjcf_path}", flush=True)
        else:
            _reset_to(env, sonic_env, mod, info_new, setup)
        task, robot = sonic_env.task, sonic_env.task.robot
        eng = sonic_env.mujoco
        m, d = eng.mjModel, eng.mjData
        if k == 0:                                # the rebuilt robot is the asked variant
            want = "_blackhands" in robot.mjcf_path
            assert want == bool(black_hands), f"robot file {robot.mjcf_path} does not match --black-hands {black_hands}"

        # ---------------------------------------------------------------- start: the live scene start, then frame 0
        agent.reset_policy()
        T._stand_on_floor(sonic_env, agent, init_pose)                    # as the live quick start placed it
        task._start_xy = np.asarray(d.qpos[:2], dtype=float).copy()       # the gates' start position (set at the first step live)
        XL.set_state(m, d, log["state0"], XL.INTEGRATION)
        if len(log["rest_idx"]) and int(log["rest_idx"][0]) == 0:
            XL.set_state(m, d, log["rest_val"][0], XL.REST)
        mujoco.mj_forward(m, d)
        XL.restore_band(robot, log["action_band_state"][0])
        eng.render_step = int(log["render_step0"])
        NS.restore(agent.node, snap)
        if reference_source != "recorded":                 # the frames already queued at frame 0 (the robot stands still): the source's tick-0 value
            vr_ = agent.node._vr_reference; Jq = list(range(12, 29)) if reference_source.endswith("_upper") else list(range(29))
            vr_.dof_pos_queue[:, Jq] = ref_q0[Jq]; vr_.prev_dof_pos[Jq] = ref_q0[Jq]; vr_.dof_vel_queue[:, Jq] = ref_dq0[Jq]; vr_.prev_dof_vel[Jq] = ref_dq0[Jq]
        A = json.loads(str(snap["agent_json"])) if "agent_json" in snap else None
        if A is not None:                         # recorded since 2026-10-18: the agent's own state at frame 0
            agent.main_state, agent._last_bits, agent._dropping = A["main_state"], int(A["last_bits"]), bool(A["dropping"])
            agent._state_t0 = float(tick_time[0]) - float(A.get("state_age", 0.0))
            arr = lambda v: None if v is None else np.asarray(v, dtype=np.float32)  # noqa: E731
            agent._left_hand, agent._right_hand = arr(A["left_hand"]), arr(A["right_hand"])
            agent._hold_real, agent._mtd_from = arr(A["hold_real"]), arr(A["mtd_from"])
            agent._quick = None
            if A.get("quick_active"):
                print(f"[rerecord] ep {ep}: the quick start was still running at frame 0; not reproduced", flush=True)
        else:                                     # older logs: a --record-on-motion / quick-start frame 0 (POLICY, no buttons held)
            agent.main_state, agent._state_t0 = "POLICY", 0.0
            agent._quick, agent._dropping, agent._last_bits = None, False, 0
            agent._hold_real = agent._mtd_from = None
        observation, privileged_info = sonic_env._get_obs(), sonic_env._get_info()
        # frame 0's object poses: live they came out of the env.step before frame 0, i.e. from the kinematics of that
        # step's last substep (before its integration), which the exact log does not hold; the source's values are the
        # measurement of that very moment, so frame 0 takes them (every later frame is recomputed and compared)
        op0 = col("observation.object_poses")[0].reshape(-1, 7)
        privileged_info = dict(privileged_info)
        for name, pose in zip(obj_names, op0):
            if name in privileged_info and np.isfinite(pose).all():
                privileged_info[name] = np.asarray(pose, dtype=np.float64).copy()

        # ---------------------------------------------------------------- the episode, tick by tick
        phys_bad = obs_bad = 0
        first_bad = None
        term_at = None
        for i in range(n):
            clock.now = float(tick_time[i])
            kf = min(int(cnt[i]), K)
            agent.sub.set_tick([{"latest_obs": frames[i, j].astype(np.float32), "frame_index": np.array([int(fmeta[i, j, 0])]),
                                 "timestamp_ns": np.array([int(round(fmeta[i, j, 1] * 1e9))])} for j in range(kf)],
                               _pico_sample(pico[i]))
            left, right = view.render_pair()
            size = (ego_shape[1], ego_shape[0])
            left = cv2.resize(left, size, interpolation=cv2.INTER_AREA) if left.shape[:2] != ego_shape[:2] else left
            right = cv2.resize(right, size, interpolation=cv2.INTER_AREA) if right.shape[:2] != ego_shape[:2] else right
            action = agent.get_action(observation, instruction=task.instruction, privileged_info=privileged_info)
            a = getattr(action, "parameters", action)
            frame_inputs = {"observation": {**observation, "head_stereo_left": left, "head_stereo_right_rect": right},
                            "action": action, "privileged_info": privileged_info, "phys": rf.sample(action)}
            if rec_ran[i]:                                                 # the policy's ONNX input/output vs the source
                o = a.get("raw_policy_obs")
                same = (o is not None and len(o) == rec_dim[i] and np.array_equal(np.asarray(o, np.float32), rec_obs[i, :rec_dim[i]].astype(np.float32))
                        and np.array_equal(np.asarray(a.get("raw_policy_out"), np.float32)[:29], rec_out[i].astype(np.float32)))
                if not same:
                    obs_bad += 1
                    first_bad = i if first_bad is None else first_bad
            xlog.before_step()
            observation, reward, terminated, truncated, privileged_info = env.step(action)
            if i == 0:
                xlog.start()
            exporter.add_frame(T._build_frame(robot_model, obj_names, **frame_inputs, policy_obs_dim=pdim))
            rf.frame_recorded()
            xlog.add_frame(action)
            if XL.get_state(m, d, XL.PHYSICS).tobytes() != log["physics"][i].tobytes():
                phys_bad += 1
                first_bad = i if first_bad is None else first_bad
            if terminated and term_at is None:
                term_at = i

        # ---------------------------------------------------------------- save, as the teleop CLI does
        ep_idx = exporter.episode_buffer["episode_index"]
        extra = {"scene_setup": setup, "init_pose": init_pose}
        RF.finish_episode(exporter.episode_buffer, rf)
        run_meta = {kk: v for kk, v in info.items() if kk not in ("version", "mujoco", "model_sha256", "timestep", "frames", "calls", "sizes",
                                                                 "episode_index", "init_pose", "environment_config", "holomotion_start")}
        run_meta.update(black_hands=bool(black_hands), robot_mjcf=robot.mjcf_path, reference_source=reference_source,
                        rerecorded_from=dict(dataset=str(src), episode=int(ep), model_sha256=info.get("model_sha256")))
        extra["realformat"] = RF.episode_meta(run_meta, setup, xlog.model_sha)
        exporter.save_episode()
        T._dump_obs_layouts(agent, str(out_dir))
        _save_episode_env_config(exporter, task, ep_idx)
        regenerated_cfg = T._env_config_json(task)
        # the scene's environment_config is the task's state_dict at save time, object poses included: a manual save
        # (left menu + A) runs one more env.step before saving, so the source took it one tick after its last frame.
        # Keep the source's own, verbatim (the physics is restored from state0 either way)
        env_cfg = info["environment_config"]
        T._save_episode_meta(exporter, ep_idx, {"environment_config": env_cfg})
        start_summary = meta.get("holomotion_start")
        start_summary = json.loads(start_summary) if isinstance(start_summary, str) else start_summary
        log_path = out_dir / "replay" / f"episode_{ep_idx:06d}.npz"
        xlog.save(log_path, setup, dict(run_meta, episode_index=int(ep_idx), init_pose=init_pose, environment_config=env_cfg,
                                        holomotion_start=start_summary), save_model=False, extra=NS.pack(snap))
        extra["exact_log"] = str(log_path.relative_to(out_dir))
        extra["holomotion_start"] = start_summary
        extra["rerecorded_from"] = dict(dataset=str(src), episode=int(ep))
        T._save_episode_meta(exporter, ep_idx, extra)
        same_cfg = json.loads(regenerated_cfg) == json.loads(info["environment_config"])
        P = dict(getattr(task, "_progress", {}) or {})
        gates = {g: bool(P.get(g)) for g in ("grasped", "at_bin", "placed") if g in P}
        manual_save = bool(pico[-1, 11] > 0)               # left menu held on the last frame: saved by hand next tick
        ok = phys_bad == 0 and obs_bad == 0
        row = dict(source_episode=int(ep), new_episode=int(ep_idx), frames=int(n), physics_bitexact=phys_bad == 0, physics_bad_frames=phys_bad,
                   policy_io_bitexact=obs_bad == 0, policy_io_bad_ticks=obs_bad, first_bad_frame=first_bad, success_frame=term_at,
                   success_on_last_frame=term_at == n - 1, task_gates_at_end=gates, source_saved_by_hand=manual_save,
                   regenerated_environment_config_same=same_cfg, environment_config="copied from the source",
                   note="frame 0 observation.object_poses copied from the source (see the module docstring)")
        summary.append(row)
        el = time.monotonic() - t_start
        print(f"[rerecord] ep {ep} -> {ep_idx}: {n} frames, physics {'BIT-EXACT' if phys_bad == 0 else f'{phys_bad} frames differ'}, "
              f"policy I/O {'BIT-EXACT' if obs_bad == 0 else f'{obs_bad} ticks differ'}, "
              f"task {'SUCCESS at the last frame' if term_at == n - 1 else ('success at frame %s' % term_at if term_at is not None else 'no success')}"
              f" gates {gates}{', source saved by hand (left menu + A)' if manual_save else ''} ({el / (k + 1):.0f} s/episode)", flush=True)
        (out_dir / "meta" / "rerecord_summary.json").write_text(json.dumps(dict(source=str(src), black_hands=bool(black_hands), reference_source=reference_source, episodes=summary), indent=1))
        if stop_on_mismatch and not ok:
            break

    n_ok = sum(r["physics_bitexact"] and r["policy_io_bitexact"] for r in summary)
    n_succ = sum(r["success_on_last_frame"] for r in summary)
    print(f"[rerecord] done: {n_ok}/{len(summary)} bit-exact, {n_succ}/{len(summary)} succeed on their last frame -> {out_dir}", flush=True)
    sys.stdout.flush()
    os._exit(0)


def cli():
    try:
        typer.run(main)
    except SystemExit as e:
        sys.stderr.flush(); sys.stdout.flush()
        os._exit(e.code if isinstance(e.code, int) else (0 if e.code is None else 1))
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stderr.flush(); sys.stdout.flush()
        os._exit(1)


if __name__ == "__main__":
    cli()
