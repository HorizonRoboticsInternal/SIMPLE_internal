#!/usr/bin/env python3
"""Replay psi0-format SIMPLE episodes in simulation, optionally with two wrist cameras.

Each psi0 episode stores its full scene (meta/episodes.jsonl -> environment_config: DR state + layout)
and the 36-D psi0 action per 20 ms frame. This tool rebuilds the scene, feeds the recorded actions
through the SAME adapter the psi0 evaluation uses (Psi0DecoupledWbcAgent for the *Teleop tasks,
Psi0Agent for the *MP tasks) with a local client that returns the recorded actions instead of a
model server, and saves per episode:

    <out>/ep_NNNNNN/wrist_left.mp4, wrist_right.mp4   the new wrist views (if --wrist-cams)
    <out>/ep_NNNNNN/egocentric.mp4                    the head view re-rendered in the replay
    <out>/ep_NNNNNN/track.npz                         replayed joints / base pose / object poses per frame
    <out>/ep_NNNNNN/summary.json                      task success + tracking error vs the recording

Then `update_dataset.py` writes a new psi0 dataset with the wrist videos added.

Run with the SIMPLE venv from the repo root, e.g.
    MUJOCO_GL=egl .venv/bin/python scripts/psi0_replay/replay_psi0.py \
        --data-dir /mnt/nas28/kerou.zhang/datasets/SIMPLE_data/simple-0721/simple/G1WholebodyXMovePickTeleop-v0 \
        --episodes 0 --sim-mode mujoco --wrist-cams --out data/psi0_replay/G1WholebodyXMovePickTeleop-v0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SIMPLE_DETERMINISTIC_RUNTIME", "1")
os.environ.setdefault("SIMPLE_EVAL_SEED", "0")
os.environ["_TYPER_STANDARD_TRACEBACK"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import wrist_cams  # noqa: E402


# ----------------------------------------------------------------------------------------------
# psi0 dataset access
# ----------------------------------------------------------------------------------------------
def read_info(root: Path) -> dict:
    return json.loads((root / "meta" / "info.json").read_text())


def read_episodes(root: Path) -> list[dict]:
    with (root / "meta" / "episodes.jsonl").open() as f:
        return [json.loads(line) for line in f if line.strip()]


def episode_frames(root: Path, info: dict, ep: int):
    import pyarrow.parquet as pq

    chunk = ep // int(info.get("chunks_size", 1000))
    path = root / info["data_path"].format(episode_chunk=chunk, episode_index=ep)
    return pq.read_table(path).to_pandas()


def env_config_of(row: dict) -> dict:
    conf = row["environment_config"]
    conf = json.loads(conf) if isinstance(conf, str) else conf
    # same fix-up as simple.datasets.lerobot.get_episode_lerobot
    scene = conf["dr_state_dict"]["scene"]
    scene["uid"] = scene["uid"].replace("102344280", "scene3")
    restore_material_from_layout(conf)
    restore_lighting_from_layout(conf)
    return conf


def restore_lighting_from_layout(conf: dict) -> None:
    """XMoveBendPick stores an EMPTY lighting state ({}) in every episode; loaded as-is it gives zero
    lights and black Isaac frames. The stored layout lists the lights that were drawn: use them."""
    dr = conf["dr_state_dict"]
    if dr.get("lighting"):
        return
    lights = (conf.get("layout") or {}).get("lights") or []
    need = ("uid", "type", "pose", "light_radius", "light_length", "light_intensity",
            "light_color_temperature", "center_light_postion", "center_light_orientation")
    if lights and all(all(k in l for k in need) for l in lights):
        dr["lighting"] = {l["uid"]: l for l in lights}
        conf["_lighting_restored_from_layout"] = True


def restore_material_from_layout(conf: dict) -> None:
    """Some psi0 sets (XMovePick, XMoveBendPick) store `material: null` in the DR state, so a replay
    would draw a new table material. Their stored layout still holds what Isaac drew: the table's
    vMaterials file and every object's shader values. Rebuild the material state from it (Isaac does
    not use the ground entry; the robot shader is not stored, so it gets the fixed-mode default)."""
    dr = conf["dr_state_dict"]
    if dr.get("material") is not None:
        return
    actors = (conf.get("layout") or {}).get("actors") or {}
    table = (actors.get("table") or {}).get("material")
    if not table:
        return
    objs = [a["material"] for k, a in actors.items()
            if k not in ("table", "table2", "robot") and isinstance(a, dict) and "asset" in a and "material" in a]
    dr["material"] = {
        "table_material": table,
        "ground_material": None,
        "robot_shader_params": {"reflection_roughness_constant": 0.5, "metallic_constant": 0.0, "specular_level": 0.0},
        "object_shader_params": objs,
    }
    conf["_material_restored_from_layout"] = True


class ReplayClient:
    """Stands in for HttpActionClient: each query returns the next block of recorded actions.

    The psi0 adapters queue every returned row and pop one per 20 ms step; when the queue is empty
    they query again. After the last block the query raises StopIteration, which ends the episode
    the way the eval loop expects.
    """

    def __init__(self, *blocks: np.ndarray):
        self.blocks = [np.asarray(b, dtype=np.float32) for b in blocks if len(b)]
        self.calls = 0

    def query_action(self, *args, **kwargs):
        self.calls += 1
        if not self.blocks:
            raise StopIteration("recorded actions exhausted")
        return (self.blocks.pop(0),)


# ----------------------------------------------------------------------------------------------
# environment + agent (mirrors src/simple/cli/eval_decoupled_wbc.py and eval.py)
# ----------------------------------------------------------------------------------------------
def build(env_id: str, sim_mode: str, render_hz: int, wrist: wrist_cams.WristCamSpec | None):
    import gymnasium as gym
    import simple.envs as _  # noqa: F401  (registers the envs)
    from simple.tasks.registry import TaskRegistry

    task_name = gym.spec(env_id).kwargs["task"]
    if wrist is not None:
        wrist_cams.add_to_task_class(TaskRegistry._registry[task_name], wrist)

    teleop = env_id.endswith("Teleop-v0")
    if teleop:
        from simple.cli.eval_decoupled_wbc import _make_sonic_config
        from simple.baselines.psi0_decoupled_wbc import Psi0DecoupledWbcAgent

        sonic_config = _make_sonic_config()
        raw_env = gym.make(env_id, sim_mode=sim_mode, render_hz=render_hz, headless=True, sonic_config=sonic_config)
        task = raw_env.unwrapped.task
        agent = Psi0DecoupledWbcAgent(task.robot, "127.0.0.1", 0, sonic_config=sonic_config)
        env = raw_env
    else:
        from simple.envs.wrappers import StandStabilizationWrapper
        from simple.baselines.psi0 import Psi0Agent

        raw_env = gym.make(env_id, sim_mode=sim_mode, render_hz=render_hz, headless=True, success_criteria=0.9)
        task = raw_env.unwrapped.task
        env = StandStabilizationWrapper(raw_env)
        agent = Psi0Agent(task.robot, "127.0.0.1", 0, client=ReplayClient(np.zeros((1, 36))))
    return env, raw_env, task, agent, teleop


def robot_base_pose(raw_env) -> np.ndarray:
    import mujoco

    eng = raw_env.unwrapped.mujoco
    m, d = eng.mjModel, eng.mjData
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return np.concatenate([d.xpos[bid], d.xquat[bid]]).astype(np.float64)


def object_poses(raw_env) -> dict[str, np.ndarray]:
    out = {}
    for k, v in raw_env.unwrapped.mujoco.mj_objects.items():
        out[str(k)] = np.concatenate([v.xpos, v.xquat]).astype(np.float64)
    return out


def replay_episode(args, env, raw_env, task, agent, teleop, root, info, row, out_dir: Path, wrist) -> dict:
    import torch
    from simple.determinism import seed_episode

    ep = int(row["episode_index"])
    df = episode_frames(root, info, ep)
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
    rec_states = np.stack(df["states"].to_numpy()).astype(np.float32)
    env_conf = env_config_of(row)

    seed = seed_episode(int(os.environ["SIMPLE_EVAL_SEED"]), ep)
    observation, info_ = env.reset(seed=seed, options={"state_dict": env_conf})
    eng = raw_env.unwrapped.mujoco
    wrist_cams.fix_near_plane(eng.mjModel)

    padded = actions
    if args.pad_frames > 0:
        # hold the last command after the recording ends (walking zeroed), as SIMPLE's
        # ReplayDecoupledAgent does: the recordings stop at the success moment, the replay
        # needs a few more frames for the slower physical lift to cross the threshold
        pad = np.repeat(actions[-1:], args.pad_frames, axis=0)
        pad[:, 32:35] = 0.0
        padded = np.concatenate([actions, pad], axis=0)
    # pre-pose: before frame 0, hold the recording's first command (walking zeroed) so the replay
    # starts in the recorded first pose, not the controller's default stance (teleop sets: elbows
    # -0.58 vs +0.08 rad, a 300 ms swing otherwise). Not recorded.
    n_pre = args.prepose_frames if (teleop and args.prepose_frames > 0) else 0
    pre = np.repeat(actions[:1], n_pre, axis=0)
    if n_pre:
        pre[:, 32:35] = 0.0
    agent.client = ReplayClient(pre, padded)
    agent.reset()
    sonic_env = raw_env.unwrapped
    robot = task.robot
    if teleop:
        agent._wbc_policy.lower_body_policy.use_policy_action = True
        n = 0
        while not robot.stabilized and n < 300:
            observation, *_, info_ = sonic_env.step(agent.get_stabilize_action(observation))
            n += 1
        agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1,), dtype=torch.float32)
    instruction = task.instruction
    for _ in range(n_pre):
        action = agent.get_action(observation, info={**info_, "episode_index": ep}, instruction=instruction)
        observation, _r, terminated, truncated, info_ = env.step(action)
    if n_pre and teleop:
        agent._wbc_policy.lower_body_policy.gait_indices = torch.zeros((1,), dtype=torch.float32)

    writers = {}
    keys = ["head_stereo_left"] + ([k for k in wrist.keys] if wrist else [])
    names = {"head_stereo_left": "egocentric"} | ({k: k for k in wrist.keys} if wrist else {})
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in keys:
        writers[k] = wrist_cams.VideoWriter(out_dir / f"{names[k]}.mp4", fps=args.render_hz)

    # row t of the psi0 sets = action t + the state/image AFTER that action was applied
    # (checked on all six sets: replayed post-step state matches the recorded row at lag 0)
    qpos, base, objs = [], [], []
    t0 = time.perf_counter()
    step = 0
    success = False
    max_reward = 0.0
    while True:
        try:
            action = agent.get_action(observation, info={**info_, "episode_index": ep}, instruction=instruction)
        except StopIteration:
            break
        observation, _r, terminated, truncated, info_ = env.step(action)
        for k in keys:
            writers[k].write(observation[k])
        qpos.append(np.asarray(observation["joint_qpos"], dtype=np.float32))
        base.append(robot_base_pose(raw_env))
        objs.append(object_poses(raw_env))
        max_reward = max(max_reward, float(_r))
        success = success or bool(getattr(raw_env.unwrapped, "_success", False))
        step += 1
        if success and step >= len(actions) and args.stop_on_success:
            break
        if step >= len(padded):
            break
    for w in writers.values():
        w.close()
    secs = time.perf_counter() - t0

    qpos = np.stack(qpos)
    names_obj = sorted(objs[0]) if objs else []
    obj_arr = np.stack([[o[k] for k in names_obj] for o in objs]) if names_obj else np.zeros((len(qpos), 0, 7))
    np.savez_compressed(out_dir / "track.npz", joint_qpos=qpos, base_pose=np.stack(base),
                        object_poses=obj_arr, object_names=np.array(names_obj), actions=actions,
                        actions_sent=padded[: len(qpos)],
                        recorded_states=rec_states)
    err = tracking_error(qpos, rec_states)
    summary = dict(
        episode_index=ep, frames_recorded=int(len(actions)), frames_replayed=int(len(qpos)),
        pad_frames_used=int(max(0, step - len(actions))), prepose_frames=int(n_pre),
        task_success=bool(success or getattr(raw_env.unwrapped, "_success", False)),
        max_reward=round(max_reward, 4), success_criteria=float(getattr(task, "success_criteria", float("nan"))),
        sim_clock=os.environ.get("SIMPLE_DETERMINISTIC_RUNTIME"),
        seconds=round(secs, 1), sim_mode=args.sim_mode, env_id=args.env_id,
        wrist=(wrist.to_json() if wrist else None), tracking=err,
        instruction=instruction,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


# psi0 `states` = hands 0:14 | arms 14:28 | waist rpy 28:31 | height 31 (see postprocess_psi0*.py)
STATE_SLICES = [("left_hand_thumb", 29, 32), ("left_hand_middle", 34, 36), ("left_hand_index", 32, 34),
                ("right_hand", 36, 43), ("left_arm", 15, 22), ("right_arm", 22, 29)]


def to_psi0_states(qpos: np.ndarray) -> np.ndarray:
    parts = [qpos[:, s:e] for _, s, e in STATE_SLICES] + [qpos[:, [13, 14, 12]]]
    return np.concatenate(parts, axis=1)


def tracking_error(qpos: np.ndarray, rec_states: np.ndarray) -> dict:
    sim = to_psi0_states(qpos)                     # (T+1, 31)
    n = min(len(sim), len(rec_states))
    best = None
    for lag in (0, 1, -1):
        a, b = (sim[lag:n], rec_states[: n - lag, :31]) if lag >= 0 else (sim[: n + lag], rec_states[-lag:n, :31])
        e = np.abs(a - b)
        row = dict(lag_frames=lag, arms_mean_rad=float(e[:, 14:28].mean()), arms_max_rad=float(e[:, 14:28].max()),
                   hands_mean_rad=float(e[:, 0:14].mean()), waist_mean_rad=float(e[:, 28:31].mean()))
        if best is None or row["arms_mean_rad"] < best["arms_mean_rad"]:
            best = row
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path, help="psi0 dataset root (meta/, data/, videos/)")
    ap.add_argument("--env-id", default=None, help="default: simple/<dataset folder name>")
    ap.add_argument("--episodes", default="0", help="comma list or a-b range of episode indices, or 'all'")
    ap.add_argument("--sim-mode", default="mujoco_isaac", choices=["mujoco", "mujoco_isaac"],
                    help="mujoco_isaac = Isaac renders (the look of the psi0 videos); mujoco = fast check")
    ap.add_argument("--render-hz", type=int, default=50)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--wrist-cams", action="store_true", help="add the two wrist cameras")
    ap.add_argument("--wrist-side", default="index", choices=["index", "middle", "back", "palm"],
                    help="where the collar housing sits around the forearm: index-finger side (default: sees the grasp), "
                         "middle-finger side, back of the hand, or palm side (the Revo 2 design; on Dex3 the thumb swings into it)")
    ap.add_argument("--wrist-tilt", type=float, default=wrist_cams.DEFAULT_TILT_DEG,
                    help="camera tilt toward the hand axis, deg (the printed collar holds 0)")
    ap.add_argument("--wrist-hfov", type=float, default=wrist_cams.DEFAULT_HFOV_DEG, help="horizontal FOV, deg")
    ap.add_argument("--wrist-size", default="640x360")
    ap.add_argument("--prepose-frames", type=int, default=100,
                    help="teleop sets: hold the recording's first command this many 20 ms frames before frame 0")
    ap.add_argument("--pad-frames", type=int, default=25,
                    help="hold the last recorded command this many extra 20 ms frames (walking zeroed)")
    ap.add_argument("--stop-on-success", action=argparse.BooleanOptionalAction, default=True,
                    help="end the padded hold as soon as the task succeeds")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    root = args.data_dir
    args.env_id = args.env_id or f"simple/{root.name}"
    info = read_info(root)
    rows = {int(r["episode_index"]): r for r in read_episodes(root)}
    if args.episodes == "all":
        eps = sorted(rows)
    elif "-" in args.episodes:
        a, b = args.episodes.split("-")
        eps = list(range(int(a), int(b) + 1))
    else:
        eps = [int(x) for x in args.episodes.split(",") if x != ""]

    wrist = None
    if args.wrist_cams:
        w, h = (int(v) for v in args.wrist_size.lower().split("x"))
        wrist = wrist_cams.WristCamSpec(tilt_deg=args.wrist_tilt, hfov_deg=args.wrist_hfov, width=w, height=h,
                                        mount_side=args.wrist_side)

    env, raw_env, task, agent, teleop = build(args.env_id, args.sim_mode, args.render_hz, wrist)
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for ep in eps:
        out_dir = args.out / f"ep_{ep:06d}"
        if args.skip_existing and (out_dir / "summary.json").exists():
            continue
        s = replay_episode(args, env, raw_env, task, agent, teleop, root, info, rows[ep], out_dir, wrist)
        results.append(s)
        print(f"[replay] {args.env_id} ep {ep}: success={s['task_success']} frames {s['frames_replayed']}/"
              f"{s['frames_recorded']} arms err {s['tracking']['arms_mean_rad']:.4f} rad ({s['seconds']} s)", flush=True)
    with (args.out / "results.jsonl").open("a") as f:
        for s in results:
            f.write(json.dumps(s) + "\n")
    os._exit(0)  # Isaac's shutdown can hang; everything is written


if __name__ == "__main__":
    main()
