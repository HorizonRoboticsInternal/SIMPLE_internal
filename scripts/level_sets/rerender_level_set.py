#!/usr/bin/env python
"""Re-render a kit level set EXACTLY from its stored scene configs (Isaac head camera), writing a new set in the same layout.

    MUJOCO_GL=egl .venv/bin/python scripts/level_sets/rerender_level_set.py <kit> --src <set dir> --out <new set dir> [--note "..."]

<src>/meta/episodes.jsonl supplies every scene's environment_config (reloaded verbatim: DRManager.load_state_dict with
dr_level=None) and <src>/meta/scene_env.json the kit's env vars (applied before the kit task is imported, as eval_scene.py
does). Each scene is settled exactly as make_levels.py / eval_decoupled_wbc do before the first frame (controller start-up,
objects at rest), mirrored into Isaac and rendered once -> <out>/{data,videos,frames/epNN.png,meta/*}. meta/rerender_check.json
records per scene the settle steps, how far every free body moved while settling, the robot start, and the 'wall' body when
the kit has one. Other meta files of <src> (feasibility_*, table_material_fix, scene_env) are copied; scene_env gets a
"rerendered" note.  2026-10-21: bowl_sink geometry aligned with the recordings; coffee_cart with the wall.
"""
import argparse, json, os, shutil, sys, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
ROOT = Path(__file__).resolve().parents[2]; KITS = Path.home() / "wrk/robot_orchard_deploy/holobrain_g1_deploy/sim"

ap = argparse.ArgumentParser(); ap.add_argument("kit"); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--render-hz", type=int, default=50); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--note", default="")
a = ap.parse_args(); SRC, OUT = Path(a.src).resolve(), Path(a.out).resolve()
scene_env = json.load(open(SRC / "meta/scene_env.json"))
for k, v in scene_env.get("env", {}).items(): os.environ.setdefault(k, v)
print("[rerender] env: " + " ".join(f"{k}={os.environ[k]}" for k in scene_env.get("env", {})), flush=True)
sys.path.insert(0, str(KITS / a.kit)); sys.path.insert(0, str(KITS))
import importlib, numpy as np, gymnasium as gym, mujoco
from PIL import Image
mod = importlib.import_module(f"{a.kit}_task")
from simple.cli.dr_decoupled_wbc import _make_sonic_config, _init_exporter, _save_episode_env_config
from simple.agents.pico_decoupled_agent import PicoDecoupledAgent

eps = [json.loads(l) for l in open(SRC / "meta/episodes.jsonl")]
cfgs = [json.loads(e["environment_config"]) if isinstance(e["environment_config"], str) else e["environment_config"] for e in eps]
if a.limit: cfgs = cfgs[:a.limit]
if OUT.exists(): shutil.rmtree(OUT)
env = gym.make(mod.ENV_ID, sim_mode="mujoco_isaac", render_hz=a.render_hz, headless=True, sonic_config=_make_sonic_config())
sonic_env = env.unwrapped; task = sonic_env.task; robot = task.robot; agent = PicoDecoupledAgent(task.robot)
exporter = None; checks = []; t0 = time.time()


def free_bodies(m):
    out = []
    for j in range(m.njnt):
        if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            b = int(m.jnt_bodyid[j]); n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or str(b)
            if "pelvis" not in n and not n.startswith("g1"): out.append((n, b))
    return out


for i, cfg in enumerate(cfgs):
    s = scene_env.get("seed", 0) + 1000 * (int(scene_env.get("level", 3)) + 1) + i
    obs, _ = env.reset(seed=s, options={"state_dict": cfg, "dr_level": None})
    mj = sonic_env.mujoco; m, d = mj.mjModel, mj.mjData
    fb = free_bodies(m); before = {n: d.xpos[b].copy() for n, b in fb}
    agent.reset(); agent._wbc_policy.lower_body_policy.use_policy_action = True
    n_settle = 0
    while not robot.stabilized and n_settle < 300:
        act = agent.get_stabilize_action(obs)
        for _ in range(sonic_env.control_decimal):
            mj.apply_action(act); mj.step(render=False)
        obs = {"joint_qpos": np.asarray(list(mj.get_robot_qpos().values()), dtype=np.float32)}
        n_settle += 1
    for _ in range(3): sonic_env.isaac.step(mj)
    obs = sonic_env._get_obs()
    moved = {n: round(float(np.linalg.norm(d.xpos[b] - before[n])), 4) for n, b in fb}
    wall = {}
    for n in ("wall",):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        if b >= 0: wall = dict(body=n, pos=[round(float(x), 3) for x in d.xpos[b]])
    sd = task.state_dict(); sp = sd["dr_state_dict"]["spatial"]; rk = next(k for k in sp if k.startswith("g1"))
    if exporter is None:
        obj_names = list(mj.mj_objects.keys())
        exporter = _init_exporter(str(OUT), task.instruction, agent._dwbc_robot_model, obj_names, robot.joint_names, obs["head_stereo_left"].shape)
        (OUT / "frames").mkdir(parents=True, exist_ok=True)
    vel = np.concatenate([np.asarray(obs.get("base_lin_vel", np.zeros(3))), np.asarray(obs.get("base_ang_vel", np.zeros(3)))])
    frame = {"observation.images.ego_view": obs["head_stereo_left"], "observation.state": np.asarray(obs["joint_qpos"], dtype=np.float64),
             "observation.base_pose": np.asarray(obs.get("base_pose_quat", obs.get("floating_base_pose", np.zeros(7))), dtype=np.float64),
             "observation.base_vel": np.asarray(vel, dtype=np.float64), "observation.eef_state": np.zeros(14, dtype=np.float64),
             "observation.img_state_delta": np.zeros(1, dtype=np.float32), "teleop.navigate_command": np.zeros(4, dtype=np.float64),
             "teleop.base_height_command": np.zeros(1, dtype=np.float64), "action": np.zeros(43, dtype=np.float64), "action.eef": np.zeros(14, dtype=np.float64),
             "observation.torso_rpy_command": np.zeros(3, dtype=np.float64)}
    frame["observation.object_poses"] = np.asarray(obs["object_poses"], dtype=np.float64) if "object_poses" in obs else np.zeros(len(obj_names) * 7, dtype=np.float64)
    exporter.add_frame(frame); time.sleep(0.5); exporter.save_episode(); _save_episode_env_config(exporter, sd, i)
    Image.fromarray(obs["head_stereo_left"]).save(OUT / "frames" / f"ep{i:02d}.png")
    checks.append(dict(scene=i, settle_steps=n_settle, robot_start_stored=[round(float(x), 4) for x in cfg["dr_state_dict"]["spatial"][rk]["position"][:2]],
                       robot_start=[round(float(x), 4) for x in sp[rk]["position"][:2]], free_body_motion_m=moved, max_motion_m=max(moved.values()) if moved else 0.0, wall=wall))
    json.dump(dict(kit=a.kit, src=str(SRC), out=str(OUT), date="2026-10-21", note=a.note, scenes=checks), open(OUT / "rerender_check.json", "w"), indent=1)
    print(f"[rerender] {a.kit} scene {i + 1}/{len(cfgs)}: settled {n_settle} steps, max free-body motion {checks[-1]['max_motion_m']:.3f} m"
          f"{', wall at ' + str(wall['pos']) if wall else ''} ({(time.time() - t0) / (i + 1):.0f} s/scene)", flush=True)
exporter.stop_video_writers()
for f in SRC.glob("meta/*.json*"):
    if f.name not in ("episodes.jsonl", "info.json", "episodes_stats.jsonl", "tasks.jsonl", "modality.json"): shutil.copy2(f, OUT / "meta" / f.name)
shutil.move(str(OUT / "rerender_check.json"), str(OUT / "meta/rerender_check.json"))
se = json.load(open(OUT / "meta/scene_env.json")); se["rerendered"] = dict(date="2026-10-21", from_set=str(SRC), note=a.note, tool="scripts/level_sets/rerender_level_set.py"); se["eval"] = se.get("eval", "").replace(str(SRC), str(OUT)); json.dump(se, open(OUT / "meta/scene_env.json", "w"), indent=1)
env.close(); agent.close()
print(f"[rerender] done: {len(cfgs)} scenes -> {OUT}", flush=True); sys.stdout.flush(); os._exit(0)
