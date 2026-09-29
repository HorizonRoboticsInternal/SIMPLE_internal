#!/usr/bin/env python
"""Does the target stand still? Loads every scene of a kit level set exactly (MuJoCo only), runs the same stabilisation the
eval and make_levels.py run before the policy starts, holds the start pose 2 s more, and measures how far the target
moved and tilted and whether it is still on the table.

    python kit_settle.py bottle_bin <set dir> --out settle.jsonl            (a kit)
    python kit_settle.py simple/G1WholebodyHandoverTeleop-v0 <set dir> --out settle.jsonl   (a benchmark teleop task)
A scene passes when the target moved < 1 cm, tilted < 5 deg and stays on the table top (within 1.5 cm of its spawn height).
"""
import argparse, importlib, json, math, os, sys
from pathlib import Path

import numpy as np

SIM = Path(__file__).resolve().parents[1]           # scenes/: the kits live in scenes/<kit>
TARGETS = {"bottle_bin": "bottle_500ml", "bowl_sink": "bowl_15cm", "coffee_cart": "coffee_cup"}
ap = argparse.ArgumentParser(); ap.add_argument("kit"); ap.add_argument("set_dir"); ap.add_argument("--out", required=True)
ap.add_argument("--scenes", type=int, nargs="*", default=None); ap.add_argument("--hold-s", type=float, default=2.0)
ap.add_argument("--frames", default=None, help="save the MuJoCo head-camera view after the hold here (ep<i>.png): a preview before the Isaac render")
a = ap.parse_args()
sd = Path(a.set_dir)
if (sd / "meta/scene_env.json").exists():
    for k, v in json.load(open(sd / "meta/scene_env.json")).get("env", {}).items():
        os.environ.setdefault(k, v)
os.environ.setdefault("MUJOCO_GL", "egl")
DOM = int(os.environ.get("KIT_DDS_DOMAIN", "38"))

import simple.cli.dr_decoupled_wbc as D  # noqa: E402
_mk = D._make_sonic_config
D._make_sonic_config = lambda: {**_mk(), "DOMAIN_ID": DOM}
if a.kit.startswith("simple/"):                               # a benchmark teleop task (no kit module): the env id itself
    mod = type("M", (), {"ENV_ID": a.kit})
else:
    sys.path.insert(0, str(SIM / a.kit))
    mod = importlib.import_module(f"{a.kit}_task")
import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
from simple.agents.pico_decoupled_agent import PicoDecoupledAgent  # noqa: E402

rows = [json.loads(l) for l in open(sd / "meta/episodes.jsonl") if l.strip()]
env = gym.make(mod.ENV_ID, sim_mode="mujoco", render_hz=50, headless=True, sonic_config=D._make_sonic_config())
u = env.unwrapped; task = u.task; robot = task.robot
agent = PicoDecoupledAgent(task.robot)
done = {json.loads(l)["scene"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()


def tpose():
    mj = u.mujoco; m, d = mj.mjModel, mj.mjData
    if a.kit not in TARGETS:                                     # benchmark tasks: the engine's "target" object
        o = mj.mj_objects.get("target")
        return np.array(o.xpos, dtype=float).copy(), np.array(o.xquat, dtype=float).copy()
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TARGETS[a.kit])
    if b < 0:
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)]
        b = next(i for i, n in enumerate(names) if n and TARGETS[a.kit] in n)
    return d.xpos[b].copy(), d.xquat[b].copy()


def tilt_deg(q):
    w, x, y, z = q
    zz = 1 - 2 * (x * x + y * y)                              # world-z component of the body z axis
    return math.degrees(math.acos(max(-1.0, min(1.0, zz))))


def step_stab(obs, n):
    mj = u.mujoco
    for _ in range(n):
        act = agent.get_stabilize_action(obs)
        for _ in range(u.control_decimal):
            mj.apply_action(act); mj.step(render=False)
        obs = {"joint_qpos": np.asarray(list(mj.get_robot_qpos().values()), dtype=np.float32)}
    return obs


for i, r in enumerate(rows):
    if (a.scenes is not None and i not in a.scenes) or i in done:
        continue
    c = r["environment_config"]
    while isinstance(c, str):
        c = json.loads(c)
    obs, _ = env.reset(options={"state_dict": c})
    mujoco.mj_forward(u.mujoco.mjModel, u.mujoco.mjData)
    p0, q0 = tpose(); tilt0 = tilt_deg(q0)
    agent.reset(); agent._wbc_policy.lower_body_policy.use_policy_action = True
    n = 0
    while not robot.stabilized and n < 300:
        obs = step_stab(obs, 1); n += 1
    p1, q1 = tpose()
    obs = step_stab(obs, int(a.hold_s * 50))
    p2, q2 = tpose()
    moved = float(np.linalg.norm(p2[:2] - p0[:2])); dz = float(p2[2] - p0[2]); tilt = tilt_deg(q2) - tilt0
    ok = moved < 0.01 and abs(tilt) < 5.0 and abs(dz) < 0.015
    rec = {"scene": i, "ok": ok, "moved_m": round(moved, 4), "dz_m": round(dz, 4), "tilt_deg": round(tilt, 2),
           "moved_at_settle_m": round(float(np.linalg.norm(p1[:2] - p0[:2])), 4), "settle_steps": n, "spawn": p0.round(4).tolist(), "end": p2.round(4).tolist()}
    if a.frames:
        from PIL import Image
        Path(a.frames).mkdir(parents=True, exist_ok=True)
        o = u._get_obs(); img = o.get("head_stereo_left") if isinstance(o, dict) else None
        if img is not None:
            Image.fromarray(np.asarray(img)).save(Path(a.frames) / f"ep{i:02d}.png")
    print(f"scene {i}: {'OK' if ok else 'MOVED'} moved {moved * 100:.1f} cm dz {dz * 100:+.1f} cm tilt {tilt:+.1f} deg (settle {n} steps)", flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
print("DONE", flush=True)
os._exit(0)
