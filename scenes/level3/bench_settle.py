#!/usr/bin/env python
"""Does the target stay put while the env steps? Loads every scene of a benchmark MP level set exactly (MuJoCo), lets the
robot stand still for --hold-s seconds (the motion-planning agent's stand command: legs balance, arms keep the reset
pose, nothing is grasped) and records the target: how far it moved, its lowest height, its tilt.

    python bench_settle.py simple/G1WholebodyBendPickMP-v0 <set dir> --out settle.jsonl [--hold-s 5]
A scene passes when the target moved < 1 cm, tilted < 5 deg and never dropped more than 2 cm.
"""
import argparse, json, math, os

import gymnasium as gym
import numpy as np
import simple.envs  # noqa
from simple.agents.mp import MotionPlannerAgent
from simple.datagen.subtask_spec import StandSpec

ap = argparse.ArgumentParser(); ap.add_argument("env_id"); ap.add_argument("set_dir"); ap.add_argument("--out", required=True)
ap.add_argument("--hold-s", type=float, default=5.0); ap.add_argument("--render-hz", type=int, default=50)
ap.add_argument("--scenes", type=int, nargs="*", default=None)
a = ap.parse_args()
rows = [json.loads(l) for l in open(os.path.join(a.set_dir, "meta/episodes.jsonl")) if l.strip()]
env = gym.make(a.env_id, sim_mode="mujoco", headless=True, webrtc=False, render_hz=a.render_hz, max_episode_steps=10 ** 6)
u = env.unwrapped; task = u.task
agent = MotionPlannerAgent(task, None, debug=False, plan_batch_size=1)      # the stand command never calls the planner
done = {json.loads(l)["scene"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()


def tpose():
    o = u.mujoco.mj_objects.get("target")
    return np.array(o.xpos, dtype=float).copy(), np.array(o.xquat, dtype=float).copy()


def tilt(q):
    w, x, y, z = q
    return math.degrees(math.acos(max(-1.0, min(1.0, 1 - 2 * (x * x + y * y)))))


for i, r in enumerate(rows):
    if i in done or (a.scenes is not None and i not in a.scenes):
        continue
    c = r["environment_config"]
    while isinstance(c, str):
        c = json.loads(c)
    obs, info = env.reset(options={"state_dict": c})
    p0, q0 = tpose(); t0 = tilt(q0); zmin = p0[2]
    task.decompose = lambda: [StandSpec("stand", steps=int(a.hold_s * a.render_hz))]
    agent.reset(); agent.synthesize(); n = 0
    try:
        while True:
            obs, rew, term, trunc, info = env.step(agent.get_action(obs, info)); n += 1
            zmin = min(zmin, tpose()[0][2])
    except StopIteration:
        pass
    p1, q1 = tpose()
    import mujoco
    m, dd = u.mujoco.mjModel, u.mujoco.mjData
    tb = u.mujoco.mj_objects.get("target").id
    touch = sorted({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) for k in range(dd.ncon)
                    for g1, g2 in [(dd.contact[k].geom1, dd.contact[k].geom2)] for g, other in ((g1, g2), (g2, g1))
                    if int(m.geom_bodyid[other]) == tb and int(m.geom_bodyid[g]) != tb})
    moved = float(np.linalg.norm(p1[:2] - p0[:2])); drop = float(p0[2] - zmin); tl = tilt(q1) - t0
    ok = moved < 0.01 and abs(tl) < 5.0 and drop < 0.02
    rec = {"scene": i, "ok": ok, "moved_m": round(moved, 4), "max_drop_m": round(drop, 4), "tilt_deg": round(tl, 2), "steps": n,
           "spawn": p0.round(4).tolist(), "end": p1.round(4).tolist(), "touching": touch}
    print(f"scene {i} (#{i + 1}): {'OK' if ok else 'MOVED'} moved {moved * 100:.1f} cm, drop {drop * 100:.1f} cm, tilt {tl:+.1f} deg ({n} steps) touching {touch}", flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
print("DONE", flush=True)
os._exit(0)
