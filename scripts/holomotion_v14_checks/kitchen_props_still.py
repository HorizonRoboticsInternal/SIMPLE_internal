#!/usr/bin/env python
"""Stillness check of the kitchen's free props inside the SIMPLE env (2026-10-26).

Builds the kitchen teleop env (Revo 2 robot, uncontrolled, it falls near the origin 1 m from the props), steps 30 s of
physics and prints each free prop's max speed in seconds 5/15/30, its tilt and xy drift.  Every prop should read 0.0 cm/s
after the first second; a growing number is a rocking contact (the bottles did this before kitchen_task._bottle_feet:
one contact point under a cylinder, pumped by SIMPLE's impratio 10 / noslip 2).  Run after adding or editing kitchen props.
    MUJOCO_GL=egl .venv/bin/python scripts/holomotion_v14_checks/kitchen_props_still.py [--seconds 30]
"""
import argparse
import sys, numpy as np, mujoco, copy
ap = argparse.ArgumentParser(); ap.add_argument("--seconds", type=float, default=30.0); A = ap.parse_args()
sys.path.insert(0, "src")
import gymnasium as gym
from simple.cli.teleop_holomotion_v14 import _load_sonic_config, load_scene
mod = load_scene("kitchen"); sc = _load_sonic_config()
env = gym.make(mod.ENV_ID, sim_mode="mujoco", render_hz=50, physics_dt=sc["SIMULATE_DT"], headless=True, max_episode_steps=10**9, sonic_config=sc, target=mod.TARGET, dr_level=0, success_criteria=0.9, robot_uid="g1_sonic_revo2")
senv = env.unwrapped; env.reset(seed=0); M, D = senv.mujoco.mjModel, senv.mujoco.mjData
feet = [M.geom(g).name for g in range(M.ngeom) if "_foot_" in M.geom(g).name]; print("feet geoms:", feet)
free = [M.body(i).name for i in range(M.nbody) if M.body(i).name.startswith("k_") and M.body_jntnum[i] == 1 and M.jnt_type[M.body_jntadr[i]] == mujoco.mjtJoint.mjJNT_FREE] + ["bottle_red"] + [n for n in ("ball_small", "ball_large") if mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_BODY, n) >= 0]
ids = {b: M.body(b).id for b in free}; per_s = int(round(1 / M.opt.timestep)); rep = {b: [] for b in free}; cur = {b: 0.0 for b in free}; start = {b: D.xpos[i].copy() for b, i in ids.items()}
for i in range(int(A.seconds * per_s)):
    mujoco.mj_step(M, D)
    for b, bi in ids.items(): cur[b] = max(cur[b], float(np.linalg.norm(D.cvel[bi][3:])))
    if (i + 1) % per_s == 0:
        for b in free: rep[b].append(cur[b]); cur[b] = 0.0
for b, bi in ids.items():
    q = D.xquat[bi]; tilt = float(np.degrees(np.arccos(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1)))); drift = np.linalg.norm(D.xpos[bi][:2] - start[b][:2]) * 100
    print(f"  {b}: max speed 5/15/last s {rep[b][4]*100:.1f}/{rep[b][14]*100:.1f}/{rep[b][-1]*100:.1f} cm/s; tilt {tilt:.1f} deg; drift {drift:.1f} cm")
lid = mujoco.mj_name2id(M, mujoco.mjtObj.mjOBJ_JOINT, "k_bin_lid")
if lid >= 0:                                                            # the pedal bin's lid (2026-10-26): held open by its spring
    print(f"  pedal bin lid: {np.degrees(D.qpos[M.jnt_qposadr[lid]]):.1f} deg after {A.seconds:g} s (open limit {np.degrees(1.45):.1f})")
