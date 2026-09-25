#!/usr/bin/env python3
"""Load the coffee-cart SIMPLE task, reset it once and report where everything landed.

    MUJOCO_GL=egl ~/wrk/SIMPLE/.venv/bin/python probe_scene.py
Writes probe/probe.json, probe/head_stereo_left.png and probe/third_person.png.
"""
import os
import sys
from pathlib import Path

import numpy as np
import PIL.Image as I

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HZ = 50
THIRD = dict(lookat=[0.85, -0.35, 0.70], distance=4.6, azimuth=145, elevation=-18)


def make_env():
    import gymnasium as gym
    import coffee_cart_task as T
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    cfg = SimLoopConfig().load_wbc_yaml(); cfg["ENV_NAME"] = "simple"
    env = gym.make(T.ENV_ID, sim_mode="mujoco", render_hz=HZ, physics_dt=cfg["SIMULATE_DT"], headless=True,
                   max_episode_steps=10**6, sonic_config=cfg, target=T.TARGET, dr_level=0, success_criteria=1e9)
    return env, T


def third_person_camera(m):
    import mujoco
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = THIRD["lookat"]; cam.distance = THIRD["distance"]
    cam.azimuth = THIRD["azimuth"]; cam.elevation = THIRD["elevation"]
    return cam


def scene_checks(env, T, obs):
    import mujoco
    robot = env.unwrapped.task.robot; m, d = robot.mjModel, robot.mjData
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    xpos = lambda n: d.xpos[bid(n)].round(3).tolist() if bid(n) >= 0 else None
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "head_stereo_left")
    axis = -d.cam_xmat[cam].reshape(3, 3)[:, 2]
    tgt = env.unwrapped.task.layout.actors["target"].asset.label
    table = env.unwrapped.task.layout.scene.table
    L = T.L
    return {
        "robot_to_cart": round(T.ROBOT_TO_CART, 3), "route_forward": round(T.ROUTE_FORWARD, 3),
        "table_from_line": round(T.TABLE_FROM_LINE, 3),
        "cup_mass_kg": T.CUP_MASS, "cart_mass_kg": T.CART_MASS,
        "pelvis": xpos("pelvis"),
        "cart": xpos("cart"), "cart_expected": [round(L["cart_cx"], 3), round(L["cart_cy"], 3), 0.0],
        "target": {tgt: xpos(tgt)},
        "target_expected": [round(L["cup_x"], 3), round(L["cup_y"], 3), round(L["box_top"], 3)],
        "table_top_z": round(float(table.pose.position[2] + 0.5 * table.size[2]), 3),
        "table_near_edge_x": round(float(table.pose.position[0] - 0.5 * table.size[0]), 3),
        "table_centre": [round(float(v), 3) for v in table.pose.position[:2]],
        "table_left_edge_y": round(float(table.pose.position[1] + 0.5 * table.size[1]), 3),
        "table_legs": xpos("table_legs"),
        "cart_body_mass_kg": round(float(m.body_mass[bid("cart")]), 3) if bid("cart") >= 0 else None,
        "cup_body_mass_kg": round(float(m.body_mass[bid(tgt)]), 3) if bid(tgt) >= 0 else None,
        "head_camera": {"pos": d.cam_xpos[cam].round(3).tolist(),
                        "pitch_down_deg": round(float(np.degrees(np.arctan2(-axis[2], np.hypot(axis[0], axis[1])))), 1),
                        "fovy_deg": round(float(m.cam_fovy[cam]), 1),
                        "image": list(np.asarray(obs["head_stereo_left"]).shape[:2][::-1])},
        "obs_keys": sorted(obs.keys()), "instruction": env.unwrapped.task.instruction,
    }


def main():
    import json
    import mujoco
    out_dir = HERE / "probe"; out_dir.mkdir(parents=True, exist_ok=True)
    env, T = make_env()
    obs, info = env.reset()
    robot = env.unwrapped.task.robot; m, d = robot.mjModel, robot.mjData
    T.recolor_hands(m)
    checks = scene_checks(env, T, obs)
    print(json.dumps(checks, indent=1))
    (out_dir / "probe.json").write_text(json.dumps(checks, indent=1))
    # re-render the head camera AFTER recolor_hands (obs was rendered during reset, before it)
    rh = mujoco.Renderer(m, height=360, width=640)
    rh.update_scene(d, camera="head_stereo_left")
    I.fromarray(rh.render()).save(out_dir / "head_stereo_left.png")
    rh.close()
    r3 = mujoco.Renderer(m, height=720, width=1280)
    r3.update_scene(d, third_person_camera(m))
    I.fromarray(r3.render()).save(out_dir / "third_person.png")
    r3.close()
    print("wrote", out_dir)
    sys.stdout.flush(); sys.stderr.flush()     # os._exit does not flush a piped stdout
    os._exit(0)          # EGL / SDK teardown aborts otherwise


if __name__ == "__main__":
    main()
