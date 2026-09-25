#!/usr/bin/env python
"""Build SIMPLE-style evaluation sets (levels 0/1/2/3) for the real-world scene kits, rendered in Isaac.

    python sim/make_levels.py bowl_sink --levels 0 1 2 3 --episodes 20 --out data/evals_scenes

writes  <out>/<gym id>/dr-level-<n>/  in the same LeRobot layout as SIMPLE's data/evals_new20 sets
(meta/info.json, meta/episodes.jsonl with environment_config, data/*.parquet, videos ego_view) plus
frames/ep<i>.png (the Isaac head camera of every scene) and meta/scene_env.json (the env vars the level
was generated with; eval_scene.py re-applies them so the extra furniture lands in the same place).

Levels follow SIMPLE's DRManager.load_state_dict, one process per level (the kit reads its env vars at import):
  0  new distractors and table material          (base scene otherwise)
  1  + new lighting
  2  + new object pose (target region widened by <KIT>_TARGET_JITTER)
  3  + alternative layout: LEVEL3_EPISODES scenes, each with a different robot start inside the threshold box
     (+-ROBOT_BACK forward/back, +-ROBOT_LEFT sideways; level3_starts spreads them evenly, shuffled by the seed);
     SIMPLE's dr_level 3 re-samples the robot start from the config, and the start is stored in the scene's state
The thresholds are the constants below.
"""
from __future__ import annotations
import argparse, json, os, random, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KITS = {
    "bottle_bin":  dict(prefix="BOTTLE_BIN",  env_id="simple/G1WholebodyBottleBinTeleop-v0",  dist_var="BOTTLE_BIN_ROBOT_TO_EDGE",  dist_key="pelvis_to_edge", jitter="0.03,0.08"),
    "bowl_sink":   dict(prefix="BOWL_SINK",   env_id="simple/G1WholebodyBowlSinkTeleop-v0",   dist_var="BOWL_SINK_ROBOT_TO_EDGE",   dist_key="pelvis_to_edge", jitter="0,0.08"),        # the bowl stays on the edge line
    "coffee_cart": dict(prefix="COFFEE_CART", env_id="simple/G1WholebodyCoffeeCartTeleop-v0", dist_var="COFFEE_CART_ROBOT_TO_CART", dist_key="robot_to_cart",  jitter="0.02,0.03",
                        region="0.475,0.65,-0.115,0.115"),   # level 2/3: anywhere on the near half of the box top, cup fully on it
}
NUM_DISTRACTORS = 3       # SIMPLE's own tasks use 3 GraspNet distractors
ROBOT_BACK = 0.10         # level 3 threshold: the robot's start moves up to this far back/forward (m) ...
ROBOT_LEFT = 0.05         # ... and up to this far sideways (m); every level-3 scene gets a different pair
LEVEL3_EPISODES = 10      # level 3 = 10 scenes, like the ablation report's alternative-layout sets
TABLE_DZ = 0.04           # level 3: the table / counter / cart-box height also changes per scene, evenly spread over +-TABLE_DZ (m)


def level3_starts(n: int, seed: int) -> list[tuple[float, float]]:
    """n distinct robot start offsets (x forward, y left) spread over the threshold box, deterministic in the seed.
    The furniture stays where the kit built it; SIMPLE's dr_level 3 re-samples the robot start from the config."""
    import numpy as np
    rng = np.random.RandomState(seed)
    xs = np.linspace(-ROBOT_BACK, ROBOT_BACK, n); ys = np.linspace(-ROBOT_LEFT, ROBOT_LEFT, n)
    rng.shuffle(xs); rng.shuffle(ys)
    return [(round(float(x), 4), round(float(y), 4)) for x, y in zip(xs, ys)]


def level_env(kit: str, level: int) -> dict[str, str]:
    k = KITS[kit]; P = k["prefix"]
    layout = json.load(open(HERE / kit / "layout.json"))["layout"]
    env = {f"{P}_NUM_DISTRACTORS": str(NUM_DISTRACTORS)}
    if level >= 2:
        env[f"{P}_TARGET_JITTER"] = k["jitter"]
        if k.get("region"):
            env[f"{P}_TARGET_REGION"] = k["region"]
    return env                                                   # level 3 moves the robot start per scene (level3_starts), not the furniture


def describe(kit: str, level: int) -> str:
    return {0: "new distractors and table material", 1: "+ new lighting", 2: "+ new object pose",
            3: f"+ alternative layout: a different robot start (within {ROBOT_BACK*100:.0f} cm back/forward, {ROBOT_LEFT*100:.0f} cm sideways) and a different table/counter/cart-box height (within +-{TABLE_DZ*100:.0f} cm) every scene"}[level]


def run_level(kit: str, level: int, episodes: int, out: Path, seed: int, render_hz: int) -> None:
    """Runs inside the per-level process: the kit's env vars are already set."""
    sys.path.insert(0, str(HERE / kit))
    import importlib, numpy as np, gymnasium as gym
    from PIL import Image
    mod = importlib.import_module(f"{kit}_task")
    from simple.cli.dr_decoupled_wbc import _make_sonic_config, _init_exporter, _save_episode_env_config
    from simple.agents.pico_decoupled_agent import PicoDecoupledAgent

    env = gym.make(mod.ENV_ID, sim_mode="mujoco_isaac", render_hz=render_hz, headless=True, sonic_config=_make_sonic_config())
    sonic_env = env.unwrapped; task = sonic_env.task; robot = task.robot
    agent = PicoDecoupledAgent(task.robot)
    run_dir = out / sonic_env.spec.id.split("/")[-1] / f"dr-level-{level}"     # data/evals_new20 layout: <task id>/dr-level-<n>
    if run_dir.exists():
        shutil.rmtree(run_dir)                                  # the exporter must find no directory (it would try to resume)

    random.seed(seed); np.random.seed(seed)
    env.reset(seed=seed)
    base = task.state_dict()                                   # the base scene this level is derived from
    starts = [] if level == 3 else None                        # level 3: SIMPLE's DRManager moves the robot start per scene (SIMPLE_LEVEL3_* env vars)
    dzs = []
    exporter = None; saved = 0; obj_names = []
    t_start = time.time()
    for i in range(episodes):
        s = seed + 1000 * (level + 1) + i
        random.seed(s); np.random.seed(s)
        obs, _ = env.reset(seed=s, options={"state_dict": base, "dr_level": level})
        # settle exactly as eval_decoupled_wbc does before the policy starts: the upper body ramps to the controller's
        # default pose and the objects come to rest, so the recorded first frame is the pose the policy will see
        agent.reset(); agent._wbc_policy.lower_body_policy.use_policy_action = True
        n_settle = 0; mj = sonic_env.mujoco
        while not robot.stabilized and n_settle < 300:                  # MuJoCo only: no Isaac sync / render per step (the stabilise
            a = agent.get_stabilize_action(obs)                          # action reads the robot's proprio, not the images)
            for _ in range(sonic_env.control_decimal):
                mj.apply_action(a); mj.step(render=False)
            obs = {"joint_qpos": np.asarray(list(mj.get_robot_qpos().values()), dtype=np.float32)}
            n_settle += 1
        for _ in range(3):
            sonic_env.isaac.step(mj)                                     # mirror the settled state into Isaac, then render once
        obs = sonic_env._get_obs()
        if starts is not None:
            sp = task.state_dict()["dr_state_dict"]["spatial"]; rk = next(k for k in sp if k.startswith("g1"))
            starts.append([round(float(v), 4) for v in sp[rk]["position"][:2]])
            dzs.append(round(float(getattr(task, "_table_dz", 0.0)), 4))
        if exporter is None:
            obj_names = list(sonic_env.mujoco.mj_objects.keys())
            exporter = _init_exporter(str(run_dir), task.instruction, agent._dwbc_robot_model, obj_names, robot.joint_names, obs["head_stereo_left"].shape)
            (run_dir / "frames").mkdir(parents=True, exist_ok=True)
        vel = np.concatenate([np.asarray(obs.get("base_lin_vel", np.zeros(3))), np.asarray(obs.get("base_ang_vel", np.zeros(3)))])
        frame = {
            "observation.images.ego_view": obs["head_stereo_left"],
            "observation.state": np.asarray(obs["joint_qpos"], dtype=np.float64),
            "observation.base_pose": np.asarray(obs.get("base_pose_quat", obs.get("floating_base_pose", np.zeros(7))), dtype=np.float64),
            "observation.base_vel": np.asarray(vel, dtype=np.float64),
            "observation.eef_state": np.zeros(14, dtype=np.float64),
            "observation.img_state_delta": np.zeros(1, dtype=np.float32),
            "teleop.navigate_command": np.zeros(4, dtype=np.float64),
            "teleop.base_height_command": np.zeros(1, dtype=np.float64),
            "action": np.zeros(43, dtype=np.float64),
            "action.eef": np.zeros(14, dtype=np.float64),
            "observation.torso_rpy_command": np.zeros(3, dtype=np.float64),
        }
        if "object_poses" in obs:
            frame["observation.object_poses"] = np.asarray(obs["object_poses"], dtype=np.float64)
        elif obj_names:
            frame["observation.object_poses"] = np.zeros(len(obj_names) * 7, dtype=np.float64)
        exporter.add_frame(frame)
        time.sleep(0.5)                                          # let the async video writer drain: saving straight away races its encoder (EOFError in avcodec_send_frame)
        exporter.save_episode()
        _save_episode_env_config(exporter, task.state_dict(), saved)
        Image.fromarray(obs["head_stereo_left"]).save(run_dir / "frames" / f"ep{saved:02d}.png")
        saved += 1
        extra = f" robot start x {starts[-1][0]:+.3f} y {starts[-1][1]:+.3f}" if starts else ""
        print(f"[levels] {kit} level {level}: scene {saved}/{episodes} ({(time.time()-t_start)/saved:.0f} s each, settled in {n_settle} steps){extra}", flush=True)
    if exporter is not None:
        exporter.stop_video_writers()
    json.dump({"kit": kit, "level": level, "description": describe(kit, level), "episodes": saved, "seed": seed,
               "robot_starts": starts if starts else None, "table_dz": dzs if starts else None,
               "threshold": {"back_forward_m": ROBOT_BACK, "sideways_m": ROBOT_LEFT, "table_dz_m": TABLE_DZ} if starts else None,
               "env": {k: v for k, v in os.environ.items() if k.startswith(KITS[kit]["prefix"] + "_")},
               "eval": f"python sim/eval_scene.py {kit} {mod.ENV_ID} psi0_decoupled_wbc train --data-format lerobot --data-dir {run_dir} --port 21000 --headless --num-episodes {saved}"},
              open(run_dir / "meta" / "scene_env.json", "w"), indent=1)
    env.close(); agent.close()
    print(f"[levels] {kit} level {level}: {saved} scenes -> {run_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kit", choices=sorted(KITS))
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--episodes-level3", type=int, default=LEVEL3_EPISODES)
    ap.add_argument("--out", default="data/evals_scenes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render-hz", type=int, default=50)
    ap.add_argument("--force", action="store_true", help="regenerate levels that are already complete")
    ap.add_argument("--_level", type=int, default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if a._level is not None:
        run_level(a.kit, a._level, a.episodes, out, a.seed, a.render_hz); return
    env_id = KITS[a.kit]["env_id"].split("/")[-1]
    for lv in a.levels:
        done = out / env_id / f"dr-level-{lv}" / "meta" / "scene_env.json"
        if done.exists() and not a.force:
            print(f"[levels] {a.kit} level {lv}: already complete ({done}), skipping (--force to redo)", flush=True); continue
        n_ep = a.episodes_level3 if lv == 3 else a.episodes
        env = dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "egl"), OMNI_KIT_ACCEPT_EULA="YES")
        env.update(level_env(a.kit, lv))
        if lv == 3:
            env.update(SIMPLE_LEVEL3_EPISODES=str(n_ep), SIMPLE_LEVEL3_BACK=str(ROBOT_BACK), SIMPLE_LEVEL3_SIDE=str(ROBOT_LEFT), SIMPLE_LEVEL3_SEED=str(a.seed),
                       SIMPLE_LEVEL3_TABLE_DZ=str(TABLE_DZ))
        print(f"[levels] {a.kit} level {lv}: {describe(a.kit, lv)} | " + " ".join(f"{k}={v}" for k, v in level_env(a.kit, lv).items()), flush=True)
        cmd = [sys.executable, __file__, a.kit, "--_level", str(lv), "--episodes", str(n_ep), "--out", str(out), "--seed", str(a.seed), "--render-hz", str(a.render_hz)]
        for attempt in range(1, 4):
            r = subprocess.run(cmd, env=env)
            if r.returncode == 0 and done.exists():
                break
            print(f"[levels] {a.kit} level {lv} attempt {attempt} failed (exit {r.returncode}); the level is regenerated from scratch", flush=True)
        else:
            print(f"[levels] {a.kit} level {lv} FAILED after 3 attempts", flush=True); sys.exit(1)


if __name__ == "__main__":
    main()
