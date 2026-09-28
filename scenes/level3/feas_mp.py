"""Physical feasibility of SIMPLE MP-task scenes with the motion planner that generated their training data (cuRobo,
MotionPlannerAgent, dexterous-hand grasps).  Each attempt draws one random cached grasp (Bodex.load_cached_grasps), plans
approach + lift, and executes it in MuJoCo; a scene is feasible when an attempt ends in task success.  Up to --tries
attempts per scene (batch planning > 1 is broken for bodex grasps: 'trajs' referenced before assignment).

    python feas_mp.py <LeRobot set dir> --env-id simple/G1WholebodyTabletopGraspMP-v0 [--start 0 --n 30 --tries 6] --out res.jsonl
Results are appended one JSON line per scene (resumable: scenes already in --out are skipped)."""
import argparse, json, os, sys, time, traceback

import gymnasium as gym
import numpy as np
import simple.envs  # noqa
from simple.agents.mp import MotionPlannerAgent
from simple.mp.curobo import CuRoboPlanner

ap = argparse.ArgumentParser()
ap.add_argument("level_dir"); ap.add_argument("--env-id", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--start", type=int, default=0); ap.add_argument("--n", type=int, default=10 ** 6); ap.add_argument("--tries", type=int, default=6)
ap.add_argument("--sim-mode", default="mujoco"); ap.add_argument("--max-steps", type=int, default=800); ap.add_argument("--render-hz", type=int, default=30)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shifts", default=None, help="'dz,dx,dy;dz,dx,dy;...' in cm: table(+objects) z, robot x, robot y; every scene is tested at every shift")
a = ap.parse_args()

cfgs = []
for l in open(os.path.join(a.level_dir, "meta/episodes.jsonl")):
    if l.strip():
        c = json.loads(l)["environment_config"]
        while isinstance(c, str):
            c = json.loads(c)
        cfgs.append(c)
done = set()
if os.path.exists(a.out):
    done = {(json.loads(l)["scene"], tuple(json.loads(l).get("shift_cm") or (0, 0, 0))) for l in open(a.out) if l.strip()}
SHIFTS = [tuple(float(v) for v in s.split(",")) for s in a.shifts.split(";")] if a.shifts else [(0.0, 0.0, 0.0)]
ROBOTS = ("g1_sonic", "g1_wholebody", "robot")


def shifted(c, dz, dx, dy):
    """Table (+ everything on it) up by dz, robot start moved by (dx, dy); metres."""
    import copy
    c = copy.deepcopy(c); d = c["dr_state_dict"]; sc = d.get("scene") or {}
    if sc.get("table"):
        sc["table"]["pose"]["position"][2] += dz
    for k, v in d["spatial"].items():
        if k in ROBOTS:
            v["position"][0] += dx; v["position"][1] += dy
        elif sc.get("table") and v["position"][2] > sc["table"]["pose"]["position"][2] - dz - 0.2:
            v["position"][2] += dz
    for name, act in ((c.get("layout") or {}).get("actors") or {}).items():
        pos = (act.get("pose") or {}).get("position")
        if not pos:
            continue
        if name == "robot":
            pos[0] += dx; pos[1] += dy
        elif name != "table2":
            pos[2] += dz
    return c

env = gym.make(a.env_id, sim_mode=a.sim_mode, headless=True, webrtc=False, max_episode_steps=a.max_steps, render_hz=a.render_hz)
task = env.unwrapped.task
planner = CuRoboPlanner(robot=task.robot, plan_dt=0.01, plan_batch_size=1, easy_motion_gen=False, ignore_target_collisions=False)
agent = MotionPlannerAgent(task, planner, debug=False, plan_batch_size=1)


def attempt(c):
    obs, info = env.reset(options={"state_dict": c}); steps = 0; outcome = "unknown"
    try:
        while True:
            st = agent.synthesize()
            if st is False:
                return "plan_failed", steps
            fin = False
            while True:
                try:
                    act = agent.get_action(obs, info); obs, r, term, trunc, info = env.step(act); steps += 1
                    if term or trunc:
                        outcome = "success" if term else "timeout"; fin = True; break
                except StopIteration:
                    if st == "phase_break":
                        break
                    outcome = "plan_exhausted"; fin = True; break
            if fin or st != "phase_break":
                return (outcome if outcome != "unknown" else "executed_no_success"), steps
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return "error: " + repr(e)[:120], steps
    finally:
        agent.reset()


for i, c0 in list(enumerate(cfgs))[a.start: a.start + a.n]:
  for sh in SHIFTS:
    if (i, sh) in done:
        continue
    c = shifted(c0, sh[0] / 100, sh[1] / 100, sh[2] / 100) if any(sh) else c0
    t0 = time.time(); tried = []
    for k in range(a.tries):
        np.random.seed(a.seed * 1000 + i * 17 + k)            # the grasp draw (Bodex picks np.random.randint over the cached set)
        out, steps = attempt(c); tried.append(out)
        if out == "success":
            break
    rec = {"scene": i, "shift_cm": list(sh), "feasible": tried[-1] == "success", "attempts": len(tried), "outcomes": tried, "s": round(time.time() - t0, 1)}
    print(f"scene {i} shift {sh}: {'FEASIBLE' if rec['feasible'] else 'infeasible'} after {len(tried)} attempt(s) {tried} {rec['s']} s", flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
env.close()
print("DONE", flush=True)
