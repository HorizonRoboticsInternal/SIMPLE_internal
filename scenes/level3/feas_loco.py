"""Walk-then-grasp feasibility of the locomotion teleop scenes, checked with the motion planner of their MP twins.

The two locomotion teleop tasks have MP twins in SIMPLE (same G1, AMO controller + cuRobo grasp planning):
    XMovePickTeleop        -> XMoveAndPickMP        walk to the table, grasp, lift
    LocomotionPickBetween… -> LocomotionPickBetweenTablesMP   grasp from the start, lift (the walk to table 2 is not checked)
Each teleop scene is transplanted into its twin: target, distractors, container, table(s) and robot start are copied. Like
every SIMPLE MP task the twin keeps the table top at z = 0 (the dexterous-grasp code picks the object's resting pose from
its height above z = 0) and moves the robot instead: pelvis z = TELEOP_PELVIS - teleop table top, objects z - table top. "After walking to the table" is modelled
by placing the robot at the grasp stance straight ahead of its start (lateral offset kept): the object at the
demonstrations' median distance ahead of the pelvis at lift (XMovePick 30.7 cm; p5 27.4, p95 35.9 are tried next), or at
the demonstrations' median stance at lift
including their sidestep (XMovePick 30.7 ahead / 7.8 right). The planner's own open-loop walk is not used: it
covers ~0.6 of short commanded distances. A scene is feasible when, from one of these stances, the planner grasps and the
target ends >= 5 cm above its start height.

    python scenes/level3/feas_loco.py <teleop LeRobot set> --task G1WholebodyXMovePickTeleop-v0 --out res.jsonl [--tries 2 --scenes 3 11]
Tasks: XMovePick, LocomotionPickBetweenTables, Handover (twin: G1WholebodyTabletopHandoverMP; records the right-hand
grasp and the left-hand takeover separately).
"""
import argparse, copy, json, math, os, time, traceback

import gymnasium as gym
import numpy as np
import simple.envs  # noqa
from simple.agents.mp import MotionPlannerAgent
from simple.mp.curobo import CuRoboPlanner

TELEOP_PELVIS = 0.75          # teleop pelvis height above the floor while walking / grasping (XMovePick demos, height command 0.74)
TWINS = {
    # stances: (target ahead of the pelvis, target left of the pelvis or None = keep the start's lateral offset) at the grasp.
    # Straight walks first, then the demonstrations' median sidestep stance.
    "G1WholebodyXMovePickTeleop-v0": dict(twin="simple/G1WholebodyXMoveAndPickMP-v0", kind="walk",
                                          stances=[(0.307, None), (0.274, None), (0.359, None), (0.307, -0.078)]),
    "G1WholebodyLocomotionPickBetweenTablesTeleop-v0": dict(twin="simple/G1WholebodyLocomotionPickBetweenTablesMP-v0", stances=[None], kind="static"),
    # Handover: right hand grasps + lifts 10 cm, the left hand takes it over (the twin's success: left-hand contact held), the right
    # hand opens and the left hand moves away. The teleop basket (placing into it) is not part of the twin.
    "G1WholebodyHandoverTeleop-v0": dict(twin="simple/G1WholebodyTabletopHandoverMP-v0", stances=[None], kind="handover"),
}
ROBOTS = ("g1_sonic", "g1_wholebody", "robot")

ap = argparse.ArgumentParser()
ap.add_argument("level_dir"); ap.add_argument("--task", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--start", type=int, default=0); ap.add_argument("--n", type=int, default=10 ** 6); ap.add_argument("--tries", type=int, default=4)
ap.add_argument("--max-steps", type=int, default=4000); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--stance-idx", type=int, nargs="*", default=None, help="only these entries of the task's stance list")
ap.add_argument("--scenes", type=int, nargs="*", default=None, help="only these scenes")
a = ap.parse_args()
T = TWINS[a.task]


def load(p):
    out = []
    for l in open(os.path.join(p, "meta/episodes.jsonl")):
        if l.strip():
            c = json.loads(l)["environment_config"]
            while isinstance(c, str):
                c = json.loads(c)
            out.append(c)
    return out


cfgs = load(a.level_dir)
done = {json.loads(l)["scene"] for l in open(a.out) if l.strip()} if os.path.exists(a.out) else set()

env = gym.make(T["twin"], sim_mode="mujoco", headless=True, webrtc=False, render_hz=30, max_episode_steps=a.max_steps)
task = env.unwrapped.task
env.reset(seed=0)
enc = lambda o: o.tolist() if hasattr(o, "tolist") else str(o)
TWIN_BASE = json.loads(json.dumps(task.state_dict(), default=enc))
TW_SP = TWIN_BASE["dr_state_dict"]["spatial"]
TW_RK = next(k for k in TW_SP if k in ROBOTS)
TW_PELVIS = float(TW_SP[TW_RK]["position"][2])
planner = CuRoboPlanner(robot=task.robot, plan_dt=0.01, plan_batch_size=1, easy_motion_gen=False, ignore_target_collisions=False)
agent = MotionPlannerAgent(task, planner, debug=False, plan_batch_size=1)


def transplant(c, stance):
    """Teleop scene -> twin scene state with the robot at the grasp stance; returns (state, walk distance, info)."""
    d = c["dr_state_dict"]; sp = d["spatial"]; rk = next(k for k in sp if k in ROBOTS)
    t1 = d["scene"]["table"]; top1 = float(t1["pose"]["position"][2]) + float(t1["size"][2]) / 2
    zmap = lambda z: float(z) - top1                            # table top -> 0
    pelvis = TELEOP_PELVIS - top1                               # the robot moves instead
    s = copy.deepcopy(TWIN_BASE); sd = s["dr_state_dict"]; ssp = {}
    r = sp[rk]; ssp[TW_RK] = {"position": [float(r["position"][0]), float(r["position"][1]), pelvis],
                              "quaternion": list(r.get("quaternion") or [1.0, 0.0, 0.0, 0.0])}
    sd["target"] = copy.deepcopy(d["target"])
    sd["distractors"] = copy.deepcopy(d.get("distractors") or {})
    tw_cont = (TWIN_BASE["dr_state_dict"].get("container") or None)
    if d.get("container") and "container" in sd:
        sd["container"] = copy.deepcopy(d["container"])
    for k, v in sp.items():
        if k == rk or not isinstance(v, dict):
            continue
        if str(k).startswith("container") and "container" not in sd:     # the twin has no container: leave the teleop basket out
            continue
        ssp[k] = {"position": [float(v["position"][0]), float(v["position"][1]), zmap(v["position"][2])],
                  "quaternion": list(v.get("quaternion") or [1.0, 0.0, 0.0, 0.0])}
    if tw_cont is not None and not d.get("container"):          # the twin has a container the teleop task lacks: park it far away
        ck = next((k for k in TW_SP if str(k).startswith("container")), None)
        if ck:
            ssp[ck] = {"position": [6.0, 6.0, pelvis - 0.7], "quaternion": [1.0, 0.0, 0.0, 0.0]}
    sd["spatial"] = ssp
    if T["kind"] == "handover":
        sd.pop("material", None)       # the twin's stored per-object shader list does not match the copied distractors (visual only)
    for key in ("table", "table2"):
        t = (d.get("scene") or {}).get(key)
        if t is not None and key in sd["scene"]:
            nt = copy.deepcopy(t); nt["pose"]["position"][2] = zmap(t["pose"]["position"][2]); sd["scene"][key] = nt
        elif t is None and key in sd["scene"] and key == "table2":
            sd["scene"][key]["pose"]["position"] = [8.0, 8.0, pelvis - 0.8]
    tgt = sp[str(d["target"]["uid"])]["position"]
    rel_x = float(tgt[0]) - float(r["position"][0]); rel_y = float(tgt[1]) - float(r["position"][1])
    ahead, left = (stance[0], stance[1]) if stance is not None else (None, None)
    walk = 0.0 if ahead is None else max(0.0, rel_x - ahead)
    side = 0.0 if left is None else rel_y - left                        # sidestep (+ = left) that puts the target at `left`
    ssp[TW_RK]["position"][0] = float(r["position"][0]) + walk          # arrived: ahead by the walk distance
    ssp[TW_RK]["position"][1] = float(r["position"][1]) + side
    return s, walk, {"rel_x_start": round(rel_x, 4), "rel_y": round(rel_y, 4), "walk_m": round(walk, 4), "sidestep_m": round(side, 4),
                     "table_top": round(top1, 4), "pelvis_above_table": round(TELEOP_PELVIS - top1, 4)}


SQUAT = [-0.3]


ORIG_DECOMPOSE = task.decompose


def subtasks(walk):
    from simple.datagen.subtask_spec import GraspObjectSpec, HeightAdjustSpec, LiftSpec, PhaseBreakSpec, StandSpec, WalkSpec
    if T["kind"] == "handover":
        return _handover_specs()
    g = GraspObjectSpec("approach", target_uid=task.target.uid, pregrasp=False, grasp_type="bodex", hand_uid="dex3_right", lock_links=["left_hand_palm_link"])
    # No StandSpec("initialize") before the grasp: the twin's start routine swings the arms forward and knocks an object that
    # stands at the table edge; the grasp is planned straight from the arrived pose (the reset state).
    if T["kind"] == "walk":
        return [PhaseBreakSpec("phase_break_before_pick", grasp_type="bodex"), g, LiftSpec("lift", up=0.15, grasp_type="bodex", hand_uid="dex3_right")]
    if T["kind"] == "walk_bend":
        # a 2-step stand first: the squat (keep_waist_pose) reuses the controller command a stand step sets
        return [StandSpec("stand", steps=2), HeightAdjustSpec("adjust_height", height=SQUAT[0], keep_waist_pose=True), PhaseBreakSpec("phase_break_before_pick", grasp_type="bodex"), g,
                HeightAdjustSpec("adjust_height", height=0, keep_waist_pose=True)]
    return [PhaseBreakSpec("phase_break_before_grasp", grasp_type="bodex"), g, LiftSpec("lift", up=0.08, grasp_type="bodex", hand_uid="dex3_right")]


def _handover_specs():
    from simple.datagen.subtask_spec import StandSpec
    sts = list(ORIG_DECOMPOSE())
    if sts and isinstance(sts[0], StandSpec):
        sts = sts[1:]                                                   # no start-up arm routine (the grasp starts from the reset pose)
    return sts


PLAN_OK = [False]
PB_Z = []                                                               # target z at every phase break (handover: after the right-hand lift)


def target_z():
    m = env.unwrapped.mujoco
    obj = m.mj_objects.get("target") if hasattr(m, "mj_objects") else None
    return float(obj.xpos[2]) if obj is not None else None


def pelvis_xy():
    import mujoco
    m = env.unwrapped.mujoco; b = mujoco.mj_name2id(m.mjModel, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    return [round(float(v), 4) for v in m.mjData.xpos[b][:3]]            # x, y, z (z: squat depth at the grasp)


def attempt(state, walk):
    obs, info = env.reset(options={"state_dict": state})
    task.decompose = lambda: subtasks(walk)                   # per-scene walk; the agent reads decompose() in synthesize()
    z0 = target_z(); steps = 0; outcome = "unknown"; stance = None; n_syn = 0
    try:
        while True:
            st = agent.synthesize(); n_syn += 1
            if st is False:
                return ("plan_failed" if n_syn > 1 else "plan_failed_pre"), z0, target_z(), stance
            if n_syn >= 2:
                PLAN_OK[0] = True                                   # the grasp phase was planned: a collision-free grasp exists
            fin = False
            while True:
                try:
                    act = agent.get_action(obs, info); obs, r, term, trunc, info = env.step(act); steps += 1
                    if term and T["kind"] == "handover":
                        outcome = "handover_success"; fin = True; break
                    if trunc:
                        outcome = "timeout"; fin = True; break
                except StopIteration:
                    if st == "phase_break":
                        PB_Z.append(target_z())
                        stance = stance or pelvis_xy(); break       # the stance the walk ended at, right before the grasp
                    outcome = "executed"; fin = True; break
            if fin or st != "phase_break":
                return outcome, z0, target_z(), stance
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return "error: " + repr(e)[:120], z0, target_z(), stance
    finally:
        agent.reset()


for i, c in list(enumerate(cfgs))[a.start: a.start + a.n]:
    if i in done or (a.scenes is not None and i not in a.scenes):
        continue
    t0 = time.time(); tried = []; ok = False; last = None; used = None; plan_ok = False
    for stance_d in T["stances"]:
        state, walk, meta = transplant(c, stance_d)
        SQUAT[0] = stance_d[2] if (stance_d is not None and len(stance_d) > 2) else -0.3
        if a.stance_idx is not None and T["stances"].index(stance_d) not in a.stance_idx:
            continue
        for k in range(a.tries):
            np.random.seed(a.seed * 1000 + i * 17 + k)
            PLAN_OK[0] = False; PB_Z.clear()
            out, z0, z1, stance = attempt(state, walk)
            lift = None if (z0 is None or z1 is None) else round(z1 - z0, 4)
            if T["kind"] == "handover":                                     # lift after the right-hand grasp, then the handover itself
                rlift = None if (z0 is None or len(PB_Z) < 2 or PB_Z[1] is None) else round(PB_Z[1] - z0, 4)
                ok = rlift is not None and rlift >= 0.05 and out == "handover_success"
                lift = rlift if not ok else lift
                out = out if rlift is None or rlift >= 0.05 else f"right_grasp_lift_{rlift}"
            else:
                ok = out == "executed" and lift is not None and lift >= 0.05
            plan_ok = plan_ok or PLAN_OK[0]
            tag = "" if stance_d is None else f"@{stance_d[0]:.3f}" + ("" if stance_d[1] is None else f"/{stance_d[1]:+.3f}") + (f"/sq{stance_d[2]:+.2f}" if len(stance_d) > 2 else "")
            tried.append((out if not (out == "executed" and not ok) else f"executed_lift_{lift}") + tag)
            last = {"lift_m": lift, "stance_pelvis_xy": stance, **meta}
            if ok:
                used = stance_d; break
        if ok:
            break
    rec = {"scene": i, "feasible": ok, "grasp_planned": plan_ok, "attempts": len(tried), "outcomes": tried, "stance_m": used, **(last or {}), "s": round(time.time() - t0, 1)}
    print(f"scene {i}: {'FEASIBLE' if ok else 'infeasible'} planned {plan_ok} {tried} stance {used} lift {rec.get('lift_m')} {rec['s']} s", flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
env.close()
print("DONE", flush=True)
