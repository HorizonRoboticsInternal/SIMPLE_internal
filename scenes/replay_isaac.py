#!/usr/bin/env python
"""Replay a recorded real episode inside a kit's SIMPLE level-N scene with Isaac rendering, using the kit's own
replay_in_scene.py unchanged: the env is built in mujoco_isaac mode, reset to one scene of data/evals_scenes (its
distractors, table material, lighting), and the kit's video (third person | head camera | real head camera) then
carries the Isaac head camera in the middle panel.

    python sim/replay_isaac.py bowl_sink --episode 7 --nav-gain 1.5 --out sim/bowl_sink/replay/isaac_level0_ep7.mp4
    python sim/replay_isaac.py bottle_bin --session 2026-09-17-00-16-29-G1-sim --episode 6
    python sim/replay_isaac.py coffee_cart --episode 84
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent   # the SIMPLE checkout
DATA = Path(os.environ.get("SIMPLE_DATA_DIR") or ROOT / "data")   # SIMPLE data root (datasets, recordings, scene sets)
ENV_IDS = {"bottle_bin": "G1WholebodyBottleBinTeleop-v0", "bowl_sink": "G1WholebodyBowlSinkTeleop-v0", "coffee_cart": "G1WholebodyCoffeeCartTeleop-v0"}
SESSIONS = {"bottle_bin": "2026-09-17-00-16-29-G1-sim", "bowl_sink": "psi0/BowlToSink_0918"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("kit", choices=sorted(ENV_IDS)); ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--session", default=None); ap.add_argument("--level", type=int, default=0); ap.add_argument("--scene", type=int, default=0)
ap.add_argument("--nav-gain", type=float, default=1.0); ap.add_argument("--sets", default=str(DATA / "evals_scenes"))
ap.add_argument("--out", default=None)
ap.add_argument("--sim-mode", default="mujoco_isaac", choices=("mujoco_isaac", "mujoco"), help="diagnostics: mujoco = no Isaac")
ap.add_argument("--hide-shell", action="store_true", help="third-person pass: hide the HSSD room walls/ceiling so the kit's own (outside-the-room) camera sees the scene")
ap.add_argument("--third-light", type=float, default=0.0, help="pass B: dome light intensity added for the third-person camera (0 = none)")
ap.add_argument("--export", default=None, help="write this episode as a one-episode SIMPLE LeRobot dataset (Gr00t exporter) into this directory")
ap.add_argument("--seed", type=int, default=None, help="seed numpy/random before every reset and pass it to env.reset: a fresh table/ground material and lighting per seed (no level state is injected)")
ap.add_argument("--trace-out", default=None, help="write the kit's full trace (rows + summary) as JSON here, as the kit's own worker mode does")
ap.add_argument("--summary-out", default=None, help="write the kit summary (gates) plus the robot-furniture contact statistics as JSON here")
ap.add_argument("--no-third", action="store_true", help="skip the Isaac third-person camera (the kit video keeps its MuJoCo third-person panel)")
ap.add_argument("--ext-clock", action="store_true", help="also for coffee_cart: one external virtual clock for every module (its own shim covers only the agent + teleop policy)")
ap.add_argument("--driver", default=None, choices=("replay", "sweep"), help="bottle_bin: 'sweep' drives sweep_replay.py (the tool that produced the verified successes; own virtual clock, --video-dir); default sweep for bottle_bin, replay otherwise")
ap.add_argument("--no-state", action="store_true", help="diagnostics: plain reset (the kit scene as MuJoCo replays see it) instead of the level scene")
ap.add_argument("--state-out", default=None, help="record the full MuJoCo integration state before every env.step into this .npy (a MuJoCo-only run whose result is checked)")
ap.add_argument("--state-in", default=None, help="force the MuJoCo state recorded by --state-out before every env.step: the Isaac render then shows exactly that run")
a = ap.parse_args()

set_dir = Path(a.sets) / ENV_IDS[a.kit] / f"dr-level-{a.level}"
meta = json.load(open(set_dir / "meta" / "scene_env.json"))
for k, v in meta.get("env", {}).items():
    os.environ.setdefault(k, v)                                   # the kit's level knobs (distractor count etc.)
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ["REPLAY_NAV_GAIN"] = str(a.nav_gain)
if a.ext_clock:
    os.environ["REPLAY_SIM_CLOCK"] = "0"                          # one clock only: the external one
rows = [json.loads(l) for l in open(set_dir / "meta" / "episodes.jsonl") if l.strip()]
STATE = json.loads(rows[a.scene]["environment_config"])
print(f"[isaac-replay] {a.kit} episode {a.episode} in level {a.level} scene {a.scene} ({meta['description']}); env {meta.get('env')}", flush=True)

sys.path.insert(0, str(HERE / a.kit))
import gymnasium
_orig_make = gymnasium.make


def _export_gate_open():
    """The kits open EXPORT_GATE when the recorded rows start: the stand-up settle and the first-pose hold stay out of the dataset."""
    mod = sys.modules.get("sweep_replay" if a.kit == "bottle_bin" else "replay_in_scene")
    gate = getattr(mod, "EXPORT_GATE", None)
    return gate is None or bool(gate.get("on", True))


def _make(env_id, *args, **kw):
    kw["sim_mode"] = a.sim_mode
    env = _orig_make(env_id, *args, **kw)
    orig_reset, orig_step = env.reset, env.step

    def reset(*ra, **rk):
        if a.seed is not None:                                   # per-episode look: seed the DR sampling (materials, lighting) and the env
            import random as _random; _random.seed(a.seed); np.random.seed(a.seed); rk["seed"] = a.seed
        elif rk.get("options") is None and not a.no_state:
            rk["options"] = {"state_dict": STATE}
        out = orig_reset(*ra, **rk)
        EXPORT["last"] = out                                     # (obs, info) paired with the next action
        if a.sim_mode == "mujoco_isaac" and not G1COMP["done"]:
            try: _add_g1comp_head_isaac(env)
            except Exception as exc: print(f"[isaac-replay] g1comp head: not added ({exc!r})", flush=True); G1COMP["done"] = True
        if THIRD["cam"] is None and not a.no_third and a.sim_mode == "mujoco_isaac":
            _make_third_person_camera(env)
        return out

    def step(*sa, **sk):
        if a.state_out or a.state_in:
            _state_io(env)
        if a.export and sa and _export_gate_open():
            _export_frame(env, sa[0])
        out = orig_step(*sa, **sk)
        EXPORT["last"] = (out[0], out[-1]) if isinstance(out, tuple) and len(out) >= 2 else None
        _count_contacts(env)
        if CLOCK["clock"] is not None:
            CLOCK["clock"].tick()
        if THIRD["cam"] is not None:
            _capture_third_person(out[0] if isinstance(out, tuple) and out else None)
        return out
    env.reset, env.step = reset, step
    if a.state_out:
        _close0 = env.close
        def close_save(*ca, **ck):                               # the bottle sweep os._exit()s right after env.close: save the states first
            _state_save(); return _close0(*ca, **ck)
        env.close = close_save
    if a.export:
        orig_close = env.close
        def close(*ca, **ck):                                    # the kit's env.close() (SimulationApp shutdown) segfaults the process: skip it.
            if a.kit == "bottle_bin":                            # the sweep os._exit()s right after close: save here (result read from its json);
                _export_finish(None)                             # the other kits return their summary after close, so they save in the driver
            print("[isaac-replay] export: env.close skipped (the process exits with os._exit after saving)", flush=True)
        env.close = close
    return env


gymnasium.make = _make

KIT_INSTRUCTIONS = {"bottle_bin": "pick up the bottle, turn around and drop it into the trash bin",
                    "bowl_sink": "pick up the green bowl from the counter, turn right, move towards the sink, and place the bowl in the sink",
                    "coffee_cart": "push the cart to the table, then pick up the coffee cup and place it on the table"}
EXPORT = {"agent": None, "last": None, "frames": None, "n": 0, "obj_names": None, "task": None}


def _build_frame(agent, obj_names, observation, privileged_info, action, target_waist):
    """One recording frame from the pre-step observation/privileged info and the controller action: a copy of
    simple.cli.teleop_decoupled_wbc._build_frame (that module cannot be imported inside the Isaac process)."""
    proprio = privileged_info["proprio"]; rm = agent._dwbc_robot_model
    action_q = rm.get_configuration_from_actuated_joints(body_actuated_joint_values=action["target_q"], left_hand_actuated_joint_values=action["left_hand_q"], right_hand_actuated_joint_values=action["right_hand_q"])
    action_q = np.array(action_q, dtype=np.float64); action_q[12:15] = target_waist
    eef = action["action_eef"]; eef = np.zeros(14) if eef is None else np.asarray(eef, dtype=np.float64)
    frame = {
        "observation.images.ego_view": observation["head_stereo_left"],
        "observation.state": np.asarray(observation["joint_qpos"], dtype=np.float64),
        "observation.eef_state": eef,
        "action": action_q,
        "action.eef": eef,
        "observation.img_state_delta": np.array([0.0], dtype=np.float32),
        "teleop.navigate_command": np.asarray(action["navigate_cmd"], dtype=np.float64),
        "teleop.base_height_command": np.atleast_1d(np.asarray(action["base_height_command"], dtype=np.float64)),
        "observation.base_pose": np.asarray(observation["base_pose_quat"] if "base_pose_quat" in observation else proprio["floating_base_pose"], dtype=np.float64),
        "observation.base_vel": np.asarray(np.concatenate([observation["base_lin_vel"], observation["base_ang_vel"]]) if "base_lin_vel" in observation and "base_ang_vel" in observation else proprio["floating_base_vel"], dtype=np.float64),
        "observation.torso_rpy_command": np.asarray(action["torso_rpy_cmd"], dtype=np.float64),
    }
    if obj_names:
        frame["observation.object_poses"] = np.concatenate([np.asarray(privileged_info[n], dtype=np.float64) for n in obj_names])
    return frame


def _export_frame(env, action):
    """One raw frame per env.step: the observation/privileged info the action was computed from (cached from the previous
    step or the reset) plus the controller action. Kept in memory (numbers) and as JPEGs (images); build_dataset.py turns
    the folder into a SIMPLE LeRobot dataset outside Isaac (its video encoder crashes inside the Isaac process)."""
    try:                                                         # the agent returns a simple.core.action.ActionCmd (dict-like, None for missing keys)
        has_q = action["target_q"] is not None and action["left_hand_q"] is not None
    except Exception:
        has_q = False
    if EXPORT["last"] is None or EXPORT["agent"] is None or not has_q:
        EXPORT["skipped"] = EXPORT.get("skipped", 0) + 1
        return
    import cv2
    obs, info = EXPORT["last"]
    se = env.unwrapped
    if EXPORT["frames"] is None:
        task = se.task; robot = task.robot
        EXPORT["obj_names"] = list(se.mujoco.mj_objects.keys()); EXPORT["task"] = task; EXPORT["joint_names"] = list(robot.joint_names)
        kit_mod = sys.modules.get(f"{a.kit}_task"); EXPORT["instruction"] = os.environ.get(f"{a.kit.upper()}_INSTRUCTION") or KIT_INSTRUCTIONS.get(a.kit) or getattr(kit_mod, "INSTRUCTION", None) or task.instruction
        print(f"[isaac-replay] export: obs keys {sorted(obs.keys())[:20]}; proprio keys {sorted((info.get('proprio') or {}).keys())[:20]}", flush=True)
        EXPORT["frames"] = []; Path(a.export, "images").mkdir(parents=True, exist_ok=True)
        print(f"[isaac-replay] export: {a.export} objects {EXPORT['obj_names']} image {obs['head_stereo_left'].shape} (first frame after {EXPORT.get('skipped', 0)} skipped steps)", flush=True)
    act = {k: action[k] for k in ("target_q", "left_hand_q", "right_hand_q", "action_eef", "navigate_cmd", "base_height_command", "torso_rpy_cmd", "target_waist")}
    if act["navigate_cmd"] is None or act["base_height_command"] is None or act["torso_rpy_cmd"] is None:
        try:
            goal = EXPORT["agent"]._teleop_policy.get_action()
        except Exception:
            goal = {}
        up = np.asarray(goal.get("target_upper_body_pose", np.zeros(31)), dtype=np.float64)
        if act["navigate_cmd"] is None: act["navigate_cmd"] = np.asarray(goal.get("navigate_cmd", [0.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        if act["base_height_command"] is None: act["base_height_command"] = np.atleast_1d(np.asarray(goal.get("base_height_command", 0.74), dtype=np.float64))
        if act["torso_rpy_cmd"] is None: act["torso_rpy_cmd"] = up[-3:] if up.shape[0] >= 3 else np.zeros(3)
    tw = act["target_waist"]
    if tw is None:
        rm = EXPORT["agent"]._dwbc_robot_model
        q43 = rm.get_configuration_from_actuated_joints(body_actuated_joint_values=act["target_q"], left_hand_actuated_joint_values=act["left_hand_q"], right_hand_actuated_joint_values=act["right_hand_q"])
        tw = np.asarray(q43)[12:15].copy()
    obs2 = dict(obs)                                             # the sim proprio carries no global base pose: take MuJoCo's ground truth of this state
    try:
        md = se.task.robot.mjData; obs2["base_pose_quat"] = np.asarray(md.qpos[:7], dtype=np.float64).copy()
        obs2["base_lin_vel"] = np.asarray(md.qvel[:3], dtype=np.float64).copy(); obs2["base_ang_vel"] = np.asarray(md.qvel[3:6], dtype=np.float64).copy()
    except Exception:
        pass
    frame = _build_frame(EXPORT["agent"], EXPORT["obj_names"], obs2, info, act, tw)
    k = len(EXPORT["frames"]); img = frame.pop("observation.images.ego_view")
    cv2.imwrite(str(Path(a.export, "images", f"{k:06d}.jpg")), cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    EXPORT["frames"].append({kk: np.asarray(v, dtype=np.float64 if kk != "observation.img_state_delta" else np.float32) for kk, v in frame.items()})
    EXPORT["n"] += 1


def _export_finish(result=None):
    if EXPORT["frames"] is None or EXPORT.get("finished"):
        return
    EXPORT["finished"] = True
    if result is None and a.kit == "bottle_bin":                 # the sweep wrote its result json just before closing the env
        try:
            import glob as _glob
            cands = sorted(_glob.glob(str(HERE / "bottle_bin/replay/sweep" / "*" / f"ep{a.episode:03d}.json")), key=os.path.getmtime)
            result = json.load(open(cands[-1])) if cands else None
            if result is not None: result.pop("trace", None)
        except Exception:
            result = None
    keys = list(EXPORT["frames"][0].keys()); arrays = {k: np.stack([f[k] for f in EXPORT["frames"]]) for k in keys}
    np.savez(Path(a.export, "frames.npz"), **arrays)
    try:
        env_config = json.loads(json.dumps(EXPORT["task"].state_dict(), default=str))
    except Exception as exc:
        env_config = {"error": repr(exc)}
    meta = {"state_in": a.state_in, "forced_steps": STATEIO["forced"], "state_in_problem": STATEIO["bad"], "steps": STATEIO["i"],
            "kit": a.kit, "episode": a.episode, "seed": a.seed, "frames": EXPORT["n"], "skipped_steps": EXPORT.get("skipped", 0), "sim_mode": a.sim_mode, "session": a.session,
            "instruction": EXPORT["instruction"], "obj_names": EXPORT["obj_names"], "joint_names": EXPORT["joint_names"], "image_shape": [360, 640, 3], "fps": 50,
            "scene_env": {k: v for k, v in os.environ.items() if k.startswith(("BOTTLE_BIN_", "BOWL_SINK_", "COFFEE_CART_", "REPLAY_"))}, "env_config": env_config,
            "contacts": {"steps": CONTACT.get("n"), "touching_ticks": CONTACT.get("ticks"), "first_tick": CONTACT.get("first"),
                         "pairs": sorted(((f"{k[0]}|{k[1]}", v) for k, v in (CONTACT.get("names") or {}).items()), key=lambda kv: -kv[1])[:12]}}
    if result is not None:
        try:
            meta["result"] = json.loads(json.dumps(result, default=str))
        except Exception:
            meta["result"] = str(result)[:2000]
    Path(a.export, "replay_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(f"[isaac-replay] export: saved {EXPORT['n']} raw frames ({EXPORT.get('skipped', 0)} steps skipped) -> {a.export} (build the dataset with build_dataset.py)", flush=True)


if a.export:
    import simple.agents.pico_decoupled_agent as _pda_e
    _AgentE = _pda_e.PicoDecoupledAgent

    class _ExportAgent(_AgentE):
        def __init__(self, *aa, **kk):
            super().__init__(*aa, **kk); EXPORT["agent"] = self
    _pda_e.PicoDecoupledAgent = _ExportAgent

STATEIO = {"i": 0, "rec": [], "arr": None, "forced": 0, "saved": False, "bad": None}


def _state_io(env):
    """2026-09-29 exact-state Isaac render of a validated physical MuJoCo run: --state-out records the full integration state
    (qpos, qvel, act, ctrl, warmstart, time...) before every env.step; --state-in sets it back before the same step index, so the
    physics steps from the recorded state and the Isaac frame after the step shows the recorded run (no drift, no lost grasp)."""
    import mujoco
    se = env.unwrapped; m, d = se.task.robot.mjModel, se.task.robot.mjData; spec = mujoco.mjtState.mjSTATE_INTEGRATION
    n = mujoco.mj_stateSize(m, spec); i = STATEIO["i"]; STATEIO["i"] += 1
    if a.state_out:
        buf = np.empty(n); mujoco.mj_getState(m, d, buf, spec); STATEIO["rec"].append(buf)
    if a.state_in:
        if STATEIO["arr"] is None:
            STATEIO["arr"] = np.load(a.state_in)
            if STATEIO["arr"].shape[1] != n:
                # 2026-09-29 g1comp: the model has joint-less head bodies the recording (stock G1) did not -> same qpos / qvel /
                # ctrl / warmstart; only the per-body applied-force block (6 per body, all zero in these replays) is wider
                arr = STATEIO["arr"]; dn = n - arr.shape[1]; S = mujoco.mjtState
                off = sum(mujoco.mj_stateSize(m, f) for f in (S.mjSTATE_TIME, S.mjSTATE_QPOS, S.mjSTATE_QVEL, S.mjSTATE_ACT, S.mjSTATE_WARMSTART, S.mjSTATE_CTRL, S.mjSTATE_QFRC_APPLIED))
                nx = mujoco.mj_stateSize(m, S.mjSTATE_XFRC_APPLIED)
                # bodies the recording did not have: the g1comp head chain and any static kit asset added since (2026-10-14: the
                # coffee wall); their applied-force blocks are zero, so the recorded blocks map onto the remaining bodies in order
                head = {"head_servo_link", "xl330_link", "d455_link"} | {b for b in os.environ.get("REPLAY_STATE_NEW_BODIES", "wall").split(",") if b}
                keep = [i for i in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "") not in head]
                if dn > 0 and dn % 6 == 0 and nx - dn >= 0 and 6 * len(keep) == nx - dn:
                    xo = arr[:, off:off + nx - dn]                 # recorded per-body forces, old body order = new order without the head
                    xn = np.zeros((len(arr), m.nbody, 6)); xn[:, keep, :] = xo.reshape(len(arr), len(keep), 6)
                    if os.environ.get("REPLAY_STATE_ZERO_XFRC", "0") == "1":
                        xn[:] = 0.0
                    STATEIO["arr"] = np.concatenate([arr[:, :off], xn.reshape(len(arr), nx), arr[:, off + nx - dn:]], axis=1)
                    hit = sorted({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in np.flatnonzero(np.abs(xn).reshape(len(arr), m.nbody, 6).max(axis=(0, 2)) > 0)})
                    print(f"[isaac-replay] state-in: mapped {arr.shape[1]} -> {n} (+{dn // 6} body, recorded applied forces max {float(np.abs(xo).max()) if xo.size else 0.0:.3g} on {hit}; new bodies {sorted(head & set(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody)))}; order {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(m.nbody) if i == 0 or i > m.nbody - 8]})", flush=True)
                else:
                    STATEIO["bad"] = f"state size {STATEIO['arr'].shape[1]} != model {n}"; print(f"[isaac-replay] STATE-IN DISABLED: {STATEIO['bad']}", flush=True)
        if STATEIO["bad"] is None and i < len(STATEIO["arr"]):
            mujoco.mj_setState(m, d, STATEIO["arr"][i], spec); STATEIO["forced"] += 1


def _state_save():
    if a.state_out and not STATEIO["saved"] and STATEIO["rec"]:
        Path(a.state_out).parent.mkdir(parents=True, exist_ok=True); np.save(a.state_out, np.stack(STATEIO["rec"])); STATEIO["saved"] = True
        print(f"[isaac-replay] states: {len(STATEIO['rec'])} steps -> {a.state_out}", flush=True)


CONTACT = {"ticks": 0, "n": 0, "first": None, "names": {}, "ids": None}
FURNITURE = ("table", "counter", "sink_counter", "cart", "trash_bin", "cover_board", "table_legs")


def _count_contacts(env):
    """Ticks in which any robot geom touches the furniture (the SIMPLE table or the kit's counters / cart / bin)."""
    import mujoco
    se = env.unwrapped; robot = se.task.robot; m, d = robot.mjModel, robot.mjData
    if CONTACT["ids"] is None or CONTACT["ids"][0] is not m:
        furn, rob = set(), set()
        for b in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            root = b
            while m.body_parentid[root] != 0 and m.body_parentid[root] != root:
                root = m.body_parentid[root]
            rname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, root) or ""
            if any(name == f or name.startswith(f) for f in FURNITURE) or any(rname == f or rname.startswith(f) for f in FURNITURE):
                furn.add(b)
            elif "pelvis" in rname or "torso" in rname or rname.endswith("_link") or "pelvis" in name:
                rob.add(b)
        CONTACT["ids"] = (m, furn, rob)
    _, furn, rob = CONTACT["ids"]
    CONTACT["n"] += 1; hit = False
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in rob and b2 in furn) or (b2 in rob and b1 in furn):
            rb = b1 if b1 in rob else b2; fb = b2 if b1 in rob else b1
            key = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, rb), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, fb))
            CONTACT["names"][key] = CONTACT["names"].get(key, 0) + 1; hit = True
    if hit:
        CONTACT["ticks"] += 1
        if CONTACT["first"] is None:
            CONTACT["first"] = CONTACT["n"]; print(f"[isaac-replay] first robot-furniture contact at tick {CONTACT['n']}", flush=True)
    if CONTACT["n"] % 250 == 0:
        top = sorted(CONTACT["names"].items(), key=lambda kv: -kv[1])[:4]
        print(f"[isaac-replay] contacts: {CONTACT['ticks']} of {CONTACT['n']} ticks touching furniture; top pairs {top}", flush=True)


THIRD = {"cam": None, "n": 0, "k": 0, "path": None}
THIRD_IN_ROOM = {}   # the kits' own third-person pose (the room shell is hidden for this pass instead of moving the camera)   # (distance factor, elevation) per kit, chosen from third_pose_probe.py


def _third_params():
    """The kit's third-person MuJoCo free camera (lookat / distance / azimuth / elevation). REPLAY_THIRD_CLOSEUP=1: a close-up of
    the robot's head at the start pose (front-left, 0.9 m) to check the robot model."""
    if os.environ.get("REPLAY_THIRD_CLOSEUP") == "1":
        return dict(lookat=[0.05, 0.0, 1.22], distance=0.9, azimuth=210.0, elevation=-12.0)
    mod = sys.modules.get("replay_in_scene") or sys.modules.get("sweep_replay")
    if mod is None or not hasattr(mod, "THIRD"):
        import importlib
        mod = importlib.import_module("replay_in_scene")
    return dict(mod.THIRD)


ROOM_SHELL_PATTERNS = ("walls", "ceilings", "openings")      # prim-name substrings hidden by --hide-shell (see isaac_smoke/room_prims_probe.py)


def _hide_room_shell():
    """Hide the HSSD room's walls / ceilings / openings so the kit's own (outside-the-room) camera sees the scene. The HSSD
    floor is a child of the `walls` group, so the group prims are not hidden as a whole: their children are hidden one by
    one and flat ones (world bbox thinner than 5 cm = floor slabs) are kept (2026-09-24: the third-person views sat on black)."""
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdGeom, Usd
    st = get_current_stage(); n = 0; kept = []
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    for p in st.Traverse():
        path = str(p.GetPath())
        if not path.startswith("/World/scene"):
            continue
        name = p.GetName().lower()
        if not (any(k in name for k in ROOM_SHELL_PATTERNS) and p.IsA(UsdGeom.Imageable)):
            continue
        kids = [c for c in p.GetChildren() if c.IsA(UsdGeom.Imageable)]
        if not kids:
            UsdGeom.Imageable(p).MakeInvisible(); n += 1; continue
        for c in kids:
            try:
                r = bb.ComputeWorldBound(c).ComputeAlignedRange()
                flat = (r.GetMax()[2] - r.GetMin()[2]) < 0.05 and r.GetMax()[2] < 0.1     # a floor slab: thin AND at ground level (ceiling slabs are thin too)
            except Exception:
                flat = False
            if flat:
                kept.append(c.GetName())
            else:
                UsdGeom.Imageable(c).MakeInvisible(); n += 1
    print(f"[isaac-replay] hid {n} room shell prims ({ROOM_SHELL_PATTERNS}); kept flat (floor) prims {kept}", flush=True)


def _add_third_person_light(intensity: float):
    """Pass B only: a dome light so the third-person camera (up to 7 m away, outside the SIMPLE table lights) sees the
    whole route. The head panel of the final video comes from pass A, which is unaffected."""
    if intensity <= 0:
        return
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdLux, Sdf
    st = get_current_stage(); lp = "/World/third_person_light"
    if st.GetPrimAtPath(lp):
        st.RemovePrim(lp)
    UsdLux.DomeLight.Define(st, Sdf.Path(lp)).CreateIntensityAttr(float(intensity))
    print(f"[isaac-replay] third-person dome light {intensity:.0f}", flush=True)


def _add_g1comp_head_isaac(env):
    """2026-09-29 g1comp: the Isaac robot USD is SIMPLE's stock G1; add the g1comp head chain (servo mast -> XL330 -> D455) as
    visual meshes under its torso_link prim, at the MuJoCo head chain's fixed transforms (so they move with the torso and the
    render shows the g1comp). The head camera already follows the MuJoCo d455_link camera."""
    import mujoco, trimesh
    from pxr import UsdGeom, UsdShade, Gf, Sdf, Vt
    from omni.isaac.core.utils.stage import get_current_stage
    se = env.unwrapped; m = se.task.robot.mjModel
    if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "d455_link") < 0:
        return
    st = get_current_stage()
    if any(p.GetName() == "d455_link" and "/Robot" in str(p.GetPath()) for p in st.Traverse()):
        print("[isaac-replay] g1comp head: already in the Isaac robot (simple.robots.g1comp_sonic)", flush=True); return
    torso = [p for p in st.Traverse() if p.GetName() == "torso_link" and "/Robot" in str(p.GetPath())]
    if not torso:
        print("[isaac-replay] g1comp head: no torso_link prim found", flush=True); return
    parent = torso[0].GetPath()
    from simple.utils import resolve_data_path
    mesh_dir = Path(resolve_data_path("robots/g1/meshes"))
    colors = {"head_servo_link": (0.2, 0.2, 0.2), "xl330_link": (0.2, 0.2, 0.2), "d455_link": (0.9, 0.9, 0.9)}
    for name in ("head_servo_link", "xl330_link", "d455_link"):
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name); pos, q = m.body_pos[b], m.body_quat[b]
        xp = Sdf.Path(f"{parent}/g1comp_{name}"); xf = UsdGeom.Xform.Define(st, xp)
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in pos])); xf.AddOrientOp().Set(Gf.Quatf(float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        tm = trimesh.load(str(mesh_dir / f"{name}.STL"), force="mesh")
        mesh = UsdGeom.Mesh.Define(st, xp.AppendChild("mesh"))
        mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in tm.vertices]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tm.faces))); mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([int(i) for i in tm.faces.ravel()]))
        mesh.CreateSubdivisionSchemeAttr("none"); mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*colors[name])]))
        mat = UsdShade.Material.Define(st, xp.AppendChild("look")); sh = UsdShade.Shader.Define(st, xp.AppendChild("look").AppendChild("shader"))
        sh.CreateIdAttr("UsdPreviewSurface"); sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colors[name]))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5); mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        parent = xp
    print(f"[isaac-replay] g1comp head: servo mast, XL330 and D455 meshes added under {torso[0].GetPath()}", flush=True)
    G1COMP["done"] = True


G1COMP = {"done": False}


def _make_third_person_camera(env):
    """An Isaac camera at the pose of the kit's MuJoCo third-person camera (same look-at, distance, angles, 45 deg fovy)."""
    import mujoco, numpy as np, transforms3d as t3d
    from omni.isaac.sensor import Camera
    se = env.unwrapped; robot = se.task.robot; m, d = robot.mjModel, robot.mjData
    P = _third_params()
    fac, elev = THIRD_IN_ROOM.get(a.kit, (1.0, None))          # the kits' MuJoCo camera stands outside the HSSD room; pull it inside
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = P["lookat"]; cam.distance = P["distance"] * fac; cam.azimuth = P["azimuth"]; cam.elevation = P["elevation"] if elev is None else elev
    scn = mujoco.MjvScene(m, maxgeom=2000)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    c0, c1 = scn.camera[0], scn.camera[1]
    pos = (np.asarray(c0.pos) + np.asarray(c1.pos)) / 2.0; fwd = np.asarray(c0.forward, dtype=float); up = np.asarray(c0.up, dtype=float)
    fwd /= np.linalg.norm(fwd); up -= fwd * np.dot(up, fwd); up /= np.linalg.norm(up); right = np.cross(fwd, up)
    R = np.stack([right, up, -fwd], axis=1)                       # USD camera: +x right, +y up, looks along -z
    quat = t3d.quaternions.mat2quat(R)
    W, H = 640, 480; fovy = float(m.vis.global_.fovy) if m.vis.global_.fovy > 0 else 45.0
    hfov = 2 * np.degrees(np.arctan(np.tan(np.radians(fovy) / 2) * W / H))
    c = Camera(prim_path="/World/third_person_cam", name="third_person_cam", resolution=(W, H))
    c.initialize()
    c.set_world_pose(pos, quat, camera_axes="usd")
    f = 10.0; c.set_focal_length(f); c.set_horizontal_aperture(2 * f * np.tan(np.radians(hfov) / 2)); c.set_vertical_aperture(2 * f * np.tan(np.radians(fovy) / 2))
    c.set_clipping_range(0.05, 50.0)
    THIRD["cam"] = c; THIRD["pos"] = pos.tolist(); THIRD["quat"] = quat.tolist()
    if a.hide_shell:
        _hide_room_shell()
    _add_third_person_light(a.third_light)
    print(f"[isaac-replay] third-person Isaac camera at {np.round(pos, 2).tolist()} fovy {fovy:.0f} deg (hfov {hfov:.0f})", flush=True)


def _capture_third_person(obs=None):
    """Save the Isaac third-person frame AND the head-camera frame of the same tick as JPEGs (cv2: PIL's encoders are broken
    inside the Isaac process; the process dies in Isaac's shutdown, so nothing can be post-processed here). Every tick:
    compose_third.py aligns the kit video (written every 2nd tick once the replay starts) structurally: kit frame k = tick S + 2k, S = M - 2N."""
    import numpy as np, traceback
    THIRD["n"] += 1
    try:
        rgba = THIRD["cam"].get_rgba()
        if rgba is None or getattr(rgba, "ndim", 0) != 3 or rgba.shape[0] == 0:
            fr = THIRD["cam"].get_current_frame(); rgba = fr.get("rgba") if isinstance(fr, dict) else None
        if rgba is None or getattr(rgba, "ndim", 0) != 3 or rgba.shape[0] == 0:
            if THIRD["n"] <= 10: print(f"[isaac-replay] third-person tick {THIRD['n']}: no frame", flush=True)
            return
        if THIRD["path"] is None:
            THIRD["path"] = HERE / a.kit / "replay" / f"third_{a.kit}_ep{a.episode}"
            THIRD["path"].mkdir(parents=True, exist_ok=True)
            for old in THIRD["path"].glob("*.jpg"):
                old.unlink()
            THIRD["k"] = 0
        import cv2
        img = np.array(rgba, dtype=np.uint8)[..., :3].copy()
        cv2.imwrite(str(THIRD["path"] / f"f{THIRD['k']:05d}.jpg"), np.ascontiguousarray(img[..., ::-1]), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        head = None
        if isinstance(obs, dict) and "head_stereo_left" in obs:
            head = np.asarray(obs["head_stereo_left"])
            cv2.imwrite(str(THIRD["path"] / f"h{THIRD['k']:05d}.jpg"), np.ascontiguousarray(head[..., ::-1]), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        THIRD["k"] += 1
        if THIRD["n"] <= 6 or THIRD["n"] % 500 == 0:
            print(f"[isaac-replay] third-person tick {THIRD['n']}: saved frame {THIRD['k']} {img.shape} head {None if head is None else head.shape}", flush=True)
    except BaseException as exc:  # noqa: BLE001
        print(f"[isaac-replay] third-person capture failed at tick {THIRD['n']}: {exc!r}", flush=True)
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()


def _close_third_person():
    if THIRD["path"] is None:
        return None
    print(f"[isaac-replay] third-person frames: {THIRD.get('k', 0)} -> {THIRD['path']}", flush=True)
    return THIRD["path"]


def compose_final(kit_video: Path, third_video: Path, out: Path, poster: Path, level: int):
    """Final video: Isaac third person | Isaac head camera (the kit video's middle panel) | real head camera (its right panel)."""
    import av, numpy as np
    from PIL import Image, ImageDraw, ImageFont
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception: font = ImageFont.load_default()
    thirds = []
    with av.open(str(third_video)) as tc:
        for fr in tc.decode(video=0): thirds.append(fr.to_image().convert("RGB"))
    with av.open(str(kit_video)) as ic, av.open(str(out), "w") as oc:
        vi = ic.streams.video[0]; n_kit = vi.frames or 0
        vs = oc.add_stream("libx264", rate=vi.average_rate or 25); vs.pix_fmt = "yuv420p"; vs.options = {"crf": "23", "preset": "veryfast", "movflags": "+faststart"}
        vs.width, vs.height = vi.codec_context.width, vi.codec_context.height
        frames = [fr.to_image().convert("RGB") for fr in ic.decode(video=0)]
        N, M = len(frames), len(thirds)
        for k, im in enumerate(frames):
            W = im.width // 3; H = im.height
            if M:
                t = thirds[min(M - 1, round(k * (M - 1) / max(N - 1, 1)))].resize((W, H)); im.paste(t, (0, 0))
            d = ImageDraw.Draw(im)
            d.rectangle((4, 30, 300, 58), fill=(0, 0, 0)); d.text((10, 36), f"Isaac third person (level {level})", fill=(255, 235, 80), font=font)
            d.rectangle((W + 4, 30, W + 260, 58), fill=(0, 0, 0)); d.text((W + 10, 36), f"Isaac D455 (sim, level {level})", fill=(255, 235, 80), font=font)
            if k == 0: im.save(poster, quality=85)
            for pkt in vs.encode(av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")): oc.mux(pkt)
        for pkt in vs.encode(): oc.mux(pkt)
    print(f"[isaac-replay] composed {N} frames (third-person stream {M}) -> {out}", flush=True)

CLOCK = {"clock": None}

if a.ext_clock or (a.kit in ("bottle_bin", "bowl_sink") and (a.driver or ("sweep" if a.kit == "bottle_bin" else "replay")) == "replay"):   # coffee_cart has its own sim clock (fast=True): leave it alone; the other two pace on the wall clock
    # these two replays pace on the wall clock; under Isaac's slow steps the controller must run on a virtual 50 Hz clock
    # instead (the kits' own --fast mode does that but records no video). Install bowl_sink's VirtualClock once the agent
    # (and decoupled_wbc) exist, and tick it after every env.step.
    import importlib, importlib.util
    import simple.agents.pico_decoupled_agent as _pda
    _Agent = _pda.PicoDecoupledAgent

    class ClockedAgent(_Agent):
        def __init__(self, *aa, **kk):
            super().__init__(*aa, **kk)
            VC = getattr(sys.modules.get("replay_in_scene"), "VirtualClock", None)
            if VC is None:                                        # bottle_bin has no VirtualClock: borrow bowl_sink's
                spec = importlib.util.spec_from_file_location("bowl_sink_replay", HERE / "bowl_sink" / "replay_in_scene.py")
                m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); VC = m.VirtualClock
            CLOCK["clock"] = VC(); patched = CLOCK["clock"].install()
            R = sys.modules.get("replay_in_scene")
            if R is not None and getattr(R, "time", None) is not None:
                R.time = CLOCK["clock"]; patched.append("replay_in_scene.time")     # the kit's settle (2 s) and pacing now count ticks, not wall time
            print(f"[isaac-replay] virtual clock installed in {len(patched)} modules", flush=True)
    _pda.PicoDecoupledAgent = ClockedAgent

import importlib
driver = a.driver or ("sweep" if a.kit == "bottle_bin" else "replay")
if driver == "sweep":
    # sweep_replay.py: virtual clock built in, records the same 3-panel video with --video-dir; run its main() with our
    # patched gymnasium.make (Isaac + level scene) and the sweep's verified scene settings (feet 0.03, bottle 0.5 kg)
    SW = importlib.import_module("sweep_replay")
    vd = Path(a.out).parent if a.out else HERE / a.kit / "replay"
    tag = f"isaac_level{a.level}s{a.scene}" + ("" if a.sim_mode == "mujoco_isaac" else "_mujoco") + ("_nostate" if a.no_state else "")
    # 2026-09-27: drive the sweep through a JOBS file exactly like the fit rounds (bottle teleported after the reset, bin moved
    # by body_pos) instead of --bottle-xy/--bin-xy (scene built with them): the two paths differ by ~1-2 cm at the release
    # (chaotic walk), which cost ~10 % of the fitted episodes in the export. The fitted geometry reaches here through the env.
    _job = {"ep": int(a.episode), "grasp_assist": os.environ.get("REPLAY_GRASP_ASSIST") == "1", "drop_assist": os.environ.get("REPLAY_DROP_ASSIST") == "1"}
    if os.environ.get("BOTTLE_BIN_BOTTLE_XY"): _job["bottle_xy"] = [float(v) for v in os.environ["BOTTLE_BIN_BOTTLE_XY"].split(",")]
    if os.environ.get("BOTTLE_BIN_BIN_XY"): _job["bin_xy"] = [float(v) for v in os.environ["BOTTLE_BIN_BIN_XY"].split(",")]
    # 2026-09-29: JOB_* = placed by the job only (scene built with the layout defaults, exactly like the fit sweeps that validated
    # the physical runs; building the scene with the fitted bin/bottle changes the contact order and the drop into the bin)
    if os.environ.get("BOTTLE_BIN_JOB_BOTTLE_XY"): _job["bottle_xy"] = [float(v) for v in os.environ["BOTTLE_BIN_JOB_BOTTLE_XY"].split(",")]
    if os.environ.get("BOTTLE_BIN_JOB_BIN_XY"): _job["bin_xy"] = [float(v) for v in os.environ["BOTTLE_BIN_JOB_BIN_XY"].split(",")]
    if os.environ.get("BOTTLE_BIN_BIN_YAW"): _job["bin_yaw"] = float(os.environ["BOTTLE_BIN_BIN_YAW"])
    vd.mkdir(parents=True, exist_ok=True); _jobs_path = vd / f"jobs_ep{a.episode}_{tag}.json"; _jobs_path.write_text(json.dumps([_job]))
    sys.argv = ["sweep_replay.py", "--session", a.session or SESSIONS[a.kit], "--tag", tag, "--jobs", str(_jobs_path), "--shard", "0/1",
                "--video-dir", str(vd), "--robot-to-edge", os.environ.get("BOTTLE_BIN_ROBOT_TO_EDGE", "0.03"), "--bottle-mass", "0.5"]
    print(f"[isaac-replay] driver sweep_replay: {' '.join(sys.argv[1:])}", flush=True)
    try:
        SW.main()
    except SystemExit:
        pass
    _sw_res = None
    try:
        _sw_res = json.load(open(SW.OUT / tag / f"ep{a.episode:03d}.json"))
    except Exception:
        pass
    _state_save(); _export_finish(_sw_res)
    kit_video = vd / f"replay_{(a.session or SESSIONS[a.kit]).replace('/', '_')}_ep{a.episode:03d}_{tag}.mp4"
    if not kit_video.exists():
        kit_video = next(iter(sorted(vd.glob(f"replay_*_ep{a.episode:03d}_{tag}.mp4")) + sorted(vd.glob(f"replay_*_ep{a.episode}_{tag}.mp4"))), kit_video)
    _close_third_person()
    print(f"[isaac-replay] done -> {kit_video}", flush=True)
    sys.stdout.flush(); sys.stderr.flush()
    if a.export:
        os._exit(0)                                              # no Isaac teardown (it segfaults); everything is on disk
    sys.exit(0)
R = importlib.import_module("replay_in_scene")
suffix = ("" if a.sim_mode == "mujoco_isaac" else "_mujoco") + ("_nostate" if a.no_state else "") + ("_extclock" if a.ext_clock else "")
out = Path(a.out) if a.out else HERE / a.kit / "replay" / f"isaac_level{a.level}_ep{a.episode}{suffix}.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
if a.kit == "coffee_cart":
    res = R.replay_episode(a.episode, out, fast=True, video=True)
else:
    if hasattr(R, "NAV_GAIN"):
        R.NAV_GAIN = a.nav_gain
    res = R.replay_episode(a.session or SESSIONS[a.kit], a.episode, out)
_summary = next((r["summary"] for r in res if isinstance(r, dict) and "summary" in r), None) if isinstance(res, list) else res
_state_save(); _export_finish(_summary)
if a.state_in: print(f"[isaac-replay] state-in: forced {STATEIO['forced']} of {STATEIO['i']} steps ({STATEIO['bad'] or 'ok'})", flush=True)
if a.trace_out and isinstance(res, list):
    try:
        Path(a.trace_out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.trace_out, "w"), default=str)
        print(f"[isaac-replay] trace written -> {a.trace_out} ({len(res)} rows)", flush=True)
    except Exception as exc:
        print(f"[isaac-replay] trace not written: {exc!r}", flush=True)
if a.summary_out:
    try:
        Path(a.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.summary_out).write_text(json.dumps({"summary": _summary, "contacts": {"steps": CONTACT.get("n"), "touching_ticks": CONTACT.get("ticks"), "first_tick": CONTACT.get("first"),
                                                   "pairs": sorted(((f"{k[0]}|{k[1]}", v) for k, v in (CONTACT.get("names") or {}).items()), key=lambda kv: -kv[1])[:8]},
                                                   "kit": a.kit, "episode": a.episode}, default=str))
        print(f"[isaac-replay] summary written -> {a.summary_out}", flush=True)
    except Exception as exc:
        print(f"[isaac-replay] summary not written: {exc!r}", flush=True)
_close_third_person()
print(f"[isaac-replay] done -> {out}", flush=True)
try:
    print("[isaac-replay] summary:", json.dumps(res, default=str)[:600], flush=True)
except Exception:
    pass
if a.export:
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)
