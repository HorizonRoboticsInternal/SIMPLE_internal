#!/usr/bin/env python
"""Apply a grip pose set on the tuner page (docs/revo2_pose_tuner) to the Revo 2 robot and check it with the closed-loop
replay: writes data/robots/g1_rot0_teleop_revo2/grip_poses.json (what simple.robots.g1_sonic_revo2 loads for the teleop
agent and the replays), runs scripts/holomotion_v14_checks/replay_rot0_revo2.py on a few recorded episodes with it, and
summarises into docs/revo2_pose_tuner/apply_status.json (shown by the page after a rebuild).
    .venv/bin/python scripts/holomotion_v14_checks/apply_revo2_grip_poses.py <pose.json> [--episodes 0,10,20,30,40] [--video-every N] [--no-replay] [--replay-dir <existing replay dir>]
pose.json = {"open": [11], "closed": [11], "close2": [11] (optional, the trigger's pinch pose), "threshold": 0.5, "note": "...", "set_at": "..."} (the page's JSON / the artifact's poses/current).
"""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser(); ap.add_argument("pose"); ap.add_argument("--episodes", default="0,10,20,30,40"); ap.add_argument("--no-replay", action="store_true"); ap.add_argument("--replay-dir", default="", help="summarise this existing replay directory instead of running a new replay")
ap.add_argument("--video-every", type=int, default=2); ap.add_argument("--dataset", default="data/teleop_holomotion_v14_rot0_blackhands/simple/G1WholebodyBottleBinTeleop-v0/level-3_pinhole"); ap.add_argument("--out", default="docs/revo2_pose_tuner")
a = ap.parse_args(); pose = json.load(open(a.pose)); OUT = ROOT / a.out
assert len(pose["open"]) == 11 and len(pose["closed"]) == 11, "11 joint values per pose (MJCF order)"
cfg = dict(open=[round(float(x), 4) for x in pose["open"]], closed=[round(float(x), 4) for x in pose["closed"]], threshold=float(pose.get("threshold", 0.5)),
           note=pose.get("note", ""), set_at=pose.get("set_at", ""), applied_at=time.strftime("%Y-%m-%d %H:%M"), source="docs/revo2_pose_tuner")
if pose.get("close2") is not None:                                   # the second closing pose (PICO trigger): the pinch
    assert len(pose["close2"]) == 11; cfg["close2"] = [round(float(x), 4) for x in pose["close2"]]
else:                                                                # keep the robot's current pinch
    prev = ROOT / "data/robots/g1_rot0_teleop_revo2/grip_poses.json"
    if prev.exists() and json.loads(prev.read_text()).get("close2"): cfg["close2"] = json.loads(prev.read_text())["close2"]
target = ROOT / "data/robots/g1_rot0_teleop_revo2/grip_poses.json"; target.write_text(json.dumps(cfg, indent=1)); print(f"[apply] wrote {target}", flush=True)
prev = json.loads((OUT / "apply_status.json").read_text()) if (OUT / "apply_status.json").exists() else {}
same = lambda v: all(abs(float(a) - float(b)) < 1e-3 for a, b in zip(v.get("open", []), cfg["open"])) and all(abs(float(a) - float(b)) < 1e-3 for a, b in zip(v.get("closed", []), cfg["closed"]))
status = dict(cfg, replay=None, variants=[v for v in prev.get("variants", []) if not same(v)])   # keep the trial runs listed under "Also tried", minus the one being applied
if not a.no_replay:
    if a.replay_dir: rdir = Path(a.replay_dir); rc = 0
    else:
        stamp = time.strftime("%Y%m%d_%H%M"); rdir = Path(a.out) / f"replay_{stamp}"
        cmd = [str(ROOT / ".venv/bin/python"), "scripts/holomotion_v14_checks/replay_rot0_revo2.py", a.dataset, "--episodes", a.episodes, "--out", str(rdir), "--video-every", str(a.video_every)]
        print("[apply] " + " ".join(cmd), flush=True); env = dict(os.environ, MUJOCO_GL="egl"); rc = subprocess.call(cmd, cwd=ROOT, env=env)
    sys.path.insert(0, str(ROOT / "scripts/holomotion_v14_checks")); from revo2_replay_summary import summarise
    status["replay"] = dict(rc=rc, **summarise(ROOT / rdir, ROOT)) if (ROOT / rdir / "results.json").exists() else dict(dir=str(rdir), rc=rc, episodes=0, placed=0, grasped=0, held_0p5s=0, causes=[], per_episode=[])
    print(f"[apply] replay: {status['replay']['placed']}/{status['replay']['episodes']} placed, {status['replay']['grasped']} grasped, {status['replay']['held_0p5s']} held >= 0.5 s", flush=True)
    for c in status["replay"]["causes"]: print(f"[apply]   {c['count']:3d}  {c['cause']}: {c['episodes']}", flush=True)
(OUT / "apply_status.json").write_text(json.dumps(status, indent=1)); print(f"[apply] status -> {OUT / 'apply_status.json'}", flush=True)
