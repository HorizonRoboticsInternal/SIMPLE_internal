#!/usr/bin/env python3
"""Export stage: replay the fitted, successful episodes in Isaac (MuJoCo physics, Isaac head camera) with a different table/ground
material and lighting per episode (seed = 1000 + episode), each into its own one-episode SIMPLE LeRobot dataset:
    <out>/<kit>/ep_<N>/   (merge them afterwards with merge_lerobot.py)

    python sim/export_batch.py bowl_sink   --fits <fits.json> --results <fitbatch.jsonl> --jobs 3
    python sim/export_batch.py coffee_cart --fits <fits.json> --results <fitbatch.jsonl> --jobs 3
    python sim/export_batch.py bottle_bin  --jobs-json <jobs.json> --results-dir sim/bottle_bin/replay/sweep/<tag> --jobs 3
"""
import argparse, json, os, subprocess, sys, time, glob
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent; PY = sys.executable; ROOT = HERE.parent   # the SIMPLE checkout
DATA = Path(os.environ.get("SIMPLE_DATA_DIR") or ROOT / "data")   # SIMPLE data root (datasets, recordings, scene sets)
LOGS = Path(os.environ.get("REPLAY_LOG_DIR") or DATA / "replay_logs")   # run logs / per-run videos (not tracked)
SESSIONS = {"bottle_bin": "psi0/ChipCanToTrash_0225_keep97", "bowl_sink": "psi0/BowlToSink_0918", "coffee_cart": "psi0/CartCoffeeCup_0919"}


def env_for(kit, p, ep=None):
    env = dict(os.environ, MUJOCO_GL="egl", OMNI_KIT_ACCEPT_EULA="YES")
    if kit == "bowl_sink":
        env["REPLAY_GRASP_ASSIST"] = "1" if p.get("grasp_assist") else "0"
        if p.get("sink_recess") is not None: env["BOWL_SINK_SINK_RECESS"] = f"{p['sink_recess']:.4f}"
        env.update(BOWL_SINK_ROBOT_TO_EDGE=f"{p['robot_to_edge']:.4f}", BOWL_SINK_BOWL_XY=f"{p['bowl_xy'][0]:.4f},{p['bowl_xy'][1]:.4f}",
                   BOWL_SINK_SINK_ALONG=f"{p['sink_along']:.4f}", BOWL_SINK_ROBOT_DY=f"{p.get('robot_dy', 0.0):.4f}", BOWL_SINK_NUM_DISTRACTORS="0")
    elif kit == "coffee_cart":
        env.update(COFFEE_CART_CUP_XY=f"{p['cup_xy'][0]:.4f},{p['cup_xy'][1]:.4f}", COFFEE_CART_ROUTE_FORWARD=f"{p['route_forward']:.4f}",
                   COFFEE_CART_TABLE_FROM_LINE=f"{p['table_from_line']:.4f}", COFFEE_CART_ROBOT_DY=f"{p.get('robot_dy', 0.0):.4f}", COFFEE_CART_NUM_DISTRACTORS="0")
        if p.get("robot_to_cart") is not None:
            env["COFFEE_CART_ROBOT_TO_CART"] = f"{p['robot_to_cart']:.4f}"
        # the wrapper runs the kit's replay_episode directly (not its main()), so the replay's own env switches must be set here
        env.update(REPLAY_CART_ATTACH=p.get("cart_attach", "none"), REPLAY_WAIST_MODE="freeze_while_walking", REPLAY_NAV_GAIN=f"{p.get('nav_gain', 1.0):.3f}",
                   REPLAY_WALK_STRETCH="1.000", REPLAY_POST_TURN_GAIN="1.000")
        if p.get("hand_delta"): env["REPLAY_HAND_DELTA_R"] = ",".join(f"{v:.4f}" for v in p["hand_delta"])
        env["REPLAY_GRASP_ASSIST"] = "1" if p.get("grasp_assist") else "0"
        if p.get("box_from_handle") is not None: env["COFFEE_CART_BOX_FROM_HANDLE"] = f"{p['box_from_handle']:.4f}"
        env["REPLAY_CART_PARK"] = "1" if p.get("cart_park") else "0"
        env["REPLAY_CUP_PIN_PLACED"] = "1" if p.get("cup_pin") else "0"
        try:                                                      # 2026-09-28: per-episode lighting level (70 % bright / 30 % dim, fixed draw)
            lv = json.load(open(HERE / "coffee_cart/replay/light_levels.json"))["levels"]
            if ep is not None and str(ep) in lv: env["COFFEE_CART_LIGHT_LEVEL"] = lv[str(ep)]
        except FileNotFoundError:
            pass
    if kit in ("bowl_sink", "bottle_bin"):                        # 2026-09-29: the same 70 % bright / 30 % dim route lighting as coffee_cart
        try:
            lv = json.load(open(HERE / f"{kit}/replay/light_levels.json"))["levels"]
            if ep is not None and str(ep) in lv: env[f"{kit.upper()}_LIGHT_LEVEL"] = lv[str(ep)]
        except FileNotFoundError:
            pass
    if kit == "bottle_bin":
        env.update(BOTTLE_BIN_BOTTLE_XY=f"{p['bottle_xy'][0]:.4f},{p['bottle_xy'][1]:.4f}", BOTTLE_BIN_NUM_DISTRACTORS="0")
        env["REPLAY_GRASP_ASSIST"] = "1" if p.get("grasp_assist") else "0"
        env["REPLAY_DROP_ASSIST"] = "1" if p.get("drop_assist") else "0"
        if p.get("bin_yaw") is not None: env["BOTTLE_BIN_BIN_YAW"] = f"{float(p['bin_yaw']):.1f}"
        if p.get("bin_xy"):
            env["BOTTLE_BIN_BIN_XY"] = f"{p['bin_xy'][0]:.4f},{p['bin_xy'][1]:.4f}"
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kit"); ap.add_argument("--fits", default=None); ap.add_argument("--results", default=None, help="fitbatch jsonl: only successful episodes are exported")
    ap.add_argument("--jobs-json", default=None); ap.add_argument("--results-dir", default=None, help="bottle_bin: sweep result dir (ep###.json with success)")
    ap.add_argument("--out", default=str(DATA / "replay_datasets")); ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--nav-gain", type=float, default=None); ap.add_argument("--all", action="store_true", help="export every fitted episode, not only the successful ones")
    ap.add_argument("--skip-existing", action="store_true"); ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    if a.kit == "bottle_bin" and a.jobs_json:
        params = {int(j["ep"]): j for j in json.load(open(a.jobs_json))}
        ok = {int(json.load(open(f))["ep"]) for f in glob.glob(f"{a.results_dir}/ep*.json") if json.load(open(f)).get("success")}
    else:
        params = {int(k): v for k, v in json.load(open(a.fits)).items() if "skip" not in v}
        ok = {int(r["ep"]) for r in (json.loads(l) for l in open(a.results) if l.strip()) if r.get("success")}
    eps = a.episodes if a.episodes else sorted(params if a.all else ok & set(params))
    nav = a.nav_gain if a.nav_gain is not None else (1.5 if a.kit == "bowl_sink" else 1.0)
    out_root = Path(a.out) / a.kit; out_root.mkdir(parents=True, exist_ok=True); log_dir = LOGS / f"export_{a.kit}"; log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[export] {a.kit}: {len(eps)} episodes ({len(ok)} successful of {len(params)} fitted), {a.jobs} Isaac instances", flush=True)

    def run(ep):
        d = out_root / f"ep_{ep:03d}"
        if a.skip_existing and (d / "replay_meta.json").exists():
            return {"ep": ep, "skipped": True}
        if d.exists():
            import shutil; shutil.rmtree(d)                       # the exporter resumes an existing dataset: start clean
        cmd = [PY, "-u", str(HERE / "replay_isaac.py"), a.kit, "--episode", str(ep), "--seed", str(1000 + ep), "--no-third", "--no-state",
               "--export", str(d), "--out", str(log_dir / f"ep{ep}.mp4"), "--session", SESSIONS[a.kit]]
        if a.kit == "bowl_sink": cmd += ["--nav-gain", str(nav)]
        # (coffee_cart: NOT --ext-clock; its replay runs fast=True with its own sim clock, a second clock double-steps the controller)
        t0 = time.time()
        with open(log_dir / f"ep{ep}.log", "w") as lf:
            rc = subprocess.run(cmd, env=env_for(a.kit, params[ep], ep), stdout=lf, stderr=subprocess.STDOUT, timeout=a.timeout, cwd=str(ROOT)).returncode
        meta = json.load(open(d / "replay_meta.json")) if (d / "replay_meta.json").exists() else {}
        res = meta.get("result") or {}
        g = res.get("gates") if isinstance(res, dict) else None
        succ = bool((g or {}).get("placed") or res.get("success") or res.get("task_success") or (isinstance(res, dict) and (res.get("gates") or {}).get("success")))
        if a.kit == "bottle_bin" and isinstance(res, dict) and ((res.get("robot_bin_leg_ticks") if res.get("robot_bin_leg_ticks") is not None else (res.get("robot_bin_contact_ticks") or 0)) > 25):
            succ = False                                          # the walk ran into the bin: not a clean episode
        if a.kit == "bowl_sink":
            leg = sum(int(n) for pair, n in ((meta.get("contacts") or {}).get("pairs") or []) if any(k in pair.split("|")[0] for k in ("knee", "hip", "ankle", "pelvis", "torso", "waist", "foot")))
            if leg > 150: succ = False                             # legs / pelvis in the counters: the walk was obstructed
        if a.kit == "coffee_cart" and isinstance(res, dict):
            succ = bool(res.get("on_table"))                      # the cup is on the table at the end (upright, inside the footprint)
        built = None
        if meta.get("frames"):
            rb = subprocess.run([PY, "-u", str(HERE / "build_dataset.py"), str(d)], capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
            built = (d / "dataset" / "meta" / "info.json").exists()
            if not built:
                print(f"[export] ep {ep}: dataset build failed: {rb.stdout[-300:]} {rb.stderr[-300:]}", flush=True)
        s = {"ep": ep, "rc": rc, "frames": meta.get("frames"), "built": built, "success": succ, "contact_ticks": (meta.get("contacts") or {}).get("touching_ticks"), "wall_s": round(time.time() - t0, 1)}
        print(f"[export] ep {ep}: frames {s['frames']} built={built} success={succ} contacts={s['contact_ticks']} rc={rc} ({s['wall_s']} s)", flush=True); return s

    results = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for s in ex.map(run, eps):
            results.append(s)
            (out_root / "export_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    print(f"[export] DONE {a.kit}: exported {sum(1 for r in results if r.get('frames'))}, successful {sum(1 for r in results if r.get('success'))} of {len(results)}", flush=True)


if __name__ == "__main__":
    main()
