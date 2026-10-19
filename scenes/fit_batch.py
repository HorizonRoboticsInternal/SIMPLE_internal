#!/usr/bin/env python3
"""Run one kit's replay per episode with PER-EPISODE scene parameters (from a fits json), N processes at a time (MuJoCo only,
fast mode), and collect the per-episode summaries into replay/fitbatch_<tag>.jsonl.

    python sim/fit_batch.py bowl_sink   --fits sim/bowl_sink/replay/fit_psi0_1.json   --tag fit1 --jobs 8
    python sim/fit_batch.py coffee_cart --fits sim/coffee_cart/replay/fit_psi0_1.json --tag fit1 --jobs 8

fits json: {"<ep>": {...params...}}; params per kit:
  bowl_sink:   robot_to_edge, bowl_xy [x,y], sink_along, robot_dy            (+ optional nav_gain)
  coffee_cart: cup_xy [x,y], route_forward, table_from_line, robot_to_cart?, robot_dy?
"""
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
SESSIONS = {"bowl_sink": "psi0/BowlToSink_0918", "coffee_cart": "psi0/CartCoffeeCup_0919"}


def command(kit, ep, p, tag, nav_gain):
    env = dict(os.environ, MUJOCO_GL="egl", REPLAY_FAST="1")
    if kit == "bowl_sink":
        # through the replay wrapper in MuJoCo mode: same clock as the Isaac export, and it counts robot-furniture contacts
        # (the user wants the walk unobstructed: a placed bowl with the knees in the counter does not count)
        out = HERE / "bowl_sink/replay" / f"summary_ep{ep}_{tag}.json"
        cmd = [PY, "-u", str(HERE / "replay_isaac.py"), "bowl_sink", "--session", SESSIONS[kit], "--episode", str(ep), "--sim-mode", "mujoco", "--no-state", "--no-third",
               "--nav-gain", str(p.get("nav_gain", nav_gain)), "--summary-out", str(out), "--trace-out", str(HERE / "bowl_sink/replay" / f"trace_ep{ep}_{tag}.json"),
               "--out", str(LOGS / f"fitbatch_{kit}_{tag}_ep{ep}.mp4")]
        env["REPLAY_GRASP_ASSIST"] = "1" if p.get("grasp_assist") else "0"
        if p.get("sink_recess") is not None: env["BOWL_SINK_SINK_RECESS"] = f"{p['sink_recess']:.4f}"
        if p.get("sink_far"): env["BOWL_SINK_SINK_FAR"] = f"{p['sink_far']:.3f}"            # carry twin (walk check)
        if os.environ.get("REPLAY_QPOS_DIR"): env["REPLAY_QPOS_OUT"] = os.path.join(os.environ["REPLAY_QPOS_DIR"], f"bowl_ep{ep:03d}.npz")
        env.update(BOWL_SINK_ROBOT_TO_EDGE=f"{p['robot_to_edge']:.4f}", BOWL_SINK_BOWL_XY=f"{p['bowl_xy'][0]:.4f},{p['bowl_xy'][1]:.4f}", BOWL_SINK_SINK_ALONG=f"{p['sink_along']:.4f}",
                   BOWL_SINK_ROBOT_DY=f"{p.get('robot_dy', 0.0):.4f}", BOWL_SINK_NUM_DISTRACTORS="0")
        env.pop("REPLAY_FAST", None)                             # the wrapper installs its own virtual clock (fast mode records no trace summary the same way)
    elif kit == "coffee_cart":
        cmd = [PY, "-u", str(HERE / "coffee_cart/replay_in_scene.py"), "--episodes", str(ep), "--worker", "--fast", "--tag", tag,
               "--cup-xy", str(p["cup_xy"][0]), str(p["cup_xy"][1]), "--route-forward", str(p["route_forward"]), "--table-from-line", str(p["table_from_line"]),
               "--nav-gain", str(p.get("nav_gain", nav_gain))]
        if p.get("robot_to_cart") is not None:
            cmd += ["--robot-to-cart", str(p["robot_to_cart"])]
        if p.get("cart_attach"):
            cmd += ["--cart-attach", p["cart_attach"]]
        env["COFFEE_CART_ROBOT_DY"] = f"{p.get('robot_dy', 0.0):.4f}"
        # the coffee_cart main() only exports its scene env vars on the non-worker path: set them here for the worker
        env["COFFEE_CART_ROUTE_FORWARD"] = f"{p['route_forward']:.4f}"; env["COFFEE_CART_TABLE_FROM_LINE"] = f"{p['table_from_line']:.4f}"
        env["COFFEE_CART_CUP_XY"] = f"{p['cup_xy'][0]:.4f},{p['cup_xy'][1]:.4f}"
        if p.get("hand_delta"): env["REPLAY_HAND_DELTA_R"] = ",".join(f"{v:.4f}" for v in p["hand_delta"])
        env["REPLAY_GRASP_ASSIST"] = "1" if p.get("grasp_assist") else "0"
        if p.get("box_from_handle") is not None: env["COFFEE_CART_BOX_FROM_HANDLE"] = f"{p['box_from_handle']:.4f}"
        env["REPLAY_CART_PARK"] = "1" if p.get("cart_park") else "0"
        if os.environ.get("REPLAY_QPOS_DIR"): env["REPLAY_QPOS_OUT"] = os.path.join(os.environ["REPLAY_QPOS_DIR"], f"coffee_ep{ep:03d}.npz")
        env["REPLAY_CUP_PIN_PLACED"] = "1" if p.get("cup_pin") else "0"
        if p.get("robot_to_cart") is not None:
            env["COFFEE_CART_ROBOT_TO_CART"] = f"{p['robot_to_cart']:.4f}"
        env["REPLAY_NAV_GAIN"] = f"{p.get('nav_gain', nav_gain):.3f}"; env["REPLAY_SIM_CLOCK"] = "1"; env["REPLAY_WAIST_MODE"] = "freeze_while_walking"
        env["REPLAY_CART_ATTACH"] = p.get("cart_attach", "none"); env["REPLAY_WALK_STRETCH"] = "1.000"; env["REPLAY_POST_TURN_GAIN"] = "1.000"
        out = HERE / "coffee_cart/replay" / f"summary_ep{ep}_{tag}.json"
    else:
        raise SystemExit(f"kit {kit}: use sweep_replay.py --jobs for bottle_bin")
    return cmd, env, out


def summary_of(kit, ep, tag, out):
    if not out.exists():
        return {"ep": ep, "error": "no output"}
    d = json.load(open(out))
    if kit == "bowl_sink":
        s = d.get("summary") or {}; c = d.get("contacts") or {}
        g = s.get("gates", {}) or {}; touching = int(c.get("touching_ticks") or 0)
        # only the legs / pelvis / torso count as "the counter blocks the walk"; hands and elbows touch counter tops while manipulating
        leg = sum(int(n) for pair, n in (c.get("pairs") or []) if any(k in pair.split("|")[0] for k in ("knee", "hip", "ankle", "pelvis", "torso", "waist", "foot")))
        return {"ep": ep, "success": bool(g.get("placed")) and leg <= 150, "placed_any": bool(g.get("placed")), "contact_ticks": touching, "leg_contact": leg, "first_contact_tick": c.get("first_tick"), "contact_pairs": c.get("pairs"),
                "grasped": bool(g.get("grasped")), "at_basin": bool(g.get("at_basin")), "placed": bool(g.get("placed")), "assist_captured": s.get("assist_captured"), "open_row": s.get("open_row"),
                "max_lift": g.get("max_lift"), "stalled_at": g.get("stalled_at"), "t_grasped": g.get("t_grasped"), "t_placed": g.get("t_placed"),
                "target_start": s.get("target_start"), "target_end": s.get("target_end"), "basin": s.get("basin"), "bowl_end_minus_basin": g.get("bowl_end_minus_basin")}
    tp = d.get("task_progress") or {}
    ok = bool(d.get("on_table")) and bool((d.get("pre_grasp") or {"ok": True}).get("ok", True))
    if os.environ.get("FIT_STRICT") == "1":                          # 2026-09-29 data check: the kit's own scorer (cup resting on the desk with
        ok = ok and bool(d.get("task_success")) and float(d.get("max_lift_m") or 0.0) >= 0.04   # the hand off it) + lifted, not slid
    return {"ep": ep, "success": ok, "task_success": d.get("task_success"), "pre_grasp": d.get("pre_grasp"), "grasped": bool(d.get("grasped")), "cup_lifted": bool(tp.get("cup_lifted_ever", tp.get("cup_lifted"))),
            "cart_pushed": bool(tp.get("cart_pushed")), "on_table": bool(d.get("on_table")), "still_on_cart": bool(d.get("still_on_cart")),
            "cup_start": d.get("cup_start"), "cup_end": d.get("cup_end"), "table_centre": d.get("table_centre"), "cup_end_minus_table_centre": d.get("cup_end_minus_table_centre"),
            "assist_captured": d.get("assist_captured"), "cup_close_row": d.get("cup_close_row"), "cup_open_row": d.get("cup_open_row"),
            "cup_in_hand_end": bool((d.get("task_progress") or {}).get("cup_in_hand")), "cup_at_reset": d.get("cup_at_reset"),
            "base_end_xy": d.get("base_end_xy"), "max_lift_m": d.get("max_lift_m")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kit"); ap.add_argument("--fits", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--episodes", type=int, nargs="*", default=None); ap.add_argument("--nav-gain", type=float, default=None)
    ap.add_argument("--skip-existing", action="store_true"); ap.add_argument("--timeout", type=int, default=2400)
    a = ap.parse_args()
    nav_gain = a.nav_gain if a.nav_gain is not None else (1.5 if a.kit == "bowl_sink" else 1.0)
    fits = {int(k): v for k, v in json.load(open(a.fits)).items() if "skip" not in v}
    eps = a.episodes if a.episodes else sorted(fits)
    log_dir = LOGS; log_dir.mkdir(exist_ok=True)
    res_path = HERE / a.kit / "replay" / f"fitbatch_{a.tag}.jsonl"
    print(f"[fit-batch] {a.kit} tag={a.tag}: {len(eps)} episodes, {a.jobs} at a time, nav gain {nav_gain}", flush=True)

    def run(ep):
        cmd, env, out = command(a.kit, ep, fits[ep], a.tag, nav_gain)
        if a.skip_existing and out.exists():
            return summary_of(a.kit, ep, a.tag, out)
        t0 = time.time()
        with open(log_dir / f"fitbatch_{a.kit}_{a.tag}_ep{ep}.log", "w") as lf:
            rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=a.timeout, cwd=str(ROOT)).returncode
        s = summary_of(a.kit, ep, a.tag, out); s["rc"] = rc; s["wall_s"] = round(time.time() - t0, 1); s["params"] = fits[ep]
        print(f"[fit-batch] ep {ep}: success={s.get('success')} grasped={s.get('grasped')} rc={rc} ({s['wall_s']} s)", flush=True)
        return s

    results = []
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for s in ex.map(run, eps):
            results.append(s)
            with open(res_path, "w") as f:
                for r in results:
                    f.write(json.dumps(r) + "\n")
    n_ok = sum(1 for r in results if r.get("success")); n_g = sum(1 for r in results if r.get("grasped"))
    print(f"[fit-batch] DONE {a.kit} {a.tag}: success {n_ok}/{len(results)}, grasped {n_g} -> {res_path}", flush=True)


if __name__ == "__main__":
    main()
