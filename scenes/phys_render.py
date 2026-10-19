#!/usr/bin/env python3
"""Isaac renders of the VALIDATED PHYSICAL replays (2026-09-29, user: as many hard-contact episodes as possible, a different
seed per episode, 70 % bright / 30 % dim lighting). Per episode:
  1. record: replay_isaac.py --sim-mode mujoco --seed 1000+ep --state-out  (the validated placement; full MuJoCo state before
     every env.step). Kept only if the task succeeds again AND the base path stays within PATH_TOL of the validated run.
  2. render: the same command in mujoco_isaac with --state-in (every step starts from the recorded state, so the Isaac frames show
     exactly the validated run) and --export; kept only if every step was forced and the task succeeds; dataset built.
Output: data/replay_datasets_phys/<kit>/ep_NNN (raw frames + dataset + video.mp4 + replay_meta.json), _states/, results.jsonl.
    python sim/phys_render.py coffee_cart --jobs 3 [--episodes 0 3]"""
import argparse, glob, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; PY = sys.executable; ROOT = HERE.parent   # the SIMPLE checkout
DATA = Path(os.environ.get("SIMPLE_DATA_DIR") or ROOT / "data")   # SIMPLE data root (datasets, recordings, scene sets)
LOGS = Path(os.environ.get("REPLAY_LOG_DIR") or DATA / "replay_logs")   # run logs / per-run videos (not tracked)
OUT = Path(os.environ.get("PHYS_OUT") or DATA / "replay_datasets_phys")   # PHYS_OUT: another output root
sys.path.insert(0, str(HERE)); from export_batch import env_for, SESSIONS
PATH_TOL = 0.03


def bowl_end_ok(trace_rows, ep):
    """2026-09-29 data check: at the REAL end of the episode the bowl rests in the basin with the hand out of it (index fingertip
    >= 0.09 m from the bowl centre); episodes 18/31 had the bowl hooked on the fingers after the placed gate latched."""
    real = {json.loads(l)["episode_index"]: json.loads(l)["length"] for l in open(ROOT / "data/real_recordings/psi0/BowlToSink_0918/meta/episodes.jsonl")}
    rows = [x for x in trace_rows if isinstance(x, dict) and "row" in x and "summary" not in x and x["row"] <= real[ep] - 1]
    if not rows or "index_tip_r" not in rows[-1]: return True
    return float(np.linalg.norm(np.array(rows[-1]["index_tip_r"]) - np.array(rows[-1]["target_xyz"]))) >= 0.09


def validated(kit):
    """{ep: (params, reference base path {t or row: xy})} from the validated reruns that produced the physical review videos.
    scenes/<kit>/replay/validated_episodes.json (shipped with the repo, built from the full fit/sweep/trace data on the workstation
    that produced data/replay_datasets_phys) is used when present; otherwise the raw valid_*/walkgap_*/trace files are read."""
    eps = {}
    shipped = HERE / kit / "replay" / "validated_episodes.json"
    if shipped.exists():
        for e, v in json.load(open(shipped)).items():
            ref = {(float(k) if kit == "bottle_bin" else int(k)): v2 for k, v2 in v["ref_path"].items()}
            eps[int(e)] = (dict(v["params"]), ref)
        return eps
    if kit in ("coffee_cart", "bowl_sink"):
        # every validated set (not the free twins). coffee (2026-09-29): only the STRICT rounds cs* (corrected psi0 left-hand mapping,
        # scorer + lift rule); the older coffeevid* sets were validated with the swapped left hand and are no longer used
        pre = "cs" if kit == "coffee_cart" else "bowlvid"
        tags = sorted({Path(f).stem[len("walkgap_"):] for f in glob.glob(str(HERE / kit / "replay" / f"walkgap_{pre}*.json"))} - {f"{pre}f"}, key=lambda t: (len(t), t))
        tags = [t for t in tags if not t.endswith(("f", "t"))]
        for t in tags:
            f = HERE / kit / "replay" / f"walkgap_{t}.json"
            if not f.exists(): continue
            for e, v in json.load(open(f)).items():
                if v.get("ok") and (int(e) not in eps or v.get("gap", 9) < eps[int(e)][0].get("_gap", 9)):
                    tr = HERE / kit / "replay" / f"trace_ep{e}_{t}.json"
                    rows_ = json.load(open(tr)) if tr.exists() else []
                    if kit == "bowl_sink" and not bowl_end_ok(rows_, int(e)): continue
                    ref = {x["row"]: x["sim_xy"] for x in rows_ if "row" in x and "summary" not in x}
                    eps[int(e)] = (dict(v["params"], ref_tag=t, _gap=v.get("gap", 9)), ref)
    else:
        for t in sorted({Path(f).stem[len("valid_"):] for f in glob.glob(str(HERE / "bottle_bin/replay/valid_bottlevid*.json"))}, key=lambda t: (len(t), t)):
            f = HERE / "bottle_bin/replay" / f"valid_{t}.json"
            if not f.exists(): continue
            for k, v in json.load(open(f)).items():
                if not v.get("ok"): continue
                r = json.load(open(HERE / "bottle_bin/replay/sweep" / t / f"{k}.json"))
                p = {"ep": v["ep"], "bottle_xy": v["job"]["bottle_xy_cmd"], "bin_xy": v["job"]["bin_xy_cmd"], "bin_yaw": v["job"]["bin_yaw_deg"], "ref_tag": t}
                eps[v["ep"]] = (p, {round(q[0], 1): q[1:3] for q in r.get("base_path") or []})
    return eps


def relabel(src, dst, text):
    """Re-encode the kit video with the middle panel labelled as what it is (the kit prints 'MuJoCo D455 (sim)'; in mujoco_isaac
    mode that panel is the Isaac head camera)."""
    import av
    from PIL import ImageDraw, ImageFont
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception: font = ImageFont.load_default()
    with av.open(str(src)) as ic, av.open(str(dst), "w") as oc:
        vi = ic.streams.video[0]; vs = oc.add_stream("libx264", rate=vi.average_rate or 25); vs.pix_fmt = "yuv420p"
        vs.width, vs.height = vi.codec_context.width, vi.codec_context.height; vs.options = {"crf": "23", "preset": "veryfast", "movflags": "+faststart"}
        for fr in ic.decode(video=0):
            im = fr.to_image().convert("RGB"); W = im.width // 3; d = ImageDraw.Draw(im)
            d.rectangle((W + 4, 24, 2 * W - 6, 46), fill=(0, 0, 0)); d.text((W + 10, 27), text, fill=(255, 235, 80), font=font)
            for pkt in vs.encode(av.VideoFrame.from_image(im)): oc.mux(pkt)
        for pkt in vs.encode(): oc.mux(pkt)


def success(kit, res):
    if not isinstance(res, dict): return False
    if kit == "coffee_cart":                                        # 2026-09-29: + the kit's scorer (hand off the cup) + lifted >= 4 cm (not slid)
        return (bool(res.get("on_table")) and bool((res.get("pre_grasp") or {"ok": True}).get("ok", True)) and bool(res.get("task_success"))
                and float(res.get("max_lift_m") or 0.0) >= 0.04)
    if kit == "bowl_sink": return bool((res.get("gates") or {}).get("placed"))
    return bool(res.get("in_bin") and res.get("grasp_ok"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("kit"); ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--skip-existing", action="store_true"); ap.add_argument("--timeout", type=int, default=3600); a = ap.parse_args()
    kit = a.kit; V = validated(kit); eps = [e for e in sorted(V) if not a.episodes or e in a.episodes]
    root = OUT / kit; root.mkdir(parents=True, exist_ok=True); st_dir = OUT / "_states"; st_dir.mkdir(exist_ok=True)
    logd = LOGS / f"phys_render_{kit}"; logd.mkdir(parents=True, exist_ok=True); resf = root / "results.jsonl"
    done = {json.loads(l)["ep"] for l in open(resf)} if (a.skip_existing and resf.exists()) else set()
    print(f"[phys-render] {kit}: {len(eps)} validated episodes, {len(done)} already done, {a.jobs} at a time", flush=True)

    def base_cmd(ep, p):
        c = [PY, "-u", str(HERE / "replay_isaac.py"), kit, "--episode", str(ep), "--seed", str(1000 + ep), "--no-third", "--no-state", "--session", SESSIONS[kit]]
        if kit == "bowl_sink": c += ["--nav-gain", str(p.get("nav_gain", 1.5))]
        return c

    def run(ep):
        if ep in done: return None
        p, ref = V[ep]; env = env_for(kit, p, ep); env["FIT_PHYSICAL"] = "1"; env["REPLAY_GRASP_ASSIST"] = "0"; env["REPLAY_DROP_ASSIST"] = "0"
        if kit == "coffee_cart": env["REPLAY_SIM_CLOCK"] = "1"
        if kit == "bottle_bin":                                    # the scene exactly as the validating sweep built it; the job places bottle + bin
            env["BOTTLE_BIN_JOB_BOTTLE_XY"] = env.pop("BOTTLE_BIN_BOTTLE_XY"); env["BOTTLE_BIN_JOB_BIN_XY"] = env.pop("BOTTLE_BIN_BIN_XY")
        s = {"ep": ep, "kit": kit, "seed": 1000 + ep, "light": env.get(f"{kit.upper()}_LIGHT_LEVEL"), "ref_tag": p.get("ref_tag")}; t0 = time.time()
        stf = st_dir / f"{kit}_ep{ep:03d}.npy"; summ = logd / f"rec_ep{ep}_summary.json"; trace = logd / f"rec_ep{ep}_trace.json"
        cmd = base_cmd(ep, p) + ["--sim-mode", "mujoco", "--state-out", str(stf), "--out", str(logd / f"rec_ep{ep}.mp4")]
        if kit != "bottle_bin": cmd += ["--summary-out", str(summ), "--trace-out", str(trace)]
        with open(logd / f"rec_ep{ep}.log", "w") as lf:
            s["rec_rc"] = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=a.timeout, cwd=str(ROOT)).returncode
        try:
            if kit == "bottle_bin":
                res = json.load(open(HERE / "bottle_bin/replay/sweep/isaac_level0s0_mujoco_nostate" / f"ep{ep:03d}.json")); path = {round(q[0], 1): q[1:3] for q in res.get("base_path") or []}
            else:
                res = (json.load(open(summ)).get("summary") or {}); path = {x["row"]: x["sim_xy"] for x in json.load(open(trace)) if isinstance(x, dict) and "row" in x and "summary" not in x}
        except Exception as exc:
            res, path = {}, {}; s["rec_error"] = repr(exc)[:200]
        ks = sorted(set(path) & set(ref)); s["rec_success"] = success(kit, res)
        s["rec_path_dev"] = round(max(float(np.linalg.norm(np.array(path[k]) - np.array(ref[k]))) for k in ks), 3) if ks else None
        if kit == "bowl_sink" and s["rec_success"]:
            try: s["rec_success"] = bowl_end_ok(json.load(open(trace)), ep)
            except Exception: pass
        s["rec_ok"] = bool(s["rec_success"] and stf.exists() and s["rec_path_dev"] is not None and s["rec_path_dev"] <= PATH_TOL)
        if s["rec_ok"]:
            d = root / f"ep_{ep:03d}"
            if d.exists(): shutil.rmtree(d)
            cmd = base_cmd(ep, p) + ["--state-in", str(stf), "--export", str(d), "--out", str(logd / f"isaac_ep{ep}.mp4")]
            for attempt in (1, 2):                                   # Isaac occasionally segfaults at reset under load: one retry
                with open(logd / f"isaac_ep{ep}.log", "w") as lf:
                    s["isaac_rc"] = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, timeout=a.timeout, cwd=str(ROOT)).returncode
                if (d / "replay_meta.json").exists(): break
                s["isaac_retry"] = attempt
                if d.exists(): shutil.rmtree(d)
            meta = json.load(open(d / "replay_meta.json")) if (d / "replay_meta.json").exists() else {}
            s.update(frames=meta.get("frames"), forced=meta.get("forced_steps"), steps=meta.get("steps"), state_problem=meta.get("state_in_problem"))
            s["isaac_success"] = success(kit, meta.get("result"))
            vids = sorted(glob.glob(str(logd / f"isaac_ep{ep}.mp4"))) or sorted(glob.glob(str(logd / f"replay_*_ep{ep:03d}_isaac_level0s0_nostate.mp4")))
            if vids:
                try: relabel(vids[-1], d / "video.mp4", f"Isaac head camera · {s['light'] or 'stock'} lighting · seed {1000 + ep}")
                except Exception as exc: print(f"[phys-render] relabel failed ({exc!r}); raw video kept", flush=True); shutil.copy2(vids[-1], d / "video.mp4")
            if meta.get("frames"):
                subprocess.run([PY, "-u", str(HERE / "build_dataset.py"), str(d)], capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
            s["built"] = (d / "dataset/meta/info.json").exists()
            s["ok"] = bool(s["isaac_success"] and s["built"] and s.get("forced") and s.get("forced") == s.get("steps") and not s.get("state_problem") and (d / "video.mp4").exists())
        else:
            s["ok"] = False
        s["wall_s"] = round(time.time() - t0, 1)
        print(f"[phys-render] {kit} ep {ep}: " + json.dumps({k: s.get(k) for k in ("light", "rec_success", "rec_path_dev", "rec_ok", "frames", "forced", "steps", "isaac_success", "built", "ok", "wall_s")}), flush=True)
        with open(resf, "a") as f: f.write(json.dumps(s) + "\n")
        return s

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        res = [r for r in ex.map(run, eps) if r]
    print(f"[phys-render] DONE {kit}: {sum(1 for r in res if r.get('ok'))} rendered of {len(res)} (recording ok {sum(1 for r in res if r.get('rec_ok'))})", flush=True)


if __name__ == "__main__":
    main()
