#!/usr/bin/env python3
"""SAME-BACKGROUND variant of render_fix.py (2026-10-08, user: "another dataset ... with exactly same background instead of
different background, other keeps the same"): every episode is re-rendered from its EXISTING recorded states
(data/replay_datasets_phys/_states/<kit>_epNNN.npy) with ONE seed (SAMEBG_SEED, default 1000 -> the same table / floor
materials and the same lights in every episode) and ONE light level (SAMEBG_LIGHT, default bright), neutral look
(RENDER_FIX_LOOK=new), into data/replay_datasets_phys_samebg/<kit>/ep_NNN (the current renders are not touched).
Usage: render_samebg.py KIT EP [EP ...]"""
import glob, json, os, shutil, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from phys_render import env_for, SESSIONS, relabel, ROOT, DATA, LOGS, OUT as PHYS, PY
OUT = (ROOT / os.environ["SAMEBG_OUT"] if os.environ.get("SAMEBG_OUT") else DATA / "replay_datasets_phys_samebg")   # SAMEBG_OUT: another output root (e.g. a scene revision)
SEED = int(os.environ.get("SAMEBG_SEED", "1000")); LIGHT = os.environ.get("SAMEBG_LIGHT", "bright")
TAG = os.environ.get("SAMEBG_TAG", "samebg")
kit = sys.argv[1]; eps = [int(x) for x in sys.argv[2:]]
# ORIGINAL look (2026-09-29, user: fix only the flagged episodes): the same lighting as the dataset's current renders -- the parent
# task's 2000-8000 K colour temperature and the morning gains (bowl 4.5 / 3.0, bottle 3.0 / 2.0; coffee unchanged); only the
# table / floor materials change (opaque allowlist). RENDER_FIX_LOOK=new uses the kits' current neutral defaults instead.
ORIG = os.environ.get("RENDER_FIX_LOOK", "new") == "orig"
ORIG_GAIN = {"bowl_sink": {"bright": 4.5, "dim": 3.0}, "bottle_bin": {"bright": 3.0, "dim": 2.0}}
params = {}
if kit == "bottle_bin":                                                # the validated jobs the morning renders used
    from phys_render import validated
    params = {e: p for e, (p, _) in validated(kit).items()}
else:
    pre = "coffeevid" if kit == "coffee_cart" else "bowlvid"
    for f in sorted(glob.glob(str(HERE / kit / "replay" / f"walkgap_{pre}*.json"))):
        t = Path(f).stem[len("walkgap_"):]
        if t.endswith(("f", "t")) or t == "bowlvid5": continue           # bowlvid5 = the new 18/31 placements, not a morning render
        for e, v in json.load(open(f)).items():
            if v.get("ok"): params[int(e)] = dict(v["params"], ref_tag=t)
logd = LOGS / f"render_{TAG}_{kit}"; logd.mkdir(parents=True, exist_ok=True)
for ep in eps:
    p = params[ep]; env = env_for(kit, p, ep); env.update(FIT_PHYSICAL="1", REPLAY_GRASP_ASSIST="0", REPLAY_DROP_ASSIST="0")
    env[f"{kit.upper()}_LIGHT_LEVEL"] = LIGHT                            # one light level for every episode
    if kit == "coffee_cart": env.update(REPLAY_SIM_CLOCK="1", REPLAY_PSI0_LEFT_HAND="legacy")
    if kit == "bottle_bin":                                              # placed by the job, exactly like the validating sweep
        env["BOTTLE_BIN_JOB_BOTTLE_XY"] = env.pop("BOTTLE_BIN_BOTTLE_XY"); env["BOTTLE_BIN_JOB_BIN_XY"] = env.pop("BOTTLE_BIN_BIN_XY")
    if ORIG:
        env[f"{kit.upper()}_LIGHT_CCT"] = "2001,8001"
        lv = env.get(f"{kit.upper()}_LIGHT_LEVEL")
        if kit in ORIG_GAIN and lv in ORIG_GAIN[kit]: env[f"{kit.upper()}_LIGHT_GAIN"] = str(ORIG_GAIN[kit][lv])
    d = OUT / kit / f"ep_{ep:03d}"; d.parent.mkdir(parents=True, exist_ok=True)
    if d.exists(): shutil.rmtree(d)
    cmd = [PY, "-u", str(HERE / "replay_isaac.py"), kit, "--episode", str(ep), "--seed", str(SEED), "--no-third", "--no-state", "--session", SESSIONS[kit], "--sim-mode", os.environ.get("SAMEBG_SIM_MODE", "mujoco_isaac"),
           "--state-in", str(PHYS / "_states" / f"{kit}_ep{ep:03d}.npy"), "--export", str(d), "--out", str(logd / f"isaac_ep{ep}.mp4")]
    for attempt in (1, 2):
        with open(logd / f"isaac_ep{ep}.log", "w") as lf: rc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT), timeout=3600).returncode
        if (d / "replay_meta.json").exists(): break
        if d.exists(): shutil.rmtree(d)
    meta = json.load(open(d / "replay_meta.json")) if (d / "replay_meta.json").exists() else {}
    res = meta.get("result") or {}
    if kit == "coffee_cart": same = bool(res.get("on_table")) and bool((res.get("pre_grasp") or {"ok": True}).get("ok", True))
    elif kit == "bowl_sink": same = bool((res.get("gates") or {}).get("placed"))
    else: same = bool(res.get("in_bin") and res.get("grasp_ok"))
    if meta.get("frames"): subprocess.run([PY, "-u", str(HERE / "build_dataset.py"), str(d)], capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
    light = env.get(f"{kit.upper()}_LIGHT_LEVEL")
    vids = sorted(glob.glob(str(logd / f"isaac_ep{ep}.mp4"))) or sorted(glob.glob(str(logd / f"replay_*_ep{ep:03d}_isaac_level0s0_nostate.mp4")))
    if vids: relabel(vids[-1], d / "video.mp4", f"Isaac head camera · {light or 'stock'} lighting · seed {SEED} (same background)")
    mats = (meta.get("env_config") or {}).get("dr_state_dict", {}).get("material", {})
    s = {"ep": ep, "kit": kit, "seed": SEED, "light": light, "ref_tag": p.get("ref_tag"), "render_fix": True, "samebg": True, "frames": meta.get("frames"),
         "forced": meta.get("forced_steps"), "steps": meta.get("steps"), "isaac_success": same, "built": (d / "dataset/meta/info.json").exists(),
         "table": (mats.get("table_material") or {}).get("path", "").split("vMaterials_2/")[-1], "floor": (mats.get("ground_material") or {}).get("path", "").split("vMaterials_2/")[-1]}
    # SAMEBG_REQUIRE_SUCCESS=0 (2026-10-14, scene revisions): the motion is forced from the validated states; the task's re-evaluated
    # success/pre-grasp heuristics on the free step are reported (isaac_success) but do not reject the render
    need_same = same if os.environ.get("SAMEBG_REQUIRE_SUCCESS", "1") == "1" else True
    s["ok"] = bool(need_same and s["built"] and s["forced"] and s["forced"] == s["steps"] and (d / "video.mp4").exists())
    s["why"] = {"on_table": res.get("on_table"), "pre_grasp": (res.get("pre_grasp") or {}).get("ok") if isinstance(res.get("pre_grasp"), dict) else res.get("pre_grasp"),
                "task_success": res.get("task_success"), "cup_end": res.get("cup_end"), "table_near_x": res.get("table_near_x")}
    if not s["ok"] and d.exists() and os.environ.get("SAMEBG_KEEP_FAILED", "0") != "1":
        shutil.rmtree(d)
    with open(OUT / kit / "results.jsonl", "a") as f: f.write(json.dumps(s) + "\n")
    print("[render-samebg]", json.dumps(s), flush=True)
