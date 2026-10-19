#!/usr/bin/env python3
"""Convert the Isaac-rendered PHYSICAL replays (data/replay_datasets_phys/<kit>/ep_NNN/dataset, SIMPLE teleop schema, 50 fps)
into the psi0 LeRobot schema of data/real_recordings/psi0/<real>, one dataset per task, KEEPING 50 fps (user 2026-09-29):
    data/replay_datasets_phys/psi0/<real>_sim/
Field mapping (checked against the real sets; same order conventions as the replay kits used to drive the sim):
  observation.hand_joints (14)  measured Dex3 joints, per hand thumb_0 thumb_1 thumb_2 index_0 index_1 middle_0 middle_1 (left, right)
  observation.arm_joints (14)   measured arm joints, shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw (left, right)
  observation.leg_joints (15)   measured legs (left then right: hip pitch/roll/yaw, knee, ankle pitch/roll) + waist yaw, roll, pitch
  observation.prev_torso_rpy (3) the previous frame's states rpy (0 on the first frame, as in the real sets)
  observation.prev_height (1)   the previous frame's commanded height (the first frame's own command on frame 0)
  states (32)  hands (with the real sets' left-hand swap: index and middle pairs exchanged vs hand_joints) | arms | measured waist
               roll, pitch, yaw | prev_height   -- identical layout to the real sets
  action (36)  hand targets | arm targets | torso rpy command | height command | vx vy vyaw target_yaw, i.e. the commands the sim
               controller actually executed (bowl walks with 1.5x the recorded vx/vy: environment_config.nav_gain)
Meta: info.json / tasks.jsonl / modality.json as the real set (fps 50), episodes.jsonl with the real episode as source and the sim
settings in environment_config, episodes_stats.jsonl, stats.json = stats_psi0.json (float32 stats), lang_map.json and
relative_stats.json empty as in the real sets.
END HOLD (2026-09-29, data check http://10.40.11.11:8899/replay_phys/): the replay appends a settle hold after the last real row
(coffee 2.0 s, bowl/bottle 1.5 s, one frozen action) before scoring; the real episodes end without it. Every episode is cut at
the real recording's duration: keep = ceil(real_rows * 50 / 30) frames (parquet AND video), so it ends where the real one ends.
Usage: to_psi0.py KIT [--only-confirmed LIST.txt] [--no-trim]"""
import os
import argparse, json, shutil
from pathlib import Path
import math
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("SIMPLE_DATA_DIR") or HERE.parent / "data")   # the SIMPLE data root (this tool's ROOT is data/, not the checkout)
PHYS = Path(os.environ.get("PHYS_OUT") or ROOT / "replay_datasets_phys")
REAL = {"coffee_cart": "CartCoffeeCup_0919", "bowl_sink": "BowlToSink_0918", "bottle_bin": "ChipCanToTrash_0225_keep97"}
OUTN = {"coffee_cart": "CartCoffeeCup_0919_sim", "bowl_sink": "BowlToSink_0918_sim", "bottle_bin": "ChipCanToTrash_0225_sim"}
FPS = 50
HAND = [f"{s}_hand_{j}_joint" for s in ("left", "right") for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
# psi0 ACTION hand order (SIMPLE baselines/psi0.py from_psi0_upper_joints): left thumb, MIDDLE, index; right thumb, index, middle
ACT_HAND = ([f"left_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]
            + [f"right_hand_{j}_joint" for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")])
TASK_TEXT = {"bottle_bin": "pick up the water bottle, move towards the trash bin, and place the water bottle in the trash bin"}   # the real set's string says "chip can"; the sim (and the real videos) show a 500 ml water bottle
ARM = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
LEG = [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")] + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
WAIST_RPY = ["waist_roll_joint", "waist_pitch_joint", "waist_yaw_joint"]
STATES_HAND_ORDER = [0, 1, 2, 5, 6, 3, 4] + list(range(7, 14))      # the real sets' states carry the left hand's index/middle pairs swapped
COLS = ["states", "action", "observation.hand_joints", "observation.arm_joints", "observation.leg_joints", "observation.prev_torso_rpy", "observation.prev_height",
        "timestamp", "frame_index", "episode_index", "index", "task_index", "next.done"]


def stats(x):
    x = np.asarray(x, dtype=np.float32); x = x.reshape(len(x), -1)
    return {"mean": x.mean(0).tolist(), "std": x.std(0).tolist(), "min": x.min(0).tolist(), "max": x.max(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).astype(np.float32).tolist(), "q99": np.quantile(x, 0.99, axis=0).astype(np.float32).tolist()}


def trim_video(src, dst, keep):
    """The first `keep` frames of src, re-encoded (H.264, yuv420p, same size, 50 fps)."""
    import av
    with av.open(str(src)) as ic, av.open(str(dst), "w") as oc:
        vi = ic.streams.video[0]; vs = oc.add_stream("libx264", rate=FPS); vs.pix_fmt = "yuv420p"
        vs.width, vs.height = vi.codec_context.width, vi.codec_context.height; vs.options = {"crf": "18", "preset": "medium"}
        k = 0
        for fr in ic.decode(video=0):
            if k >= keep: break
            for pkt in vs.encode(av.VideoFrame.from_ndarray(fr.to_ndarray(format="rgb24"), format="rgb24")): oc.mux(pkt)
            k += 1
        for pkt in vs.encode(): oc.mux(pkt)
    return k


def convert_episode(d, new_ep, start_index, keep=None, nav_gain=1.0):
    info = json.load(open(d / "dataset/meta/info.json")); f = info["features"]
    df = pd.read_parquet(sorted((d / "dataset/data").rglob("*.parquet"))[0])
    if keep is not None: df = df.iloc[:keep].reset_index(drop=True)
    sn, an = f["observation.state"]["names"], f["action"]["names"]
    S = np.stack(df["observation.state"]).astype(np.float64); A = np.stack(df["action"]).astype(np.float64)
    si = lambda names: [sn.index(n) for n in names]; ai = lambda names: [an.index(n) for n in names]
    hand, arm, leg, waist = S[:, si(HAND)], S[:, si(ARM)], S[:, si(LEG)], S[:, si(WAIST_RPY)]
    h = np.stack(df["teleop.base_height_command"]).astype(np.float64).reshape(len(df), 1)
    rpy_cmd = np.stack(df["observation.torso_rpy_command"]).astype(np.float64); nav = np.stack(df["teleop.navigate_command"]).astype(np.float64)
    prev_h = np.vstack([h[:1], h[:-1]]); prev_rpy = np.vstack([np.zeros((1, 3)), waist[:-1]])
    states = np.concatenate([hand[:, STATES_HAND_ORDER], arm, waist, prev_h], axis=1)
    nav = nav.copy(); nav[:, :2] /= nav_gain                      # the operator's base command (the replay scaled vx, vy by its nav gain)
    action = np.concatenate([A[:, ai(ACT_HAND)], A[:, ai(ARM)], rpy_cmd, h, nav[:, :4]], axis=1)
    # 2026-09-30: a render replayed with REPLAY_PSI0_LEFT_HAND=legacy (coffee: recorded with the original left-hand
    # wiring) drives the left middle joint with the recorded index command and vice versa, so reading the hand back by
    # joint name swaps middle and index against the recording. Swap them back: the label must equal the operator's
    # recorded action (checked against the real psi0 recordings: 0.001-0.002 mean error vs 0.09-0.16 swapped).
    try:
        legacy = json.load(open(d / "replay_meta.json")).get("scene_env", {}).get("REPLAY_PSI0_LEFT_HAND") == "legacy"
    except (OSError, ValueError):
        legacy = False
    if legacy:
        action[:, 3:5], action[:, 5:7] = action[:, 5:7].copy(), action[:, 3:5].copy()
    n = len(df); assert states.shape == (n, 32) and action.shape == (n, 36)
    cols = {"states": states, "action": action, "observation.hand_joints": hand, "observation.arm_joints": arm, "observation.leg_joints": leg,
            "observation.prev_torso_rpy": prev_rpy, "observation.prev_height": prev_h}
    arrays = [pa.array([r.tolist() for r in cols[c]], type=pa.list_(pa.float64())) for c in COLS[:7]]
    arrays += [pa.array(np.arange(n) / FPS, type=pa.float64()), pa.array(np.arange(n), type=pa.int64()), pa.array(np.full(n, new_ep), type=pa.int64()),
               pa.array(start_index + np.arange(n), type=pa.int64()), pa.array(np.zeros(n, dtype=np.int64), type=pa.int64()), pa.array(np.arange(n) == n - 1, type=pa.bool_())]
    return pa.Table.from_arrays(arrays, names=COLS), n, states, action


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("kit", choices=sorted(REAL)); ap.add_argument("--only-confirmed", default=None); ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--phys-root", default=None, help="render root instead of data/replay_datasets_phys (e.g. the same-background renders)")
    ap.add_argument("--out-name", default=None, help="dataset name under replay_datasets_phys/psi0 instead of <real>_sim")
    ap.add_argument("--light-level", default=None, help="record this lighting level for every episode (same-background renders) instead of the per-episode draw")
    a = ap.parse_args()
    global PHYS
    out = PHYS / "psi0" / (a.out_name or OUTN[a.kit])
    if a.phys_root: PHYS = Path(a.phys_root)
    kit = a.kit; real = ROOT / "real_recordings/psi0" / REAL[kit]
    rows = {}
    for l in open(PHYS / kit / "results.jsonl"):
        r = json.loads(l); rows[r["ep"]] = r                          # the last attempt of an episode wins
    eps = [e for e, r in sorted(rows.items()) if r.get("ok") and (PHYS / kit / f"ep_{e:03d}/dataset/meta/info.json").exists()]
    if a.only_confirmed:
        keep = {int(x.split(":")[1]) for x in open(a.only_confirmed).read().split() if x.startswith(kit + ":")}; eps = [e for e in eps if e in keep]
    if out.exists(): shutil.rmtree(out)
    (out / "data/chunk-000").mkdir(parents=True); (out / "videos/chunk-000/egocentric").mkdir(parents=True); (out / "meta").mkdir()
    real_eps = {json.loads(l)["episode_index"]: json.loads(l) for l in open(real / "meta/episodes.jsonl")}
    task = json.loads(open(real / "meta/tasks.jsonl").readline())
    if kit in TASK_TEXT: task = dict(task, task=TASK_TEXT[kit], description=TASK_TEXT[kit])
    light = json.load(open(HERE / kit / "replay/light_levels.json"))["levels"]
    real_fps = float(json.load(open(real / "meta/info.json"))["fps"])
    keeps, full = {}, {}
    for e in eps:
        n_full = pq.read_metadata(sorted((PHYS / kit / f"ep_{e:03d}/dataset/data").rglob("*.parquet"))[0]).num_rows; full[e] = n_full
        keeps[e] = n_full if a.no_trim else min(n_full, math.ceil(real_eps[e]["length"] * FPS / real_fps - 1e-9))
    vids_src = {e: sorted((PHYS / kit / f"ep_{e:03d}/dataset/videos").rglob("*.mp4"))[0] for e in eps}
    with ProcessPoolExecutor(max_workers=8) as ex:                 # videos: copy (no trim) or cut to the kept frames
        futs = {k: (ex.submit(shutil.copy2, vids_src[e], out / f"videos/chunk-000/egocentric/episode_{k:06d}.mp4") if keeps[e] == full[e]
                    else ex.submit(trim_video, vids_src[e], out / f"videos/chunk-000/egocentric/episode_{k:06d}.mp4", keeps[e])) for k, e in enumerate(eps)}
        for k, f_ in futs.items():
            r_ = f_.result()
            if isinstance(r_, int): assert r_ == keeps[eps[k]], (eps[k], r_, keeps[eps[k]])
    all_s, all_a, all_t, all_f, all_e, all_i, all_d = [], [], [], [], [], [], []; epl, eps_stats = [], []; idx = 0
    for k, e in enumerate(eps):
        d = PHYS / kit / f"ep_{e:03d}"; meta = json.load(open(d / "replay_meta.json")); env = meta.get("scene_env", {})
        nav_gain = float(env.get("REPLAY_NAV_GAIN", 1.5 if kit == "bowl_sink" else 1.0))
        tbl, n, st, ac = convert_episode(d, k, idx, keeps[e], nav_gain)
        pq.write_table(tbl, out / f"data/chunk-000/episode_{k:06d}.parquet")
        epl.append({"episode_index": k, "tasks": [0], "length": n, "dataset_from_index": idx, "dataset_to_index": idx + n - 1, "robot_type": "g1",
                    "instruction": {"task_index": 0, "task": task["task"]},
                    "environment_config": {"sim": "MuJoCo physics (hard contact only, no kinematic aids) rendered in Isaac Sim from the exact recorded state",
                                           "kit": kit, "seed": meta.get("seed"), "lighting": a.light_level or light.get(str(e)), "nav_gain_in_sim": nav_gain, "label_base_command": "operator's (sim executed it x nav_gain_in_sim)", "fps": FPS,
                                           "trimmed_end_hold_frames": full[e] - keeps[e],
                                           "scene": {kk: v for kk, v in env.items() if kk.startswith(kit.upper()) or kk.startswith("BOTTLE_BIN_JOB")}},
                    "source": {"dataset": f"psi0/{REAL[kit]}", "episode_index": e, "real_source": (real_eps.get(e) or {}).get("source")}})
        t = np.arange(n) / FPS
        eps_stats.append({"episode_index": k, "stats": {"action": dict(stats(ac), count=[n]), "timestamp": dict(stats(t), count=[n])}})
        all_s.append(st); all_a.append(ac); all_t.append(t); all_f.append(np.arange(n)); all_e.append(np.full(n, k)); all_i.append(idx + np.arange(n)); all_d.append(np.arange(n) == n - 1)
        idx += n
    cat = np.concatenate
    st_all = {"states": stats(cat(all_s)), "action": stats(cat(all_a)), "timestamp": stats(cat(all_t)), "frame_index": stats(cat(all_f)), "episode_index": stats(cat(all_e)),
              "index": stats(cat(all_i)), "task_index": stats(np.zeros(idx)), "next.done": stats(cat(all_d).astype(np.float32))}
    info = json.load(open(real / "meta/info.json"))
    info.update(fps=FPS, total_episodes=len(eps), total_frames=idx, total_videos=len(eps), total_tasks=1, total_chunks=1)
    for kk, v in info["features"].items():
        if v.get("dtype") == "video": v["video_info"] = {"video.fps": FPS, "video.codec": "h264", "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "has_audio": False}
    json.dump(info, open(out / "meta/info.json", "w"), indent=4)
    open(out / "meta/tasks.jsonl", "w").write(json.dumps(task) + "\n"); shutil.copy2(real / "meta/modality.json", out / "meta/modality.json")
    for fn in ("lang_map.json", "relative_stats.json"): shutil.copy2(real / f"meta/{fn}", out / f"meta/{fn}")
    open(out / "meta/episodes.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in epl))
    open(out / "meta/episodes_stats.jsonl", "w").write("".join(json.dumps(x) + "\n" for x in eps_stats))
    json.dump(st_all, open(out / "meta/stats.json", "w"), indent=4); json.dump(st_all, open(out / "meta/stats_psi0.json", "w"), indent=4)
    cut = [full[e] - keeps[e] for e in eps]
    print(f"[to-psi0] {kit}: {len(eps)} episodes, {idx} frames at {FPS} fps (end hold cut: {sum(cut)} frames, {min(cut)}-{max(cut)} per episode) -> {out}", flush=True)


if __name__ == "__main__":
    main()
