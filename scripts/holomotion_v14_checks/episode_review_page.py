"""Review page for one recorded HoloMotion v1.4 teleop episode with the Revo 2 robot (2026-10-27).

    # 1. images + tactile pads (a lean recording has neither): bit-exact replay with --fill -> <level dir>_filled
    MUJOCO_GL=egl .venv/bin/python -m simple.cli.replay_holomotion_v14 <level dir> --all --fill
    # 2. third-person chase camera, rendered during another bit-exact replay
    MUJOCO_GL=egl .venv/bin/python scripts/holomotion_v14_checks/chase_view.py <level dir> <episode> <chase.mp4>
    # 3. this page
    .venv/bin/python scripts/holomotion_v14_checks/episode_review_page.py <level dir> --episode N --chase <chase.mp4> --name <page name>

Writes docs/<name>/ (NAS: index.html, grid.mp4, tactile.json) and docs/_scan/<name>/ (artifact: page.html, grid_small.mp4,
tactile.json). The page has the camera grid (head camera as recorded, chase camera, both wrist cameras; time and both grip
commands on the frames) and a tactile section synced to the video with every recorded tactile column: the 5 fingertip pads
of each hand as 3x3 taxel maps seen from outside the pad, fingertip up (normal channel, or the x/y shear channels as arrows),
the pad's tangential force as an arrow, a readout table of every field at the current time (physical values and the SDK
fields as the real hand reports them), and charts over the episode of normal force, tangential force, tangential direction,
proximity and contact count with the grip command, click to seek. Values are the filled dataset's tactile.{s}.* columns,
sampled every second frame (25 Hz) like the video. Taxel layout (measured, 2026-10-27): taxels_N is (finger, channel z/x/y,
row, col) with row along pad +y and col along pad +x (toward the fingertip).
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2
import imageio.v2 as iio
import imageio_ffmpeg
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PADS = ["thumb", "index", "middle", "ring", "pinky"]
SIDES = ("left", "right")


def col(d, c):
    return np.stack([np.asarray(v) for v in d[c]])


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("level_dir"); ap.add_argument("--episode", type=int, required=True); ap.add_argument("--chase", required=True)
    ap.add_argument("--name", required=True); ap.add_argument("--title", default=""); ap.add_argument("--small-mb", type=float, default=14.0)
    ap.add_argument("--reuse-video", action="store_true", help="keep grid.mp4 / grid_small.mp4 from the last build (page-only changes)")
    a = ap.parse_args()
    src = Path(a.level_dir).resolve(); filled = src.with_name(src.name + "_filled") if not src.name.endswith("_filled") else src
    ep = a.episode; nas, scan = ROOT / "docs" / a.name, ROOT / "docs/_scan" / a.name
    nas.mkdir(parents=True, exist_ok=True); scan.mkdir(parents=True, exist_ok=True)
    meta = [json.loads(l) for l in open(filled / "meta/episodes.jsonl") if l.strip()]
    em = next(e for e in meta if e["episode_index"] == ep)
    fill = json.loads(em["fill"]) if isinstance(em.get("fill"), str) else (em.get("fill") or {})
    if fill and int(fill.get("source_episode", ep)) != ep:                  # a single-episode fill (--episode N) is index 0 of its copy
        print(f"[page] {filled}: filled episode {ep} = source episode {fill.get('source_episode')}", flush=True)
    d = pd.read_parquet(filled / f"data/chunk-000/episode_{ep:06d}.parquet")
    n50 = len(d); idx = np.arange(0, n50, 2); n = len(idx)
    mode = {s: col(d, f"action.revo2.{s}.grip_mode").ravel().astype(int) for s in SIDES}

    # ---- the camera grid video (25 fps: every second recorded frame) ------------------------------------------------------
    V = filled / "videos/chunk-000"
    readers = [iio.get_reader(str(V / f"observation.images.ego_view_left/episode_{ep:06d}.mp4")), iio.get_reader(a.chase),
               iio.get_reader(str(V / f"observation.images.wrist_left/episode_{ep:06d}.mp4")), iio.get_reader(str(V / f"observation.images.wrist_right/episode_{ep:06d}.mp4"))]
    labels = ["head camera (rectified left eye, as recorded)", "third person (chase camera)", "left wrist camera (150 deg)", "right wrist camera (150 deg)"]
    font = cv2.FONT_HERSHEY_SIMPLEX
    names = {0: "open", 1: "CLOSED (grip)", 2: "PINCH (trigger)"}

    def put(img, txt, xy, s=0.5, c=(255, 255, 255)):
        cv2.putText(img, txt, xy, font, s, (0, 0, 0), 3, cv2.LINE_AA); cv2.putText(img, txt, xy, font, s, c, 1, cv2.LINE_AA)

    big_path, small_path = nas / "grid.mp4", scan / "grid_small.mp4"
    if a.reuse_video and big_path.exists() and small_path.exists():
        readers = []
        print(f"[page] reusing {big_path} and {small_path}", flush=True)
    big = None if not readers else iio.get_writer(str(big_path), fps=25, codec="libx264", quality=None, ffmpeg_params=["-crf", "23", "-preset", "medium", "-movflags", "+faststart"], macro_block_size=1)   # index first: the NAS server sends no byte ranges
    for t, frames in enumerate(zip(*readers) if readers else ()):
        if t >= n50:
            break
        if t % 2:
            continue
        tiles = [np.ascontiguousarray(cv2.resize(x, (640, 360), interpolation=cv2.INTER_AREA) if x.shape[:2] != (360, 640) else x.copy()) for x in frames]
        for tile, lab in zip(tiles, labels):
            put(tile, lab, (10, 22))
        put(tiles[1], f"t = {t / 50:6.2f} s", (10, 352))
        for k, s in ((2, "left"), (3, "right")):
            mm = int(mode[s][t]); put(tiles[k], f"{s} grip: {names[mm]}", (10, 352), 0.5, (255, 255, 255) if mm == 0 else (120, 220, 255))
        big.append_data(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])]))
    if big:
        big.close()
    ff = imageio_ffmpeg.get_ffmpeg_exe(); import subprocess
    for crf in (32, 34, 36, 38) if big else ():
        subprocess.run([ff, "-loglevel", "error", "-y", "-i", str(big_path), "-vf", "scale=854:480", "-c:v", "libx264", "-crf", str(crf), "-preset", "slow",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(small_path)], check=True)
        if small_path.stat().st_size / 1e6 <= a.small_mb:
            break
    if big:
        print(f"[page] grid.mp4 {big_path.stat().st_size / 1e6:.1f} MB (NAS), grid_small.mp4 {small_path.stat().st_size / 1e6:.1f} MB (crf {crf}, artifact)", flush=True)

    # ---- tactile data (25 Hz) -----------------------------------------------------------------------------------------------
    tac = {}
    for s in SIDES:
        tx = col(d, f"tactile.{s}.taxels_N")[idx].reshape(n, 5, 3, 3, 3)       # (finger, channel z/x/y, row = pad +y, col = pad +x toward the tip)
        c = lambda k: col(d, f"tactile.{s}.{k}")[idx]
        # MuJoCo's touch_grid negates all three channels when the pad is the second geom of a contact (measured 2026-10-27); a
        # contact only pushes into the pad, so a negative normal channel marks a flipped taxel: fold its sign into the shear too
        sg = np.where(tx[:, :, 0] < 0, -1.0, 1.0)
        tac[s] = dict(taxel=np.abs(tx[:, :, 0]).reshape(n, 45), shear_x=(tx[:, :, 1] * sg).reshape(n, 45), shear_y=(tx[:, :, 2] * sg).reshape(n, 45),
                      normal=np.abs(c("force_normal_N")), tang=np.abs(c("force_tangential_N")), dirn=c("force_direction"), prox=c("proximity_m"),
                      contacts=c("contacts"), status=c("status"), raw_normal=c("force_normal"), raw_tang=c("force_tangential"), raw_prox=c("proximity"))
    # colour scale per taxel: 0..TAXEL_CAP N normal, 0..SHEAR_CAP N shear (brighter = more); pad totals at 0.1 N (uint16), charts
    # on the real sensor's 0..25 N range with larger values clipped (the sim hand has no motor current limit and can press far harder)
    TAXEL_CAP = 10.0; SHEAR_CAP = 5.0; tscale = 0.05; nscale = 0.1
    nmax = float(max(tac[s]["normal"].max() for s in SIDES))
    u16 = lambda a: base64.b64encode(np.clip(np.round(a), 0, 65535).astype("<u2").tobytes()).decode()
    i8 = lambda a: base64.b64encode(np.clip(np.round(a / tscale), -127, 127).astype(np.int8).tobytes()).decode()   # signed shear, 0.05 N steps
    data = dict(fps=25, n=n, pads=PADS, seconds=round(n50 / 50, 2), taxel_scale=tscale, normal_scale=nscale,
                taxel_cap=TAXEL_CAP, shear_cap=SHEAR_CAP, normal_max=round(nmax, 2), sensor_max=25.0)
    for s in SIDES:
        T = tac[s]
        data[s] = dict(taxel=b64(np.clip(np.round(T["taxel"] / tscale), 0, 255)), shear_x=i8(T["shear_x"]), shear_y=i8(T["shear_y"]),
                       normal=u16(T["normal"] / nscale), tang=u16(T["tang"] / nscale), dirn=u16(T["dirn"]),               # direction: deg, 65535 = none
                       prox=b64(np.where(T["prox"] < 0, 255, np.clip(np.round(T["prox"] * 1000 * 20), 0, 254))),       # 0.05 mm steps, 255 = none
                       contacts=b64(np.clip(T["contacts"], 0, 255)), status=b64(np.clip(T["status"], 0, 255)),
                       raw_normal=u16(T["raw_normal"]), raw_tang=u16(T["raw_tang"]),                                   # SDK units: 0.01 N, saturating at 2500
                       raw_prox=base64.b64encode(np.clip(T["raw_prox"], 0, 2**32 - 1).astype("<u4").tobytes()).decode(),  # SDK capacitance counts
                       grip=b64(mode[s][idx]))
    for p in (nas / "tactile.json", scan / "tactile.json"):
        p.write_text(json.dumps(data))
    gsum = {s: tac[s]["taxel"].reshape(n, 5, 9).sum(-1) for s in SIDES}; grid_on = {s: gsum[s] > 0.1 for s in SIDES}; touch = {s: tac[s]["contacts"] > 0 for s in SIDES}
    stats = {s: dict(peak_N=round(float(tac[s]["normal"].max()), 1), peak_tang_N=round(float(tac[s]["tang"].max()), 1),
                     touching_s=round(float((tac[s]["normal"].sum(1) > 0.5).sum()) / 25, 1),
                     over_sensor_pct=round(100 * float((tac[s]["normal"] > 25).sum()) / max(1, float((tac[s]["normal"] > 0.05).sum())), 1),
                     status_nonzero=int((tac[s]["status"] != 0).sum()), max_contacts=int(tac[s]["contacts"].max()),
                     # the recorded taxel grid against the pad's own contacts (touch_grid also counts contacts that are not on the pad)
                     grid_no_contact_pct=round(100 * float((grid_on[s] & (tac[s]["contacts"] == 0)).sum()) / max(1, float(grid_on[s].sum())), 1),
                     grid_match_pct=round(100 * float(np.mean(np.abs(gsum[s][touch[s]] / np.maximum(tac[s]["normal"][touch[s]], 1e-6) - 1) < 0.2)) if touch[s].any() else 0.0, 1),
                     grip_closed_s=round(float((mode[s][idx] == 1).sum()) / 25, 1), pinch_s=round(float((mode[s][idx] == 2).sum()) / 25, 1)) for s in SIDES}
    print(f"[page] tactile.json {(scan / 'tactile.json').stat().st_size / 1e6:.2f} MB; taxel colour cap {TAXEL_CAP:g} N, pad normal max {nmax:.1f} N; {stats}", flush=True)

    # ---- the page ----------------------------------------------------------------------------------------------------------
    rf = json.loads(em["realformat"]) if isinstance(em.get("realformat"), str) else (em.get("realformat") or {})
    setup = json.loads(em["scene_setup"]) if isinstance(em.get("scene_setup"), str) else (em.get("scene_setup") or {})
    xl = np.load(src / f"replay/episode_{ep:06d}.npz", allow_pickle=True); info = json.loads(str(xl["info_json"]))
    facts = dict(dataset=str(src.relative_to(ROOT)), filled=str(filled.relative_to(ROOT)), episode=ep, frames=n50, seconds=round(n50 / 50, 1),
                 task=em["tasks"][0], success=rf.get("success"), seed=setup.get("seed"), plate=setup.get("plate"), robot=info.get("robot_mjcf", ""),
                 scene_rev=info.get("env_knobs", {}).get("KITCHEN_REV", "1"), stats=stats, bit_exact=bool(fill.get("bit_exact", True)))
    te = json.loads(em["task_eval"]) if isinstance(em.get("task_eval"), str) else (em.get("task_eval") or {})     # relabel_kitchen_task.py
    if te:
        names = dict(door_open="door open", grasped="first dish grasped", placed="first dish in a rack")
        facts["stages"] = [dict(label=names.get(g, g), t=round(f / 50, 2)) for g, f in te.get("frames_first", {}).items() if f is not None]
        facts["loaded"] = te.get("loaded_dishes_at_end", "").replace("_", " ").replace(",", ", ")
    cj = Path(str(a.chase) + ".json")                                                                               # chase_view.py stats
    if cj.exists():
        facts["chase"] = json.loads(cj.read_text())
    tpl = (Path(__file__).resolve().parent / "episode_review_page.html").read_text()
    title = a.title or f"{src.parts[-4].replace('teleop_holomotion_v14_', '').capitalize()} Episode {ep}"
    for target, video, out in ((scan, "grid_small.mp4", scan / "page.html"), (nas, "grid.mp4", nas / "index.html")):
        page = tpl.replace("__TITLE__", title).replace("__FACTS__", json.dumps(facts)).replace("__VIDEO__", video)
        if target is nas:
            page = page.replace("<!--FONTS-->", "")
            page = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"></head><body>\n'
                    + page + "\n</body></html>\n")
        else:
            page = page.replace("<!--FONTS-->", '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
                                '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">')
        out.write_text(page)
    print(f"[page] {scan / 'page.html'} (artifact), {nas / 'index.html'} (NAS)", flush=True)


if __name__ == "__main__":
    main()
