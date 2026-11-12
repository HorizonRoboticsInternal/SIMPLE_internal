"""Process video of an annotated episode: the recorded head camera and the chase camera with the annotation playing along (2026-10-27).

    .venv/bin/python scripts/holomotion_v14_checks/episode_process_video.py <annotation dir> <head.mp4> <chase.mp4> <out.mp4> \
        [--small <small.mp4>] [--slices <dir>]

<annotation dir> holds caption.json, slices.json and traj.json from episode_annotate.py. Every second recorded frame (25 fps,
1280x720): top, the head camera as recorded and the third-person chase camera; below, the current sub task (k of N), the actions
running now with the main one's description, both hands' states, the interaction sentence and the objects in view (from the
nearest 2 fps sample at or before the frame), and a timeline of all sub tasks with the playhead, the incidental events (not
task-relevant actions) and the trimmed tail greyed out. --small also writes an 854x480 copy under 14 MB; --slices cuts one clip
per sub task (the task-related slices) from the full video.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import textwrap
from pathlib import Path

import imageio.v2 as iio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PAL = ["#3987e5", "#e8743b", "#1baf7a", "#c98500", "#d55181", "#7f6cd6", "#2aa6b8", "#a0602c", "#5f8f2f", "#c2454f"]
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("ann"); ap.add_argument("head"); ap.add_argument("chase"); ap.add_argument("out")
    ap.add_argument("--small"); ap.add_argument("--slices"); ap.add_argument("--start", type=float, default=0); ap.add_argument("--limit", type=float, default=0)
    a = ap.parse_args()
    D = Path(a.ann); C = json.load(open(D / "caption.json")); SL = json.load(open(D / "slices.json"))
    fps = C["config"]["fps"]; T = C["config"]["trim"]["original_frames"]; kept = C["config"]["trim"]["kept_frames"]
    acts = {}
    for seg in C["segments"]:
        for x in seg["Current Action"]:
            acts[(x["label"], x["start_frame"])] = x
    acts = sorted(acts.values(), key=lambda x: x["start_frame"])
    samples = {}
    for seg in C["segments"]:
        for f in seg["interaction"]:
            samples[int(f)] = dict(interaction=seg["interaction"][f], hands=seg["hands"][f], objects=seg["objects"][f])
    skeys = np.array(sorted(samples))
    sub_of = np.zeros(T + 1, int)
    for k, s in enumerate(SL):
        sub_of[s["start_frame"]:s["end_frame"] + 1] = k
    f_big = ImageFont.truetype(BOLD, 24); f_mid = ImageFont.truetype(BOLD, 17); f_txt = ImageFont.truetype(FONT, 16); f_small = ImageFont.truetype(FONT, 13)
    f_mono = ImageFont.truetype(MONO, 15)
    W, PH = 1280, 360; X0, X1 = 24, W - 24; TLY = PH - 58

    # the static part of the panel: timeline blocks
    base = Image.new("RGB", (W, PH), (17, 19, 21)); g = ImageDraw.Draw(base)
    xf = lambda f: X0 + (f - 1) / T * (X1 - X0)
    for k, s in enumerate(SL):
        x0, x1 = xf(s["start_frame"]), xf(s["end_frame"] + 1)
        col = hexrgb(PAL[k % len(PAL)]) if s["task_relevant"] else (70, 72, 75)
        g.rectangle([x0, TLY, max(x0 + 1, x1 - 1), TLY + 22], fill=col)
        if x1 - x0 > 22:
            g.text(((x0 + x1) / 2, TLY + 11), str(k + 1), font=f_small, fill=(255, 255, 255), anchor="mm")
    if kept < T:
        g.rectangle([xf(kept + 1), TLY - 6, X1, TLY + 28], outline=(150, 150, 150), width=1)
        g.text((X1, TLY + 34), "trimmed", font=f_small, fill=(150, 150, 150), anchor="ra")
    for x in acts:
        if not x["task_relevant"]:
            g.polygon([(xf(x["start_frame"]) - 5, TLY + 30), (xf(x["start_frame"]) + 5, TLY + 30), (xf(x["start_frame"]), TLY + 23)], fill=(230, 230, 230))
    for tt in range(0, int(T / fps) + 1, 30):
        g.text((xf(tt * fps + 1), TLY + 34), f"{tt} s", font=f_small, fill=(150, 150, 150), anchor="ma")

    rd_h, rd_c = iio.get_reader(a.head), iio.get_reader(a.chase)
    out = iio.get_writer(a.out, fps=fps / 2, codec="libx264", quality=None, ffmpeg_params=["-crf", "23", "-preset", "medium", "-movflags", "+faststart", "-pix_fmt", "yuv420p"], macro_block_size=1)
    hand_txt = lambda h: ("not in view, " if not h["visible"] else "") + (f"{h['grasp_state']} {h['object']}" if h["grasp_state"] else "free")
    for i, (fh, fc) in enumerate(zip(rd_h, rd_c)):
        if i >= T:
            break
        if i % 2 or i < a.start * fps:
            continue
        if a.limit and i >= (a.start + a.limit) * fps:
            break
        f = i + 1; k = sub_of[f]; s = SL[k]
        top = np.hstack([np.asarray(Image.fromarray(fh).resize((640, 360))), np.asarray(Image.fromarray(fc).resize((640, 360)))])
        pan = base.copy(); g = ImageDraw.Draw(pan)
        col = hexrgb(PAL[k % len(PAL)]) if s["task_relevant"] else (150, 150, 150)
        g.rectangle([X0, 18, X0 + 8, 46], fill=col)
        g.text((X0 + 18, 16), f"Sub task {k + 1}/{len(SL)}: {s['sub_task']}", font=f_big, fill=(240, 240, 240))
        g.text((X1, 20), f"t = {i / fps:6.2f} s   frame {f}", font=f_mono, fill=(170, 170, 170), anchor="ra")
        now = [x for x in acts if x["start_frame"] <= f <= x["end_frame"]]
        main = [x for x in now if x["task_relevant"]]; side = [x for x in now if not x["task_relevant"]]
        y = 58
        if main:
            m = main[-1]
            g.text((X0, y), "Action: " + m["label"], font=f_mid, fill=(240, 240, 240)); y += 24
            for line in textwrap.wrap(m["description"], 120)[:2]:
                g.text((X0, y), line, font=f_txt, fill=(200, 200, 200)); y += 21
        for x in side:
            g.text((X0, y), f"▲ also: {x['label']} (incidental)", font=f_txt, fill=(235, 200, 120)); y += 21
        smp = samples[int(skeys[np.searchsorted(skeys, f, side="right") - 1])]
        y = max(y + 6, 146)
        g.text((X0, y), "Left hand: " + hand_txt(smp["hands"]["left"]), font=f_txt, fill=(200, 220, 255))
        g.text((W // 2, y), "Right hand: " + hand_txt(smp["hands"]["right"]), font=f_txt, fill=(200, 220, 255)); y += 24
        for line in textwrap.wrap("Interaction: " + smp["interaction"], 135)[:2]:
            g.text((X0, y), line, font=f_txt, fill=(220, 220, 220)); y += 21
        objs = ", ".join(f"{o['name']} ({o['state']})" for o in smp["objects"][:7])
        for line in textwrap.wrap("In view: " + (objs or "nothing tracked"), 150)[:2]:
            g.text((X0, y), line, font=f_small, fill=(160, 160, 160)); y += 18
        px = xf(f); g.line([px, TLY - 8, px, TLY + 26], fill=(255, 255, 255), width=3)
        out.append_data(np.vstack([top, np.asarray(pan)]))
    out.close()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    if a.small:
        for crf in (30, 32, 34, 36):
            subprocess.run([ff, "-loglevel", "error", "-y", "-i", a.out, "-vf", "scale=854:480", "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", "-an", a.small], check=True)
            if Path(a.small).stat().st_size < 14e6:
                break
    if a.slices:
        sd = Path(a.slices); sd.mkdir(parents=True, exist_ok=True)
        for k, s in enumerate(SL):
            name = f"{k + 1:02d}_" + "".join(ch if ch.isalnum() else "_" for ch in s["sub_task"].lower()).strip("_")[:60] + ".mp4"
            subprocess.run([ff, "-loglevel", "error", "-y", "-ss", f"{(s['start_frame'] - 1) / fps:.2f}", "-to", f"{s['end_frame'] / fps:.2f}", "-i", a.out,
                            "-c:v", "libx264", "-crf", "26", "-preset", "fast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(sd / name)], check=True)
            s["clip"] = f"slices/{name}"
        (sd / "slices.json").write_text(json.dumps(SL, indent=1))
    print(f"[process video] {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)" + (f", {a.small} ({Path(a.small).stat().st_size / 1e6:.1f} MB)" if a.small else "")
          + (f", {len(SL)} slice clips in {a.slices}" if a.slices else ""), flush=True)


if __name__ == "__main__":
    main()
