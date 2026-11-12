"""Process video of an episode sliced the RoboAtom way (2026-11-05): the head camera and the chase camera with the whole-body
segment, the hand lane, the lower-body lane and the cut rule in force playing along, over a three-lane timeline.

    .venv/bin/python scripts/holomotion_v14_checks/roboatom_process_video.py docs/_scan/roboatom_k7ep0/segments.json \
        <head.mp4> <chase.mp4> <out.mp4> [--small <small.mp4>] [--start S] [--limit S]

segments.json comes from roboatom_slice_episode.py (whole-body segments with 1-based 50 Hz frames, the upper and lower lanes,
the idle tail). Every second recorded frame (25 fps, 1280x720): top, the head camera as recorded and the third-person chase
camera; below, the current whole-body segment (k of N) with its skill and lane, its description, what the hands and the legs are
doing right now, the rule that decided the label, and a timeline with three lanes (whole body, hands, legs) and the playhead.
Skill colours are read from the page builder (docs/_scan/roboatom_k7ep0/build.py, the SC dict) so the video matches the page.
--small also writes an 854x480 copy under 14 MB.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import textwrap
from pathlib import Path

import imageio.v2 as iio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"; BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
LANE_NAME = {"upper": "hand skill (upper lane)", "lower": "lower-body skill (lower lane)"}


def hexrgb(h):
    return tuple(int(h[i:i + 2], 16) for i in (1, 3, 5))


def skill_colours(seg_path: Path) -> dict:
    """the page's SC dict (skill -> hex), parsed from its builder next to the segments file; fallback = a fixed copy"""
    try:
        s = (seg_path.parent / "build.py").read_text()
        return ast.literal_eval(re.search(r"^SC\s*=\s*(\{.*?\})\s*$", s, re.S | re.M).group(1))
    except Exception:
        return {"Move": "#2e8b57", "Pick": "#1f6f5f", "Step": "#b5533c", "Wipe": "#8a5fbf", "Handover": "#0b3d91", "Open": "#c98a1b", "Pull": "#d9772a",
                "Place": "#6a3d9a", "Push": "#e0a84f", "Close": "#8c6d31", "Squat": "#c2185b", "Turn": "#3b8ea5", "adjust": "#aab2b8"}


def hatch(g, box, colour, step=7):
    x0, y0, x1, y1 = box
    g.rectangle(box, outline=colour, width=1)
    for x in range(int(x0 - (y1 - y0)), int(x1), step):
        g.line([(max(x0, x), min(y1, y0 + max(0, x0 - x))), (min(x1, x + (y1 - y0)), max(y0, y1 - max(0, x + (y1 - y0) - x1)))], fill=colour, width=1)


def rule_text(seg, lower_now, upper_now):
    """why the whole-body label is what it is (the page's cut rules, section 1)"""
    sk = seg["skill"]
    if seg["lane"] == "upper":
        base = f"Rule: the hands do more than hold, so the hand skill ({sk}) labels the whole body"
        if lower_now and lower_now["skill"] == "adjust":
            return base + "; the stance adjustment (< 0.30 m net) is absorbed, it never cuts"
        if lower_now and lower_now["skill"] == "Squat":
            return base + "; a Squat never cuts, it stays inside the hand skill"
        if lower_now and lower_now["skill"] == "Move":
            return base + "; a short walk inside the skill is absorbed (< 0.30 m net)"
        if lower_now and lower_now["skill"] == "Step":
            return base + "; the foot stays on the pedal, so the Step continues inside it"
        return base
    if sk == "Step":
        return "Rule: a foot on the pedal always cuts: Step, even with the plate in hand"
    if sk == "Move":
        return "Rule: the hands only hold, so the lower-body skill (Move) labels the whole body" if seg.get("holding") else \
            "Rule: the hands are free and the base translates >= 0.30 m net, so Move labels the whole body"
    return f"Rule: the lower-body skill ({sk}) labels the whole body"


def lower_text(lower_now, seg, t):
    if lower_now is None:
        return "Legs: standing"
    k = lower_now["skill"]
    if k == "Move":
        return f"Legs: walking (Move), {lower_now.get('path', 0):.1f} m path, {lower_now.get('net', 0):.1f} m net, {abs(lower_now.get('turn', 0)):.0f} deg turned"
    if k == "Step":
        return f"Legs: foot on the bin pedal (Step), {lower_now.get('n_press', '?')} presses, lid to {lower_now.get('lid_after', 0):.2f} rad"
    if k == "Squat":
        return f"Legs: Squat, pelvis down to {lower_now.get('zmin', 0):.2f} m (standing 0.78 m), inside {seg['skill']}"
    if k == "adjust":
        return f"Legs: stance adjustment, {lower_now.get('net', 0):.2f} m net, {abs(lower_now.get('turn', 0)):.0f} deg (absorbed, not a segment)"
    return f"Legs: {k}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("segments"); ap.add_argument("head"); ap.add_argument("chase"); ap.add_argument("out")
    ap.add_argument("--small"); ap.add_argument("--start", type=float, default=0); ap.add_argument("--limit", type=float, default=0)
    a = ap.parse_args()
    SP = Path(a.segments); D = json.load(open(SP)); SEG = D["segments"]; UP = D["lanes"]["upper"]; LOW = D["lanes"]["lower"]; IDLE = D.get("idle", [])
    fps = D["config"]["fps"]; T = D["config"]["frames"]; SC = {k: hexrgb(v) for k, v in skill_colours(SP).items()}
    kept = SEG[-1]["end_frame"]
    seg_of = np.full(T + 1, -1, int)
    for k, s in enumerate(SEG):
        seg_of[s["start_frame"]:s["end_frame"] + 1] = k
    f_big = ImageFont.truetype(BOLD, 24); f_mid = ImageFont.truetype(BOLD, 17); f_txt = ImageFont.truetype(FONT, 16); f_small = ImageFont.truetype(FONT, 13)
    f_tag = ImageFont.truetype(BOLD, 13); f_mono = ImageFont.truetype(MONO, 15)
    W, PH = 1280, 360; X0, X1 = 92, W - 24
    xf = lambda f: X0 + (f - 1) / T * (X1 - X0)
    xt = lambda t: X0 + t * fps / T * (X1 - X0)
    GREY, HOLD, INK, MUTE = (70, 72, 75), (92, 98, 104), (240, 240, 240), (160, 160, 160)

    # ---- static panel: the three-lane timeline
    base = Image.new("RGB", (W, PH), (17, 19, 21)); g = ImageDraw.Draw(base)
    Y_WB, Y_UP, Y_LO, H_WB, H_LN = PH - 104, PH - 76, PH - 54, 24, 18
    g.text((X0 - 10, Y_WB + H_WB / 2), "whole body", font=f_small, fill=MUTE, anchor="rm")
    g.text((X0 - 10, Y_UP + H_LN / 2), "hands", font=f_small, fill=MUTE, anchor="rm")
    g.text((X0 - 10, Y_LO + H_LN / 2), "legs", font=f_small, fill=MUTE, anchor="rm")
    for s in SEG:
        x0, x1 = xf(s["start_frame"]), xf(s["end_frame"] + 1)
        g.rectangle([x0, Y_WB, max(x0 + 1, x1 - 1), Y_WB + H_WB], fill=SC.get(s["skill"], GREY))
        if x1 - x0 > 16:
            g.text(((x0 + x1) / 2, Y_WB + H_WB / 2), str(s["index"]), font=f_small, fill=(255, 255, 255), anchor="mm")
    for idl in IDLE:
        hatch(g, [xt(idl["start_s"]), Y_WB, X1, Y_WB + H_WB], MUTE)
    # hands lane: holding state (grey) under the skills
    hold_from = None
    for s in SEG:            # a continuous "holding" state spans the segments that report the same held object
        if s.get("holding") and hold_from is None: hold_from = s["start_frame"]
        if not s.get("holding") and hold_from is not None:
            g.rectangle([xf(hold_from), Y_UP + 4, xf(s["start_frame"]), Y_UP + H_LN - 4], fill=HOLD); hold_from = None
    if hold_from is not None:
        g.rectangle([xf(hold_from), Y_UP + 4, xf(kept + 1), Y_UP + H_LN - 4], fill=HOLD)
    for u in UP:
        g.rectangle([xt(u["start_s"]), Y_UP, max(xt(u["start_s"]) + 1, xt(u["end_s"]) - 1), Y_UP + H_LN], fill=SC.get(u["skill"], GREY))
    for l in LOW:
        box = [xt(l["start_s"]), Y_LO, max(xt(l["start_s"]) + 1, xt(l["end_s"]) - 1), Y_LO + H_LN]
        if l["skill"] == "adjust": hatch(g, box, SC.get("adjust", MUTE))
        else: g.rectangle(box, fill=SC.get(l["skill"], GREY))
    for tt in range(0, int(T / fps) + 1, 30):
        g.text((xt(tt), PH - 30), f"{tt} s", font=f_small, fill=MUTE, anchor="ma")
    g.text((X1, Y_WB - 16), "hatched = stance adjustment (absorbed) · grey under the hand lane = holding a plate (a state, not a skill)", font=f_small, fill=(120, 120, 120), anchor="rm")

    rd_h, rd_c = iio.get_reader(a.head), iio.get_reader(a.chase)
    out = iio.get_writer(a.out, fps=fps / 2, codec="libx264", quality=None, ffmpeg_params=["-crf", "23", "-preset", "medium", "-movflags", "+faststart", "-pix_fmt", "yuv420p"], macro_block_size=1)
    now_in = lambda L, t: next((x for x in L if x["start_s"] <= t <= x["end_s"]), None)
    for i, (fh, fc) in enumerate(zip(rd_h, rd_c)):
        if i >= T: break
        if i % 2 or i < a.start * fps: continue
        if a.limit and i >= (a.start + a.limit) * fps: break
        f = i + 1; t = i / fps; k = seg_of[f]
        top = np.hstack([np.asarray(Image.fromarray(fh).resize((640, 360))), np.asarray(Image.fromarray(fc).resize((640, 360)))])
        pan = base.copy(); g = ImageDraw.Draw(pan)
        g.text((X1, 20), f"t = {t:6.2f} s   frame {f}", font=f_mono, fill=MUTE, anchor="ra")
        if k < 0:
            g.rectangle([24, 18, 32, 46], fill=GREY)
            g.text((42, 16), "Idle tail (not a segment)", font=f_big, fill=INK)
            idl = now_in(IDLE, t)
            if idl:
                for n, line in enumerate(textwrap.wrap(idl.get("text", ""), 120)[:2]): g.text((24, 58 + 21 * n), line, font=f_txt, fill=(200, 200, 200))
        else:
            s = SEG[k]; col = SC.get(s["skill"], GREY); up = now_in(UP, t); lo = now_in(LOW, t)
            g.rectangle([24, 18, 32, 46], fill=col)
            g.text((42, 16), f"Segment {s['index']}/{len(SEG)}: {s['skill']}", font=f_big, fill=INK)
            tw = g.textlength(f"Segment {s['index']}/{len(SEG)}: {s['skill']}", font=f_big)
            g.rounded_rectangle([42 + tw + 14, 22, 42 + tw + 14 + g.textlength(LANE_NAME[s["lane"]], font=f_tag) + 14, 42], radius=5, fill=col)
            g.text((42 + tw + 21, 32), LANE_NAME[s["lane"]], font=f_tag, fill=(255, 255, 255), anchor="lm")
            y = 58
            for line in textwrap.wrap(s["description"], 118)[:2]:
                g.text((24, y), line, font=f_txt, fill=(215, 215, 215)); y += 21
            y = max(y, 100) + 4
            # hands now
            if up:
                hands = f"Hands: {up['skill']}" + (f" ({up['object']})" if up.get("object") else "") + \
                        (f", the other hand holds {s['holding']}" if s.get("holding") and up["skill"] not in ("Pick", "Place", "Handover") else "")
            elif s.get("holding"): hands = f"Hands: holding {s['holding']} (a state, not a skill)"
            else: hands = "Hands: free"
            hl = textwrap.wrap(hands, 58)[:2]; ll = textwrap.wrap(lower_text(lo, s, t), 60)[:2]
            for n, line in enumerate(hl): g.text((24, y + 23 * n), line, font=f_mid, fill=(200, 220, 255))
            for n, line in enumerate(ll): g.text((W // 2, y + 23 * n), line, font=f_mid, fill=(200, 235, 200))
            y += 23 * max(len(hl), len(ll)) + 4
            for line in textwrap.wrap(rule_text(s, lo, up), 125)[:2]:
                g.text((24, y), line, font=f_txt, fill=(235, 200, 120)); y += 21
            g.text((24, y + 2), f"{s['start_s']:.1f}–{s['end_s']:.1f} s · {s['seconds']:.1f} s · progress {min(1, max(0, (t - s['start_s']) / max(s['seconds'], 1e-6))) * 100:3.0f} %", font=f_small, fill=MUTE)
            nxt = SEG[k + 1] if k + 1 < len(SEG) else None
            if nxt:
                g.rectangle([24, Y_WB - 42, 30, Y_WB - 26], fill=SC.get(nxt["skill"], GREY))
                g.text((38, Y_WB - 44), f"Next cut in {max(0, s['end_s'] - t):4.1f} s \u2192 {nxt['index']} {nxt['skill']}: " + textwrap.shorten(nxt["description"], 105, placeholder="\u2026"), font=f_txt, fill=(200, 200, 200))
            else:
                g.text((24, Y_WB - 44), f"Last segment; the idle tail starts at {s['end_s']:.1f} s", font=f_txt, fill=(200, 200, 200))
        px = xf(f); g.line([px, Y_WB - 4, px, Y_LO + H_LN + 4], fill=(255, 255, 255), width=3)
        out.append_data(np.vstack([top, np.asarray(pan)]))
    out.close()
    if a.small:
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        for crf in (30, 32, 34, 36):
            subprocess.run([ff, "-loglevel", "error", "-y", "-i", a.out, "-vf", "scale=854:480", "-c:v", "libx264", "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p",
                            "-movflags", "+faststart", "-an", a.small], check=True)
            if Path(a.small).stat().st_size < 14e6: break
    print(f"[roboatom video] {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)" + (f", {a.small} ({Path(a.small).stat().st_size / 1e6:.1f} MB)" if a.small else ""), flush=True)


if __name__ == "__main__":
    main()
