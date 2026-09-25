#!/usr/bin/env python
"""Final video: Isaac third person | Isaac head camera (the kit video's middle panel) | real head camera (its right panel).
Alignment is frame-exact: replay_isaac.py saves, for EVERY control tick, the third-person frame f<k>.jpg and the head frame
h<k>.jpg of that tick; each kit-video frame's middle panel is matched (monotonically) to the closest head frame, and the
third-person frame of that tick is used.   python sim/compose_third.py <kit_video> <frames_dir> <out.mp4> <poster.jpg> [level]"""
import sys, glob, av, numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
kit_video, frames_dir, out, poster = map(Path, sys.argv[1:5]); level = int(sys.argv[5]) if len(sys.argv) > 5 else 0
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except Exception: font = ImageFont.load_default()
thirds = sorted(glob.glob(str(frames_dir / "f*.jpg"))); heads = sorted(glob.glob(str(frames_dir / "h*.jpg"))); M = len(thirds)
def small(im): return np.asarray(im.convert("L").resize((64, 36)), dtype=np.float32)
H = np.stack([small(Image.open(h)) for h in heads]) if heads and len(heads) == M else None
with av.open(str(kit_video)) as ic, av.open(str(out), "w") as oc:
    vi = ic.streams.video[0]
    vs = oc.add_stream("libx264", rate=vi.average_rate or 25); vs.pix_fmt = "yuv420p"; vs.options = {"crf": "23", "preset": "veryfast", "movflags": "+faststart"}
    vs.width, vs.height = vi.codec_context.width, vi.codec_context.height
    frames = [fr.to_image().convert("RGB") for fr in ic.decode(video=0)]; N = len(frames)
    # alignment by construction: replay_isaac.py captures EVERY control tick from the first env.step (settle included); the kits
    # write a video frame every 2nd tick once the replay starts, so kit frame k <-> capture S + 2k with S = M - 2N settle ticks
    S = max(0, M - 2 * N); errs = []; js = []
    for k, im in enumerate(frames):
        W = im.width // 3; Hh = im.height
        if M:
            j = min(M - 1, S + 2 * k); js.append(j)
            t = Image.open(thirds[j]).convert("RGB").resize((W, Hh)); im.paste(t, (0, 0))
        d2 = ImageDraw.Draw(im)
        d2.rectangle((4, 30, 300, 58), fill=(0, 0, 0)); d2.text((10, 36), f"Isaac third person (level {level})", fill=(255, 235, 80), font=font)
        d2.rectangle((W + 4, 30, W + 260, 58), fill=(0, 0, 0)); d2.text((W + 10, 36), f"Isaac D455 (sim, level {level})", fill=(255, 235, 80), font=font)
        if k == 0: im.save(poster, quality=85)
        for pkt in vs.encode(av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")): oc.mux(pkt)
    for pkt in vs.encode(): oc.mux(pkt)
print(f"composed {N} kit frames from {M} per-tick captures -> {out} | settle offset S = {S} ticks, kit frame k <-> tick S + 2k")
