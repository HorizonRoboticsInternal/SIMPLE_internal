#!/usr/bin/env python
"""6 x 5 grid of a level set's 30 head-camera frames (frames/epNN.png) -> one JPEG, for docs/simple_scenes_readme.
    .venv/bin/python scripts/level_sets/grid30.py <set dir> <out.jpg> [--cols 6] [--w 320]"""
import argparse, sys
from pathlib import Path
from PIL import Image, ImageDraw
ap = argparse.ArgumentParser(); ap.add_argument("set"); ap.add_argument("out"); ap.add_argument("--cols", type=int, default=6); ap.add_argument("--w", type=int, default=320)
a = ap.parse_args(); fr = sorted(Path(a.set).glob("frames/ep*.png")); assert fr, a.set
w = a.w; h = round(w * 9 / 16); rows = -(-len(fr) // a.cols); grid = Image.new("RGB", (a.cols * w, rows * h), "black")
for i, p in enumerate(fr):
    im = Image.open(p).convert("RGB").resize((w, h)); d = ImageDraw.Draw(im); d.rectangle([0, 0, 34, 16], fill=(0, 0, 0)); d.text((4, 2), f"{i + 1}", fill=(255, 255, 255))
    grid.paste(im, ((i % a.cols) * w, (i // a.cols) * h))
grid.save(a.out, quality=86); print(f"[grid] {len(fr)} frames -> {a.out} {grid.size}")
