#!/usr/bin/env python
"""30-scene level-3 grids for the README: 6 columns x 5 rows of 320x180 Isaac head-camera frames (the first frame of each
scene), scene numbers in the corner, 1920x900 JPEG per benchmark task.

    python make_lv3_grids.py <out_img_dir>
"""
import json
import sys
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]      # the SIMPLE checkout
BENCH = ["G1WholebodyTabletopGraspMP-v0", "G1WholebodyBendPickMP-v0", "G1WholebodyXMovePickTeleop-v0", "G1WholebodyHandoverTeleop-v0",
         "G1WholebodyLocomotionPickBetweenTablesTeleop-v0"]
KITS = {"bottle_bin": "G1WholebodyBottleBinTeleop-v0", "bowl_sink": "G1WholebodyBowlSinkTeleop-v0", "coffee_cart": "G1WholebodyCoffeeCartTeleop-v0"}
COLS, ROWS, TW, TH = 6, 5, 320, 180
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
except OSError:
    FONT = ImageFont.load_default()


def first_frame(mp4: Path) -> Image.Image:
    with av.open(str(mp4)) as c:
        for fr in c.decode(video=0):
            return fr.to_image().convert("RGB")
    raise RuntimeError(f"no frame in {mp4}")


def frames_for(set_dir: Path) -> list:
    n = sum(1 for l in open(set_dir / "meta" / "episodes.jsonl") if l.strip())
    pngs = sorted((set_dir / "frames").glob("ep*.png"))
    if len(pngs) >= n:
        return [Image.open(p).convert("RGB") for p in pngs[:n]]
    vids = sorted((set_dir / "videos").rglob("episode_*.mp4"))
    vids = ([v for v in vids if "ego_view" in str(v)] or [v for v in vids if "rgb_head_stereo_left" in str(v)] or vids)   # the head camera the policy sees
    return [first_frame(v) for v in vids[:n]]


def grid(imgs, out: Path) -> None:
    sheet = Image.new("RGB", (COLS * TW, ROWS * TH), (20, 20, 20))
    for k, im in enumerate(imgs[:COLS * ROWS]):
        im = im.resize((TW, TH))
        d = ImageDraw.Draw(im)
        label = str(k + 1)
        w = d.textlength(label, font=FONT)
        d.rectangle([0, 0, w + 12, 24], fill=(0, 0, 0))
        d.text((6, 3), label, font=FONT, fill=(255, 235, 80))
        sheet.paste(im, ((k % COLS) * TW, (k // COLS) * TH))
    sheet.save(out, quality=84)
    print(f"{out.name}: {len(imgs)} scenes")


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for t in BENCH:
        d = ROOT / "data/evals_scenes_benchmark" / t / "dr-level-3"
        grid(frames_for(d), out / f"grid_{t.replace('G1Wholebody', '').replace('-v0', '')}_lv3_30.jpg")


if __name__ == "__main__":
    main()
