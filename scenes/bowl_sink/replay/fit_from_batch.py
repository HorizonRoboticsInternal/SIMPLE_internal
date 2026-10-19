#!/usr/bin/env python3
"""Fit the sink's position from a replay batch of the 2026-09-18 teleop episodes.

The walking in the replay does not depend on where the basin is, so one run per episode is enough:
take the point where the hand opens (the sim's release), then choose the basin position that catches
the most of those points.  Run after replay/run_fit_batch.sh.

    python replay/fit_from_batch.py [--tag fit] [--basin 0.54 0.33] [--bowl-r 0.075]
"""
import argparse, json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAY = json.load(open(HERE.parent / "layout.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="fit")
    ap.add_argument("--basin", type=float, nargs=2, default=[LAY["sink"]["basin"][0], LAY["sink"]["basin"][1]])
    ap.add_argument("--bowl-r", type=float, default=LAY["bowl"]["diameter"] / 2)
    ap.add_argument("--lift", type=float, default=0.10, help="a grasp counts when the bowl rises this far")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(HERE / f"batch_{a.tag}.jsonl") if l.strip()]
    ok = [r for r in rows if not r.get("error")]
    print(f"episodes replayed: {len(ok)}   (errors: {len(rows) - len(ok)})")
    grasped = [r for r in ok if (r.get("lift") or 0) >= a.lift]
    rel = [r for r in grasped if r.get("release")]
    print(f"  grasped (lift >= {a.lift:.2f} m): {len(grasped)}/{len(ok)}")
    print(f"  and released:                    {len(rel)}/{len(ok)}")
    if not rel:
        return
    R = np.array([r["release"]["xy"] for r in rel])
    Z = np.array([r["release"]["z"] for r in rel])
    base = np.array([r["basin"][:2] for r in rel])          # the basin each run was built with
    # everything in the frame of that episode's own scene: offset of the release from the basin used
    d = R - base
    print("\nrelease point minus the basin it was built with (m):")
    for k, lab in ((0, "along the sink counter (x)"), (1, "across it (y)")):
        v = d[:, k]
        print(f"  {lab:28s} p10 {np.percentile(v,10):+.3f}  p50 {np.percentile(v,50):+.3f}  p90 {np.percentile(v,90):+.3f}  sd {v.std():.3f}")
    print(f"  release height                p10 {np.percentile(Z,10):.3f}  p50 {np.percentile(Z,50):.3f}  p90 {np.percentile(Z,90):.3f} m")

    hx, hy = a.basin[0] / 2 - a.bowl_r, a.basin[1] / 2 - a.bowl_r
    def caught(dx, dy):
        return (np.abs(d[:, 0] - dx) < hx) & (np.abs(d[:, 1] - dy) < hy)
    grid_x = np.arange(-0.60, 0.61, 0.005); grid_y = np.arange(-0.40, 0.41, 0.005)
    best = max(((int(caught(x, y).sum()), x, y) for x in grid_x for y in grid_y), key=lambda t: t[0])
    n_best, dx, dy = best
    print(f"\nbest basin shift from where these runs had it: x {dx:+.3f} m, y {dy:+.3f} m")
    print(f"  catches {n_best}/{len(rel)} of the released bowls ({n_best/len(ok)*100:.0f}% of all {len(ok)} episodes)")
    print(f"  as built (no shift): {int(caught(0, 0).sum())}/{len(rel)}")
    along_now = LAY["layout"]["sink_along"]
    print(f"\n  => sink_along {along_now:.3f} - ({dx:+.3f}) = {along_now - dx:.3f} m   (the 127 on the drawing, to the basin's far edge)")
    print(f"  => the sink counter moves {dy:+.3f} m across, i.e. robot_from_left {LAY['layout']['left_y']:.3f} + ({dy:+.3f}) = {LAY['layout']['left_y'] + dy:.3f} m")
    for bl, ba in ((a.basin[0], a.basin[1]), (0.60, 0.40), (0.70, 0.45)):
        hx2, hy2 = bl / 2 - a.bowl_r, ba / 2 - a.bowl_r
        c = int(((np.abs(d[:, 0] - dx) < hx2) & (np.abs(d[:, 1] - dy) < hy2)).sum())
        print(f"  with a {bl:.2f} x {ba:.2f} m basin: {c}/{len(rel)}")
    json.dump(dict(tag=a.tag, n=len(ok), grasped=len(grasped), released=len(rel), caught=n_best,
                   shift=[float(dx), float(dy)], sink_along=float(along_now - dx),
                   robot_from_left=float(LAY["layout"]["left_y"] + dy)),
              open(HERE / f"fit_from_batch_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
