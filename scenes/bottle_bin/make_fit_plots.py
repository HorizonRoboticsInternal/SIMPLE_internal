#!/usr/bin/env python
"""Figures for the "fitted to all 97 episodes" section of the scene report.

  python make_fit_plots.py --grid replay/sweep/grid3cm --before replay/sweep/base_3cm \
                           --after replay/sweep/final --bin -0.31 -0.80 --out site/img

Writes fit_grid.png (can position -> how many of the 97 replays carry the can off the table),
release_cloud.png (where the 97 replays let go of the can, with the bin opening drawn on top) and
outcomes.png (per-episode outcome, before vs after).
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

import fit_layout as F

INK, GRID, OK, BAD, MID = "#1a1a1a", "#d8d8d4", "#2f7d4f", "#b03030", "#c08a2e"


def load_runs(d):
    return F.load([d])


def fig_grid(grid_dirs, out, best=None, spawn_map=None):
    """Can position -> replays that carry the can off the table; cells where the can spawns inside the robot's
    open thumbs are hatched grey (the robot flings it before the recording starts)."""
    cells = {}
    for d in grid_dirs:                              # a cell measured by more than one sweep: the first dir given wins
        per = {}
        for r in F.load([d]):
            per.setdefault(tuple(round(v, 3) for v in (r["bottle_xy_cmd"] or r["scene"]["layout_bottle_xy"])), []).append(r)
        for b, rs in per.items():
            if b not in cells and len(rs) >= 90:
                cells[b] = rs
    sm = json.load(open(spawn_map)) if spawn_map else None
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    if sm:                                           # the thumb zone, from the placement-only check
        xs, ys = np.array(sm["xs"]), np.array(sm["ys"])
        bad = np.array([[0 if sm["cells"][f"{x:.3f},{y:.3f}"]["ok"] else 1 for x in xs] for y in ys], float)
        ax.contourf(xs, ys, bad, levels=[0.5, 1.5], colors="none", hatches=["///"], zorder=1)
        ax.contour(xs, ys, bad, levels=[0.5], colors=["#888888"], linewidths=1, zorder=1)
    vals = [sum(F.carried(r) for r in rs) for rs in cells.values()]
    vmin, vmax = min(vals), max(vals)
    cmap = plt.get_cmap("YlGnBu")
    for (x, y), rs in cells.items():
        if len(rs) < 90:
            continue
        spawn_bad = sm and not sm["cells"].get(f"{x:.3f},{y:.3f}", {"ok": True})["ok"]
        v = sum(F.carried(r) for r in rs)
        col = "#d9d9d9" if spawn_bad else cmap(0.15 + 0.85 * (v - vmin) / max(1, vmax - vmin))
        ax.add_patch(Rectangle((x - 0.0048, y - 0.0048), 0.0096, 0.0096, fc=col, ec="white", lw=0.8, zorder=2))
        ax.text(x, y, "x" if spawn_bad else f"{v}", ha="center", va="center", fontsize=8.5, zorder=3,
                color="#777777" if spawn_bad else ("white" if (v - vmin) / max(1, vmax - vmin) > 0.6 else INK),
                fontweight="bold")
    ax.plot(0.245, -0.004, "o", ms=20, mfc="none", mec=BAD, mew=2.2, zorder=4, label="published can (0.245, -0.004)")
    if best:
        ax.plot(*best, "o", ms=20, mfc="none", mec=OK, mew=2.6, zorder=4, label=f"fitted can ({best[0]:.3f}, {best[1]:+.3f})")
    ax.plot([], [], "s", ms=10, mfc="#d9d9d9", mec="#888888", label="can spawns inside a thumb (invalid)")
    ax.set_xlim(0.195, 0.325); ax.set_ylim(-0.052, 0.044); ax.set_aspect("equal")
    ax.set_xlabel("can axis, distance ahead of the pelvis  (m)")
    ax.set_ylabel("left (+) / right (-) of the pelvis  (m)")
    ax.set_title("Replays (of 97) that pick the can up and carry it off, by can position", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)


def fig_binscan(scan_dir, out, best=None, old=(-0.25, -0.84), free_dir=None, yaw=None, title=None):
    """Bin centre -> cans that end in the bin (each cell = all 97 episodes replayed with the bin there).  With
    free_dir, the count is of cans in the bin whose walk the bin did not shorten (end position within 6 cm of the
    same episode's free walk)."""
    runs = F.load([scan_dir])
    free = {r["ep"]: r for r in F.load([free_dir])} if free_dir else None
    def score(rs):
        if free is None:
            return sum(bool(r.get("in_bin")) for r in rs)
        return sum(bool(r.get("in_bin")) and np.hypot(r["base_end_xy"][0] - free[r["ep"]]["base_end_xy"][0],
                                                     r["base_end_xy"][1] - free[r["ep"]]["base_end_xy"][1]) <= 0.06 for r in rs)
    cells = {}
    for r in runs:
        if yaw is not None and r.get("bin_yaw_deg", -90) != yaw:
            continue
        cells.setdefault(tuple(round(v, 3) for v in r["bin_xy_cmd"]), []).append(r)
    cells = {k: v for k, v in cells.items() if len(v) >= 90}
    vals = [score(v) for v in cells.values()]
    vmin, vmax = min(vals), max(vals)
    cmap = plt.get_cmap("YlGnBu")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for (x, y), rs in cells.items():
        v = score(rs)
        t = (v - vmin) / max(1, vmax - vmin)
        ax.add_patch(Rectangle((x - 0.0145, y - 0.0145), 0.029, 0.029, fc=cmap(0.15 + 0.85 * t), ec="white", lw=1))
        ax.text(x, y, f"{v}", ha="center", va="center", fontsize=10, fontweight="bold", color="white" if t > 0.6 else INK)
    if old:
        ax.plot(*old, "o", ms=22, mfc="none", mec=BAD, mew=2.2, label=f"previous bin ({old[0]:+.2f}, {old[1]:+.2f})")
    if best:
        ax.plot(*best, "o", ms=22, mfc="none", mec=OK, mew=2.6, label=f"chosen bin ({best[0]:+.2f}, {best[1]:+.2f})")
    xs = [k[0] for k in cells]; ys = [k[1] for k in cells]
    ax.set_xlim(min(xs) - 0.03, max(xs) + 0.03); ax.set_ylim(min(ys + ([old[1]] if old else [])) - 0.03, max(ys) + 0.03); ax.set_aspect("equal")
    ax.set_xticks(sorted(set(xs))); ax.set_yticks(sorted(set(ys + ([old[1]] if old else []))))
    ax.tick_params(labelsize=8.5)
    ax.set_xlabel("bin centre, x: toward the table  (m)"); ax.set_ylabel("bin centre, y: robot's left (+)  (m)")
    ax.set_title(title or "Cans (of 97) that end in the bin, by bin position\n(can at the fitted position)", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5, frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)


def fig_release(dirs_labels, bins, out, half=(0.1395, 0.0795), bin_wd=(0.36, 0.24), feet_dir=None):
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    if feet_dir:                                      # where the feet go in the free walk, all episodes
        for r in load_runs(feet_dir):
            fp = np.array(r.get("feet_path", []))
            if len(fp):
                ax.plot(fp[:, 1], fp[:, 2], "-", color="#b8c4d0", lw=.6, alpha=.6, zorder=1)
                ax.plot(fp[:, 3], fp[:, 4], "-", color="#b8c4d0", lw=.6, alpha=.6, zorder=1)
        ax.plot([], [], "-", color="#b8c4d0", lw=1.5, label="feet paths, free walk (97 episodes)")
    for (d, lab, col) in dirs_labels:
        runs = load_runs(d)
        P = np.array([r["release"]["xy"] for r in runs if r.get("release") and F.carried(r)], float).reshape(-1, 2)
        ax.plot(P[:, 0], P[:, 1], "o", ms=5.5, mfc=col, mec="white", mew=.6, label=f"{lab} ({len(P)} releases)", zorder=3)
    for (c, lab, col, ls) in bins:
        ax.add_patch(Rectangle((c[0] - bin_wd[0] / 2, c[1] - bin_wd[1] / 2), bin_wd[0], bin_wd[1],
                               fill=False, ec=col, lw=2, ls=ls, zorder=4, label=f"{lab} bin ({c[0]:+.2f}, {c[1]:+.2f})"))
        ax.add_patch(Rectangle((c[0] - half[0], c[1] - half[1]), 2 * half[0], 2 * half[1],
                               fill=False, ec=col, lw=1, ls=":", alpha=.8, zorder=4))
    ax.plot(0, 0, "^", ms=13, mfc="#4a6fa5", mec=INK, zorder=5)
    ax.annotate("robot start", (0, 0), (0.06, 0.06), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK, lw=.8))
    ax.set_xlabel("x: toward the table  (m)"); ax.set_ylabel("y: robot's left (+) / right (-)  (m)")
    ax.set_title("Where the replays let go of the can when nothing is in the way")
    ax.axis("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="upper right", framealpha=.95)
    ax.set_xlim(-0.95, 0.75); ax.set_ylim(-1.95, 0.25)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def fig_outcomes(before, after, out, labels=("published layout", "fitted layout"), free_dir=None):
    data = []
    for d in (before, after):
        runs = sorted(load_runs(d), key=lambda r: r["ep"])
        data.append({r["ep"]: F.outcome(r) for r in runs})
    eps = sorted(set(data[0]) | set(data[1]))
    colours = {"in_bin": OK, "on_floor": MID, "on_table": BAD, None: "#eeeeee"}
    fig, ax = plt.subplots(figsize=(11.5, 2.5))
    for row, d in enumerate(data):
        for k, e in enumerate(eps):
            ax.add_patch(Rectangle((k, -row), 0.92, 0.9, color=colours.get(d.get(e)), lw=0))
    ax.set_xlim(0, len(eps)); ax.set_ylim(-1.05, 1.0)
    ax.set_yticks([0.45, -0.55]); ax.set_yticklabels([f"{labels[0]}\n{sum(v == 'in_bin' for v in data[0].values())}/{len(eps)} in the bin",
                                                      f"{labels[1]}\n{sum(v == 'in_bin' for v in data[1].values())}/{len(eps)} in the bin"], fontsize=9)
    ax.set_xticks(np.arange(0, len(eps), 5) + 0.5); ax.set_xticklabels([eps[i] for i in range(0, len(eps), 5)], fontsize=8)
    ax.set_xlabel("episode of session 2026-09-17-02-25-56 (the 97 kept for training)", fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    hand = [Rectangle((0, 0), 1, 1, color=colours[k]) for k in ("in_bin", "on_floor", "on_table")]
    ax.legend(hand, ["can ends in the bin", "dropped on the floor", "never left the table"], ncol=3, fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, 1.45), frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", nargs="*"); ap.add_argument("--before"); ap.add_argument("--after")
    ap.add_argument("--binscan"); ap.add_argument("--spawn-map")
    ap.add_argument("--farscan"); ap.add_argument("--free"); ap.add_argument("--labels", nargs=2, default=None)
    ap.add_argument("--bin", type=float, nargs=2, default=None); ap.add_argument("--old-bin", type=float, nargs=2, default=[-0.25, -0.84])
    ap.add_argument("--best-bottle", type=float, nargs=2, default=None)
    ap.add_argument("--out", default="site/img")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if a.grid:
        fig_grid(a.grid, out / "fit_grid.png", best=a.best_bottle, spawn_map=a.spawn_map)
        print("wrote", out / "fit_grid.png")
    if a.binscan:
        fig_binscan(a.binscan, out / "fit_bin.png", best=None, old=a.old_bin,
                    title="Cans (of 97) that end in the bin, by bin position\n(the 09-23 morning scan: every one of these bins stops the robot)")
        print("wrote", out / "fit_bin.png")
    if a.farscan:
        fig_binscan(a.farscan, out / "fit_bin_far.png", best=a.bin, old=None, free_dir=a.free, yaw=-90,
                    title="Cans in the bin with the walk untouched (of 97), by bin position\n(36 cm side toward the table)")
        print("wrote", out / "fit_bin_far.png")
    if a.before or a.after:
        dl = []
        lab = a.labels or ("published layout", "fitted layout")
        if a.free:
            dl.append((a.free, "free walk, no bin", "#2f7d4f"))
        else:
            if a.before:
                dl.append((a.before, lab[0], "#b03030"))
            if a.after:
                dl.append((a.after, lab[1], "#2f7d4f"))
        bins = [(a.old_bin, "previous", BAD, "--")] + ([(a.bin, "chosen", OK, "-")] if a.bin else [])
        fig_release(dl, bins, out / "release_cloud.png", feet_dir=a.free)
        print("wrote", out / "release_cloud.png")
    if a.before and a.after:
        fig_outcomes(a.before, a.after, out / "outcomes.png", labels=tuple(a.labels) if a.labels else ("published layout", "fitted layout"))
        print("wrote", out / "outcomes.png")


if __name__ == "__main__":
    main()
