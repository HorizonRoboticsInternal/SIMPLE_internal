#!/usr/bin/env python3
"""Replace the three kits' "Deployed model in this scene" blocks of docs/simple_scenes_readme/index.html (the level-0,
one-episode runs of 2026-09-24) with the deploy presets' results on the 30-scene level-3 sets, run through the standard
SIMPLE evaluator in Isaac (holobrain_g1_deploy/sim/run_lv3_eval.sh).

    python build_lv3_blocks.py <report_root> [--page docs/simple_scenes_readme] [--apply] [--max-videos 2]

Without --apply it prints the new blocks. With --apply it backs index.html up (index_<date>_before_lv3_eval.html), composes
one combined Isaac video (third person | head camera) per chosen episode with sim/compose_views.py, copies them with
content-hashed names into the page's vid/ + img/, and rewrites the blocks. Publishing is a separate rsync (see README).
"""
import argparse, hashlib, html, json, re, shutil, subprocess, sys
from datetime import date
from pathlib import Path

SIMPLE = Path.home() / "wrk" / "SIMPLE"
PKG = Path.home() / "wrk" / "robot_orchard_deploy" / "holobrain_g1_deploy"
KITS = {"bottle_bin": ("chipcan_nativec9", "G1WholebodyBottleBinTeleop-v0"),
        "bowl_sink": ("bowltosink_c9", "G1WholebodyBowlSinkTeleop-v0"),
        "coffee_cart": ("cart_c19", "G1WholebodyCoffeeCartTeleop-v0")}
# The kits' ordered gates (the keys the gates_ep json uses in first_true_t); the other booleans are state flags.
KIT_GATES = {"bottle_bin": ["grasped", "at_bin", "placed"],
             "bowl_sink": ["at_bowl", "grasped", "at_basin", "placed"],
             "coffee_cart": ["cart_pushed_ever", "cup_lifted_ever", "success"]}   # success = placed and released (gate 3)
GATE_LABEL = {"grasped": "grasped", "at_bin": "at bin", "placed": "placed", "at_bowl": "at bowl", "at_basin": "at sink",
              "cart_pushed_ever": "cart pushed", "cup_lifted_ever": "cup lifted", "success": "placed"}


def sha8(p: Path) -> str:
    h = hashlib.sha1()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def load_kit(root: Path, kit: str):
    preset, env = KITS[kit]
    d = root / f"{kit}_{preset}"
    eps = []
    for g in sorted(d.glob("gates_ep*.json"), key=lambda p: int(re.search(r"ep(\d+)", p.name).group(1))):
        i = int(re.search(r"ep(\d+)", g.name).group(1))
        j = json.load(open(g))
        gates = KIT_GATES[kit]
        eps.append(dict(i=i, success=j.get("success_t") is not None, success_t=j.get("success_t"),
                        first=j.get("first_true_t", {}), final=j["final"], seconds=j.get("seconds"), gates=gates))
    stats = {}
    st = d / "eval_stats.txt"
    if st.exists():
        for line in open(st):
            m = re.match(r"episode_(\d+): (True|False)", line.strip())
            if m:
                stats[int(m.group(1))] = m.group(2) == "True"
    for e in eps:                      # the evaluator's verdict is the authority; gates_ep is its live view
        if e["i"] in stats:
            e["success"] = stats[e["i"]]
    return d, eps


def compose(d: Path, ep: int, kit: str, preset: str, out_mp4: Path, poster: Path) -> bool:
    frames = d / f"frames_ep{ep}"
    if not frames.is_dir() or not any(frames.glob("f*.jpg")):
        return False
    cmd = [str(SIMPLE / ".venv/bin/python"), str(PKG / "sim/compose_views.py"), str(frames), str(out_mp4), "--hz", "50",
           "--every", "2", "--fps", "25", "--label", f"{kit} · {preset} · Isaac, standard eval, level 3", "--poster", str(poster)]
    env = dict(**__import__("os").environ); env["PATH"] = f"{Path.home()}/wrk/simple_validate/.ffbin:" + env["PATH"]
    return subprocess.run(cmd, env=env, capture_output=True).returncode == 0 and out_mp4.exists()


def table(eps):
    gates = eps[0]["gates"] if eps else []
    cols = "".join(f"<th>{e['i'] + 1}</th>" for e in eps)
    rows = []
    for g in gates:
        cells = []
        for e in eps:
            t = e["first"].get(g)
            cells.append(f"<td title='{t:.1f} s'>✓</td>" if t is not None else "<td>·</td>")
        rows.append(f"<tr><td>{html.escape(GATE_LABEL.get(g, g))}</td>{''.join(cells)}</tr>")
    rows.append("<tr><td><b>success</b></td>" + "".join("<td><b>✓</b></td>" if e["success"] else "<td>·</td>" for e in eps) + "</tr>")
    return f"<div class='tbl'><table class='rtab'><thead><tr><th>scene</th>{cols}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def model_text(old_block: str) -> str:
    """The existing block's model description, up to '; run through'."""
    m = re.search(r"(model <code>.*?)(?:; run through)", old_block, re.S)
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--page", default=str(SIMPLE / "docs/simple_scenes_readme"))
    ap.add_argument("--apply", action="store_true"); ap.add_argument("--max-videos", type=int, default=2)
    ap.add_argument("--kits", default="", help="comma-separated subset of kits to (re)build; default all with results")
    a = ap.parse_args()
    root, page = Path(a.root).resolve(), Path(a.page).resolve()
    index = page / "index.html"; s = open(index, encoding="utf-8").read()
    summary = {}
    only = {k for k in a.kits.split(",") if k}
    for kit, (preset, env) in KITS.items():
        if only and kit not in only:
            continue
        d, eps = load_kit(root, kit)
        if not eps:
            print(f"[{kit}] no episodes under {d}", file=sys.stderr); continue
        n_ok = sum(e["success"] for e in eps); n = len(eps)
        gate_counts = {g: sum(1 for e in eps if e["first"].get(g) is not None) for g in eps[0]["gates"]}
        m = re.search(rf"<!-- pipetest-{kit} -->(.*?)<!-- pipetest-{kit} -->", s, re.S)
        if not m:
            print(f"[{kit}] block anchor not found", file=sys.stderr); continue
        mtext = model_text(m.group(1))
        # example videos: successes first, then scene 0
        picks = [e["i"] for e in eps if e["success"]][: a.max_videos]
        if not picks:                  # no success: the furthest attempt (most gates reached, then the highest lift)
            best = max(eps, key=lambda e: (len(e["first"]), float(e["final"].get("max_lift") or 0.0)))
            picks = [best["i"]]
        figs = []
        for ep in picks:
            src = f"vid/lv3_{kit}_ep{ep:02d}.mp4"; pst = f"img/poster_lv3_{kit}_ep{ep:02d}.jpg"
            if a.apply:
                tmp_mp4, tmp_jpg = page / "vid" / f"_tmp_{kit}_{ep}.mp4", page / "img" / f"_tmp_{kit}_{ep}.jpg"
                if compose(d, ep, kit, preset, tmp_mp4, tmp_jpg):
                    hv, hp = sha8(tmp_mp4), sha8(tmp_jpg)
                    src, pst = f"vid/lv3_{kit}_ep{ep:02d}.{hv}.mp4", f"img/poster_lv3_{kit}_ep{ep:02d}.{hp}.jpg"
                    shutil.move(tmp_mp4, page / src); shutil.move(tmp_jpg, page / pst)
                else:
                    print(f"[{kit}] compose failed for ep{ep}", file=sys.stderr); continue
            e = next(x for x in eps if x["i"] == ep)
            if e["success"]:
                verdict = f"success at {e['success_t']:.1f} s" if e["success_t"] else "success"
            else:
                reached = [GATE_LABEL.get(g, g) for g in e["gates"] if e["first"].get(g) is not None]
                lift = e["final"].get("max_lift")
                verdict = ("no success; gates reached: " + ", ".join(reached) if reached else "no success, no gate reached") + \
                          (f" (object lifted at most {lift * 100:.0f} cm)" if isinstance(lift, (int, float)) else "")
            figs.append(f"<figure class='wide'><video controls preload='metadata' poster='{pst}' src='{src}' style='aspect-ratio:1280/480'></video>"
                        f"<figcaption><b>Level-3 scene {ep + 1}</b> &mdash; {verdict}: left an Isaac third-person camera, right the head camera the policy saw.</figcaption></figure>")
        secs = sorted(e["seconds"] for e in eps if e.get("seconds"))
        ep_len = f"{secs[len(secs) // 2]:.0f} s per episode (stand-up included)" if secs else "34 s per episode"
        gates_html = " &middot; ".join(f"<span class='gate'>{html.escape(GATE_LABEL.get(g, g))}</span> {c}/{n}" for g, c in gate_counts.items())
        block = (f"<!-- pipetest-{kit} --><figure class='wide'><figcaption><b>Deployed model on the level-3 set ({date.today():%Y-%m-%d}, standard SIMPLE eval in Isaac, {n} scenes)</b> &mdash; "
                 f"{mtext}; run through <code>simple.cli.eval_decoupled_wbc</code> with SIMPLE's <code>psi0_decoupled_wbc</code> agent (HTTP /act, a whole chunk per query; "
                 f"8 demasking steps, horizon 15) on <b>all {n} level-3 scenes</b> with Isaac rendering, {ep_len}. "
                 f"<b>Success {n_ok}/{n}.</b> Gates reached: {gates_html}.</figcaption></figure>"
                 f"{''.join(figs)}{table(eps)}<!-- pipetest-{kit} -->")
        s = s[: m.start()] + block + s[m.end():]
        summary[kit] = dict(preset=preset, episodes=n, success=n_ok, gates=gate_counts, videos=picks)
        print(f"[{kit}] {n_ok}/{n} success; gates {gate_counts}; videos {picks}")
    if a.apply:
        bak = page / f"index_{date.today():%Y%m%d}_before_lv3_eval.html"
        if not bak.exists():
            shutil.copy(index, bak)
        open(index, "w", encoding="utf-8").write(s)
        json.dump(summary, open(Path(__file__).with_name("lv3_summary.json"), "w"), indent=1)
        print(f"applied -> {index} (backup {bak.name})")
    else:
        print("(dry run; --apply to write)")


if __name__ == "__main__":
    main()
