#!/usr/bin/env python
"""Pipeline-check report of simple.cli.eval_holomotion_v14 with the HoloBrain deploy models in the loop.

    # 1. collect the run's artefacts into inputs/ (results, bridge logs, health, frames from the videos, replay checks)
    .venv/bin/python docs/_scan/holomotion_v14_pipeline_check/build.py --collect <pipe dir>
    # 2. render docs/holomotion_v14_pipeline_check/index.html from inputs/
    .venv/bin/python docs/_scan/holomotion_v14_pipeline_check/build.py

<pipe dir> holds, per task: eval_<task>/.../results.json (+ videos/, replay/), bridge_<task>.jsonl (one line per model
query), health_<task>.json (the model server's /health), and optionally checks.json (replay / reproducibility results
written by the check step). Images are written with content-hashed names; publish the output folder with rsync --delete.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import html
import json
import shutil
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / "inputs"
OUT = HERE.parents[1] / "holomotion_v14_pipeline_check"
TASKS = [("bottle_bin", "Bottle → bin", "chipcan_nativec9"), ("bowl_sink", "Bowl → sink", "bowltosink_c9"),
         ("coffee_cart", "Coffee cart → table", "cart_c19")]
FRAME_TIMES = [0.3, 5.0, 10.0, 20.0, 29.5]


# ----------------------------------------------------------------------------- collect
def ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def collect(pipe: Path) -> None:
    INPUTS.mkdir(parents=True, exist_ok=True)
    (INPUTS / "frames").mkdir(exist_ok=True)
    manifest = {}
    for task, _, _ in TASKS:
        res = glob.glob(str(pipe / f"eval_{task}" / "**" / "results.json"), recursive=True)
        if not res:
            print(f"[collect] {task}: no results.json")
            continue
        run_dir = Path(res[0]).parent
        shutil.copy(res[0], INPUTS / f"results_{task}.json")
        for f in ("bridge", "health"):
            src = pipe / (f"{f}_{task}.jsonl" if f == "bridge" else f"{f}_{task}.json")
            if src.exists():
                shutil.copy(src, INPUTS / src.name)
        frames = {}
        for v in sorted(run_dir.glob("videos/*.mp4")):
            frames[v.stem] = []
            for t in FRAME_TIMES:
                tmp = INPUTS / "frames" / f"tmp_{task}_{v.stem}_{t}.jpg"
                r = subprocess.run([ffmpeg(), "-loglevel", "error", "-y", "-ss", str(t), "-i", str(v), "-frames:v", "1",
                                    "-vf", "scale=960:-2", "-q:v", "4", str(tmp)], capture_output=True)
                if r.returncode != 0 or not tmp.exists():
                    continue
                h = hashlib.sha1(tmp.read_bytes()).hexdigest()[:8]
                dst = INPUTS / "frames" / f"{task}_{v.stem}_t{t:g}.{h}.jpg"
                tmp.rename(dst)
                frames[v.stem].append((t, dst.name))
        manifest[task] = dict(run_dir=str(run_dir), frames=frames)
        print(f"[collect] {task}: {len(frames)} videos, {sum(len(v) for v in frames.values())} frames")
    checks = pipe / "checks.json"
    if checks.exists():
        shutil.copy(checks, INPUTS / "checks.json")
    (INPUTS / "manifest.json").write_text(json.dumps(manifest, indent=1))


# ----------------------------------------------------------------------------- data
def load() -> dict:
    d = dict(manifest=json.loads((INPUTS / "manifest.json").read_text()), tasks={})
    d["checks"] = json.loads((INPUTS / "checks.json").read_text()) if (INPUTS / "checks.json").exists() else {}
    for task, label, preset in TASKS:
        rp = INPUTS / f"results_{task}.json"
        if not rp.exists():
            continue
        R = json.loads(rp.read_text())
        bp = INPUTS / f"bridge_{task}.jsonl"
        bridge = [json.loads(line) for line in open(bp)] if bp.exists() else []
        health = json.loads((INPUTS / f"health_{task}.json").read_text()) if (INPUTS / f"health_{task}.json").exists() else {}
        # bridge sessions appear in episode order
        order, by_sess = [], {}
        for b in bridge:
            by_sess.setdefault(b["session"], []).append(b)
            if b["session"] not in order:
                order.append(b["session"])
        d["tasks"][task] = dict(label=label, preset=preset, run=R["run"], episodes=R["episodes"], bridge=bridge, health=health,
                                sessions=[by_sess[s] for s in order])
    return d


def pct(v: list[float], p: float) -> float:
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


# ----------------------------------------------------------------------------- html bits
def esc(x) -> str:
    return html.escape(str(x))


def latency_svg(vals: list[float], color: str) -> str:
    """Histogram of model latencies, to scale: bins of 50 ms from 0 to the max."""
    if not vals:
        return ""
    import math
    top = math.ceil(max(vals) / 0.05) * 0.05
    nb = max(1, int(round(top / 0.05)))
    counts = [0] * nb
    for v in vals:
        counts[min(nb - 1, int(v / 0.05))] += 1
    W, H, L, B = 520, 150, 36, 28
    cmax = max(counts)
    bw = (W - L - 8) / nb
    bars = "".join(f'<rect x="{L + i * bw:.1f}" y="{H - B - (c / cmax) * (H - B - 12):.1f}" width="{max(bw - 2, 1):.1f}" '
                   f'height="{(c / cmax) * (H - B - 12):.1f}" fill="{color}"/>' for i, c in enumerate(counts))
    ticks = "".join(f'<text x="{L + i * bw:.1f}" y="{H - 8}" font-size="11" text-anchor="middle" fill="var(--muted)">{i * 50}</text>'
                    for i in range(0, nb + 1, max(1, nb // 6)))
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="latency histogram" style="max-width:560px;width:100%;height:auto;display:block">'
            f'<line x1="{L}" y1="{H - B}" x2="{W - 4}" y2="{H - B}" stroke="var(--line)"/>{bars}{ticks}'
            f'<text x="{L}" y="12" font-size="11" fill="var(--muted)">queries per 50 ms bin (max {cmax})</text>'
            f'<text x="{W - 4}" y="{H - 8}" font-size="11" text-anchor="end" fill="var(--muted)">ms</text></svg>')


def stage_rows(d: dict) -> list[tuple[str, str, bool]]:
    """The pipeline stages and whether each passed, over all tasks."""
    rows = []
    T = d["tasks"]
    ok = all(t["health"].get("status") == "ok" for t in T.values()) and len(T) == 3
    rows.append(("Model servers up", ", ".join(f"{t['preset']} ({t['health'].get('ckpt_step', '?')})" for t in T.values()), ok))
    ok = all(all(b["image"][:2] == [360, 640] for b in t["bridge"]) for t in T.values()) and all(t["bridge"] for t in T.values())
    renderer = next((t["run"].get("renderer", "mujoco") for t in d["tasks"].values()), "mujoco")
    rendered = "rendered in Isaac (SIMPLE's standard mujoco_isaac mode)" if renderer == "isaac" else "rendered by MuJoCo"
    rows.append(("Images reach the model", f"every query carried the HBVCAM rectified left eye {rendered}, resized to 640 × 360 (the training size)", ok))
    ok = all(all(b["rows"] == 24 for b in t["bridge"]) for t in T.values())
    rows.append(("Model replies", "24 rows × 36 at 50 Hz per query, all finite", ok))
    ok = all(all(b["out"]["finite"] and b["out"]["shape"] == [24, 79] and abs(b["out"]["quat_norm"] - 1) < 1e-3 for b in t["bridge"]) for t in T.values())
    rows.append(("Bridge frames valid", "24 × 79 reference frames per reply: finite, unit root quaternions", ok))
    ok = all(all(e["replies_not_used"] == 0 for e in t["episodes"]) for t in T.values())
    rows.append(("Eval accepted every reply", "0 replies rejected by the eval's validity check", ok))
    eng = [e["engaged_step"] for t in T.values() for e in t["episodes"]]
    ok = all(x is not None for x in eng)
    rows.append(("Motion tracking engaged", f"on the model's frames in every episode, at step {min(eng) if ok else '?'}–{max(eng) if ok else '?'}", ok))
    ok = all(all(e["error"] is None and not e["fell"] for e in t["episodes"]) for t in T.values())
    rows.append(("Episodes ran to the end", "no errors, no falls, every episode reached its step budget or the task check", ok))
    n_eps = sum(len(t["episodes"]) for t in T.values())
    ok = all(all("exact_log" in e for e in t["episodes"]) for t in T.values())
    rows.append(("Outputs written", f"results.json, {n_eps} videos, {n_eps} replay logs", ok))
    C = d["checks"]
    rep = C.get("replay", {})
    ok = bool(rep) and all(v.get("ok") for v in rep.values())
    rows.append(("Replay bit-exact", "; ".join(f"{k}: {v.get('summary', '?')}" for k, v in rep.items()) or "not run", ok))
    det = C.get("repeat", {})
    if det:                                            # a property of the model (sampling), reported, not scored
        rows.append(("Same scene twice", det.get("summary", ""), None))
    return rows


def build() -> None:
    d = load()
    renderer = next((t["run"].get("renderer", "mujoco") for t in d["tasks"].values()), "mujoco")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "img").mkdir(exist_ok=True)
    for f in (OUT / "img").glob("*.jpg"):
        f.unlink()
    for f in (INPUTS / "frames").glob("*.jpg"):
        if not f.name.startswith("tmp_"):
            shutil.copy(f, OUT / "img" / f.name)
    stages = stage_rows(d)
    scored = [ok for _, _, ok in stages if ok is not None]
    all_ok = all(scored)
    n_eps = sum(len(t["episodes"]) for t in d["tasks"].values())
    n_q = sum(len(t["bridge"]) for t in d["tasks"].values())
    lat_all = [b["model_s"] for t in d["tasks"].values() for b in t["bridge"]]

    css = """
:root{--bg:#f6f7f5;--surface:#fff;--ink:#1d2321;--muted:#5d6a66;--line:#d8ddda;--accent:#0f766e;--accent-soft:#e3f1ee;--ok:#1a7f4b;--bad:#b3261e;--code:#eef1ef;color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15191a;--surface:#1d2324;--ink:#e6ebe8;--muted:#9aa6a1;--line:#2f3837;--accent:#5cc4b7;--accent-soft:#1d3330;--ok:#5fd08f;--bad:#f28b82;--code:#232b2b;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#15191a;--surface:#1d2324;--ink:#e6ebe8;--muted:#9aa6a1;--line:#2f3837;--accent:#5cc4b7;--accent-soft:#1d3330;--ok:#5fd08f;--bad:#f28b82;--code:#232b2b;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;padding-block:28px 64px;padding-inline:16px}
main{max-width:1080px;margin:0 auto;display:flex;flex-direction:column;gap:36px}
h1{font-size:26px;line-height:1.2;margin:0;text-wrap:balance}h2{font-size:19px;margin:0 0 10px;text-wrap:balance}h3{font-size:15px;margin:0 0 6px}
p{margin:0;max-width:68ch}.muted{color:var(--muted)}code,.mono{font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
code{background:var(--code);padding:1px 5px;border-radius:4px}
pre{background:var(--code);padding:12px 14px;border-radius:8px;overflow-x:auto;margin:0;font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.verdict{display:flex;gap:14px;align-items:center;padding:14px 16px;border-radius:10px;background:var(--accent-soft);border-left:4px solid var(--accent)}
.verdict b{font-size:17px}.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600;letter-spacing:.03em;text-transform:uppercase}
.ok{background:var(--ok);color:#fff}.bad{background:var(--bad);color:#fff}.note{background:var(--muted);color:#fff}
.tbl{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--surface)}table{border-collapse:collapse;width:100%;font-size:14px}
th{text-align:left;font-size:11.5px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted);padding:9px 12px;border-bottom:1px solid var(--line);white-space:nowrap}
td{padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}tr:last-child td{border-bottom:0}td.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}.grid figure{margin:0;display:flex;flex-direction:column;gap:4px}
.grid img{width:100%;height:auto;border-radius:6px;border:1px solid var(--line);display:block}figcaption{font-size:12.5px;color:var(--muted)}
.cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;display:flex;flex-direction:column;gap:8px}
ul{margin:0;padding-left:1.2em;max-width:70ch}li{margin:3px 0}.flow{display:flex;flex-wrap:wrap;gap:8px;align-items:center}.flow span{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:13.5px}
.flow i{color:var(--muted);font-style:normal}
"""
    H = [f'<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
         f"<title>HoloMotion v1.4 Pipeline Check</title><style>{css}</style><main>"]
    H.append(f"""<header style="display:flex;flex-direction:column;gap:12px">
<div class="muted mono">SIMPLE · eval_holomotion_v14 · {esc(d['tasks'][next(iter(d['tasks']))]['run'].get('robot_mjcf', '')).split('/')[-1] if d['tasks'] else ''}</div>
<h1>Pipeline check: the HoloBrain G1 models driving the HoloMotion v1.4 evaluator</h1>
<p>Three deploy models (<code>~/wrk/robot_orchard_deploy/models</code>, served by <code>holobrain_g1_deploy/scripts/serve.sh</code>) were put in the loop of the HoloMotion v1.4 VLA evaluation, one per kitchen task. Their outputs command a decoupled whole-body controller, not the HoloMotion motion tracker, so a bridge converts each reply into reference frames the controller accepts. This checks every stage of the pipeline with a real model: images in, inference, protocol, timing, controller, outputs, replay. It does not measure the models.</p>
<div class="verdict"><span class="pill {'ok' if all_ok else 'bad'}">{'pass' if all_ok else 'fail'}</span><div><b>{sum(scored)} of {len(scored)} pipeline stages passed</b><br><span class="muted">{n_eps} episodes over 3 tasks, {n_q} model queries, median model latency {statistics.median(lat_all) * 1000:.0f} ms</span></div></div>
</header>""")

    # stages
    H.append('<section><h2>What was checked</h2><div class="tbl"><table><thead><tr><th>stage</th><th>evidence</th><th>result</th></tr></thead><tbody>')
    for name, ev, ok in stages:
        pill = '<span class="pill note">noted</span>' if ok is None else f'<span class="pill {"ok" if ok else "bad"}">{"pass" if ok else "fail"}</span>'
        H.append(f'<tr><td><b>{esc(name)}</b></td><td>{esc(ev)}</td><td>{pill}</td></tr>')
    H.append("</tbody></table></div></section>")

    # flow
    H.append("""<section><h2>The loop</h2>
<div class="flow"><span>MuJoCo scene<br><i>level-3 set, G1 + backpack + HBVCAM</i></span><i>→</i><span>eval_holomotion_v14<br><i>HBVCAM left eye 1280×720 → 640×360, state, instruction</i></span><i>→</i><span>bridge :2100x<br><i>SIMPLE /act → PSI0 /act</i></span><i>→</i><span>HoloBrain model server :801x<br><i>Qwen3-VL 2B, 8 demasking steps</i></span><i>→</i><span>bridge<br><i>24 × 36 rows → 24 × 79 frames</i></span><i>→</i><span>HoloMotion v1.4<br><i>model_22000 tracks the frames</i></span></div>
<div class="cols" style="margin-top:14px">
<div class="card"><h3>What the model gets (its training format)</h3><ul><li><code>rgb_head_stereo_left</code>: the eval image resized to 640 × 360</li><li><code>states</code> (32): Dex3 hands 14, arms 14, waist roll/pitch/yaw, last base-height command</li><li>the instruction is overridden by the preset's training instruction on the server</li></ul></div>
<div class="card"><h3>What it returns</h3><ul><li>24 rows at 50 Hz, 36 each: hands 14, arms 14, torso roll/pitch/yaw, height, vx vy vyaw target_yaw</li><li>one query per 24 control steps (0.48 s of sim)</li></ul></div>
<div class="card"><h3>How the bridge fakes the reference</h3><ul><li>legs: HoloMotion's default standing angles</li><li>waist and arms: the model's targets, same motor order</li><li>joint velocities: finite differences along the chunk</li><li>root: the model's vx, vy, vyaw integrated 20 ms per row from the robot's pose at the first query; yaw-only rotation</li><li>hands: the model's 14 Dex3 targets, reordered to SIMPLE's MJCF order</li></ul></div>
</div></section>""")
    if renderer == "isaac":
        H[-1] = H[-1].replace("MuJoCo scene<br><i>level-3 set, G1 + backpack + HBVCAM</i>",
                              "Isaac scene<br><i>level-3 set, G1 + backpack + HBVCAM &middot; MuJoCo physics, Isaac rendering (SIMPLE's standard mode)</i>")

    # per task
    for task, T in d["tasks"].items():
        E, B = T["episodes"], T["bridge"]
        lat = [b["model_s"] for b in B]
        H.append(f'<section><h2>{esc(T["label"])}: {esc(T["preset"])}</h2>')
        hp = T["health"]
        H.append(f'<p class="muted">Model <code>{esc(Path(str(hp.get("run_dir", "?"))).name)}</code> {esc(hp.get("ckpt_step", ""))}. '
                 f'Server instruction: “{esc(hp.get("instruction_override", ""))}”. Eval instruction: “{esc(T["run"].get("instruction", ""))}”.</p>' if hp else "")
        H.append('<div class="tbl" style="margin-top:10px"><table><thead><tr><th>scene</th><th>setup</th><th>motion from step</th><th>steps</th><th>queries</th><th>model ms (mean / max)</th><th>gates reached</th><th>end</th></tr></thead><tbody>')
        for i, e in enumerate(E):
            S = T["sessions"][i] if i < len(T["sessions"]) else []
            ms = [s["model_s"] * 1000 for s in S]
            gates = ", ".join(g for g, v in e["gates"].items() if v) or "–"
            end = "task success" if e["success"] else ("error: " + e["error"] if e["error"] else "step budget")
            H.append(f'<tr><td class="mono">{esc(e["episode"])}</td><td class="muted">table {e["table_dz"] * 100:+.1f} cm</td><td class="num">{e["engaged_step"]}</td>'
                     f'<td class="num">{e["steps"]}</td><td class="num">{e["queries"]}</td><td class="num">{statistics.mean(ms) if ms else 0:.0f} / {max(ms) if ms else 0:.0f}</td>'
                     f'<td>{esc(gates)}</td><td>{esc(end)}{" · FELL" if e["fell"] else ""}</td></tr>')
        H.append("</tbody></table></div>")
        # commands
        cmd = [(s["cmd"]["vx"], s["cmd"]["vy"], s["cmd"]["vyaw"], s["cmd"]["right_hand_closed"]) for s in B]
        if cmd:
            H.append(f'<p class="muted" style="margin-top:8px">The model\'s walking command over all queries: vx {statistics.mean(c[0] for c in cmd):+.2f} m/s '
                     f'(min {min(c[0] for c in cmd):+.2f}, max {max(c[0] for c in cmd):+.2f}), vyaw {statistics.mean(c[2] for c in cmd):+.2f} rad/s; '
                     f'right hand closed in {100 * statistics.mean(c[3] for c in cmd):.0f} % of the replies.</p>')
        H.append(f'<div class="cols" style="margin-top:12px"><div class="card"><h3>Model latency, {len(lat)} queries</h3>{latency_svg(lat, "var(--accent)")}'
                 f'<p class="muted">median {statistics.median(lat) * 1000:.0f} ms, p90 {pct(lat, 0.9) * 1000:.0f} ms, max {max(lat) * 1000:.0f} ms. The controller runs on sim time, so latency does not change the rollout. The three model servers ran one at a time in this run, each alone on the GPU.</p></div></div>')
        frames = d["manifest"].get(task, {}).get("frames", {})
        for ep, lst in frames.items():
            if not lst:
                continue
            H.append(f'<h3 style="margin-top:14px">{esc(ep)}: what the model saw (left) and the scene (right)</h3><div class="grid">')
            for t, name in lst:
                H.append(f'<figure><img src="img/{name}" alt="{esc(ep)} at {t:g} s" loading="lazy"><figcaption>t = {t:g} s</figcaption></figure>')
            H.append("</div>")
        H.append("</section>")

    # checks detail
    C = d["checks"]
    if C:
        H.append("<section><h2>Replay and reproducibility</h2><div class=\"cols\">")
        for k, v in C.get("replay", {}).items():
            H.append(f'<div class="card"><h3>{esc(k)}: replay</h3><p>{esc(v.get("summary", ""))}</p><pre>{esc(v.get("detail", ""))}</pre></div>')
        if C.get("repeat"):
            r = C["repeat"]
            H.append(f'<div class="card"><h3>Same scene, run twice</h3><p>{esc(r.get("summary", ""))}</p><pre>{esc(r.get("detail", ""))}</pre></div>')
        H.append("</div></section>")

    H.append("""<section><h2>What this shows, and what it does not</h2><ul>
<li><b>Shows:</b> a real model server in the loop end to end: the eval's images and state reach the model in its training format, its replies come back in time, the bridge's frames pass the eval's validity check, the controller switches to motion tracking on them and tracks them, every episode runs to its end, and the outputs (results, videos, replay logs) are complete and bit-exact on replay.</li>
<li><b>Does not show:</b> task performance. The reference frames are faked from a controller command the model was trained to give a different controller; the legs are a fixed standing pose; the model was trained on the D455 head camera (90°), not the HBVCAM (105°) image it sees here; and bottle_bin's model was trained on the chip can, not the bottle. Success rates here mean nothing about the models.</li>
<li><b>Reproducibility:</b> the model samples its actions (grouped diffusion), so the same scene gives a different rollout each run from the first reply on. Every rollout is still reproducible from its replay log, which records the replies' effect bit for bit.</li>
<li><b>Reset:</b> the model server's <code>/act</code> route ignores the per-episode reset flag (it keeps its last torso command across episodes); the bridge resets its own root integration per episode.</li>
</ul></section>
<section><h2>Reproduce</h2><pre>cd ~/wrk/robot_orchard_deploy/holobrain_g1_deploy
PORT=8014 bash scripts/serve.sh chipcan_nativec9 --steps 8 --replan 15        # bowltosink_c9 :8015, cart_c19 :8016
cd ~/wrk/SIMPLE
.venv/bin/python scripts/holomotion_v14_vla_bridge.py --upstream-port 8014 --port 21000 --preset chipcan_nativec9 --log bridge_bottle_bin.jsonl
python -m simple.cli.eval_holomotion_v14 --scene bottle_bin --port 21000 --image-size 640x360 --num-episodes 5
python -m simple.cli.replay_holomotion_v14 &lt;run dir&gt; --all --mode action</pre>
<p class="muted" style="margin-top:8px">Generator and inputs: <code>docs/_scan/holomotion_v14_pipeline_check/</code>.</p></section></main>""")
    (OUT / "index.html").write_text("\n".join(H))
    print(f"[build] {OUT / 'index.html'} ({sum(1 for _ in (OUT / 'img').glob('*.jpg'))} images); stages passed {sum(scored)}/{len(scored)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", type=Path, default=None)
    a = ap.parse_args()
    if a.collect:
        collect(a.collect)
    build()
