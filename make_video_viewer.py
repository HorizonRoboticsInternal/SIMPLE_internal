#!/usr/bin/env python3
"""Build a local HTML viewer for generated SIMPLE datasets.

Scans LeRobot-format dataset roots for videos/chunk-*/observation.rgb_*/episode_*.mp4
and writes an HTML page that plays them grouped by task -> episode -> camera.

Re-run any time to pick up newly finished tasks:
    python make_video_viewer.py && xdg-open video_viewer.html

Paths in the page are relative to the repo root, so it works over file://.
"""
from __future__ import annotations

import html
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "video_viewer.html"

# dataset roots to scan; label -> directory
SOURCES = {
    "Dex3 sweep": ROOT / "data" / "dex3_all",
    "G1 tabletop": ROOT / "data" / "datagen_g1",
    "Franka (MuJoCo)": ROOT / "data" / "datagen_example",
    "Franka (Isaac)": ROOT / "data" / "datagen_isaac",
    # point at one dataset dir; the parent holds 123 of them and scanning all is unusable
    "Teleop (Isaac-rendered)": Path("/home/Horizon/wrk/teleop_new/datasets/G1WholebodyHandoverTeleop-v0"),
    "Real robot (teleop)": ROOT / "data" / "real_recordings",
}

# For these sources, read each episode's parquet and annotate frame count and
# joint-motion std (max over joints) -- makes static junk episodes obvious.
STATS_SOURCES = {"Real robot (teleop)"}

SUMMARY = Path(
    "/tmp/claude-1000/-home-Horizon-wrk-SIMPLE/"
    "66a4ba73-f074-4719-b3c1-1268e8145abc/scratchpad/dex3_logs/summary.tsv"
)

CAM_ORDER = [
    "head_stereo_left", "head_stereo_right",
    "front_stereo_left", "front_stereo_right",
    "side_left", "wrist", "wrist_left",
]

# Only these cameras are shown. Set SHOW_CAMERAS = None to include every camera.
SHOW_CAMERAS: set[str] | None = {"head_stereo_left", "front_stereo_left", "egocentric", "ego_view"}


def cam_rank(name: str) -> tuple[int, str]:
    return (CAM_ORDER.index(name), "") if name in CAM_ORDER else (len(CAM_ORDER), name)


def keep_cam(name: str) -> bool:
    return SHOW_CAMERAS is None or name in SHOW_CAMERAS


def episode_meta(level_dir: Path) -> dict:
    info = level_dir / "meta" / "info.json"
    if not info.is_file():
        return {}
    try:
        d = json.loads(info.read_text())
    except (OSError, ValueError):
        return {}
    meta = {"fps": d.get("fps"), "frames": d.get("total_frames"), "eps": d.get("total_episodes")}
    # real-robot exporter stores the language prompt here
    tasks_file = level_dir / "meta" / "tasks.jsonl"
    if tasks_file.is_file():
        try:
            first = json.loads(tasks_file.read_text().splitlines()[0])
            meta["task_prompt"] = first.get("task")
        except (OSError, ValueError, IndexError):
            pass
    return meta


def episode_stats(level_dir: Path, chunk: str, ep: int) -> str | None:
    """Frame count + max joint-motion std from the episode parquet ('' on failure)."""
    p = level_dir / "data" / chunk / f"episode_{ep:06d}.parquet"
    if not p.is_file():
        return None
    try:
        import numpy as np
        import pandas as pd

        df = pd.read_parquet(p, columns=["observation.state"])
        st = np.stack(df["observation.state"].values)
        return f"{len(df)} frames · motion {st.std(0).max():.2f}"
    except Exception:
        return None


def scan(source_dir: Path, with_stats: bool = False) -> dict[str, dict]:
    """task name -> {meta, episodes: {ep_index: [(camera, relpath)]}, ep_stats}"""
    tasks: dict[str, dict] = {}
    if not source_dir.is_dir():
        return tasks
    for mp4 in sorted(source_dir.rglob("videos/chunk-*/*/episode_*.mp4")):
        cam = mp4.parent.name.replace("observation.rgb_", "").replace("observation.images.", "")
        if not keep_cam(cam):
            continue
        m = re.search(r"episode_(\d+)\.mp4$", mp4.name)
        if not m:
            continue
        ep = int(m.group(1))
        level_dir = mp4.parents[3]          # .../level-0, or the dataset root itself
        # SIMPLE datagen nests videos under <Task>/simple/<Task>/level-N/; a
        # post-processed dataset puts them straight under <Task>/. Name the task
        # from whichever of those two layouts we are looking at.
        task = level_dir.parent.name if level_dir.name.startswith("level-") else level_dir.name
        entry = tasks.setdefault(
            task, {"meta": episode_meta(level_dir), "episodes": defaultdict(list), "ep_stats": {}}
        )
        entry["episodes"][ep].append((cam, os.path.relpath(mp4, ROOT).replace(os.sep, "/")))
        if with_stats and ep not in entry["ep_stats"]:
            stats = episode_stats(level_dir, mp4.parents[1].name, ep)
            if stats:
                entry["ep_stats"][ep] = stats
    return tasks


def load_summary() -> dict[str, tuple[int, int, int]]:
    out: dict[str, tuple[int, int, int]] = {}
    if not SUMMARY.is_file():
        return out
    for line in SUMMARY.read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 4:
            try:
                out[parts[0]] = (int(parts[1]), int(parts[2]), int(parts[3]))
            except ValueError:
                pass
    return out


CSS = """
:root{--bg:#f7f7f8;--panel:#fff;--fg:#18181b;--muted:#6b7280;--line:#e4e4e7;
 --ok:#047857;--okbg:#d1fae5;--bad:#b91c1c;--badbg:#fee2e2;--warn:#92400e;--warnbg:#fef3c7;--accent:#4f46e5}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0b0b0e;--panel:#151519;--fg:#ececf1;
 --muted:#9ca3af;--line:#2a2a31;--ok:#6ee7b7;--okbg:#064e3b;--bad:#fca5a5;--badbg:#7f1d1d;
 --warn:#fcd34d;--warnbg:#78350f;--accent:#a5b4fc}}
:root[data-theme=dark]{--bg:#0b0b0e;--panel:#151519;--fg:#ececf1;--muted:#9ca3af;--line:#2a2a31;
 --ok:#6ee7b7;--okbg:#064e3b;--bad:#fca5a5;--badbg:#7f1d1d;--warn:#fcd34d;--warnbg:#78350f;--accent:#a5b4fc}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
header{position:sticky;top:0;z-index:10;background:var(--panel);border-bottom:1px solid var(--line);
 padding:14px 20px;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0;font-weight:650;letter-spacing:-.01em}
.stat{color:var(--muted);font-size:13px}
input[type=search]{flex:1;min-width:200px;padding:7px 11px;border:1px solid var(--line);
 border-radius:7px;background:var(--bg);color:var(--fg);font-size:14px}
button{padding:7px 12px;border:1px solid var(--line);border-radius:7px;background:var(--bg);
 color:var(--fg);cursor:pointer;font-size:13px}
button:hover{border-color:var(--accent)}
main{padding:20px;max-width:1600px;margin:0 auto}
.task{background:var(--panel);border:1px solid var(--line);border-radius:11px;margin-bottom:18px;overflow:hidden}
.task>summary{cursor:pointer;padding:13px 16px;display:flex;gap:11px;align-items:center;
 flex-wrap:wrap;font-weight:600;list-style:none}
.task>summary::-webkit-details-marker{display:none}
.task>summary::before{content:"▸";color:var(--muted);font-weight:400}
.task[open]>summary::before{content:"▾"}
.badge{font-size:11px;font-weight:600;padding:2px 8px;border-radius:99px;letter-spacing:.02em}
.ok{background:var(--okbg);color:var(--ok)}.bad{background:var(--badbg);color:var(--bad)}
.warn{background:var(--warnbg);color:var(--warn)}
.sub{font-weight:400;color:var(--muted);font-size:12.5px}
.ep{border-top:1px solid var(--line);padding:13px 16px}
.ep h3{margin:0 0 10px;font-size:13px;font-weight:600;color:var(--muted);
 display:flex;gap:10px;align-items:center}
.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}
figure{margin:0}
video{width:100%;border-radius:7px;background:#000;display:block;aspect-ratio:16/9;object-fit:contain}
figcaption{font-size:11.5px;color:var(--muted);margin-top:4px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.empty{padding:14px 16px;color:var(--muted);font-size:13.5px;border-top:1px solid var(--line)}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
"""

JS = """
const q=document.getElementById('q');
q.addEventListener('input',()=>{const v=q.value.toLowerCase();
 document.querySelectorAll('.task').forEach(t=>{
  t.style.display=t.dataset.name.toLowerCase().includes(v)?'':'none';});});
document.getElementById('expand').onclick=()=>document.querySelectorAll('.task').forEach(t=>t.open=true);
document.getElementById('collapse').onclick=()=>document.querySelectorAll('.task').forEach(t=>t.open=false);
// play/pause every video inside one episode row together
document.addEventListener('click',e=>{
 if(!e.target.matches('[data-play]'))return;
 const vids=e.target.closest('.ep').querySelectorAll('video');
 const anyPaused=[...vids].some(v=>v.paused);
 vids.forEach(v=>{v.currentTime=0; anyPaused?v.play():v.pause();});
 e.target.textContent=anyPaused?'Pause all':'Play all';});
"""


def build() -> str:
    summary = load_summary()
    sections: list[str] = []
    n_tasks = n_eps = n_vids = 0

    for label, src in SOURCES.items():
        tasks = scan(src, with_stats=label in STATS_SOURCES)
        failed = {
            name: rc for name, (rc, eps, _v) in summary.items()
            if eps == 0 and src.name == "dex3_all"
        }
        if not tasks and not failed:
            continue
        sections.append(f'<h2 style="font-size:14px;color:var(--muted);margin:26px 0 10px">'
                        f'{html.escape(label)} <span class="sub">· {html.escape(os.path.relpath(src, ROOT))}</span></h2>')

        for task in sorted(tasks):
            info = tasks[task]
            eps = info["episodes"]
            meta = info["meta"]
            n_tasks += 1
            n_eps += len(eps)
            bits = []
            if meta.get("fps"):
                bits.append(f'{meta["fps"]} fps')
            if meta.get("frames"):
                bits.append(f'{meta["frames"]} frames')
            cams = len(next(iter(eps.values()))) if eps else 0
            bits.append(f'{cams} cameras')
            if meta.get("task_prompt"):
                bits.append(f'task: “{meta["task_prompt"]}”')
            ep_stats = info.get("ep_stats", {})
            rows = []
            for ep in sorted(eps):
                vids = sorted(eps[ep], key=lambda cv: cam_rank(cv[0]))
                n_vids += len(vids)
                cells = "".join(
                    f'<figure><video controls muted loop preload="none" src="{html.escape(rel)}"></video>'
                    f'<figcaption>{html.escape(cam)}</figcaption></figure>'
                    for cam, rel in vids
                )
                stat = ep_stats.get(ep)
                stat_html = ""
                if stat:
                    static = "motion 0.00" in stat
                    stat_html = (f'<span class="badge {"bad" if static else "ok"}">{html.escape(stat)}'
                                 + (" — static" if static else "") + "</span>")
                rows.append(
                    f'<div class="ep"><h3>Episode {ep:03d}'
                    f'<button data-play>Play all</button>{stat_html}</h3>'
                    f'<div class="grid">{cells}</div></div>'
                )
            sections.append(
                f'<details class="task" data-name="{html.escape(task)}"><summary>'
                f'{html.escape(task)} <span class="badge ok">{len(eps)} ep</span>'
                f'<span class="sub">{html.escape(" · ".join(bits))}</span></summary>'
                + "".join(rows) + "</details>"
            )

        for name, rc in sorted(failed.items()):
            if name in tasks:
                continue
            n_tasks += 1
            why = ("timed out before 2 successes" if rc == 124
                   else "errored on startup" if rc == 1 else f"exit {rc}")
            sections.append(
                f'<details class="task" data-name="{html.escape(name)}"><summary>'
                f'{html.escape(name)} <span class="badge {"warn" if rc==124 else "bad"}">no video</span>'
                f'<span class="sub">{html.escape(why)}</span></summary>'
                f'<div class="empty">No episodes recorded. Log: '
                f'<code>dex3_logs/{html.escape(name)}.log</code></div></details>'
            )

    head = (f'<span class="stat">{n_tasks} tasks · {n_eps} episodes · {n_vids} videos</span>')
    return (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>SIMPLE episode viewer</title><style>{CSS}</style></head><body>'
        f'<header><h1>SIMPLE episodes</h1>{head}'
        f'<input id="q" type="search" placeholder="Filter tasks…">'
        f'<button id="expand">Expand all</button><button id="collapse">Collapse all</button>'
        f'</header><main>{"".join(sections) or "<p>No datasets found yet.</p>"}</main>'
        f'<script>{JS}</script></body></html>'
    )


if __name__ == "__main__":
    OUT.write_text(build())
    print(f"wrote {OUT}")
