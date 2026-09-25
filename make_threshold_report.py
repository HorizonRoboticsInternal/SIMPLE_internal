#!/usr/bin/env python
"""Build cmd_threshold_report.html from data/cmd_threshold/results_*.json.

Companion to make_sweep_report.py: that one maps the upper command envelope
(0.3 .. 1.8), this one maps the *onset* -- the command magnitude below which
the robot does not move and above which it does.
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/cmd_threshold"
OUT = ROOT / "cmd_threshold_report.html"

AXES = {
    "vx": ("Forward (vx)", "m/s", "speed_mps"),
    "vy": ("Lateral (vy)", "m/s", "speed_mps"),
    "vyaw": ("Turn (vyaw_flag)", "rad/s", "yaw_rate_rps"),
}

CSS = """
body{font-family:system-ui,sans-serif;margin:0;background:#0f1115;color:#e8eaed}
.wrap{padding:26px 34px 60px;max-width:1100px}
h1{margin:0 0 6px;font-size:1.5em}
h2{color:#7fd0ff;font-size:1.1em;margin:30px 0 6px}
.sub{color:#9aa3ad;font-size:0.88em;line-height:1.6;margin:4px 0 10px}
table{border-collapse:collapse;width:100%;font-size:0.85em;margin:10px 0}
th,td{border:1px solid #2a2f37;padding:5px 9px;text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:#1a1f26;color:#cfd6dd}
tr.below td{color:#8a9099}
tr.onset td{background:#16242c;color:#7fe09a;font-weight:600}
tr.above td{color:#cfd6dd}
.note{background:#171b21;border-left:4px solid #7fd0ff;padding:11px 15px;margin:12px 0;
  border-radius:5px;font-size:0.9em;line-height:1.65}
.note.ok{border-left-color:#7fe09a}
.note.warn{border-left-color:#e0c77f}
code{background:#1a1f26;padding:1px 5px;border-radius:4px;font-size:0.92em}
.bar{display:inline-block;height:9px;background:#7fd0ff;border-radius:2px;vertical-align:middle}
.barwrap{width:150px;display:inline-block;text-align:left}
"""


def load():
    rows = []
    for f in sorted(DATA.glob("results_*.json")):
        if f.name.endswith("_trace.json"):
            continue
        rows += json.loads(f.read_text())
    return rows


def main():
    rows = load()
    zeros = [r for r in rows if r.get("axis") == "zero" and "error" not in r]
    floor = max((r["net_disp_m"] for r in zeros), default=0.0)

    body = ["<div class='wrap'><h1>SIMPLE sim teleop — motion-onset threshold</h1>"]
    body.append(
        "<div class='sub'>Where locomotion <i>starts</i>. "
        f"{len(rows)} trials, 8 s hold with the first 1.5 s discarded, "
        f"drift floor {floor*1000:.1f} mm from {len(zeros)} zero-command repeats "
        "(bit-identical, so the sim is deterministic and every value below is "
        "signal, not noise). <code>Walk %</code> counts actual Walk-ONNX "
        "invocations, so the policy switch is observed rather than inferred.</div>"
    )

    body.append(
        "<div class='note ok'><b>Result: the onset is 0.10 on all three axes</b>, "
        "bracketed to (0.095, 0.100]. It is a discontinuity, not a ramp — the "
        "Balance/Walk ONNX switch at "
        "<code>‖[vx, vy, vyaw_flag]‖ &lt; 0.1</code>, not a gradual walk-policy "
        "floor. Forward speed jumps 17× across a 0.005 change in command.</div>"
    )

    for axis, (title, unit, metric) in AXES.items():
        sel = [r for r in rows if r.get("axis") == axis and "error" not in r]
        if not sel:
            continue
        sel.sort(key=lambda r: r["value"])
        peak = max((r[metric] or 0) for r in sel) or 1.0
        body.append(f"<h2>{html.escape(title)}</h2><table><thead><tr>"
                    f"<th>command</th><th>Walk %</th><th>achieved ({unit})</th>"
                    f"<th>net disp (m)</th><th>foot lift (mm)</th><th></th>"
                    "</tr></thead><tbody>")
        for r in sel:
            v = r["value"]
            cls = "onset" if abs(v - 0.10) < 1e-9 else ("below" if v < 0.10 else "above")
            val = r[metric] or 0.0
            w = int(150 * abs(val) / peak)
            body.append(
                f"<tr class='{cls}'><td>{v:g}</td>"
                f"<td>{(r['walk_frac'] or 0)*100:.0f}</td>"
                f"<td>{val:.4f}</td><td>{r['net_disp_m']:.3f}</td>"
                f"<td>{r['foot_z_range']*1000:.0f}</td>"
                f"<td><span class='barwrap'><span class='bar' style='width:{w}px'></span></span></td></tr>"
            )
        body.append("</tbody></table>")

    body.append(
        "<div class='note warn'><b>Two different mechanisms.</b> For vx/vy the "
        "sub-threshold command still reaches the Balance policy, which leans "
        "into it: the robot <i>creeps without stepping</i> (foot lift 0 mm, "
        "~20 mm of travel at cmd 0.095, ≈19 cm per held minute). For turning, "
        "<code>vyaw</code> is forced to exactly 0 below the flag threshold, so "
        "there is no creep at all — sub-threshold turn trials are bit-identical "
        "to the zero baseline.</div>"
    )

    body.append(
        "<div class='note warn'><b><code>vyaw_flag</code> is binary.</b> Turn "
        "rate is 0.399–0.400 rad/s for every flag from 0.10 to 0.30, and the "
        "0.15/0.20/0.25/0.30 trials are bit-identical on every metric. The rate "
        "comes from yaw <i>error</i> (<code>yaw_error/0.5</code>, clipped at "
        "1.0 rad/s), not from the stick — pushing the turn stick further does "
        "nothing.</div>"
    )

    body.append(
        "<div class='note'><b>What the operator feels.</b> The PICO dead zone "
        "rescales rather than clips: <code>v = sign·(|axis|−0.1)/0.9 · MAX</code>. "
        "Reaching the 0.1 command threshold therefore needs "
        "<b>28% stick deflection to walk</b> (MAX 0.5 m/s) but only "
        "<b>19% to turn</b> (MAX 1.0 rad/s). Between the nominal 0.1 dead zone "
        "and those points the stick emits a non-zero command that is logged and "
        "recorded but that the Walk policy never sees.</div>"
    )

    body.append("</div>")

    OUT.write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>SIMPLE teleop — motion-onset threshold</title>"
        f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>",
        encoding="utf-8",
    )
    print(f"wrote {OUT} ({len(rows)} trials)")


if __name__ == "__main__":
    main()
