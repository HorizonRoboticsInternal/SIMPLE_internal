#!/usr/bin/env python3
"""Build cmd_sweep_report.html from data/cmd_sweep/results.json + mp4s."""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "cmd_sweep"
OUT = ROOT / "cmd_sweep_report.html"

CSS = """
:root{--bg:#f7f7f8;--panel:#fff;--fg:#18181b;--muted:#6b7280;--line:#e4e4e7;
 --ok:#047857;--okbg:#d1fae5;--bad:#b91c1c;--badbg:#fee2e2;--accent:#4f46e5}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#0b0b0e;--panel:#151519;
 --fg:#ececf1;--muted:#9ca3af;--line:#2a2a31;--ok:#6ee7b7;--okbg:#064e3b;--bad:#fca5a5;
 --badbg:#7f1d1d;--accent:#a5b4fc}}
:root[data-theme=dark]{--bg:#0b0b0e;--panel:#151519;--fg:#ececf1;--muted:#9ca3af;--line:#2a2a31;
 --ok:#6ee7b7;--okbg:#064e3b;--bad:#fca5a5;--badbg:#7f1d1d;--accent:#a5b4fc}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.6 ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1200px;margin:0 auto;padding:28px 20px}
h1{font-size:22px;letter-spacing:-.01em;margin:0 0 4px}
h2{font-size:16px;margin:30px 0 10px}
p,li{color:var(--fg)} .muted{color:var(--muted)}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--panel);
 border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{padding:8px 11px;text-align:left;border-top:1px solid var(--line)}
th{background:none;color:var(--muted);font-weight:600;font-size:12px;
 text-transform:uppercase;letter-spacing:.04em;border-top:none}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
 background:var(--line);padding:1px 5px;border-radius:4px}
.badge{font-size:11px;font-weight:600;padding:2px 8px;border-radius:99px}
.ok{background:var(--okbg);color:var(--ok)}.bad{background:var(--badbg);color:var(--bad)}
.card{background:var(--panel);border:1px solid var(--line);border-radius:11px;
 padding:14px 16px;margin:14px 0}
.card h3{margin:0 0 6px;font-size:14.5px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
video{width:100%;max-width:960px;border-radius:8px;background:#000;display:block;margin-top:8px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.kv{font-size:13px;color:var(--muted)}
.overflow{overflow-x:auto}
"""

FINDINGS_HTML = """
<h2>3 · Findings</h2>
<div class="card"><ul>
<li><b>Stable everywhere tested</b> — 14/14 trials upright, including commands 3.6× beyond
 the teleop cap. The Walk policy saturates rather than falling.</li>
<li><b>Forward speed saturates ≈ 0.15 m/s.</b> Commanded 0.3 → 0.08, 0.5 → 0.13,
 0.8 → 0.15, 1.2/1.8 → no further gain. Commanding beyond the 0.5 teleop cap is pointless.</li>
<li><b>Lateral is the fastest axis</b>: vy = 0.8 achieved 0.21 m/s.</li>
<li><b>Turning saturates ≈ 0.35 rad/s</b> regardless of vyaw_flag 0.5 vs 1.0 (~18°/s;
 a 90° turn takes ~4.5 s).</li>
<li><b>Height envelope</b>: 0.50 m crouch tracks cleanly; 0.30 m is a sustained deep crouch
 (0.29 m held, video-verified — the harness fall detector misfired and was corrected);
 walking while crouched at 0.50 nearly stalls (0.026 m/s).</li>
<li><b>Startup matters more than any command value</b>: engaging the lower-body RL on a
 statically-held stance falls 100% of the time. The working flow is the recording-mode
 reset — spawn standing, <code>reset_policy()</code>, engage RL immediately. The PICO toggle
 in sim <i>disables</i> an already-enabled policy (the CLI pre-enables it), which is why
 pressing it drops the robot.</li>
<li><b>Benchmark context</b>: the released XMovePickTeleop episodes command only
 {0, 0.5} m/s (full-stick bang-bang) — by these measurements they walk at ≈0.13 m/s real.</li>
</ul></div>
"""

CONTROLLER_HTML = """
<h2>1 · The controller, from code</h2>
<p>The sim teleop pipeline (<code>simple.cli.teleop_decoupled_wbc</code>) runs the
<b>decoupled WBC</b> stack — not AMO (scripted MP data) and not SONIC (unused TODO):</p>
<div class="card"><ul>
<li><b>Upper body</b> — <code>InterpolationPolicy</code>: rate-limited interpolation toward
 28 joint targets (2×7 arm + 2×7 hand from teleop IK). Rate limit =
 <code>upper_body_joint_speed</code>.</li>
<li><b>Lower body</b> — <code>G1GearWbcPolicy</code>: <b>two ONNX RL policies</b>,
 <code>GR00T-WholeBodyControl-Balance</code> and <code>-Walk</code>, switched every tick on
 the command norm: <code>‖[vx, vy, vyaw_flag]‖ &lt; 0.1 → Balance</code>, else Walk.
 Output: 15 lower-body joint targets. Legs obey the RL output only while
 <code>use_policy_action</code> is on (<code>toggle_policy_action</code>); otherwise they hold
 measured position.</li>
<li><b>Actuation</b> — everything merges into 29 joint position targets executed by PD
 (<code>ActionCmd("decoupled_wbc")</code>) at 50&nbsp;Hz control over a 200&nbsp;Hz MuJoCo step.</li>
</ul></div>

<h2>2 · The commands that move the robot</h2>
<div class="card"><p>Locomotion is driven by exactly two goal fields, consumed by the
lower-body policy every tick:</p>
<ul>
<li><code>navigate_cmd = [vx, vy, vyaw_flag, target_yaw]</code> — m/s, m/s, turn flag, absolute
 heading (rad). Turning is <i>doubly controlled</i>: the robot keeps turning while
 <code>vyaw_flag</code> is set and homes onto <code>target_yaw</code>.</li>
<li><code>base_height_command</code> — pelvis height target (m).</li>
</ul>
<p>The PICO joystick maps onto these with hard caps:
<code>MAX_LINEAR_VEL = 0.5 m/s</code>, <code>MAX_ANGULAR_VEL = 1.0 rad/s</code>, height
clamped to <code>0.2–0.74 m</code>. The sweep below drives the same stack
<i>directly</i>, so it can exceed the teleop caps and find the policy's true envelope.</p></div>
"""


def fmt(v, unit=""):
    return f"{v}{unit}" if v is not None else "—"


def main():
    results = json.loads((DATA / "results.json").read_text())
    rows, cards = [], []
    for r in results:
        nav = r.get("nav_cmd", [0, 0, 0, 0])
        cmd = f"vx={nav[0]} vy={nav[1]} flag={nav[2]} Δyaw={nav[3]}"
        if r.get("height_cmd", 0.74) != 0.74:
            cmd += f" h={r['height_cmd']}"
        ok = not r.get("fell") and "error" not in r
        verdict = ("<span class='badge bad'>FELL</span>" if r.get("fell")
                   else "<span class='badge bad'>ERROR</span>" if "error" in r
                   else "<span class='badge ok'>OK</span>")
        rows.append(
            f"<tr><td><code>{html.escape(r['trial'])}</code></td>"
            f"<td class='kv'>{html.escape(cmd)}</td>"
            f"<td>{fmt(r.get('achieved_speed'), ' m/s')}</td>"
            f"<td>{fmt(r.get('achieved_yaw_rate'), ' rad/s')}</td>"
            f"<td>{fmt(r.get('pelvis_z_min'), ' m')}</td>"
            f"<td>{verdict}</td></tr>"
        )
        vid = r.get("video")
        vid_html = (f"<video controls muted loop preload='none' "
                    f"src='{html.escape(str(vid))}'></video>"
                    if vid else "<p class='muted'>no video captured</p>")
        detail = (f"achieved {fmt(r.get('achieved_speed'), ' m/s')} · "
                  f"yaw {fmt(r.get('achieved_yaw_rate'), ' rad/s')} · "
                  f"pelvis min {fmt(r.get('pelvis_z_min'), ' m')}"
                  + (f" · <b>{html.escape(r['error'])}</b>" if "error" in r else "")
                  + (f" · <i>{html.escape(r['detector_note'])}</i>" if "detector_note" in r else ""))
        cards.append(
            f"<div class='card'><h3><code>{html.escape(r['trial'])}</code> {verdict}"
            f"<span class='kv'>{html.escape(r.get('note', ''))}</span></h3>"
            f"<div class='kv'>command: {html.escape(cmd)} · {detail}</div>{vid_html}</div>"
        )

    n_ok = sum(1 for r in results if not r.get("fell") and "error" not in r)
    body = (
        f"<main><h1>Teleop command envelope — decoupled WBC</h1>"
        f"<p class='muted'>G1WholebodyXMovePickTeleop-v0 · MuJoCo · scripted commands injected "
        f"into the real teleop stack (no headset) · {len(results)} trials, {n_ok} OK · videos: "
        f"head_stereo_left | front_stereo_left</p>"
        + CONTROLLER_HTML
        + FINDINGS_HTML
        + "<h2>4 · Sweep results</h2><div class='overflow'><table><tr>"
          "<th>trial</th><th>command</th><th>achieved speed</th><th>achieved yaw rate</th>"
          "<th>pelvis z min</th><th>outcome</th></tr>"
        + "".join(rows) + "</table></div>"
        + "<h2>5 · Videos</h2>" + "".join(cards)
        + "</main>"
    )
    OUT.write_text(
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Teleop command envelope</title><style>{CSS}</style></head>"
        f"<body>{body}</body></html>"
    )
    print(f"wrote {OUT} ({len(results)} trials)")


if __name__ == "__main__":
    main()
