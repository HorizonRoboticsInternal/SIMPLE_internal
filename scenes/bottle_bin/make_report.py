#!/usr/bin/env python3
"""Scene page for the bottle / trash-bin scene, generated from layout.json, renders/ and replay/.

Writes  site/index.html      full document with RELATIVE links to site/img/ and site/vid/ (publish the folder as a unit:
                             cp -r site/ /mnt/nas26/alan.jiang/fleet_status/holobrain_bottle_bin_scene/)
        site/artifact.html   the same body as an Artifact page fragment (published with site/img + site/vid as files)
        report.html          standalone single file (images inlined, videos as links) for local viewing
Videos are expected in site/vid/ (remuxed with +faststart by the shell snippet in README.md)."""
import base64, html, io, json, re, shutil
from pathlib import Path
from PIL import Image

HERE = Path(__file__).resolve().parent
R, RP, SITE = HERE / "renders", HERE / "replay", HERE / "site"
IMGSRC = HERE / "replay" / "sweep" / "fig"          # the 09-23 fit figures (make_fit_plots.py --out)
IMG = SITE / "img"; IMG.mkdir(parents=True, exist_ok=True)
L = json.load(open(HERE / "layout.json")); lay, cam, pose = L["layout"], L["ego_camera"], L["pose"]
SESSION = "2026-09-17-02-25-56-G1-sim"; EPS = [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]


def _prep(src: Path, maxw=1280, q=82):
    """Copy an image into site/img (JPEG at <= maxw, PNG kept for the plan / sheets with text) and return (relpath, data URI)."""
    im = Image.open(src).convert("RGB")
    keep_png = src.name.startswith(("plan", "bottle_options", "fit_", "outcomes")) or "sheet" in src.name
    if im.width > maxw and not keep_png:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    name = src.stem + (".png" if keep_png else ".jpg"); out = IMG / name; b = io.BytesIO()
    if keep_png:
        im.save(out, "PNG", optimize=True); im.save(b, "PNG", optimize=True); mime = "image/png"
    else:
        im.save(out, "JPEG", quality=q, optimize=True); im.save(b, "JPEG", quality=q, optimize=True); mime = "image/jpeg"
    return f"img/{name}", f"data:{mime};base64," + base64.b64encode(b.getvalue()).decode()


class Fig:
    """fig(...) renders with relative paths (site) or data URIs (standalone) depending on MODE."""
    mode = "site"
    def __call__(self, src, cap):
        p = Path(src) if Path(src).is_absolute() else (R / src)
        if not p.exists():
            return f"<p class='missing'>missing image: {html.escape(str(src))}</p>"
        rel, uri = _prep(p)
        return f"<figure><img src='{rel if self.mode == 'site' else uri}' alt='{html.escape(cap)}' loading='lazy'><figcaption>{html.escape(cap)}</figcaption></figure>"
fig = Fig()


def video(name, cap):
    p = SITE / "vid" / name
    if not p.exists():
        return f"<p class='missing'>missing video: {html.escape(name)}</p>"
    if fig.mode == "site":
        return f"<figure><video controls preload='metadata' src='vid/{name}'></video><figcaption>{html.escape(cap)}</figcaption></figure>"
    return f"<figure><p class='sub'><b>{html.escape(name)}</b> (video, in site/vid/)</p><figcaption>{html.escape(cap)}</figcaption></figure>"


# ---------------- replay results ----------------
def run_summary(session, ep, tag):
    f = RP / f"trace_ep{ep}_{tag}.json"
    if not f.exists():
        return None
    tr = json.load(open(f)); s = next((r["summary"] for r in tr if "summary" in r), {}); g = s.get("grasp", {})
    return dict(in_bin=g.get("in_bin"), grasp=g.get("success"), release=g.get("release"), end=s.get("target_end"), bin=s.get("bin"), lift=g.get("max_lift_m"))
closure = json.load(open(RP / "bottle_from_closure_02-25-56.json")) if (RP / "bottle_from_closure_02-25-56.json").exists() else {}
fit = json.load(open(RP / "bin_fit_union_3cm.json")) if (RP / "bin_fit_union_3cm.json").exists() else {}
rows = []
for ep in EPS:
    s7, s3, v = run_summary(SESSION, ep, "scan"), run_summary(SESSION, ep, "scan3cm"), run_summary(SESSION, ep, "verify")
    c = closure.get(str(ep), {})
    def yn(r): return "—" if r is None else ("yes" if r["in_bin"] else ("no grasp" if not r["release"] or r["release"]["t"] < 8 else "missed"))
    def rel(r): return "—" if not r or not r["release"] or r["release"]["t"] < 8 else f"({r['release']['xy'][0]:+.2f}, {r['release']['xy'][1]:+.2f})"
    rows.append(f"<tr><td>{ep}</td><td>{c.get('bottle_xy_est', ['?'])[0] if c else '?'}</td><td>{yn(s7)}</td><td>{yn(s3)}</td><td>{rel(s3)}</td><td>{yn(v)}</td><td>{rel(v)}</td></tr>")
n_in = {t: sum(1 for ep in EPS if (run_summary(SESSION, ep, t) or {}).get("in_bin")) for t in ("scan", "scan3cm", "verify")}
# 2026-09-23: the layout fitted to all 97 kept episodes (replay/sweep/, sweep_replay.py -> make_summary.py)
SUM = json.load(open(RP / "sweep" / "summary_20260923.json"))
B, FN = SUM["base_3cm"], SUM["final2"]              # B = the 09-17 layout, FN = the layout on this page
N = FN["n"]
OLD_CAN, OLD_BIN = B["bottle_xy"], B["bin_xy"]
NEAR = SUM["final"]                                  # the 09-23 morning layout: bin at (-0.26, -0.80), withdrawn the same day
W, FW = SUM["walk"], SUM["free_walk"]
GT = SUM.get("gates")
W_OLD, W_NEAR, W_FAR = W["contact_old"], W["contact_current"], W["final2"]
far_cells = {k: v for k, v in SUM["far_bin_scan"].items() if k.startswith("-90")}
far_best = max(v["in_bin_and_free"] for v in far_cells.values())
vid = {int(k): v for k, v in SUM["video_runs"].items()}


def vcap(ep):
    v = vid.get(ep)
    if not v:
        return f"episode {ep}"
    o = v["outcome"]
    walk = "walk untouched" if v.get("shortfall_m", 0) <= 0.06 else f"walk cut {v['shortfall_m'] * 100:.0f} cm short by the bin"
    touch = "" if not v.get("touch_s") else f", foot touches the bin for {v['touch_s']:.1f} s"
    if o == "in_bin":
        return f"episode {ep}: carried and dropped in the bin; {walk}{touch}"
    if o == "on_floor":
        b = v.get("bin") or [lay["bin_x"], lay["bin_y"]]
        dx, dy = v["end"][0] - b[0], v["end"][1] - b[1]
        where = f"{abs(dy) * 100:.0f} cm {'short of' if dy > 0 else 'beyond'} the bin" if abs(dy) > abs(dx) else f"{abs(dx) * 100:.0f} cm {'toward the table from' if dx > 0 else 'behind'} the bin"
        return f"episode {ep}: carried, dropped on the floor {where}; {walk}{touch}"
    if v.get("tilt", 0) > 45:
        return f"episode {ep}: the closing hand knocks the can over; it stays on the table, on its side"
    return f"episode {ep}: the recorded hand closes beside the can; it never leaves the table"


def eps_str(e):
    return ", ".join(str(x) for x in e)

spec = [
    ("Robot", "g1comp: G1 29 DOF, Dex3 hands, pan/tilt D455 head, no wrist cams (holomotion g1_comp_45dof.xml), facing the table"),
    ("Table", "2.07 m across (facing the robot) × 0.60 m deep, top 0.72 m above the floor, 3 cm slab, four 4 cm legs"),
    ("Cover board", "along the far edge, standing on the table top, top edge 0.92 m above the floor (a 20 cm lip, 1.8 cm thick, full width)"),
    ("Bottle", f"500 ml plastic bottle standing upright: 6.5 cm diameter × 21.2 cm, {L['bottle']['mass']:.2f} kg (full; the recorded bottle has water in it, and at SIMPLE's 0.1 kg default the replayed fingers tip it over), cap on"),
    ("Bottle position", f"axis 56.4 cm from the left table edge and {(lay['bottle_x'] - lay['near_x'])*100:.1f} cm from the near edge, i.e. {abs(lay['bottle_y'])*100:.1f} cm right of the robot's centre line and {lay['bottle_x']*100:.1f} cm ahead of the pelvis (fitted 2026-09-23 to all {N} kept episodes; was 9 cm / {OLD_CAN[0]*100:.1f} cm)"),
    ("Robot placement", f"front of the feet {(lay['near_x'] - lay['toe_offset'])*100:.0f} cm from the near edge (pelvis {lay['pelvis_to_edge']*100:.1f} cm; the spec said 17 cm, moved closer after the replays), pelvis centre 56 cm from the left table edge; the table runs {abs(lay['right_y'])*100:.0f} cm to its right"),
    ("Trash bin", f"36 × 24 × 39 cm, black, 4 cm corner radius, open top, turned 90° clockwise (36 cm side toward the table); on the floor {abs(lay['bin_y'])*100:.0f} cm to the robot's right and {abs(lay['bin_x'])*100:.0f} cm behind the pelvis start = {(lay['near_x'] - lay['bin_x'])*100:.0f} cm out from the near table edge, {(lay['bin_y'] - lay['right_y'])*100:.0f} cm from the table's right end (2026-09-23: where a free-walking replay of the {N} kept episodes lets go, {len(far_cells)}-position scan; was ({OLD_BIN[0]:+.2f}, {OLD_BIN[1]:+.2f}), in the robot's path)"),
    ("Body pose", "every upper-body joint 0 except the elbows at −0.66, hands open; waist 0; pelvis at 0.74 m; legs solved for flat feet (the elbows_raised init pose of the recordings)"),
    ("Head servos", f"ID1 pan 2517 ticks (centre) · ID0 tilt {pose['head_tilt_deg']:.0f}° down = home 2086 + 114 ticks → goal 2200"),
    ("Camera axis", f"{cam['pitch_down_deg']:.1f}° below horizontal (47.7° bracket at servo zero + 10° servo), lens {cam['pos'][2]:.2f} m above the floor"),
    ("Camera FOV", f"{cam['hfov_deg']:.0f}° horizontal × {cam['fovy_deg']:.1f}° vertical (16:9), rendered here at {cam['size'][0]} × {cam['size'][1]}; SIMPLE renders 640 × 360"),
    ("Colours", "Dex3 hands black, floor plain light grey"),
]
spec_rows = "".join(f"<div class='row'><dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd></div>" for k, v in spec)
stats = [("0.72 m", "table top above floor"), ("0.74 m", "pelvis above floor"), (f"{lay['pelvis_to_edge']:.3f} m", "pelvis to near edge"),
         (f"{lay['near_x'] - lay['toe_offset']:.2f} m", "feet to near edge"), (f"{cam['pos'][2]:.2f} m", "camera above floor"),
         (f"{cam['pitch_down_deg']:.1f}°", "axis pitch down"), (f"({lay['bin_x']:+.2f}, {lay['bin_y']:+.2f}) m", "bin centre from the start pose"), (f"{FN['in_bin']}/{N}", "kept episodes whose can ends in the bin, walk free in " + f"{N - W_FAR['blocked']}")]
stats_html = "".join(f"<div class='stat'><span class='v'>{html.escape(v)}</span><span class='k'>{html.escape(k)}</span></div>" for v, k in stats)

CSS = """
:root{--bg:#F3F4F1;--ink:#1B1F24;--mute:#5C6570;--rule:#D5D8D2;--card:#FFFFFF;--acc:#B7371F;--axis:#2E7D32;--chip:#E8EAE4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15181C;--ink:#E9ECE6;--mute:#99A2AB;--rule:#2C323A;--card:#1C2127;--acc:#E4664F;--axis:#6FBF73;--chip:#232930}}
:root[data-theme="dark"]{--bg:#15181C;--ink:#E9ECE6;--mute:#99A2AB;--rule:#2C323A;--card:#1C2127;--acc:#E4664F;--axis:#6FBF73;--chip:#232930}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.5 "Source Sans 3",system-ui,sans-serif;-webkit-text-size-adjust:100%}
main{max-width:820px;margin:0 auto;padding-block:1.25rem 4rem;padding-inline:1rem}
h1,h2,h3{font-family:"Barlow Condensed","Arial Narrow",sans-serif;text-wrap:balance;margin:0;line-height:1.05}
h1{font-size:2.6rem;font-weight:600;letter-spacing:-.01em} h2{font-size:1.65rem;font-weight:600;margin:2.6rem 0 .9rem;padding-top:.9rem;border-top:2px solid var(--rule)} h3{font-size:1.25rem;font-weight:600;margin:1.6rem 0 .6rem}
.eyebrow{font-family:"Barlow Condensed",sans-serif;font-weight:500;text-transform:uppercase;letter-spacing:.12em;font-size:.85rem;color:var(--acc)}
.lede{color:var(--mute);margin:.6rem 0 0;max-width:38em}
nav{display:flex;flex-wrap:wrap;gap:.5rem;margin:1.2rem 0 0}
nav a{font-family:"Barlow Condensed",sans-serif;font-size:1rem;letter-spacing:.04em;text-transform:uppercase;text-decoration:none;color:var(--ink);background:var(--chip);padding:.3rem .7rem;border-radius:999px}
nav a:focus-visible,a:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
dl{margin:0;display:grid;gap:0} .row{display:grid;grid-template-columns:9.5rem 1fr;gap:.6rem;padding:.55rem 0;border-bottom:1px solid var(--rule)}
dt{font-family:"Barlow Condensed",sans-serif;font-weight:600;letter-spacing:.02em;color:var(--mute)} dd{margin:0;font-variant-numeric:tabular-nums}
@media (max-width:480px){.row{grid-template-columns:1fr;gap:.1rem}}
figure{margin:0;background:var(--card);border:1px solid var(--rule);border-radius:6px;overflow:hidden}
figure img,figure video{display:block;width:100%;height:auto;max-width:100%}
figcaption{font-size:.9rem;color:var(--mute);padding:.5rem .7rem;border-top:1px solid var(--rule)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.6rem;margin:1rem 0}
.stat{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:.6rem .7rem;display:flex;flex-direction:column}
.stat .v{font-family:"Barlow Condensed",sans-serif;font-size:1.5rem;font-weight:600;font-variant-numeric:tabular-nums;line-height:1.1} .stat .k{font-size:.82rem;color:var(--mute)}
p{max-width:44em} li{max-width:44em;margin:.25rem 0}
code{font:.9em ui-monospace,Menlo,Consolas,monospace;background:var(--chip);padding:.05em .35em;border-radius:3px}
pre{background:var(--chip);padding:.7rem .9rem;border-radius:6px;overflow-x:auto;font:.85rem/1.45 ui-monospace,Menlo,Consolas,monospace}
.tw{overflow-x:auto} table{border-collapse:collapse;font-size:.95rem;font-variant-numeric:tabular-nums} th,td{padding:.3rem .6rem;border-bottom:1px solid var(--rule);text-align:left} th{font-family:"Barlow Condensed",sans-serif;color:var(--mute);font-weight:600}
.sub{color:var(--mute);font-size:.92rem} .missing{color:var(--acc)}
"""


def body():
    return f"""<title>G1 Bottle Bin Scene</title>
<style>{CSS}</style>
<main>
<p class="eyebrow">HoloBrain G1 deploy · sim setup · 2026-09-16 / 17 · layout refitted 2026-09-23</p>
<h1>G1 bottle &amp; bin scene</h1>
<p class="lede">The bottle-to-trash-bin setup rebuilt in MuJoCo the same way as the tabletop box scene: a 2.07 × 0.60 × 0.72 m table with a
cover board along its far edge, the g1comp at the table's left end with a 500 ml bottle straight ahead, and a black trash bin on the floor
behind its right. The can and bin positions are fitted to all {N} kept episodes of the real session
(<code>ChipCanToTrash_0225_keep97</code>): every episode is replayed through the same whole-body controller inside the scene, and
{FN['in_bin']} of {N} end with the can in the bin and the robot's walk untouched in {N - W_FAR['blocked']}. (Two earlier bins that scored {B['in_bin']} and {NEAR['in_bin']} did so by
stopping the robot; see below.) MuJoCo only; the Isaac rendering is the next step.</p>
<nav><a href="#fit">Fit to 97 episodes</a><a href="#settings">Settings</a><a href="#plan">Plan</a><a href="#views">Views</a><a href="#robot">Robot view</a><a href="#replay">Replay</a><a href="#videos">Videos</a><a href="#assumptions">Assumptions</a><a href="#bottle">Bottle options</a><a href="#files">Files</a><a href="#nas28">Get it from nas28</a></nav>


<h2 id="fit">Can and bin fitted to all {N} episodes</h2>
<p>Every one of the {N} kept episodes of session {SESSION} (<code>psi0/ChipCanToTrash_0225_keep97</code>) is replayed in this scene for
each candidate can and bin position, {SUM['n_replays']:,} replays in all: the recorded upper-body targets, walking command and base height drive the
decoupled whole-body controller, the lower body walks by itself. Two things are checked per episode: does the can end in the bin, and did the bin get in
the robot's way, measured as how far short of its own free walk (the same episode replayed with no bin) the robot stops.</p>
<div class="stats">
<div class="stat"><span class="v">{FN['in_bin']} / {N}</span><span class="k">cans end in the bin</span></div>
<div class="stat"><span class="v">{W_FAR['in_bin_and_free']} / {N}</span><span class="k">…with the walk untouched</span></div>
<div class="stat"><span class="v">{W_FAR['blocked']} / {N}</span><span class="k">walks the bin cuts short (&gt; 6 cm)</span></div>
<div class="stat"><span class="v">{FN['on_table']} / {N}</span><span class="k">cans never leave the table</span></div>
</div>
<h3>The earlier bins stopped the robot</h3>
<p>The 09-17 layout had the bin at ({OLD_BIN[0]:+.2f}, {OLD_BIN[1]:+.2f}) and a first 09-23 fit moved it to ({NEAR['bin_xy'][0]:+.2f}, {NEAR['bin_xy'][1]:+.2f}), scoring
{B['in_bin']} and {NEAR['in_bin']} cans in the bin. Both sit in the robot's path: in {W_NEAR['blocked']} of {N} episodes the robot walks into the bin
about {W_NEAR['first_touch_median_t']:.0f} s in, pushes against it for a second and stops {W_NEAR['median_shortfall_m'] * 100:.0f} cm short of where the same episode walks
with nothing in the way. The can then falls into the bin because the bin is where the robot got stuck, not where the recording put it. Those two counts are
withdrawn; every number below is with the walk checked.</p>
{fig(str(IMGSRC / "outcomes.png"), f"every kept episode: the first 09-23 fit, which stops the robot in all {N} episodes (top), and the layout on this page (bottom)")}
<h3>Where a free walk ends</h3>
<p>With the bin removed, the robot walks {FW['walk_m_p5_50_95'][0]:.2f} to {FW['walk_m_p5_50_95'][2]:.2f} m (median {FW['walk_m_p5_50_95'][1]:.2f}) to its right, turns
{abs(FW['heading_deg_p5_50_95'][2]):.0f}° to {abs(FW['heading_deg_p5_50_95'][0]):.0f}° and lets go of the can at ({FW['release_x_p5_50_95'][1]:+.2f}, {FW['release_y_p5_50_95'][1]:+.2f}) m, 5th to 95th
percentile x {FW['release_x_p5_50_95'][0]:+.2f} to {FW['release_x_p5_50_95'][2]:+.2f}, y {FW['release_y_p5_50_95'][0]:+.2f} to {FW['release_y_p5_50_95'][2]:+.2f}. That is the recording, not the
simulator: the sim tracks each episode's recorded heading to ±{FW['sim_minus_real_heading_deg_std']:.0f}°, and the turn and walk length differ from episode to episode because the
operator steered by eye to wherever the bin was relative to that episode's start pose (the real base position is not recorded, so how much of the spread is the
start pose and how much the real robot's own odometry cannot be told apart). The spread, 28 × 37 cm at 5–95 %, is wider than the bin's 34 × 22 cm opening, so
one fixed bin catches part of it.</p>
{fig(str(IMGSRC / "release_cloud.png"), "the free walk: feet paths of all 97 episodes and where the can is let go, with the previous and the chosen bin")}
<h3>Choosing the bin</h3>
<p>{len(far_cells)} bin positions around that cloud, each with all {N} episodes replayed against it, scored by cans in the bin whose walk the bin did not
shorten. The best cells reach {far_best}; the chosen ({lay['bin_x']:+.2f}, {lay['bin_y']:+.2f}) is one of them with the fewest interrupted walks
({W_FAR['blocked']} cut short by more than 6 cm, {W_FAR['touch']} touched at all, {W_FAR['touch_over_half_s']} for longer than half a second, all at the end of the walk).
Turned the other way (24 cm side toward the table) every cell scores lower. The bin now stands {abs(lay['bin_y']) * 100:.0f} cm to the robot's right and
{abs(lay['bin_x']) * 100:.0f} cm behind the pelvis start, i.e. {abs(lay['bin_y'] - lay['right_y']) * 100:.0f} cm {'past' if lay['bin_y'] < lay['right_y'] else 'short of'} the table's right end, {(lay['near_x'] - lay['bin_x']) * 100:.0f} cm out from
the near-edge line — where the very first build (09-16) had put it before the replays moved it in.</p>
<div class="grid">
{fig(str(IMGSRC / "fit_bin_far.png"), "bin position: cans in the bin with the walk untouched (of 97)")}
{fig(str(IMGSRC / "fit_grid.png"), "can position: replays (of 97) that pick the can up and carry it off; grey = spawns inside a thumb")}
</div>
<h3>The can</h3>
<ul>
<li><b>{(lay['bottle_x'] - OLD_CAN[0])*100:.1f} cm further from the robot than on 09-17</b>: {lay['bottle_x']*100:.1f} cm ahead of the pelvis, {(lay['bottle_x'] - lay['near_x'])*100:.1f} cm from the near
table edge (was 9). Over {len(SUM['can_grid_carried'])} can positions the replays carry the can off the table in 81 to 84 episodes anywhere from 25.5 to 28.5 cm ahead
(vs {B['carried']} at the old spot); {lay['bottle_x']*100:.1f} cm is the middle of that plateau. The grasp does not depend on the bin, so this part of the morning's fit stands.</li>
<li><b>On the centre line.</b> In the start pose the open thumbs hang at 26.4 cm ahead, 6.1 cm either side, 0.89 m up. A can more than ~5 mm off the line at
23–30 cm spawns inside a thumb and is flung off the table before the recording starts ({SUM['spawn_clear']} of {SUM['spawn_total']} spots on a 5 mm grid are clear, hatched above).</li>
<li><b>Table distance and can mass do not matter</b> for the grasp: feet 3, 5, 7 cm from the edge carry {SUM['final']['carried']}, {SUM['edge05']['carried']}, {SUM['edge07']['carried']};
0.35 kg carries {SUM['mass35']['carried']} vs {SUM['final']['carried']} at 0.5 kg. Placing the can per episode where the recording says it stood carries only {SUM['perep']['carried']}: most of those
estimates lie inside the thumb zone.</li>
</ul>
<h3>What still fails, and what would fix it</h3>
<p>{FN['on_table']} cans never leave the table ({eps_str(FN['never_picked_eps'])}; in all but one the closing hand knocks the can over — the real can stood elsewhere that episode).
{FN['on_floor']} are carried and dropped on the floor: the robot ends its walk too far from where the bin is. A fixed bin cannot follow the operator's per-episode
steering; a per-episode bin, placed at each episode's free-walk release point, would catch nearly all {FW['n_carried']} carried cans and is the option for data generation.</p>
<h3>The task's own checkers</h3>
<p><code>bottle_bin_task.py</code> now scores an episode with three gates, in the order the task happens, each latched once met, and
reports them in <code>info["task_progress"]</code> with the time each was met: <b>grasped</b>, the right hand touches the bottle and holds it at least 3 cm
off the table, within 60° of upright, for half a second (a can that is knocked over and scooped up for a moment does not count); <b>at_bin</b>, after
grasping, the pelvis has walked at least 0.5 m, is within 0.8 m of the bin centre and has stood still (under 0.15 m/s) for 0.4 s; <b>placed</b>, after
grasping, the bottle axis is inside the bin's inner opening below the rim and the hand has let go. The reward is 0.3 for grasped, 0.6 with at_bin,
1.0 for placed, and as in SIMPLE's other tasks <code>check_success</code> is reward ≥ 0.9: <b>the task succeeds when the bottle is placed in the bin.</b>
A fourth flag, <code>robot_touched_bin</code>, is a diagnostic, not a gate. The bin's pose is read from the simulator each step, so a bin moved at run time
is scored where it is.</p>
{'<div class="stats"><div class="stat"><span class="v">' + str(GT['grasped']) + ' / ' + str(GT['n']) + '</span><span class="k">grasped</span></div><div class="stat"><span class="v">' + str(GT['at_bin']) + ' / ' + str(GT['n']) + '</span><span class="k">stopped at the bin</span></div><div class="stat"><span class="v">' + str(GT['placed']) + ' / ' + str(GT['n']) + '</span><span class="k">placed = task success</span></div><div class="stat"><span class="v">' + str(GT['placed_matches_in_bin']) + ' / ' + str(GT['n']) + '</span><span class="k">agree with the replay tool’s own in-bin check</span></div></div>' if GT else ''}
{('<p class="sub">Grasped fires a median ' + str(GT['t_grasped_after_closure_median']) + ' s after the recorded hand closure (lift + the half-second hold). Grasped but never stopped at the bin: ' + (eps_str(GT['eps_grasped_not_at_bin']) or 'none') + '. Stopped at the bin but the can missed: ' + (eps_str(GT['eps_at_bin_not_placed']) or 'none') + '.</p>') if GT else ''}
<h3>How the replays were run</h3>
<p><code>sweep_replay.py</code> is <code>replay_in_scene.py</code> without the video, the wall-clock pacing and the Pico TCP server, so ten run at once at about
8× real time. The whole control stack (agent, WBC, interpolation) is put on one virtual clock that advances exactly 20 ms per tick: it reproduces the paced
replays (episode 0: same 24 cm lift, release within 1 mm) and is deterministic. Contact is still chaotic, a millimetre in where the can starts flips a marginal
grasp, so counts from different sweeps of the same layout differ by 2 or 3; the numbers above are the scene rebuilt from <code>layout.json</code>. Robot–bin contact is
counted from MuJoCo's contact list each tick.</p>

<h2 id="settings">Settings</h2>
<dl>{spec_rows}</dl>
<div class="stats">{stats_html}</div>

<h2 id="plan">Plan view with the dimensions</h2>
<p>Orthographic top view ({lay['plan_extent'] / lay['plan_h'] * 100:.2f} cm per pixel). The robot faces up the page; its right is the right of the image. Red = the distances, drawn from the computed positions.</p>
{fig("plan.png", "plan view: table, cover board, bottle, robot and bin with the distances")}

<h2 id="views">Views</h2>
<div class="grid">
{fig("overview.png", "overview from behind the robot's left")}
{fig("front_left.png", "from the far left, over the cover board")}
{fig("front_right.png", "from the far right, down the length of the table")}
{fig("side.png", "side view from the robot's right, green = the D455 optical axis")}
{fig("bottle_closeup.png", f"the bottle {(lay['bottle_x'] - lay['near_x'])*100:.1f} cm from the near edge, between the robot's open hands")}
{fig("bin_closeup.png", "the bin: rounded corners, open top, 36 cm side toward the table")}
</div>

<h2 id="robot">What the D455 sees</h2>
{fig("ego_d455.png", f"robot's head camera at the start ({cam['size'][0]} × {cam['size'][1]}, 90° × {cam['fovy_deg']:.1f}°, servo 10°)")}

<h2 id="replay">Earlier tuning (2026-09-16 / 17)</h2>
<p>Recorded episodes are replayed through the decoupled-WBC stack inside this scene (<code>replay_in_scene.py</code>, SIMPLE task
<code>simple/G1WholebodyBottleBinTeleop-v0</code>): the recorded upper-body targets, navigation command and base height drive the
controller tick by tick (30 fps rows resampled onto its 50 Hz ticks); the lower body walks by itself. The bottle must be grasped, carried
and released into the bin.</p>
<h3>What the sweep changed</h3>
<ul>
<li><b>Bottle mass 0.1 → 0.5 kg.</b> With SIMPLE's 0.1 kg default the fingers tip the bottle at the same instant at 17, 12 and 7 cm start distance (a 0.3 N finger touch tips it). The recorded bottle has water in it; at 0.5 kg the same motion closes on it.</li>
<li><b>Start distance 17 → 3 cm.</b> Session 02-25-56, ten episodes (0, 11, 22, … 99): {n_in['scan']}/10 land in the bin with the feet 7 cm from the edge, {n_in['scan3cm']}/10 at 3 cm. The real bottle position varied per episode (the wrist at hand closure spans 0.20 to 0.33 m ahead of the pelvis), so episodes 66, 77 and 99, whose bottle stood farthest away, never grasp at 3 cm.</li>
<li><b>Bin centre fitted to the releases.</b> Fourteen release points from the two 3 cm passes span x −0.35 to −0.13 m and y −0.90 to −0.78 m relative to the start pose; the bin's inner opening is 34 × 22 cm, so all fit at centre ({fit.get('best', [0, 0])[0]:+.2f}, {fit.get('best', [0, 0])[1]:+.2f}) with {fit.get('margin', 0)*100:.1f} cm to spare. The same episode releases up to 8 cm apart between two runs (the replay is wall-clock paced), which is what eats the margin.</li>
</ul>
<h3>Session 2026-09-17-02-25-56, ten episodes, earlier layouts</h3>
<div class="tw"><table>
<tr><th>episode</th><th>real bottle x, est. (m)</th><th>7 cm</th><th>3 cm, pass 1</th><th>release (x, y)</th><th>3 cm, verification</th><th>release (x, y)</th></tr>
{''.join(rows)}
</table></div>
<p class="sub">"real bottle x" = the wrist position at hand closure + the hand offset measured on the successful replays; the scene bottle was then at {OLD_CAN[0]:.3f} m.
The verification pass ran with the bin at (−0.28, −0.86); the 09-17 centre ({OLD_BIN[0]:+.2f}, {OLD_BIN[1]:+.2f}) came from these release points
(<code>replay/bin_fit.py</code>, <code>replay/bin_fit_union_3cm.json</code>). Both were replaced on 09-23 by the fit above.</p>

<h2 id="videos">Simulation videos</h2>
<p>Left: MuJoCo third-person view of the scene. Middle: MuJoCo D455 (what a policy would see). Right: the real recording's D455 at the same row.
This layout: feet 3 cm from the edge, can {lay['bottle_x']*100:.1f} cm ahead of the pelvis, 0.5 kg, bin at ({lay['bin_x']:+.2f}, {lay['bin_y']:+.2f}). The robot now walks
its full recorded path; the same ten episodes as the 09-17 page, so they can be compared. Cans that miss land short of the bin: the robot ends its walk
where this episode's operator steering took it, not where the bin is.</p>
<div class="grid">
{''.join(video(f"s0225_ep{ep:03d}.mp4", vcap(ep)) for ep in EPS)}
</div>
<h3>Session 2026-09-17-00-16-29, episodes 6 and 12 (50 fps)</h3>
<p class="sub">The two episodes the layout was first tuned on (feet 7 cm, bin at (−0.27, −0.87)); both end with the bottle in the bin.</p>
<div class="grid">
{video("s0016_ep006.mp4", "episode 6: grasped, carried, released into the bin")}
{video("s0016_ep012.mp4", "episode 12: grasped, carried, released into the bin")}
</div>
<div class="grid">
{fig(str(RP / "grasp_phase_sheet.png"), "why the 0.1 kg bottle failed: sim D455 at 17 cm (top) and 12 cm (middle) vs the real D455 (bottom), the bottle tips at 2.2 s in both")}
{fig(str(RP / "ep12_feet07cm_m500g_sheet.png"), "episode 12 (00-16-29) with the 0.5 kg bottle: grasped, lifted and carried like the recording")}
</div>

<h2 id="assumptions">Assumptions, and the switch for each</h2>
<ul>
<li><b>Cover board</b>: a vertical board on the far edge whose top is 0.92 m above the floor. <code>--cover shelf --cover-depth 0.30</code> makes it a horizontal board instead; <code>--cover-top</code> moves the top edge.</li>
<li><b>Left / right</b>: the robot's left and right as it faces the table. Both the bottle's 56.4 cm and the robot's 56 cm are measured from the left edge (confirmed), so the bottle is 0.4 cm right of the robot's centre line.</li>
<li><b>Feet to edge</b> is measured from the front of the feet (<code>--robot-to-edge</code>, spec 0.17, now 0.03); <code>--robot-ref pelvis</code> measures from the pelvis centre.</li>
<li><b>Bottle</b>: axis position, {(lay['bottle_x'] - lay['near_x'])*100:.1f} cm from the near edge (<code>BOTTLE_FROM_FRONT</code> in <code>build_scene.py</code>; <code>--bottle-ref near</code> measures to its near surface instead); 6.5 × 21.2 cm; <code>--bottle-mass</code> (0.5). <code>--bottle-xy X Y</code> on the replay tools overrides it in the pelvis frame; per-episode estimates of the real position are in <code>replay/episode_geometry_97.json</code>.</li>
<li><b>Can vs the thumbs</b>: the start pose (elbows −0.66, hands open) puts the thumbs at (0.264, ±0.061, 0.888) m. Any can position must clear them; <code>replay/sweep/spawn_map.json</code> lists which spots do. If the start pose changes, the fit has to be redone.</li>
<li><b>Bin</b>: <code>--bin-xy X Y</code> in the pelvis frame (x toward the table, robot's right = −y); <code>--bin-xy 0 -1.40</code> gives the original table-end position; <code>--bin-yaw</code> turns it (−90 = 36 cm side toward the table).</li>
<li><b>Pose</b>: elbows −0.66, everything else 0 (<code>--pose 3</code>, default); <code>--pose 2</code> = arms hanging, <code>--pose 1</code> = the Ψ₀ client pose.</li>
</ul>

<h2 id="bottle">Bottle options</h2>
<p>The scene uses a generated 500 ml PET bottle (6.5 cm × 21 cm, translucent, no label). SIMPLE's GraspNet-1B library has no plain water bottle;
these bottle-shaped objects come closest (textures, stable poses, grasps and Isaac USDs included). Pick one by id
(<code>BOTTLE_BIN_TARGET=graspnet1b:66</code>) and it replaces the generated bottle.</p>
{fig("bottle_options.png", "standing renders with the scanned textures; sizes are the bounding box in the standing pose, volume from the mesh")}

<h2 id="files">Files and commands</h2>
<p>Everything is in <code>holobrain_g1_deploy/sim/bottle_bin/</code> (in the nas28 kit below): <code>build_scene.py</code> (MuJoCo scene, renders, plan),
<code>scene.xml</code>, <code>layout.json</code> (the numbers every other file reads), <code>bottle_bin_task.py</code> (SIMPLE task), <code>replay_in_scene.py</code>
(action replay), <code>sweep_replay.py</code> (the batch replay behind the fit), <code>fit_layout.py</code> and <code>make_fit_plots.py</code> (counts and figures),
<code>replay/</code> (videos, traces, fits; <code>replay/sweep/</code> = every 09-23 replay as one JSON per episode), <code>assets/</code> (bottle and bin meshes), <code>make_report.py</code> (this page).</p>
<pre>MUJOCO_GL=glfw DISPLAY=:1 ~/wrk/SIMPLE/.venv/bin/python build_scene.py                       # scene.xml, layout.json, renders/
MUJOCO_GL=egl ~/wrk/SIMPLE/.venv/bin/python replay_in_scene.py --probe                          # the scene inside SIMPLE
MUJOCO_GL=egl ~/wrk/SIMPLE/.venv/bin/python replay_in_scene.py --session {SESSION} \\
    --episodes 0 11 22 33 44 55 66 77 88 99 --robot-to-edge 0.03 --bottle-mass 0.5 --bin-xy {lay['bin_x']} {lay['bin_y']} --tag verify
# the 09-23 fit: all kept episodes at one layout, 10 workers (use --jobs with bottle_xy / bin_xy per job for a grid)
./replay/sweep/run_shards.sh final 10 --jobs replay/sweep/jobs_keep97.json --robot-to-edge 0.03 --bottle-mass 0.5 \
    --bottle-xy {lay['bottle_x']:.3f} {lay['bottle_y']:.3f} --bin-xy {lay['bin_x']} {lay['bin_y']} --fast-obs
python fit_layout.py replay/sweep/final                                                         # in bin / carried / never picked
python replay/sweep/make_summary.py &amp;&amp; python make_fit_plots.py --grid replay/sweep/grid2 replay/sweep/grid3cm \
    --spawn-map replay/sweep/spawn_map.json --binscan replay/sweep/binscan --before replay/sweep/base_3cm --after replay/sweep/final
python make_report.py                                                                           # this page (site/)</pre>

<h2 id="nas28">Get it from nas28</h2>
<p class="sub">The kit below is the 2026-09-17 state (the earlier layout). The 09-23 fit (<code>sweep_replay.py</code>, <code>fit_layout.py</code>,
<code>replay/sweep/</code>, the new <code>layout.json</code>) is on the workstation and not in it yet.</p>
<p>The scene, the SIMPLE task, the replay tool, every run behind the numbers above and this page are in the SIMPLE tasks kit on NAS-28.
It is the 2026-09-09 kit (tabletop box and chips-can tasks) plus this scene; the 09-09 kit is untouched.</p>
<pre>/mnt/nas28/alan.jiang/holobrain_simple_tasks_20260917/            (NAS-28 = 10.40.11.28:/volume1/NAS-28, mounted at /mnt/nas28)
├── README_DEPLOY.md                                              set-up of the whole kit; section "Added 2026-09-17" for this scene
├── holobrain_g1_deploy/sim/bottle_bin/
│   ├── build_scene.py  scene.xml  layout.json  renders/  assets/  standalone MuJoCo scene, the numbers, the meshes (OBJ + USDA)
│   ├── bottle_bin_task.py                                        SIMPLE task  simple/G1WholebodyBottleBinTeleop-v0  (reads layout.json)
│   ├── replay_in_scene.py                                        action replay of a real recording inside the scene
│   ├── replay/                                                   all runs: *_scan (7 cm), *_scan3cm, *_verify, 00-16-29 finals; traces, bin_fit.py
│   ├── make_report.py  site/  report.html                        this page and its generator
│   └── README.md                                                 the scene's own notes and change log
├── holobrain_g1_deploy/sim/tabletop_box/                         the box / chips-can tasks and the g1comp robot (from the 09-09 kit)
├── holobrain_g1_deploy/tools/simple_eval_ws.py                   the holobrain_ws agent, psi0_protocol.py
├── data/evals/                                                   the box / can eval sets
└── docs/bottle_bin_scene/index.html                              this page (offline copy), docs/tabletop_scene.html</pre>
<h3>Requirements on the other machine</h3>
<ul>
<li><code>/mnt/nas28</code> mounted (<code>sudo mount -t nfs4 10.40.11.28:/volume1/NAS-28 /mnt/nas28</code>).</li>
<li>A SIMPLE checkout with its venv, the G1 assets and HSSD scene 0, and the sonic teleop stack importable in that venv
(<code>gear_sonic</code>, <code>decoupled_wbc</code>, <code>unitree_sdk2_python</code>; the alt-layout kit at
<code>/mnt/nas28/alan.jiang/altlayout_eval_kit/</code> carries SIMPLE's data, patch and third-party checkouts if the machine has none).</li>
<li>Rendering: <code>MUJOCO_GL=egl</code> headless (the replay and the probe), or <code>MUJOCO_GL=glfw DISPLAY=:1</code> for the standalone scene renders.
The standalone scene also needs the holomotion g1comp model (<code>G1_COMP_MJCF=/path/to/g1_comp_45dof.xml</code>; the meshes are in the kit's
<code>sim/tabletop_box/g1comp_meshes/</code>).</li>
<li>Real recordings for the replay: <code>SIMPLE/data/real_recordings/&lt;session&gt;/</code> (LeRobot v2.1, 30 or 50 fps); the sessions used here are
<code>2026-09-17-02-25-56-G1-sim</code> and <code>2026-09-17-00-16-29-G1-sim</code> on the workstation (copy them from there or from the psi_g1_chipcantotrash LMDB source).</li>
</ul>
<h3>Set up and run</h3>
<pre>cp -r /mnt/nas28/alan.jiang/holobrain_simple_tasks_20260917/holobrain_g1_deploy ~/wrk/robot_orchard_deploy/   # any path
cd ~/wrk/robot_orchard_deploy/holobrain_g1_deploy/sim/bottle_bin
sed -i 's#/home/Horizon/wrk/SIMPLE#/path/to/SIMPLE#' replay_in_scene.py         # ROOT = the SIMPLE checkout (recordings live under its data/)

# 1. the scene inside SIMPLE (no policy, no recording): probe.json + third_person.png + head_stereo_left.png in replay/probe/
MUJOCO_GL=egl /path/to/SIMPLE/.venv/bin/python replay_in_scene.py --probe

# 2. replay a recording in the scene (one process per episode; mp4 + trace + grasp / in-bin check per episode, index.html)
MUJOCO_GL=egl /path/to/SIMPLE/.venv/bin/python replay_in_scene.py --session 2026-09-17-02-25-56-G1-sim \
    --episodes 0 11 22 --robot-to-edge 0.03 --bottle-mass 0.5 --bin-xy -0.25 -0.84 --tag mine
python replay/bin_fit.py --tag mine                                             # bin centre from the release points of that pass

# 3. the standalone MuJoCo scene and this page
MUJOCO_GL=glfw DISPLAY=:1 /path/to/SIMPLE/.venv/bin/python build_scene.py      # scene.xml, layout.json, renders/, plan
python make_report.py                                                           # site/index.html + img/ (+ vid/ if present)

# 4. use the task from your own code
import sys; sys.path.insert(0, "…/sim/bottle_bin"); import bottle_bin_task      # registers simple/G1WholebodyBottleBinTeleop-v0
env = gym.make("simple/G1WholebodyBottleBinTeleop-v0", sim_mode="mujoco", sonic_config=cfg, headless=True, target="bottle_bin:bottle_500ml")</pre>
<p class="sub">Knobs: <code>BOTTLE_BIN_ROBOT_TO_EDGE</code>, <code>BOTTLE_BIN_BOTTLE_MASS</code>, <code>BOTTLE_BIN_BIN_XY</code>, <code>BOTTLE_BIN_BOTTLE_XY</code>,
<code>BOTTLE_BIN_TARGET</code> (env vars read by the task; the replay's flags set them). Every distance is in the pelvis frame of the start pose:
x toward the table, the robot's right = −y, floor at z = 0. Changing <code>build_scene.py</code>'s defaults and rebuilding changes <code>layout.json</code>, which the task reads.</p>
</main>
"""


def rel_str(ep):
    r = run_summary(SESSION, ep, "verify")
    return f"({r['release']['xy'][0]:+.2f}, {r['release']['xy'][1]:+.2f})" if r and r["release"] else "?"


# site (relative links) ------------------------------------------------------------------
fig.mode = "site"; b = body()
(SITE / "index.html").write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + b + "</html>")
(SITE / "artifact.html").write_text(b)
# standalone -----------------------------------------------------------------------------
fig.mode = "inline"; b2 = body()
(HERE / "report.html").write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' + b2 + "</html>")
for f in ("site/index.html", "site/artifact.html", "report.html"):
    print("wrote", HERE / f, f"{(HERE / f).stat().st_size/1e6:.2f} MB")
print("images:", len(list(IMG.iterdir())), "videos:", len(list((SITE / 'vid').iterdir())))
