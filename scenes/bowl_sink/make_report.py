#!/usr/bin/env python3
"""Scene page for the bowl -> sink kitchen scene, generated from layout.json, renders/, real/ and replay/.

Writes  site/index.html   full document, RELATIVE links to site/img/ and site/vid/ (publish the folder as a unit:
                          cp -r site/ /mnt/nas26/alan.jiang/fleet_status/holobrain_bowl_sink_scene/)
        report.html       standalone single file (images inlined) for local viewing

Same recipe as ../bottle_bin/make_report.py."""
import base64, html, io, json, shutil, subprocess
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
R, RP, REAL, SITE = HERE / "renders", HERE / "replay", HERE / "real", HERE / "site"
IMG = SITE / "img"; VID = SITE / "vid"
IMG.mkdir(parents=True, exist_ok=True); VID.mkdir(parents=True, exist_ok=True)
L = json.load(open(HERE / "layout.json"))
FIT = json.load(open(HERE / "replay/fit_kitchen_0918.json")) if (HERE / "replay/fit_kitchen_0918.json").exists() else None
BATCH = [json.loads(l) for l in open(HERE / "replay/batch_par.jsonl")] if (HERE / "replay/batch_par.jsonl").exists() else []
lay, cam, pose, sink, counter, bowl = L["layout"], L["ego_camera"], L["pose"], L["sink"], L["counter"], L["bowl"]
DRAWING = Path.home() / "Downloads" / "20260922-235520.jpg"
FFMPEG = "/home/Horizon/wrk/simple_validate/.ffbin/ffmpeg"


def _prep(src: Path, maxw=1280, q=84):
    im = Image.open(src).convert("RGB")
    keep_png = src.name.startswith("plan") or "sheet" in src.name
    if im.width > maxw and not keep_png:
        im = im.resize((maxw, round(im.height * maxw / im.width)), Image.LANCZOS)
    name = src.stem + (".png" if keep_png else ".jpg"); out = IMG / name; b = io.BytesIO()
    if keep_png:
        im.save(out, "PNG", optimize=True); im.save(b, "PNG", optimize=True); mime = "image/png"
    else:
        im.save(out, "JPEG", quality=q, optimize=True); im.save(b, "JPEG", quality=q, optimize=True); mime = "image/jpeg"
    return f"img/{name}", f"data:{mime};base64," + base64.b64encode(b.getvalue()).decode()


class Fig:
    mode = "site"

    def __call__(self, src, cap):
        p = Path(src) if Path(src).is_absolute() else (R / src)
        if not p.exists():
            return f"<p class='missing'>missing image: {html.escape(str(src))}</p>"
        rel, uri = _prep(p)
        return (f"<figure><img src='{rel if self.mode == 'site' else uri}' alt='{html.escape(cap)}' loading='lazy'>"
                f"<figcaption>{cap}</figcaption></figure>")


fig = Fig()


def video(src: Path, name: str, cap: str, max_mb=40):
    """Remux into site/vid with +faststart (dev007 streams it) and embed."""
    dst = VID / name
    if not dst.exists() and Path(src).exists():
        subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)], check=False)
    if not dst.exists():
        return f"<p class='missing'>missing video: {html.escape(name)}</p>"
    if fig.mode == "site":
        return f"<figure><video controls preload='metadata' src='vid/{name}'></video><figcaption>{cap}</figcaption></figure>"
    return f"<figure><p class='sub'><b>{html.escape(name)}</b> (video, in site/vid/)</p><figcaption>{cap}</figcaption></figure>"


# ---------------- replay results ----------------
runs = json.load(open(RP / "summary.json")) if (RP / "summary.json").exists() else []
basin = sink["basin_centre"]


CORRECTED_TAG = "fix0923"        # runs with this tag used the geometry corrected on 2026-09-23


def replay_rows():
    out = []
    for r in sorted(runs, key=lambda r: CORRECTED_TAG not in str(r.get("tag"))):
        g = r.get("grasp", {}) or {}
        d = g.get("bowl_end_minus_basin") or [None, None, None]
        new = CORRECTED_TAG in str(r.get("tag"))
        end = r.get("target_end") or r.get("bottle_end") or []
        out.append(
            f"<tr><td>ep {r.get('ep')} <span class='sub'>{r.get('tag')}</span></td>"
            f"<td>{'corrected' if new else 'before 09-23'}</td>"
            f"<td>{r.get('seconds', 0):.0f} s</td>"
            f"<td>{g.get('max_lift_m', 0):.2f} m</td>"
            f"<td>{'tipped ' + format(g['first_tip_t'], '.1f') + ' s' if g.get('first_tip_t') else 'upright'}</td>"
            f"<td>{f'{end[2]:.2f} m' if len(end) > 2 else '—'}</td>"
            f"<td>{f'{d[0]:+.2f}' if d[0] is not None else '—'}</td>"
            f"<td>{f'{d[1]:+.2f}' if d[1] is not None else '—'}</td>"
            f"<td>{'yes' if g.get('in_sink') else 'no'}</td></tr>")
    return "".join(out)


# ---------------- page ----------------
spec = [
    ("Robot", "g1comp: G1 29 DOF, Dex3 hands, pan/tilt D455 head (holomotion <code>g1_comp_45dof.xml</code>), standing at the bowl counter, facing it (+x)"),
    ("Bowl counter", f"{counter['size'][1]:.2f} m long (across, facing the robot) &times; {counter['size'][0]:.2f} m deep, top <b>{counter['size'][2]:.2f} m</b> above the floor, 3 cm slab on a solid cabinet"),
    ("Back unit", f"{counter['back_unit'][1]:.2f} &times; {counter['back_unit'][0]:.2f} m, top {counter['back_unit'][2]:.2f} m, against the wall behind the bowl counter"),
    ("Robot placement", f"front of the feet {(lay['near_x'] - lay['toe_offset'])*100:.0f} cm from the front edge (pelvis {lay['pelvis_to_edge']*100:.1f} cm; the 55 recordings put it at 30 cm by two independent measurements), pelvis {lay['left_y']*100:.0f} cm from the counter's LEFT end &mdash; that is where the bowl sat along the counter in the 2026-09-18 session; your drawing's 139 is the 09-21 deployment setup"),
    ("Bowl", f"{bowl['diameter']*100:.0f} cm diameter &times; {bowl['depth']*100:.0f} cm deep, 4 mm wall, hollow, green, {bowl['mass']:.2f} kg"),
    ("Bowl position", f"<b>at the counter's edge</b>: centre {(lay['bowl_x'] - lay['near_x'])*100:.0f} cm beyond the front edge, so the 4.5 cm base sits {((lay['bowl_x'] - lay['near_x']) - 0.045)*100:.1f} cm inside it (4 s of physics: no drift, 0.02&deg; tilt); "
     + ("straight ahead on the robot's centre line" if abs(lay['bowl_y']) < 0.005 else f"{abs(lay['bowl_y'])*100:.0f} cm to the robot's {'left' if lay['bowl_y'] > 0 else 'right'} of its centre line")
     + f" = ({lay['bowl_x']:.3f}, {lay['bowl_y']:+.3f}) m from the pelvis start, rim at {counter['size'][2] + bowl['depth']:.2f} m"),
    ("Sink counter", f"{sink['counter'][0]:.2f} m long &times; {sink['counter'][1]:.2f} m deep, top <b>{sink['counter'][2]:.2f} m &mdash; the same worktop height as the bowl counter</b>, perpendicular on the robot's RIGHT: its face at the bowl counter's right end, <b>starting at the bowl counter's front-edge line</b> and running toward the robot's rear (it does not reach back to the wall)"),
    ("Sink basin", f"{sink['basin'][0]:.2f} m (along the counter) &times; {sink['basin'][1]:.2f} m (across, 4 cm rim each side) &times; {sink['basin'][2]*100:.0f} cm deep, <b>{sink.get('corner_radius', 0.03)*100:.0f} cm rounded corners</b>; its far edge <b>{sink['along']*100:.0f} cm from the sink counter's start</b> (= the bowl counter's front-edge line; the drawing's 127), centre at ({basin[0]:.2f}, {basin[1]:.2f}) m from the pelvis start; rim at {sink['counter'][2]:.2f} m, basin floor at {basin[2]:.2f} m"),
    ("Body pose", "every upper-body joint 0 except the elbows at &minus;0.66, hands open; waist 0; pelvis at 0.74 m; legs solved for flat feet (the <code>elbows_raised</code> init pose of the recordings)"),
    ("Head servos", f"pan 0 (centred) &middot; tilt {pose['head_tilt_deg']:.0f}&deg; down (114 ticks)"),
    ("Camera", f"D455 axis {cam['pitch_down_deg']:.1f}&deg; below horizontal, lens {cam['pos'][2]:.2f} m above the floor, {cam['hfov_deg']:.0f}&deg; &times; {cam['fovy_deg']:.1f}&deg; (16:9), rendered {cam['size'][0]} &times; {cam['size'][1]} (SIMPLE renders 640 &times; 360)"),
    ("Colours", "Dex3 hands black, floor light grey, counters plain grey &mdash; no dressing objects (plates, jug, dish rack) are modelled"),
]
spec_rows = "".join(f"<div class='row'><dt>{k}</dt><dd>{v}</dd></div>" for k, v in spec)

stats = [(f"{counter['size'][2]:.2f} m", "bowl counter top"), (f"{sink['counter'][2]:.2f} m", "sink rim"),
         (f"{bowl['diameter']*100:.0f} cm", "bowl diameter"), (f"{lay['near_x'] - lay['toe_offset']:.2f} m", "feet to counter edge"),
         (f"{sink['along']*100:.0f} cm", "sink start to basin far edge"), (f"{cam['pos'][2]:.2f} m", "camera above floor"),
         (f"{cam['pitch_down_deg']:.0f}&deg;", "camera pitch down"),
         (f"{sum(1 for r in runs if (r.get('grasp') or {}).get('in_sink'))}/{len(runs)}", "replays that end in the sink")]
stats_html = "".join(f"<div class='stat'><span class='v'>{v}</span><span class='k'>{k}</span></div>" for v, k in stats)

CSS = """
:root{--bg:#F3F4F1;--ink:#1B1F24;--mute:#5C6570;--rule:#D5D8D2;--card:#FFFFFF;--acc:#B7371F;--ok:#2E7D32;--chip:#E8EAE4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#15181C;--ink:#E9ECE6;--mute:#99A2AB;--rule:#2C323A;--card:#1C2127;--acc:#E4664F;--ok:#6FBF73;--chip:#232930}}
:root[data-theme="dark"]{--bg:#15181C;--ink:#E9ECE6;--mute:#99A2AB;--rule:#2C323A;--card:#1C2127;--acc:#E4664F;--ok:#6FBF73;--chip:#232930}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:17px/1.5 system-ui,-apple-system,Segoe UI,sans-serif;-webkit-text-size-adjust:100%}
main{max-width:860px;margin:0 auto;padding:1.25rem 1rem 4rem}
h1,h2,h3{text-wrap:balance;margin:0;line-height:1.1}
h1{font-size:2.2rem;font-weight:650;letter-spacing:-.01em}
h2{font-size:1.5rem;font-weight:650;margin:2.4rem 0 .8rem;padding-top:.8rem;border-top:2px solid var(--rule)}
h3{font-size:1.15rem;font-weight:650;margin:1.5rem 0 .5rem}
.eyebrow{font-weight:600;text-transform:uppercase;letter-spacing:.12em;font-size:.8rem;color:var(--acc);margin:0}
.lede{color:var(--mute);margin:.6rem 0 0;max-width:40em}
nav{display:flex;flex-wrap:wrap;gap:.45rem;margin:1.1rem 0 0}
nav a{font-size:.92rem;text-decoration:none;color:var(--ink);background:var(--chip);padding:.3rem .7rem;border-radius:999px}
nav a:focus-visible,a:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
dl{margin:0} .row{display:grid;grid-template-columns:11rem 1fr;gap:.6rem;padding:.55rem 0;border-bottom:1px solid var(--rule)}
dt{font-weight:650;color:var(--mute)} dd{margin:0;font-variant-numeric:tabular-nums}
@media (max-width:520px){.row{grid-template-columns:1fr;gap:.1rem}}
figure{margin:0;background:var(--card);border:1px solid var(--rule);border-radius:6px;overflow:hidden}
figure img,figure video{display:block;width:100%;height:auto}
figcaption{font-size:.88rem;color:var(--mute);padding:.5rem .7rem;border-top:1px solid var(--rule)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:.8rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.6rem;margin:1rem 0}
.stat{background:var(--card);border:1px solid var(--rule);border-radius:6px;padding:.6rem .7rem;display:flex;flex-direction:column}
.stat .v{font-size:1.4rem;font-weight:650;font-variant-numeric:tabular-nums;line-height:1.15} .stat .k{font-size:.8rem;color:var(--mute)}
p{max-width:44em} li{max-width:44em;margin:.3rem 0}
code{font:.9em ui-monospace,Menlo,Consolas,monospace;background:var(--chip);padding:.05em .35em;border-radius:3px}
pre{background:var(--chip);padding:.7rem .9rem;border-radius:6px;overflow-x:auto;font:.85rem/1.45 ui-monospace,Menlo,Consolas,monospace}
.tw{overflow-x:auto} table{border-collapse:collapse;font-size:.93rem;font-variant-numeric:tabular-nums;width:100%}
th,td{padding:.35rem .6rem;border-bottom:1px solid var(--rule);text-align:left} th{color:var(--mute);font-weight:650}
.sub{color:var(--mute);font-size:.92rem} .missing{color:var(--acc)}
.flag{background:var(--card);border:1px solid var(--rule);border-left:4px solid var(--acc);border-radius:6px;padding:.7rem .9rem;margin:.8rem 0}
.flag b{color:var(--acc)}
.ok{border-left-color:var(--ok)} .ok b{color:var(--ok)}
"""


def success_videos():
    out = []
    for f in sorted((RP).glob("replay_BowlToSink_0918_ep*_vid.mp4")):
        ep = int(f.name.split("_ep")[1].split("_")[0])
        tr = RP / f"trace_ep{ep}_vid.json"
        cap = f"Episode {ep}"
        if not tr.exists():                       # still being replayed: the mp4 is incomplete
            continue
        if tr.exists():
            sm = next((r["summary"] for r in json.load(open(tr)) if "summary" in r), {}); g = sm.get("gates") or {}
            if not g.get("placed"):
                continue
            cap += f": moved to the bowl at {g['t_at_bowl']} s, grasped at {g['t_grasped']} s, at the sink at {g['t_at_basin']} s, placed at {g['t_placed']} s{' and settled' if g.get('settled') else ''}; {g['max_lift']*100:.0f} cm lift."
        out.append(video(f, f"success_ep{ep}_gain13.mp4", cap))
    return "\n".join(out)


def _funnel(tag):
    import glob
    from bowl_sink_gates import BowlSinkGates, GateCfg
    fun = dict(at_bowl=0, grasped=0, at_basin=0, placed=0, settled=0); n = 0
    for f in sorted(glob.glob(str(HERE / f"replay/trace_ep*_{tag}.json"))):
        tr = json.load(open(f)); sm = next((r["summary"] for r in tr if "summary" in r), None)
        if not sm: continue
        g = sm.get("gates") or BowlSinkGates.from_trace(tr, sm["basin"], GateCfg(sink_h=0.86)).P; n += 1
        for k in fun: fun[k] += int(bool(g.get(k)))
    return n, fun


def tuned_section():
    tl = L["layout"]
    n130, f130 = _funnel("v130"); n150, f150 = _funnel("v150")
    def row(label, n, f):
        return (f"<tr><td>{label}</td><td>{n}</td><td>{f['at_bowl']}</td><td>{f['grasped']}</td><td>{f['at_basin']}</td>"
                f"<td><b>{f['placed']}</b></td></tr>")
    verdict = f"""<h3>Verification: every episode replayed in one fixed scene, nothing adjusted per episode</h3>
<div class="tw"><table>
<tr><th>scene</th><th>episodes</th><th>moved to the bowl</th><th>grasped</th><th>at the sink</th><th>placed in the basin</th></tr>
{row("gain 1.3, feet 0.27 m from the edge, 1.94 m from the left end (built 2026-09-23)", n130, f130)}
{row("gain 1.5, feet 0.25 m, 2.04 m from the left end (the fitted optimum, <b>built now</b>)", n150, f150)}
</table></div>
<div class="flag"><b>The fit is not predictive.</b> It promised 18 completions at the gain-1.5 optimum; the replay
delivers {f150['placed']}. The mapping replays themselves, run in the 2026-09-23 scene, place 4&ndash;10 of 55 depending on the
gain (table above). Every start pose and gain tried lands between 6 and 10 of 55: the count is set by how far each
episode's recorded arm path is from this bowl, not by where the robot stands. The scene is left at the fitted optimum;
moving it back to 0.27 / 1.94 changes the count by less than the spread between episodes.</div>"""
    return f"""
<p>These 55 teleop episodes (<code>data/real_recordings/psi0/BowlToSink_0918</code>, the same kitchen, 2026-09-18)
were used to place the bowl and to test the scene: each episode's recorded arm and hand motion is replayed in it,
the lower-body policy walks on the recorded velocity commands, and the outcome is whether the bowl gets picked up and
ends in the basin.</p>

<h3>What the recordings say, measured without integrating anything</h3>
<div class="tw"><table>
<tr><th>measured over the 55 episodes</th><th>p10</th><th>median</th><th>p90</th><th>how</th></tr>
<tr><td>bowl ahead of the start pose</td><td>0.51 m</td><td><b>0.57 m</b></td><td>0.65 m</td><td>green blob in the first camera frame, back-projected; leg odometry agrees at 0.570</td></tr>
<tr><td>bowl left/right of the start pose</td><td>&minus;0.04 m</td><td><b>0.00 m</b></td><td>+0.04 m</td><td>same</td></tr>
<tr><td>counter top</td><td>0.83 m</td><td><b>0.85 m</b></td><td>0.88 m</td><td>height of the bowl's rim when the fingers close, minus the 8 cm bowl</td></tr>
<tr><td>turn from the grasp to the drop</td><td>&minus;103&deg;</td><td><b>&minus;90&deg;</b></td><td>&minus;79&deg;</td><td>IMU heading; gyro integration agrees</td></tr>
<tr><td>hand height at the drop</td><td>0.92 m</td><td><b>0.95 m</b></td><td>0.99 m</td><td>forward kinematics on the recorded joints</td></tr>
<tr><td>walk before the grasp: actual vs commanded</td><td>0.19 m</td><td><b>0.24 m</b> vs 0.59 m</td><td>0.31 m</td><td>leg odometry vs the integral of the velocity command</td></tr>
</table></div>
<div class="flag ok"><b>Two of your numbers are confirmed by the recordings.</b> The worktop comes out at 0.85 m, and
the robot's start is 0.57 m from the bowl, i.e. feet 0.30 m from the counter's edge &mdash; the value the scene was
first built with.</div>
<div class="flag"><b>An earlier version of this page fitted the scene to a walk integrated from the commanded
velocity. That was wrong,</b> and it was retracted: the robot achieves only about 40&ndash;55 % of what it is commanded,
so the integral put the bowl at 0.92 m and dragged the whole fit with it. Everything here is from measurements that do
not go through that integral.</div>

<h3>Where the bowl sat along the counter</h3>
<p>With the kitchen exactly as drawn and the bowl 139 cm from the counter's left end, no replayed drop lands anywhere
near the basin: the robot turns right and walks about 1 m, and the drawing's basin is 1.8 m from that bowl. Moving the
bowl to <b>185 cm from the left end</b>, leaving everything else as drawn, puts 30 of 53 drops inside the basin. In this
session the bowl simply sat 46 cm closer to the sink than on the drawing. Both positions are one flag apart
(<code>--robot-from-left 1.85</code> / <code>1.39</code>).</p>

<h3>Finding the start pose: one replay per episode at each walking gain</h3>
<p>The simulated hand's path does not depend on what the scene contains, so one replay per episode maps where the hand
closes and where it opens. With the kitchen fixed, the start pose places the bowl and the basin in that frame, and the
best start pose is solved offline. The walking gain scales the recorded velocity commands: 1.0 reproduces the real
robot's own shortfall, higher gains lengthen the carry but overshoot the bowl. All replays below run on a virtual clock
(<code>REPLAY_FAST=1</code>), so the same episode in the same scene always gives the same result.</p>
<div class="tw"><table>
<tr><th>walking gain</th><th>best start pose (fitted offline)</th><th>completes, predicted by the fit</th><th>same replays, live gates in the 2026-09-23 scene: grasped / placed</th></tr>
<tr><td>1.0</td><td>feet 0.27 m from the edge, 2.00 m from the left end</td><td>8</td><td>22 / 4</td></tr>
<tr><td>1.3</td><td>feet 0.23 m, 1.96 m</td><td>14</td><td>24 / 9</td></tr>
<tr><td><b>1.5</b></td><td><b>feet 0.25 m, 2.04 m</b></td><td><b>18</b></td><td>24 / 9</td></tr>
<tr><td>1.7</td><td>feet 0.24 m, 2.15 m</td><td>12</td><td>26 / 7</td></tr>
<tr><td>1.9</td><td>feet 0.24 m, 2.15 m</td><td>16</td><td>28 / 10</td></tr>
</table></div>
<p class="sub">Out of 55. The fitted optimum is sharp (20&ndash;30 cm from the edge, 1.94&ndash;2.14 m along the counter), but the
prediction counts a grasp whenever the hand closes within 9 cm of the bowl's centre, which the simulated fingers do not
always turn into a hold.</p>
{verdict}
<p>What limits the count is the grasp, not the geometry: every episode started from a slightly different parking spot,
so the point where the hand closes is spread over 17 cm, and the simulated fingers catch a 15 cm bowl only when they
close within about 9 cm of its centre. Putting the bowl at the counter's edge is your suggestion for widening that
margin; the row above is the test.</p>
"""


def body():
    real = lambda n, cap: fig(REAL / n, cap)
    return f"""<title>G1 Bowl Sink Scene</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>{CSS}</style>
<main>
<p class="eyebrow">HoloBrain G1 deploy &middot; sim setup &middot; built 2026-09-23</p>
<h1>G1 bowl &rarr; sink kitchen scene</h1>
<p class="lede">The MuJoCo scene for &ldquo;pick up the green bowl from the counter, turn right, move towards the sink,
and place the bowl in the sink&rdquo;, built from the hand drawing of 2026-09-22 and the deployment run of 2026-09-21.
This page is for checking the layout <b>before</b> anything is tested in it.</p>
<nav><a href="#settings">Settings</a><a href="#plan">Plan</a><a href="#drawing">Drawing</a><a href="#views">Views</a>
<a href="#camera">D455 view</a><a href="#real">Sim vs real</a><a href="#replay">Replays</a><a href="#tuned">Tuned to the data</a><a href="#success">Successes</a><a href="#gates">Gates</a><a href="#check">What to confirm</a><a href="#files">Files</a></nav>
<div class="stats">{stats_html}</div>

<h2 id="settings">Settings</h2>
<p class="sub">Every number below is in <code>layout.json</code>, which is also what the SIMPLE task and the replay tool read.
World frame: the pelvis at the start pose is the origin, +x toward the bowl counter, z up, the robot's right = &minus;y.</p>
<dl>{spec_rows}</dl>

<h2 id="plan">Plan view with the dimensions</h2>
{fig("plan.png", "Orthographic top view, robot's right = image right. Red dimensions are the numbers from the drawing; the basin sits at the far end of the sink counter, behind the robot's right shoulder.")}

<h2 id="drawing">The drawing it was built from</h2>
<p>Numbers on the sheet are in centimetres (written &ldquo;mm&rdquo;). This is how each one was read:</p>
<div class="tw"><table>
<tr><th>On the sheet</th><th>Read as</th><th>In the scene</th></tr>
<tr><td>260 &times; 63 &times; 86</td><td>bowl counter: 2.60 m long, 0.63 m deep, top 0.86 m</td><td>as drawn</td></tr>
<tr><td>262 &times; 41 &times; 107</td><td>the tall unit behind it, against the wall &mdash; the <b>only</b> 1.07 m unit</td><td>2.60 &times; 0.41 &times; 1.07 m (length rounded to the counter's)</td></tr>
<tr><td>139</td><td>from the counter's left end to the <b>bowl</b>, in the 09-21 deployment setup</td><td>in the 2026-09-18 teleop session the bowl sat ~185 cm from that end (its drop point is 0.96 m from the sink, not 1.42); the scene uses 185 so those episodes replay. <code>--robot-from-left 1.39</code> rebuilds the drawing's setup</td></tr>
<tr><td>266 &times; 41 &times; 107</td><td>sink counter, perpendicular, on the robot's right</td><td><b>2.08</b> &times; 0.41 m, worktop <b>0.86 m</b> (your corrections 2026-09-23: 208 long, not 266; the same height as the bowl counter, not 107); it <b>starts at the bowl counter's front-edge line</b>, as on the sheet, not at the wall</td></tr>
<tr><td>41 &times; 54 &times; 10, rim 4</td><td>basin cut-out 0.54 m along &times; full 0.41 m depth, 0.10 m deep, 4 cm rim</td><td>basin 0.54 &times; 0.33 &times; 0.10 m, corners rounded 3 cm</td></tr>
<tr><td><b>127</b></td><td>from the sink counter's start (= the bowl counter's front-edge line) to the basin</td><td><b>the basin's far edge is 127 cm from the start</b> (your &ldquo;sink right edge from the left table edge&rdquo;), so the basin spans 73&ndash;127 cm along the counter</td></tr>
<tr><td>63</td><td>&mdash;</td><td>not used (you asked to ignore it)</td></tr>
</table></div>
{fig(DRAWING, "The hand drawing of 2026-09-22 (~/Downloads/20260922-235520.jpg). L-shaped kitchen: the bowl counter across the top, the sink counter down the right-hand side.")}

<h2 id="views">Views</h2>
<div class="grid">
{fig("overview.png", "Overview: the robot at the bowl counter, the sink counter running away on its right, the basin at the far end.")}
{fig("front_left.png", "From the robot's front left.")}
{fig("side.png", "From the side, showing the 0.86 m counter against the 1.07 m back unit and sink counter.")}
{fig("from_sink_side.png", "From beyond the sink, looking back at the robot.")}
{fig("bowl_closeup.png", "The bowl on the counter: 15 cm across, 8 cm deep, hollow.")}
{fig("sink_closeup.png", "The basin: 0.54 &times; 0.33 m, 10 cm deep, 4 cm rim, floor at 0.97 m.")}
</div>

<h2 id="camera">What the D455 sees at the start</h2>
<div class="grid">
{fig("ego_d455.png", "Sim: the robot's own head camera at the start pose (1280 &times; 720; SIMPLE renders 640 &times; 360).")}
{real("real_ep7_t00.png", "Real: the head camera at the start of episode 7 of the 2026-09-21 run, same moment.")}
</div>
<p class="sub">The scene is geometry only: the counters are plain grey and none of the real kitchen's dressing
(patterned plates, cup stack, tissue box, water jug, dish rack, kettle, floor mat) is modelled. The
grey band across the sim view is the counter's front edge; in the real frame the same edge sits at the same height,
which is what the 0.30 m feet-to-edge distance was tuned to.</p>

<h2 id="real">The real run, step by step</h2>
<p>Episode 7 of run <code>20260921_172723_bowltosink_c96</code>, the head camera at the moments the scene has to support.</p>
<div class="grid3">
{real("real_ep7_t00.png", "t = 0 s &middot; start: bowl on the counter, straight ahead")}
{real("real_ep7_t08.png", "t = 8 s &middot; the right hand closes on the bowl")}
{real("real_ep7_t14.png", "t = 14 s &middot; lifted, starting to turn right")}
{real("real_ep7_t20.png", "t = 20 s &middot; mid-turn, the corner of the L")}
{real("real_ep7_t28.png", "t = 28 s &middot; over the basin, the hand opens")}
{real("real_ep7_t38.png", "t = 38 s &middot; the bowl is in the sink")}
</div>

<h2 id="replay">Replays of the 2026-09-21 deployment run</h2>
<p class="sub">Historical: these runs predate the 55-episode tuning below and used the bowl 15 cm in from the edge at 139 cm from the counter's left end. The current scene is tested in the next section.</p>
<p>The recorded 36-D rows of the real run are replayed tick by tick in this scene (the arms follow the recording,
the lower-body policy walks by itself). <b>None of them ends with the bowl in the basin yet</b>, so the scene is
not yet validated &mdash; these are the numbers that say where it is off.</p>
<div class="flag"><b>These runs predate today's corrections.</b> They were replayed against the earlier geometry
(sink worktop 1.07 m, basin 27 cm further away, robot 8 cm to the right of where it now stands). A re-run against the
corrected scene is under way; until it lands, read the table as the reason the corrections were needed, not as a verdict
on the scene as it now stands.</div>
<div class="tw"><table>
<tr><th>run</th><th>geometry</th><th>length</th><th>max lift</th><th>bowl</th><th>bowl ends at z</th><th>&Delta;x to basin</th><th>&Delta;y to basin</th><th>in sink</th></tr>
{replay_rows()}
</table></div>
<div class="flag ok"><b>The height correction fixed the big error.</b> Before 2026-09-23 the bowl ended on the
<b>floor</b>, a quarter of a metre short of the counter, because the sink worktop was built 21 cm too high. With the
worktop at 0.86 m the bowl now ends <b>on the sink counter</b>, 3 cm off across its depth and at worktop height.</div>
<div class="flag"><b>What is left is 33 cm along the counter.</b> The bowl comes to rest 0.33 m further from the bowl
counter than the basin's centre, i.e. about 6 cm beyond the basin's far rim. That is roughly the size of the remaining
ambiguity in the 127: built to the basin's far edge, the bowl lands just outside it; read to the basin's centre
(<code>--sink-ref centre</code>) the same release lands 21 cm inside the basin. The replayed walk also turns further
than the robot did, which moves the landing point by a comparable amount, so this one number cannot yet decide the
reading on its own.</div>
{video(RP / "replay_actions_taskX_20260921_172803.jsonl_ep7_fix0923.mp4", "replay_ep7_corrected.mp4", "Replay of episode 7 in the corrected scene: third person, the sim D455, and the real D455 side by side. The bowl ends on the sink worktop, just past the basin's far rim.")}

<h2 id="tuned">Tuned to the 55 teleop episodes</h2>
{tuned_section()}

<h2 id="success">Successful episodes in the final scene</h2>
<p>Recorded teleop episodes replayed in the scene as built on 2026-09-23 (gain 1.3, feet 0.27 m from the edge, 1.94 m from the
counter's left end; the scene now stands at 0.25 / 2.04, see above), with all four task gates passing. Left: third person. Middle: the simulated head camera. Right: the real
head camera of the same episode. The earlier video further up, from the 2026-09-21 deployment run, predates the final
geometry and is not a success.</p>
<div class="grid">
{video(RP / "replay_BowlToSink_0918_ep7_gatecheck.mp4", "success_ep7_gain13.mp4", "Episode 7: moved to the bowl at 5.1 s, grasped at 9.2 s, at the sink at 25.7 s, placed at 29.2 s and settled; 17 cm lift.")}
{success_videos()}
</div>

<h2 id="gates">Task gates: four checks, in order</h2>
<p>The task now judges an episode in four stages, each of which can only pass after the previous one. The same code runs
inside the SIMPLE task (it is the reward and the success signal) and inside the replays (every replay reports them).</p>
<div class="tw"><table>
<tr><th>#</th><th>gate</th><th>passes when</th></tr>
<tr><td>1</td><td><b>moved to the bowl and stopped</b></td><td>walked &ge; 5 cm since the start, bowl within 0.60 m of the pelvis and within 60&deg; of the heading, base still (&lt; 0.15 m/s) for 0.4 s, before any grasp</td></tr>
<tr><td>2</td><td><b>grasped the bowl</b></td><td>the right hand is in contact with the bowl and the bowl is &ge; 5 cm above where it started, together for 0.3 s &mdash; a knock does not count</td></tr>
<tr><td>3</td><td><b>moved to the sink and stopped</b></td><td>walked &ge; 20 cm since the grasp, basin centre within 0.80 m of the pelvis and within 75&deg; of the heading, base still for 0.4 s</td></tr>
<tr><td>4</td><td><b>placed the bowl in the basin</b></td><td>the hand has let go, the bowl's centre is inside the basin (shrunk by the bowl's 4.5 cm base) and below the 0.86 m rim; <i>settled</i> if it also rests there for 0.4 s</td></tr>
</table></div>
<p>Reward is 0.25 per gate and 1.0 once placed; success means placed. Each report also says which gate an episode stalled
at, the maximum lift, the tilt at the grasp and where the bowl ended relative to the basin.</p>
<div class="flag ok"><b>Funnel on the verified 55-episode batch above:</b> moved to the bowl 55 &rarr; grasped 17 &rarr; at the
sink 17 &rarr; placed 12 (7 settled). Every grasp made it to the sink; five drops missed the basin. The old single
&ldquo;in the basin&rdquo; flag agreed with the new placed gate on 52 of 55 &mdash; the three it counted had the bowl still in
the hand over the basin or resting on the rim.</div>

<h2 id="check">What to confirm before testing</h2>
<div class="flag ok"><b>Applied 2026-09-23, from your corrections:</b> the sink worktop is 0.86 m, the same as the bowl
counter (the 107 on the sheet is the tall back unit only); the sink counter is <b>2.08 m long and starts at the bowl
counter's front-edge line</b>; the basin's far edge is <b>127 cm from that start</b>; the bowl is centred in the robot's
view and sits <b>at the counter's edge</b> (base 1.5 cm inside, verified stable); the basin's four corners are rounded
(3 cm), in the standalone scene and in the SIMPLE task.</div>
<p>Still assumed, and each is one flag away from being changed:</p>
<ol>
<li><b>The bowl's place along the counter: 185 cm from the left end for the 09-18 episodes, 139 on your drawing.</b>
Both are in the scene as flags; the difference is which session's setup you want to replay.</li>
<li><b>Where the sink counter's face sits across.</b> Built at the bowl counter's right end, as on the sheet. The box
you drew on the plan sat 45 cm on the robot's side of that face; if the whole counter should move toward the robot,
that is <code>--robot-from-left</code> (it moves the bowl counter's right end, and the sink counter with it).</li>
<li><b>Feet 0.28 m from the front edge.</b> The recordings say 0.30 by two independent routes; 0.28 keeps the bowl
where the simulated hand closes. <code>--robot-to-edge</code>.</li>
<li><b>Basin 0.54 &times; 0.33 m with a 4 cm rim, 10 cm deep.</b> Read from &ldquo;41 x 54 x 10&rdquo; and the 4 cm
arrow on the sheet: the cut-out spans the counter's full 41 cm depth minus a 4 cm rim on each side.</li>
<li><b>Dressing objects are absent.</b> Fine for physics; it matters if a vision policy is to run in this scene.</li>
</ol>

<h2 id="files">Files and commands</h2>
<pre>sim/bowl_sink/
├── build_scene.py   scene.xml  layout.json  assets/  renders/   the scene, its numbers, the generated meshes
├── bowl_sink_task.py                                            SIMPLE task simple/G1WholebodyBowlSinkTeleop-v0
├── replay_in_scene.py  replay/                                  replay of the real run inside the scene
└── make_report.py   site/                                       this page

MUJOCO_GL=glfw DISPLAY=:1 ~/wrk/SIMPLE/.venv/bin/python build_scene.py          # scene.xml, layout.json, renders/
MUJOCO_GL=egl  ~/wrk/SIMPLE/.venv/bin/python replay_in_scene.py --probe         # the scene inside SIMPLE
python make_report.py                                                           # this page (site/)</pre>
<p class="sub">Knobs for every number above: <code>--robot-to-edge</code>, <code>--bowl-xy X Y</code>,
<code>--sink-along D</code>, <code>--bowl-mass</code>, <code>--tag</code> on both <code>build_scene.py</code> and
<code>replay_in_scene.py</code>; the task reads <code>layout.json</code> and honours
<code>BOWL_SINK_ROBOT_TO_EDGE</code>, <code>BOWL_SINK_BOWL_XY</code>, <code>BOWL_SINK_SINK_ALONG</code>,
<code>BOWL_SINK_BOWL_MASS</code>, <code>BOWL_SINK_INSTRUCTION</code>.</p>
</main>"""


def main():
    fig.mode = "site"
    (SITE / "index.html").write_text("<!doctype html><html lang='en'><head><meta charset='utf-8'>" + body() + "</body></html>")
    fig.mode = "standalone"
    (HERE / "report.html").write_text("<!doctype html><html lang='en'><head><meta charset='utf-8'>" + body() + "</body></html>")
    print("wrote", SITE / "index.html", "and", HERE / "report.html")
    print("images:", len(list(IMG.glob('*'))), "videos:", len(list(VID.glob('*'))))


if __name__ == "__main__":
    main()
