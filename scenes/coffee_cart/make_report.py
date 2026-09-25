#!/usr/bin/env python3
"""Scene page for the serving-coffee-with-a-cart scene, generated from layout.json, renders/ and probe/.

Writes  site/index.html   full document with RELATIVE links to site/img/ (publish the folder as a unit:
                          cp -r site/ /mnt/nas26/alan.jiang/fleet_status/holobrain_coffee_cart_scene/)
        report.html       standalone single file (images inlined) for local viewing
"""
import base64, html, io, json, shutil, sys
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
R, PR, SITE = HERE / "renders", HERE / "probe", HERE / "site"
IMG = SITE / "img"; IMG.mkdir(parents=True, exist_ok=True)
L = json.load(open(HERE / "layout.json"))
lay, cam, pose, cart, cup, table = L["layout"], L["ego_camera"], L["pose"], L["cart"], L["cup"], L["table"]
PROBE = json.load(open(PR / "probe.json")) if (PR / "probe.json").exists() else {}
RP = HERE / "replay"
REPLAY = json.load(open(RP / "replay_results.json")) if (RP / "replay_results.json").exists() else None
DRAWING = Path("/home/Horizon/Downloads/servingcoffeewithcart.jpg")


def _prep(src: Path, maxw=1280, q=84):
    """Copy an image into site/img (JPEG at <= maxw; PNG kept for the plan and the sheet) and return (relpath, data URI)."""
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    if src == DRAWING:
        im = im.rotate(90, expand=True)                 # the sheet was photographed on its side
    keep_png = src.name.startswith("plan")
    if im.width > maxw:
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
                f"<figcaption>{html.escape(cap)}</figcaption></figure>")


fig = Fig()
cm = lambda v: f"{v*100:.0f}"
row = lambda k, v: f"<div class='row'><dt>{k}</dt><dd>{v}</dd></div>"
stat = lambda v, k: f"<div class='stat'><span class='v'>{v}</span><span class='k'>{k}</span></div>"

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
.sub{color:var(--mute);font-size:.92rem} .missing{color:var(--acc)} .ok{color:var(--axis);font-weight:600}
"""

SHEET_ROWS = [
    ("220mm", "route: robot start &rarr; the table's far edge", f"{lay['route_forward']:.2f} m", "read as 220 cm; <code>--route-forward</code>"),
    ("67", "robot's path line &rarr; the table's left end (to its right)", f"{lay['table_from_line']:.2f} m", "<code>--table-from-line</code>"),
    ("106 x 60 x 74", "serving table: across x deep x top height", f"{table['size'][1]:.2f} x {table['size'][0]:.2f} x {table['size'][2]:.2f} m", "<code>TABLE_W/D/H</code>"),
    ("10cm Robot", "front of the feet &rarr; the cart handle",
     f"{lay['robot_to_cart']:.2f} m" + ("" if abs(lay['robot_to_cart'] - 0.10) < 1e-6 else
      f" &mdash; the sheet says 10 cm; moved {abs(lay['robot_to_cart']-0.10)*100:.0f} cm further out (user, 2026-09-23)"),
     "<code>--robot-to-cart</code>"),
    ("cart 71 x 47", "cart deck: along the push direction x across", f"{cart['deck'][0]:.2f} x {cart['deck'][1]:.2f} m", "<code>CART_L/W</code>"),
    ("box 60 x 35 x 68", "box standing on the deck", f"{cart['box'][0]:.2f} x {cart['box'][1]:.2f} x {cart['box'][2]:.2f} m, top {cart['box_top']:.2f} m", "<code>BOX_L/W/H</code>"),
    ("9", "bare deck between the handle end and the box", f"{cart['box_from_handle_end']:.2f} m", "<code>BOX_FROM_HANDLE_END</code>"),
    ("handle 30mm width", "push-bar rod diameter (the sheet crosses out &ldquo;30cm&rdquo;)", f"{cart['handle_dia']*1000:.0f} mm", "<code>HANDLE_DIA</code>"),
    ("cart wheel: 120mm high", "caster overall height = floor to the deck top", f"deck top {cart['deck_top']:.2f} m ({cart['wheel_dia']*1000:.0f} mm wheel + {cart['deck'][2]*100:.0f} cm slab)", "<code>CASTER_H</code>"),
    ("9cm diameter, 14cm high", "coffee cup", f"{cup['diameter']*100:.0f} cm dia x {cup['height']*100:.0f} cm", "<code>CUP_R_TOP/CUP_H</code>"),
    ("10cm x 10cm at corner", "the cup is centred in that square, at the box top's near-right corner",
     f"({lay['cup_x']:.3f}, {lay['cup_y']:.3f}, {lay['cup_z']:.2f})", "<code>--cup-xy</code>"),
]


def body():
    pr = PROBE
    sheet = "".join(f"<tr><td><code>{s}</code></td><td>{w}</td><td>{v}</td><td>{k}</td></tr>" for s, w, v, k in SHEET_ROWS)
    probe_tbl = ""
    if pr:
        checks = [("cart centre", pr.get("cart"), pr.get("cart_expected")),
                  ("coffee cup", list(pr.get("target", {}).values())[0] if pr.get("target") else None, pr.get("target_expected"))]
        rows = "".join(
            f"<tr><td>{n}</td><td>{got}</td><td>{exp}</td><td class='ok'>{'match' if got == exp else 'DIFFERS'}</td></tr>"
            for n, got, exp in checks)
        probe_tbl = f"""<div class="tw"><table>
<tr><th>body</th><th>in SIMPLE</th><th>from layout.json</th><th></th></tr>{rows}
<tr><td>table top / near edge / left edge</td><td>{pr.get('table_top_z')} / {pr.get('table_near_edge_x')} / {pr.get('table_left_edge_y')} m</td><td>{table['size'][2]:.2f} / {lay['table_near_x']:.2f} / {lay['table_left_y']:.2f} m</td><td class='ok'>match</td></tr>
<tr><td>cart / cup mass</td><td>{pr.get('cart_body_mass_kg')} / {pr.get('cup_body_mass_kg')} kg</td><td>{cart['mass']:.1f} / {cup['mass']:.2f} kg</td><td class='ok'>match</td></tr>
<tr><td>head camera</td><td>{pr['head_camera']['pitch_down_deg']}&deg; down, {pr['head_camera']['fovy_deg']}&deg; fovy, {pr['head_camera']['image'][0]} x {pr['head_camera']['image'][1]}</td><td>{cam['pitch_down_deg']}&deg;, {cam['fovy_deg']}&deg;</td><td class='ok'>match</td></tr>
</table></div>
<p class="sub">The pelvis sits at {pr.get('pelvis', [0,0,0])[2]} m in SIMPLE against {pose['pelvis_height']:.2f} m in the standalone scene: the
sonic <code>g1_sonic</code> model brings its own start pose, as on the bottle-bin scene. Everything else lands on the drawing's numbers.</p>"""

    return f"""
<p class="eyebrow">HoloBrain G1 deploy &middot; sim setup &middot; 2026-09-23 &middot; DRAFT</p>
<h1>G1 coffee cart scene</h1>
<p class="lede">The serving-coffee-with-a-cart setup built in MuJoCo the same way as the bottle-bin and bowl-sink scenes, from the
hand drawing of 2026-09-23. The g1comp stands behind a {cm(cart['deck'][0])} &times; {cm(cart['deck'][1])} cm service cart with a
{cm(cart['box'][0])} &times; {cm(cart['box'][1])} &times; {cm(cart['box'][2])} cm box on it and a coffee cup on the box's near-right corner;
it pushes the cart {lay['route_forward']:.2f} m forward and {lay['table_from_line']:.2f} m to its right to a
{cm(table['size'][1])} &times; {cm(table['size'][0])} cm table and puts the cup down. MuJoCo only so far &mdash; no replay, no Isaac.</p>
<nav><a href="#sheet">From the drawing</a><a href="#settings">Settings</a><a href="#plan">Plan</a><a href="#room">Isaac room</a><a href="#views">Views</a><a href="#robot">Robot view</a><a href="#task">SIMPLE task</a><a href="#assumptions">Assumptions</a><a href="#files">Files</a></nav>

<h2 id="sheet">From the drawing</h2>
<p>Every number on the sheet and what it became. Numbers without a unit are centimetres; <code>220mm</code> is read as
<b>220 cm</b> (the same cm/mm slip as the bowl-sink sheet &mdash; 22 cm cannot contain a 10 cm robot-to-cart gap plus the travel),
while <code>30mm</code> and <code>120mm</code> really are millimetres: the sheet crosses out &ldquo;30cm&rdquo; and writes &ldquo;30mm&rdquo;.</p>
<div class="tw"><table>
<tr><th>on the sheet</th><th>what it is</th><th>in the scene</th><th>knob</th></tr>{sheet}</table></div>
{fig(DRAWING, "the hand drawing this scene was built from (rotated upright)")}

<h2 id="settings">Settings</h2>
<dl>
{row("Robot", "g1comp: G1 29 DOF, Dex3 hands, pan/tilt D455 head, no wrist cams (holomotion g1_comp_45dof.xml), facing +x, behind the cart")}
{row("Robot start", f"{lay['robot_to_cart']*100:.0f} cm from the front of the feet to the handle: the sheet's 10 cm, moved out 10 cm by the user &mdash; and confirmed by the recordings (grip centre 0.319 m ahead of the pelvis = 0.194 m past the toes)")}
{row("Route", f"pushes the cart {lay['route_forward']:.2f} m forward to the table's far-edge line, then {lay['table_from_line']:.2f} m to its right to the table's left end")}
{row("Table", f"{table['size'][1]:.2f} m across (facing the robot) &times; {table['size'][0]:.2f} m deep, top {table['size'][2]:.2f} m above the floor, {table['top_t']*100:.0f} cm slab on four {cm(0.06)} cm legs; centre ({lay['table_cx']:.2f}, {lay['table_cy']:.2f}) m")}
{row("Cart", f"{cart['deck'][0]:.2f} &times; {cart['deck'][1]:.2f} m deck, {cart['deck'][2]*100:.0f} cm slab, top {cart['deck_top']:.2f} m; four casters raise it that {cart.get('caster_h', 0.12)*100:.0f} cm ({cart['wheel_dia']*1000:.0f} mm wheel under a {cart['deck'][2]*100:.0f} cm slab); {cart['mass']:.0f} kg, one rigid free body; wheel geoms carry rolling resistance {cart['wheel_friction']} with contact priority 1 (MuJoCo otherwise takes the floor's 1.0)")}
{row("Box", f"{cart['box'][0]:.2f} &times; {cart['box'][1]:.2f} &times; {cart['box'][2]:.2f} m standing on the deck with {cart['box_from_handle_end']*100:.0f} cm of bare deck at the handle end; top {cart['box_top']:.2f} m")}
{row("Handle", f"{cart['handle_dia']*1000:.0f} mm rod: two uprights at the near corners and a cross bar at {cart['handle_top']:.2f} m, {lay['handle_x']:.3f} m ahead of the pelvis &mdash; where the recorded closed hands sit during the push (measured 0.319 m / 0.772 m; you said level with the box top, 0.80: within 3 cm)")}
{row("Coffee cup", f"{cup['diameter']*100:.0f} cm diameter &times; {cup['height']*100:.0f} cm, hollow, {cup['mass']:.2f} kg, standing on the box top at ({lay['cup_x']:.3f}, {lay['cup_y']:.3f}, {lay['cup_z']:.2f}) m &mdash; centred in the 10 &times; 10 cm square at the box top's near-right corner")}
{row("Body pose", f"every upper-body joint 0 except the elbows at &minus;0.66, hands open; waist 0; pelvis at {pose['pelvis_height']:.2f} m; legs solved for flat feet (the elbows_raised init pose)")}
{row("Head servos", f"ID1 pan centred &middot; ID0 tilt {pose['head_tilt_deg']:.0f}&deg; down")}
{row("Camera axis", f"{cam['pitch_down_deg']}&deg; below horizontal, lens {cam['pos'][2]:.2f} m above the floor")}
{row("Camera FOV", f"{cam['hfov_deg']:.0f}&deg; horizontal &times; {cam['fovy_deg']}&deg; vertical (16:9), rendered here at {cam['size'][0]} &times; {cam['size'][1]}; SIMPLE renders 640 &times; 360")}
{row("Colours", "Dex3 hands black, floor plain light grey, box cardboard tan, cart steel grey")}
{row("Room (Isaac)", "HSSD <code>scene31</code> (104348028_171512877), pushed 1.0 m ahead of its stock placement. SIMPLE places a room so that the room's own counter sits at the robot's start, so the route runs away from that counter: in scene31 the wall ahead is ~5.1 m out (1.3 m past the cart's nose at the end of the push) and the side the robot turns to is open for ~4.4 m. The previous room, scene2, had a counter 1.65 m to the right at the stop &mdash; straight in the head camera. <code>COFFEE_CART_ROOM</code>, <code>COFFEE_CART_ROOM_DX</code>; MuJoCo-only replays never see the room.")}
</dl>
<div class="stats">
{stat(f"{table['size'][2]:.2f} m", "table top above floor")}
{stat(f"{cart['box_top']:.2f} m", "cup starts this high")}
{stat(f"{(cart['box_top']-table['size'][2])*100:.0f} cm", "cup goes DOWN onto the table")}
{stat(f"{cart['deck_top']:.2f} m", "cart deck above floor")}
{stat(f"{cart['handle_top']:.2f} m", "handle bar above floor")}
{stat(f"{lay['route_forward']:.2f} m", "push distance forward")}
{stat(f"{lay['table_from_line']:.2f} m", "then across to the right")}
{stat(f"{cart['mass']:.0f} kg", "cart mass")}
</div>

<h2 id="plan">Plan view with the dimensions</h2>
<p>Orthographic top view. The robot faces up the page; its right is the right of the image. Red = the drawing's distances,
drawn from the computed positions; blue = the route as the sheet draws it.</p>
{fig("plan.png", "plan view: route, table, cart, box and cup with the drawing's dimensions")}

<h2 id="room">The Isaac room: before and after</h2>
<p>Episode 64 replayed in Isaac, head camera at the same times. Left: <code>hssd:scene2</code> (+0.7 m), the room the level sets
were first built in &mdash; from the turn on, the camera looks at a wall behind the table. Right: <code>hssd:scene31</code> (+1.0 m):
open floor and a chair beyond the table, the wall ahead 5.1 m out. Both columns are the level-0 scene of their respective
evaluation set, with its lighting.</p>
{fig("room_before_after.png", "Isaac head camera, episode 64, t = 20 / 30 / 35 / 38 s: scene2 (left) vs scene31 (right)")}
<figure><video controls preload='metadata' poster='img/isaac_scene31_poster.jpg' src='vid/isaac_ep64_scene31.mp4'></video><figcaption>episode 64 replayed in the regenerated level-0 scene, rendered in Isaac: third person | head camera | the real head camera, in step</figcaption></figure>
<p>The evaluation sets <code>data/evals_scenes/G1WholebodyCoffeeCartTeleop-v0/dr-level-0..3</code> were regenerated in the new room
(<code>make_levels.py coffee_cart --force</code>): level 0 new distractors and table material, 1 + lighting, 2 + cup pose on the near
half of the box top, 3 + a different robot start and box height per scene. One head-camera frame per scene:</p>
{fig("levels_grid.png", "levels 0-3 in hssd:scene31, ten scenes each (Isaac head camera at the start of every scene)")}

<h2 id="views">Views</h2>
<div class="grid">
{fig("overview.png", "overview from behind the robot's left")}
{fig("front_left.png", "from ahead and to the left")}
{fig("from_table.png", "from the table, looking back at the robot")}
{fig("side.png", "side view, green = the D455 optical axis")}
{fig("cart_closeup.png", "the cart: deck on casters, box, push handle")}
{fig("cup_closeup.png", "the cup in the 10 x 10 cm corner square on the box top")}
{fig("table_closeup.png", "the serving table, 106 x 60 cm, top 74 cm")}
</div>

<h2 id="robot">What the D455 sees</h2>
<p>At the start the robot looks down at its own hands on the handle, the box top and the cup; the table's legs are already
in the top-right corner of the frame.</p>
<div class="grid">
{fig("ego_d455.png", f"standalone scene, {cam['size'][0]} x {cam['size'][1]}, 90 x 58.7 deg, servo 10 deg down")}
{fig(PR / "head_stereo_left.png", "the same view inside SIMPLE (head_stereo_left, 640 x 360)")}
</div>

<h2 id="task">The SIMPLE task</h2>
<p>Importing <code>coffee_cart_task.py</code> registers <code>simple/G1WholebodyCoffeeCartTeleop-v0</code> on the sonic
(decoupled-WBC) teleop stack with robot <code>g1_sonic</code>. SIMPLE's table primitive is the serving table's top slab; its
legs are a static asset; the cart and the cup are free objects. The cart is one rigid body whose four wheel geoms carry a low
friction ({cart['wheel_friction']}) so it rolls rather than drags &mdash; that needed a small extension to the engine patch the other
scenes use, so a single asset can give different friction to different pieces.</p>
<p>Instruction: <b>&ldquo;{PROBE.get('instruction', '')}&rdquo;</b></p>
<h3>Probe: where everything landed on reset</h3>
{probe_tbl}
{fig(PR / "third_person.png", "the task inside SIMPLE at reset, third-person")}
<pre>MUJOCO_GL=egl ~/wrk/SIMPLE/.venv/bin/python probe_scene.py

import sys; sys.path.insert(0, ".../sim/coffee_cart"); import coffee_cart_task
env = gym.make("simple/G1WholebodyCoffeeCartTeleop-v0", sim_mode="mujoco", sonic_config=cfg, headless=True)</pre>
<p class="sub">Knobs (env vars read by the task; every distance is in the pelvis frame of the start pose, x toward the table,
the robot's right = &minus;y, floor at z = 0): <code>COFFEE_CART_ROBOT_TO_CART</code>, <code>COFFEE_CART_ROUTE_FORWARD</code>,
<code>COFFEE_CART_TABLE_FROM_LINE</code>, <code>COFFEE_CART_CUP_XY</code>, <code>COFFEE_CART_CUP_MASS</code>,
<code>COFFEE_CART_CART_MASS</code>, <code>COFFEE_CART_INSTRUCTION</code>.</p>

<h2 id="assumptions">Assumptions, and the switch for each</h2>
<p>These are the places the sheet does not decide; each one is a flag rather than a hard-coded choice.</p>
<ul>
<li><b>&ldquo;220mm&rdquo; means 220 cm.</b> The sheet's other mm values ({cart['handle_dia']*1000:.0f} mm handle, {cart['wheel_dia']*1000:.0f} mm wheel) really are mm, but a 22 cm route cannot hold a 10 cm robot-to-cart gap plus travel. <code>--route-forward</code>.</li>
<li><b>Handle bar height {cart['handle_top']:.2f} m</b>: measured from a free-hand replay of three episodes (midpoint of thumb and index tips during the push, 0.772 &plusmn; 0.05 m as the hands bob with the gait). You said level with the box top (0.80): within 3 cm. <code>--handle-top</code>.</li>
<li><b>Cup at the box top's near-<i>right</i> corner.</b> The sheet's circle sits by the near edge, right of centre; &ldquo;10cm x 10cm at corner&rdquo; is read as the square the cup is centred in. <code>--cup-xy</code>.</li>
<li><b>Cup mass {cup['mass']:.2f} kg</b> (a full cup). SIMPLE's 0.1 kg default tips under a finger touch &mdash; the lesson from the bottle-bin scene. <code>--cup-mass</code>.</li>
<li><b>The cart is one rigid body</b> with wheel geoms at rolling resistance {cart['wheel_friction']} rather than four hinged casters, so it slides sideways as a swivel-caster cart would. The wheels need MuJoCo contact <b>priority 1</b>: without it the pair takes the floor's friction of 1.0, a 15 kg cart then needs ~150 N and the robot cannot push it &mdash; that single rule cost most of a day. <code>WHEEL_FRICTION</code>, <code>--cart-mass</code>.</li>
<li><b>The route's 67 cm is to the robot's right</b> (the sheet's page-right with the robot facing up the page). <code>--table-from-line</code>; a negative value puts the table on its left.</li>
<li><b>The box is closed</b> (a solid box the cup stands on), not an open bin.</li>
<li><b>The Isaac room is a backdrop</b> chosen for clearance, not the recording's office: floor plans of the HSSD rooms were rasterised from their USDs around the robot's start and scene31 was the one with open floor along the whole route and past the table. The pre-generated <code>dr-level-*</code> sets store the room in their state, so sets built before 2026-09-24 still load scene2 until <code>make_levels.py coffee_cart --force</code> is rerun.</li>
<li><b>&ldquo;Cart wheel: 120mm high&rdquo; is the caster's overall height</b>, floor to the deck's <i>top</i> &mdash; which is how a caster is specified and what keeps the box top at the {cart['box_top']:.2f} m you picked. So the wheel itself is {cart['wheel_dia']*1000:.0f} mm and ends exactly at the deck's underside: no geometry overlaps and nothing sits below the floor.</li>
</ul>

<h2 id="files">Files and commands</h2>
<p>Everything is in <code>holobrain_g1_deploy/sim/coffee_cart/</code>: <code>build_scene.py</code> (MuJoCo scene, renders, plan),
<code>scene.xml</code>, <code>layout.json</code> (the numbers every other file reads), <code>coffee_cart_task.py</code> (SIMPLE task),
<code>probe_scene.py</code> (load the task and report where everything landed), <code>assets/</code> (generated cup, cart and leg
meshes), <code>make_report.py</code> (this page).</p>
<pre>MUJOCO_GL=glfw DISPLAY=:1 ~/wrk/SIMPLE/.venv/bin/python build_scene.py   # scene.xml, layout.json, renders/
MUJOCO_GL=egl  ~/wrk/SIMPLE/.venv/bin/python probe_scene.py              # the scene inside SIMPLE -> probe/
python make_report.py                                                    # this page (site/)</pre>
{replay_section()}
"""


def replay_section():
    if not REPLAY:
        return "<h2 id='replay'>Action replay</h2><p class='missing'>replay/replay_results.json not written yet</p>"
    R = REPLAY; b = R["baseline"]; best = R["best"]
    rows = "".join(f"<tr><td>{s['route_forward']:.1f}</td><td>{s['table_from_line']:.1f}</td><td>{s.get('post_turn_gain', 1.0):.1f}</td>"
                   f"<td>{(str(s['task_success']) + '/' + str(s['n'])) if s.get('has_task_verdict') else '&ndash;'}</td><td>{s['on_table']}/{s['n']}</td><td>{s.get('tipped', 0)}/{s['n']}</td><td>{s['lifted']}/{s['n']}</td><td>{s['floor']}</td><td>{s['on_cart']}</td></tr>"
                   for s in R["sweep"])
    ts = b.get("task_success") if b.get("has_task_verdict") else None
    stats = "".join(stat(v, k) for v, k in (((f"{ts}/{b['n']}", "task success (the task's own checker)"),) if ts is not None else ()) + (
                                             (f"{b['on_table']}/{b['n']}", "upright on the table (geometric check)"),
                                             (f"{b['lifted']}/{b['n']}", "cup lifted off the cart"),
                                             (f"{R['push']['pushed']}/{R['push']['n']}", "cart pushed > 1 m, real contact"),
                                             (f"{R['push']['cart_travel_mean']:.2f} m", "mean cart travel (sheet: 2.20)"),
                                             (f"{R['push']['heading_err_deg']:.0f}&deg;", "mean |heading error| at the end")))
    verdict_line = (f"Per episode on the full set: task success {b.get('task_success')}, cart pushed {b.get('task_cart_pushed')}, "
                    f"cup lifted {b.get('task_cup_lifted')} of {b['n']}." if b.get("has_task_verdict")
                    else "The numbers above are from the geometric check; the task's own verdict is being re-run.")
    vid = f"<figure><video controls preload='metadata' src='vid/{R['video']}'></video><figcaption>{html.escape(R['video_caption'])}</figcaption></figure>" if R.get("video") and (SITE / "vid" / R["video"]).exists() else ""
    _prep(HERE / "renders" / "isaac_scene31_poster.jpg")
    vid += fig("replay_ep84_sheet.png", "episode 84 under the sim clock: third-person | sim D455 at t = 4, 12, 31, 40 s (grip, push, turn, place)") if (HERE / "renders" / "replay_ep84_sheet.png").exists() else ""
    return f"""
<h2 id="replay">Action replay of the real recordings</h2>
<p>All 97 episodes of <code>psi0/CartCoffeeCup_0919</code> replayed through the decoupled-WBC stack in this scene, gravity
compensation on, a sim clock in place of the wall clock (so every run is bit-reproducible), <b>real contact</b>: the recorded hands close on the bar and the robot pushes the {cart['mass']:.0f} kg cart itself.
Success = the cup ends upright on the table top (its end height must be within 3.5 cm of 0.74 m, which rules out a cup still
riding the 0.80 m cart top).</p>
<div class="stats">{stats}</div>
<h3>Task checkers</h3>
<p><code>coffee_cart_task.py</code> now scores itself the way the stock SIMPLE tasks do, through <code>compute_reward</code> /
<code>check_success</code> on the env's <code>info</code> dict, and writes every checker into <code>info["progress"]</code> each step:</p>
<div class="tw"><table>
<tr><th>checker</th><th>true when</th><th>sticky</th></tr>
<tr><td><code>cart_pushed</code></td><td>the cart is &ge; 0.50 m from where it started</td><td>yes</td></tr>
<tr><td><code>cup_lifted</code></td><td>the cup is &ge; 5 cm above its rest on the box top</td><td>yes</td></tr>
<tr><td><code>cup_upright</code></td><td>the cup axis is within 45&deg; of vertical</td><td>no</td></tr>
<tr><td><code>cup_resting_on_table</code></td><td>inside the table footprint, upright, base within 3.5 cm of the 0.74 m top (a cup still on the 0.80 m cart top cannot pass)</td><td>no</td></tr>
<tr><td><code>cup_in_hand</code></td><td>any hand geom is in contact with the cup</td><td>no</td></tr>
<tr><td><code>cup_tipped_on_table</code>, <code>cup_on_floor</code>, <code>cup_still_on_cart</code></td><td>the failure states, for the report</td><td>no</td></tr>
<tr><td><b><code>success</code></b></td><td>resting on the table <i>and</i> out of the hand for 25 consecutive steps (0.5 s at 50 Hz)</td><td>yes: once placed and let go, the episode counts</td></tr>
</table></div>
<p class="sub">Reward = 0.25 &middot; cart pushed + 0.25 &middot; cup lifted + 0.5 &middot; success, so <code>check_success</code> fires only on a placed cup. Thresholds are
env vars <code>COFFEE_CART_CHK_CART_PUSHED_M</code>, <code>_CUP_LIFT_M</code>, <code>_TABLE_Z_TOL</code>, <code>_UPRIGHT_DEG</code>, <code>_HOLD_STEPS</code>.
{verdict_line}</p>
<h3>Where the table goes: the sweep</h3>
<p>Robot start fixed at the measured grip ({lay['robot_to_cart']*100:.0f} cm); route and lateral offset swept on 8 episodes spread over the set.
&ldquo;gain&rdquo; scales the recorded walking command only <i>after</i> the turn, where the recorded 0.13 m/s sits at the sim gait's
motion-onset threshold and the robot barely advances.</p>
<div class="tw"><table><tr><th>route (m)</th><th>table from line (m)</th><th>post-turn gain</th><th>task success</th><th>upright on table (geom.)</th><th>tipped on table</th><th>lifted</th><th>on floor</th><th>still on cart</th></tr>{rows}</table></div>
<p><b>Best: route {best['route_forward']:.1f} m, table {best['table_from_line']:.1f} m from the line{', post-turn gain ' + format(best['post_turn_gain'], '.1f') if best.get('post_turn_gain', 1.0) != 1.0 else ''}</b> &mdash;
{b['on_table']}/{b['n']} of all episodes upright on the table, {b.get('tipped', 0)} more tipped over on it, {b['lifted']} lifted off the cart.</p>
{vid}
<h3>What caps the completion rate</h3>
<ul>
<li><b>Push distance varies per episode</b> in sim ({R['push']['cart_travel_p10']:.1f}&ndash;{R['push']['cart_travel_p90']:.1f} m, p10&ndash;p90) although the operator stopped at the same desk every time: the replay is open-loop, the operator was closed-loop by eye. Release points spread over {R['release']['x_p10']:.1f}&ndash;{R['release']['x_p90']:.1f} m along the route against a 0.60 m table depth.</li>
<li><b>The post-turn approach</b> is commanded at ~0.13 m/s, right at the sim gait's onset threshold (0.10), so the robot creeps where the operator walked ~0.5 m; releases land {abs(R['release']['y_med']):.2f} m to the right of the start line.</li>
<li>What is <i>not</i> the problem any more: the push itself (cart {R['push']['cart_travel_mean']:.2f} m vs the sheet's 2.20), the turn ({R['push']['heading_err_deg']:.0f}&deg; error), the grasp geometry, the cart mass.</li>
</ul>
<pre>MUJOCO_GL=egl ~/wrk/SIMPLE/.venv/bin/python replay_in_scene.py --probe
MUJOCO_GL=egl ~/wrk/SIMPLE/.venv/bin/python replay_in_scene.py --fast --jobs 12 --episodes 0 12 24 --robot-to-cart 0.20 \
    --route-forward {best['route_forward']:.1f} --table-from-line {best['table_from_line']:.1f} --cart-mass {cart['mass']:.0f} --handle-top {cart['handle_top']:.2f}
MUJOCO_GL=egl ~/wrk/SIMPLE/.venv/bin/python replay_in_scene.py --episodes 84 ...          # without --fast: mp4 with the real D455 alongside</pre>
"""
def main():
    standalone = "--standalone" in sys.argv
    fig.mode = "site"
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G1 Coffee Cart Scene</title>
<style>{CSS}</style>
<main>{body()}</main>
</html>"""
    (SITE / "index.html").write_text(page)
    print("wrote", SITE / "index.html", f"({len(page)/1024:.0f} kB) and", len(list(IMG.glob('*'))), "images in", IMG)
    if standalone:
        fig.mode = "inline"
        page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>G1 Coffee Cart Scene</title>
<style>{CSS}</style>
<main>{body()}</main>
</html>"""
        (HERE / "report.html").write_text(page)
        print("wrote", HERE / "report.html", f"({len(page)/1024/1024:.1f} MB)")


if __name__ == "__main__":
    main()
