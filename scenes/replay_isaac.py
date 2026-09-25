#!/usr/bin/env python
"""Replay a recorded real episode inside a kit's SIMPLE level-N scene with Isaac rendering, using the kit's own
replay_in_scene.py unchanged: the env is built in mujoco_isaac mode, reset to one scene of data/evals_scenes (its
distractors, table material, lighting), and the kit's video (third person | head camera | real head camera) then
carries the Isaac head camera in the middle panel.

    python sim/replay_isaac.py bowl_sink --episode 7 --nav-gain 1.5 --out sim/bowl_sink/replay/isaac_level0_ep7.mp4
    python sim/replay_isaac.py bottle_bin --session 2026-09-17-00-16-29-G1-sim --episode 6
    python sim/replay_isaac.py coffee_cart --episode 84
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV_IDS = {"bottle_bin": "G1WholebodyBottleBinTeleop-v0", "bowl_sink": "G1WholebodyBowlSinkTeleop-v0", "coffee_cart": "G1WholebodyCoffeeCartTeleop-v0"}
SESSIONS = {"bottle_bin": "2026-09-17-00-16-29-G1-sim", "bowl_sink": "psi0/BowlToSink_0918"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("kit", choices=sorted(ENV_IDS)); ap.add_argument("--episode", type=int, required=True)
ap.add_argument("--session", default=None); ap.add_argument("--level", type=int, default=0); ap.add_argument("--scene", type=int, default=0)
ap.add_argument("--nav-gain", type=float, default=1.0); ap.add_argument("--sets", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "evals_scenes"))
ap.add_argument("--out", default=None)
ap.add_argument("--sim-mode", default="mujoco_isaac", choices=("mujoco_isaac", "mujoco"), help="diagnostics: mujoco = no Isaac")
ap.add_argument("--hide-shell", action="store_true", help="third-person pass: hide the HSSD room walls/ceiling so the kit's own (outside-the-room) camera sees the scene")
ap.add_argument("--third-light", type=float, default=0.0, help="pass B: dome light intensity added for the third-person camera (0 = none)")
ap.add_argument("--no-third", action="store_true", help="skip the Isaac third-person camera (the kit video keeps its MuJoCo third-person panel)")
ap.add_argument("--ext-clock", action="store_true", help="also for coffee_cart: one external virtual clock for every module (its own shim covers only the agent + teleop policy)")
ap.add_argument("--driver", default=None, choices=("replay", "sweep"), help="bottle_bin: 'sweep' drives sweep_replay.py (the tool that produced the verified successes; own virtual clock, --video-dir); default sweep for bottle_bin, replay otherwise")
ap.add_argument("--no-state", action="store_true", help="diagnostics: plain reset (the kit scene as MuJoCo replays see it) instead of the level scene")
a = ap.parse_args()

set_dir = Path(a.sets) / ENV_IDS[a.kit] / f"dr-level-{a.level}"
meta = json.load(open(set_dir / "meta" / "scene_env.json"))
for k, v in meta.get("env", {}).items():
    os.environ.setdefault(k, v)                                   # the kit's level knobs (distractor count etc.)
os.environ.setdefault("MUJOCO_GL", "egl"); os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ["REPLAY_NAV_GAIN"] = str(a.nav_gain)
if a.ext_clock:
    os.environ["REPLAY_SIM_CLOCK"] = "0"                          # one clock only: the external one
rows = [json.loads(l) for l in open(set_dir / "meta" / "episodes.jsonl") if l.strip()]
STATE = json.loads(rows[a.scene]["environment_config"])
print(f"[isaac-replay] {a.kit} episode {a.episode} in level {a.level} scene {a.scene} ({meta['description']}); env {meta.get('env')}", flush=True)

sys.path.insert(0, str(HERE / a.kit))
import gymnasium
_orig_make = gymnasium.make


def _make(env_id, *args, **kw):
    kw["sim_mode"] = a.sim_mode
    env = _orig_make(env_id, *args, **kw)
    orig_reset, orig_step = env.reset, env.step

    def reset(*ra, **rk):
        if rk.get("options") is None and not a.no_state:
            rk["options"] = {"state_dict": STATE}
        out = orig_reset(*ra, **rk)
        if THIRD["cam"] is None and not a.no_third and a.sim_mode == "mujoco_isaac":
            _make_third_person_camera(env)
        return out

    def step(*sa, **sk):
        out = orig_step(*sa, **sk)
        _count_contacts(env)
        if CLOCK["clock"] is not None:
            CLOCK["clock"].tick()
        if THIRD["cam"] is not None:
            _capture_third_person(out[0] if isinstance(out, tuple) and out else None)
        return out
    env.reset, env.step = reset, step
    return env


gymnasium.make = _make

CONTACT = {"ticks": 0, "n": 0, "first": None, "names": {}, "ids": None}
FURNITURE = ("table", "counter", "sink_counter", "cart", "trash_bin", "cover_board", "table_legs")


def _count_contacts(env):
    """Ticks in which any robot geom touches the furniture (the SIMPLE table or the kit's counters / cart / bin)."""
    import mujoco
    se = env.unwrapped; robot = se.task.robot; m, d = robot.mjModel, robot.mjData
    if CONTACT["ids"] is None or CONTACT["ids"][0] is not m:
        furn, rob = set(), set()
        for b in range(m.nbody):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            root = b
            while m.body_parentid[root] != 0 and m.body_parentid[root] != root:
                root = m.body_parentid[root]
            rname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, root) or ""
            if any(name == f or name.startswith(f) for f in FURNITURE) or any(rname == f or rname.startswith(f) for f in FURNITURE):
                furn.add(b)
            elif "pelvis" in rname or "torso" in rname or rname.endswith("_link") or "pelvis" in name:
                rob.add(b)
        CONTACT["ids"] = (m, furn, rob)
    _, furn, rob = CONTACT["ids"]
    CONTACT["n"] += 1; hit = False
    for i in range(d.ncon):
        c = d.contact[i]; b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
        if (b1 in rob and b2 in furn) or (b2 in rob and b1 in furn):
            rb = b1 if b1 in rob else b2; fb = b2 if b1 in rob else b1
            key = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, rb), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, fb))
            CONTACT["names"][key] = CONTACT["names"].get(key, 0) + 1; hit = True
    if hit:
        CONTACT["ticks"] += 1
        if CONTACT["first"] is None:
            CONTACT["first"] = CONTACT["n"]; print(f"[isaac-replay] first robot-furniture contact at tick {CONTACT['n']}", flush=True)
    if CONTACT["n"] % 250 == 0:
        top = sorted(CONTACT["names"].items(), key=lambda kv: -kv[1])[:4]
        print(f"[isaac-replay] contacts: {CONTACT['ticks']} of {CONTACT['n']} ticks touching furniture; top pairs {top}", flush=True)


THIRD = {"cam": None, "n": 0, "k": 0, "path": None}
THIRD_IN_ROOM = {}   # the kits' own third-person pose (the room shell is hidden for this pass instead of moving the camera)   # (distance factor, elevation) per kit, chosen from third_pose_probe.py


def _third_params():
    """The kit's third-person MuJoCo free camera (lookat / distance / azimuth / elevation)."""
    mod = sys.modules.get("replay_in_scene") or sys.modules.get("sweep_replay")
    if mod is None or not hasattr(mod, "THIRD"):
        import importlib
        mod = importlib.import_module("replay_in_scene")
    return dict(mod.THIRD)


ROOM_SHELL_PATTERNS = ("walls", "ceilings", "openings")      # prim-name substrings hidden by --hide-shell (see isaac_smoke/room_prims_probe.py)


def _hide_room_shell():
    """Hide the HSSD room's walls / ceilings / openings so the kit's own (outside-the-room) camera sees the scene. The HSSD
    floor is a child of the `walls` group, so the group prims are not hidden as a whole: their children are hidden one by
    one and flat ones (world bbox thinner than 5 cm = floor slabs) are kept (2026-09-24: the third-person views sat on black)."""
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdGeom, Usd
    st = get_current_stage(); n = 0; kept = []
    bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    for p in st.Traverse():
        path = str(p.GetPath())
        if not path.startswith("/World/scene"):
            continue
        name = p.GetName().lower()
        if not (any(k in name for k in ROOM_SHELL_PATTERNS) and p.IsA(UsdGeom.Imageable)):
            continue
        kids = [c for c in p.GetChildren() if c.IsA(UsdGeom.Imageable)]
        if not kids:
            UsdGeom.Imageable(p).MakeInvisible(); n += 1; continue
        for c in kids:
            try:
                r = bb.ComputeWorldBound(c).ComputeAlignedRange()
                flat = (r.GetMax()[2] - r.GetMin()[2]) < 0.05 and r.GetMax()[2] < 0.1     # a floor slab: thin AND at ground level (ceiling slabs are thin too)
            except Exception:
                flat = False
            if flat:
                kept.append(c.GetName())
            else:
                UsdGeom.Imageable(c).MakeInvisible(); n += 1
    print(f"[isaac-replay] hid {n} room shell prims ({ROOM_SHELL_PATTERNS}); kept flat (floor) prims {kept}", flush=True)


def _add_third_person_light(intensity: float):
    """Pass B only: a dome light so the third-person camera (up to 7 m away, outside the SIMPLE table lights) sees the
    whole route. The head panel of the final video comes from pass A, which is unaffected."""
    if intensity <= 0:
        return
    from omni.isaac.core.utils.stage import get_current_stage
    from pxr import UsdLux, Sdf
    st = get_current_stage(); lp = "/World/third_person_light"
    if st.GetPrimAtPath(lp):
        st.RemovePrim(lp)
    UsdLux.DomeLight.Define(st, Sdf.Path(lp)).CreateIntensityAttr(float(intensity))
    print(f"[isaac-replay] third-person dome light {intensity:.0f}", flush=True)


def _make_third_person_camera(env):
    """An Isaac camera at the pose of the kit's MuJoCo third-person camera (same look-at, distance, angles, 45 deg fovy)."""
    import mujoco, numpy as np, transforms3d as t3d
    from omni.isaac.sensor import Camera
    se = env.unwrapped; robot = se.task.robot; m, d = robot.mjModel, robot.mjData
    P = _third_params()
    fac, elev = THIRD_IN_ROOM.get(a.kit, (1.0, None))          # the kits' MuJoCo camera stands outside the HSSD room; pull it inside
    cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = P["lookat"]; cam.distance = P["distance"] * fac; cam.azimuth = P["azimuth"]; cam.elevation = P["elevation"] if elev is None else elev
    scn = mujoco.MjvScene(m, maxgeom=2000)
    mujoco.mjv_updateScene(m, d, mujoco.MjvOption(), None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
    c0, c1 = scn.camera[0], scn.camera[1]
    pos = (np.asarray(c0.pos) + np.asarray(c1.pos)) / 2.0; fwd = np.asarray(c0.forward, dtype=float); up = np.asarray(c0.up, dtype=float)
    fwd /= np.linalg.norm(fwd); up -= fwd * np.dot(up, fwd); up /= np.linalg.norm(up); right = np.cross(fwd, up)
    R = np.stack([right, up, -fwd], axis=1)                       # USD camera: +x right, +y up, looks along -z
    quat = t3d.quaternions.mat2quat(R)
    W, H = 640, 480; fovy = float(m.vis.global_.fovy) if m.vis.global_.fovy > 0 else 45.0
    hfov = 2 * np.degrees(np.arctan(np.tan(np.radians(fovy) / 2) * W / H))
    c = Camera(prim_path="/World/third_person_cam", name="third_person_cam", resolution=(W, H))
    c.initialize()
    c.set_world_pose(pos, quat, camera_axes="usd")
    f = 10.0; c.set_focal_length(f); c.set_horizontal_aperture(2 * f * np.tan(np.radians(hfov) / 2)); c.set_vertical_aperture(2 * f * np.tan(np.radians(fovy) / 2))
    c.set_clipping_range(0.05, 50.0)
    THIRD["cam"] = c; THIRD["pos"] = pos.tolist(); THIRD["quat"] = quat.tolist()
    if a.hide_shell:
        _hide_room_shell()
    _add_third_person_light(a.third_light)
    print(f"[isaac-replay] third-person Isaac camera at {np.round(pos, 2).tolist()} fovy {fovy:.0f} deg (hfov {hfov:.0f})", flush=True)


def _capture_third_person(obs=None):
    """Save the Isaac third-person frame AND the head-camera frame of the same tick as JPEGs (cv2: PIL's encoders are broken
    inside the Isaac process; the process dies in Isaac's shutdown, so nothing can be post-processed here). Every tick:
    compose_third.py aligns the kit video (written every 2nd tick once the replay starts) structurally: kit frame k = tick S + 2k, S = M - 2N."""
    import numpy as np, traceback
    THIRD["n"] += 1
    try:
        rgba = THIRD["cam"].get_rgba()
        if rgba is None or getattr(rgba, "ndim", 0) != 3 or rgba.shape[0] == 0:
            fr = THIRD["cam"].get_current_frame(); rgba = fr.get("rgba") if isinstance(fr, dict) else None
        if rgba is None or getattr(rgba, "ndim", 0) != 3 or rgba.shape[0] == 0:
            if THIRD["n"] <= 10: print(f"[isaac-replay] third-person tick {THIRD['n']}: no frame", flush=True)
            return
        if THIRD["path"] is None:
            THIRD["path"] = HERE / a.kit / "replay" / f"third_{a.kit}_ep{a.episode}"
            THIRD["path"].mkdir(parents=True, exist_ok=True)
            for old in THIRD["path"].glob("*.jpg"):
                old.unlink()
            THIRD["k"] = 0
        import cv2
        img = np.array(rgba, dtype=np.uint8)[..., :3].copy()
        cv2.imwrite(str(THIRD["path"] / f"f{THIRD['k']:05d}.jpg"), np.ascontiguousarray(img[..., ::-1]), [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        head = None
        if isinstance(obs, dict) and "head_stereo_left" in obs:
            head = np.asarray(obs["head_stereo_left"])
            cv2.imwrite(str(THIRD["path"] / f"h{THIRD['k']:05d}.jpg"), np.ascontiguousarray(head[..., ::-1]), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        THIRD["k"] += 1
        if THIRD["n"] <= 6 or THIRD["n"] % 500 == 0:
            print(f"[isaac-replay] third-person tick {THIRD['n']}: saved frame {THIRD['k']} {img.shape} head {None if head is None else head.shape}", flush=True)
    except BaseException as exc:  # noqa: BLE001
        print(f"[isaac-replay] third-person capture failed at tick {THIRD['n']}: {exc!r}", flush=True)
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()


def _close_third_person():
    if THIRD["path"] is None:
        return None
    print(f"[isaac-replay] third-person frames: {THIRD.get('k', 0)} -> {THIRD['path']}", flush=True)
    return THIRD["path"]


def compose_final(kit_video: Path, third_video: Path, out: Path, poster: Path, level: int):
    """Final video: Isaac third person | Isaac head camera (the kit video's middle panel) | real head camera (its right panel)."""
    import av, numpy as np
    from PIL import Image, ImageDraw, ImageFont
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception: font = ImageFont.load_default()
    thirds = []
    with av.open(str(third_video)) as tc:
        for fr in tc.decode(video=0): thirds.append(fr.to_image().convert("RGB"))
    with av.open(str(kit_video)) as ic, av.open(str(out), "w") as oc:
        vi = ic.streams.video[0]; n_kit = vi.frames or 0
        vs = oc.add_stream("libx264", rate=vi.average_rate or 25); vs.pix_fmt = "yuv420p"; vs.options = {"crf": "23", "preset": "veryfast", "movflags": "+faststart"}
        vs.width, vs.height = vi.codec_context.width, vi.codec_context.height
        frames = [fr.to_image().convert("RGB") for fr in ic.decode(video=0)]
        N, M = len(frames), len(thirds)
        for k, im in enumerate(frames):
            W = im.width // 3; H = im.height
            if M:
                t = thirds[min(M - 1, round(k * (M - 1) / max(N - 1, 1)))].resize((W, H)); im.paste(t, (0, 0))
            d = ImageDraw.Draw(im)
            d.rectangle((4, 30, 300, 58), fill=(0, 0, 0)); d.text((10, 36), f"Isaac third person (level {level})", fill=(255, 235, 80), font=font)
            d.rectangle((W + 4, 30, W + 260, 58), fill=(0, 0, 0)); d.text((W + 10, 36), f"Isaac D455 (sim, level {level})", fill=(255, 235, 80), font=font)
            if k == 0: im.save(poster, quality=85)
            for pkt in vs.encode(av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")): oc.mux(pkt)
        for pkt in vs.encode(): oc.mux(pkt)
    print(f"[isaac-replay] composed {N} frames (third-person stream {M}) -> {out}", flush=True)

CLOCK = {"clock": None}

if a.ext_clock or (a.kit in ("bottle_bin", "bowl_sink") and (a.driver or ("sweep" if a.kit == "bottle_bin" else "replay")) == "replay"):   # coffee_cart has its own sim clock (fast=True): leave it alone; the other two pace on the wall clock
    # these two replays pace on the wall clock; under Isaac's slow steps the controller must run on a virtual 50 Hz clock
    # instead (the kits' own --fast mode does that but records no video). Install bowl_sink's VirtualClock once the agent
    # (and decoupled_wbc) exist, and tick it after every env.step.
    import importlib, importlib.util
    import simple.agents.pico_decoupled_agent as _pda
    _Agent = _pda.PicoDecoupledAgent

    class ClockedAgent(_Agent):
        def __init__(self, *aa, **kk):
            super().__init__(*aa, **kk)
            VC = getattr(sys.modules.get("replay_in_scene"), "VirtualClock", None)
            if VC is None:                                        # bottle_bin has no VirtualClock: borrow bowl_sink's
                spec = importlib.util.spec_from_file_location("bowl_sink_replay", HERE / "bowl_sink" / "replay_in_scene.py")
                m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); VC = m.VirtualClock
            CLOCK["clock"] = VC(); patched = CLOCK["clock"].install()
            R = sys.modules.get("replay_in_scene")
            if R is not None and getattr(R, "time", None) is not None:
                R.time = CLOCK["clock"]; patched.append("replay_in_scene.time")     # the kit's settle (2 s) and pacing now count ticks, not wall time
            print(f"[isaac-replay] virtual clock installed in {len(patched)} modules", flush=True)
    _pda.PicoDecoupledAgent = ClockedAgent

import importlib
driver = a.driver or ("sweep" if a.kit == "bottle_bin" else "replay")
if driver == "sweep":
    # sweep_replay.py: virtual clock built in, records the same 3-panel video with --video-dir; run its main() with our
    # patched gymnasium.make (Isaac + level scene) and the sweep's verified scene settings (feet 0.03, bottle 0.5 kg)
    SW = importlib.import_module("sweep_replay")
    vd = Path(a.out).parent if a.out else HERE / a.kit / "replay"
    tag = f"isaac_level{a.level}s{a.scene}" + ("" if a.sim_mode == "mujoco_isaac" else "_mujoco") + ("_nostate" if a.no_state else "")
    sys.argv = ["sweep_replay.py", "--session", a.session or SESSIONS[a.kit], "--tag", tag, "--episodes", str(a.episode),
                "--video-dir", str(vd), "--robot-to-edge", "0.03", "--bottle-mass", "0.5"]
    print(f"[isaac-replay] driver sweep_replay: {' '.join(sys.argv[1:])}", flush=True)
    try:
        SW.main()
    except SystemExit:
        pass
    kit_video = vd / f"replay_{a.session or SESSIONS[a.kit]}_ep{a.episode:03d}_{tag}.mp4"
    if not kit_video.exists():
        kit_video = next(iter(sorted(vd.glob(f"replay_*_ep{a.episode:03d}_{tag}.mp4")) + sorted(vd.glob(f"replay_*_ep{a.episode}_{tag}.mp4"))), kit_video)
    _close_third_person()
    print(f"[isaac-replay] done -> {kit_video}", flush=True)
    sys.exit(0)
R = importlib.import_module("replay_in_scene")
suffix = ("" if a.sim_mode == "mujoco_isaac" else "_mujoco") + ("_nostate" if a.no_state else "") + ("_extclock" if a.ext_clock else "")
out = Path(a.out) if a.out else HERE / a.kit / "replay" / f"isaac_level{a.level}_ep{a.episode}{suffix}.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
if a.kit == "coffee_cart":
    res = R.replay_episode(a.episode, out, fast=True, video=True)
else:
    if hasattr(R, "NAV_GAIN"):
        R.NAV_GAIN = a.nav_gain
    res = R.replay_episode(a.session or SESSIONS[a.kit], a.episode, out)
_close_third_person()
print(f"[isaac-replay] done -> {out}", flush=True)
try:
    print("[isaac-replay] summary:", json.dumps(res, default=str)[:600], flush=True)
except Exception:
    pass
