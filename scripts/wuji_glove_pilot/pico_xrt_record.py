#!/usr/bin/env python
"""Record what the PICO sends through the XRoboToolkit SDK, and draw the human skeleton from it (2026-11-04).

record   polls the SDK and keeps every sample in which something changed: the headset pose (7), the 24-joint body pose
         (24 x 7: x y z qx qy qz qw, SMPL joint order), both hands (2 x 26 x 7, OpenXR joint order) with their quality
         flags, and the controller sample (sticks, triggers, grips, buttons). One sample (--one) or N seconds.
skeleton draws the body tree and the two hand trees (hands zoomed) from one frame of a recording, or every k-th frame to an
         MP4, and checks that the body's wrists and the hands' wrist joints agree (same frame, same units).

    .venv/bin/python scripts/wuji_glove_pilot/pico_xrt_record.py live [--record]              # real-time skeleton window (1 body / 2 hands)
    .venv/bin/python scripts/wuji_glove_pilot/pico_xrt_record.py record --seconds 10            # -> data/wuji_glove_pilot/xrt_<time>/xrt.npz
    .venv/bin/python scripts/wuji_glove_pilot/pico_xrt_record.py record --one                   # a single sample
    .venv/bin/python scripts/wuji_glove_pilot/pico_xrt_record.py skeleton data/wuji_glove_pilot/xrt_<time>/xrt.npz            # middle frame -> PNG
    .venv/bin/python scripts/wuji_glove_pilot/pico_xrt_record.py skeleton <npz> --video         # all frames -> MP4
    .venv/bin/python scripts/wuji_glove_pilot/pico_xrt_record.py --selftest                     # synthetic recording -> PNG, no headset

The headset must be on and connected to the PC service (it drops off a few minutes after being taken off); body tracking
needs the XRoboToolkit app's full-body mode, hand tracking needs the controllers put down. Frames: the SDK's own world
frame, metres; the body and the hands share it (the wrist check below says by how much they agree).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "wuji_glove_pilot"

# SMPL 24-joint order (the publisher's body stream) and its kinematic tree
BODY_NAMES = ["pelvis", "L_hip", "R_hip", "spine1", "L_knee", "R_knee", "spine2", "L_ankle", "R_ankle", "spine3", "L_foot", "R_foot",
              "neck", "L_collar", "R_collar", "head", "L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist", "L_hand", "R_hand"]
BODY_PARENT = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
L_WRIST, R_WRIST = 20, 21
# OpenXR XR_HAND_JOINT order (26) and tree: wrist is the root, palm hangs off it, each finger = metacarpal .. tip off the wrist
HAND_NAMES = ["palm", "wrist"] + [f"thumb_{j}" for j in ("metacarpal", "proximal", "distal", "tip")] + [
    f"{f}_{j}" for f in ("index", "middle", "ring", "little") for j in ("metacarpal", "proximal", "intermediate", "distal", "tip")]
HAND_PARENT = [1, -1, 1, 2, 3, 4] + sum(([1, 6 + 5 * k, 7 + 5 * k, 8 + 5 * k, 9 + 5 * k] for k in range(4)), [])
HAND_WRIST = 1
CTRL_NAMES = ["left_axis_x", "left_axis_y", "right_axis_x", "right_axis_y", "left_trigger", "right_trigger", "left_grip", "right_grip",
              "A", "B", "X", "Y", "left_menu", "right_menu", "left_axis_click", "right_axis_click"]
HAND_MIN_SPREAD_M = 5e-3                              # a hand whose 26 joints lie within this of each other is "no hand data"


def hand_present(h: np.ndarray) -> np.ndarray:
    """(..., 26, 7) -> bool per hand: while it is not tracking a hand the SDK repeats one pose for all 26 joints (zeros, or
    the last pose), so a real hand is one whose joints are spread out."""
    p = np.asarray(h)[..., :, :3]
    return (p.max(axis=-2) - p.min(axis=-2)).max(axis=-1) > HAND_MIN_SPREAD_M


def upright(p: np.ndarray, y_up: bool) -> np.ndarray:
    """Points (..., 3) in the SDK frame -> plot frame with z up (the SDK's world is y-up: x, y, z -> x, z, y)."""
    return p[..., [0, 2, 1]] if y_up else p


# ---------------------------------------------------------------------------------------------------------------- record
def read_sample(x) -> dict:
    """Everything the SDK exposes, one poll."""
    def f(name, default):
        try:
            return getattr(x, name)()
        except Exception:                                 # noqa: BLE001  (a function missing in this SDK build)
            return default
    ctrl = [*f("get_left_axis", (0.0, 0.0)), *f("get_right_axis", (0.0, 0.0)), f("get_left_trigger", 0.0), f("get_right_trigger", 0.0),
            f("get_left_grip", 0.0), f("get_right_grip", 0.0), f("get_A_button", 0), f("get_B_button", 0), f("get_X_button", 0), f("get_Y_button", 0),
            f("get_left_menu_button", 0), f("get_right_menu_button", 0), f("get_left_axis_click", 0), f("get_right_axis_click", 0)]
    body_avail = bool(f("is_body_data_available", False))
    return dict(
        head=np.asarray(f("get_headset_pose", [0.0] * 7), dtype=np.float64).reshape(7),
        body=np.asarray(f("get_body_joints_pose", [[0.0] * 7] * 24), dtype=np.float64).reshape(24, 7) if body_avail else np.zeros((24, 7)),
        body_avail=body_avail, body_stamp_ns=int(f("get_body_timestamp_ns", 0)), stamp_ns=int(f("get_time_stamp_ns", 0)),
        hands=np.stack([np.asarray(f("get_left_hand_tracking_state", [[0.0] * 7] * 26), dtype=np.float64).reshape(26, 7),
                        np.asarray(f("get_right_hand_tracking_state", [[0.0] * 7] * 26), dtype=np.float64).reshape(26, 7)]),
        hand_active=np.array([int(f("get_left_hand_is_active", 0)), int(f("get_right_hand_is_active", 0))]),
        ctrl=np.asarray(ctrl, dtype=np.float64))


def record(a) -> Path:
    import xrobotoolkit_sdk as x
    out = Path(a.out) if a.out else OUT_DIR / time.strftime("xrt_%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    x.init()
    rows, last, t0 = [], None, time.monotonic()
    seconds = 0.0 if a.one else a.seconds
    waited_msg = 0.0
    try:
        while True:
            now = time.monotonic() - t0
            s = read_sample(x)
            live = np.any(s["head"]) or s["body_avail"] or bool(hand_present(s["hands"]).any())
            if not live:                                  # nothing from the headset yet (asleep / not connected / hands out of view)
                if now - waited_msg >= 5.0:
                    print(f"   waiting for the headset ({now:.0f} s): no pose, no body, no hands from the PC service", flush=True); waited_msg = now
                if now > a.wait:
                    sys.exit(f"[xrt] nothing from the headset in {a.wait:.0f} s: is it on, connected to the PC service, hands in view?")
                time.sleep(0.02); continue
            key = (s["stamp_ns"], s["body_stamp_ns"], s["head"].tobytes(), s["hands"].tobytes())
            if key != last:                               # keep every poll in which something changed
                last = key
                rows.append(dict(s, t=now, wall=time.time()))
                if a.one:
                    break
                if len(rows) % 100 == 0:
                    print(f"   {now:5.1f} s: {len(rows)} samples, body {int(s['body_avail'])}, hands active {s['hand_active'].tolist()}", flush=True)
            if seconds and now - (rows[0]["t"] if rows else now) >= seconds:
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        pass
    finally:
        x.close()                                         # the SDK thread aborts the process if left open at exit
    if not rows:
        sys.exit("[xrt] no samples")
    R = {k: np.array([r[k] for r in rows]) for k in ("t", "wall", "head", "body", "body_avail", "body_stamp_ns", "stamp_ns", "hands", "hand_active", "ctrl")}
    np.savez_compressed(out / "xrt.npz", **R, body_names=np.array(BODY_NAMES), hand_names=np.array(HAND_NAMES), ctrl_names=np.array(CTRL_NAMES))
    span = float(R["t"][-1] - R["t"][0]) if len(rows) > 1 else 0.0
    body_rate = len({b.tobytes() for b in R["body"][R["body_avail"]]}) / span if span > 0 else 0.0     # get_body_timestamp_ns() is always 0
    hand_rate = [sum(1 for i in range(1, len(rows)) if not np.array_equal(R["hands"][i, k], R["hands"][i - 1, k])) / span if span > 0 else 0.0 for k in range(2)]
    meta = dict(samples=len(rows), seconds=round(span, 2), body_frames_hz=round(body_rate, 1), hand_updates_hz=[round(h, 1) for h in hand_rate],
                hand_active_fraction=[round(float(R["hand_active"][:, k].mean()), 2) for k in range(2)], body_available_fraction=round(float(R["body_avail"].mean()), 2),
                layout=dict(head="x y z qx qy qz qw", body="24 x 7 SMPL order, see body_names", hands="2 (left, right) x 26 x 7 OpenXR order, see hand_names",
                            ctrl=CTRL_NAMES, t="seconds since the first poll (monotonic)", wall="time.time()"), made=time.strftime("%Y-%m-%d %H:%M:%S"))
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[xrt] {len(rows)} samples over {span:.1f} s -> {out}/xrt.npz; body {body_rate:.0f} Hz ({meta['body_available_fraction'] * 100:.0f} % available), "
          f"hands {hand_rate[0]:.0f} / {hand_rate[1]:.0f} Hz (active {meta['hand_active_fraction']})", flush=True)
    return out


# -------------------------------------------------------------------------------------------------------------- skeleton
def _bones(parent):
    return [(i, p) for i, p in enumerate(parent) if p >= 0]


def _set_equal(ax, pts: np.ndarray, pad: float = 0.05) -> None:
    c, r = pts.mean(axis=0), max(float(np.ptp(pts, axis=0).max()) / 2, 1e-3) + pad
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)


def is_y_up(R: dict) -> bool:
    """The SDK's world frame is y-up (Unity); decided from the body (head above the pelvis) when there is one, else the headset height."""
    ok = R["body_avail"]
    if ok.any():
        d = (R["body"][ok][:, 15, :3] - R["body"][ok][:, 0, :3]).mean(axis=0)
        return abs(d[1]) > abs(d[2])
    h = R["head"][:, :3].mean(axis=0)
    return abs(h[1]) > abs(h[2])


def draw_frame(R: dict, i: int, path: Path, title: str = "") -> dict:
    """Body + both hands of frame i -> PNG (drawn upright); returns the wrist agreement (mm) per side."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yup = is_y_up(R)
    body, hands, active = upright(R["body"][i, :, :3], yup), upright(R["hands"][i, :, :, :3], yup), R["hand_active"][i]
    head = upright(R["head"][i, :3], yup)
    present = hand_present(R["hands"][i])
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    if R["body_avail"][i]:
        for c, p in _bones(BODY_PARENT):
            ax.plot(*zip(body[c], body[p]), c="tab:blue", lw=2)
        ax.scatter(*body.T, s=12, c="tab:blue")
        for j in (L_WRIST, R_WRIST, 15):
            ax.text(*body[j], BODY_NAMES[j], fontsize=7)
        _set_equal(ax, np.vstack([body, head[None]]))
    else:
        ax.text2D(0.1, 0.5, "no body data in this frame", transform=ax.transAxes)
    ax.scatter(*head, s=40, c="k", marker="^"); ax.set_title("body (24 SMPL joints) + headset" + (", SDK frame y-up, drawn z-up" if yup else ""))
    agree = {}
    for k, (side, col) in enumerate((("left", "tab:green"), ("right", "tab:red"))):
        axh = fig.add_subplot(1, 3, 2 + k, projection="3d")
        h = hands[k]
        if present[k]:
            for c, p in _bones(HAND_PARENT):
                axh.plot(*zip(h[c], h[p]), c=col, lw=2)
            axh.scatter(*h.T, s=10, c=col)
            for j in (1, 5, 10, 25):
                axh.text(*h[j], HAND_NAMES[j], fontsize=6)
            _set_equal(axh, h, pad=0.01)
            if R["body_avail"][i]:
                bw = body[L_WRIST if side == "left" else R_WRIST]
                agree[side] = float(np.linalg.norm(bw - h[HAND_WRIST]) * 1000)
                axh.scatter(*bw, s=50, c="tab:blue", marker="x")
        else:
            axh.text2D(0.1, 0.5, "no hand data in this frame", transform=axh.transAxes)
        axh.set_title(f"{side} hand (26 joints), quality {int(active[k])}" + (f"\nbody wrist (x) {agree[side]:.0f} mm from the hand wrist" if side in agree else ""), fontsize=10)
    fig.suptitle(title or f"frame {i} / {len(R['t'])}  t = {R['t'][i]:.2f} s")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)
    return agree


def skeleton(a) -> None:
    R = dict(np.load(a.npz, allow_pickle=True))
    n = len(R["t"]); out = Path(a.npz).parent
    if a.video:
        import imageio.v2 as iio
        step = max(1, int(round((R["t"][-1] - R["t"][0]) / n / (1.0 / a.fps)))) if n > 1 else 1
        frames = list(range(0, n, step)); tmp = out / "_frames"; tmp.mkdir(exist_ok=True)
        w = iio.get_writer(str(out / "skeleton.mp4"), fps=a.fps, codec="libx264", quality=7, macro_block_size=8)
        for i in frames:
            draw_frame(R, i, tmp / "f.png"); w.append_data(iio.imread(tmp / "f.png"))
        w.close(); (tmp / "f.png").unlink(); tmp.rmdir()
        print(f"[xrt] {len(frames)} frames at {a.fps} fps -> {out}/skeleton.mp4")
    i = n // 2 if a.frame is None else a.frame
    agree = draw_frame(R, i, out / f"skeleton_{i}.png")
    # stats: rates, bone lengths (are these real metres?), body/hand frame agreement over the recording
    body_ok = R["body_avail"]; yup = is_y_up(R)
    if body_ok.any():
        b = R["body"][body_ok][:, :, :3]
        arm = np.linalg.norm(b[:, 16] - b[:, 18], axis=1).mean() + np.linalg.norm(b[:, 18] - b[:, 20], axis=1).mean()
        up = (b[:, 15, 1 if yup else 2] - b[:, 0, 1 if yup else 2]).mean()
        print(f"[xrt] body: {body_ok.sum()} frames with data; left upper arm + forearm {arm * 100:.0f} cm; head {up * 100:.0f} cm above the pelvis; the SDK frame is {'y' if yup else 'z'}-up")
    for k, side in enumerate(("left", "right")):
        hk = R["hands"][:, k, :, :3]; ok = hand_present(R["hands"][:, k])
        if ok.any():
            h = hk[ok]; mid = (np.linalg.norm(h[:, 12] - h[:, 11], axis=1) + np.linalg.norm(h[:, 13] - h[:, 12], axis=1) + np.linalg.norm(h[:, 14] - h[:, 13], axis=1) + np.linalg.norm(h[:, 15] - h[:, 14], axis=1)).mean()
            line = f"[xrt] {side} hand: {ok.sum()} frames with data, quality high {R['hand_active'][ok, k].mean() * 100:.0f} %; middle finger metacarpal+phalanges {mid * 100:.1f} cm"
            both = ok & body_ok
            if both.any():
                dw = np.linalg.norm(R["body"][both][:, L_WRIST if side == "left" else R_WRIST, :3] - hk[both][:, HAND_WRIST], axis=1) * 1000
                line += f"; body wrist vs hand wrist {np.median(dw):.0f} mm median ({dw.min():.0f}-{dw.max():.0f})"
            print(line)
        else:
            print(f"[xrt] {side} hand: no hand data in any frame (hand tracking off: controllers held, or the hand out of the headset's view)")
    print(f"[xrt] drew frame {i} -> {out}/skeleton_{i}.png" + (f"; wrist agreement {agree}" if agree else ""))


# ------------------------------------------------------------------------------------------------------------------ live
LIVE_XML = """<mujoco><visual><headlight ambient=".5 .5 .5" diffuse=".6 .6 .6"/></visual>
<asset><texture name="sky" type="skybox" builtin="gradient" rgb1=".93 .94 .95" rgb2=".72 .75 .78" width="256" height="1536"/></asset>
<worldbody><light pos="0 0 3" dir="0 0 -1" directional="true"/></worldbody></mujoco>"""
BODY_R, HAND_R, JOINT_R = 0.02, 0.004, 0.012
COL = dict(body=(0.2, 0.45, 0.85, 1.0), left=(0.1, 0.65, 0.2, 1.0), right=(0.85, 0.2, 0.15, 1.0), head=(0.1, 0.1, 0.1, 1.0), floor=(0.8, 0.8, 0.8, 0.5))


def _capsule(scn, a, b, r, rgba) -> None:
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, r, np.asarray(a, np.float64), np.asarray(b, np.float64))
    scn.ngeom += 1


def _sphere(scn, p, r, rgba) -> None:
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([r, 0, 0], np.float64), np.asarray(p, np.float64), np.eye(3).ravel(), np.array(rgba, np.float32))
    scn.ngeom += 1


def _box(scn, centre, half, rgba) -> None:
    import mujoco
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_BOX, np.asarray(half, np.float64), np.asarray(centre, np.float64), np.eye(3).ravel(), np.array(rgba, np.float32))
    scn.ngeom += 1


def live(a) -> None:
    """Real-time skeleton: the SDK polled at up to 60 Hz, body + hands drawn as capsules in a MuJoCo viewer (mouse orbits /
    zooms; keys: 1 = body view, 2 = hands view, 0 = free camera). --record keeps every changed sample and saves it at exit."""
    import mujoco
    from mujoco import viewer as mjv
    import xrobotoolkit_sdk as x
    m = mujoco.MjModel.from_xml_string(LIVE_XML); d = mujoco.MjData(m)
    mode = {"view": 1, "set": True}

    def on_key(key):
        if key in (ord("1"), ord("2"), ord("0")):
            mode["view"], mode["set"] = int(chr(key)), True

    x.init()
    print("[xrt live] window open: mouse orbits / zooms; keys 1 = body view, 2 = hands view, 0 = leave the camera alone. "
          "Headset on, controllers down for the hands. Close the window to stop.", flush=True)
    rows, last_key, t0, yup = [], None, time.monotonic(), None
    stats = dict(polls=0, avail=0, body=0, hands=[0, 0], last_body_stamp=None, last_hands=[None, None], printed=t0)
    try:
        with mjv.launch_passive(m, d, key_callback=on_key, show_left_ui=False, show_right_ui=False) as v:
            while v.is_running():
                tick = time.monotonic()
                s = read_sample(x); stats["polls"] += 1
                present = hand_present(s["hands"])
                if s["body_avail"]:                                                   # get_body_timestamp_ns() is always 0: count pose changes
                    stats["avail"] += 1
                    if stats["last_body_stamp"] is None or not np.array_equal(s["body"], stats["last_body_stamp"]):
                        stats["body"] += 1; stats["last_body_stamp"] = s["body"].copy()
                for k in range(2):
                    if present[k] and (stats["last_hands"][k] is None or not np.array_equal(s["hands"][k], stats["last_hands"][k])):
                        stats["hands"][k] += 1; stats["last_hands"][k] = s["hands"][k].copy()
                if yup is None and (s["body_avail"] or np.any(s["head"])):          # decide the up axis once, from the first body (else head) sample
                    R1 = dict(body_avail=np.array([s["body_avail"]]), body=s["body"][None], head=s["head"][None]); yup = is_y_up(R1)
                if a.record:
                    key = (s["stamp_ns"], s["body_stamp_ns"], s["head"].tobytes(), s["hands"].tobytes())
                    if key != last_key:
                        last_key = key; rows.append(dict(s, t=tick - t0, wall=time.time()))
                body = upright(s["body"][:, :3], bool(yup)); hands = upright(s["hands"][:, :, :3], bool(yup)); head = upright(s["head"][:3], bool(yup))
                with v.lock():
                    scn = v.user_scn; scn.ngeom = 0
                    if s["body_avail"]:
                        for c, p in _bones(BODY_PARENT):
                            _capsule(scn, body[c], body[p], BODY_R, COL["body"])
                        for j in range(24):
                            _sphere(scn, body[j], JOINT_R, COL["body"])
                        floor = min(body[10, 2], body[11, 2]) - 0.03
                        _box(scn, [body[0, 0], body[0, 1], floor], [1.0, 1.0, 0.005], COL["floor"])
                    if np.any(head):
                        _sphere(scn, head, 0.08, COL["head"])
                    for k, side in enumerate(("left", "right")):
                        if present[k]:
                            h = hands[k]
                            for c, p in _bones(HAND_PARENT):
                                _capsule(scn, h[c], h[p], HAND_R, COL[side])
                            _sphere(scn, h[HAND_WRIST], 0.01, COL[side])
                    if mode["view"] and mode["set"]:
                        mode["set"] = False
                        v.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                        if mode["view"] == 1:
                            v.cam.distance, v.cam.elevation = 3.0, -10
                        else:
                            v.cam.distance, v.cam.elevation = 0.45, -25
                    if mode["view"] == 1 and s["body_avail"]:
                        v.cam.lookat[:] = body[0] + [0, 0, 0.25]
                    elif mode["view"] == 2:
                        pts = [hands[k][HAND_WRIST] for k in range(2) if present[k]] or ([body[L_WRIST], body[R_WRIST]] if s["body_avail"] else [])
                        if pts:
                            v.cam.lookat[:] = np.mean(pts, axis=0)
                v.sync()
                if tick - stats["printed"] >= 1.0:
                    dt = tick - stats["printed"]
                    print(f"[xrt live] {tick - t0:6.1f} s  body {stats['body'] / dt:4.0f} Hz (available {100 * stats['avail'] / max(stats['polls'], 1):3.0f} %)  "
                          f"hands L {stats['hands'][0] / dt:4.0f} / R {stats['hands'][1] / dt:4.0f} Hz  quality {s['hand_active'].tolist()}  "
                          f"{'' if s['body_avail'] else 'NO BODY  '}{'' if present.any() else 'NO HANDS (controllers down? hands in view?)'}", flush=True)
                    stats.update(polls=0, avail=0, body=0, hands=[0, 0], printed=tick)
                time.sleep(max(0.0, 1 / 60 - (time.monotonic() - tick)))
    finally:
        x.close()
    if a.record and rows:
        out = Path(a.out) if a.out else OUT_DIR / time.strftime("xrt_%Y%m%d_%H%M%S"); out.mkdir(parents=True, exist_ok=True)
        R = {k: np.array([r[k] for r in rows]) for k in ("t", "wall", "head", "body", "body_avail", "body_stamp_ns", "stamp_ns", "hands", "hand_active", "ctrl")}
        np.savez_compressed(out / "xrt.npz", **R, body_names=np.array(BODY_NAMES), hand_names=np.array(HAND_NAMES), ctrl_names=np.array(CTRL_NAMES))
        print(f"[xrt live] saved {len(rows)} samples -> {out}/xrt.npz (skeleton: pico_xrt_record.py skeleton {out}/xrt.npz --video)", flush=True)


# -------------------------------------------------------------------------------------------------------------- selftest
def _synthetic_recording(path: Path, n: int = 40) -> None:
    """A standing figure (metres, z up) with the left arm swinging and both hands open, in the same frame."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pico_hand import mp_to_xr
    from wuji_glove import FLAT, FIST, human_keypoints
    rng = np.random.default_rng(0)
    body0 = np.zeros((24, 3)); z = {0: 0.95, 1: 0.9, 2: 0.9, 3: 1.05, 4: 0.5, 5: 0.5, 6: 1.2, 7: 0.1, 8: 0.1, 9: 1.35, 10: 0.02, 11: 0.02, 12: 1.5, 13: 1.42, 14: 1.42,
                                  15: 1.62, 16: 1.42, 17: 1.42, 18: 1.15, 19: 1.15, 20: 0.9, 21: 0.9, 22: 0.82, 23: 0.82}
    y = {1: 0.1, 2: -0.1, 4: 0.1, 5: -0.1, 7: 0.1, 8: -0.1, 10: 0.1, 11: -0.1, 13: 0.08, 14: -0.08, 16: 0.2, 17: -0.2, 18: 0.24, 19: -0.24, 20: 0.26, 21: -0.26, 22: 0.27, 23: -0.27}
    for j in range(24): body0[j] = [0.0, y.get(j, 0.0), z[j]]
    rows = []
    for i in range(n):
        body = body0.copy(); a_ = 0.5 * np.sin(2 * np.pi * i / n)
        body[18] = body[16] + np.array([0.27 * np.sin(a_), 0.04, -0.27 * np.cos(a_)]); body[20] = body[18] + np.array([0.26 * np.sin(a_), 0.02, -0.26 * np.cos(a_)]); body[22] = body[20] + [0.0, 0.01, -0.08]
        hands = np.zeros((2, 26, 7)); hands[:, :, 6] = 1.0
        for k, (side, w) in enumerate((("left", L_WRIST), ("right", R_WRIST))):
            kp = human_keypoints(FLAT if (i // 10) % 2 == 0 else FIST, side)
            hands[k] = mp_to_xr(kp); hands[k, :, :3] += body[w] - hands[k, 1, :3] + rng.normal(0, 0.002, 3)
        pose = np.concatenate([body, np.tile([0, 0, 0, 1.0], (24, 1))], axis=1)
        rows.append(dict(t=i * 0.02, wall=1e9 + i * 0.02, head=np.array([0.05, 0, 1.65, 0, 0, 0, 1.0]), body=pose, body_avail=True, body_stamp_ns=i, stamp_ns=i,
                         hands=hands, hand_active=np.array([1, 1]), ctrl=np.zeros(len(CTRL_NAMES))))
    R = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(path, **R, body_names=np.array(BODY_NAMES), hand_names=np.array(HAND_NAMES), ctrl_names=np.array(CTRL_NAMES))


def _selftest() -> None:
    out = OUT_DIR / "xrt_selftest"; out.mkdir(parents=True, exist_ok=True)
    _synthetic_recording(out / "xrt.npz")
    skeleton(argparse.Namespace(npz=str(out / "xrt.npz"), video=False, fps=10, frame=None))
    R = dict(np.load(out / "xrt.npz", allow_pickle=True)); agree = draw_frame(R, 5, out / "skeleton_5.png")
    assert max(agree.values()) < 10.0, agree                       # the synthetic hands sit on the body wrists
    print("[xrt] selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("record"); r.add_argument("--seconds", type=float, default=10.0); r.add_argument("--one", action="store_true", help="a single sample")
    r.add_argument("--wait", type=float, default=60.0, help="seconds to wait for the headset before giving up"); r.add_argument("--out", default="")
    s = sub.add_parser("skeleton"); s.add_argument("npz"); s.add_argument("--frame", type=int, default=None); s.add_argument("--video", action="store_true"); s.add_argument("--fps", type=int, default=10)
    lv = sub.add_parser("live", help="real-time skeleton in a MuJoCo viewer"); lv.add_argument("--record", action="store_true", help="save every changed sample at exit"); lv.add_argument("--out", default="")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.cmd == "live":
        live(a)
    elif a.cmd == "record":
        out = record(a)
        skeleton(argparse.Namespace(npz=str(out / "xrt.npz"), video=False, fps=10, frame=None))
    elif a.cmd == "skeleton":
        skeleton(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
