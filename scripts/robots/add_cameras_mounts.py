#!/usr/bin/env python
"""Adds the in-house camera hardware to the rot0 Revo 2 robot (2026-10-26, user: "add two wrist camera with mount ... and head
camera with mount ... to this robot").  Text insertion into g1_rot0_teleop_revo2_notouch.xml (idempotent, marker comments);
run scripts/robots/add_revo2_touch.py afterwards to regenerate g1_rot0_teleop_revo2.xml.

Head: the printed bracket ~/Downloads/head_stereo_mount-rot0.stl (STEP frame: camera side -x, z up, mm) placed so that its
  80 x 25 mm module pocket (plane n.p = -30.0 mm, n = (-cos 49.3, 0, -sin 49.3), centre (-62.51, 0, 93.26)) carries the back
  face of the HBVCAM module the robot already has (head_stereo_body_0: centre (0.0589, 0, 0.43797), pitch 49.3 deg, 1.6 mm
  board).  Mapping = 180 deg about z (camera side -x -> +x) + translation; the bosses land at the visor corners.
Wrist (x2): ~/Downloads/wrist_monocular_mount.step = two-piece collar (opening 59 x 48 mm, 12 mm thick, z = collar axis,
  +z toward the hand) + camera housing on its -y side; the HBVCAM-F2619HD board (20 x 14 x 8 mm) lies in the housing's
  pocket (floor z = 6.05, centre (0, -38.4)), lens on +z.  Installed as in the reference video (kitchen_scene/
  Feishu20261026-140125.qt): collar on the wrist-yaw link right behind the hand flange, housing on the PALM side (+y of the
  wrist_yaw frame: the fingertip pads face +y), camera looking along the forearm toward the fingers.  Mount frame -> wrist_yaw
  frame: x_m -> +z (thumb side), y_m -> -y, z_m -> +x; collar z -11..1 -> x 25.5..37.5 mm, housing to x 51.6 mm.
  WRIST_CAM_SIDE=palm|back|thumb|pinky flips the housing side if the video reading turns out wrong.
Cameras: wrist_left / wrist_right (MuJoCo cameras in the housing; plus wrist_{side}_{front,left,right} 90-deg faces for the
  150-deg fisheye view the recorder saves (user 2026-10-26: "save the 150 fov view each"): lens centre (0, -38.4, 14) mm of the mount frame, optical
  axis +z_m, image up = +y_m toward the forearm axis), fovy 40.4 deg = the F2619HD's vertical FOV (EFL 4.4 mm, 3 um pixels,
  1920x1080 -> 66.3 x 40.4 deg, 76 deg diagonal), rendered at 640 x 360 by the teleop recorder.
    .venv/bin/python scripts/robots/add_cameras_mounts.py [--robot-dir data/robots/g1_rot0_teleop_revo2] [--side palm]
"""
import argparse, json, os, shutil, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
MARK = "<!-- cameras+mounts 2026-10-26"
ENDMARK = "<!-- /cameras+mounts -->"
HEAD_POSE_FILE = "head_mount_pose.json"               # optional, written by apply_head_mount_pose.py from the tuner page: {"pos": [3], "quat": [w,x,y,z], "move_module": bool}
HEAD_STL = Path.home() / "Downloads/head_stereo_mount-rot0.stl"
WRIST_STLS = [Path("/tmp/claude-1000/-home-Horizon-wrk-SIMPLE/8c40e913-4e1e-4a48-82bd-486b2ab20a27/scratchpad/wristcam/cad") / f"wrist_mount_solid{i}_coarse.stl" for i in (0, 1)]
FOVY = 2 * np.degrees(np.arctan(1.62 / 4.4))            # 40.4 deg: 1080 px * 3 um / 2 = 1.62 mm half-height, EFL 4.4 mm
# mount frame -> wrist_yaw frame, housing side; rotation as quaternion (w x y z); collar axis z_m -> +x always
# The two wrist-yaw frames are NOT mirror images with respect to the hands: the Revo 2 palm is on -y of the LEFT wrist-yaw
# frame and on +y of the RIGHT one (fingertip pads sit 20 mm from the finger links toward -y / +y; the fingers curl that way).
# The thumb side is +z on both.  A collar pose is therefore given for the LEFT wrist and mirrored through the palm plane
# (y -> -y) for the right wrist: the housing lands on the same side of each hand.  (2026-10-26 22:30, after the right
# hand came out on the back of the hand with the earlier "same local pose" rule.)
PALM_Y = {"left": -1.0, "right": +1.0}
HOUSING_QUAT = {-1.0: [0.70710678, 0, 0.70710678, 0],      # housing (-y_m) -> -y of the wrist frame;  z_m -> +x
                +1.0: [0, 0.70710678, 0, 0.70710678]}      # housing -> +y;  z_m -> +x
SIDE_QUAT = {"palm": HOUSING_QUAT[PALM_Y["left"]], "back": HOUSING_QUAT[-PALM_Y["left"]],        # named for the LEFT hand (then mirrored for the right)
             "thumb": [0.5, -0.5, 0.5, -0.5],              # housing -> +z = thumb side
             "pinky": [0.5, 0.5, 0.5, 0.5]}                # housing -> -z = pinky side


def _mirror_y(q):
    """The SAME printed part installed as the mirror image of the left installation: R' = M R S with M = diag(1,-1,1) (the
    palm plane) and S = diag(-1,1,1) (the part's own symmetry plane, x_m), a proper rotation: the housing (-y_m) lands on the
    mirrored side, z_m stays along +x (toward the hand)."""
    import mujoco
    R = np.empty(9); mujoco.mju_quat2Mat(R, np.array(q, float)); R = R.reshape(3, 3)
    Rp = np.diag([1.0, -1.0, 1.0]) @ R @ np.diag([-1.0, 1.0, 1.0]); qp = np.empty(4); mujoco.mju_mat2Quat(qp, Rp.reshape(9))
    return [float(v) for v in qp]


def _mirror_pos_y(p):
    return [p[0], -p[1], p[2]]
MOUNT_X = 0.0365                                           # mount origin on the wrist x axis: collar z -11..1 mm -> x 25.5..37.5 mm
WRIST_POSE_FILE = "wrist_mount_pose.json"                  # optional, from the wrist tuner page: {"pos": [3], "quat": [w,x,y,z], "cam_tilt_deg": a} in the wrist_yaw frame (same local pose on both wrists)
CAM_BASE = [0.0, -0.0384, 0.014]                           # lens centre in the mount frame (pocket centre, 14 mm above the collar face)
POCKET = [0.0, -0.0384, 0.00605]                           # pocket floor centre: the camera tilt pivots here


def _cam_quats(tilt_deg: float):
    """Camera quaternions (w x y z) in the mount frame: the view along +z_m tilted by tilt_deg toward +y_m (the forearm axis side;
    with the housing on the back of the hand that is toward the palm).  Returns (pinhole/front, left face, right face, tilt q)."""
    import math
    a = math.radians(tilt_deg); qt = [math.cos(-a / 2), math.sin(-a / 2), 0.0, 0.0]          # rotation about x_m by -a: (0,0,1) -> (0, sin a, cos a)
    front = _quat_mul(qt, [0, 0, 1, 0]); left = _quat_mul(qt, [0.70710678, 0, -0.70710678, 0]); right = _quat_mul(qt, [0.70710678, 0, 0.70710678, 0])
    return front, left, right, qt


def _strip_blocks(t: str) -> str:
    """Remove every block inserted by this script (with or without the end marker of the first version)."""
    import re
    lines = t.split("\n"); out = []; i = 0
    while i < len(lines):
        if MARK in lines[i]:
            j = i + 1
            if lines[j].strip().startswith("<mesh"):
                ours = ('<mesh name="head_stereo_mount_rot0"', '<mesh name="wrist_cam_mount_')
                while j < len(lines) and (lines[j].strip().startswith(ours) or ENDMARK in lines[j]): j += 1
            else:
                while j < len(lines) and "</body>" not in lines[j]: j += 1
                j += 1
                if j < len(lines) and ENDMARK in lines[j]: j += 1
            i = j; continue
        out.append(lines[i]); i += 1
    return "\n".join(out)


def _quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return [w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2, w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2]


def _quat_rot(q, v):
    w, x, y, z = q; vx, vy, vz = v
    # v' = v + 2 w (q x v) + 2 q x (q x v)
    cx, cy, cz = y*vz - z*vy, z*vx - x*vz, x*vy - y*vx
    dx, dy, dz = y*cz - z*cy, z*cx - x*cz, x*cy - y*cx
    return [vx + 2*(w*cx + dx), vy + 2*(w*cy + dy), vz + 2*(w*cz + dz)]


def _place_module(t: str, rd: Path, head_pos, head_quat, move_module: bool) -> str:
    """The stereo module (5 head_stereo_body_* geoms, 2 optical sites, 12 hbvcam cameras) rides in the bracket's pocket:
    with move_module the delta between the pocket-derived pose and the given one is applied to them (from their ORIGINAL
    values kept in module_original.json), otherwise the originals are restored."""
    import re
    orig_f = rd / "module_original.json"
    pat = re.compile(r'<(geom|site|camera) name="(head_stereo_body_\d|head_stereo_(?:left|right)_optical_frame|hbvcam_\w+)"([^>]*)>')
    if not orig_f.exists():
        orig = {}
        for m in pat.finditer(t):
            a = dict(re.findall(r'(\w+)="([^"]*)"', m.group(3))); orig[m.group(2)] = dict(pos=a["pos"], quat=a["quat"])
        orig_f.write_text(json.dumps(orig, indent=1))
    orig = json.loads(orig_f.read_text())
    P0, Q0 = [-0.0041, 0, 0.3453], [0, 0, 0, 1]                                        # the pocket-derived bracket pose
    if move_module:
        # delta T = T_new * T_default^-1 ; q0 is its own inverse (180 deg), general form below
        q0inv = [Q0[0], -Q0[1], -Q0[2], -Q0[3]]; qd = _quat_mul(head_quat, q0inv)
        td = [a - b for a, b in zip(head_pos, _quat_rot(qd, P0))]
    def fix(m):
        name = m.group(2); o = orig[name]; pos = [float(v) for v in o["pos"].split()]; quat = [float(v) for v in o["quat"].split()]
        if move_module:
            pos = [a + b for a, b in zip(_quat_rot(qd, pos), td)]; quat = _quat_mul(qd, quat)
        attrs = m.group(3); attrs = re.sub(r'pos="[^"]*"', 'pos="' + " ".join(f"{v:.7g}" for v in pos) + '"', attrs); attrs = re.sub(r'quat="[^"]*"', 'quat="' + " ".join(f"{v:.7g}" for v in quat) + '"', attrs)
        return f'<{m.group(1)} name="{name}"{attrs}>'
    return pat.sub(fix, t)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--robot-dir", default="data/robots/g1_rot0_teleop_revo2"); ap.add_argument("--side", default=os.environ.get("WRIST_CAM_SIDE", "palm"), choices=sorted(SIDE_QUAT))
    ap.add_argument("--redo", action="store_true", help="strip the previously inserted blocks first (re-place the mounts)")
    ap.add_argument("--head-pose", default=None, help=f"JSON file with the head bracket pose (default: <robot-dir>/{HEAD_POSE_FILE} when it exists; else the pocket-derived pose)")
    ap.add_argument("--wrist-pose", default=None, help=f"JSON file with the wrist collar pose in the wrist_yaw frame (default: <robot-dir>/{WRIST_POSE_FILE} when it exists; else --side at x {MOUNT_X} m)")
    a = ap.parse_args(); rd = ROOT / a.robot_dir; src = rd / "g1_rot0_teleop_revo2_notouch.xml"; t = src.read_text()
    if MARK in t and not a.redo:
        print(f"[cams] {src.name} already has the cameras+mounts block; use --redo to re-place"); return
    if MARK in t:
        t = _strip_blocks(t); print("[cams] previous blocks stripped")
    head_pos, head_quat, move_module = [-0.0041, 0, 0.3453], [0, 0, 0, 1], False
    hp = Path(a.head_pose) if a.head_pose else (rd / HEAD_POSE_FILE if (rd / HEAD_POSE_FILE).exists() else None)
    if hp is not None:
        H = json.loads(hp.read_text()); head_pos, head_quat, move_module = [float(v) for v in H["pos"]], [float(v) for v in H["quat"]], bool(H.get("move_module", False))
        print(f"[cams] head bracket pose from {hp}: pos {head_pos} quat {head_quat}" + (" + module/cameras follow" if move_module else ""))
    t = _place_module(t, rd, head_pos, head_quat, move_module)
    md = rd / "meshes"; shutil.copy(HEAD_STL, md / "head_stereo_mount_rot0.STL")
    for i, p in enumerate(WRIST_STLS): shutil.copy(p, md / f"wrist_cam_mount_{'clamp_cam' if i == 0 else 'clamp_back'}.STL")
    ap_wrist = Path(a.wrist_pose) if a.wrist_pose else (rd / WRIST_POSE_FILE if (rd / WRIST_POSE_FILE).exists() else None)
    q, mount_pos, cam_tilt = SIDE_QUAT[a.side], [MOUNT_X, 0.0, 0.0], 0.0
    if ap_wrist is not None:
        Wp = json.loads(ap_wrist.read_text()); q = [float(v) for v in Wp["quat"]]; mount_pos = [float(v) for v in Wp["pos"]]; cam_tilt = float(Wp.get("cam_tilt_deg", 0.0))
        print(f"[cams] wrist collar pose from {ap_wrist}: pos {mount_pos} quat {q}, camera tilt {cam_tilt:g} deg toward the forearm axis side")
    side_pose = {"left": (mount_pos, q), "right": (_mirror_pos_y(mount_pos), _mirror_y(q))}              # the given pose is the LEFT wrist's
    qf, ql, qr, qt = _cam_quats(cam_tilt); fq = lambda qq: " ".join(f"{v:.8g}" for v in qq)
    def tilted(p):                                                      # module geoms pivot about the pocket floor centre
        rel = [p[i] - POCKET[i] for i in range(3)]; r = _quat_rot(qt, rel); return " ".join(f"{POCKET[i] + r[i]:.6g}" for i in range(3))
    cam_pos = tilted(CAM_BASE)
    assets = f'''    {MARK}: printed mounts (mm STL) -->
    <mesh name="head_stereo_mount_rot0" file="head_stereo_mount_rot0.STL" scale="0.001 0.001 0.001" />
    <mesh name="wrist_cam_mount_clamp_cam" file="wrist_cam_mount_clamp_cam.STL" scale="0.001 0.001 0.001" />
    <mesh name="wrist_cam_mount_clamp_back" file="wrist_cam_mount_clamp_back.STL" scale="0.001 0.001 0.001" />
    {ENDMARK}
'''
    head = f'''            {MARK}: the rot0 head bracket, placed by its module pocket (see scripts/robots/add_cameras_mounts.py) -->
            <body name="head_stereo_mount" pos="{' '.join(f'{v:.6g}' for v in head_pos)}" quat="{' '.join(f'{v:.8g}' for v in head_quat)}">
              <geom name="head_stereo_mount" type="mesh" mesh="head_stereo_mount_rot0" contype="0" conaffinity="0" group="1" mass="0" rgba="0.12 0.12 0.13 1" />
            </body>
            {ENDMARK}
'''
    def wrist(side):
        ps = " ".join(f"{v:.6g}" for v in side_pose[side][0]); qs = " ".join(f"{v:.8g}" for v in side_pose[side][1])
        return f'''                          {MARK}: wrist camera collar + HBVCAM-F2619HD ({side}: palm on {"-y" if PALM_Y[side] < 0 else "+y"} of this wrist frame) -->
                          <body name="{side}_wrist_cam_mount" pos="{ps}" quat="{qs}">
                            <geom name="{side}_wrist_cam_clamp_cam" type="mesh" mesh="wrist_cam_mount_clamp_cam" contype="0" conaffinity="0" group="1" mass="0" rgba="0.12 0.12 0.13 1" />
                            <geom name="{side}_wrist_cam_clamp_back" type="mesh" mesh="wrist_cam_mount_clamp_back" contype="0" conaffinity="0" group="1" mass="0" rgba="0.12 0.12 0.13 1" />
                            <geom name="{side}_wrist_cam_board" type="box" size="0.010 0.007 0.0006" pos="{tilted([0, -0.0384, 0.00665])}" quat="{fq(qt)}" contype="0" conaffinity="0" group="1" mass="0" rgba="0.05 0.25 0.12 1" />
                            <geom name="{side}_wrist_cam_lens" type="cylinder" size="0.004 0.00335" pos="{tilted([0, -0.0384, 0.01065])}" quat="{fq(qt)}" contype="0" conaffinity="0" group="1" mass="0" rgba="0.05 0.05 0.05 1" />
                            <geom name="{side}_wrist_cam_glass" type="cylinder" size="0.0025 0.0002" pos="{tilted([0, -0.0384, 0.0142])}" quat="{fq(qt)}" contype="0" conaffinity="0" group="1" mass="0" rgba="0.1 0.2 0.35 1" />
                            <site name="{side}_wrist_cam_optical" pos="{cam_pos}" quat="{fq(qt)}" size="0.002" group="4" rgba="0.1 0.65 0.6 1" />
                            <camera name="wrist_{side}" pos="{cam_pos}" quat="{fq(qf)}" fovy="{FOVY:.2f}" />
                            <camera name="wrist_{side}_front" pos="{cam_pos}" quat="{fq(qf)}" fovy="90" />
                            <camera name="wrist_{side}_left" pos="{cam_pos}" quat="{fq(ql)}" fovy="90" />
                            <camera name="wrist_{side}_right" pos="{cam_pos}" quat="{fq(qr)}" fovy="90" />
                          </body>
                          {ENDMARK}
'''
    def insert_before(text, anchor, block, count=1):
        assert text.count(anchor) == count, (anchor, text.count(anchor)); return text.replace(anchor, block + anchor)
    t = insert_before(t, '    <mesh name="left_wrist_yaw_link" file="left_wrist_yaw_link.STL" />\n', assets)
    # head block after the right pinhole camera line
    i = t.index('<camera name="hbvcam_right_pinhole"'); j = t.index("\n", i) + 1; t = t[:j] + head + t[j:]
    for side in ("left", "right"):
        t = insert_before(t, f'                          <body name="{side}_hand_base2_link" pos="0.0415 0 0">\n', wrist(side))
    src.write_text(t)
    info = dict(made="2026-10-26", wrist_side=a.side if ap_wrist is None else f"tuner page ({ap_wrist.name})", mount_quat_wxyz=q, mount_pos_m=mount_pos, mount_x_m=mount_pos[0], cam_tilt_deg=cam_tilt, cam_quat_in_mount=qf, cam_pos_in_mount=[float(v) for v in cam_pos.split()],
                palm_y=PALM_Y, right_wrist_pose=dict(pos=side_pose["right"][0], quat_wxyz=side_pose["right"][1], note="the left pose mirrored through the palm plane (y -> -y)"), fovy_deg=round(float(FOVY), 2), image=[640, 360],
                head_mount=dict(stl=str(HEAD_STL), pos=head_pos, quat_wxyz=head_quat, module_follows=move_module, placed_by="module pocket = back face of head_stereo_body_0" if hp is None else f"tuner page ({hp.name})"),
                wrist_camera=dict(lens_in_mount_frame_m=[0, -0.0384, 0.014], optical_axis="+z_m = +x of wrist_yaw (toward the hand)", image_up="+y_m (toward the forearm axis)"),
                module=dict(name="HBVCAM-F2619HD V11", board_mm=[20, 14, 8], efl_mm=4.4, pixel_um=3.0, sensor="OS02H10 1/2.8in 1920x1080", hfov_deg=66.3, vfov_deg=40.4, dfov_deg=73.8))
    (rd / "cameras.json").write_text(json.dumps(info, indent=1)); print(f"[cams] inserted into {src}; meshes copied; {rd / 'cameras.json'} written (wrist side {a.side}, fovy {FOVY:.2f})")


if __name__ == "__main__":
    main()
