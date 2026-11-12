"""HBVCAM-F2439GS-2 stereo fisheye on the Unitree G1: calibration, fisheye images from pinhole renders, and remapping.

MuJoCo, Isaac and most renderers only have pinhole cameras. This module turns pinhole images into the HBVCAM fisheye
image with the calibrated OpenCV fisheye model (calibration/stereo_calib.yaml, 1280x720 per eye):

  * fisheye_rays()        the viewing ray of every fisheye pixel (inverse of cv2.fisheye.projectPoints)
  * remap_faces()         full fisheye image from five 90 deg pinhole renders at the eye (front/left/right/up/down)
  * pinhole_to_fisheye()  warp ONE existing pinhole image into the fisheye (black where it has no data)
  * render_hbvcam()       MuJoCo: render the five faces of each eye from g1_29dof_with_hand_hbvcam_stereo.xml and
                          return left/right fisheye (raw or rectified) images

Command line:
  python hbvcam_fisheye.py selftest
  MUJOCO_GL=egl python hbvcam_fisheye.py render --out out/
  python hbvcam_fisheye.py remap frame.png --fovy 77.55 --source simple_head --out frame_fisheye.png

Needs numpy and opencv-python; mujoco >= 3.2 only for render / the MJCF checks.
"""
import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CALIB = HERE / "calibration" / "stereo_calib.yaml"
URDF = HERE / "g1_29dof_with_hand_hbvcam_stereo.urdf"
MJCF = HERE / "g1_29dof_with_hand_hbvcam_stereo.xml"
W, H = 1280, 720                                   # per eye; the camera streams both side by side as 2560 x 720

# the five 90 deg faces, as (forward, right, down) in the eye's optical frame (x right, y down, z forward)
FACES = {
    "front": ([0, 0, 1], [1, 0, 0], [0, 1, 0]),
    "right": ([1, 0, 0], [0, 0, -1], [0, 1, 0]),
    "left": ([-1, 0, 0], [0, 0, 1], [0, 1, 0]),
    "up": ([0, -1, 0], [1, 0, 0], [0, 0, 1]),
    "down": ([0, 1, 0], [1, 0, 0], [0, 0, -1]),
}
BODY_TO_OPTICAL = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # x fwd / y left / z up -> optical
OPT_TO_MJCAM = np.diag([1.0, -1.0, -1.0])                                 # optical -> MuJoCo camera (looks along -z)


# ---------------------------------------------------------------- rotations
def quat_to_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    """wxyz quaternion of a rotation matrix."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = s / 4
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q if q[0] >= 0 else -q


def rpy_to_mat(r, p, y):
    """URDF rpy: fixed-axis roll about x, then pitch about y, then yaw about z."""
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# ---------------------------------------------------------------- calibration and frames
def load_calib(path=CALIB):
    """K1, D1, K2, D2 (per eye), R, T (right w.r.t. left), R1, R2, P1, P2 (rectification) as numpy arrays."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    c = {k: fs.getNode(k).mat() for k in ["K1", "D1", "K2", "D2", "R", "T", "R1", "R2", "P1", "P2"]}
    fs.release()
    c["D1"], c["D2"] = c["D1"].ravel(), c["D2"].ravel()
    return c


def eye_poses(urdf=URDF):
    """4x4 poses of head_stereo_{left,right}_optical_frame in torso_link, read from the URDF joints."""
    out = {}
    for j in ET.parse(urdf).getroot().findall("joint"):
        for eye in ("left", "right"):
            if j.get("name") == f"head_stereo_{eye}_optical_joint":
                assert j.find("parent").get("link") == "torso_link"
                o = j.find("origin")
                T = np.eye(4)
                T[:3, :3] = rpy_to_mat(*[float(v) for v in o.get("rpy").split()])
                T[:3, 3] = [float(v) for v in o.get("xyz").split()]
                out[eye] = T
    return out


def eye_calib(c, eye):
    return (c["K1"], c["D1"], c["R1"], c["P1"]) if eye == "left" else (c["K2"], c["D2"], c["R2"], c["P2"])


# other cameras' optical frames in torso_link, for remapping their images ('none' = keep the source pose)
SOURCES = {
    # SIMPLE engines/mujoco.py eye_in_head camera ("Realsense_D435i" config: 640x360, fov 110 deg horizontal)
    "simple_head": quat_to_mat([0.91496, 0.0, 0.40355, 0.0]) @ BODY_TO_OPTICAL,
    # stock G1 head d435_link (rpy 0 0.8307767 0)
    "d435_link": rpy_to_mat(0, 0.8307767239493009, 0) @ BODY_TO_OPTICAL,
}


# ---------------------------------------------------------------- the fisheye model
def distort_theta(th, D):
    t2 = th * th
    return th * (1 + t2 * (D[0] + t2 * (D[1] + t2 * (D[2] + t2 * D[3]))))


def fisheye_rays(K, D, w=W, h=H, R_rect=None, P=None):
    """Unit viewing ray (h, w, 3) of every pixel in the eye's optical frame, a valid mask, and the angle off axis.

    Raw image: invert thd = th (1 + k1 th^2 + k2 th^4 + k3 th^6 + k4 th^8) per pixel (Newton); pixels past the 180 deg
    lens circle (th > 90 deg) are invalid. Rectified image (R_rect = R1/R2, P = P1/P2): rays of the rectified pinhole
    turned by R_rect^T, valid where the raw image has data. Pixel centres are at integer coordinates (OpenCV)."""
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    D = np.asarray(D, float).ravel()
    half = np.pi / 2
    if R_rect is None:
        x, y = (u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1]
        thd = np.hypot(x, y)
        t = np.minimum(thd, distort_theta(half, D))
        th = t.copy()
        for _ in range(20):
            t2 = th * th
            f = distort_theta(th, D) - t
            df = 1 + t2 * (3 * D[0] + t2 * (5 * D[1] + t2 * (7 * D[2] + 9 * D[3] * t2)))
            th = th - f / df
        s = np.where(thd > 1e-12, np.sin(th) / np.maximum(thd, 1e-12), 1.0)
        rays = np.stack([x * s, y * s, np.cos(th)], -1)
        valid = thd <= distort_theta(half, D)
    else:
        f, cx, cy = P[0, 0], P[0, 2], P[1, 2]
        r = np.stack([(u - cx) / f, (v - cy) / f, np.ones_like(u)], -1) @ np.asarray(R_rect, float)   # R^T per row
        rays = r / np.linalg.norm(r, axis=-1, keepdims=True)
        th = np.arccos(np.clip(rays[..., 2], -1, 1))
        rr = np.hypot(rays[..., 0], rays[..., 1])
        sc = np.where(rr > 1e-12, distort_theta(th, D) / np.maximum(rr, 1e-12), 0)
        pu, pv = K[0, 0] * rays[..., 0] * sc + K[0, 2], K[1, 1] * rays[..., 1] * sc + K[1, 2]
        valid = (pu >= -0.5) & (pu <= w - 0.5) & (pv >= -0.5) & (pv <= h - 0.5)
    return rays, valid & (th <= half), th


def remap_faces(faces, rays, valid):
    """Fisheye image from the five square 90 deg face renders (dict name -> HxWx3, image row 0 = top)."""
    names = list(FACES)
    pick = np.argmax(np.stack([rays @ np.array(FACES[n][0], float) for n in names], -1), -1)
    out = np.zeros(rays.shape[:2] + (3,), np.uint8)
    for i, n in enumerate(names):
        N = faces[n].shape[0]
        fwd, right, down = (np.array(a, float) for a in FACES[n])
        z = np.maximum(rays @ fwd, 1e-6)
        mu = ((rays @ right) / z + 1) * N / 2 - 0.5
        mv = ((rays @ down) / z + 1) * N / 2 - 0.5
        img = cv2.remap(faces[n], mu.astype(np.float32), mv.astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE)
        sel = (pick == i) & valid
        out[sel] = img[sel]
    return out


def pinhole_to_fisheye(img, fovy_deg, R_src_from_fish, K, D, w=W, h=H, R_rect=None, P=None):
    """Warp one pinhole image (principal point at the centre, square pixels, vertical fov fovy_deg) into the fisheye.

    R_src_from_fish turns fisheye-eye optical rays into the source camera's optical frame (np.eye(3) = same pose;
    translation between the two cameras is ignored). Returns (image, fraction of the lens circle the source covers)."""
    sh, sw = img.shape[:2]
    f = (sh / 2) / np.tan(np.radians(fovy_deg) / 2)
    rays, valid, _ = fisheye_rays(K, D, w, h, R_rect, P)
    r = rays @ np.asarray(R_src_from_fish, float).T
    z = r[..., 2]
    mu = f * r[..., 0] / np.maximum(z, 1e-9) + (sw - 1) / 2
    mv = f * r[..., 1] / np.maximum(z, 1e-9) + (sh - 1) / 2
    inside = valid & (z > 1e-6) & (mu >= -0.5) & (mu <= sw - 0.5) & (mv >= -0.5) & (mv <= sh - 0.5)
    out = cv2.remap(img, mu.astype(np.float32), mv.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    out[~inside] = 0
    return out, float(inside.sum() / max(valid.sum(), 1))


# ---------------------------------------------------------------- MuJoCo
def render_hbvcam(model, data, renderer, calib, rectified=False, eyes=("left", "right"), prefix="head_stereo"):
    """Fisheye images of both eyes from the model's cameras '<prefix>_<eye>_<face>' (fovy 90, square renderer).
    The renderer must be square (e.g. mujoco.Renderer(model, 1536, 1536)). Returns {eye: HxWx3 uint8}."""
    out = {}
    for eye in eyes:
        faces = {}
        for f in FACES:
            renderer.update_scene(data, camera=f"{prefix}_{eye}_{f}")
            faces[f] = renderer.render().copy()
        K, D, R, P = eye_calib(calib, eye)
        rays, valid, _ = fisheye_rays(K, D, R_rect=R if rectified else None, P=P if rectified else None)
        out[eye] = remap_faces(faces, rays, valid)
    return out


def add_hbvcam_cameras(body, eye_T, prefix="head_stereo"):
    """Add the five face cameras of each eye (eye_T: {eye: 4x4 in the body's frame}) to an MjSpec body."""
    for eye, T in eye_T.items():
        for f, (fwd, right, down) in FACES.items():
            R = T[:3, :3] @ np.column_stack([right, down, fwd]).astype(float)
            body.add_camera(name=f"{prefix}_{eye}_{f}", pos=T[:3, 3].tolist(), quat=mat_to_quat(R @ OPT_TO_MJCAM).tolist(),
                            fovy=90.0)


# ---------------------------------------------------------------- command line
def cmd_selftest(_):
    ok = True

    def check(name, cond, detail):
        nonlocal ok
        ok &= bool(cond)
        print(f"{'PASS' if cond else 'FAIL'}  {name}: {detail}")

    c = load_calib()
    check("calibration", abs(c["K1"][0, 0] - 471.197) < 0.01 and abs(np.linalg.norm(c["T"]) - 60.517) < 0.01,
          f"fx_left {c['K1'][0, 0]:.3f} px, baseline {np.linalg.norm(c['T']):.3f} mm")
    rng = np.random.default_rng(0)
    for eye in ("left", "right"):
        K, D, R, P = eye_calib(c, eye)
        rays, valid, th = fisheye_rays(K, D)
        vv, uu = np.nonzero(valid & (th < np.radians(88)))
        i = rng.choice(len(uu), 4000, replace=False)
        px, _ = cv2.fisheye.projectPoints(rays[vv[i], uu[i]].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
        err = np.abs(px.reshape(-1, 2) - np.c_[uu[i], vv[i]]).max()
        check(f"{eye} raw rays vs cv2.fisheye.projectPoints", err < 1e-3, f"max {err:.2e} px over 4000 pixels")
        rays, valid, _ = fisheye_rays(K, D, R_rect=R, P=P)
        vv, uu = np.nonzero(valid)
        i = rng.choice(len(uu), 4000, replace=False)
        x = rays[vv[i], uu[i]] @ R.T
        err = np.abs(np.c_[P[0, 0] * x[:, 0] / x[:, 2] + P[0, 2], P[1, 1] * x[:, 1] / x[:, 2] + P[1, 2]] - np.c_[uu[i], vv[i]]).max()
        check(f"{eye} rectified rays vs R/P", err < 1e-6, f"max {err:.2e} px")
    E = eye_poses()
    b = np.linalg.norm(E["left"][:3, 3] - E["right"][:3, 3]) * 1000
    check("URDF eye frames", 55 < b < 65, f"baseline {b:.2f} mm, left axis {np.degrees(np.arcsin(-E['left'][2, 2])):.1f} deg down")
    try:
        import mujoco
    except ImportError:
        print("SKIP  MJCF checks: mujoco is not installed")
        return 0 if ok else 1
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    worst = 0.0
    for eye, T in E.items():
        s = m.site(f"head_stereo_{eye}_optical_frame")
        worst = max(worst, np.abs(s.pos - T[:3, 3]).max(), np.abs(quat_to_mat(s.quat) - T[:3, :3]).max())
        for f, (fwd, right, down) in FACES.items():
            cam = m.cam(f"head_stereo_{eye}_{f}")
            assert cam.bodyid[0] == torso
            Rw = T[:3, :3] @ np.column_stack([right, down, fwd]).astype(float) @ OPT_TO_MJCAM
            worst = max(worst, np.abs(cam.pos - T[:3, 3]).max(), np.abs(quat_to_mat(cam.quat) - Rw).max())
    check("MJCF sites and 10 face cameras vs URDF", worst < 1e-5, f"max deviation {worst:.1e} (m / rotation-matrix entries)")
    return 0 if ok else 1


def demo_spec(mjcf=MJCF):
    """The robot on a floor with a calibration-style board 1.6 m ahead, for a quick look."""
    import mujoco
    spec = mujoco.MjSpec.from_file(str(mjcf))
    tex = spec.add_texture(name="demo_grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.42, 0.44, 0.46], rgb2=[0.34, 0.36, 0.38], width=512, height=512)
    mat = spec.add_material(name="demo_grid", texrepeat=[80, 80])
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
    spec.add_texture(name="demo_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.78, 0.83, 0.88], rgb2=[0.35, 0.4, 0.45], width=512, height=3072)
    wb = spec.worldbody
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[20, 20, 0.1], material="demo_grid", contype=0, conaffinity=0)
    wb.add_light(pos=[1.5, 0.5, 3.0], dir=[-0.4, -0.15, -0.9], diffuse=[0.6] * 3, castshadow=False)
    wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.002, 0.5, 0.45], pos=[1.6, 0, 1.2], rgba=[0.95, 0.95, 0.93, 1],
                contype=0, conaffinity=0)
    for r in range(8):
        for col in range(9):
            if (r + col) % 2 == 0:
                wb.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.0012, 0.05, 0.05], contype=0, conaffinity=0,
                            pos=[1.597, 0.4 - col * 0.1, 1.55 - r * 0.1], rgba=[0.08, 0.08, 0.08, 1])
    return spec


def cmd_render(a):
    import mujoco
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = demo_spec(a.model).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    c = load_calib()
    r = mujoco.Renderer(model, a.face, a.face)
    try:
        for kind in ("raw", "rect"):
            ims = render_hbvcam(model, data, r, c, rectified=kind == "rect")
            sbs = np.hstack([ims["left"], ims["right"]])
            cv2.imwrite(str(out / f"hbvcam_sbs_{kind}.png"), cv2.cvtColor(sbs, cv2.COLOR_RGB2BGR))
            print(f"{out / f'hbvcam_sbs_{kind}.png'}  {sbs.shape[1]}x{sbs.shape[0]}")
    finally:
        r.close()
    return 0


def cmd_remap(a):
    c = load_calib()
    K, D, R, P = eye_calib(c, a.eye)
    img = cv2.imread(a.image, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"cannot read {a.image}")
    if a.source == "none":
        Rsf = np.eye(3)
    else:
        Rsf = SOURCES[a.source].T @ eye_poses()[a.eye][:3, :3]
    fish, cov = pinhole_to_fisheye(img, a.fovy, Rsf, K, D, R_rect=R if a.rect else None, P=P if a.rect else None)
    cv2.imwrite(a.out, fish)
    print(f"{a.out}: {fish.shape[1]}x{fish.shape[0]}, the source fills {100 * cov:.0f} % of the lens circle")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest", help="check the model maths against OpenCV and the MJCF cameras against the URDF")
    p = sub.add_parser("render", help="MuJoCo: render both eyes (raw and rectified) as 2560x720 frames")
    p.add_argument("--model", default=str(MJCF))
    p.add_argument("--out", default="out")
    p.add_argument("--face", type=int, default=1536, help="pixels per 90 deg face render (<= 2048)")
    p = sub.add_parser("remap", help="warp one pinhole image into the HBVCAM fisheye")
    p.add_argument("image")
    p.add_argument("--fovy", type=float, required=True, help="vertical field of view of the source image, degrees")
    p.add_argument("--source", choices=["none", *SOURCES], default="none",
                   help="source camera orientation on torso_link; 'none' keeps its pose and only changes the lens")
    p.add_argument("--eye", choices=["left", "right"], default="left")
    p.add_argument("--rect", action="store_true", help="output the rectified view instead of the raw fisheye")
    p.add_argument("--out", default="fisheye.png")
    a = ap.parse_args(argv)
    return {"selftest": cmd_selftest, "render": cmd_render, "remap": cmd_remap}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
