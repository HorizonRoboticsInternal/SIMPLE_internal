#!/usr/bin/env python
"""Camera intrinsics + extrinsics of the rot0 Revo 2 robot in one file (2026-10-28, user: "does the file include the intrinsic
and extrinsic of the cameras?" -- the robot folder had the poses in the MJCF/URDF but the recorded-image intrinsics only
inside each recording's meta/holomotion_raw.json).

    .venv/bin/python scripts/robots/export_camera_calibration.py      # -> data/robots/g1_rot0_teleop_revo2/camera_calibration.json

Contents: the four recorded image streams (model, size, K / fisheye f, the exact numbers the recorder writes to
meta/holomotion_raw.json "images", the MuJoCo cameras they are rendered from, their ROS optical frame = the URDF link with
its parent and xyz / rpy, and the pose in the pelvis at the zero pose), the raw HBVCAM fisheye stereo calibration of the
real head module (K, D per eye, R, T, R1/R2/P1/P2), every MuJoCo camera of the model, and the IMU frames.
Checks: each exported frame equals the URDF joint origin (1e-6 m / 1e-5 rad), and the recorded intrinsics equal those in
an existing recording's meta/holomotion_raw.json when one is present.
"""
from __future__ import annotations

import datetime, json, re, sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2, mujoco, numpy as np
from scipy.spatial.transform import Rotation as Rot

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from simple.teleop.holomotion_v14.wrist_cams import FACES, FOV_DEG, H as WRIST_H, NEAR_M, W as WRIST_W, image_spec  # noqa: E402

RB = ROOT / "data/robots/g1_rot0_teleop_revo2"; MJCF = RB / "g1_rot0_teleop_revo2_v3.xml"; URDF = RB / "g1_rot0_teleop_revo2.urdf"
CALIB = ROOT / "third_party/hbvcam_stereo/calibration/stereo_calib.yaml"; OUT = RB / "camera_calibration.json"
PINHOLE_W, PINHOLE_H = 1280, 720; REC_W, REC_H = 640, 360
RX_PI = np.diag([1.0, -1.0, -1.0])                       # MuJoCo camera (-z forward, y up) -> ROS optical (z forward, y down)
RECORDING = ROOT / "data/teleop_holomotion_v14_kitchen5/simple/G1WholebodyKitchenSinkTeleop-v0/level-3_pinhole_filled/meta/holomotion_raw.json"
R6 = lambda a: np.round(np.asarray(a, float), 6).tolist()


def R_of(q_wxyz):
    return Rot.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()


def quat_wxyz(R):
    q = Rot.from_matrix(R).as_quat(); q = q if q[3] >= 0 else -q
    return [float(q[3]), float(q[0]), float(q[1]), float(q[2])]


def frame(parent: str, p, R) -> dict:
    return dict(parent=parent, xyz_m=R6(p), rpy_rad=R6(Rot.from_matrix(R).as_euler("xyz")), quat_wxyz=R6(quat_wxyz(R)))


def load_calib(path: Path) -> dict:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    c = {k: fs.getNode(k).mat() for k in ("K1", "D1", "K2", "D2", "R", "T", "R1", "R2", "P1", "P2")}
    size = (int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real())); fs.release()
    assert size == (PINHOLE_W, PINHOLE_H), size
    return c


def recorded_pinhole(P, eye: str) -> dict:
    """The numbers simple.cli.teleop_holomotion_v14._pinhole_image_spec writes (640 x 360 from a 1280 x 720 render)."""
    sx, sy = REC_W / PINHOLE_W, REC_H / PINHOLE_H
    d = dict(width=REC_W, height=REC_H, fx=round(float(P[0, 0]) * sx, 4), fy=round(float(P[1, 1]) * sy, 4),
             cx=round((float(P[0, 2]) + 0.5) * sx - 0.5, 4), cy=round((float(P[1, 2]) + 0.5) * sy - 0.5, 4))
    if eye == "right":
        d["baseline_mm"] = round(-float(P[0, 3]) / float(P[0, 0]), 3)
    return d


def urdf_frames(path: Path) -> dict:
    out = {}
    for j in ET.parse(path).getroot().findall("joint"):
        o = j.find("origin")
        out[j.find("child").get("link")] = dict(parent=j.find("parent").get("link"), xyz=[float(v) for v in o.get("xyz").split()], rpy=[float(v) for v in o.get("rpy").split()])
    return out


def main():
    m = mujoco.MjModel.from_xml_path(str(MJCF)); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    xml = MJCF.read_text(); uf = urdf_frames(URDF); cal = load_calib(CALIB)
    body_name = lambda b: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b))
    pelvis = m.body("pelvis").id; Rp, pp = d.xmat[pelvis].reshape(3, 3), d.xpos[pelvis]

    def world(body: int, p, R):
        Rb, pb = d.xmat[body].reshape(3, 3), d.xpos[body]
        return pb + Rb @ np.asarray(p), Rb @ R

    def in_pelvis(body: int, p, R):
        pw, Rw = world(body, p, R)
        return frame("pelvis (zero pose, pelvis at z 0.793)", Rp.T @ (pw - pp), Rp.T @ Rw)

    def check_urdf(link: str, parent: str, p, R):
        u = uf[link]; assert u["parent"] == parent, (link, u["parent"], parent)
        dp = np.abs(np.asarray(u["xyz"]) - p).max(); dR = np.abs(Rot.from_euler("xyz", u["rpy"]).as_matrix() - R).max()
        assert dp < 1e-6 and dR < 1e-5, (link, dp, dR)

    def cam_attrs(name: str) -> dict:
        tag = re.search(rf'<camera name="{name}"[^>]*/>', xml).group(0); at = dict(re.findall(r'(\w+)="([^"]*)"', tag))
        return {k: v for k, v in at.items() if k not in ("name", "pos", "quat")}

    cams = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, i): i for i in range(m.ncam)}
    recorded = {}
    for eye, key, rp in (("left", "observation.images.ego_view_left", "R1/P1"), ("right", "observation.images.ego_view_right", "R2/P2")):
        cam = f"hbvcam_{eye}_pinhole"; ci = cams[cam]; body = int(m.cam_bodyid[ci]); R = R_of(m.cam_quat[ci]) @ RX_PI; p = m.cam_pos[ci]
        link = f"{cam}_optical_frame"; check_urdf(link, body_name(body), p, R)
        P, Rr = cal["P" + rp[1]], cal["R" + rp[1]]; k = recorded_pinhole(P, eye)
        fovh, fovv = 2 * np.degrees(np.arctan(REC_W / 2 / k["fx"])), 2 * np.degrees(np.arctan(REC_H / 2 / k["fy"]))
        recorded[key] = dict(
            what=f"rectified {eye} eye of the HBVCAM stereo head ({rp} of the stereo calibration), rendered by MuJoCo at {PINHOLE_W}x{PINHOLE_H} and area-downscaled to {REC_W}x{REC_H}",
            model="pinhole, rectified: no distortion; u = fx x/z + cx, v = fy y/z + cy in the optical frame", **k,
            K=[[k["fx"], 0.0, k["cx"]], [0.0, k["fy"], k["cy"]], [0.0, 0.0, 1.0]], fov_deg=dict(h=round(fovh, 2), v=round(fovv, 2)),
            render=dict(mujoco_camera=cam, size=[PINHOLE_W, PINHOLE_H], mjcf_camera_attributes=cam_attrs(cam), downscale="area 2x; pixel centres x' = (x + 0.5) s - 0.5",
                        P_1280x720=R6(P), R_rect=R6(Rr), note="K at 1280x720 = P[:3,:3]; the rectified eye frame = the raw eye frame rotated by R_rect"),
            optical_frame=dict(urdf_link=link, **frame(body_name(body), p, R), in_pelvis=in_pelvis(body, p, R)))
    recorded["observation.images.ego_view_right"]["stereo"] = dict(baseline_mm=recorded["observation.images.ego_view_right"]["baseline_mm"],
                                                                 T_left_to_right_m=[round(recorded["observation.images.ego_view_right"]["baseline_mm"] / 1000, 6), 0.0, 0.0],
                                                                 note="rectified pair: the right optical frame is the left one shifted along +x; disparity = fx * baseline / depth")
    ws = image_spec()
    for side in ("left", "right"):
        key = f"observation.images.wrist_{side}"; cam = f"wrist_{side}"; ci = cams[cam]; body = int(m.cam_bodyid[ci]); R = R_of(m.cam_quat[ci]) @ RX_PI; p = m.cam_pos[ci]
        link = f"{cam}_optical_frame"; check_urdf(link, body_name(body), p, R)
        mount = body; wy = m.body(f"{side}_wrist_yaw_link").id; assert int(m.body_parentid[mount]) == wy
        Rm, pm = R_of(m.body_quat[mount]), m.body_pos[mount]
        s = ws[key]; s.pop("mujoco_camera", None)
        recorded[key] = dict(what=s.pop("what"), model=s.pop("model") + "; theta = angle to the optical axis, phi = atan2(v - cy, u - cx); no further distortion; black outside the lens circle", **s,
                             render=dict(mujoco_faces=[f"{cam}_{f}" for f in FACES], face_fovy_deg=90, face_px=448, near_plane_m=NEAR_M, schedule_hz=30,
                                         pinhole_camera_of_the_module=dict(name=cam, mjcf_camera_attributes=cam_attrs(cam), note="the F2619HD's own 76-deg (diagonal) pinhole at the same pose; not recorded")),
                             optical_frame=dict(urdf_link=link, **frame(body_name(body), p, R), in_wrist_yaw_link=frame(f"{side}_wrist_yaw_link", pm + Rm @ p, Rm @ R),
                                                in_pelvis=in_pelvis(body, p, R)))
    # raw fisheye eyes = the MJCF sites head_stereo_{eye}_optical_frame (already optical convention)
    raw = {}
    for eye, k in (("left", "1"), ("right", "2")):
        si = m.site(f"head_stereo_{eye}_optical_frame").id; body = int(m.site_bodyid[si]); R = R_of(m.site_quat[si]); p = m.site_pos[si]
        link = f"head_stereo_{eye}_optical_frame"; check_urdf(link, body_name(body), p, R)
        raw[eye] = dict(K=R6(cal["K" + k]), D_fisheye_k1_k4=R6(cal["D" + k].ravel()), image=[PINHOLE_W, PINHOLE_H], model="OpenCV fisheye (cv2.fisheye): r = f theta_d, theta_d = theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)",
                        optical_frame=dict(urdf_link=link, **frame(body_name(body), p, R), in_pelvis=in_pelvis(body, p, R)))
    raw["stereo"] = dict(R_right_wrt_left=R6(cal["R"]), T_right_wrt_left_mm=R6(cal["T"].ravel()), note="OpenCV stereoCalibrate: x_right = R x_left + T")
    raw["rectification"] = dict(R1=R6(cal["R1"]), R2=R6(cal["R2"]), P1=R6(cal["P1"]), P2=R6(cal["P2"]), note="OpenCV fisheye.stereoRectify outputs at 1280x720; the recorded ego_view K = P[:3,:3] scaled to 640x360")
    raw["calibration"] = dict(file="third_party/hbvcam_stereo/calibration/stereo_calib.yaml", report="third_party/hbvcam_stereo/calibration/calibration_report.txt", date="2026-09-10",
                              board="8x7 inner corners, 100 mm squares, 25 of 35 pairs used", stereo_rms_px=0.11668, baseline_mm=60.517121,
                              note="calibration of the real HBVCAM module; its eye poses on the robot are the sites above (the 10-26 head tuning moved them, a visual fit, not a calibration)")
    # every MuJoCo camera
    role = {"pinhole": "rectified eye, recorded", "front": "fisheye face", "left": "fisheye face", "right": "fisheye face", "up": "fisheye face", "down": "fisheye face"}
    all_cams = []
    for name, ci in cams.items():
        body = int(m.cam_bodyid[ci]); kind = name.split("_")[-1]
        all_cams.append(dict(name=name, body=body_name(body), pos_m=R6(m.cam_pos[ci]), quat_wxyz=R6(m.cam_quat[ci]), attributes=cam_attrs(name),
                             role=("wrist module pinhole (not recorded)" if name in ("wrist_left", "wrist_right") else role.get(kind, "")),
                             optical_frame_rpy_rad=R6(Rot.from_matrix(R_of(m.cam_quat[ci]) @ RX_PI).as_euler("xyz"))))
    imu = {}
    si = m.site("imu").id; imu["mjcf_site_imu"] = frame(body_name(int(m.site_bodyid[si])), m.site_pos[si], R_of(m.site_quat[si]))
    for link in ("imu_link", "pelvis_imu_link", "d435_link", "mid360_link"):
        if link in uf:
            imu[f"urdf_{link}"] = dict(parent=uf[link]["parent"], xyz_m=R6(uf[link]["xyz"]), rpy_rad=R6(uf[link]["rpy"]))
    imu["note"] = "the recorded states.robot.imu_* columns are defined at pelvis_imu_link; imu_link is the stock G1 torso IMU (= the MJCF site imu)"
    out = dict(
        made=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), source_mjcf=str(MJCF.relative_to(ROOT)), source_urdf=str(URDF.relative_to(ROOT)), source_calibration=str(CALIB.relative_to(ROOT)),
        conventions=dict(optical_frame="ROS optical: x right, y down, z forward = the image axes (u along x, v along y); the URDF *_optical_frame links",
                         mujoco_camera="looks along its -z with y up; optical = MuJoCo camera frame x Rx(pi)", quaternions="w x y z", units="m, rad, px; calibration T in mm",
                         rpy="URDF: R = Rz(yaw) Ry(pitch) Rx(roll)", pixels="OpenCV: pixel (0,0) is centred at x = 0, y = 0",
                         frames_in_pelvis="world pose at the model's zero pose (every joint 0, pelvis at (0, 0, 0.793), identity), expressed in the pelvis; the wrist entries move with the arm"),
        recorded_images=recorded, raw_head_fisheye=raw, mujoco_cameras=all_cams, imu=imu)
    OUT.write_text(json.dumps(out, indent=1))
    # cross-check with a recording's own image spec
    if RECORDING.exists():
        ref = json.loads(RECORDING.read_text())["images"]
        for key, spec in ref.items():
            mine = recorded[key]
            for k in ("width", "height", "fx", "fy", "cx", "cy", "f", "baseline_mm", "fov_diag_deg"):
                if k in spec: assert abs(float(spec[k]) - float(mine[k])) < 1e-6, (key, k, spec[k], mine[k])
        print(f"[cams] intrinsics equal to {RECORDING.relative_to(ROOT)} for {len(ref)} streams")
    print(f"[cams] wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.1f} kB): {len(recorded)} recorded streams, {len(all_cams)} MuJoCo cameras; URDF frames checked: "
          + ", ".join(k["optical_frame"]["urdf_link"] for k in recorded.values()) + ", head_stereo_left/right_optical_frame")


if __name__ == "__main__":
    main()
