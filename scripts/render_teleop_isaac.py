"""Re-render the ego_view of HoloMotion v14 sim-teleop episodes in Isaac, from their bit-exact replay logs.

    MUJOCO_GL=egl .venv/bin/python scripts/render_teleop_isaac.py <dataset> --out <dir> [--episodes 0,1] [--compare]

<dataset> is a teleop LeRobot dir with replay/ (e.g. .../G1WholebodyBottleBinTeleop-v0/level-3_pinhole_realformat). For each
episode the scene is rebuilt through SIMPLE in mujoco_isaac mode from the recorded setup and DR state (as
simple.cli.replay_holomotion_v14 does for MuJoCo), then every dataset frame is shown to Isaac: the MuJoCo state before that
frame's action (the log's state0, then its per-frame physics state, the exact states the recording saw) is set, Isaac is
synced to it, and an Isaac camera identical to the teleop's hbvcam_left_pinhole camera (the HBVCAM's rectified left eye:
1280 x 720, f and principal point from P1 of the stereo calibration, MuJoCo's clip planes, its pose read from MuJoCo every
frame) is rendered. The robot in the image is the recording's MuJoCo robot (the backpack G1 of the teleop MJCF: its meshes,
primitives and colours posed from MuJoCo every frame, MujocoRobot); SIMPLE's stock Isaac G1 is hidden.
Writes <out>/episode_XXXXXX.mp4 (H.264, yuv420p, the dataset fps -- a drop-in for observation.images.ego_view) and, with
--compare, <out>/episode_XXXXXX_compare.mp4 (the dataset's current video | the Isaac render).

Look: --fresh-look keeps the recorded layout, objects and room but re-samples the materials and lights with seed
1000 + episode instead of reusing the recording's; --light bright|dim sets <SCENE>_LIGHT_LEVEL (the scene's calibrated
bright / dim light ranges, the 70 % / 30 % rule; read when the task is built, so one level per process); --hide furniture
hides the HSSD room's furniture in Isaac and --hide walls everything of the room but its floor (the teleop ran in MuJoCo,
without the room); --room replay uses the room (and pose) of the kit's Isaac replay datasets and eval scenes instead of the
recorded one.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

# The HSSD room (and its pose) of each kit's Isaac replay datasets (data/replay_datasets/<kit>/*/replay_meta.json,
# env_config.dr_state_dict.scene), so --room replay puts the teleop renders in the same room as that data. The teleop episodes
# saved SIMPLE's level-3 room (hssd:scene0). Isaac-only: MuJoCo has no room, the physics and model hash are unchanged.
# The same room and pose as the kit's level-3 eval scenes (data/evals_scenes/G1WholebodyBottleBinTeleop-v0/dr-level-3).
REPLAY_ROOMS = {"bottle_bin": {"uid": "hssd:scene2", "center_offset": [0.15, 0.0, 0.0], "center_orientation": [90.0, 0.0, 90.0]}}
# <SCENE>_LIGHT_GAIN per light level with --hide walls (the floor-only room): the scene's own gains (calibrated on its 640x360
# head camera with the walls up) x 3, so the HBVCAM image keeps the scene's brightness targets without the walls' bounce light
# (2026-09-30, episode 0, mean frame luma: bright 112 at 3.9, target 110-115; dim 25 at 1.25 -> 73 at 5.0, ~59 at 3.75, target 55-65).
FLOOR_ONLY_LIGHT_GAIN = {"bottle_bin": {"bright": 3.9, "dim": 3.75}}
# <SCENE>_LIGHT_GAIN per light level with --room replay and the room untouched (as the action-replay data renders it): the
# scene's gains were calibrated on the replay's head camera, which looks down at the lit table; the HBVCAM view takes in more
# of the room, so x ~2.1 keeps the same targets (2026-09-30, episode 0, mean frame luma: bright 72 at 1.3 -> 110 at 2.7;
# dim 58 at 2.6; targets 110-115 / 55-65).
REPLAY_ROOM_LIGHT_GAIN = {"bottle_bin": {"bright": 2.7, "dim": 2.6}}


def build_env(info: dict, hide_groups: tuple[str, ...] = ()):
    """simple.cli.replay_holomotion_v14._build_env, in mujoco_isaac mode (plus Isaac-only hidden HSSD room groups); not reset."""
    import gymnasium as gym
    from simple.cli.teleop_holomotion_v14 import _load_sonic_config, load_scene
    from simple.teleop.holomotion_v14.robot_variants import REPO_ROOT, backpack_mjcf, teleop_mjcf, tilt_head_sensor

    for k, v in info.get("env_knobs", {}).items():
        os.environ[k] = v
    if info.get("scene_root") and "HOLOBRAIN_SIM_DIR" not in os.environ and os.path.isdir(info["scene_root"]):
        os.environ["HOLOBRAIN_SIM_DIR"] = info["scene_root"]
    mod = load_scene(info["scene"])
    env = gym.make(info["env_id"], sim_mode="mujoco_isaac", render_hz=info["render_hz"], physics_dt=info["physics_dt"],
                   headless=True, max_episode_steps=10 ** 9, sonic_config=_load_sonic_config(),
                   target=getattr(mod, "TARGET", None), dr_level=info["dr_level"], success_criteria=0.9)
    se = env.unwrapped
    if os.environ.get(f"{info['scene'].upper()}_LIGHT_LEVEL"):
        # SIMPLE's DR manager puts lighting in "fixed" mode at every level > 0 (a 2 x 4 grid at a constant 6001): give the
        # randomizer back the task's own config (the scene's bright / dim ranges). Isaac-only; MuJoCo physics is unchanged.
        from copy import deepcopy
        se.task.dr.randomizers["lighting"].cfg = deepcopy(se.task.dr_cfgs["lighting"])
    if hide_groups:                          # visual only: MuJoCo has no room, physics and stored states are untouched
        se.task.isaac_hidden_scene_groups = tuple(hide_groups)
    robot = se.task.robot
    tilt = float(info.get("head_tilt_deg", 0.0))
    if "robot" in info:
        robot.mjcf_path = teleop_mjcf(info["robot"], info.get("backpack_kg", 0.0), tilt_deg=tilt)
    if tilt:
        tilt_head_sensor(se.task, tilt)
    elif info.get("backpack_kg", 0) > 0:
        robot.mjcf_path = backpack_mjcf(info["backpack_kg"])
    if info.get("robot_mjcf") and robot.mjcf_path != info["robot_mjcf"] and (REPO_ROOT / "data" / info["robot_mjcf"]).exists():
        robot.mjcf_path = info["robot_mjcf"]
    return env, se, mod


def reset_episode(env, se, mod, info: dict, setup: dict, seed: int | None = None, room: dict | None = None) -> None:
    """The recorded episode's scene (its setup, then SIMPLE's saved DR state). With a seed the materials and lights are not
    taken from the recording: they are re-sampled with that seed (the lighting config follows <SCENE>_LIGHT_LEVEL). With a
    room (uid, center_offset, center_orientation) the HSSD room is that one instead of the recorded one (Isaac-only)."""
    import random
    from simple.teleop.holomotion_v14.scene_setup import SceneSetups
    setups = SceneSetups(info["scene"], mod, se.task)
    setups.apply(setup if setup.get("randomized") else setups.nominal())
    state = json.loads(info["environment_config"])
    if room:
        state["dr_state_dict"]["scene"].update(room)
    if seed is None:
        env.reset(options={"state_dict": state})
        return
    for k in ("material", "lighting"):
        state["dr_state_dict"].pop(k, None)
    random.seed(seed)
    np.random.seed(seed)
    env.reset(seed=seed, options={"state_dict": state})


def floor_only_room(room_uid: str) -> dict:
    """--hide walls: keep only the HSSD room's floor. Its openings and ceilings are hidden, and every mesh of its walls group
    keeps only its up-facing faces at floor height (hssd:scene2 merges the floor into the wall meshes, so hiding the walls
    group would take the floor with it). Faces, normals and face-varying / uniform primvars are subset together. Isaac-only;
    idempotent (the room prim is kept across resets)."""
    import omni.usd
    from pxr import Usd, UsdGeom, Vt
    st = omni.usd.get_context().get_stage()
    scene_path = f"/World/scene/s_{room_uid.replace(':', '_')}"                 # IsaacSimEngine: SCENE_PRIM_PATH/s_<uid>
    scene = st.GetPrimAtPath(scene_path)
    if not scene.IsValid():
        raise RuntimeError(f"no room prim at {scene_path}")
    walls = st.GetPrimAtPath(f"{scene_path}/walls")
    for q in Usd.PrimRange(walls, Usd.TraverseInstanceProxies()):           # instanced pieces cannot be edited: un-instance them
        if q.IsInstance():
            q.SetInstanceable(False)
    xc = UsdGeom.XformCache()
    to_room = np.asarray(xc.GetLocalToWorldTransform(scene).GetInverse(), dtype=np.float64)      # world -> the room's own Y-up frame
    out = {"hidden_meshes": 0, "floor_faces": 0, "floor_m2": 0.0}
    for g in ("openings", "ceilings"):
        p = st.GetPrimAtPath(f"{scene_path}/{g}")
        if p.IsValid():
            UsdGeom.Imageable(p).MakeInvisible()
    for p in Usd.PrimRange(walls):
        if not p.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(p)
        pts = np.asarray(m.GetPointsAttr().Get(), dtype=np.float64)
        cnt = np.asarray(m.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if len(cnt) == 0:
            continue
        M = np.asarray(xc.GetLocalToWorldTransform(p), dtype=np.float64) @ to_room              # row vectors: local -> room
        w = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3]
        starts = np.r_[0, np.cumsum(cnt)[:-1]]
        a, b, c = w[idx[starts]], w[idx[starts + 1]], w[idx[starts + 2]]
        n = np.cross(b - a, c - a)
        area = 0.5 * np.linalg.norm(n, axis=1)
        ny = n[:, 1] / (2 * area + 1e-12) * (-1 if m.GetOrientationAttr().Get() == "leftHanded" else 1)
        keep = (np.maximum.reduceat(w[idx, 1], starts) < 0.02) & (ny > 0.9)
        if not keep.any():
            UsdGeom.Imageable(p).MakeInvisible()
            out["hidden_meshes"] += 1
            continue
        out["floor_faces"] += int(keep.sum())
        out["floor_m2"] += float(area[keep].sum())
        if keep.all():                                                   # trimmed by an earlier episode of this process
            continue
        fv = np.repeat(keep, cnt)                                        # per face-vertex
        m.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(cnt[keep].astype(np.int32)))
        m.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(idx[fv].astype(np.int32)))
        nattr = m.GetNormalsAttr()
        if nattr.HasValue():
            nv = np.asarray(nattr.Get(), dtype=np.float32)
            interp = m.GetNormalsInterpolation()
            if interp == UsdGeom.Tokens.faceVarying:
                nattr.Set(Vt.Vec3fArray.FromNumpy(nv[fv]))
            elif interp == UsdGeom.Tokens.uniform:
                nattr.Set(Vt.Vec3fArray.FromNumpy(nv[keep]))
        for pv in UsdGeom.PrimvarsAPI(p).GetPrimvars():
            interp = pv.GetInterpolation()
            if interp not in (UsdGeom.Tokens.faceVarying, UsdGeom.Tokens.uniform):
                continue
            sel = fv if interp == UsdGeom.Tokens.faceVarying else keep
            if pv.IsIndexed():
                pv.SetIndices(Vt.IntArray.FromNumpy(np.asarray(pv.GetIndices(), dtype=np.int32)[sel]))
            else:
                vals = pv.Get()
                pv.Set(type(vals)(list(np.asarray(vals)[sel].tolist())) if not hasattr(type(vals), "FromNumpy")
                       else type(vals).FromNumpy(np.asarray(vals)[sel]))
    out["floor_m2"] = round(out["floor_m2"], 1)
    return out


from simple.teleop.holomotion_v14.isaac_views import MujocoRobot, PinholeCamera  # noqa: E402  (moved 2026-10-01)


def encoder(path: Path, w: int, h: int, fps: float, crf: int):
    import imageio_ffmpeg
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "medium",
                             "-crf", str(crf), "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", default="", help="comma-separated indices (default: all)")
    ap.add_argument("--warmup", type=int, default=24, help="Isaac steps on the first frame before recording (renderer settle)")
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--compare", action="store_true", help="also write <ep>_compare.mp4: the dataset's video | Isaac")
    ap.add_argument("--hide", default="", help="comma-separated HSSD room parts to hide in Isaac: furniture (or any other top-level "
                    "room group) and walls (walls, openings and ceilings; the floor faces kept) -- the teleop ran without the room")
    ap.add_argument("--room", default="recorded", choices=("recorded", "replay"),
                    help="replay: the room and room pose of the kit's Isaac replay datasets (REPLAY_ROOMS)")
    ap.add_argument("--fresh-look", action="store_true", help="re-sample materials and lights with seed 1000 + episode")
    ap.add_argument("--light", default="", choices=("", "bright", "dim"), help="<SCENE>_LIGHT_LEVEL for this process")
    ap.add_argument("--light-gain", type=float, default=None, help="<SCENE>_LIGHT_GAIN (default: FLOOR_ONLY_LIGHT_GAIN with "
                    "--hide walls, REPLAY_ROOM_LIGHT_GAIN with --room replay, else the scene's own)")
    ap.add_argument("--skip-existing", action="store_true", help="skip episodes whose <out>/episode_XXXXXX.json exists (resume)")
    a = ap.parse_args()

    import imageio_ffmpeg
    import mujoco
    from simple.teleop.holomotion_v14 import exact_log as XL
    from simple.teleop.holomotion_v14.robot_variants import PINHOLE_CAMERA

    info_ds = json.loads((a.dataset / "meta/info.json").read_text())
    fps = float(info_ds["fps"])
    episodes = [json.loads(l) for l in (a.dataset / "meta/episodes.jsonl").read_text().splitlines()]
    wanted = {int(x) for x in a.episodes.split(",") if x}
    episodes = [e for e in episodes if not wanted or e["episode_index"] in wanted]
    if a.skip_existing:
        episodes = [e for e in episodes if not (a.out / f"episode_{e['episode_index']:06d}.json").exists()]
    a.out.mkdir(parents=True, exist_ok=True)
    chunks = int(info_ds.get("chunks_size", 1000))
    env = se = mod = cam = first = None
    for e in episodes:
        ep = e["episode_index"]
        t0 = time.time()
        log = XL.load(a.dataset / e["exact_log"])
        info, setup = log["info"], log["setup"]
        if env is None:
            if a.light:
                os.environ[f"{info['scene'].upper()}_LIGHT_LEVEL"] = a.light
                gain = a.light_gain
                if gain is None and "walls" in a.hide.split(","):
                    gain = FLOOR_ONLY_LIGHT_GAIN[info["scene"]][a.light]
                elif gain is None and a.room == "replay":
                    gain = REPLAY_ROOM_LIGHT_GAIN[info["scene"]][a.light]
                if gain is not None:
                    os.environ[f"{info['scene'].upper()}_LIGHT_GAIN"] = str(gain)
            hide = [g for g in a.hide.split(",") if g]
            env, se, mod = build_env(info, tuple(g for g in hide if g != "walls"))
            room = REPLAY_ROOMS[info["scene"]] if a.room == "replay" else None
            first = info
        elif info.get("scene") != first.get("scene") or info.get("backpack_kg") != first.get("backpack_kg"):
            sys.exit(f"episode {ep}: different scene/robot from the first episode; render them separately")
        seed = 1000 + ep if a.fresh_look else None
        reset_episode(env, se, mod, info, setup, seed, room)
        floor = floor_only_room(se.task.dr.state_dict()["scene"]["uid"]) if "walls" in hide else None
        m, d = se.mujoco.mjModel, se.mujoco.mjData
        same_model = XL.model_sha256(m) == info["model_sha256"]
        if log["physics"].shape[1] != m.nq + m.nv:
            print(f"[isaac-render] episode {ep}: SKIPPED, the rebuilt model's state size differs from the log", flush=True)
            continue
        drs = se.task.dr.state_dict()
        look = {"seed": seed, "light_level": a.light or None, "light_gain": os.environ.get(f"{info['scene'].upper()}_LIGHT_GAIN"), "room": (drs.get("scene") or {}).get("uid"), "floor_only": floor,
                "table_material": ((drs.get("material") or {}).get("table_material") or {}).get("name"),
                "ground_material": ((drs.get("material") or {}).get("ground_material") or {}).get("name"),
                "lights": len(drs.get("lighting") or {}),
                "light_intensity": [round(float(v.get("light_intensity", 0))) for v in (drs.get("lighting") or {}).values()][:6]}
        if cam is None or not cam.cam.prim.IsValid():
            cam = PinholeCamera()
        cam.clip_like(m)
        body = MujocoRobot(se)                    # rebuilt per episode: the compiled model (geom ids) follows the layout
        n = len(log["frame_calls"])
        out = a.out / f"episode_{ep:06d}.mp4"
        enc = encoder(out, cam.W, cam.H, fps, a.crf)
        blank = 0
        for t in range(n):
            if t == 0:
                XL.set_state(m, d, log["state0"], XL.INTEGRATION)
            else:
                XL.set_state(m, d, log["physics"][t - 1], XL.PHYSICS)
            mujoco.mj_forward(m, d)
            cam.follow(m, d, PINHOLE_CAMERA)
            body.update(d)
            for _ in range(a.warmup if t == 0 else 1):
                se.isaac.step(se.mujoco)
            img = cam.rgb()
            if img is None or img.shape[:2] != (cam.H, cam.W):
                img = np.zeros((cam.H, cam.W, 3), np.uint8)
                blank += 1
            enc.stdin.write(img.tobytes())
        enc.stdin.close()
        enc.wait()
        line = {"episode": ep, "frames": n, "blank_frames": blank, "model_identical": bool(same_model),
                "seconds": round(time.time() - t0, 1), "video": str(out), "robot": f"mujoco ({len(body.geoms)} geoms)", **look}
        if a.compare:
            vkey = next(k for k, v in info_ds["features"].items() if v.get("dtype") == "video")
            orig = a.dataset / info_ds["video_path"].format(episode_chunk=ep // chunks, video_key=vkey, episode_index=ep)
            cmp = a.out / f"episode_{ep:06d}_compare.mp4"
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", "-i", str(orig), "-i", str(out),
                            "-filter_complex", "[0:v][1:v]hstack=inputs=2", "-c:v", "libx264", "-crf", "20",
                            "-pix_fmt", "yuv420p", str(cmp)], check=True)
            line["compare"] = str(cmp)
        if blank == 0:                   # the done-marker: an interrupted episode has none and is rendered again on resume
            (a.out / f"episode_{ep:06d}.json").write_text(json.dumps(line))
        print("[isaac-render] " + json.dumps(line), flush=True)
    sys.stdout.flush()
    os._exit(0)                           # Isaac's shutdown segfaults; everything is written by now


if __name__ == "__main__":
    main()
