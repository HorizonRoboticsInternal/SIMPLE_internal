"""Isaac views of the HoloMotion v1.4 teleop robot: the HBVCAM rectified-left camera, the backpack G1 drawn from the
compiled MuJoCo geoms (SIMPLE's stock Isaac G1 USD is a different robot), and a fixed third-person camera. Used online by
simple.cli.eval_holomotion_v14 (--sim-mode simple) and offline by scripts/render_teleop_isaac.py."""
from __future__ import annotations

import numpy as np


class PinholeCamera:
    """An Isaac camera matching the MuJoCo hbvcam_left_pinhole camera."""

    def __init__(self) -> None:
        from omni.isaac.sensor import Camera
        from pxr import UsdGeom
        from simple.teleop.holomotion_v14.robot_variants import PINHOLE_H, PINHOLE_W, rectified_left

        _, P1 = rectified_left()
        W, H = PINHOLE_W, PINHOLE_H
        fx, fy, cx, cy = float(P1[0, 0]), float(P1[1, 1]), float(P1[0, 2]), float(P1[1, 2])
        self.cam = Camera(prim_path="/World/hbvcam_left_pinhole", name="hbvcam_left_pinhole", resolution=(W, H))
        self.cam.initialize()
        f = 10.0                                                          # SIMPLE's engine convention (__update_cameras)
        ha, va = f * W / fx, f * H / fy
        self.cam.set_focal_length(f)
        self.cam.set_horizontal_aperture(ha)
        self.cam.set_vertical_aperture(va)
        usd = UsdGeom.Camera(self.cam.prim)
        usd.GetHorizontalApertureOffsetAttr().Set(float((cx - 0.5 * W) * ha / W))
        usd.GetVerticalApertureOffsetAttr().Set(float((cy - 0.5 * H) * va / H))
        self.cam.set_clipping_range(0.01, 50.0)
        self.W, self.H = W, H

    def follow(self, m, d, name: str) -> None:
        """Place it at the MuJoCo camera's world pose (both look along -z with +y up)."""
        import mujoco
        cid = m.camera(name).id
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.asarray(d.cam_xmat[cid], dtype=np.float64))
        self.cam.set_world_pose(np.asarray(d.cam_xpos[cid], dtype=np.float64), q, camera_axes="usd")

    def rgb(self):
        a = self.cam.get_rgba()
        return None if a is None or a.size == 0 else np.ascontiguousarray(a[..., :3], dtype=np.uint8)

    def clip_like(self, m) -> None:
        """MuJoCo's near / far planes for this model (znear, zfar x the model extent)."""
        self.cam.set_clipping_range(float(m.vis.map.znear * m.stat.extent), float(m.vis.map.zfar * m.stat.extent))


class MujocoRobot:
    """The recording's robot drawn in Isaac as MuJoCo draws it: every robot geom of the compiled model that MuJoCo's
    default render options show (groups 0-2, alpha > 0; a visual/collision pair on the same mesh and pose drawn once), with
    its mesh (vertices and normals of the compiled model) or primitive shape and its colour, posed from MuJoCo's
    geom_xpos / geom_xmat every frame. This is the backpack G1 with the HBVCAM head of the teleop MJCF; SIMPLE's own Isaac
    robot (the stock G1 USD, synced by joint angles only) is hidden."""

    ROOT = "/mj_robot"                                        # top level: no /World transform in between

    def __init__(self, se, root_body: str = "pelvis") -> None:
        import mujoco
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

        m = se.mujoco.mjModel
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(self.ROOT):
            stage.RemovePrim(self.ROOT)
        stock = stage.GetPrimAtPath(f"{se.isaac.workspace_prim_path}/Robot")
        if stock.IsValid():
            UsdGeom.Imageable(stock).MakeInvisible()
        root = UsdGeom.Xform.Define(stage, self.ROOT)
        mpu = UsdGeom.GetStageMetersPerUnit(stage) or 1.0
        if abs(mpu - 1.0) > 1e-9:
            root.AddScaleOp().Set(Gf.Vec3f(*(3 * [1.0 / mpu])))

        rb = m.body(root_body).id
        def is_robot(b):
            while b > 0:
                if b == rb:
                    return True
                b = int(m.body_parentid[b])
            return False

        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        T = mujoco.mjtGeom
        looks, seen, self.geoms, self.ops = {}, set(), [], []
        for g in range(m.ngeom):
            if not is_robot(int(m.geom_bodyid[g])) or not opt.geomgroup[min(int(m.geom_group[g]), 5)]:
                continue
            mat = int(m.geom_matid[g])
            rgba = m.geom_rgba[g]
            if mat >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):           # MuJoCo: a non-default geom rgba overrides
                rgba = m.mat_rgba[mat]
            if rgba[3] <= 0:
                continue
            key = (int(m.geom_bodyid[g]), int(m.geom_type[g]), int(m.geom_dataid[g]), *np.round(m.geom_size[g], 6),
                   *np.round(m.geom_pos[g], 6), *np.round(m.geom_quat[g], 6))
            if key in seen:
                continue
            seen.add(key)
            path, typ, s = f"{self.ROOT}/g{g}", int(m.geom_type[g]), m.geom_size[g]
            scale = None
            if typ == T.mjGEOM_MESH:
                mid = int(m.geom_dataid[g])
                va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
                fa, fn = int(m.mesh_faceadr[mid]), int(m.mesh_facenum[mid])
                na = int(m.mesh_normaladr[mid])
                faces = np.asarray(m.mesh_face[fa:fa + fn], dtype=np.int32)
                prim = UsdGeom.Mesh.Define(stage, path)
                prim.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(m.mesh_vert[va:va + vn], dtype=np.float32)))
                prim.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(fn, 3, dtype=np.int32)))
                prim.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.reshape(-1)))
                normals = np.asarray(m.mesh_normal[na + np.asarray(m.mesh_facenormal[fa:fa + fn]).reshape(-1)], dtype=np.float32)
                prim.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals))
                prim.SetNormalsInterpolation(UsdGeom.Tokens.faceVarying)
                prim.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
            elif typ == T.mjGEOM_SPHERE:
                prim = UsdGeom.Sphere.Define(stage, path); prim.CreateRadiusAttr(float(s[0]))
            elif typ == T.mjGEOM_CYLINDER:
                prim = UsdGeom.Cylinder.Define(stage, path); prim.CreateAxisAttr("Z")
                prim.CreateRadiusAttr(float(s[0])); prim.CreateHeightAttr(float(2 * s[1]))
            elif typ == T.mjGEOM_CAPSULE:
                prim = UsdGeom.Capsule.Define(stage, path); prim.CreateAxisAttr("Z")
                prim.CreateRadiusAttr(float(s[0])); prim.CreateHeightAttr(float(2 * s[1]))
            elif typ == T.mjGEOM_BOX:
                prim = UsdGeom.Cube.Define(stage, path); prim.CreateSizeAttr(1.0); scale = 2 * s
            elif typ == T.mjGEOM_ELLIPSOID:
                prim = UsdGeom.Sphere.Define(stage, path); prim.CreateRadiusAttr(1.0); scale = s
            else:
                print(f"[isaac-view] robot geom {g}: type {typ} not drawn", flush=True)
                continue
            x = UsdGeom.Xformable(prim)
            t_op = x.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
            o_op = x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
            if scale is not None:
                x.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*map(float, scale)))
            col = tuple(np.round(rgba[:3], 4))
            if col not in looks:                               # MuJoCo draws rgba as display (sRGB) values: RTX wants linear
                mpath = f"{self.ROOT}/Looks/c{len(looks)}"
                mtl = UsdShade.Material.Define(stage, mpath)
                sh = UsdShade.Shader.Define(stage, f"{mpath}/Shader")
                sh.CreateIdAttr("UsdPreviewSurface")
                sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) ** 2.2 for c in col]))
                sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
                sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
                mtl.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
                looks[col] = mtl
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(looks[col])
            self.geoms.append(g)
            self.ops.append((t_op, o_op))
        self.n_faces = sum(int(m.mesh_facenum[m.geom_dataid[g]]) for g in self.geoms if m.geom_type[g] == T.mjGEOM_MESH)
        print(f"[isaac-view] MuJoCo robot in Isaac: {len(self.geoms)} geoms ({self.n_faces} mesh faces), "
              f"{len(looks)} colours; stock Isaac robot hidden: {stock.IsValid()}", flush=True)

    def update(self, d) -> None:
        import mujoco
        from pxr import Gf
        q = np.zeros(4)
        for g, (t_op, o_op) in zip(self.geoms, self.ops):
            mujoco.mju_mat2Quat(q, np.asarray(d.geom_xmat[g], dtype=np.float64))
            t_op.Set(Gf.Vec3d(*map(float, d.geom_xpos[g])))
            o_op.Set(Gf.Quatd(float(q[0]), float(q[1]), float(q[2]), float(q[3])))


class IsaacFreeCamera:
    """A fixed Isaac camera at a MuJoCo free-camera pose (lookat, distance, azimuth, elevation; MuJoCo's conventions), with
    MuJoCo's default vertical field of view, for the episode video's third-person view."""

    def __init__(self, lookat, distance: float, azimuth_deg: float, elevation_deg: float, size=(640, 360), fovy_deg: float = 45.0,
                 prim_path: str = "/World/third_person_cam") -> None:
        from omni.isaac.sensor import Camera
        W, H = size
        self.cam = Camera(prim_path=prim_path, name=prim_path.rsplit("/", 1)[-1], resolution=(W, H))
        self.cam.initialize()
        f = 10.0
        va = 2.0 * f * np.tan(np.radians(fovy_deg) / 2.0)
        self.cam.set_focal_length(f)
        self.cam.set_vertical_aperture(va)
        self.cam.set_horizontal_aperture(va * W / H)
        self.cam.set_clipping_range(0.05, 60.0)
        self.W, self.H = W, H
        self.place(lookat, distance, azimuth_deg, elevation_deg)

    def place(self, lookat, distance: float, azimuth_deg: float, elevation_deg: float) -> None:
        az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
        forward = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])   # MuJoCo mjvCamera forward
        pos = np.asarray(lookat, dtype=np.float64) - distance * forward
        up_w = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up_w); right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        R = np.stack([right, up, -forward], axis=1)            # USD camera: +x right, +y up, looks along -z
        q = _quat_wxyz(R)
        self.cam.set_world_pose(pos, q, camera_axes="usd")

    def rgb(self):
        a = self.cam.get_rgba()
        return None if a is None or a.size == 0 else np.ascontiguousarray(a[..., :3], dtype=np.uint8)


def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])
