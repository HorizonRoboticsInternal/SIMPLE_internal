from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R

SMPL_PARENTS_24 = np.array(
    [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21],
    dtype=np.int32,
)

GMR_LR_SWAP_PAIRS = [
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_foot", "right_foot"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
]


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(floor))
    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5,
        :,
    ].reshape(batch_dim + (4,))


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    orig_shape = axis_angle.shape[:-1]
    aa = axis_angle.reshape(-1, 3)

    theta = torch.linalg.norm(aa, dim=-1, keepdim=True)
    axis = aa / torch.clamp(theta, min=1e-8)

    x = axis[:, 0]
    y = axis[:, 1]
    z = axis[:, 2]
    zeros = torch.zeros_like(x)

    k_mat = torch.stack(
        [
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ],
        dim=-1,
    ).reshape(-1, 3, 3)

    eye = torch.eye(3, dtype=aa.dtype, device=aa.device).unsqueeze(0).expand(aa.shape[0], -1, -1)
    sin_theta = torch.sin(theta).unsqueeze(-1)
    cos_theta = torch.cos(theta).unsqueeze(-1)
    axis_outer = axis.unsqueeze(-1) @ axis.unsqueeze(-2)

    small = (theta.squeeze(-1) < 1e-8).unsqueeze(-1).unsqueeze(-1)
    rot = cos_theta * eye + (1.0 - cos_theta) * axis_outer + sin_theta * k_mat
    rot = torch.where(small, eye, rot)
    return rot.reshape(orig_shape + (3, 3))


class HumanoidBatchV2:
    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.smpl_24_parents = [
            -1, 0, 0, 0, 1, 2, 3,
            4, 5, 6, 7, 8, 9, 9,
            9, 12, 13, 14, 16, 17,
            18, 19, 20, 21,
        ]

    @staticmethod
    def _relative_link_position(joints_world: torch.Tensor, root_pos: torch.Tensor) -> torch.Tensor:
        return joints_world - root_pos.unsqueeze(0)

    def _relative_link_pose(self, full_pose_aa: torch.Tensor) -> torch.Tensor:
        joint_count = full_pose_aa.shape[0]
        if joint_count != len(self.smpl_24_parents):
            raise ValueError(f"Joint count mismatch: {joint_count} vs {len(self.smpl_24_parents)}")

        rotation_local = axis_angle_to_matrix(full_pose_aa)
        rotation_global = torch.empty_like(rotation_local)
        for joint_idx in range(joint_count):
            parent = self.smpl_24_parents[joint_idx]
            if parent == -1:
                rotation_global[joint_idx] = rotation_local[joint_idx]
            else:
                rotation_global[joint_idx] = rotation_global[parent] @ rotation_local[joint_idx]
        return rotation_global

    def step_per_frame(
        self,
        full_pose_aa: torch.Tensor,
        root_pos: torch.Tensor,
        joints: torch.Tensor,
    ) -> SimpleNamespace:
        global_joints_position = joints
        global_joints2root_pos = self._relative_link_position(joints[1:, :], root_pos)
        global_joints_rotation_mat = self._relative_link_pose(full_pose_aa)
        return SimpleNamespace(
            global_joints2root_pos=global_joints2root_pos,
            global_joints_rotation_mat=global_joints_rotation_mat,
            global_joints_position=global_joints_position,
        )


@dataclass
class PicoToSmplConfig:
    quat_scalar_first: bool = False
    apply_global_y_180: bool = True
    apply_root_rx90: bool = True
    root_align_degrees: float = 90.0
    root_align_axis: str = "x"


def body_poses_to_smpl_pose_trans(
    body_poses: np.ndarray,
    parents: np.ndarray = SMPL_PARENTS_24,
    cfg: PicoToSmplConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if cfg is None:
        cfg = PicoToSmplConfig()

    body_poses = np.asarray(body_poses, dtype=np.float32)
    if body_poses.shape != (24, 7):
        raise ValueError(f"body_poses shape must be (24,7), got {body_poses.shape}")

    positions = body_poses[:, 0:3].astype(np.float32)
    qx, qy, qz, qw = body_poses[:, 3], body_poses[:, 4], body_poses[:, 5], body_poses[:, 6]
    global_quats_sfirst = np.stack([qw, qx, qy, qz], axis=1).astype(np.float32)
    global_rots = R.from_quat(global_quats_sfirst, scalar_first=True)

    if cfg.apply_global_y_180:
        global_rots = global_rots * R.from_euler("y", 180.0, degrees=True)

    local_rots = []
    for i in range(24):
        parent = int(parents[i])
        if parent == -1:
            local_rots.append(global_rots[i])
        else:
            local_rots.append(global_rots[parent].inv() * global_rots[i])

    pose_aa_24x3 = np.stack([rot.as_rotvec() for rot in local_rots], axis=0).astype(np.float32)
    trans = positions[0].astype(np.float32)

    if cfg.apply_root_rx90:
        rot_align = R.from_euler(cfg.root_align_axis, cfg.root_align_degrees, degrees=True).as_matrix().astype(
            np.float32
        )
        root_matrix = R.from_rotvec(pose_aa_24x3[0]).as_matrix().astype(np.float32)
        pose_aa_24x3[0] = R.from_matrix(rot_align @ root_matrix).as_rotvec().astype(np.float32)
        trans = (rot_align @ trans.reshape(3, 1)).reshape(3).astype(np.float32)

    return pose_aa_24x3, trans


def _mirror_matrix(mirror_axis: str) -> np.ndarray:
    if mirror_axis == "x":
        return np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
    if mirror_axis == "y":
        return np.diag([1.0, -1.0, 1.0]).astype(np.float32)
    if mirror_axis == "z":
        return np.diag([1.0, 1.0, -1.0]).astype(np.float32)
    raise ValueError(f"mirror_axis must be one of x/y/z, got {mirror_axis}")


def safe_normalize_quat_wxyz(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32).reshape(4,)
    norm = float(np.linalg.norm(q))
    if norm < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def mirror_pos_and_quat_wxyz(pos: np.ndarray, quat_wxyz: np.ndarray, mirror_axis: str) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(pos, dtype=np.float32).reshape(3,)
    q = safe_normalize_quat_wxyz(quat_wxyz)
    mirror = _mirror_matrix(mirror_axis)

    pos_m = (mirror @ pos).astype(np.float32)
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)
    rot_m = R.from_quat(q_xyzw).as_matrix().astype(np.float32)
    rot_m = (mirror @ rot_m @ mirror).astype(np.float32)
    q_m_xyzw = R.from_matrix(rot_m).as_quat().astype(np.float32)
    quat_m_wxyz = np.array([q_m_xyzw[3], q_m_xyzw[0], q_m_xyzw[1], q_m_xyzw[2]], dtype=np.float32)
    return pos_m, safe_normalize_quat_wxyz(quat_m_wxyz)


def mirror_and_swap_gmr_input(gmr_input: dict[str, Any], mirror_axis: str = "x") -> dict[str, Any]:
    mirrored: dict[str, Any] = {}
    for key, (pos, quat) in gmr_input.items():
        mirrored[key] = mirror_pos_and_quat_wxyz(pos, quat, mirror_axis)

    out = dict(mirrored)
    for a, b in GMR_LR_SWAP_PAIRS:
        if a in out and b in out:
            out[a], out[b] = out[b], out[a]
    return out


class PicoSmplGmrRetargeter:
    def __init__(
        self,
        smpl_asset_dir: str,
        src_human: str = "smplx",
        tgt_robot: str = "unitree_g1",
        device: str = "cpu",
        mirror_pose: bool = False,
        mirror_axis: str = "x",
        gmr_verbose: bool = False,
    ) -> None:
        from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting as GMR
        from smpl_sim.smpllib.smpl_parser import SMPL_Parser

        self.device = device
        self.gmr = GMR(src_human=src_human, tgt_robot=tgt_robot, verbose=gmr_verbose)
        self.smpl_parser = SMPL_Parser(model_path=smpl_asset_dir, gender="neutral")
        if hasattr(self.smpl_parser, "to"):
            self.smpl_parser = self.smpl_parser.to(self.device)

        self.betas = torch.zeros(1, 10, device=self.device)
        self.humanoid_fk = HumanoidBatchV2(torch.device(self.device))
        self.prev_dof_pos: np.ndarray | None = None
        self.lasttime: float | None = None
        self.height_offset: float | None = None
        self.mirror_pose = mirror_pose
        self.mirror_axis = mirror_axis

    def body_poses_to_gmr_input(self, body_poses: np.ndarray) -> dict[str, Any]:
        pose_aa, trans = body_poses_to_smpl_pose_trans(
            body_poses,
            cfg=PicoToSmplConfig(
                apply_global_y_180=True,
                apply_root_rx90=True,
                root_align_axis="x",
                root_align_degrees=90.0,
            ),
        )
        return self.smpl_pose_trans_to_gmr_input(pose_aa, trans)

    def smpl_pose_trans_to_gmr_input(self, smpl_pose_aa: np.ndarray, smpl_trans: np.ndarray) -> dict[str, Any]:
        if not isinstance(smpl_pose_aa, torch.Tensor):
            smpl_pose_aa = torch.tensor(smpl_pose_aa, dtype=torch.float32)
        if not isinstance(smpl_trans, torch.Tensor):
            smpl_trans = torch.tensor(smpl_trans, dtype=torch.float32)

        pose = smpl_pose_aa.to(self.device, dtype=torch.float32)
        trans = smpl_trans.to(self.device, dtype=torch.float32)
        if pose.ndim == 2:
            pose = pose.unsqueeze(0)
        if trans.ndim == 1:
            trans = trans.unsqueeze(0)

        verts, joints = self.smpl_parser.get_joints_verts(pose, self.betas, trans)
        if self.height_offset is None:
            self.height_offset = float(verts[0, :, 2].min().item())
        joints[..., 2] -= self.height_offset

        pose = pose.squeeze(0)
        trans = trans.squeeze(0)
        joints = joints.squeeze(0)
        motion_state = self.humanoid_fk.step_per_frame(pose, trans, joints)

        global_joints_position = motion_state.global_joints_position
        global_joints_rotation_mat = motion_state.global_joints_rotation_mat
        global_joints_qua_wxyz = matrix_to_quaternion(global_joints_rotation_mat)

        smpl_to_gmr = {
            "pelvis": 0,
            "spine3": 9,
            "left_hip": 1,
            "right_hip": 2,
            "left_knee": 4,
            "right_knee": 5,
            "left_foot": 10,
            "right_foot": 11,
            "left_shoulder": 16,
            "right_shoulder": 17,
            "left_elbow": 18,
            "right_elbow": 19,
            "left_wrist": 20,
            "right_wrist": 21,
        }

        gmr_input_data: dict[str, Any] = {}
        for name, idx in smpl_to_gmr.items():
            pos = global_joints_position[idx].detach().cpu().numpy()
            quat = global_joints_qua_wxyz[idx].detach().cpu().numpy()
            gmr_input_data[name] = (pos, quat)

        if self.mirror_pose:
            gmr_input_data = mirror_and_swap_gmr_input(gmr_input_data, mirror_axis=self.mirror_axis)

        return gmr_input_data

    def retarget(self, body_poses: np.ndarray, now: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        gmr_input_data = self.body_poses_to_gmr_input(body_poses)
        qpos = np.asarray(self.gmr.retarget(gmr_input_data), dtype=np.float32)
        if qpos.shape[0] < 8:
            raise ValueError(f"GMR qpos must include root pose and dof positions, got {qpos.shape}")

        root_pos = qpos[:3].astype(np.float32)
        root_rot = qpos[3:7].astype(np.float32)
        dof_pos = qpos[7:].astype(np.float32)
        if dof_pos.shape[0] != 29:
            raise ValueError(f"Expected 29 dof positions, got {dof_pos.shape[0]}")

        delta_time = 1.0 / 50.0 if self.lasttime is None else (now - self.lasttime)
        self.lasttime = now
        if self.prev_dof_pos is None:
            dof_vel = np.zeros_like(dof_pos, dtype=np.float32)
        else:
            dof_vel = (dof_pos - self.prev_dof_pos) / max(delta_time, 1e-6)
        self.prev_dof_pos = dof_pos.copy()

        latest_obs = np.concatenate([dof_pos, dof_vel, root_pos, root_rot], axis=0).astype(np.float32)
        if latest_obs.shape[0] != 65:
            raise ValueError(f"latest_obs must have 65 values, got {latest_obs.shape}")
        if np.isnan(latest_obs).any():
            raise ValueError("NaN detected in motion reference")

        return dof_pos, dof_vel.astype(np.float32), root_pos, root_rot, latest_obs
