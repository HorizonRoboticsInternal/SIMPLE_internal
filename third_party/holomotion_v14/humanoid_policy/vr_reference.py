"""VR/latest_obs reference queue state for the 29DOF policy node."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


LATEST_OBS_DIM = 65
REFERENCE_DOF_DIM = 29


@dataclass
class VrLatestObsReference:
    """Maintain Phase 3B latest_obs current/previous/future queues."""

    n_fut_frames: int
    num_actions: int = REFERENCE_DOF_DIM
    expected_dim: int = LATEST_OBS_DIM

    def __post_init__(self) -> None:
        self.n_fut_frames = max(int(self.n_fut_frames), 0)
        self.num_actions = int(self.num_actions)
        self.expected_dim = int(self.expected_dim)

        self.latest_obs: np.ndarray | None = None
        self.received = False
        self.last_obs_time: float | None = None
        self.seen_frames = 0
        self.latest_frame_idx = -1
        self.latest_source_stamp_sec: float | None = None

        self.prev_dof_pos: np.ndarray | None = None
        self.prev_dof_vel: np.ndarray | None = None
        self.prev_root_pos: np.ndarray | None = None
        self.prev_root_rot: np.ndarray | None = None
        self.prev_frame_idx: int | None = None
        self.prev_source_stamp_sec: float | None = None

        if self.n_fut_frames > 0:
            self.dof_pos_queue = np.zeros(
                (self.n_fut_frames, self.num_actions),
                dtype=np.float32,
            )
            self.dof_vel_queue = np.zeros(
                (self.n_fut_frames, self.num_actions),
                dtype=np.float32,
            )
            self.root_pos_queue = np.zeros((self.n_fut_frames, 3), dtype=np.float32)
            self.root_rot_queue = np.zeros((self.n_fut_frames, 4), dtype=np.float32)
            self.frame_idx_queue = np.full((self.n_fut_frames,), -1, dtype=np.int64)
            self.source_stamp_queue = np.full(
                (self.n_fut_frames,),
                np.nan,
                dtype=np.float64,
            )
        else:
            self.dof_pos_queue = None
            self.dof_vel_queue = None
            self.root_pos_queue = None
            self.root_rot_queue = None
            self.frame_idx_queue = None
            self.source_stamp_queue = None

    def is_ready_for_motion(self, enable_teleop_reference: bool, delay_frames: int) -> bool:
        if not bool(enable_teleop_reference):
            return False
        if not (self.received and self.latest_obs is not None):
            return False
        if self.n_fut_frames <= 0:
            return True
        needed = self.n_fut_frames + max(int(delay_frames), 0) + 1
        return int(self.seen_frames) >= needed

    @property
    def has_latest_obs(self) -> bool:
        return self.received and self.latest_obs is not None

    def data_age(self, current_time: float) -> float:
        if self.last_obs_time is None:
            return float("inf")
        return float(current_time) - self.last_obs_time

    def has_future_sequence(self, n_frames: int | None = None) -> bool:
        n_frames = self._future_count(n_frames)
        if n_frames <= 0:
            return False
        return (
            self.dof_pos_queue is not None
            and self.dof_vel_queue is not None
            and self.root_pos_queue is not None
            and self.root_rot_queue is not None
            and self.dof_pos_queue.shape[0] >= n_frames
            and self.dof_vel_queue.shape[0] >= n_frames
            and self.root_pos_queue.shape[0] >= n_frames
            and self.root_rot_queue.shape[0] >= n_frames
        )

    def _future_count(self, n_frames: int | None = None) -> int:
        if n_frames is None:
            return self.n_fut_frames
        return max(int(n_frames), 0)

    @staticmethod
    def _coerce_frame_index(frame_index: int | None) -> int:
        try:
            return int(frame_index) if frame_index is not None else -1
        except Exception:
            return -1

    @staticmethod
    def _coerce_stamp_sec(stamp_sec: float | None) -> float | None:
        try:
            stamp = float(stamp_sec) if stamp_sec is not None else float("nan")
        except Exception:
            return None
        if not np.isfinite(stamp) or stamp <= 0.0:
            return None
        return stamp

    def store(
        self,
        arr: np.ndarray,
        *,
        current_time: float,
        frame_index: int | None = None,
        source_stamp_sec: float | None = None,
    ) -> bool:
        """Store one latest_obs packet and advance future queues."""
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.shape[1] < self.expected_dim:
            return False

        latest_frame_idx = self._coerce_frame_index(frame_index)
        latest_source_stamp_sec = self._coerce_stamp_sec(source_stamp_sec)

        clipped = arr[:, : self.expected_dim].astype(np.float32, copy=False)
        self.latest_obs = clipped
        self.received = True
        self.last_obs_time = float(current_time)
        self.latest_frame_idx = latest_frame_idx
        self.latest_source_stamp_sec = latest_source_stamp_sec
        self.seen_frames += 1

        if self.n_fut_frames <= 0 or self.dof_pos_queue is None:
            return True

        latest_root_pos = clipped[0, 58:61]
        latest_root_rot = clipped[0, 61:65]
        latest_dof_pos = clipped[0, : self.num_actions]
        latest_dof_vel = clipped[0, self.num_actions : 2 * self.num_actions]

        if self.prev_dof_pos is None:
            self.prev_dof_pos = np.empty_like(self.dof_pos_queue[0])
            self.prev_dof_vel = np.empty_like(self.dof_vel_queue[0])
            self.prev_root_pos = np.empty_like(self.root_pos_queue[0])
            if self.root_rot_queue is not None:
                self.prev_root_rot = np.empty_like(self.root_rot_queue[0])

        np.copyto(self.prev_dof_pos, self.dof_pos_queue[0])
        np.copyto(self.prev_dof_vel, self.dof_vel_queue[0])
        np.copyto(self.prev_root_pos, self.root_pos_queue[0])
        if self.root_rot_queue is not None:
            np.copyto(self.prev_root_rot, self.root_rot_queue[0])
        if self.frame_idx_queue is not None:
            try:
                self.prev_frame_idx = int(self.frame_idx_queue[0])
            except Exception:
                self.prev_frame_idx = -1
        if self.source_stamp_queue is not None:
            self.prev_source_stamp_sec = self._coerce_stamp_sec(
                float(self.source_stamp_queue[0])
            )

        self.dof_pos_queue[:-1] = self.dof_pos_queue[1:]
        self.dof_pos_queue[-1] = latest_dof_pos
        self.dof_vel_queue[:-1] = self.dof_vel_queue[1:]
        self.dof_vel_queue[-1] = latest_dof_vel
        self.root_pos_queue[:-1] = self.root_pos_queue[1:]
        self.root_pos_queue[-1] = latest_root_pos
        if self.root_rot_queue is not None:
            self.root_rot_queue[:-1] = self.root_rot_queue[1:]
            self.root_rot_queue[-1] = latest_root_rot
        if self.frame_idx_queue is not None:
            self.frame_idx_queue[:-1] = self.frame_idx_queue[1:]
            self.frame_idx_queue[-1] = latest_frame_idx
        if self.source_stamp_queue is not None:
            self.source_stamp_queue[:-1] = self.source_stamp_queue[1:]
            self.source_stamp_queue[-1] = (
                latest_source_stamp_sec
                if latest_source_stamp_sec is not None
                else np.nan
            )
        return True

    def freeze_at_current_pose(self) -> bool:
        """Replace the live trajectory with a stationary copy of its current pose."""
        if not self.has_latest_obs:
            return False

        current_dof_pos = np.asarray(self.current_dof_pos(), dtype=np.float32).copy()
        current_root_pos = np.asarray(self.current_root_pos(), dtype=np.float32).copy()
        current_root_rot = self.current_root_rot()
        if current_root_rot is None:
            return False
        current_root_rot = np.asarray(current_root_rot, dtype=np.float32).copy()
        current_frame_index = self.current_frame_index()
        current_source_stamp_sec = self.current_source_stamp_sec()

        self.latest_obs[0, : self.num_actions] = current_dof_pos
        self.latest_obs[0, self.num_actions : 2 * self.num_actions] = 0.0
        self.latest_obs[0, 2 * self.num_actions : 2 * self.num_actions + 3] = (
            current_root_pos
        )
        self.latest_obs[0, 2 * self.num_actions + 3 : 2 * self.num_actions + 7] = (
            current_root_rot
        )
        self.latest_frame_idx = current_frame_index
        self.latest_source_stamp_sec = current_source_stamp_sec

        if self.n_fut_frames <= 0 or self.dof_pos_queue is None:
            return True

        self.prev_dof_pos = current_dof_pos.copy()
        self.prev_dof_vel = np.zeros(self.num_actions, dtype=np.float32)
        self.prev_root_pos = current_root_pos.copy()
        self.prev_root_rot = current_root_rot.copy()
        self.prev_frame_idx = current_frame_index
        self.prev_source_stamp_sec = current_source_stamp_sec

        self.dof_pos_queue[:] = current_dof_pos
        self.dof_vel_queue.fill(0.0)
        self.root_pos_queue[:] = current_root_pos
        self.root_rot_queue[:] = current_root_rot
        self.frame_idx_queue.fill(current_frame_index)
        if current_source_stamp_sec is None:
            self.source_stamp_queue.fill(np.nan)
        else:
            self.source_stamp_queue.fill(current_source_stamp_sec)
        return True

    def current_frame_index(self) -> int:
        if self.n_fut_frames > 0 and self.frame_idx_queue is not None:
            if self.prev_frame_idx is not None:
                return int(self.prev_frame_idx)
            return int(self.frame_idx_queue[0])
        return int(self.latest_frame_idx)

    def current_source_stamp_sec(self) -> float | None:
        if self.n_fut_frames > 0 and self.source_stamp_queue is not None:
            if self.prev_source_stamp_sec is not None:
                return float(self.prev_source_stamp_sec)
            return self._coerce_stamp_sec(float(self.source_stamp_queue[0]))
        return self.latest_source_stamp_sec

    def future_frame_indices(self, n_frames: int | None = None) -> np.ndarray:
        T = self._future_count(n_frames)
        if T <= 0 or self.frame_idx_queue is None:
            return np.zeros(0, dtype=np.int64)
        return self.frame_idx_queue[:T].astype(np.int64, copy=True)

    def future_source_stamp_secs(self, n_frames: int | None = None) -> np.ndarray:
        T = self._future_count(n_frames)
        if T <= 0 or self.source_stamp_queue is None:
            return np.zeros(0, dtype=np.float64)
        return self.source_stamp_queue[:T].astype(np.float64, copy=True)

    def latest_dof_pos(self) -> np.ndarray | None:
        if self.latest_obs is None:
            return None
        return self.latest_obs[0, : self.num_actions]

    def latest_root_pos(self) -> np.ndarray | None:
        if self.latest_obs is None:
            return None
        return self.latest_obs[0, 58:61].astype(np.float32)

    def latest_root_rot(self) -> np.ndarray | None:
        if self.latest_obs is None:
            return None
        return self.latest_obs[0, 61:65].astype(np.float32)

    def current_dof_pos(self, offline_fallback: np.ndarray | None = None) -> np.ndarray:
        if self.n_fut_frames > 0 and self.dof_pos_queue is not None:
            if self.prev_dof_pos is not None:
                return self.prev_dof_pos
            return self.dof_pos_queue[0]
        if self.latest_obs is None:
            if offline_fallback is None:
                return np.zeros(self.num_actions, dtype=np.float32)
            return offline_fallback
        return self.latest_obs[0, : self.num_actions]

    def current_dof_vel(self, offline_fallback: np.ndarray | None = None) -> np.ndarray:
        if self.n_fut_frames > 0 and self.dof_vel_queue is not None:
            if self.prev_dof_vel is not None:
                return self.prev_dof_vel
            return self.dof_vel_queue[0]
        if self.latest_obs is None:
            if offline_fallback is None:
                return np.zeros(self.num_actions, dtype=np.float32)
            return offline_fallback
        return self.latest_obs[0, self.num_actions : 2 * self.num_actions]

    def current_root_pos(self) -> np.ndarray:
        if self.n_fut_frames > 0 and self.root_pos_queue is not None:
            if self.prev_root_pos is not None:
                return self.prev_root_pos.astype(np.float32)
            return self.root_pos_queue[0].astype(np.float32)
        if self.latest_obs is None:
            return np.zeros(3, dtype=np.float32)
        return self.latest_obs[0, 58:61].astype(np.float32)

    def current_root_rot(self) -> np.ndarray | None:
        if self.prev_root_rot is not None:
            return self.prev_root_rot
        if self.root_rot_queue is not None:
            return self.root_rot_queue[0].astype(np.float32)
        if self.latest_obs is None:
            return None
        return self.latest_obs[0, 61:65].astype(np.float32)

    def obs_ref_dof_pos_fut(
        self,
        *,
        ref_to_onnx: np.ndarray,
        pos_fut_buffer: np.ndarray,
        n_frames: int | None = None,
    ) -> np.ndarray:
        T = self._future_count(n_frames)
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if not self.has_future_sequence(T):
            return np.zeros(self.num_actions * T, dtype=np.float32)
        pos_fut_buffer[:, :T] = self.dof_pos_queue[:T].T
        pos_fut_onnx = pos_fut_buffer[ref_to_onnx, :T].transpose(1, 0)
        return pos_fut_onnx.reshape(-1).astype(np.float32)

    def obs_ref_root_height_fut(self, n_frames: int | None = None) -> np.ndarray:
        T = self._future_count(n_frames)
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if not self.has_future_sequence(T):
            return np.zeros(T, dtype=np.float32)
        root_pos_fut = self.root_pos_queue[:T, 2].astype(np.float32)
        return root_pos_fut.reshape(-1)

    def obs_ref_root_pos_fut(self, n_frames: int | None = None) -> np.ndarray:
        T = self._future_count(n_frames)
        if T <= 0:
            return np.zeros(0, dtype=np.float32)
        if not self.has_future_sequence(T):
            return np.zeros(3 * T, dtype=np.float32)
        return self.root_pos_queue[:T].astype(np.float32).reshape(-1)

    def copy_fk_sequence_inputs(
        self,
        *,
        root_pos_seq: np.ndarray,
        root_rot_seq: np.ndarray,
        dof_pos_seq: np.ndarray,
        cur_root_pos: np.ndarray,
        cur_root_rot: np.ndarray,
        cur_dof_pos: np.ndarray,
        n_frames: int | None = None,
    ) -> bool:
        T = self._future_count(n_frames)
        if T <= 0 or not self.has_future_sequence(T):
            return False
        np.copyto(root_pos_seq[0, 0], cur_root_pos)
        np.copyto(root_rot_seq[0, 0], cur_root_rot)
        np.copyto(dof_pos_seq[0, 0], cur_dof_pos)
        np.copyto(root_pos_seq[0, 1 : 1 + T], self.root_pos_queue[:T])
        np.copyto(root_rot_seq[0, 1 : 1 + T], self.root_rot_queue[:T])
        np.copyto(dof_pos_seq[0, 1 : 1 + T], self.dof_pos_queue[:T])
        return True
