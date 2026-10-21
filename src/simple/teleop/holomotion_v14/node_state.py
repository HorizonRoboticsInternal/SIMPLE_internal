"""Snapshot and restore of the HoloMotion v1.4 policy node's internal state (2026-10-16).

Everything in the node that is not a function of the current tick's inputs: the reference queue (the 10 future frames,
the current frame, counters), the policy runtime state (mode, enabled flags, last ONNX action, publish double buffer,
button-edge memory, velocity command), the velocity policy's 8-step observation history, the motion policy's KV cache and
RoPE step, the motion yaw alignment and FK flags, the PICO stick filter and the node's receive-time bookkeeping.

``snapshot(node)`` is taken at the start of a control tick, before the tick's low state, controller sample and reference
frames are fed. ``restore(fresh_node, snap)`` puts a freshly built node (sim_node.build_sim_policy_node) into exactly that
state; fed the same per-tick inputs on the same clock, it reproduces the live ONNX inputs and outputs bit for bit from
that tick on. The recorder keeps the snapshot of the tick a recording starts on (frame 0) in the episode's exact log, so
a recording replays from its first frame whether it started cold (motion tracking engaged on frame 0, the
--record-on-motion flow) or warm (walking, or mid-motion). The vendored node code is not modified.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np

PREFIX = "holomotion_start."       # key prefix of a packed snapshot (exact-log npz arrays)
VERSION = 1

_RT_SCALARS = ("policy_enabled", "robot_state_ready", "current_policy_mode", "motion_uses_vr_reference", "motion_frame_idx",
               "motion_step_idx", "motion_in_progress", "motion_reference_frozen", "current_motion_clip_index", "vx", "vy",
               "vyaw", "target_dof_pos_real_buffer_idx", "last_vr_status_log_time", "vr_queue_ready_logged",
               "vr_fk_started_logged", "vr_cold_start_logged", "policy_slow_count")
_RT_ARRAYS = ("actions_onnx", "target_dof_pos_onnx", "motion_filtered_actions_onnx")
_VR_SCALARS = ("received", "last_obs_time", "seen_frames", "latest_frame_idx", "latest_source_stamp_sec", "prev_frame_idx",
               "prev_source_stamp_sec")
_VR_ARRAYS = ("latest_obs", "prev_dof_pos", "prev_dof_vel", "prev_root_pos", "prev_root_rot", "dof_pos_queue",
              "dof_vel_queue", "root_pos_queue", "root_rot_queue", "frame_idx_queue", "source_stamp_queue")
_EV_SCALARS = ("_motion_ref_yaw_alignment_ready", "_use_fk_vr", "_fk_vr_cache_ready")
# the low-state cache is keyed by message identity (cache_lowstate: refreshed with force=True once per policy step, else
# only for a new message object); the sim node feeds the same object every tick, so between two policy steps the
# evaluator still reads the PREVIOUS tick's IMU / joints (the yaw alignment at the motion engage does): kept as state
_EV_ARRAYS = ("_motion_ref_yaw_alignment_quat_wxyz", "_fk_root_rot_seq_np", "_fk_root_pos_seq_np",
              "_robot_root_quat_wxyz_buffer", "_robot_root_ang_vel_buffer", "_real_dof_pos_buffer", "_real_dof_vel_buffer")
_PICO = ("lx", "ly", "rx", "ry", "lx_prev", "ly_prev", "rx_prev", "ry_prev", "button")
_NODE_SCALARS = ("_pico_enable_velocity", "_pico_switch_to_motion", "_pico_last_receive_time", "_lowstate_last_receive_time",
                 "_motion_reference_last_receive_time", "_latest_reference_timestamp")


def _py(v: Any) -> Any:
    """JSON-safe python value."""
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return _py(v.tolist())
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    return v


def _put(arrays: dict, prefix: str, obj: Any, names: tuple) -> None:
    for k in names:
        v = getattr(obj, k, None)
        if v is not None:
            arrays[prefix + k] = np.array(v, copy=True)


def node_time() -> float:
    """The clock the node reads (time.time(); the ControllerClock when one is installed)."""
    import humanoid_policy.policy_node_29dof as P
    return float(P.time.time())


def snapshot(node) -> dict[str, np.ndarray]:
    """The node's state right now, as a flat {name: array} dict (scalars and flags in 'meta_json')."""
    st, ev = node.runtime.state, node.observation_evaluator
    vr = node._vr_reference
    arrays: dict[str, np.ndarray] = {}
    now = node_time()
    meta: dict[str, Any] = {"version": VERSION, "clock_now": now}
    rt = {k: _py(getattr(st, k)) for k in _RT_SCALARS}
    rt["last_button_states"] = {str(k): int(v) for k, v in st.last_button_states.items()}
    bufs = st.target_dof_pos_real_buffers
    rt["target_dof_pos_real_alias"] = next((i for i, b in enumerate(bufs) if st.target_dof_pos_real is b), None)
    meta["runtime"] = rt
    for i, b in enumerate(bufs):
        arrays[f"rt.target_dof_pos_real_buffers.{i}"] = np.array(b, copy=True)
    if rt["target_dof_pos_real_alias"] is None and st.target_dof_pos_real is not None:    # a hold target
        arrays["rt.target_dof_pos_real"] = np.array(st.target_dof_pos_real, copy=True)
    _put(arrays, "rt.", st, _RT_ARRAYS)
    meta["vr"] = None if vr is None else {k: _py(getattr(vr, k)) for k in _VR_SCALARS}
    if vr is not None:
        _put(arrays, "vr.", vr, _VR_ARRAYS)
    meta["evaluator"] = {k: _py(getattr(ev, k, None)) for k in _EV_SCALARS}
    meta["evaluator"]["lowstate_cache_warm"] = getattr(ev, "_lowstate_cache_msg", None) is not None
    _put(arrays, "ev.", ev, _EV_ARRAYS)
    if getattr(node, "use_kv_cache", False) and node.motion_kv_cache is not None:
        arrays["kv.motion_kv_cache"] = np.array(node.motion_kv_cache, copy=True)
    meta["pico_controller"] = {k: _py(getattr(node.pico_controller, k)) for k in _PICO}
    meta["node"] = {k: _py(getattr(node, k, None)) for k in _NODE_SCALARS}
    builders = {}
    for mode, b in (("velocity", node.velocity_obs_builder), ("motion", node.motion_obs_builder)):
        if getattr(b, "_obs_buffer", None) is None:        # never built: a fresh builder's first call lays it out, as live
            continue
        builders[mode] = dict(slices=[[str(n), int(s), int(e), bool(h)] for n, s, e, h in b._obs_slices],
                              dim=int(b._obs_buffer.shape[0]),
                              buffers={n: dict(pointer=int(c._pointer), num_pushes=int(c._num_pushes)) for n, c in b._buffers.items()})
        arrays[f"obs.{mode}.buffer"] = np.array(b._obs_buffer, copy=True)
        for n, c in b._buffers.items():
            arrays[f"hist.{mode}.{n}"] = np.array(c._buffer, copy=True)
    meta["obs_builders"] = builders
    arrays["meta_json"] = np.array(json.dumps(meta, sort_keys=True))
    arrays["clock_now"] = np.float64(now)
    return arrays


def clock_now(snap: dict[str, np.ndarray]) -> float:
    return float(snap["clock_now"])


def fresh_start(node) -> None:
    """The node as after re-launching the robot stack: runtime state, observation builders (unbuilt: no history, none of
    setup's warm-up pushes), VR queue, evaluator state, KV cache, PICO filter. What the teleop agent does at every scene
    reset; ``restore`` starts from it, so a node built by build_sim_policy_node matches the live one before the snapshot."""
    from simple.teleop.holomotion_v14 import ensure_vendor_path
    ensure_vendor_path()
    from humanoid_policy.policy_runtime import PolicyRuntimeState
    from humanoid_policy.utils.remote_controller_filter import RemoteController
    node.runtime.state = PolicyRuntimeState(num_actions=node.num_actions)
    node._init_obs_buffers()
    node.observation_evaluator.initialize_observation_state()
    for fn in ("clear_motion_yaw_alignment", "clear_vr_fk_cache"):
        if hasattr(node.observation_evaluator, fn):
            getattr(node.observation_evaluator, fn)()
    if node.motion_kv_cache is not None:
        node.motion_kv_cache.fill(0)
    node.pico_controller = RemoteController()
    node._pico_enable_velocity = node._pico_switch_to_motion = False
    node._pico_last_receive_time = None


def restore(node, snap: dict[str, np.ndarray]) -> None:
    """Put a (freshly built) node into the snapshot's state."""
    from simple.teleop.holomotion_v14 import ensure_vendor_path
    ensure_vendor_path()
    from holomotion_policy_core.obs_builder import _CircularBuffer, _RuntimeTerm
    from humanoid_policy.vr_reference import VrLatestObsReference

    meta = json.loads(str(snap["meta_json"]))
    if int(meta.get("version", 0)) != VERSION:
        raise RuntimeError(f"node_state snapshot version {meta.get('version')} != {VERSION}")
    fresh_start(node)
    st, ev = node.runtime.state, node.observation_evaluator
    # ---- policy runtime state
    R = meta["runtime"]
    for k in _RT_SCALARS:
        setattr(st, k, R[k])
    st.last_button_states = {k: int(v) for k, v in R["last_button_states"].items()}
    for i in range(len(st.target_dof_pos_real_buffers)):
        np.copyto(st.target_dof_pos_real_buffers[i], snap[f"rt.target_dof_pos_real_buffers.{i}"])
    st.target_dof_pos_real_buffer = st.target_dof_pos_real_buffers[int(st.target_dof_pos_real_buffer_idx)]
    alias = R["target_dof_pos_real_alias"]
    if alias is not None:
        st.target_dof_pos_real = st.target_dof_pos_real_buffers[int(alias)]
    else:
        st.target_dof_pos_real = snap["rt.target_dof_pos_real"].copy() if "rt.target_dof_pos_real" in snap else None
    st.actions_onnx = snap["rt.actions_onnx"].copy()
    st.target_dof_pos_onnx = snap["rt.target_dof_pos_onnx"].copy()
    st.motion_filtered_actions_onnx = (snap["rt.motion_filtered_actions_onnx"].copy()
                                       if "rt.motion_filtered_actions_onnx" in snap else None)
    # ---- reference queue
    V = meta["vr"]
    if V is not None:
        vr = node._vr_reference
        if vr is None:
            vr = node._vr_reference = VrLatestObsReference(n_fut_frames=int(node.n_fut_frames), num_actions=node.num_actions,
                                                           expected_dim=node.latest_obs_expected_dim)
        for k in _VR_SCALARS:
            setattr(vr, k, V[k])
        for k in _VR_ARRAYS:
            setattr(vr, k, snap[f"vr.{k}"].copy() if f"vr.{k}" in snap else None)
    # ---- observation evaluator (yaw alignment, FK flags; the FK caches themselves are rebuilt every motion tick)
    E = meta["evaluator"]
    for k in _EV_SCALARS:
        if E.get(k) is not None:
            setattr(ev, k, E[k])
    for k in _EV_ARRAYS:
        key = f"ev.{k}"
        if key in snap:
            cur = getattr(ev, k, None)
            if isinstance(cur, np.ndarray) and cur.shape == snap[key].shape:
                np.copyto(cur, snap[key])             # keeps the tensor views on the same memory
            else:
                setattr(ev, k, snap[key].copy())
    if E.get("lowstate_cache_warm"):
        # the cache held the previous tick's low state live; the sim node feeds one message object every tick, so marking
        # it as the cached one makes the non-forced reads before the next policy step see the restored buffers, as live
        ev._lowstate_cache_msg = getattr(node, "_sim_lowstate", None) or node._lowstate_msg
    # ---- motion KV cache
    if "kv.motion_kv_cache" in snap and node.motion_kv_cache is not None:
        np.copyto(node.motion_kv_cache, snap["kv.motion_kv_cache"])
    # ---- PICO stick filter, node bookkeeping
    for k, v in meta["pico_controller"].items():
        setattr(node.pico_controller, k, list(v) if isinstance(v, list) else v)
    for k, v in meta["node"].items():
        setattr(node, k, v)
    # ---- observation builders: lay a never-built builder out exactly as its first build_policy_obs() would, then the history
    for mode, b in (("velocity", node.velocity_obs_builder), ("motion", node.motion_obs_builder)):
        B = meta["obs_builders"].get(mode)
        if B is None:
            continue
        if getattr(b, "_obs_buffer", None) is None:
            hist = {name: int(hl) for name, _s, hl in b._term_entries}
            scales = {name: float(sc) for name, sc, _h in b._term_entries}
            b._buffers, b._history_names, b._obs_slices, b._runtime_terms = {}, [], [], []
            for name, s, e, has_hist in B["slices"]:
                feat = (e - s) // hist[name] if has_hist else (e - s)
                sc = scales[name]
                hb = None
                if has_hist:
                    hb = _CircularBuffer(hist[name], feat)
                    b._buffers[name] = hb
                    b._history_names.append(name)
                b._obs_slices.append((name, s, e, has_hist))
                b._runtime_terms.append(_RuntimeTerm(name=name, getter=b._resolve_term_getter(name), scale=sc,
                                                     scale_buffer=np.zeros(feat, dtype=np.float32) if sc != 1.0 else None,
                                                     history_buffer=hb, start=s, end=e))
            b._obs_buffer = np.zeros(int(B["dim"]), dtype=np.float32)
            b._obs_batch_buffer = b._obs_buffer.reshape(1, -1)
        np.copyto(b._obs_buffer, snap[f"obs.{mode}.buffer"])
        for name, c in b._buffers.items():
            info = B["buffers"][name]
            c._pointer, c._num_pushes = int(info["pointer"]), int(info["num_pushes"])
            np.copyto(c._buffer, snap[f"hist.{mode}.{name}"])
            if c._num_pushes:
                c.flatten_oldest_first()


def summary(snap: dict[str, np.ndarray], ran_motion_on_frame0: bool | None = None) -> dict[str, Any]:
    """A small JSON summary (episodes.jsonl / exact-log info): what frame 0 started from."""
    meta = json.loads(str(snap["meta_json"]))
    R, V = meta["runtime"], meta["vr"] or {}
    out = dict(version=meta["version"], clock_now=meta["clock_now"], policy_mode=R["current_policy_mode"],
               policy_enabled=bool(R["policy_enabled"]), motion_step_idx=int(R["motion_step_idx"]),
               seen_frames=V.get("seen_frames"), current_frame_index=V.get("prev_frame_idx"),
               queue_frame_index=snap["vr.frame_idx_queue"].astype(int).tolist() if "vr.frame_idx_queue" in snap else None,
               obs_history=sorted(meta["obs_builders"]))
    if ran_motion_on_frame0 is not None:
        # cold: the motion policy ran on frame 0 with its state reset on that very tick (engaged on frame 0)
        out["cold_start"] = bool(ran_motion_on_frame0) and not (R["current_policy_mode"] == "motion" and R["policy_enabled"])
    return out


def pack(snap: dict[str, np.ndarray] | None, prefix: str = PREFIX) -> dict[str, np.ndarray]:
    return {} if snap is None else {prefix + k: v for k, v in snap.items()}


def unpack(arrays: dict[str, Any], prefix: str = PREFIX) -> dict[str, np.ndarray] | None:
    out = {k[len(prefix):]: np.asarray(v) for k, v in arrays.items() if isinstance(k, str) and k.startswith(prefix)}
    return out or None
