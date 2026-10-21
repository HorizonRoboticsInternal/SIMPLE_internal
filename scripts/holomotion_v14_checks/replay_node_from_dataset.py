"""Re-run the HoloMotion v1.4 policy node from a recording's raw inputs ONLY (holomotion.LowState.*, holomotion.VRState.*,
holomotion.MotionReference.frames) and compare, tick by tick, its ONNX input / output with what the recording stored
(holomotion.policy_obs, holomotion.actions_onnx). Answers: can the controller's inputs be replayed bit-exactly from the data?

2026-10-16 cold start: a fresh node is first put into the recording's frame-0 start state (replay/episode_XXXXXX.npz,
holomotion_start.* arrays, node_state.restore) and runs on the recorded node clock (holomotion.tick_time), so the
comparison starts at frame 0. --no-snapshot skips the restore (the earlier behaviour: exact only from the first motion tick).

    MUJOCO_GL=egl .venv/bin/python replay_node_from_dataset.py <lerobot dir> [--episode N] [--no-snapshot] [--out result.json]"""
import argparse, json, os, sys
import numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from simple.teleop.holomotion_v14 import BACKPACK_MODELS_DIR
os.environ.setdefault("HOLOMOTION_UNITREE_POLICY_MODELS_DIR", str(BACKPACK_MODELS_DIR))
from simple.teleop.holomotion_v14.sim_node import build_sim_policy_node
from simple.teleop.holomotion_v14.sim_clock import ControllerClock
from simple.teleop.holomotion_v14 import exact_log as XL, node_state as NS
from simple.agents.holomotion_v14_agent import VRSTATE_FIELDS

ap = argparse.ArgumentParser(); ap.add_argument("dataset"); ap.add_argument("--episode", type=int, default=0); ap.add_argument("--no-snapshot", action="store_true")
ap.add_argument("--out", default=""); a = ap.parse_args()
D, ep = a.dataset, a.episode
t = pq.read_table(f"{D}/data/chunk-000/episode_{ep:06d}.parquet"); n = t.num_rows
col = lambda c: np.stack(t[c].to_pandas().values).reshape(n, -1)
q, dq = col("holomotion.LowState.motor_state.q"), col("holomotion.LowState.motor_state.dq"); quat, gyro = col("holomotion.LowState.imu_state.quaternion"), col("holomotion.LowState.imu_state.gyroscope")
vl, vr = col("holomotion.VRState.left_controller"), col("holomotion.VRState.right_controller")
K = col("holomotion.MotionReference.frames").shape[1] // 65
frames, meta, cnt = col("holomotion.MotionReference.frames").reshape(n, K, 65), col("holomotion.MotionReference.frames_meta").reshape(n, K, 2), col("holomotion.MotionReference.frames_this_tick")[:, 0].astype(int)
pobs, pdim, ran, aout = col("holomotion.policy_obs"), col("holomotion.policy_obs_dim")[:, 0].astype(int), col("holomotion.policy_ran")[:, 0].astype(int), col("holomotion.actions_onnx")
tick_time = col("holomotion.tick_time")[:, 0] if "holomotion.tick_time" in t.column_names else None
use_tick_time = tick_time is not None and np.isfinite(tick_time).all()
log_path = f"{D}/replay/episode_{ep:06d}.npz"; snap = None; start_summary = None
if os.path.exists(log_path):
    log = XL.load(log_path); snap = NS.unpack(log); start_summary = log["info"].get("holomotion_start")
use_snapshot = snap is not None and not a.no_snapshot
print(f"[replay-node] {n} ticks, frames kept per tick up to {K}; recorded policy_ran counts {dict(zip(*np.unique(ran, return_counts=True)))}; ticks with more frames than stored: {int((cnt > K).sum())}")
print(f"[replay-node] start snapshot in the exact log: {snap is not None} (used: {use_snapshot}); recorded tick clock: {use_tick_time}; start: {start_summary}")

def sample(i):
    L = dict(zip(VRSTATE_FIELDS, vl[i])); R = dict(zip(VRSTATE_FIELDS, vr[i]))
    if np.isnan(vl[i][0]): return None
    return dict(l_active=int(L["status"]), r_active=int(R["status"]), A=int(R["a_button"]), B=int(R["b_button"]), X=int(L["x_button"]), Y=int(L["y_button"]),
                left_axis=[float(L["axis_x"]), float(L["axis_y"])], right_axis=[float(R["axis_x"]), float(R["axis_y"])], left_trigger=float(L["trigger"]), right_trigger=float(R["trigger"]),
                left_grip=float(L["grip"]), right_grip=float(R["grip"]), L3=int(L["axis_click"]), R3=int(R["axis_click"]), left_menu=int(L["menu_button"]), right_menu=int(R["menu_button"]), button_bits=0)

node = build_sim_policy_node(quiet=True)
t0 = float(tick_time[0]) if use_tick_time else (float(meta[cnt > 0][0, -1, 1]) if (cnt > 0).any() else 1.0e9)
clock = ControllerClock(t0=t0); clock.install()
node.set_robot_state("MOVE_TO_DEFAULT"); node.set_robot_state("POLICY")
if use_snapshot:
    NS.restore(node, snap)
    print(f"[replay-node] node restored to the frame-0 start state: mode {node.runtime.state.current_policy_mode}, enabled {node.runtime.state.policy_enabled}, "
          f"queue frames {node._vr_reference.frame_idx_queue.tolist() if node._vr_reference is not None else None}, current {node._vr_reference.prev_frame_idx if node._vr_reference is not None else None}")
obs_exact = np.zeros(n, bool); out_exact = np.zeros(n, bool); mode_ok = np.zeros(n, bool); dmax = np.full(n, np.nan)
for i in range(n):
    k = min(int(cnt[i]), K)
    if use_tick_time:
        clock.now = float(tick_time[i])                                   # the node's own clock at this tick, as recorded
    elif k:                                                               # old recordings: follow the newest frame's stamp
        clock.now = float(meta[i, k - 1, 1]) + 1e-3
    else:
        clock.now += 0.02
    node.feed_lowstate(q[i], dq[i], quat[i], gyro[i])
    s = sample(i)
    if s is not None: node.feed_pico(s)
    for j in range(k):
        node.feed_reference(frames[i, j], int(meta[i, j, 0]), int(round(meta[i, j, 1] * 1e9)))
    node.last_policy_io = None; node.run(); io = node.last_policy_io
    rec_ran = int(ran[i]); got = 0 if io is None else (2 if io["mode"] == "motion" else 1); mode_ok[i] = got == rec_ran
    if io is not None and rec_ran:
        o = pobs[i, :pdim[i]].astype(np.float32); d = np.abs(io["obs"] - o).max() if io["obs"].shape == o.shape else np.inf
        dmax[i] = d; obs_exact[i] = d == 0.0; out_exact[i] = np.array_equal(io["out"], aout[i].astype(np.float32))
mot = np.where(ran == 2)[0]; vel = np.where(ran == 1)[0]; any_ = np.where(ran > 0)[0]
first_mot = int(mot[0]) if len(mot) else -1
def first_all_exact(idx, flags):
    for s_ in idx:
        if flags[idx[idx >= s_]].all(): return int(s_)
    return -1
ok = obs_exact & out_exact & mode_ok
print(f"[replay-node] mode matched on {int(mode_ok.sum())}/{n} ticks; recording's first motion tick {first_mot}")
print(f"[replay-node] velocity phase ({len(vel)} ticks): obs bit-exact on {int(obs_exact[vel].sum())}, out bit-exact on {int(out_exact[vel].sum())}; max |obs diff| {np.nanmax(dmax[vel]) if len(vel) else 0:.3e}")
print(f"[replay-node] motion phase ({len(mot)} ticks): obs bit-exact on {int(obs_exact[mot].sum())}, out bit-exact on {int(out_exact[mot].sum())}; exact from tick {first_all_exact(mot, ok)} onward; max |obs diff| {np.nanmax(dmax[mot]) if len(mot) else 0:.3e}")
print(f"[replay-node] ALL policy ticks ({len(any_)}): bit-exact (mode, obs, out) on {int(ok[any_].sum())}; exact from tick {first_all_exact(any_, ok)} onward (0 = from frame 0)")
bad = [int(i) for i in any_ if not ok[i]][:10]; print("[replay-node] first non-exact policy ticks:", bad, "| diffs:", [f"{dmax[i]:.2e}" for i in bad])
res = dict(dataset=D, episode=ep, n=n, used_snapshot=bool(use_snapshot), snapshot_present=snap is not None, used_tick_time=bool(use_tick_time), start=start_summary,
           first_motion_tick=first_mot, motion_ticks=len(mot), motion_obs_exact=int(obs_exact[mot].sum()), motion_out_exact=int(out_exact[mot].sum()), motion_exact_from=first_all_exact(mot, ok),
           velocity_ticks=len(vel), velocity_obs_exact=int(obs_exact[vel].sum()), velocity_out_exact=int(out_exact[vel].sum()), velocity_max_diff=float(np.nanmax(dmax[vel])) if len(vel) else 0.0,
           all_policy_ticks=len(any_), all_exact=int(ok[any_].sum()), all_exact_from=first_all_exact(any_, ok), mode_matched=int(mode_ok.sum()), first_bad=bad)
out = a.out or f"{os.path.dirname(os.path.abspath(__file__))}/replay_node_result.json"
json.dump(res, open(out, "w"), indent=1); print("->", out)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0)
