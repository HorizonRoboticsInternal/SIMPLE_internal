"""reference_actions (11 x 65) check (2026-10-19): (a) the column equals the window rebuilt from the frame-0 snapshot + the
recorded frames at every step; (b) a fresh node, restored at frame 0 and fed the column's 11 rows every 20 ms (nothing else
from MotionReference), reproduces the recorded model input and output bit for bit.
    MUJOCO_GL=egl .venv/bin/python reference_actions_column_test.py <dataset dir> [episode]"""
import json, os, sys, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from simple.teleop.holomotion_v14 import BACKPACK_MODELS_DIR
os.environ.setdefault("HOLOMOTION_UNITREE_POLICY_MODELS_DIR", str(BACKPACK_MODELS_DIR))
from simple.teleop.holomotion_v14.sim_node import build_sim_policy_node
from simple.teleop.holomotion_v14.sim_clock import ControllerClock
from simple.teleop.holomotion_v14 import exact_log as XL, node_state as NS
from simple.agents.holomotion_v14_agent import VRSTATE_FIELDS
D = sys.argv[1]; ep = int(sys.argv[2]) if len(sys.argv) > 2 else 0
t = pq.read_table(f"{D}/data/chunk-000/episode_{ep:06d}.parquet"); n = t.num_rows
col = lambda c: np.stack(t[c].to_pandas().values).reshape(n, -1)
W = col("reference_actions").reshape(n, 11, 65); WI = col("reference_actions_frame_index")
F = col("holomotion.MotionReference.frames").reshape(n, 16, 65); M = col("holomotion.MotionReference.frames_meta").reshape(n, 16, 2); C = np.minimum(col("holomotion.MotionReference.frames_this_tick")[:, 0].astype(int), 16)
q, dq, quat, gyro = col("holomotion.LowState.motor_state.q"), col("holomotion.LowState.motor_state.dq"), col("holomotion.LowState.imu_state.quaternion"), col("holomotion.LowState.imu_state.gyroscope")
vl, vr = col("holomotion.VRState.left_controller"), col("holomotion.VRState.right_controller")
pobs, pdim, ran, aout, tick = col("holomotion.policy_obs"), col("holomotion.policy_obs_dim")[:, 0].astype(int), col("holomotion.policy_ran")[:, 0].astype(int), col("holomotion.actions_onnx"), col("holomotion.tick_time")[:, 0]
log = XL.load(f"{D}/replay/episode_{ep:06d}.npz"); snap = NS.unpack(log)
# (a) rebuild the window from the snapshot queue + frames
stream = [np.concatenate([snap["vr.prev_dof_pos"], snap["vr.prev_dof_vel"], snap["vr.prev_root_pos"], snap["vr.prev_root_rot"]])] + [np.concatenate([snap[k][r] for k in ("vr.dof_pos_queue", "vr.dof_vel_queue", "vr.root_pos_queue", "vr.root_rot_queue")]) for r in range(10)]
same = 0
for i in range(n):
    stream += [F[i, j] for j in range(C[i])]
    same += np.array_equal(np.array(stream[-11:], dtype=np.float32), W[i])
print(f"[reference_actions] (a) column == rebuilt window on {same}/{n} steps")
# (b) feed the column's 11 rows every step
def sample(i):
    L = dict(zip(VRSTATE_FIELDS, vl[i])); R = dict(zip(VRSTATE_FIELDS, vr[i]))
    if np.isnan(vl[i][0]): return None
    return dict(l_active=int(L["status"]), r_active=int(R["status"]), A=int(R["a_button"]), B=int(R["b_button"]), X=int(L["x_button"]), Y=int(L["y_button"]), left_axis=[float(L["axis_x"]), float(L["axis_y"])], right_axis=[float(R["axis_x"]), float(R["axis_y"])], left_trigger=float(L["trigger"]), right_trigger=float(R["trigger"]), left_grip=float(L["grip"]), right_grip=float(R["grip"]), L3=int(L["axis_click"]), R3=int(R["axis_click"]), left_menu=int(L["menu_button"]), right_menu=int(R["menu_button"]), button_bits=0)
node = build_sim_policy_node(quiet=True); clock = ControllerClock(t0=float(tick[0])); clock.install()
node.set_robot_state("MOVE_TO_DEFAULT"); node.set_robot_state("POLICY"); NS.restore(node, snap)
ok = 0; tot = 0
for i in range(n):
    clock.now = float(tick[i]); node.feed_lowstate(q[i], dq[i], quat[i], gyro[i]); s = sample(i)
    if s is not None: node.feed_pico(s)
    if WI[i, 0] >= 0:
        for r in range(11): node.feed_reference(W[i, r], int(WI[i, r]), 0)       # the whole window, every 20 ms
    node.last_policy_io = None; node.run(); io = node.last_policy_io
    if ran[i]:
        tot += 1; ok += io is not None and np.array_equal(io["obs"], pobs[i, :pdim[i]].astype(np.float32)) and np.array_equal(io["out"], aout[i].astype(np.float32))
print(f"[reference_actions] (b) fed the window column only: model input+output bit-exact on {ok}/{tot} policy steps"); sys.stdout.flush(); os._exit(0)
