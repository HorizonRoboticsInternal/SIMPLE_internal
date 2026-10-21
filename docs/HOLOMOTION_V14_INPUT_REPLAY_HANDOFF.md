# HoloMotion v1.4 stored-input replay: hand-off

Written 2026-10-18. Workstation `Horizon-G1`, repo `~/wrk/SIMPLE` (branch `alan/Simple`, all of this uncommitted).

## What this is

A **closed-loop replay of a teleop recording through the HoloMotion v1.4 controller**. The policy node (motion model
`model_22000` + the v1.4 velocity model) runs again, live. It is fed only what the recording stored: the PICO controller
sample, the reference frames and the controller clock, every 20 ms. It starts from the episode's saved frame-0
state. The node computes the joint targets itself; MuJoCo steps them. Nothing recorded on the action side (joint
targets, `ctrl`) is copied.

Every frame is checked against the recording. Physics state and the policy's ONNX input/output must match bit for
bit, and the task outcome must match. The replay is written out as a new LeRobot dataset by the teleop recorder's own
code. That is how the 50 grey-hand episodes of 2026-10-18 were re-recorded with black hands: 50 of 50 bit-exact.

Three replay tools exist. Do not mix them up:

| tool | HoloMotion runs? | physics? | use it for |
|---|---|---|---|
| `python -m simple.cli.rerecord_holomotion_v14` | **yes, closed loop** | yes | the stored-input replay; writes a new dataset |
| `python -m simple.cli.replay_holomotion_v14` | no | yes | physics check: replays the logged `ctrl` (`--mode physics`) or the logged joint targets with PD recomputed (`--mode action`) |
| `docs/_scan/holomotion_v14_pipeline_update/replay_node_from_dataset.py` | yes, open loop | no | controller-only check: feeds the recorded LowState too and compares the ONNX I/O |

How much of the PICO sample is needed (tested 2026-10-19 on episode 0): after 0 ms only the grips matter, because
they open and close the hands; sticks, triggers and buttons can be zeroed with no change. A replay with no PICO data
at all, which presses Y itself at 0 ms and takes the hand state from the recorded `action` hand blocks, is also
bit-exact. The PICO columns are needed to replay walking phases, mode switches or stops.

## 1. Model inputs, and where the replay got each one

Left: everything the HoloMotion backpack model takes in. Right: the data the action replay used for it.

- **recording**: read from the original recording.
- **replayed physics**: computed by the replay's own physics, closed loop.
- **controller state**: from the frame-0 snapshot in the recording's replay log, then the controller's own values.

**Motion model (model_22000), input `obs`: 604 values = 18 terms**

| model input | size | what it is | where the replay got it |
|---|---|---|---|
| *reference, current frame* | | | |
| `actor_ref_dof_pos_cur` | 29 | joint positions of the current reference frame | **recording** `holomotion.MotionReference.frames`: dof_pos |
| `actor_ref_root_height_cur` | 1 | root height of the current frame | **recording** `holomotion.MotionReference.frames`: root_pos z |
| `actor_ref_gravity_projection_cur` | 3 | gravity direction seen from the reference root | **recording** `holomotion.MotionReference.frames`: root_rot_wxyz |
| `actor_ref_base_linvel_cur` | 3 | reference root linear velocity, from the window's root positions | **recording** `holomotion.MotionReference.frames`: root_pos |
| `actor_ref_base_angvel_cur` | 3 | reference root angular velocity, from the window's root orientations | **recording** `holomotion.MotionReference.frames`: root_rot_wxyz |
| `actor_ref_robot_yaw_error_sin_cos` | 2 | heading difference between the reference root and the robot | **recording** `holomotion.MotionReference.frames`: root_rot_wxyz · **replayed physics** IMU quaternion · **controller state** yaw alignment |
| *reference, 10 future frames* | | | |
| `actor_ref_dof_pos_fut` | 290 | joint positions of future frames 1 to 10 (10 × 29) | **recording** `holomotion.MotionReference.frames`: dof_pos |
| `actor_ref_root_height_fut` | 10 | root height of each future frame | **recording** `holomotion.MotionReference.frames`: root_pos z |
| `actor_ref_gravity_projection_fut` | 30 | gravity seen from each future root | **recording** `holomotion.MotionReference.frames`: root_rot_wxyz |
| `actor_ref_base_linvel_fut` | 30 | root linear velocity of each future frame | **recording** `holomotion.MotionReference.frames`: root_pos |
| `actor_ref_base_angvel_fut` | 30 | root angular velocity of each future frame | **recording** `holomotion.MotionReference.frames`: root_rot_wxyz |
| `actor_ref_future_yaw_delta_sin_cos` | 20 | heading change of each future frame against the current one | **recording** `holomotion.MotionReference.frames`: root_rot_wxyz |
| `actor_ref_future_root_ori_robot_frame_6d` | 60 | orientation of each future root, in the robot's frame | **recording** `holomotion.MotionReference.frames`: root_rot_wxyz · **replayed physics** IMU quaternion |
| *robot state* | | | |
| `actor_projected_gravity` | 3 | gravity in the robot's body frame | **replayed physics** IMU quaternion (the recorded `holomotion.LowState.imu_state.quaternion` only checks it) |
| `actor_rel_robot_root_ang_vel` | 3 | robot root angular velocity | **replayed physics** IMU gyroscope (recorded `…imu_state.gyroscope` only checks it) |
| `actor_dof_pos` | 29 | robot joint positions | **replayed physics** motor q (recorded `holomotion.LowState.motor_state.q` only checks it) |
| `actor_dof_vel` | 29 | robot joint velocities | **replayed physics** motor dq (recorded `…motor_state.dq` only checks it) |
| `actor_last_action` | 29 | the model's previous output | **controller state** frame 0: `holomotion_start.rt.actions_onnx`; after that, the model's own previous output |

**Motion model, its other two inputs**

| model input | size | what it is | where the replay got it |
|---|---|---|---|
| `past_key_values` | 1×2×1×32×4×64 | the transformer's memory of the previous steps | **controller state** frame 0: `holomotion_start.kv.motion_kv_cache`; after that, the model's own `present_key_values` |
| `step_idx` | 1 | step counter since motion tracking engaged | **controller state** frame 0: in `holomotion_start.meta_json`; then +1 per step |

**Decided by the node around the model**

| what | what it is | where the replay got it |
|---|---|---|
| which model runs | walking model or motion model (A = walking, Y = motion tracking) | **recording** `holomotion.pico`: button bits |
| the 11-frame window | which received frames are current and future 1 to 10 (below) | **recording** `holomotion.MotionReference.frames` (+ `frames_meta`, `frames_this_tick`) · **controller state** frame 0: `holomotion_start.vr.*` |
| reference freshness | the reference is frozen if older than 0.6 s; the controller sample times out after 0.5 s | **recording** `holomotion.tick_time` |

**Walking model, input `obs`: 776 values = 6 terms × 8 steps of history.** It was not used in the 50 episodes, which
ran motion tracking from frame 0 to the end.

| model input | size | where the replay got it |
|---|---|---|
| `actor_velocity_command` | 4 × 8 | **recording** `holomotion.pico`: sticks, through the stick filter (**controller state** filter state at frame 0) |
| `actor_projected_gravity`, `actor_rel_robot_root_ang_vel`, `actor_dof_pos`, `actor_dof_vel` | (3+3+29+29) × 8 | **replayed physics** IMU and motors |
| `actor_last_action` | 29 × 8 | **controller state** history at frame 0: `holomotion_start.hist.velocity.*` |

**Recorded but not a model input**

- MotionReference `dof_vel` (29 per frame) is carried in every reference frame, but no term reads it. Zeroing it
  everywhere left the model's input and output bit-exact on all 916 steps (18.3 s) of episode 0.
- The Dex3 hands open and close from the grips (**recording** `holomotion.pico`), sent to the hands directly, outside HoloMotion.

The robot-state terms are deliberately not taken from the recording. The replay reads them from its own physics, and
they still came out byte-equal to the recorded ones every 20 ms. That is the closed-loop proof.

### The current frame and the 10 future frames

The stream delivers one or two new reference frames every 20 ms, and the node queues them. The frame 10 arrivals behind the
newest is **current**, and the 10 newer ones are **future 1 to 10**; future 10 is the newest frame received. At frame 0
the queue comes from the snapshot; after that, from the frames recorded every 20 ms.

```
received frames, arrival order:  ... | cur (-10) | f1 (-9) | f2 | f3 | f4 | f5 | f6 | f7 | f8 | f9 (-1) | f10 = newest
```

- In the teleop, frames arrive at 70 Hz, so the window spans about 140 ms and current is 140 to 160 ms behind the newest
  frame. In the evaluation, a VLA's frames arrive one every 20 ms: 20 ms apart, 200 ms in total.
- A sender may also feed the whole 11 × 65 window every 20 ms; with a queue of 11 it replaces the window outright.
  Tested: the same model input and output as the recorded single frames, bit for bit, at every 20 ms step of episodes 0 and
  30 (`docs/_scan/holomotion_v14_pipeline_update/replay_node_window_test.py`, `WINDOW=1`).
- Rebuilt from the snapshot queue plus the frames recorded every 20 ms, the window matched the recorded
  `actor_ref_dof_pos_cur` / `_fut` on all 34,228 steps (11.4 minutes) of the 50 episodes.
- `reference_actions` (11 × 65) is this window since 2026-10-19 (block 0 = current, 1–10 = future). In
  datasets recorded before that date it was a look-ahead built from `teleop.latest_obs`, not the policy's window (and 11 × 79);
  the `reference_actions_frame_index` column exists only in the new sets.

## 2. Setup

```bash
cd ~/wrk/SIMPLE
export MUJOCO_GL=egl                 # headless rendering; every command below assumes it
.venv/bin/python -c "import mujoco, simple"   # the repo venv (Python 3.10)
```

- **Scene kits** (bottle_bin, bowl_sink, coffee_cart) live in `~/wrk/robot_orchard_deploy/holobrain_g1_deploy/sim`. Each
  episode's replay log records the `scene_root` it was recorded with, and the rebuild uses it. `HOLOBRAIN_SIM_DIR`
  overrides it.
- **Models** (not in git; never put them in a public repo): `data/holomotion/v14_models_backpack_3p2/`
  - `motion_tracking_model/exported/model_22000.onnx`, sha256 `d79ccce759ede01f23fe49c1e7178c1e216ec754d33a9f05adadc72d86c45f9d`
  - `velocity_tracking_model/exported/velocity_model.onnx`
  - On a new machine, copy them from NAS-28:
    ```bash
    rsync -a /mnt/nas28/alan.jiang/holomotion_models/v14_models_backpack_3p2/ data/holomotion/v14_models_backpack_3p2/
    (cd data/holomotion/v14_models_backpack_3p2 && sha256sum -c SHA256SUMS)
    ```
  - The node finds them through `HOLOMOTION_UNITREE_POLICY_MODELS_DIR`. The replay sets it from the log's
    `info["policy_models_dir"]`.
- **HoloMotion robot code** is vendored unchanged in `third_party/holomotion_v14`. Do not edit it; everything
  sim-specific lives in `src/simple/teleop/holomotion_v14/` and `src/simple/agents/holomotion_v14_agent.py`.

## 3. Which recordings can be replayed from their inputs

Recordings made with `simple.cli.teleop_holomotion_v14` from 2026-10-16 on. A dataset qualifies when it has:

- parquet columns `holomotion.pico`, `holomotion.MotionReference.frames`, `holomotion.MotionReference.frames_meta`,
  `holomotion.MotionReference.frames_this_tick`, `holomotion.tick_time`;
- per episode `replay/episode_XXXXXX.npz` with `state0`, `physics`, `rest_idx` / `rest_val`, `action_band_state`,
  `render_step0`, and the `holomotion_start.*` arrays (the controller snapshot of frame 0);
- since 2026-10-18 also `holomotion_start.agent_json` (the agent's own state at frame 0). Older logs replay with a
  default for it (section 8, trap 6).
- since 2026-10-19 also `reference_actions` (11 × 65 per 20 ms, the v1.3 name; v1.3's blocks were 79 = 65 + 14 hand
  values): the node's window after each step, block 0 = current, blocks 1–10 = future; `reference_actions_frame_index`
  (11) = the frame numbers. Optional for the replay, which rebuilds it from the frames; fed whole every 20 ms it reproduces the model input
  exactly (tested).

```bash
D=data/teleop_holomotion_v14_rot0/simple/G1WholebodyBottleBinTeleop-v0/level-3_pinhole
.venv/bin/python - <<EOF
import json, numpy as np
f = json.load(open("$D/meta/info.json"))["features"]
print({k: k in f for k in ("holomotion.pico", "holomotion.MotionReference.frames", "holomotion.tick_time")})
z = np.load("$D/replay/episode_000000.npz")
print("controller snapshot:", any(k.startswith("holomotion_start.") for k in z.files), "| agent state:", "holomotion_start.agent_json" in z.files)
EOF
```

The September 2026 sets (`data/teleop_holomotion_v14/...`) lack these. They replay only with `replay_holomotion_v14`
(physics/action), not through HoloMotion from frame 0.

## 4. Commands

```bash
SRC=data/teleop_holomotion_v14_rot0/simple/G1WholebodyBottleBinTeleop-v0/level-3_pinhole

# the whole dataset -> a NEW root (the same simple/<env>/<level> folders are made under it; the source is never written)
.venv/bin/python -m simple.cli.rerecord_holomotion_v14 $SRC --save-dir data/teleop_holomotion_v14_rot0_blackhands

# a subset, a quick test (always test 1-2 episodes into a scratch root first)
.venv/bin/python -m simple.cli.rerecord_holomotion_v14 $SRC --save-dir data/tmp_rerecord_test --episodes 0,1
#   --episodes 0,3,5-9      --no-black-hands (keep the grey hands)      --stop-on-mismatch
```

A per-episode line is printed, and `meta/rerecord_summary.json` is rewritten after every episode:

```
[rerecord] ep 0 -> 0: 916 frames, physics BIT-EXACT, policy I/O BIT-EXACT, task SUCCESS at the last frame (19 s/episode)
[rerecord] ep 30 -> 30: 933 frames, physics BIT-EXACT, policy I/O BIT-EXACT, task no success gates {...}, source saved by hand (left menu + A)
[rerecord] done: 50/50 bit-exact, 46/50 succeed on their last frame -> .../level-3_pinhole
```

It runs on CPU at about 9 to 19 s per episode, one process, so 50 episodes take about 8 minutes. For long runs use
`nohup ... > log 2>&1 &` and watch the log.

Verify the result independently:

```bash
NEW=data/teleop_holomotion_v14_rot0_blackhands/simple/G1WholebodyBottleBinTeleop-v0/level-3_pinhole
.venv/bin/python docs/_scan/teleop_runbook/compare_rerecord.py $SRC $NEW          # every non-video column vs the source
.venv/bin/python -m simple.cli.replay_holomotion_v14 $NEW --all                   # the new logs: "model identical: BIT-EXACT"
.venv/bin/python docs/_scan/holomotion_v14_pipeline_update/replay_node_from_dataset.py $NEW --episode 0   # controller only
```

## 5. What the replay does, step by step

The implementation is `src/simple/cli/rerecord_holomotion_v14.py` (about 330 lines); read it before changing anything.
Per episode:

1. **Scene.** `replay_holomotion_v14._build_env(info, setup, check_legacy=False)` for the first episode, then
   `_reset_to(env, sonic_env, mod, info, setup)`. Each applies the episode's random setup (`scene_setup`) and loads SIMPLE's
   saved scene state (`environment_config`). The robot model comes from
   `robot_variants.teleop_mjcf(info["robot"], backpack_kg, tilt_deg, black_hands=...)`.
2. **The live start, re-done.** `agent.reset_policy()`, then the CLI's `_stand_on_floor(sonic_env, agent, init_pose)`
   with the episode's recorded `init_pose`. The task's success gates take this as their start position
   (`task._start_xy`); live, they took it at the first step after the scene reset.
3. **Physics at frame 0**, from the replay log:
   - `XL.set_state(m, d, state0, XL.INTEGRATION)`;
   - the rest state `XL.set_state(m, d, rest_val[0], XL.REST)` when `rest_idx[0] == 0`;
   - `mj_forward`;
   - `XL.restore_band(robot, action_band_state[0])`;
   - `eng.render_step = render_step0`.
4. **Controller at frame 0.** `node_state.restore(agent.node, NS.unpack(log))`. It starts from the same fresh-start reset
   the agent applies at every scene reset (`node_state.fresh_start`). It then restores the reference queue (10 future
   frames + current), the runtime state, the walking policy's 8-step history, the motion KV cache, the yaw alignment,
   the evaluator's low-state cache, the PICO stick filter and the receive times.
5. **Agent at frame 0.** From `holomotion_start.agent_json`: main state, button-edge bits, hand targets, band drop, hold
   pose. Without it: `POLICY`, no buttons held, quick start finished (true for `--record-on-motion` and quick-start
   recordings).
6. **Clock.** `ControllerClock` is installed once after the agent exists (it patches `time` in the HoloMotion modules
   and the agent). Before each 20 ms step, `clock.now = tick_time[i]`. The node's data-age (0.6 s) and controller-timeout
   (0.5 s) checks then see what they saw live.
7. **Each 20 ms step i** (the teleop CLI's order):
   1. Render both eyes pre-step (`PinholeView.render_pair()`, area-downscaled to the source size).
   2. Put step i's frames and controller sample into the stand-in subscriber.
   3. `action = agent.get_action(observation, ...)`. This feeds the low state from the current physics, then the
      sample, then the frames, then `node.run()`.
   4. The real-format sampler.
   5. `xlog.before_step()`.
   6. `env.step(action)`.
   7. `xlog.start()` on i == 0.
   8. `exporter.add_frame(...)`, then `xlog.add_frame(...)`.
   9. Compare physics with `log["physics"][i]` and the ONNX I/O with the parquet.
8. **Save**, as the CLI does:
   - `RF.finish_episode`;
   - `exporter.save_episode()`;
   - the observation layouts;
   - `environment_config` copied from the source (trap 8);
   - the new replay log with the same controller snapshot;
   - `episodes.jsonl` extras (`scene_setup`, `init_pose`, `realformat`, `exact_log`, `holomotion_start`, `rerecorded_from`).

## 6. Checks and what to expect

Fields of `meta/rerecord_summary.json`, per episode:

- `physics_bitexact`: the MuJoCo physics state (`mjSTATE_PHYSICS`) after every frame equals the source log's, byte for byte.
- `policy_io_bitexact`: at every 20 ms step where a policy ran, the ONNX input and output equal the recorded ones, byte for byte.
- `success_frame`, `success_on_last_frame`: when the task's own `terminated` fired. A source episode that saved itself
  ends on its success frame.
- `task_gates_at_end`: grasped / at_bin / placed at the end.
- `source_saved_by_hand`: left menu held on the last frame, so the operator saved it (left menu + A 20 ms later).

Known, benign differences against the source:

- **Frame 0 `observation.object_poses`** is copied from the source. Live, it came out of the `env.step` before frame 0,
  from the kinematics of that step's last substep, and the log does not hold that. Every later frame is recomputed.
- **`holomotion.MotionReference.frames_this_tick`**: in a 20 ms step where more than 16 frames arrived, only the newest 16 were
  stored, so the replay feeds 16 and records 16. It happened once in 50 episodes. The controller only needs the newest
  11, so physics and ONNX I/O stay exact.
- **Images** differ by design when the robot's look changes (e.g. black hands).

Everything else in the 85 non-video columns must be identical. Use `compare_rerecord.py` to prove it.

## 7. Writing your own replay loop

For a variant, copy `rerecord_holomotion_v14.py` rather than starting from scratch. Variants include: a different motion
model, edited reference frames, VLA-produced references, or no dataset output. The core, condensed (`T` =
`simple.cli.teleop_holomotion_v14`, `XL` = `exact_log`, `NS` = `node_state`):

```python
log = XL.load(src / "replay" / f"episode_{ep:06d}.npz"); info, setup = dict(log["info"]), log["setup"]
snap = NS.unpack(log)                                         # frame-0 controller snapshot (+ agent_json)
info_new = dict(info, black_hands=True); info_new.pop("robot_mjcf", None)      # trap 2
os.environ["HOLOMOTION_UNITREE_POLICY_MODELS_DIR"] = info["policy_models_dir"] # the model the episode used
env, se, mod = _build_env(info_new, setup, check_legacy=False)  # later episodes: _reset_to(env, se, mod, info_new, setup)
agent = HoloMotionV14Agent(se.task.robot, enable_pico_stream=False, button_map="sim", quiet=True)
agent.sub.close(); agent.sub = _ReplaySubscriber()            # stand-in for the ZMQ subscriber
clock = ControllerClock(t0=tick_time[0]); clock.install()     # once, after the agent (the node) exists

agent.reset_policy(); T._stand_on_floor(se, agent, init_pose); se.task._start_xy = d.qpos[:2].copy()
XL.set_state(m, d, log["state0"], XL.INTEGRATION)
if len(log["rest_idx"]) and log["rest_idx"][0] == 0: XL.set_state(m, d, log["rest_val"][0], XL.REST)
mujoco.mj_forward(m, d); XL.restore_band(se.task.robot, log["action_band_state"][0]); se.mujoco.render_step = int(log["render_step0"])
NS.restore(agent.node, snap)                                  # then the agent fields (agent_json or the defaults)
observation, privileged_info = se._get_obs(), se._get_info()

for i in range(n):
    clock.now = float(tick_time[i])
    agent.sub.set_tick(frames_of_tick(i), _pico_sample(pico[i]))     # <- swap in other references here
    action = agent.get_action(observation, instruction=se.task.instruction, privileged_info=privileged_info)
    observation, reward, terminated, truncated, privileged_info = env.step(action)
    same = XL.get_state(m, d, XL.PHYSICS).tobytes() == log["physics"][i].tobytes()
```

Feed different references and nothing matches the recording any more. Then the comparison is a diff report, not a
pass/fail. Load a different model and the frame-0 snapshot no longer fits it, so start from a cold node instead (no
`NS.restore`, the agent in walking mode) and expect divergence from 0 ms.

## 8. Traps (each one cost a failed run)

1. **`_build_env`'s robot selection.** Until 2026-10-18 an `elif backpack_kg` hung off `if tilt:`. Every tilt-0 episode
   was rebuilt with the plain backpack robot, and only the `robot_mjcf` override restored the right file. Fixed in
   `replay_holomotion_v14.py` and `scripts/render_teleop_isaac.py`. If a rebuild picks a strange robot, print
   `robot.mjcf_path`.
2. **`info["robot_mjcf"]` wins over any variant you ask for.** `_build_env` reloads the recorded file when it exists. To
   change the look (black hands), drop that key from the info you pass. The re-recorder asserts the robot file contains
   `_blackhands`.
3. **`check_legacy`.** With a changed look the model hash differs on purpose. Without `check_legacy=False`, `_build_env`
   would "fall back" to the September rot15 robot.
4. **The task's start position.** The bottle_bin gates measure "walked from the start" against `task._start_xy`, set at
   the first gated step. Re-do the quick-start placement (`_stand_on_floor` + the recorded `init_pose`) before loading
   state0, or the gate start becomes frame 0.
5. **Restore order.** `node_state.restore` must start from `fresh_start` (it does). A node straight from `setup()` has a
   warm-up push in its walking history. Also, the evaluator caches the low state by message identity, so between policy
   steps it reads the IMU from 20 ms earlier. Both are in the snapshot; do not "simplify" them away.
6. **Agent state.** Logs before 2026-10-18 have no `agent_json`. The default (`POLICY`, `_last_bits = 0`, no quick start)
   is right for `--record-on-motion` and quick-start recordings. A recording that started mid-button-press would
   diverge; `policy_io_bitexact` would say so.
7. **Clock.** Install `ControllerClock` once and set `now` every 20 ms. Real time in a replay ages the reference past 0.6 s,
   and the node freezes it.
8. **`environment_config` of hand-saved episodes.** The CLI runs one more `env.step` before a manual save, so the source
   stored the scene state 20 ms after its last frame (distractor poses differ by <0.1 mm). Copy the source's config
   verbatim (the re-recorder does). The physics is restored from state0 either way.
9. **The bottle_bin "placed" gate tests the bottle's base point** with an upright-bottle margin. A bottle lying diagonally
   in the bin with its base against a long wall never passes (episodes 30, 35, 46, 49 of 2026-10-18; the operator saved
   them by hand). A replay faithfully reproduces "no success" there; that is not a replay bug. Analysis and a proposed
   whole-bottle check (not applied): `docs/_scan/bottle_bin_gate_check/`, http://10.40.11.11:8899/bottle_bin_gate_check/.
10. **Output folder hygiene.**
    - Never write into the source dataset; always use a new `--save-dir`.
    - Never put stray `.mp4` / `.parquet` files inside a dataset folder; the exporter counts them and crashes on the
      next save.
    - A folder holding another column set or image size is refused at startup on purpose.
11. **Processes.** Kill your own jobs by PID. `pkill -f <pattern>` matches your own shell and kills it. Do not stop jobs
    you did not start.

## 9. Reference result and file map

2026-10-18, `data/teleop_holomotion_v14_rot0` (50 bottle_bin episodes, 34,228 frames, grey hands) →
`data/teleop_holomotion_v14_rot0_blackhands`:

- 50/50 physics and policy I/O bit-exact.
- 85/85 non-video columns identical, except one `frames_this_tick` value (episode 47, at 20 ms: 17 → 16).
- 50/50 new logs replay bit-exactly with the black-hand model.
- 46/50 succeed on their last frame. The other four were hand-saved because of trap 9.

| file | role |
|---|---|
| `src/simple/cli/rerecord_holomotion_v14.py` | the stored-input replay / re-recorder |
| `src/simple/cli/teleop_holomotion_v14.py` | the teleop CLI; its recorder helpers (`_init_exporter`, `_build_frame`, `_raw_columns`, `_stand_on_floor`, ...) are reused |
| `src/simple/cli/replay_holomotion_v14.py` | `_build_env`, `_reset_to`; physics / action replay |
| `src/simple/teleop/holomotion_v14/node_state.py` | controller snapshot / restore / fresh start |
| `src/simple/teleop/holomotion_v14/exact_log.py` | replay log: `load`, `set_state`, `restore_band`, `replay_physics` |
| `src/simple/teleop/holomotion_v14/sim_clock.py` | `ControllerClock` |
| `src/simple/teleop/holomotion_v14/robot_variants.py` | `teleop_mjcf(..., black_hands=)` |
| `src/simple/agents/holomotion_v14_agent.py` | the agent: main state, button map, Dex3 grip, the node calls, `tick_state` / `tick_agent_state` |
| `docs/_scan/teleop_runbook/compare_rerecord.py` | column-by-column comparison |
| `docs/_scan/holomotion_v14_pipeline_update/replay_node_from_dataset.py` | open-loop controller check |
| `docs/holomotion_v14_teleop.md` | why each part exists (dated sections) |
| http://10.40.11.11:8899/pico_sim_teleop_runbook.html | operator runbook: data format, commands, controls |

## 10. House rules

- Do not commit or push without the user's explicit go. The model must never go into a public repo.
- Show visual results to the user as an HTML page (NAS `fleet_status` + an artifact) before asking them to decide.
- Report outcomes as measured, failures included. "Bit-exact" means byte-equal, checked.
