# Sim teleop (VR → MuJoCo) — process and runbook

How the `*Teleop-v0` simulator pipeline works and how to run it. Verified
2026-08-21 (11 episodes recorded of `G1WholebodyXMovePickTeleop-v0`).

Companion docs:
- `REAL_ROBOT_TELEOP.md` — same controller on physical hardware (concepts)
- `REAL_ROBOT_RUNBOOK.md` — real-robot step-by-step + failure table
- `SCRIPTED_G1_NOTES.md` — the *other* pipeline (`*MP-v0`, motion planning + AMO)
- `cmd_sweep_report.html` — measured command envelope of this controller

---

## 1. What runs

Same controller as the real robot: **decoupled WBC** (not AMO, not SONIC).

```
PicoStreamer (XRoboToolkit)
  → TeleopRetargetingIK        wrist poses → 28 upper-body joint targets
  → InterpolationPolicy        rate-limited (upper_body_joint_speed)
  → G1GearWbcPolicy            Balance.onnx / Walk.onnx, switched on
                               ‖[vx, vy, vyaw_flag]‖ < 0.1 → 15 leg joints
  → ActionCmd("decoupled_wbc") 43 PD position targets, 50 Hz ctrl / 200 Hz sim
```

SIMPLE imports `decoupled_wbc` as a **library** and drives its own MuJoCo env, so
— unlike `deploy_g1.py` — the sim path needs **no ROS 2 and no Docker**.

Movement commands (the only things that move the base):
`navigate_cmd = [vx, vy, vyaw_flag, target_yaw]` and `base_height_command`.
PICO caps: 0.5 m/s, 1.0 rad/s, height 0.2–0.74 m.

## 2. Run it

```bash
cd /home/Horizon/wrk/SIMPLE
source .venv/bin/activate
export MUJOCO_GL=egl

python -m simple.cli.teleop_decoupled_wbc simple/G1WholebodyXMovePickTeleop-v0 \
  --target=graspnet1b:0 --sim-mode=mujoco --record --no-headless \
  --save-dir data/teleop_recordings \
  --success-criteria 0.9 --max-episode-steps 3000
```

Prereqs: XRoboToolkit PC service running on the host (already installed —
connects on `127.0.0.1:60061`), PICO app open, ankle trackers on.

Healthy startup prints both ONNX policies loading, `XRoboToolkit SDK
initialized`, and `[TCPServer] Ready for connections on 0.0.0.0:13579`.

## 3. Operate — do NOT press policy-action in sim

| Step | Control |
|---|---|
| 1. wait for stabilize | — |
| 2. **activate teleop** | **left menu + right trigger** (match your arm pose first — PICO activation calibrates instantly, no countdown) |
| 3. drive | left joystick walk/strafe · right joystick turn · Y/X height |
| 4. save episode | automatic on task success, or **left menu + A** |
| 5. discard | **left menu + X** |
| reset teleop | keyboard `k` |

**The policy-action toggle (left menu + left trigger) must NOT be pressed in
sim.** Every sim CLI pre-enables the lower-body RL
(`agent._wbc_policy.lower_body_policy.use_policy_action = True`), so the toggle
*disables* it and the robot collapses. On the real robot the opposite is true —
nothing pre-enables it, so there you must press it. This asymmetry cost a
session to find.

In `--record` mode `_on_episode_reset()` also **skips the elastic-band drop**:
the robot spawns standing, the WBC pipeline is reset, RL engages immediately.
(Engaging RL *after* a band drop onto a statically-held stance fails 100% of the
time — measured in the command sweep.)

## 4. When an episode saves

Recording begins only when `elastic_done and teleop_active and robot.stabilized
and agent._cached_target_q is not None`, and the episode ends on
`terminated or truncated` — i.e. task success or `--max-episode-steps`.

**`--success-criteria` means different things per task** — this is the usual
reason "it just keeps recording":

| Task | Reward | Sensible value |
|---|---|---|
| `XMovePickTeleop` | `clip((lift)/0.1, 0, 1)` — **capped at 1.0** | `0.9` (CLI default of **3 can never be reached**) |
| `HandoverTeleop` | `+0.1/step while in container`, unbounded, resets to 0 on loss | `3` works (= 30 consecutive steps) |

Manual save/discard (added locally in `teleop_decoupled_wbc.py`,
`_ManualEpisodeControl`) works alongside the automatic path — whichever fires
first wins. Console prints `Started recording N`, `Manual save requested`,
`Episode abandoned`, or `[Record] Episode N saved`.

## 5. Headset first-person view

Streaming is **pull-initiated by the headset**: nothing happens until the PICO
app sends `OPEN_CAMERA` to the workstation on port **13579**. Then the agent
spawns a `StreamingThread` and pushes `head_stereo_left|right` side by side
(default 2560×720) with an episode counter overlaid top-right of each eye — if
you can read that counter, the pipeline is working end to end.

Watch for `[TCPServer] Client connected from …` and `[PicoDecoupled]
OPEN_CAMERA: {...}`. Absent those, the headset never reached 13579.

## 6. Output and the two-stage pipeline

Stage 1 (this doc) captures in MuJoCo — **untextured** renders, because Isaac is
deliberately off to keep VR latency low. Output is LeRobot format at 50 fps:

```
data/teleop_recordings/simple/<Task>-v0/level-0/{data,videos,meta}
```

Stage 2 renders photorealistically by replaying the recorded actions:

```bash
python -m simple.cli.replay_decoupled_wbc simple/<Task>-v0 \
  --data-dir=data/teleop_recordings/simple/<Task>-v0/level-0/ \
  --sim-mode=mujoco_isaac --render-hz=50 --record --resume \
  --save-dir=data/replay_out --success-criteria=0.2
```

Then `scripts/postprocess_psi0_sonic.py` projects to the Psi-0 training format
(43-D full-body state → 36-D controller-agnostic action = 28 upper joints +
3 rpy + 1 height + 4 nav). Scripted `*MP-v0` data needs no replay — it can
render Isaac in one pass.

## 7. Measured command envelope

From `cmd_sweep_report.html` (14 scripted-command trials, all video-backed):

| Command | Achieved | Note |
|---|---|---|
| vx 0.3 / 0.5 / 0.8 / 1.2 / 1.8 | 0.08 / 0.13 / 0.15 / 0.13 / 0.14 m/s | **saturates ≈0.15 m/s** — the 0.5 joystick cap is already past the policy's top speed |
| vy 0.4 / 0.8 | 0.10 / 0.21 m/s | lateral is the fastest axis |
| vyaw_flag 0.5 / 1.0 | 0.33 / 0.35 rad/s | **saturates ≈0.35 rad/s** (~18°/s; 90° ≈ 4.5 s) |
| height 0.50 / 0.30 | tracked / 0.29 held | deep crouch is stable, not a fall |
| walk while crouched (0.5 m) | 0.026 m/s | nearly stalls |

14/14 trials stayed upright, including 3.6× over-command — the policy saturates
rather than destabilising. Reproduce/extend with `teleop_cmd_sweep.py` then
`make_sweep_report.py`.

## 8. Gotchas

- `LeRobotDataset.create` uses `mkdir(exist_ok=False)`; a run that dies before
  its first save leaves a stub that makes the next run fail with *"Failed to
  resume from corrupted dataset"*. Delete the stub (or use a fresh `--save-dir`).
- `images/` is always empty — LeRobot deletes PNGs after encoding to mp4, and
  the parquet holds no image columns.
- `--sim-mode` must be `mujoco` for teleop (asserted); `mujoco_isaac` is for the
  replay stage only.
- Browse recordings with `python make_video_viewer.py` → `video_viewer.html`.
