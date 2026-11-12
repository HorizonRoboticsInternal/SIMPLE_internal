# G1 teleop control loop spec — ChipCanToTrash recordings (for the deployment stack)

Written 2026-09-17 for the agent that maintains `robot_orchard_deploy/holobrain_g1_deploy`. It describes the
control loop, every threshold and every convention that produced session `2026-09-17-02-25-56-G1-sim`
(102 episodes of *"pick up the chip can, move towards the trash bin, and place the chip can in the trash bin"*,
exported as `data/real_recordings/psi0/ChipCanToTrash_0225`, 99 episodes, and `ChipCanToTrash_0225_keep97`).
Every number is read from the code or measured on that session. Section 10 is the list of things the deployment
must match or knowingly change.

Code: `SIMPLE/third_party/decoupled_wbc` = upstream `5b304e4` + `SIMPLE/third_party/patches/decoupled_wbc_local.patch`
(all local changes live in the patch; `git -C third_party/decoupled_wbc apply ../patches/decoupled_wbc_local.patch`).
Paths below are relative to `third_party/decoupled_wbc` unless they start with `SIMPLE/`.

Launch used for the session (runbook `SIMPLE/REAL_ROBOT_RUNBOOK.md`):

```bash
python scripts/deploy_g1.py --interface real --robot-ip 192.168.123.164 \
  --body-control-device pico --hand-control-device pico \
  --upper-body-joint-speed 5 \
  --camera-host 192.168.123.164 --camera-port 5555 \
  --data-collection-frequency 30 --root-output-dir /home/Horizon/outputs \
  --home-randomize --home-random-arms
# defaults in effect: --init-pose elbows_raised, --enable-gravity-compensation (arms, 10 Nm clip),
# --home-after-save, --control-frequency 50, teleop 20 Hz, no height randomization (0.74 m)
```

## 1. Processes and rates

| Process | Rate | What it does |
|---|---|---|
| Control loop `control/main/teleop/run_g1_control_loop.py` | **50 Hz** | reads the robot state over DDS, runs the decoupled WBC (upper body = interpolated teleop targets, lower body = RL policy), sends **position targets + feed-forward torque** for the 29 body motors and position targets for the 14 Dex3 hand joints, publishes state + action for the recorder |
| Lower-body RL policy `control/policy/g1_gear_wbc_policy.py` (inside the control loop) | 50 Hz | GR00T-WholeBodyControl `Balance.onnx` / `Walk.onnx` (policy dt 5 ms × decimation 4 = 20 ms); 15 outputs = 12 leg + 3 waist joint targets = default + 0.25 × action; observation 516 = 76 × 6 history; command scale [2, 2, 0.5] on [vx, vy, vyaw]; gait period 0.9 s. **Walk network when ‖[vx, vy, turn flag]‖ ≥ 0.1, Balance otherwise** |
| Teleop policy `control/main/teleop/run_teleop_policy_loop.py` | **20 Hz** | polls the Pico (XRoboToolkit), retargeting IK → 14 arm + 14 hand targets, navigation / height commands, park–release state machine; publishes the goal to the control loop |
| Recorder `control/main/teleop/run_g1_data_exporter.py` | **30 Hz, camera-paced** | one row per D455 frame (blocking ZMQ receive on the robot's 30 fps stream); each row = the newest 50 Hz state/action message + that frame |
| Camera server (on the robot) | 30 fps | D455 colour, 640 × 360, JPEG over ZMQ |
| Motor PD (robot side) | per 50 Hz command | `q_target`, `dq_target = 0`, `tau_ff`, kp / kd of section 2 |

Consequences: **one recorded row = 33.3 ms**; a 24-row action chunk is 0.8 s. The upper-body targets are
refreshed every 50 ms (20 Hz) and interpolated to 50 Hz, so consecutive rows differ by at most one teleop step.
Datasets recorded before 2026-09-17 were labelled `fps = 50`; they were relabelled to 30 in place
(`SIMPLE/scripts/relabel_fps30.py`, content unchanged), so every dataset now carries the true rate.

## 2. Motor gains (every command; actuator order)

`MOTOR_KP / MOTOR_KD` in `control/main/teleop/configs/g1_29dof_gear_wbc.yaml`. Deployment `ARM_GAINS=teleop-all`
loads exactly these from the SIMPLE checkout (that is already the launcher default).

| Motors | kp | kd |
|---|---|---|
| hip pitch / roll / yaw (each leg) | 150 / 150 / 150 | 2 / 2 / 2 |
| knee | 200 | 4 |
| ankle pitch / roll | 40 / 40 | 2 / 2 |
| waist yaw / roll / pitch | 250 / 250 / 250 | 5 / 5 / 5 |
| shoulder pitch / roll / yaw (each arm) | 100 / 100 / 40 | 5 / 5 / 2 |
| elbow | 40 | 2 |
| wrist roll / pitch / yaw | 20 / 20 / 20 | 2 / 2 / 2 |

Hands: Dex3 position control through the hand SDK, no feed-forward torque.

## 3. Init / home pose (`--init-pose elbows_raised`, the default since 2026-09-16)

| Group | Value |
|---|---|
| 14 arm joints | 0, except **both elbows −0.66 rad** |
| 14 hand joints | 0 (open) |
| base height command | **0.74 m** |
| torso roll / pitch / yaw command | 0 / 0 / 0 |
| navigation | vx = vy = flag = 0, target yaw 0 |

Legs and waist are owned by the balance policy (its reference joint offsets: hip pitch −0.1, knee 0.3, ankle pitch
−0.2, others 0). JSON: `control/main/teleop/configs/init_pose_elbows_raised.json`; presets in
`control/robot_model/instantiation/g1.py`. The shoulder-roll limit was widened to [0, 2.25] (L) / [−2.25, 0] (R) so
roll 0 is inside the range. Spec page: `http://10.40.11.11:8899/g1_initial_pose_spec.html`.

Measured on the session (row 0 of each episode, mean over 102 episodes):

| | commanded (`action`) | measured (`observation.state`) |
|---|---|---|
| left / right elbow | −0.66 / −0.65 | −0.64 / −0.62 |
| all other arm joints | 0.00 ± 0.02 | 0.00 ± 0.02 |
| std of the elbow command over episodes | 0.06 / 0.05 (home randomization, section 5) | |

The 0.02–0.03 rad elbow sag is what remains with the gravity feed-forward on; without it the sag was ~0.09 rad.

## 4. Gravity feed-forward (default ON since 2026-09-16)

`control/envs/g1/g1_env.py` + `control/envs/g1/utils/gravity_ff.py`: every command, in every phase (tracking,
ramp home, parked), carries `tau_ff` = pinocchio generalized gravity g(q) of the **14 arm joints, evaluated at the
commanded pose, clipped to ±10 Nm per joint**. Legs, waist, hands get 0. At the init pose the values per arm are
shoulder pitch −3.07 Nm, elbow −2.82 Nm, wrist pitch −0.97 Nm (others ≈ 0). Flags:
`--enable-gravity-compensation` / `--no-enable-gravity-compensation`, `--gravity-compensation-joints`,
`--gravity-compensation-max-torque 10`.

## 5. Episode protocol (park / release, `--home-after-save` default ON)

```
left menu + right trigger  -> activation: min-jerk ramp to the home pose over 6 s, per-joint speed ≤ 0.5 rad/s  [homing]
                              held there; operator hands ignored, calibration origin re-snapped every tick      [parked]
A                          -> recording starts NOW: row 0 = the parked pose; 0.5 s min-jerk blend into
                              live tracking                                                                     [release -> tracking]
A                          -> episode saved; ramp back home (6 s)                                               [homing -> parked]
B                          -> episode discarded; ramp back home
left menu + left trigger   -> toggles the lower-body policy action (must be on before walking)
```

* The **first 15 rows (0.5 s)** of every episode are the blend from the parked pose into the operator's pose.
* Row 0 always has nav = [0, 0, 0, 0] and height 0.74 (verified on all 102 episodes).
* **Home randomization** (`--home-randomize --home-random-arms`, ON for this session): every return home draws a
  per-joint arm offset N(0, σ²) clipped to ±2.5 σ. σ per joint = spread of the first frame of Psi's real data
  (`PSI_REAL_INIT_POSE_STD` in `control/policy/teleop_policy.py`):

  | joint | σ left / right (rad) |
  |---|---|
  | shoulder pitch | 0.047 / 0.055 |
  | shoulder roll | 0.028 / 0.019 |
  | shoulder yaw | 0.015 / 0.017 |
  | elbow | 0.064 / 0.051 |
  | wrist roll | 0.026 / 0.024 |
  | wrist pitch | 0.024 / 0.020 |
  | wrist yaw | 0.018 / 0.020 |

  Hands stay open. Height stays 0.74 (`--home-random-height` would draw 0.68–0.74 m; OFF here).
  `--home-random-scale` multiplies σ, `--home-random-seed` makes the sequence repeatable.

## 6. Upper body: Pico → joint targets

* Wrist targets = controller poses relative to the headset (calibrated at activation), retargeting IK
  (`control/teleop/teleop_retargeting_ik.py`) → 14 arm joint targets. The **waist is not teleoperated**: the 3
  waist joints are outputs of the RL policy, so the recorded waist columns are the policy's, not the operator's.
* The control loop interpolates the 20 Hz targets and **rate-limits every upper-body joint to 5 rad/s**
  (`--upper-body-joint-speed 5`; the config default is 1000 = off, the flag is required).
* Safety monitor `control/envs/g1/utils/joint_safety.py`: arm joint velocity limit **6 rad/s**, finger joints
  **80 rad/s** (raised from 50), safety margin 1.0 (position limits unshrunk, position violations are warnings
  only). A velocity violation latches safe mode (kp = 0, tau = 0) and the stack must be restarted.
* **Hands** (`control/teleop/solver/hand/g1_gripper_ik_solver.py`): controller buttons select one of three fixed
  Dex3 target vectors; the joint vector order is [thumb_0, thumb_1, thumb_2, index_0, index_1, middle_0, middle_1].
  Right hand = −(left hand). While the **left menu button is held, both hands are forced open**.

  | gesture | right-hand target | left-hand target |
  |---|---|---|
  | none | 0 | 0 |
  | **trigger only** (used for the chip can) | thumb −0.5 / −0.7 / −0.7, index +1.5 / +1.5, middle +0.6 / +1.5 | thumb_0 −0.5, thumb_1/2 +0.7, index −1.5 / −1.5, middle −0.6 / −1.5 |
  | trigger + grip | thumb 0 / −0.7 / −0.7, index +1.0 / +1.5, middle +1.0 / +1.5 | negated |
  | grip only | thumb +0.5 / −0.7 / −0.7, index +0.6 / +1.5, middle +1.5 / +1.5 | negated |

  What the session contains: right hand only the *trigger-only* pattern (action maxima exactly 1.5 / 1.5 / 0.6 / 1.5,
  thumb −0.5 / −0.7 / −0.7); the measured hand stops on the can at index_0 0.80, index_1 1.49, middle_0 0.47,
  middle_1 1.42, thumb −0.48 / −0.49 / −0.62. Left hand never closes (action 0 throughout), although the left arm
  does move (left elbow range [−0.90, 1.41]).

## 7. Navigation, height, torso

Stick processing (`control/teleop/streamers/pico_streamer.py`): `n = sign(s) · (|s| − 0.1) / 0.9` for |s| > 0.1,
else 0 (dead zone 0.1, then renormalised to [−1, 1]).

| Channel | Source | Formula | Recorded as |
|---|---|---|---|
| vx | left stick Y | n × **0.5 m/s** | `nav[0]` (session: 0 … +0.50, never negative) |
| vy | −left stick X | n × 0.5 m/s | `nav[1]` (session: −0.34 … +0.45, 2.6 % of rows) |
| turn flag | −right stick X | n × **1.0** | `nav[2]`; negative = clockwise (right turn); session −0.91 … +1.0 |
| target yaw | integrated dial | dial += flag × (1/50) per teleop tick, ticks at 20 Hz → **0.40 rad/s per unit flag** (measured 0.403 on the session); continuous, never wrapped | `nav[3]` = **turn since the A press** = wrap(dial − dial_at_A) ∈ [−π, π], exactly 0 on row 0; session −2.05 … 0 |
| base height | Y (+) / X (−) buttons | ±0.01 m per teleop tick, clipped to [0.20, 0.74] | `base_height_command`; **0.74 on every row of the session** |
| torso rpy | not teleoperated | | `observation.torso_rpy_command` = 0 (unused by the export, see section 8) |

Yaw controller in the lower-body policy (`G1GearWbcPolicy`): `err = wrap(target_heading − heading)`;
commanded yaw rate = `clip(err / 0.5, ±1.0 rad/s)` **only while |flag| ≥ 0.1 and |err| > 0.01 rad, else 0**. The
robot therefore turns only while the stick is pushed, toward a target that leads the heading by whatever the dial
accumulated. The dial is re-anchored to the current heading at every A press (`yaw_dial_offset`), which is what
makes `nav[3]` relative to the heading at episode start.

Fixed on 2026-09-17: the dial used to wrap at ±π and the 5 rad/s interpolator smeared the 2π jump into a ~1 s
sweep of the recorded target yaw (episodes 4, 5, 9, 12 of `2026-09-17-00-16-29` and 6, 7 of
`2026-09-16-22-08-53`: drop them). Now flag and dial bypass the interpolator (zero-order hold) and the dial is
continuous; the largest per-row change of `nav[3]` on the session is 0.037 rad (two teleop ticks in one row).

## 8. Recorded data

Raw session (`data/real_recordings/2026-09-17-02-25-56-G1-sim`, LeRobot v2.1, **fps 30**, 102 episodes,
64 218 rows):

| column | dim | content |
|---|---|---|
| `observation.state` | 43 | measured joints in **actuator order**: legs 0:12 (L hip p/r/y, knee, ankle p/r, then R), waist 12:15 (yaw, roll, pitch), L arm 15:22, **L hand 22:29 (index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2)**, R arm 29:36, R hand 36:43 (same order) |
| `action` | 43 | the WBC's commanded joint targets, same order (legs / waist = policy output, arms / hands = interpolated teleop targets) |
| `teleop.navigate_command` | 4 | [vx, vy, turn flag, target_yaw_rel] as in section 7 |
| `teleop.base_height_command` | 1 | 0.74 |
| `observation.torso_rpy_command` | 3 | 0 |
| `observation.base_pose` | 7 | position **not recorded (zeros)**, quaternion (w, x, y, z) valid |
| `observation.base_vel` | 6 | base twist |
| `observation.eef_state`, `action.eef` | 14 | wrist poses (measured / commanded) |
| `observation.img_state_delta` | 1 | image-stamp minus state-stamp, 18–93 ms on the session (different clocks, use only relatively) |
| `observation.images.ego_view` | 640 × 360 | D455 colour, head mounted, 30 fps |

psi0 export (`SIMPLE/scripts/postprocess_psi0_sonic.py`, fps taken from the source): joints are **re-indexed by
name** into the SIMPLE order (legs, waist, L arm, R arm, L hand thumb/index/middle, R hand thumb/index/middle)
and then sliced:

| psi0 `states` (32) | | psi0 `action` (36) | |
|---|---|---|---|
| 0:3 | L thumb 0-2 (state) | 0:28 | same slices, from `action` |
| 3:5 | L middle 0-1 | 28:30 | **waist roll, pitch — the WBC's commanded waist joints** |
| 5:7 | L index 0-1 | 30:31 | **waist yaw (commanded)** |
| 7:14 | R hand: thumb 0-2, index 0-1, middle 0-1 | 31:32 | base height command (0.74) |
| 14:21 | L arm | 32:34 | vx, vy |
| 21:28 | R arm | 34:35 | turn flag |
| 28:31 | waist roll, pitch, yaw **measured joint angles** | 35:36 | target_yaw_rel |
| 31:32 | base height command of the previous row (0.74) | | |

Extra columns: `observation.prev_torso_rpy` (3), `observation.prev_height` (1), image key
`observation.images.egocentric`, task string as above. The "rpy" rows are **not zero**: they are the balance
policy's waist joints. Session ranges, action: roll [−0.13, 0.15], pitch [−0.03, 0.18], yaw [−0.06, 0.12];
state: roll [−0.11, 0.14], pitch [0.05, 0.19], yaw [−0.07, 0.12].

Quality on the session: **0 duplicate (stale) rows** in all 102 episodes (buffered state shelf + camera-first
ordering, `control/utils/state_shelf.py`, fixed 2026-09-17; earlier sessions have 3–20 % isolated duplicates),
timestamps exactly 1/30 s apart, measured row rate 30.0 Hz per the recorder's per-episode log.

## 9. What the model saw (session statistics, 102 episodes)

| quantity | value |
|---|---|
| episode length | 480–786 rows (16–26 s), mean 630 rows (21 s); one 138-row (4.6 s) episode |
| walking (vx > 0) | 16 % of rows, vx ≤ 0.50 m/s, never backwards; vy ≠ 0 on 2.6 % of rows |
| turning (flag ≠ 0) | 55 % of rows: 49 % clockwise (flag −0.4 … −0.91, median −0.41), 6 % counter-clockwise corrections; target_yaw_rel ∈ [−2.05, 0] |
| base height | 0.74 constant |
| right arm (action) | shoulder pitch [−1.07, 0.21], roll [−0.54, 0], yaw [−0.68, 0.74], elbow [−0.93, 1.43], wrist roll [−0.67, 0.55], pitch [−1.18, 0.58], yaw [−0.47, 0.13] |
| left arm (action) | shoulder pitch [−1.05, 0.66], elbow [−0.90, 1.41], wrist yaw [−0.44, 1.09] |
| waist (policy output) | roll [−0.13, 0.15], pitch [−0.03, 0.18], yaw [−0.06, 0.12] |
| hands | right: trigger-only pattern (section 6); left: open |

## 10. Deployment checklist (match, or change knowingly)

1. **Rate**: rows are 30 Hz; execute model rows at 30 Hz (24-row chunk = 0.8 s). The WBC itself keeps its own rate.
2. **Init pose**: `deploy_client.sh` defaults to `INIT_POSE=3`; pass **`INIT_POSE=5`** and change its elbows from
   **−0.65 to −0.66** (`INIT_POSES["5"]` in `patches/psi0_client_delta.patch`, preset
   `scripts/presets/init_pose5_user_20260916.json`); hands open, height 0.74, torso 0 / 0 / 0 (`INIT_TORSO=0,0,0`;
   the launcher default is −0.019, 0.057, 0). Park there between episodes; optionally jitter the arms with the σ
   table of section 5 (training start poses vary by up to ±0.15 rad).
3. **Gravity feed-forward on the arms**: `ARM_GRAVITY_COMP=on` (launcher default is off). The data has the
   0.02–0.03 rad sag of section 3, not the ~0.09 rad sag of a pure-PD start.
4. **Gains**: `ARM_GAINS=teleop-all` (launcher default, section 2).
5. **Row 35 (target yaw) is relative to the heading at episode start, wrapped to ±π.** The Psi-0 master uses the
   same anchor: `yaw_offset := IMU yaw` when `reset_yaw_offset` is set, and the patched client sets it right before
   the control thread starts an episode; `dyaw = wrap(imu_yaw − yaw_offset − row35)` goes to the lower-body
   policy as (sin, cos). Keep that reset at every episode start (SPACE), not only at pose init, and never add an
   absolute heading to row 35.
6. **Row 34 (turn flag) semantics differ between the two WBCs.** The teleop's gear WBC turns *only while*
   |row 34| ≥ 0.1 (then at `clip(err / 0.5, ±1)` rad/s); the Psi-0 WBC turns whenever `dyaw ≠ 0`, regardless of
   row 34. In the data the flag is non-zero whenever the target moves, so both behave alike during a turn; they
   differ only after the model drops the flag with a residual heading error (the deployed robot finishes aligning,
   the teleop robot would stop). Passing rows 32:35 through unchanged (`WALK_GATE=off`, the default) is correct;
   the original Ψ₀ gate (vx > 0.25 → 0.35 m/s, |vy| < 0.3 → 0) would distort the 0.5 m/s walking rows.
7. **Rows 28:31 are waist joint angles from the balance policy, not zero** (section 8). The client maps them to the
   master's torso roll / pitch / yaw command. Expect roll ±0.15, pitch up to 0.18, yaw ±0.12.
8. **Row 31 (height)** = 0.74 always in this task.
9. **Hands (rows 0:14)**: the closed targets of section 6 are what the model outputs; the Dex3 stops earlier on the
   object (state ≠ action, normal). The left-hand rows stay 0.
10. **Episode start**: the first 0.5 s of every training episode is the blend out of the parked pose; a small
    settling motion at the start of a rollout is expected.
11. **Upper-body limits**: the teleop rate-limited joints to 5 rad/s and shut down above 6 rad/s (arms) /
    80 rad/s (fingers). The client's per-step arm clip (`clip_arm_q_target`, 30 rad/s) is looser; a model step
    larger than 5 rad/s never occurred in the data.
12. Do not run a second XRoboToolkit client or PC service on the workstation while the teleop is up (shared
    stream; the teleop refuses to start if two services listen on 127.0.0.1:60061).

## 11. Files

* Teleop policy, park / release, home randomization: `control/policy/teleop_policy.py`
* Stick processing, dial, hand gestures: `control/teleop/streamers/pico_streamer.py`; Dex3 gesture targets:
  `control/teleop/solver/hand/g1_gripper_ik_solver.py`
* Yaw controller, Balance / Walk switch, `target_yaw_rel`: `control/policy/g1_gear_wbc_policy.py`; zero-order hold
  of flag + dial: `control/policy/g1_decoupled_whole_body_policy.py`; interpolation / rate limit:
  `control/policy/interpolation_policy.py`
* Gravity feed-forward: `control/envs/g1/g1_env.py`, `control/envs/g1/utils/gravity_ff.py`
* Init pose presets: `control/robot_model/instantiation/g1.py`, `control/main/teleop/configs/init_pose_elbows_raised.json`
* Config defaults and flags: `control/main/teleop/configs/configs.py`, `scripts/deploy_g1.py`
* Recorder: `control/main/teleop/run_g1_data_exporter.py`, `control/utils/state_shelf.py`, `control/utils/navigate_cmd.py`
* Export: `SIMPLE/scripts/postprocess_psi0_sonic.py`; relabel: `SIMPLE/scripts/relabel_fps30.py`
* Tests: `SIMPLE/tests/test_target_yaw_relative.py`, `test_home_randomize.py`, `test_gravity_feedforward.py`,
  `test_init_pose.py`, `test_state_shelf.py`
* Runbook: `SIMPLE/REAL_ROBOT_RUNBOOK.md`; deploy side: `holobrain_g1_deploy/scripts/deploy_client.sh`,
  `patches/psi0_client_delta.patch`, Ψ₀ master `Psi0/real/teleop/master_whole_body.py`
