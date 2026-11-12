# Handover: rot0 G1 + BrainCo Revo 2 robot (URDF / MJCF) and the full sim-teleop recording

Written 2026-10-26 for an agent picking up work on this robot or its recordings. Paths are relative to the SIMPLE repo
(`/home/Horizon/wrk/SIMPLE`). `data/` is **not** tracked by git: on another machine use the bundle
`dist/handover_revo2_20261026.tar.gz` (robot folder, URDF tools, this file, one sample full recording).

## 0. Where things are

| what | path |
|---|---|
| Robot MJCF, **source of truth** (what the sim runs) | `data/robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2_v3.xml` since 2026-10-28 (= v2 with the left thumb tactile pad moved to the fingertip; it sat 42.5 mm behind it). `_v2.xml` (2026-10-27) = `g1_rot0_teleop_revo2.xml` + low-friction hand self-contact (`scripts/robots/add_revo2_hand_selfcontact.py`). Same bodies, joints and sensors throughout. Recordings name their file in the replay log (`robot_mjcf`) and replay on it; `tactile.left` thumb columns recorded before 2026-10-28 come from the misplaced pad |
| Robot URDF, generated from the MJCF (v3) on 2026-10-28 | `data/robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2.urdf` (+ `.urdf.json` export report) |
| Camera intrinsics + extrinsics (recorded streams, raw HBVCAM fisheye calibration, all MuJoCo cameras, IMU frames), 2026-10-28 | `data/robots/g1_rot0_teleop_revo2/camera_calibration.json` (written by `scripts/robots/export_camera_calibration.py`; checked against the URDF frames and a recording's `meta/holomotion_raw.json`) |
| Meshes (82 STL; 3 mount meshes are in mm, `scale 0.001`) | `data/robots/g1_rot0_teleop_revo2/meshes/` |
| Sidecars | same folder: `cameras.json`, `touch_pads.json`, `head_mount_pose.json`, `wrist_mount_pose.json`, `module_original.json`, `grip_poses.json`, `package.json` |
| URDF export / check | `scripts/robots/export_urdf_revo2.py`, `scripts/robots/check_urdf_revo2.py` (results in `.../urdf_check/urdf_check.json`) |
| Robot build chain | `scripts/robots/make_g1_revo2.py` → `add_cameras_mounts.py` → `add_revo2_touch.py` (the last two edit `*_notouch.xml` → final XML) |
| Sim robot class | `src/simple/robots/g1_sonic_revo2.py` (`robot_uid="g1_sonic_revo2"`, hand PD, grip poses) |
| Recorder | `src/simple/cli/teleop_holomotion_v14.py`; `src/simple/teleop/holomotion_v14/{revo2_record,wrist_cams,realformat,exact_log,fill_record,hbvcam_views}.py` |
| Replay / fill | `src/simple/cli/replay_holomotion_v14.py` |
| Tactile sensor model | `src/simple/sensors/revo2_touch.py` |
| Human docs | `docs/holomotion_v14_teleop.md` (sections "Recording", "Loop pacing, --lean-record and --fill", "Bit-exact replay"); pages `docs/wrist_head_cams/`, `docs/revo2_touch/`, `docs/kitchen_teleop/` |

**Rule:** change the robot only through the build chain and the MJCF, then re-export and re-check the URDF:

```bash
.venv/bin/python scripts/robots/export_urdf_revo2.py
MUJOCO_GL=egl .venv/bin/python scripts/robots/check_urdf_revo2.py      # must print "all PASS"
```

## 1. The robot

Unitree G1 29-DoF with:

- **BrainCo Revo 2 hands**: 11 joints per hand, 6 motors, replacing the Dex3 hands. Each hand hangs on an adapter: `{side}_hand_base2_link` sits 41.5 mm from `{side}_wrist_yaw_link`, and `{side}_hand_base_link` sits at 59.1 mm.
- **HBVCAM stereo head camera** on the printed rot0 bracket. The bracket is body `head_stereo_mount`, mesh `head_stereo_mount_rot0.STL`.
- **Two HBVCAM-F2619HD wrist cameras** in printed collars. The collars are bodies `{side}_wrist_cam_mount`, with the housing on the palm side.
- **Revo 2 fingertip tactile pads**: links `{side}_hand_{finger}_touch_link`.
- **A 3.2 kg backpack**: `backpack_link`.

The hands are black (visual only). The total mass is 39.55 kg, and each hand below the wrist-yaw joint weighs 0.875 kg.

### 1.1 What changed relative to `data/robots/g1_sonic/g1_29dof_with_hand_hbvcam_stereo.urdf` (2026-10-16, Dex3)

| item | 10-16 URDF | now |
|---|---|---|
| hands | Dex3, 7 joints each | Revo 2, 11 joints each (10 mimic joints in total), adapter ring visual Ø50 × 17.6 mm |
| head eyes (`head_stereo_{l,r}_optical_frame`) | torso_link (70.52, ±30.36, 424.46) mm | **(77.02, 30.36, 392.96) / (76.62, −30.36, 393.46) mm**, i.e. **+6.5 mm forward, −31.5 mm down**, same orientation (49.3° down) |
| head bracket | none | `head_stereo_mount` on torso_link at (2.4, 0, 313.8) mm, yaw 180° |
| wrist cameras | none | collars + cameras, see 1.3 |
| tactile pads | none | 5 pad links and 5 pad frames per hand |
| new frames | | `hbvcam_{left,right}_pinhole_optical_frame` (rectified image frames), `wrist_{left,right}_optical_frame`, `pelvis_imu_link` |

The head shift is the user's own visual tuning on 2026-10-26, made to match a video of the real robot. The module and
its four mounting holes moved with the bracket. It is **not** a measured calibration.

### 1.2 Joints

- **Body.** The 29 body joints have the same names and limits as the G1 URDF. Effort limits are the MJCF actuator ctrlrange (hip and waist 88, knee 139, shoulder to wrist pitch 25, wrist yaw 5 N·m); velocity limits are copied from the G1 URDF.
- **Hands, in MJCF order per side**, prefix `{side}_hand_`, all lower limits 0:

| joint | upper (rad) | motor / coupling |
|---|---|---|
| thumb_metacarpal_joint | 1.5184 | motor "thumb_rotation" |
| thumb_proximal_joint | 1.0472 | motor "thumb_bend" |
| thumb_distal_joint | 1.0472 | mimic thumb_proximal × 1.0 |
| index_proximal_joint | 1.4661 | motor "index" |
| index_distal_joint | 1.693 | mimic index_proximal × 1.155 |
| middle_proximal / distal | 1.4661 / 1.693 | motor "middle" / mimic × 1.155 |
| ring_proximal / distal | 1.4661 / 1.693 | motor "ring" / mimic × 1.155 |
| pinky_proximal / distal | 1.4661 / 1.693 | motor "pinky" / mimic × 1.155 |

- **Couplings.** In the MJCF the distal joints are coupled through joint equalities; in the URDF they are `<mimic>`. Pinocchio and MuJoCo's URDF importer ignore `<mimic>`, so set distal = multiplier × proximal yourself when you do forward kinematics there.
- **Effort and velocity limits.** Finger effort limits are the scaled values used in sim (0.5–1.2 N·m). The module URDF's 2 N·m gave more than 30 N per finger, against the retailer's 15 N pinch. Velocity limits are copied from the source URDFs.

### 1.3 Frames you will need

Each frame is given relative to its parent: xyz in mm, URDF rpy in rad.

| frame | parent | xyz (mm) | rpy | meaning |
|---|---|---|---|---|
| `head_stereo_left_optical_frame` | torso_link | 77.02, 30.36, 392.96 | −2.431087, 0, −1.570796 | raw left fisheye eye (optical: x right, y down, z forward), 49.3° down |
| `head_stereo_right_optical_frame` | torso_link | 76.62, −30.36, 393.46 | −2.431446, −0.002786, −1.576805 | raw right eye |
| `hbvcam_left_pinhole_optical_frame` | torso_link | 77.02, 30.36, 392.96 | −2.430247, −0.007546, −1.564990 | **image frame of `observation.images.ego_view_left`** (rectified, R1/P1) |
| `hbvcam_right_pinhole_optical_frame` | torso_link | 76.62, −30.36, 393.46 | −2.432342, −0.001923, −1.562961 | image frame of `ego_view_right` (R2/P2) |
| `wrist_left_optical_frame` | left_wrist_cam_mount | (in left_wrist_yaw_link: 49.06, −33.84, 0) | see URDF | **image frame of `observation.images.wrist_left`**; looks along (0.819, 0.574, 0) of the wrist-yaw frame, i.e. forward and 35° in onto the palm |
| `wrist_right_optical_frame` | right_wrist_cam_mount | (in right_wrist_yaw_link: 49.06, 33.84, 0) | see URDF | image frame of `wrist_right`; view (0.819, −0.574, 0) |
| `{side}_wrist_cam_mount` (body) | {side}_wrist_yaw_link | 36.5, 0, 0 | left 0, π/2, 0; right mirrored | printed collar, housing on the palm side (palm = −y of the LEFT wrist-yaw frame, +y of the RIGHT one; thumb side +z on both) |
| `pelvis_imu_link` | pelvis | 45.25, 0, −83.39 | 0 | the IMU the recorded `states.robot.imu_*` columns refer to |
| `imu_link`, `d435_link`, `mid360_link` | torso_link | as the stock G1 URDF | | stock sensor frames; `imu_link` = the MJCF site `imu` |
| `{side}_hand_{finger}_touch_site` | `{side}_hand_{finger}_touch_link` | | | tactile pad frame: on the finger axis, **z into the finger**, x toward the tip |

Not exported to the URDF, by design:

- The five 90° face cameras per eye and the three per wrist. They are MuJoCo rendering helpers.
- The MJCF sites `{side}_wrist_cam_optical`. They sit at the wrist lens but are turned 180° about the viewing axis relative to the recorded wrist image. Use `wrist_{side}_optical_frame` instead.

### 1.4 Camera image models

These are what the recorder writes into `meta/holomotion_raw.json` → `"images"`, shown here at the default 640 × 360 size.

| image | model | parameters at 640 × 360 |
|---|---|---|
| `ego_view_left` | pinhole (rectified) | fx = fy = 246.7829, cx = 317.1564, cy = 179.7328 |
| `ego_view_right` | pinhole (rectified) | same intrinsics; baseline 60.517 mm |
| `wrist_left` / `wrist_right` | **equidistant fisheye**, r_px = f·θ | f = 280.483, cx = 319.5, cy = 179.5; 150° on the image diagonal (130.7° × 73.5°); outside the lens circle is black |

- **Head images.** The head images are rendered at 1280 × 720 and area-downscaled to the recorded size. `--record-size 1280x720` records them at full size, and the json then carries the scaled intrinsics.
- **Wrist images.** The wrist images are assembled from three 90° faces (448 px each) at the same optical centre. They are new renders every 1/30 s (`wrist_cams.fresh` = 1), and the previous frame is repeated in between.
- **Not a real lens.** The real F2619HD lens is 76° diagonal. The 150° wrist view is a synthetic wide view the user asked for, and "150°" was read as a diagonal FOV, which the user has not confirmed.
- **Projecting a point.** Express it in the image frame's optical coordinates (x right, y down, z forward), then apply the model above.

### 1.5 How the URDF was verified (`check_urdf_revo2.py`, all PASS 2026-10-26)

| check | result |
|---|---|
| FK of 96 links and frames at 50 random joint configs, URDF loaded by MuJoCo vs the MJCF | max 1.2e-12 m, rotation 1.4e-11 |
| mass, world CoM, world inertia tensor per link | total 39.5534 kg both; CoM 9e-13 m; inertia 2.5e-9 kg·m² |
| marker spheres through the recorder's own image pipelines vs the intrinsics above (28 markers, 4 cameras, wrist side faces included) | ≤ 0.63 px head (downscale quantisation), ≤ 0.25 px wrist |
| depth renders of the visual geometry, 6 views | 0 of 151 171 robot pixels differ by more than 1 mm |
| `yourdfpy` validate + load meshes; `pinocchio` FK vs MJCF (fingertips, feet, torso, 2 camera frames) | valid; 4.9e-12 m |

### 1.6 Caveats

**Fields a URDF cannot carry.** MuJoCo-only properties are absent from the URDF: joint armature (0.01), contact parameters, actuator gains and the hand PD, and the sensors.

- Use the MJCF for simulation.
- The sim's solver settings are a 5 ms step, elliptic cone, impratio 10, 2 noslip iterations, the implicitfast integrator, and 4 physics steps per 20 ms control tick.

**Collision geometry.** Collision elements are the mesh STLs the MJCF collides with; MuJoCo uses their convex hulls. The Revo 2 palm–thumb hull contact is excluded in the MJCF (`<contact><exclude>`), but URDF has no equivalent.

**Massless parts.** The head bracket and the wrist collars are massless visual bodies in the model. The real printed parts weigh something, which has not been measured.

**Inertia format.** Inertias are exported as full tensors in each link frame with inertial rpy 0. Some readers, MuJoCo's URDF importer among them, ignore the inertial rpy.

**Real-robot differences:**
- The wrist camera's 35° tilt onto the palm exists only in sim. The printed collar holds the module at 0°, so a wedge or a new pocket is needed to match.
- The BrainCo adapter dimensions are approximate; no adapter CAD was available.

## 2. The full recording

### 2.1 How it is made

```bash
# full recording (default): states, actions, controller inputs, 4 videos, tactile
scripts/teleop_holomotion_v14.sh --scene kitchen --robot revo2 --record --save-dir data/<run>
# lean recording: no images and no tactile while teleoperating, filled afterwards on the bit-exact replay
scripts/teleop_holomotion_v14.sh --scene kitchen --robot revo2 --record --lean-record --save-dir data/<run>
python -m simple.cli.replay_holomotion_v14 data/<run>/simple/G1WholebodyKitchenSinkTeleop-v0/level-3_pinhole --all --fill
#   -> level-3_pinhole_filled: the same columns as a full recording
```

- **Lean vs full.** A filled lean dataset has the full recording's columns. On a full recording, `--fill` reproduced every tactile column and `wrist_cams.fresh` exactly, and the images to codec precision (36 dB ego, 42 dB wrist).
- **Kitchen task knobs** are set in the environment and stored per episode:
  - `KITCHEN_TASK=bottle_sink` (the default) or `ball_bin`.
  - `KITCHEN_BALLS` (balls are off by default).
  - `KITCHEN_BIN_LID` (closed by default).
  - `KITCHEN_REV` (scene revision, 2).

### 2.2 Layout (LeRobot v2.1, 50 fps)

```
<save-dir>/simple/G1WholebodyKitchenSinkTeleop-v0/level-3_pinhole/
  data/chunk-000/episode_000000.parquet        119 columns, one row per 20 ms
  videos/chunk-000/observation.images.{ego_view_left,ego_view_right,wrist_left,wrist_right}/episode_000000.mp4
  meta/info.json  modality.json  tasks.jsonl  episodes.jsonl  episodes_stats.jsonl
  meta/realformat.json      real-G1-recorder field map and conventions (hand "brainco_revo2")
  meta/holomotion_raw.json  HoloMotion interface columns, observation layouts, image intrinsics
  meta/lean_record.json     lean recordings only (what was left out, and how --fill puts it back)
  replay/episode_000000.npz exact log for the bit-exact replay
```

### 2.3 Timing

- **Frame t.** Each frame t is one 20 ms control tick. All state, tactile and image columns are sampled on the pre-step state, the same instant as `observation.state[t]`. `action[t]` is the command applied during the step t → t+1.
- **Timestamps.** `timestamp = t / 50`. `holomotion.tick_time` is the wall clock of the tick (a loop-pacing diagnostic).
- **After a reset.** The sim restarts each episode from a new compiled scene. Frame 0 is the start pose, randomised per seed.

### 2.4 Columns

Orders below are exact. Use `meta/info.json` → `features[*].names` for the per-element names.

**Robot state and command (decoupled-WBC schema, widened for the Revo 2):**

| column | dtype × shape | content |
|---|---|---|
| `observation.state` | f64 × 51 | joint positions in **robot order**: 29 body (legs 0–11, waist 12–14, left arm 15–21, right arm 22–28), left hand 29–39, right hand 40–50 (hand joints in MJCF order, 1.2) |
| `action` | f64 × 51 | joint targets in the **decoupled layout**: legs + waist + left arm 0–21, **left hand 22–32**, right arm 33–39, right hand 40–50. ⚠ This differs from the state order; `meta/modality.json` has both slice sets |
| `observation.base_pose` | f64 × 7 | pelvis pos (m) + quat **wxyz**, world |
| `observation.base_vel` | f64 × 6 | pelvis lin vel (world), ang vel |
| `observation.object_poses` | f64 × 7·N | per slot pos + quat wxyz; slots `target` (the red bottle), `distractor_0..2` (NaN when absent), plus `ball_small`, `ball_large` when the balls are on |
| `teleop.navigate_command` | f64 × 4 | the walking policy's command vx, vy, vyaw; the 4th slot (target_yaw) is always 0 |
| `teleop.latest_obs` | f32 × 65 | newest reference frame: dof_pos 29, dof_vel 29, root_pos 3, root_quat wxyz 4 (dof in real order) |
| `policy.mode` | i64 × 1 | 0 walking (velocity), 1 motion tracking |
| `policy.raw_action`, `policy.target_real` | f32 × 29 | policy output; the joint targets in real order (identical to `holomotion.action`) |
| `observation.eef_state`, `action.eef`, `observation.img_state_delta`, `teleop.base_height_command`, `observation.torso_rpy_command` | | zeros, kept for schema compatibility with the psi0 datasets |

**Revo 2 hands** (per side `s`):

| column | shape | content |
|---|---|---|
| `states.revo2.s.joint_position` | f32 × 11 | joint positions, MJCF order |
| `states.revo2.s.motor_position` | f32 × 6 | thumb_rotation, thumb_bend, index, middle, ring, pinky (the proximal joints) |
| `states.revo2.s.joint_velocity`, `joint_effort` | f32 × 11 | from the real-format sampler |
| `action.revo2.s.cmd_position` / `motor_target` | f32 × 11 / 6 | hand target sent that step |
| `action.revo2.s.grip_mode` | i64 × 1 | 0 open, 1 closed (grip button), 2 pinch (trigger) |

**Tactile pads** (per side `s`; the 5-wide columns are in the order thumb, index, middle, ring, pinky; sensor model in `src/simple/sensors/revo2_touch.py`):

| column | shape | unit / meaning |
|---|---|---|
| `tactile.s.force_normal` | i32 × 5 | SDK counts, 0.01 N (0–2500 = 0–25 N, saturates) |
| `tactile.s.force_tangential` | i32 × 5 | 0.01 N |
| `tactile.s.force_direction` | i32 × 5 | degrees, 0 = toward the fingertip, clockwise; 65535 = none |
| `tactile.s.proximity` | i64 × 5 | capacitance-like counts, 1 058 000 · (1 − d / 1 cm) |
| `tactile.s.status` | i32 × 5 | SDK status |
| `tactile.s.force_normal_N`, `force_tangential_N` | f32 × 5 | newtons |
| `tactile.s.proximity_m` | f32 × 5 | metres; −1 = nothing within 1 cm |
| `tactile.s.contacts` | i32 × 5 | contact count on the pad |
| `tactile.s.taxels_N` | f32 × 135 | 5 pads × 3 channels (z, x, y in the pad frame) × 3 × 3 grid (row = pad +y, col = pad +x toward the tip). **Do not use as recorded (2026-10-27):** MuJoCo's touch_grid counts every contact in front of the pad, not only contacts on it, and negates all three channels when the pad is the second geom of a contact; in kitchen5 ep 1 the grid total matches the pad's own normal force within ±20% on only 2-4% of touching samples. The per-pad columns (`force_*`, `proximity*`, `contacts`, `status`) come from the pad's own contacts and are fine. |

**Images** (videos, H × W × 3 = 360 × 640 × 3 by default; models and frames in 1.3–1.4):

- `observation.images.ego_view_left`, `observation.images.ego_view_right`, `observation.images.wrist_left`, `observation.images.wrist_right`.
- `wrist_cams.fresh` (i8 × 1): 1 when the wrist frame is a new render, 0 when the previous one is repeated.

**Real-G1-recorder fields** (`meta/realformat.json` explains each; verified against a real recording):

| column | shape | convention |
|---|---|---|
| `states.robot.joint_velocity`, `joint_effort` | f32 × 29 | real joint order |
| `states.robot.root_pose` | f32 × 7 | pos + quat **xyzw** (⚠ differs from `observation.base_pose`) |
| `states.robot.root_velocity` | f32 × 6 | world linear + body angular |
| `states.robot.imu_quaternion` / `imu_rpy` / `imu_gyroscope` / `imu_accelerometer` | f32 × 4 / 3 / 3 / 3 | pelvis IMU (`pelvis_imu_link`): quat xyzw, rpy extrinsic xyz, accelerometer includes gravity |
| `joint_targets.kp`, `joint_targets.kd` | f32 × 29 | PD gains used that step |
| `reference_qpos` | f32 × 36 | root pos, root quat wxyz, 29 dof |
| `holomotion_obs.ref_future_root_ori_robot_frame_6d` / `ref_future_yaw_delta_sin_cos` / `ref_robot_yaw_error_sin_cos` | f32 × 60 / 20 / 2 | as the real recorder computes them (see realformat.json; the 6D is relative to block 0, not to the robot, despite its name) |

**HoloMotion controller interface** (`meta/holomotion_raw.json` → `columns` explains each):

| column(s) | content |
|---|---|
| `reference_actions` (f32 × 715 = 11 × 65), `reference_actions_frame_index` (i64 × 11) | the policy's reference window after this step's frames were fed: block 0 = current, blocks 1–10 = future; block k at tick t equals block 0 at tick t + k. **What a VLA has to produce**; every recording must keep it |
| `holomotion.MotionReference.*` | the newest reference message and every frame received this tick (`frames` 16 × 65, `frames_meta` 16 × (frame_index, stamp)) |
| `holomotion.LowState.*`, `holomotion.VRState.{left,right}_controller` (11 fields), `holomotion.pico` (12) | what the node was fed (body q/dq, IMU, PICO controllers, raw PICO sample) |
| `holomotion.policy_obs` (776, zero padded), `policy_obs_dim`, `policy_ran`, `holomotion.obs.*`, `holomotion.obs_velocity.*` | the exact ONNX input and its named terms for the motion and walking policies |
| `holomotion.actions_onnx`, `holomotion.action`, `holomotion.kps`, `holomotion.kds` | raw ONNX output, the published joint targets and gains (real order) |
| `holomotion.tick_time` | wall clock at the start of the tick |

### 2.5 Per-episode metadata (`meta/episodes.jsonl`, JSON strings)

| field | content |
|---|---|
| `scene_setup` | seed, robot start offset, target xy, plate id, distractors (the random setup) |
| `init_pose` | random start-pose offsets (joint dq and base) |
| `realformat` | `success` (the task's last gate latched), `success_frame`, model sha256, backpack and head-tilt settings |
| `exact_log` | path of the replay log |
| `holomotion_start` | the controller's state at frame 0 (warm or cold start) |
| `environment_config` | SIMPLE's scene state |
| `fill` | filled datasets only |

### 2.6 Exact log and replay

`replay/episode_XXXXXX.npz` holds:

- the MuJoCo integration state before frame 0;
- ctrl and step counts for every physics call, and applied forces when they change;
- qpos/qvel after every frame;
- the actions;
- the setup, the scene knobs (`env_knobs`), the run metadata and the controller's start state.

```bash
python -m simple.cli.replay_holomotion_v14 <level dir> --all                 # raw inputs, compares every frame bit for bit
python -m simple.cli.replay_holomotion_v14 <level dir> --all --mode action   # through the robot's PD
python -m simple.cli.replay_holomotion_v14 <level dir> --episode 0 --video ep0.mp4
```

The replay rebuilds the scene through SIMPLE from the stored knobs. A kitchen recording made before the knobs existed has no `KITCHEN_REV`, and it is rebuilt as revision 1, the scene before 2026-10-26 evening.

### 2.7 Existing full recordings with this robot (kitchen, all replay BIT-EXACT on 2026-10-26)

| dataset | episodes (frames) | note |
|---|---|---|
| `data/teleop_holomotion_v14_kitchen2/.../level-3_pinhole` | 1 (9001) | user's PICO session, rev 1 scene |
| `data/teleop_holomotion_v14_kitchen3/.../level-3_pinhole` | 3 (4682, 1415, 7495) | user's PICO sessions, rev 1 scene |
| `data/teleop_holomotion_v14_kitchen_plates_smoke/.../level-3_pinhole` | 3 (390, 982, 899) | scripted smoke, in the bundle as the sample |

None of them has success = true. Recordings made before the 10-26 pad-frame fix (robot file 16:27) have mirrored left-pad
taxel and ray directions; the forces are unaffected, and none of the datasets above is affected.

### 2.8 Traps

- **State and action orders differ** in where the hands sit (2.4). `meta/modality.json` gives both slice sets.
- **Quaternions:** `observation.base_pose` and the reference frames are wxyz; `states.robot.root_pose` and `imu_quaternion` are xyzw.
- **Pad saturation:** closing an empty hand in sim presses the pads together at up to 70 N, so `force_normal` saturates at 2500. The sim has no motor current limit; the real hand stalls near 10–15 N.
- **Loading:** `LeRobotDataset(..., video_backend="pyav")`. The venv's torchcodec cannot load the videos.
- **Folder hygiene:** never put extra `.mp4` / `.parquet` files inside a dataset folder. The exporter counts them when saving the next episode and aborts.
- **Save time:** saving an episode takes about 2.8 ms per frame while the sim stands still (22 s for 2.6 min). Ctrl-C during a save is now deferred until the save completes.

## Provenance of the Revo 2 hand model vs BrainCo's official repos (checked 2026-11-04)

Our hand geometry is BrainCo's own CAD export, but it reached us through Unitree's `xr_teleoperate/assets/brainco_hand` copy of the
old `BrainCoTech/stark-serialport-example` (branch `revo2`; repo archived 2026-05-27, branch gone) and the HoloMotion G1+BrainCo module
URDF (same lineage). It is NOT the current official package. BrainCo's `revo2_description` (ROS 2) and `revo2_description_ros1` carry
identical URDFs, v1.1.1 of 2026-08-20 ("aligned with the mass-production version 20260818"); BrainCo's newer `brainco-description`
(release 20261025) still ships the OLD revo2 kinematics, so the two official sources disagree and v1.1.1 is the newer statement.

Differences ours -> v1.1.1 (per hand): joint origins identical to 0.01 mm; proximal limits 84 -> 80.79 deg; distal coupling 1.155
everywhere -> 1.131 / 1.267 / 1.252 / 1.271 (index / middle / ring / pinky), thumb PIP 1.0 -> 1.0945 (limit 60 -> 66 deg); thumb
abduction zero +3 deg (limit 87 -> 89) and thumb MCP / PIP zero compensation +9 / +6 deg. Fingertips: fingers identical at the zero
pose, 3-6 mm apart fully closed; the THUMB tip is 11.6 mm away at zero and 12.6 mm closed (the one difference that matters for
pinches and the tuned grip poses). Meshes: 10 of 17 per hand identical; palm, thumb proximal/distal and the four finger distal links
were re-exported with the fingertip pad split into `*_touch_link` (union shape identical, thumb frames yawed 180 deg). Our pads sit
on the official touch-joint origins (fingers exact, left thumb 0.3 mm / 1.6 deg). Palm inertia is ~10x too small in ours AND in
v1.1.1 (brainco-description has the plausible value); harmless while the palm is welded to the arm.
Scripts, JSON and an overlay render: `docs/revo2_official_compare/` (clone commands in the script headers). A robot v4 with the
v1.1.1 kinematics has not been built; recordings pin their `robot_mjcf`, so old replays would stay valid.

### The official-hand robot (2026-11-04 night)

Built: `data/robots/g1_rot0_teleop_revo2_official/` (MJCF + `_notouch.xml`, meshes, `package.json`, `touch_pads.json`, converted
`grip_poses.json`, exported `g1_rot0_teleop_revo2_official.urdf`, camera JSONs copied). Same robot, cameras, mounts and names as v3;
hands = BrainCo revo2_description v1.1.1 (source copy in `data/robots/g1_revo2_official/`, mount from the HoloMotion module). Fingertips
match the official URDF FK to 0.1 mm; BrainCo's pad links carry the same three sensors per pad as v3. Selected with
`--revo2-hand official` (`g1_sonic_revo2.select_hand_variant`); recordings carry `revo2_hand` + `robot_mjcf`, the replay follows
`robot_mjcf`. Details: docs/holomotion_v14_teleop.md, "The official BrainCo hand as a second Revo 2 robot".

### 2026-11-05: official-hand touch zones rebuilt, comparison page

The first official build (2026-11-04) had the finger touch-site boxes centred 11 mm inside the finger with a 17 mm radial half
size (the pad frame's finger axis went through the distal link's origin instead of the tip link). `add_revo2_touch.py` fixed, the
official robot rebuilt in place (`g1_rot0_teleop_revo2_official.xml`, `touch_pads.json`, `.urdf`; counts unchanged), press test
touch_total = contact normal (23.9 N index, 26.3 N thumb), smoke 2/2 BIT-EXACT. The legacy v3 LEFT hand carries the same oversized
boxes and is unchanged (recordings replay on it); `--tip-axis` builds a tight-box legacy variant into a new file name.
Visual comparison of the two hands (3D, renders, tables): `docs/revo2_hand_compare/index.html` (NAS fleet_status/revo2_hand_compare/).
