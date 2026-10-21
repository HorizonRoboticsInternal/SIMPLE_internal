# HoloMotion v1.4 teleop in SIMPLE (teleop-collection branch)

Sim teleoperation of the G1 in the three real-to-sim scenes (bottle → bin, bowl → sink, coffee cart) with the same
controller the robot runs for v1.4 PICO collection: HoloMotion's `feat/v14-teleop-collection` branch
(`robot-lab-internal/open-source/holomotion`, `deployment/`).

## What runs

```
PICO headset + controllers
  └─ publisher.py            (holomotion_teleop conda env)
       body   → SMPL → GMR (the robot's PicoSmplGmrRetargeter) → obs65 frames ─┐  ZMQ tcp://*:6001
       controllers (v1.4.1 read_pico_control_sample)          → pico_ctrl   ─┘
  └─ teleop_holomotion_v14   (SIMPLE venv)
       main-node states (as the robot's C++ main_node): WAIT → MOVE_TO_DEFAULT → POLICY, ZERO_TORQUE, E-STOP
       policy node: the robot's HoloMotionPolicyNode, unchanged (walking model on A, motion model on B)
       Dex3 hands: the v1.4 grip gripper (grip > 0.5 → close pose)
       → G1 PD torques with the policy's own kps/kds, MuJoCo at 200 Hz, policy at 50 Hz
```

The robot's policy code is vendored unchanged in `third_party/holomotion_v14/` (commit in its `SOURCE.md`); SIMPLE only
replaces ROS: `src/simple/teleop/holomotion_v14/` (ROS stubs, the sim node, the ZMQ channel, the publisher) and
`src/simple/agents/holomotion_v14_agent.py`. Refresh the vendored copy with `third_party/holomotion_v14/sync.sh <checkout>`.

## Setup (once)

- Models, in the layout the robot's loader expects: `data/holomotion/v14_models/`
  - `velocity_tracking_model/` = HuggingFace `HorizonRobotics/HoloMotion_models/HoloMotion_velocity_tracking_model`
  - `motion_tracking_model/` → `data/holomotion/models/motion_tracking_model_v141.*` (identical to the official
    v1.4.1 release, SHA256 `2aabb53b…`)
- The backpack model (`model_22000`, the default motion model; 1.6 GB, not in git). Copy it from the NAS (NAS-28,
  `10.40.11.28:/volume1/NAS-28`, mounted at `/mnt/nas28`):

  ```bash
  rsync -a /mnt/nas28/alan.jiang/holomotion_models/v14_models_backpack_3p2/ data/holomotion/v14_models_backpack_3p2/
  (cd data/holomotion/v14_models_backpack_3p2 && sha256sum -c SHA256SUMS)
  ```

  The NAS folder is the complete loader layout with its links resolved: `motion_tracking_model/` (model_22000 +
  config), `velocity_tracking_model/` (the public walking model), `README.txt`, `SHA256SUMS`.
- The publisher env `~/miniconda3/envs/holomotion_teleop` (torch, smplx, smpl_sim, GMR, mink, pyzmq, xrobotoolkit_sdk)
  and an `SMPL_NEUTRAL.pkl` (found automatically in `~/wrk/HoloMotion_TELEOP/assets/smpl` or
  `~/wrk/robot-locomanip/models/smpl`; else `--smpl-dir` / `HOLOMOTION_SMPL_DIR`).
- The XRoboToolkit PC service running, headset connected, body tracking on.

## Run

```bash
scripts/teleop_holomotion_v14.sh --scene bottle_bin            # or bowl_sink, coffee_cart
scripts/teleop_holomotion_v14.sh --scene bowl_sink --record     # LeRobot episodes in data/teleop_holomotion_v14/
```

or the two processes by hand:

```bash
~/miniconda3/envs/holomotion_teleop/bin/python src/simple/teleop/holomotion_v14/publisher.py --source pico
.venv/bin/python -m simple.cli.teleop_holomotion_v14 --scene bottle_bin --record
```

Without a headset, `--source synthetic` publishes a standing reference and scripted controller input:
`--press 1:L3 --press 5:A --stick 8:12:left_y=0.5 --press 14:B` (times in seconds, sticks −1…1).

## Buttons

The sim map (`--button-map sim`, the default):

| PICO | does |
|---|---|
| A | walking mode (policy on; lowers the leash with `--no-quick-start`) |
| Y | motion tracking: the robot follows your body |
| A (in motion tracking) | back to walking |
| B | nothing on the robot (free for the PICO app's view toggle) |
| left stick / right stick x | walk (0.5 m/s fwd, 0.2 m/s side) / turn (0.7 rad/s), in walking mode |
| grips | close the Dex3 hands |
| X | zero torque |
| right stick click | emergency stop (2 s damping, then zero torque) |
| left stick click | stand up (MOVE_TO_DEFAULT, 3 s); only needed with `--no-quick-start` or after X / E-stop |

`--button-map robot` is the robot's own v1.4.1 map: B = motion tracking, Y or A = back to walking.

Sim only. Hold the left menu button; while it is held nothing reaches the controller:

| PICO | does |
|---|---|
| left menu + A | start recording / save the episode |
| left menu + B | abandon the episode |
| left menu + Y | skip to a new random setup |
| both triggers | reset the scene, same setup (a retry) |

An episode is also saved automatically when the scene's task check succeeds (the item placed). After every save the
scene is built again from a new random setup.

## Scenes

The scene kits are found in `HOLOBRAIN_SIM_DIR` if set, else `~/wrk/robot_orchard_deploy/holobrain_g1_deploy/sim` (the
workstation's working copies), else this repo's `scenes/`. Each recorded episode notes which folder it used, and the
replay rebuilds from the same one.

The robot starts where each scene puts it, except bottle → bin. That scene's 3 cm (feet to table) was fitted to
decoupled-WBC replays that start with the hands raised. The v1.4 controller stands up with its forearms at table height,
so at 3 cm the hands land on the table and the robot leans on it. For v1.4 teleop the CLI starts it 0.35 m back, and
the operator walks in, as in the real collection. `--start-distance` overrides this for any scene (the table, item and
bin move together; only the robot's start changes).

| scene | start (feet to …) | knob |
|---|---|---|
| bottle_bin | 0.35 m to the table (scene default 0.03) | `BOTTLE_BIN_ROBOT_TO_EDGE` |
| bowl_sink | 0.38 m to the counter | `BOWL_SINK_ROBOT_TO_EDGE` |
| coffee_cart | 0.20 m to the cart | `COFFEE_CART_ROBOT_TO_CART` |

## Quick start (the default)

```bash
scripts/teleop_holomotion_v14.sh --scene bottle_bin --backpack-kg 3.2 --record
```

Every scene starts with the robot standing on the floor at the policy's default pose, with no leash. That covers the
first scene at launch and every scene after a save, left menu + Y or both triggers.

1. The walking policy goes on at the first step. The CLI presses A for you, through the same code path as the real
   button, with a neutral stand-in controller sample if the PICO isn't connected yet, so the robot balances on its own.
2. The robot stays in walking mode; press Y for motion tracking when ready. `--auto-motion` presses it for you
   (Y; B with `--button-map robot`) 0.5 s after the walking policy is on.
3. Recording starts on the first step your PICO controllers are there: frame 0 is the start pose (random with
   `--random-init-pose`). With `--auto-motion` and a fixed start pose it starts at motion-on instead. Left menu + A
   saves. `--record-on-motion` instead starts every recording, the first included, when you press Y (motion
   tracking). The scene waits in walking mode with "not recording (Y: motion + record)" on the overlay. Left menu + Y
   (new setup) does not trigger it; after an abandon, Y starts a new recording.

Episodes recorded this way never have the leash up, so they replay bit-exactly from the actions alone.
`--no-quick-start` brings back the old flow: hanging on the leash, then L3, A, B and left menu + A by hand. Without
the leash and without the policy, the robot does not stay up: held in its start pose by the stand-up gains, it fell
backwards within 3 s with the backpack.

### Random start pose

`--random-init-pose` (the default since 2026-09-30; `--no-random-init-pose` turns it off) gives every scene init its own small random robot pose, so no two episodes start from the same
state. The pose is drawn from `--seed`, and the offsets are uniform:

| part | range at `--init-pose-scale 1` |
|---|---|
| arms (shoulder, elbow, wrist) | ±0.15 rad |
| waist | ±0.05 rad |
| legs | ±0.03 rad |
| base | ±2 cm in x and y, ±3° yaw, on top of the setup's robot start |

With quick start the robot is placed in that pose and recording starts on the first step the controllers are connected,
so frame 0 is the randomized state. If the PICO connects after the scene started, the same setup starts over from a
fresh random pose. Each episode therefore begins with the ~1 s automatic hand-over (walking policy, then motion
tracking). With `--no-quick-start` the offset is applied to the robot hanging on the leash.

Each episode's offset is saved as `init_pose` in `meta/episodes.jsonl` and in its replay log. In a scripted test the
first frames of 4 episodes differed by 0.19–0.28 rad (largest joint difference), and all 4 replayed bit-exactly.

## Robot and headset camera

The default robot (`--robot stereo`) is the G1 with the HBVCAM-F2439GS-2 stereo fisheye head camera, the v1.4
collection robot's camera. It is vendored in `third_party/hbvcam_stereo/` from the `docs/_scan/g1_head_stereo_camera`
bundle.

- **Model:** SIMPLE's G1 plus the two eye frames and a visual-only camera body. It has the same bodies, masses, joints
  and actuators, so the physics is unchanged. `robot_variants.teleop_mjcf()` adds the 3.2 kg backpack and one camera,
  `hbvcam_left_pinhole`.
- **The pinhole camera:** the stereo calibration's rectified left eye (R1, P1): 1280 × 720, f 493.6 px,
  104.7° × 72.2°, at the left optical frame. Points project to within 0.1 px of OpenCV with P1.
  MuJoCo's principal-point offset has the opposite x sign to OpenCV's (measured), so the camera uses (W/2 − cx, cy − H/2).
- **The headset** gets that picture (`--stream-camera pinhole`). `--stream-camera head` streams the scene's head camera
  instead; `--robot stock` uses SIMPLE's G1.
- **`--stream-camera fisheye`** sends both eyes' raw fisheye images side by side (2560 × 720: left eye in columns
  0–1279, right in 1280–2559), the frame the real camera streams over USB. The PICO app shows it as two pictures, or
  in 3D with depth.
  - How: `hbvcam_views.FisheyeView` renders each eye's four 90° face cameras (`hbvcam_{eye}_{front,left,right,up}`;
    "down" never enters the lens) at 960 px. It remaps them with a per-pixel lookup built once from the calibration's
    fisheye rays (`third_party/hbvcam_stereo/hbvcam_fisheye.py`).
  - The output is identical to the bundle's `render_hbvcam` (mean difference 0.000).
  - The headlight is made direction-free in the rendered scene only, so the faces show no seams; the model is not
    touched.
  - Cost: one eye per sim step (~3.6 ms), the newest left + newest right sent at the stream's 30 fps (eyes at most
    20 ms apart). Measured: the sim at 49.9 Hz, the headset ~28 fps.
  - **Recorded too:** with `--stream-camera fisheye`, `ego_view` is the fisheye pair (2560 × 720, both eyes
    rendered from the same pre-step state every step). It goes to a separate dataset folder, `level-N_fisheye`, since
    a dataset holds one image size. The headset stream reuses that render.
    - Measured: sim 49.8 Hz, headset ~24 fps.
    - `--record-camera auto|head|pinhole|fisheye` picks the recorded view explicitly; `auto` (default) records the
      headset's camera: pinhole with the pinhole stream (the default), fisheye with the fisheye stream.
  - The headset frame is built with cv2 colour conversion into one buffer (0.8 ms; it was 6 ms and slowed the sim).
  - `--fisheye-face-px` sets the face size. The default 960 is about the lens's own sharpness (480 vs 471 px/rad at
    the centre); 640 is cheaper and softer, at the same frame rate.
  - Renders of all three scenes (pinhole, fisheye, head camera):
    http://10.40.11.11:8899/g1_head_stereo_camera/index.html, section "In the sim teleop scenes".
- **The recorded `ego_view`** is, by default, what the headset shows: the pinhole, 1280 × 720, folder
  `level-N_pinhole`. Before 2026-09-30 it was the scene's head camera; `--record-camera head` still records that one
  (640 × 360, 90°). This robot has no `d455_link`, so that camera sits at SIMPLE's stock head mount on the torso,
  looking 59° below horizontal when the robot stands in the default pose. The untilted pinhole looks 46° down.
- **Camera tilt:** `--head-tilt-deg T` (default 10; 0 = the calibrated mount) pitches both head cameras T° further down (− = up), each about its own
  position. That covers the HBVCAM cameras (pinhole and fisheye faces, turned in the robot MJCF, file suffix
  `_tiltp10`) and SIMPLE's head camera (its sensor pose, through the kits' head-camera pose patch, the same composition
  as `<KIT>_HEAD_TRIM_DEG`).
  - Measured at +10 and +20: every camera turned exactly that much, positions unchanged.
  - +10 keeps the item in view from the start spot and shows both hands at the table. At +20 the head camera loses
    the bottle from bottle_bin's start spot.
  - The tilt goes into each exact log (`head_tilt_deg`) and the replay rebuilds it (2 episodes at +10, bit-exact).
  - The scene kits' own knobs (`<KIT>_NECK_TILT_DEG`, `<KIT>_HEAD_TRIM_DEG`) do not act in the teleop: the first
    moves the g1comp robot's neck, which the teleop robot replaces; the second is zeroed while the kit's robot is
    g1comp.
- **The model hash is unchanged:** the stream renderer sets the 1280 × 720 off-screen buffer only while it is created,
  so the model the exact log hashes stays byte-identical.

`--stream-port` (default 13579, the port the app uses) exists for testing. A stand-in for the app that sends
OPEN_CAMERA and decodes the returned H.264 received 31 fps of 2560 × 720 frames with the pinhole picture and the
overlay.

## Headset view

The XRoboToolkit app expects a stereo camera (2 × 16:9 side by side). By default it shows that frame as two pictures;
its right **B** toggles a single view that shows only the left half. `--stream-view` picks what SIMPLE sends:

| view | frame | what you see |
|---|---|---|
| `single` (default) | the left head camera once, centred | one picture in the app's default view |
| `mono` | the left camera in both halves | one picture after B (the app's single view); two identical ones before |
| `stereo` | left \| right | the old dual stream |

With the sim button map, B does nothing on the robot, so use it freely for the app's view: if the picture looks cut in
half, press B. (With `--button-map robot`, B is also motion tracking, so every B press flips the view as well.)

A status overlay shows:

- recording state and time (`REC 12.3 s`, `starting ...`, `not recording`);
- the mode (WAIT / STAND / WALK / MOTION), episodes saved and the setup number;
- the task gates (e.g. `grasped [x]  at_bin [ ]  placed [ ]`);
- the sim-only combos.

## Random setups

Every episode starts from a new setup drawn from its own seed (`--seed N`: the n-th setup uses N + n; default a random
N, printed). A setup holds exact values, offsets from the nominal layout above:

| scene | robot start (x, y, yaw) | item | furniture | distractors |
|---|---|---|---|---|
| bottle_bin | ±8 cm, ±10 cm, ±10° | bottle −4…+10 cm deeper, ±12 cm along the table | bin ±12 cm, ±12 cm | 0–3 on the table |
| bowl_sink | ±8 cm, ±10 cm, ±10° | bowl 0…+10 cm deeper, ±15 cm along the counter | – | 0–3 on the counter |
| coffee_cart | from the cart: −4…+5 cm, ±5 cm, ±6° | cup anywhere across the box top, first 20 cm from the handle | cart (with box and cup) ±5 cm, ±8 cm | 0–3 on the delivery table |

Distractor type and placement come from SIMPLE's distractor DR (GraspNet objects) seeded by the setup, so they change
with every setup too. The table height stays nominal with these ranges.

### Level-3 ranges (`--setup-ranges lv3`, the default since 2026-09-30; `--setup-ranges teleop` = the table above)

The ranges of the kits' level-3 evaluation sets (`holobrain_g1_deploy/sim/make_levels.py`), one uniform draw per setup
(make_levels spreads its 10 scenes evenly over the same box):

| scene | robot start (x, y; no turn) | item | table height | distractors |
|---|---|---|---|---|
| bottle_bin | ±10 cm, ±5 cm | bottle ±3 cm deeper, ±8 cm along the table | table ±4 cm (legs and cover board follow) | 3 on the table |
| bowl_sink | ±10 cm, ±5 cm | bowl on the edge line, ±8 cm along the counter | counter and sink ±4 cm | 3 on the counter |
| coffee_cart | −20…0 cm, ±5 cm | cup on the box top, 6–23.5 cm from its near edge, across the full safe width | delivery table and cart box ±4 cm | 3 on the delivery table |

The bin and the cart stay where they are, as in the level-3 sets. The table offset is in whole millimetres, because the
kits build their table-leg, counter and cart meshes per millimetre (`<asset>_p012` / `_m034`). coffee_cart's 20 cm robot
span sits 10 cm further back than level 3's ±10 cm. In the v1.4 standing pose the fingers reach the cart handle about
5 cm ahead of the nominal start, and the check (2026-09-30) found the hands inside the cart for the forward half. With the
shift, the closest start leaves 5.6 cm; bottle_bin and bowl_sink keep 10 cm or more at every extreme. lv3 recordings go to
`level-3` (`--dr-level` otherwise names the folder). Each setup stores `ranges` and `table_dz`, and the bit-exact
replay rebuilds the height (checked: 3 coffee_cart episodes at +3.3, −3.3 and −1.2 cm, action and physics replay).

The ranges are `RANGES` / `LV3_RANGES` in `src/simple/teleop/holomotion_v14/scene_setup.py`; `--max-distractors`,
`--no-randomize` (the nominal scene). The setup goes into the scene through SIMPLE's own DR (zero-width robot/target regions, the
distractor count, numpy's seed) and the scene module's layout dict; the scene task files are unchanged. Each episode's
setup is in `meta/episodes.jsonl` (`scene_setup`, next to SIMPLE's `environment_config`).

## Recording

LeRobot v2.1 at 50 fps, the decoupled-WBC schema (as the psi0 datasets) plus:
`teleop.navigate_command` (the policy's vx, vy, vyaw, 0), `teleop.latest_obs` (the 65-value reference),
`policy.mode` (0 walking, 1 motion), `policy.raw_action`, `policy.target_real`. `action` is the 43-joint target
(29 body + 2 × 7 Dex3). `observation.object_poses` has fixed slots: the scene's objects, then `distractor_0..N-1`
(NaN when a setup has fewer).

## Evaluating a VLA on this controller

`python -m simple.cli.eval_holomotion_v14` runs a VLA with this controller and these defaults: the VLA returns the
reference frames model_22000 tracks (`teleop.latest_obs`), plus grips or hand targets. See
[holomotion_v14_eval.md](holomotion_v14_eval.md).

## The real recorder's fields (recorded live)

Since 2026-09-30 every recording also carries the extra fields of the real G1 recorder (HDF5 format_version 1.1), the
ones `scripts/holomotion_sim_realformat.py` used to derive afterwards from the replay logs: 19 columns per frame
(`--no-realformat` turns them off).

| columns | contents |
|---|---|
| `joint_targets.kp`, `.kd` | PD gains of the frame's action, 29 joints |
| `states.robot.joint_velocity`, `.joint_effort` | 29 joint velocities and applied actuator forces |
| `states.robot.root_pose`, `.root_velocity` | pelvis position + quaternion (xyzw); world linear + body angular velocity |
| `states.robot.imu_quaternion`, `.imu_rpy`, `.imu_gyroscope`, `.imu_accelerometer` | the pelvis IMU at the stock mount (accelerometer with gravity) |
| `states.dex3.{left,right}.joint_velocity`, `.joint_effort` | the hands, real order (thumb 0–2, index 0–1, middle 0–1) |
| `reference_qpos`, `reference_actions` | the frame's reference, and the 11-block queue (current + 10 future frames) |
| `holomotion_obs.*` | the three reference terms the real recorder computes |

The values are sampled on a copy of the MuJoCo state right before each step (the state `observation.state` is taken
from), so the physics and the replay logs are unchanged. The reference columns need the next 10 frames and are filled
in when the episode is saved. `meta/realformat.json` holds the format spec, and each episode's `realformat` entry in
`episodes.jsonl` the model hash, seed, backpack and tilt.

Checked: the converter run on a live recording recomputes every column from the replay log and compares. All 19 match
bit for bit (two episodes, 558 and 1116 frames), and the recording replays bit-exact. Running the converter on such a
dataset now verifies the live columns instead of appending duplicates.

Not recorded live: the Isaac ego-view render of `level-3_pinhole_realformat` (videos re-rendered with
`scripts/render_teleop_isaac.py`, random look per episode). That stays a post-step; it cannot run inside the live
MuJoCo loop.

## Bit-exact replay

With `--record`, each saved episode also gets `replay/episode_XXXXXX.npz` (~125 KB per second):

- the full MuJoCo integration state before the first frame;
- ctrl before every physics call, the applied forces when they change, and the mj_step counts;
- the actions (joint targets, kp, kd, Dex3 targets);
- qpos/qvel after every frame, and the hash of the compiled model;
- the setup, the scene knobs and SIMPLE's `environment_config`.

```bash
python -m simple.cli.replay_holomotion_v14 data/teleop_holomotion_v14/<env_id>/level-0 --all                 # raw inputs
python -m simple.cli.replay_holomotion_v14 data/teleop_holomotion_v14/<env_id>/level-0 --all --mode action   # through the robot's PD
python -m simple.cli.replay_holomotion_v14 <dataset> --episode 3 --video ep3.mp4                              # head | third person
```

The replay rebuilds the scene through SIMPLE from the saved setup, checks the model hash, then compares every frame's
state bit for bit. If the scene code has changed since, the hash check says so; `--save-model` at record time also
keeps the compiled model (`.mjb`, ~125 MB, mostly meshes and collision trees) for `--model mjb`.

## G1 with a backpack

This is the default. The Dex3 G1 carries HoloMotion's backpack (`backpack_link` on the torso at (−0.095, −0.0005,
0.165), from their payload assets; no collision, as in their MuJoCo eval) at 3.2 kg (`--backpack-kg`). Motion
tracking runs the v1.4.1 BrainCo 3.2 kg backpack model, `model_22000` (`--motion-model backpack`). The walking
model is the public v1.4 one; the upstream lock pins the same file for the backpack deployment.

The model lives in `data/holomotion/` (not in git; 1.6 GB). The copy on the NAS is `/mnt/nas28/alan.jiang/holomotion_models/v14_models_backpack_3p2/` (see Setup):

- `v14_models_backpack_3p2/motion_tracking_model/exported/model_22000.onnx`: the model under its original name,
  sha256 `d79ccce7…`, equal to the upstream lock.
- `v14_models_backpack_3p2/`: the folder the policy node loads, with a README.

Its own `config.yaml` (lock `f64f778a…`) was not supplied, so the folder uses the public v1.4.1 config. That is safe:

- the node takes joint names, gains, action scale and default pose from the ONNX metadata;
- the two models have the same observation (`[1, 604]`) and KV cache, and 5 of 6 metadata fields;
- only `default_joint_pos` differs, by up to 0.01 rad.

Other setups:

| flags | robot | motion model |
|---|---|---|
| (default) | 3.2 kg backpack | backpack `model_22000` |
| `--motion-model public` | 3.2 kg backpack | public v1.4.1 `model_16200` |
| `--motion-model public --backpack-kg 0` | no backpack | public v1.4.1 (the plain robot) |
| `--motion-model-bundle DIR` | as `--backpack-kg` | another bundle, hash-checked against the lock |

In a scripted drill on the backpack robot (2026-09-29), the backpack model held motion tracking with about half the
public model's yaw drift: 0.6° vs 1.3° over 4 s, and 0.9° vs 1.8° over 6 s. Switching back to walking, its heading
jumped 1.2° against 5.3°. Walking is identical (same model). Neither fell.

## Verified, and not

Verified in all three scenes with the synthetic publisher (headless):
- stand, policy on, band lowered, standing;
- walking back, forward and turning at the commanded speeds;
- motion mode with yaw alignment, and back to walking;
- Dex3 grip close/open;
- a recorded and saved episode with all fields.

Also verified (2026-09-28), scripted in all three scenes:
- a new setup after every save, the retry and skip gestures;
- 7 recorded episodes replayed bit-exactly in both modes, including band-up, stand-up, walking, turning, grip and
  motion mode;
- the 3.2 kg backpack with the public models: stand, walk, strafe, turn, motion hold, no falls. Commanded −0.5 m/s
  backward gave 0.49 m/s with the backpack and 0.28 m/s without; motion-mode yaw drift was 6.8° vs 1.7° over 6 s.
  `model_22000` itself is not tested yet.
- quick start (2026-09-29, scripted, 3.2 kg backpack): three resets (after a save, a retry, a save). Each came up
  standing, pelvis 0.76 m, motion tracking and recording on within ~1 s, no falls. The quick-started episodes replay
  bit-exactly in action mode with no leash force.

The motion model runs in ~3 ms per step on CPU and the walking model in ~0.5 ms. The real retargeter
(SMPL + GMR, `holomotion_teleop` env) initialises and converts a body pose in ~30 ms.

Not yet verified: a live PICO session (headset body tracking → motion mode). The publisher reads the SDK exactly as
v1.4's own workstation node does (`get_body_joints_pose`, 24 × (x, y, z, qx, qy, qz, qw)), but it needs a first run
with the headset.

## 2026-10-16 update: the HoloMotion interface recorded under its own names, rot0 camera, both eyes

- Every recording carries the HoloMotion v1.4 interface per 50 Hz control tick, named as HoloMotion names it (layouts and
  the message -> column map in the dataset's `meta/holomotion_raw.json`):
  - `holomotion.MotionReference.{dof_pos, dof_vel, root_pos, root_rot_wxyz, frame_index, header.stamp}`: the newest
    reference frame fed this tick (holomotion_interfaces/msg/MotionReference; joints in `complete_dof_order`). This is what
    the PICO operator provides today and what a VLA must provide. `MotionReference.frames` / `frames_meta` /
    `frames_this_tick` keep every frame received since the previous tick (up to 4; 65 = dof_pos dof_vel root_pos root_rot_wxyz).
  - `holomotion.LowState.motor_state.{q,dq}`, `holomotion.LowState.imu_state.{quaternion,gyroscope}` (unitree_hg/msg/LowState as fed).
  - `holomotion.VRState.{left_controller,right_controller}` (robo_orchard_pico_msg_ros2/msg/VRState fields: status, a/b/x/y
    buttons, axis_x, axis_y, trigger, grip, axis_click, menu_button) and the raw `holomotion.pico` sample.
  - `holomotion.obs.<term>`: the motion-tracking policy's observation terms with the config's names and order
    (`actor_ref_gravity_projection_cur` … `actor_ref_future_root_ori_robot_frame_6d`, 18 terms = the 604-wide ONNX input);
    `holomotion.obs_velocity.<term>` the same for the walking policy (6 terms, 776). The term layout is baked in
    `src/simple/teleop/holomotion_v14/holomotion_obs_layout.json` (re-bake with `HOLOMOTION_DUMP_OBS_LAYOUT=<path>` during a
    recording with motion tracking on) and checked against the live builder.
  - `holomotion.policy_obs` / `policy_obs_dim` / `policy_ran` (the exact ONNX input), `holomotion.actions_onnx` (raw ONNX
    output, the model's joint order), `holomotion.action` / `kps` / `kds` (what the node publishes on /humanoid/action|kps|kds).
  The vendored robot code is unchanged; the capture wraps the runtime's policy call from SIMPLE's side (`sim_node.py`).
- The stereo URDF (`data/robots/g1_sonic/g1_29dof_with_hand_hbvcam_stereo.urdf`) is at the **rot0** bracket: 49.3 deg
  below horizontal on the torso (rot15 gave 34.3; the September recordings were rot15 + `--head-tilt-deg 10` = 44.3).
  `--head-tilt-deg` now defaults to 0 in the teleop and the evaluation. The previous files are kept as `*_rot15.*`;
  `replay_holomotion_v14._build_env` rebuilds old episodes with the legacy robot (`teleop_mjcf(..., legacy=True)`) when
  their model hash says so, so their exact logs stay bit-exact.
- With `--record-camera pinhole` (the default) both rectified eyes are recorded as `observation.images.ego_view_left`
  (R1/P1) and `observation.images.ego_view_right` (R2/P2), 1280 x 720 each, from the same state; the single `ego_view`
  stream is no longer written. The MJCF gets `hbvcam_right_pinhole` next to `hbvcam_left_pinhole`. The evaluation
  (`eval_holomotion_v14`) sends both eyes under the same keys.
- Confirmation page: http://10.40.11.11:8899/holomotion_v14_pipeline_update/ (built by docs/_scan/holomotion_v14_pipeline_update/build.py
  from a scripted synthetic recording in data/teleop_holomotion_v14_test_rot0).
- Replay check (2026-10-16, `docs/_scan/holomotion_v14_pipeline_update/replay_node_from_dataset.py`): a fresh node driven
  from the recorded raw inputs only (LowState, VRState, MotionReference.frames, clock on the frames' stamps) reproduces the
  motion-tracking policy's ONNX input and output bit-exactly on every motion tick from the first one (748/748 on the test
  episode); the walking-policy phase before the switch is not reproduced (its 8-step history predates the recording).
  `MotionReference.frames` keeps the newest 16 frames of each tick (`frames_this_tick` counts all arrivals).
- Cold start, replay from frame 0 (2026-10-16, later the same day): every recording also stores the controller's internal
  state at the start of its frame 0 (`replay/episode_XXXXXX.npz`: the `holomotion_start.*` arrays, written by
  `src/simple/teleop/holomotion_v14/node_state.py`; a summary with `cold_start` in `meta/episodes.jsonl` and in the exact
  log's info) and the node's clock per tick (`holomotion.tick_time`). `node_state.restore` puts a fresh node into that
  state (starting from the same fresh-start reset the agent applies at every scene reset, `node_state.fresh_start`), so
  `replay_node_from_dataset.py` reproduces the ONNX input and output bit-exactly from frame 0: cold-start recording
  (`--record-on-motion`, frame 0 = the engage tick) 744/744 policy ticks; quick-start recording (frame 0 in walking
  mode) 947/947, the walking phase included. In v1.4 the motion-tracking engage is the controller reset (last action,
  KV cache, RoPE step, yaw alignment), so `--record-on-motion` gives a cold controller start on frame 0 by construction;
  the recorder prints `COLD start` / `WARM start` when a recording begins. Collect VLA data with `--record-on-motion`.
  No per-recording environment reset is needed (the v1.3 hand-off's manual reset step): the scene is reset after every
  save and the exact log holds the physics start state (`state0`), which still replays bit-exactly.
  Two traps the snapshot had to cover: the evaluator's low-state cache is keyed by message identity and refreshed once
  per policy step, so between steps (the yaw alignment at the engage) it reads the previous tick's IMU; and a node
  straight from `setup()` carries a warm-up push in its walking history that a reset node does not.
  Test recordings: `data/teleop_holomotion_v14_test_{cold,warm}` (scripts in `docs/_scan/holomotion_v14_pipeline_update/`).

- Resume guard (2026-10-18): recording into a folder that already holds a dataset with another column set (e.g. the
  September `level-3_pinhole` sets: one `ego_view`, no `holomotion.*`) used to fail LeRobot's `validate_frame` on the
  first recorded tick, because the exporter resumes an existing folder with its own feature list. The CLI now compares
  the two at startup (`_check_resume_columns`) and exits with a message before writing anything. The default
  `--save-dir data/teleop_holomotion_v14` holds the September episodes, so new collections need a new folder (the
  runbook uses `data/teleop_holomotion_v14_rot0`). Runbook updated: docs/pico_sim_teleop_runbook.html.

- Recorded image size (2026-10-18, user: "640*360 is enough for image recording"): `--record-size` (default 640x360)
  downsizes each recorded eye with area averaging from the 1280x720 render; the headset stream stays 1280x720. The camera
  model and calibration are unchanged; meta/holomotion_raw.json `images` stores the intrinsics at the recorded size
  (fx = fy = 246.783, cx = 317.156, cy = 179.733 px; baseline 60.517 mm; x' = (x + 0.5) s - 0.5). A 15 s episode is now
  ~7 MB (each eye 0.95 MB instead of 2.7). `eval_holomotion_v14 --image-size` now defaults to 640x360 per eye, so the VLA
  gets what it was trained on (empty = as rendered). The resume guard also refuses a folder whose image size differs.
- v1.3 vs v1.4 reference (measured 2026-10-18 on data/teleop_holomotion_v14_test_640): the policy's window = the 11
  newest RECEIVED frames (current 10 behind the newest, ~14 ms apart at the 70 Hz publisher, 7-8 ticks back);
  `reference_actions` (11x79, realformat) = newest frame of ticks t..t+10, hands 0, and differs from the policy window at
  t-10 by up to 0.06 rad. Exact window: holomotion.obs.actor_ref_* or rebuilt from holomotion.MotionReference.frames.
  Runbook section 10 has the comparison table.

- Black hands and re-recording (2026-10-18, user: "the robot hand is white in the simulation, it should be black ... replay
  the holomotion inputs for these 50 eps ... save as new 50 recordings"; "in the future the robot hand should be black by
  default"): `--black-hands` (default on, teleop and eval) builds the robot as `teleop_mjcf(..., black_hands=True)`: every
  geom whose mesh is a left_hand_*/right_hand_* mesh (palm on wrist_yaw_link + fingers) gets rgba 0.10 0.10 0.10 1, as the
  scene kits draw them; only geom_rgba differs from the grey model. The exact log records `black_hands`; the replay rebuilds
  from it (logs without the key stay grey and keep matching their hash).
  `python -m simple.cli.rerecord_holomotion_v14 <dataset> --save-dir <new root>` re-records a dataset closed loop from its
  recorded HoloMotion inputs: scene rebuilt from setup + environment_config, the live quick-start placement re-done (the
  task gates' start position), frame-0 physics (state0, rest state, band), the controller restored from the frame-0
  snapshot, the agent in its frame-0 state (stored since 2026-10-18 as holomotion_start.agent_json; older logs: POLICY, no
  buttons held), fed holomotion.pico + MotionReference.frames on holomotion.tick_time, through the teleop recorder itself.
  Each frame is checked: physics bit for bit, policy ONNX I/O bit for bit, success on the source's last frame
  (meta/rerecord_summary.json). Frame 0's object poses are copied from the source (live they came from the step before
  frame 0). The user's 50 bottle_bin episodes -> data/teleop_holomotion_v14_rot0_blackhands.
  Two replay bugs found on the way: `_build_env` attached the old-episode `elif backpack` to `if tilt`, so tilt-0 episodes got
  the plain backpack robot and only the robot_mjcf override restored the recorded file (same in scripts/render_teleop_isaac.py),
  now fixed; and the robot_mjcf override would undo a requested variant (the re-recorder drops it).

- Reference window column (2026-10-19, user: "i need the reference frame at each 20 ms which is the one you feed to holomotion"):
  `holomotion.reference_window` (11 x 65 per 20 ms step) + `holomotion.reference_window_frame_index` (11) [renamed the same
  day, see the next note]: the node's
  queue after the step's frames were fed, row 0 = current (10 arrivals behind the newest), rows 1-10 = future 1-10; zeros
  until 11 frames have arrived. Agent extras raw_ref_window / raw_ref_window_index. A single 65-value frame per 20 ms
  cannot be bit-exact (the queue moved by 1-2 frames per step at 70 Hz): replaying from teleop.latest_obs diverges from
  20 ms on. Feeding the window column whole every 20 ms is bit-exact (replay_window_column_test.py: eps 0/30/47). Measured on
  the 50 episodes: the frame the model tracks is 145 ms (131-242) behind teleop.latest_obs, 1.4 deg mean / 48 deg max per joint.
  The 50 black-hand episodes were re-recorded with the column (data/teleop_holomotion_v14_rot0_blackhands; the previous copy is
  data/teleop_holomotion_v14_rot0_blackhands_20261018_nowindow). New live recordings need a new --save-dir (the runbook now says
  data/teleop_holomotion_v14_v2) because data/teleop_holomotion_v14_rot0 lacks the column and the recorder refuses to mix formats.
- Window column renamed to the v1.3 hand-off's name (2026-10-19, user: "i want the same naming"): `reference_actions`
  (715 = 11 x 65, block 0 = current, blocks 1-10 = future; v1.3's blocks were 79 = 65 + 14 hand values, user: "it only
  needs to be 11*65 slot but with reference_actions name") + `reference_actions_frame_index` (11). It replaces the realformat look-ahead column of the same name (block k = teleop.latest_obs of frame t + k), which
  is no longer written; `reference_qpos` and `holomotion_obs.*` still derive from teleop.latest_obs. The converter
  scripts/holomotion_sim_realformat.py keeps a live `reference_actions` when `reference_actions_frame_index` exists.
  Test: docs/_scan/holomotion_v14_pipeline_update/reference_actions_column_test.py. The 50 black-hand episodes were
  re-recorded with the new column (closed-loop stored-input replay: 50/50 physics + policy I/O bit-exact, 46/50 succeed, the
  same 4 hand-saved) and installed as data/teleop_holomotion_v14_rot0_blackhands; the previous copy (column named
  holomotion.reference_window) is data/teleop_holomotion_v14_rot0_blackhands_20261019_refwindowcol. All shared columns are
  identical; reference_actions == the old holomotion.reference_window on 50/50. The mp4s differ by encoder noise (mean |diff|
  1.4/255, same between any two re-records: not bit-exact video, physics is).
