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
| left menu + both triggers | reset the scene, same setup (a retry); a trigger alone is the Revo 2 pinch pose (2026-10-26) |

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
| kitchen | 0.35 m to the peninsula worktop | `KITCHEN_ROBOT_TO_EDGE`, `KITCHEN_ROBOT_Y` |

**Dishwasher door (scene revision 3, the default since 2026-10-27).** The door collides where it is drawn (front panel,
control strip, handle bar and posts; before, fingers sank 51 mm into the front and the handle collider filled the finger gap)
and its hinge has a counterbalance spring (12.4 N m/rad) and 3.5 N m friction, so it stays where it is left, like a real door;
about 8 N at the handle opens it. Recordings with `KITCHEN_REV` 2 or none keep the old door. Page: docs/dishwasher_door.

**Revo 2 hands that stay closed (fixed 2026-10-27).** A grip closed on little or nothing could wedge the opposed thumb tip
between the index and ring fingertips; with friction 1.0 between the hand's own parts (and SIMPLE's impratio 10 + noslip 2)
the fingers were then frictionally locked and stayed closed after the release, whatever was commanded (kitchen4 episode 2,
143-155 s). The Revo 2 robot now loads `g1_rot0_teleop_revo2_v2.xml`: hand-on-hand contacts at friction 0.2, hand-on-object
contacts unchanged (the 50-recording bottle replay places the same 32 episodes). Older recordings replay on the file they
name. **Recording start:** since 2026-10-27 `--record-on-motion` is the default: each recording starts when you press Y
(motion tracking engages on frame 0, a cold controller start). `--no-record-on-motion` records from the first tick the
controllers are connected (frame 0 = the random start pose). Both replay bit-exactly.

The kitchen (`--scene kitchen`, usually with `--robot revo2`) has three tasks, picked with `KITCHEN_TASK`:
`load_dishwasher` (the default since 2026-10-27: "open the dishwasher, load the plates and bowls from the counter into
its racks, push the rack back in and close the door"; gates door_open = the door past 69 deg, grasped = a hand holding a
plate or bowl 3 cm up or off the counter for 0.5 s, placed = a dish resting in the upper or lower rack with no hand on it
for 0.5 s, rack_in = both racks within 5 cm of home after a dish was placed, door_closed = the door within 5 deg of shut
with the racks in and a dish still loaded, for 1 s. Success is door_closed since 2026-10-30 (placed until then).
**The gates are not evaluated in the teleop loop any more** (2026-10-30, user: the kitchen5 pipeline without the gate
checks; they cost 0.25 ms per tick with 10 ms spikes): the recorder never stops by itself (only the env's step limit,
--max-episode-steps = 10 min, still saves), the headset shows no gate ticks, and the labels are computed on the
bit-exact replay: `replay_holomotion_v14 <level dir> --all [--fill]` (its `--gates` default) writes `task_eval` and
`realformat.success` / `success_frame` into meta/episodes.jsonl of the dataset and of the fill copy, keeping the
original line in meta/_before_gates_<date>/. `--live-gates` on the teleop CLI brings the live evaluation and the
"task done: L-menu + A to save" line back. The status line's dish count also needs --live-gates), `bottle_sink` (the default before 10-27: the kitchen's red bottle to the sink, empty bowls;
gates grasped, at_sink, placed) and `ball_bin`
(2026-10-26, an option: each bowl holds a 4 cm ball; pick up a bowl, walk to the pedal bin ahead-left and tip the ball
in; gates grasped, at_bin, placed = a ball inside the bin). `KITCHEN_BALLS=1` adds the balls to the default task too.
The pedal bin's lid is closed by default (step on the pedal on the bin's far side or lift it by hand);
`KITCHEN_BIN_LID=open` starts it open. Recordings store `KITCHEN_REV`, `KITCHEN_BALLS` and `KITCHEN_BIN_LID` and the
replay rebuilds from them; recordings made before these knobs replay as `KITCHEN_REV=1`. The kitchen page
(docs/kitchen_teleop) has the views, the numbers and the physics checks.

The task only labels the recording and drives the gates; physics is the same for all three, so a recording can be
relabelled later. `scripts/holomotion_v14_checks/relabel_kitchen_task.py <level dir> [--task T] --write` replays every
episode bit-exactly with the new task's gates and rewrites `meta/tasks.jsonl` and `meta/episodes.jsonl` (`tasks`,
`realformat.success` / `success_frame`, and a `task_eval` entry with each gate's first frame), in the dataset and its
`_filled` copy, after copying both files to `meta/_before_relabel_<date>/`. On 2026-10-27 kitchen4 and kitchen5
(recorded under bottle_sink while loading the dishwasher) were relabelled to load_dishwasher: kitchen4 ep 0 and 2 and
kitchen5 ep 1 succeed (one or two plates in a rack), kitchen4 ep 1 (plate left in the sink) and kitchen5 ep 0 (plate fell
onto the open door) do not.

Review pages: `scripts/holomotion_v14_checks/episode_review_page.py` (camera grid + tactile; see its docstring) takes a
third-person video from `chase_view.py`, which follows the robot from behind and cuts to another side whenever a wall or
cabinet would hide the pelvis, torso or a hand (sight-line test against the drawn geometry every frame; kitchen5 ep 1:
the whole robot in view on 99.7% of frames, against 87.9% for the fixed chase camera, `--mode fixed`).

## Quick start (the default)

```bash
scripts/teleop_holomotion_v14.sh --scene bottle_bin --backpack-kg 3.2 --record
```

Every scene starts with the robot standing on the floor at the policy's default pose, with no leash. That covers the
first scene at launch and every scene after a save, left menu + Y or left menu + both triggers.

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

### Loop pacing, `--lean-record` and the replay's `--fill` (2026-10-26)

Every control tick is 20 ms of simulation; the loop sleeps the rest of the 20 ms of wall time. When a tick costs more,
the simulation falls behind the operator (the headset picture lags, stick pushes and arm motions arrive late) and the
robot feels unstable although nothing in the controller changed. The status line now prints the pace:
`tick=20.1ms(max 23, late 1%)` = mean wall time per tick since the last line, its maximum, and the share of ticks
over 25 ms; a one-time warning names the lighter options when the mean passes 24 ms. The recorded
`holomotion.tick_time` column holds the same clock per frame.

Measured on the workstation (headless, 32 cores): every bottle_bin recording ticks at 20.3 ms; the kitchen with the
Revo 2 robot, wrist cameras and tactile pads ticked at 30 ms headless and 38-41 ms live (viewer + headset stream +
first-person window), i.e. half real time. Where the kitchen tick goes (bottle_bin in brackets): physics 4.2 ms (0.8),
the env's own 640 x 360 head renders 5.1 ms (1.3), the 1280 x 720 pinhole pair for the stream and the recording 6.1 ms
(2.1), the wrist cameras 4.3 ms per render at 30 Hz = 2.6 ms per tick, plus the four video encoders, the viewer and
the window. The kitchen model has 106 bodies, 1254 geoms and ~160 resting contacts (plates and bowls in the rack).

Two things changed:

- the env's per-step head renders are skipped whenever nothing shows them (the stereo pinhole / fisheye views render
  their own pictures; `[HoloMotion v1.4] the scene's per-step head renders are off` at startup);
- `--lean-record`: the dataset gets states, actions, the controller's inputs and the real-format fields, but **no
  images** (ego_view, wrist cameras) and **no tactile pads**; the headset still gets its picture (left eye at the
  stream rate). `meta/lean_record.json` lists what was left out. The bit-exact replay puts it back:

```bash
scripts/teleop_holomotion_v14.sh --scene kitchen --robot revo2 --record --lean-record --save-dir data/<run>
python -m simple.cli.replay_holomotion_v14 data/<run>/<env_id>/level-3_pinhole --all --fill        # -> level-3_pinhole_filled
```

`--fill` replays every episode from its exact log and, on each frame's pre-step state (the state `observation.state`
was sampled from), renders the rectified pinhole pair at 1280 x 720 area-averaged to the recorded size, the wrist
cameras on the recording's 30 Hz schedule (`wrist_cams.fresh`), and reads the Revo 2 tactile pads; it writes a
complete copy of the dataset with those columns added (same episode order, `meta/episodes.jsonl` extras, `replay/`
logs copied, `fill` recorded per episode). Run on a full recording it checks itself instead: on
`teleop_holomotion_v14_kitchen_plates_smoke` all 20 tactile columns and `wrist_cams.fresh` came back identical and
the images matched the recorded videos to codec precision (ego eyes PSNR 36 dB median, wrist 42 dB). `--no-tactile`
and `--no-wrist-cams` drop one of them alone; the pads cost little, the renders are what matters.

Saving takes a while, and the simulation stands still meanwhile: about 2.8 ms per frame for a lean episode (11 s
for 80 s of recording, 22 s for 2.6 min). The recorder prints `saving episode N (frames, seconds); do not stop the
program` and how long it took; Ctrl-C during a save no longer cuts it short (2026-10-26: a 4-minute episode was lost
that way, leaving only meta/tasks.jsonl -- such a half-written folder is now cleared on the next start).

After the change, headless on the same machine: kitchen + Revo 2 full recording 20.4 ms per tick (was 29.9),
`--lean-record` 20.2 ms, bottle_bin 20.1 ms. Live, the viewer, the headset stream (left eye at 30 Hz) and the
first-person window add a few ms on top; watch `tick=` and use `--lean-record` when it stays above ~22 ms.

### Live pacing measured on the recordings, and the stream resolution question (2026-11-03)

`holomotion.tick_time` of the day's lean A1 / A2 recordings: mean 27.0 ms per 20 ms tick, p95 33.7, 76 % of ticks over 25 ms,
bursts of 45-60 ms in a few episodes, the first step 0.2-1.7 s (the exact-log hash). Headless and lean the same kitchen ticks
at 20.2 ms, so the desktop side costs ~7 ms: the MuJoCo viewer window (the whole kitchen every tick), the first-person window
(cv2.imshow of the 1280 x 720 frame at 30 Hz) and the stream. Measured stream costs: the eye render is geometry-bound (1280 x 720
2.4 ms, 640 x 360 2.1 ms), the 2560 x 720 canvas 0.3 ms, x264 1.5 ms per frame in its own thread, the bitrate a constant 4 Mbps
set by the headset app's OPEN_CAMERA request (width / height / fps / bitrate come from the app). Lowering the stream resolution
therefore buys well under 1 ms per tick and no Wi-Fi relief; the headset decodes faster, that is all. What helps: `--headless`
(viewer off; the headset stream is unaffected), `--no-fpv-window`, or the new `--viewer-hz 10` (the viewer every fifth tick,
default 50 = every tick), and for Wi-Fi `HOLOMOTION_STREAM_BITRATE=2000000` (overrides the app's 4 Mbps in the sender). Watch
`tick=` on the terminal; the headset's own link (3-360 ms pings that evening) is the other half of the felt lag.

### The official BrainCo hand as a second Revo 2 robot: `--revo2-hand official` (2026-11-04 night)

User: "create a new version of the robot with the official hand and remember to have tactile sensors too, and make this new
robot an option so future commands can switch between the old Revo 2 hand and the new hand."

**Why.** The Revo 2 hand recorded so far (`data/robots/g1_rot0_teleop_revo2`, v3) is BrainCo CAD that reached us through
Unitree's xr_teleoperate copy of BrainCo's old `stark-serialport-example` URDF and the HoloMotion module. BrainCo's own
`revo2_description` / `revo2_description_ros1` (identical, v1.1.1 of 2026-08-20, "aligned with the mass-production hand") differ:
per-finger distal couplings 1.131 / 1.267 / 1.252 / 1.271 (thumb 1.0945) instead of 1.155 (thumb 1.0), finger limits 80.8 deg
instead of 84, and the thumb zeros moved by +3 deg (abduction) and +9 / +6 deg (MCP / PIP), so at equal joint values the thumb
tip sits 12 mm from the legacy one. The comparison is in `docs/revo2_official_compare/` and the handover doc.

**The robot.** `data/robots/g1_rot0_teleop_revo2_official/g1_rot0_teleop_revo2_official.xml` (+ `_notouch.xml`, `meshes/`,
`package.json`, `touch_pads.json`, `grip_poses.json`, the camera JSONs copied from the legacy folder, an exported URDF).
Built from the legacy robot WITH its cameras and mounts (`g1_rot0_teleop_revo2_notouch.xml`) by swapping the hands:

    .venv/bin/python scripts/robots/make_g1_revo2.py --hand official --src data/robots/g1_rot0_teleop_revo2/g1_rot0_teleop_revo2_notouch.xml \
        --out-dir data/robots/g1_rot0_teleop_revo2_official --out-name g1_rot0_teleop_revo2_official_notouch.xml
    MUJOCO_GL=egl .venv/bin/python scripts/robots/add_revo2_touch.py --robot-dir data/robots/g1_rot0_teleop_revo2_official \
        --src g1_rot0_teleop_revo2_official_notouch.xml --pads-from-model --out g1_rot0_teleop_revo2_official.xml
    .venv/bin/python scripts/robots/add_revo2_hand_selfcontact.py --src <that file> --out <that file>
    .venv/bin/python scripts/robots/convert_grip_poses_official.py
    .venv/bin/python scripts/robots/export_urdf_revo2.py --src <abs path of that file> --out <abs path .urdf>

The hand source is BrainCo's package copied into `data/robots/g1_revo2_official/` (urdf/ + meshes/, Apache 2.0, SOURCE.md with
the commit); the mount (adapter at 41.5 mm, hand base at 59.1 mm from wrist_yaw) stays the HoloMotion module's, whose base frame
equals BrainCo's. Everything else about the robot is identical to v3: cameras, mounts, backpack, body, names (`<side>_hand_...`,
11 joints per hand in the same order, 51 actuators in joint order). Checks: fingertips at the zero pose and fully closed equal the
official URDF's forward kinematics to 0.1 mm; 10 couplings; hand mass 413 g (390 + the 23 g adapter).

**Tactile pads.** BrainCo's own `*_touch_link` pad links (fixed joints of the official URDF) are the pad bodies; `add_revo2_touch.py
--pads-from-model` keeps them and adds the named collision geom, the box site, the ray site and the three sensors per pad as on
v3 (`touch_total`, `touch_prox`, `touch_grid`), so `Revo2Touch` and the recorder's `tactile.*` columns work unchanged (`touch_pads.json`
is read from the selected robot's folder). Press test (a 20 mm box 1 mm into the right index / thumb pad, no gravity): total 6.8 /
24 N, `physical()` normal 24.3 N = the touch sensor, tangential 5.7 N. The 3 x 3 `touch_grid` taxels were not consistent in that
test on EITHER robot (legacy thumb 0.7 of 16 N, official thumb a negative sum); the recorded SDK fields come from `physical()`.

*2026-11-05 touch-zone fix.* The first official build placed the finger box sites (the volume the `<touch>` sensor sums) from a
finger axis through the distal link's ORIGIN; the tip lies 17 deg off the distal +z, so the box came out centred 11 mm inside the
finger with a 17 mm radial half size (13.0 cm3 instead of 4.7 cm3). `add_revo2_touch.py` now puts the axis through the TIP link
whenever the pads come from the model (and with `--tip-axis` for the legacy chain); the official robot was rebuilt (same counts:
81 bodies, 122 sensors, 144 pairs; the right-finger boxes equal the legacy right hand's exactly, 13.0 x 7.2 x 6.2 mm; press test
touch_total 23.9 N = contact normal 23.9 N), the URDF re-exported and the smoke repeated: 2 episodes BIT-EXACT. The legacy v3
robot's LEFT hand has the same oversized boxes (its pads sit on the distal links) and was left as it is, because every recording
replays on that file; a tight-box legacy variant needs a new file name (v4) and a replay rule for the old recordings.

*Comparison page.* `docs/revo2_hand_compare/` (builder `docs/_scan/revo2_hand_compare/{make_data.py,build.py,page.html}`): both
right hands from the compiled models in one three.js view (side by side / overlay, own grip targets or the same command to both,
pads, touch zones, fingertip markers with live tip distances), the two renders, the joint / coupling / thumb-zero / fingertip
tables and the coupling chart. NAS `fleet_status/revo2_hand_compare/`.

**The switch.** `--revo2-hand legacy|official` (with `--robot revo2`; default `legacy`, nothing changes without the flag).
`simple.robots.g1_sonic_revo2.select_hand_variant()` sets the class file, rewrites `REVO2_UPPER` / `COUPLING` in place from the
compiled model, reloads the grip poses from that robot's folder and rebuilds the controller config; `hand_skeleton` reads the limits
at call time and picks the hand's pinch knots (`PINCH_KNOTS_BY_HAND`, the official ones measured with `pinch_poses.py --table`,
`docs/revo2_official_compare/pinch_table_official.log`). `run_meta` records `revo2_hand` and `robot_mjcf`; `replay_holomotion_v14`
selects the hand from `robot_mjcf` before it builds the env, so old recordings replay on the legacy file and new ones on the official.

    scripts/teleop_holomotion_v14.sh --scene kitchen --robot revo2 --revo2-hand official --record --slice A2 --setup-ranges lv3 --lean-record \
        --save-dir data/teleop_holomotion_v14_kitchen_a2_hands --num-episodes 30 --hand-source skeleton --auto-cycle --countdown-s 5 --plate-balls 2 --plate-set lift

**Grip poses.** `g1_rot0_teleop_revo2_official/grip_poses.json` = the legacy tuned poses converted (thumb rotation -3 deg, bend -9 deg,
distal = official coupling x proximal): the fingertips land within 0.3 mm (open fingers) to 5 mm (closed, pinch thumb) of where the
legacy poses put them. Not re-tuned on this hand yet; the pose tuner page writes the legacy file, so a tuned set for the official hand
is applied by writing this file (same format).

**Results.** Headless A2 smoke with `--revo2-hand official --hand-source skeleton --auto-cycle --plate-balls 2 --plate-set lift`:
2 episodes, both replay BIT-EXACT with the replay choosing the official robot; the evening's legacy smoke still replays BIT-EXACT
(2/2) with the new code and no flag.

**Known gaps.** The palm inertia in BrainCo's v1.1.1 URDF is about 10x too small (kept as published; the palm is welded to the arm).
`camera_calibration.json` was not regenerated (the cameras are the same; its exporter has the legacy path built in). The robot card
page and the glove pilot scripts (`scripts/wuji_glove_pilot`, their own copy of the limits) were not updated. `check_urdf_revo2.py`
has the legacy file built in; the official URDF was checked with yourdfpy only (loads, validates, 10 mimic joints).

### Indicator balls on the plates and the liftable plate set (2026-11-04 evening)

User: "add two free small balls on the plate, free to move around: when I lift the plate I need to make it horizontal, a
rolling ball helps me see that; and make sure the BrainCo fingers can reach underneath the plate, double check the plate type".

`--plate-balls 2` (kitchen only; default 0): two loose 10 mm balls, red and blue, glass-marble mass (10.5 g), resting in a row
across the well centre of every plate lying on the worktop (not a plate standing in a rack or held in a hand). The kit adds
them to the kitchen spec (`k_ball_plate_a_0` ...) and after the compile drops them from 5 mm above the plate's top surface
(a downward ray onto the plate's own geoms, so a reset-pose hand above the plate does not fool it), then settles them 0.5 s with
everything else held. Knobs `KITCHEN_PLATE_BALLS`, `KITCHEN_PLATE_BALL_MM` (10), `KITCHEN_PLATE_BALL_ROLL` (rolling friction
0.0003 m) are recorded with the scene knobs; the bit-exact replay rebuilds them. Tilt response, measured in the kitchen scene with
the plate held still (the threshold is roll / radius): at 1 deg the balls move 2 mm in 2 s, at 2 deg they roll to the wall. (The
0.001 m a bare MuJoCo scene suggested only moved from 3 deg here.) The headset view at the A2 starts shows both balls clearly on
both plates.

**Plates rattle in place -- sphere feet.** With the balls on, they crept 5-20 mm over the first 3 s of every episode on a plate
nobody touched. Cause: the plate's flat foot cylinder on the worktop box gives ONE contact point, and under SIMPLE's solver the
plate jitters about it at 40-70 deg/s of angular speed without going anywhere (the bottles had the same disease, fixed with
sphere feet on 10-26). `plate_swap.swap_plate(..., feet=4)` now puts four 4.5 mm spheres under the foot ring and lifts the
cylinder's face 1.5 mm (same resting height): 4 contacts, 0.0 deg/s, the balls do not move at all. The lift set's plates carry
`foot_spheres: 4` in their meta (so every consumer gets it), `--plate-balls` sets `KITCHEN_PLATE_FEET=4` for the v1 set too, and
without either the v1 plates are exactly as recorded so far (feet 0; this morning's smoke replays bit-exact with the new code).

**Can the fingers get under the rim?** `scripts/kitchen_plates/check_under_rim.py`: the floating right hand of check_pinch.py
beside a plate lying flat on a counter slab, the four straight fingers (index distal link 15 mm thick) or the straight thumb
(19 mm) flat on the counter with the pad up, tip 10 mm outside the rim, pushed lip + 12 mm inward; the hand is rolled / pitched
about the sliding link until nothing else rests on the counter (the thumb needs roll -60 / pitch 10 deg). Pass = the tip >= 15 mm
under the rim edge. The v1 set (rim underside 16-30 mm above the counter): the fingertips get under only 14 of 50 plates (the
other 30 are shoved away, median depth 9 mm), the thumb under 36. So the generator got `--min-clearance-mm 24 --min-lip-mm 18`
(`make_plates.py`, v1 defaults 16 / 12 unchanged) -> `data/kitchen_plates_lift`: 50 of 70 candidates (seed 11) that pass the
under-rim check (fingertips and thumb >= 15 mm under all 50, median 29 / 32 mm, none shoved), check_pinch (50/50 held through
12 cm; 10 of the 70 candidates slipped out of the one-handed edge pinch while hanging, mostly 400 g ones) and check_rack_fit
(50/50 stand in the lower rack). Rim underside 24-32 mm, lip 19-34 mm, height 29-40 mm, diameter 172-251 mm, mass 82-400 g.
Results: docs/kitchen_plates/under_rim.json (v1), docs/kitchen_plates_lift/{under_rim,pinch,rack_fit}.json. `--plate-set lift`
selects it (sets `KITCHEN_PLATES` to the absolute directory, recorded with the knobs; `--plate-set <dir>` works too); the A2
bank starts keep their plate indices (type k of the lift set instead of type k of v1; the placement clearance check and redraw
cover the different sizes). v1 stays the default everywhere else.

The A2 command with both:

    scripts/teleop_holomotion_v14.sh --scene kitchen --robot revo2 --record --slice A2 --setup-ranges lv3 --lean-record \
        --save-dir data/teleop_holomotion_v14_kitchen_a2_hands --num-episodes 30 --hand-source skeleton --auto-cycle --countdown-s 5 \
        --plate-balls 2 --plate-set lift

Checked headless (synthetic publisher, SPACE knob): 2 episodes recorded with balls + lift plates + feet, both replaying bit-exact.
Other ways to judge "horizontal" that were considered: a sim-only tilt readout in the headset stream (exact, but a cue the real
robot will never have); a tilt label on the replay (plate roll/pitch per frame, max tilt while lifted) as an episode-quality
filter, cheap and real-world neutral; a cup or a second plate stacked on the plate (falls off when tilted; dramatic, slow to
reset). The balls are the only one that is physical and visible in the recorded images.

### SPACE after the robot already stands: the countdown now starts at once (2026-11-04 16:00)

User: "i dont see the count down" (and: after SPACE the current episode should switch to motion tracking and start recording).
The auto cycle waited for `quick_ready` to start its countdown, and `quick_ready` is a one-shot the quick start fires the moment
the robot stands -- long before an operator presses SPACE -- so a cycle armed afterwards sat in wait_ready for a signal that
never came again. The headless smokes never saw it because `--auto-cycle-start` arms the cycle before the robot stands. Now
the CLI latches readiness (`ready_latch`: set by quick_ready, cleared at every new start) and feeds the level to the cycle,
so SPACE at any time after the robot stands starts the 5 s countdown immediately, then Y, the hand-over and the recording.
Test knob `KITCHEN_AUTO_CYCLE_SPACE_S=<s>` presses SPACE for you that many seconds in (headless tests only; it also lets
`--auto-cycle` run without a terminal). Checked with that knob (the robot standing 8 s before the SPACE): armed and counting down in the same tick, the hand-over 5 s later, the recording from that tick, 2 episodes saved and replaying bit-exact.

### SPACE from any of our windows (2026-11-04 15:30)

User, headset on, body tracking at 67 Hz and both hands active in the publisher log: "i dont see the countdown and i can not
control the wholebody after pressing the space". The SPACE went to whichever window had the keyboard focus -- the MuJoCo sim
window or the headset-view window, both opened after the terminal -- and only the terminal's stdin fed the keyboard stand-in,
so the cycle never armed, no countdown, no Y, and the held robot kept ignoring the operator. Now all three take the keys: the
sim window's key callback (`SonicLocoManipEnv._viewer_key`, the elastic band's keys as before plus `viewer_key_hook`) and the
headset-view window's `cv2.waitKey` (`FpvWindow(on_key=...)`) forward their key codes to `PicoKeys.press_code` (GLFW and
OpenCV codes: SPACE / p, the letters, digits, arrows), the same path as the terminal. The idle overlay line says "press SPACE
to start (terminal, sim window or headset-view window focused)". Checked: a code-level test (SPACE from both windows arms the
cycle and the countdown digit appears) and the launcher smoke; the windows themselves need a display, so the live session is
the test of the focus itself.

### Countdown in the headset stream, whole body + fingers from the PICO (2026-11-04 15:00)

User: "after space press i want to see the count down in the pico stream, and i want to use the pico control the wholebody
including the hand finger (reference scripts/wuji_glove_pilot/pico_xrt_record.py live); in holomotion motion tracking mode, i
should still control the wholebody + hands".
- The countdown is in the stream already: `--auto-cycle` prepends the cycle's lines to the headset overlay (a 4x digit, then
  "recording in 3.7 s (stand still)", then the usual status lines; rendered through `compose_stream_frame`, checked with
  scratchpad/countdown_render.py). SPACE on the keyboard starts it (the terminal running the launcher must have the focus).
- Whole body + fingers: the publisher (`--source pico --hands`, which the launcher adds for `--hand-source skeleton`) reads the
  24-joint body pose and both 26-joint hands from the same XRoboToolkit SDK session, exactly as pico_xrt_record.py does; the
  body drives the motion-tracking reference, the hands the Revo 2 fingers (`hand_skeleton.py`). Nothing else changes in motion
  tracking mode. The headset's preconditions (pico_xrt_record.py's own note): body tracking needs the XRoboToolkit app's
  full-body mode with the trackers calibrated and the headset on the head; hand tracking needs the controllers put down. The
  12:22 session had body frames at 0 Hz for its whole length, i.e. the app was not in full-body mode at the time.
- The publisher's 5 s line now says so: "body frames X Hz (PICO Y Hz), total N; hands active L a % R b %", and while no body
  frame has ever arrived with controller samples flowing: "NO BODY TRACKING YET ... the XRoboToolkit app must be in its
  full-body mode". `tail -f /tmp/holomotion_v14_publisher.log` during a session shows it.

### A2 held start no longer needs the headset to stand (2026-11-04 14:00)

User: "the robot still fall off. please confirm there will be no such cases in the future". The publisher log of that session
reads "body frames 0.0 Hz (PICO 0.0 Hz), total 0": the headset streamed no body tracking at all, so the held start got no
reference, gave up after `QUICK_HOLD_WAIT_S` 1 s into walking mode at the counter, and the walking arm pose toppled the robot
(the case measured at midday). Now the hold does not depend on the headset: in a tick without a body frame the agent feeds the
node a frame of its own (the held joints, the frozen upright root; frame indices continue, logged like any frame), so motion
tracking engages and holds the pose whether the headset streams or not, and the hold is never released. The hand-over to the
operator (Y / the auto cycle) waits until body frames are `QUICK_HOLD_LIVE_S` 0.5 s fresh and says so on the terminal; the
headset's own "NO BODY TRACKING" line still shows. Tested headless: with no body frames at all the robot stands held at the counter (pelvis height unchanged, motion tracking on its own frames from 2 s), the hand-over is refused and the cycle keeps waiting ("waiting for the headset's body tracking before the hand-over"), nothing is recorded until the frames arrive, then 2 episodes record and replay bit-exact; a 5 s dropout in the middle of the hold leaves the robot standing and the hand-over follows the resumed tracking, 2 episodes bit-exact; the ordinary flow with the headset streaming is unchanged, 2 episodes bit-exact.

### A2 start, 2026-11-04 13:30: the held start is the default -- arms as at the A1 end, standing, at the counter

User: "the arm pose should close to the A1 ending pose, but i still need the robot to stand, and close to the counter as close
as possible because the A2 is pick up plate subtasks". Only motion tracking holds an arbitrary arm pose while standing (the
walking policy owns the arms and topples at the counter, a PD hold and pinned arms fall), so `SLICES["A2"]["start_mode"]` is
`hold`: the recorded A1 end as placed, nudged a little for 4 cm of clearance (22 of 37 ends by 2-10 cm back and at most 0.1 rad
of shoulder lift), motion tracking holding the joints on a frozen upright root, the robot standing still and ignoring the
operator until Y; the headset says "standing, start pose held (SPACE: countdown, Y: take over)". The CLI's `--start-hold /
--no-start-hold` overrides the slice's mode (`--no-start-hold`: walking mode 22-36 cm further back, the paragraph below). The
CLI prints which start it builds at launch. Measured on all 37 ends with this build: 22 are nudged (2–10 cm back, at most 0.1 rad of shoulder lift), and afterwards none touches anything at placement, in the stiff-hold physics check (37 of 37) or in 6 s of the real controller's hold (37 of 37 standing: drift under 3 cm, tilt under 2 deg, the fingertips at most 3.6 cm lower); the launcher flow with the auto cycle records 2 episodes with no reset, both replaying bit-exact.

### A2 start, 2026-11-04 afternoon: walking mode, each end moved back until it stands clear

User: "please make sure each pose the robot can stand still in the walking mode". So the default is walking mode again, made to
stand: `SLICES["A2"]["start_mode"] = "walk"` (the CLI's `--start-hold` opts back into the held start). At placement the recorded
end's arms go to the walking policy's default (`grasp_start.arms_to_walk_default`; the policy puts them there within a second
anyway) and the robot is moved back along its heading 2 cm at a time (`grasp_start.nudge_back_from_edge`) until every robot
geom but the feet is `NUDGE_WALK_CLEAR_M` 10 cm behind the worktop's front edge -- 10 cm because the walking policy's stance
wanders up to 9 cm with a zero command; geometry only, since a contact pass with a 10 cm margin on every robot geom overflows
MuJoCo's 20 MB arena with robot-to-robot pairs and crashed the CLI (the 4 cm pass of the held start now drops those pairs by
zeroing the robot's contype for the pass). The terminal prints "walking-mode start: the arms at the walking policy's
default, the robot N cm further back than the recorded end" and `start.nudge` (mode walk) goes into the setup. The recorded A1
end's heading, legs, waist and plates are kept; the location is the recorded one less that step back. Measured on all 37 ends: every end moves back, 22 to 36 cm (median 28 cm); afterwards none touches anything at placement, in the stiff-hold physics check (37 of 37) or in 6 s of walking mode with the real controller (37 of 37 standing: the stance sinks 4.6 cm, drift under 10 cm, tilt under 6 deg, no fall); the launcher flow with the auto cycle records 2 episodes 24 cm back, both replaying bit-exact.

### A2 start nudged for clearance (2026-11-04 afternoon)

User: "adjust the A1 seed a little bit and make it no contact to the scene". The A2 start is a recorded A1 end, and a few of
those ends stand with the torso on the worktop front (x 0.41) or a hand on a plate rim; instead of skipping them,
`grasp_start.nudge_clear` moves the placed start a little: the first of `NUDGE_STEPS` (back along the heading 0-12 cm, both
shoulders pitched up 0-0.15 rad so the hands rise; smallest first) after which nothing of the robot is within `NUDGE_CLEAR_M`
4 cm of the scene (floor plane, own feet and a held plate excepted; `clearance_contacts`: one forward pass with the robot's geom
margins set to 4 cm, so the collision pipeline reports exactly the near pairs with their distance). Clearance rather than
contact because the fingertips come down up to 4 cm under the hold. The CLI applies it to the held A2 start (`--start-hold`,
the default), prints "start nudged for clearance: N cm back, shoulders X rad up" and records `start.nudge` in the setup; the
start check, the headset grid and the hold measurement apply the same call. measured on all 37 ends: 22 of them get nudged (2–10 cm back, the shoulders 0–0.1 rad up; ends 1 and 3 need the most, 10 cm and 0.1 rad), and afterwards no end touches anything at placement, in the stiff-hold physics check (37 of 37) or in 6 s of the real controller's hold (37 of 37; drift under 3 cm, tilt under 2 deg, the fingertips at most 3.6 cm lower), so no end is skipped any more; the launcher flow with the auto cycle records 2 episodes with no reset, both replaying bit-exact.

### A2 start, 2026-11-04 midday: the held start stays the default; walking mode only further back

User, after two live sessions with the held start: "for A2 at the beginning it would be the same as other in walking mode, after
i press space 5 s wait, it will enter motion tracking mode just like we use gripper, and why the window keeps open and close?".
Both sessions had reset start after start ("start touched ... while held"): the robot followed, then leaned with, the operator's
body reference while they set up (leaning over the keyboard pitches the headset's pelvis), and every reset rebuilds the scene
and its viewer window. Fixed and measured:
- No contact watch in the loop any more (user rule 2026-10-30: nothing resets under the operator). The placed start is still
  checked once (`start_robot`: a touching end is skipped for the next, before the scene is shown).
- The held start's frozen root is upright (its heading only), so an operator leaning over the keyboard no longer tilts the
  robot. Launcher smoke with it: 2 episodes, no reset, bit-exact.
- Walking mode at the recorded A1 ends does not exist physically: the walking policy's arm pose puts the hand bases 10 cm below
  the worktop top at the worktop edge (the A1 ends stand 6-22 cm from it), so the hands are inside the drawer fronts the moment
  the robot is placed (37 of 37 ends, 5-36 mm) and the policy, pushing against them, topples the robot within 3 s (37 of 37 fell
  in the headless measurement; the recorded arms placed instead are swept down into the same place). Walking mode is therefore
  `--no-start-hold`: the arms at the walking default and the robot `A2_WALK_BACK_M` 0.35 m further back from the counter than the
  recorded end (the operator steps in after Y). measured on all 37 ends with the real controller headless: placed 0.35 m back, no end touches anything at placement or in 3 s of walking mode (the stance sinks 4.6 cm, drift under 10 cm, tilt under 6 deg), and the launcher flow with --no-start-hold records 2 episodes, no redraw, both replaying bit-exact.
- So the default for A2 stays the held start: the robot stands still in the recorded end pose, SPACE, the countdown, Y -> motion
  tracking and a 0.5 s hand-over to the operator's pose; it does not follow the operator before Y, exactly as walking mode would
  not.

### No contact at the start: A2 only, arms held under the walking policy (2026-11-03 night)

User, with a live A2 start whose hands rested on the plates: "the robot should not touch anything at the beginning", then
"the rule only for A2, not for the rest". Why the drawn pose was not enough: the quick start presses A at once, and the
walking policy drives all 29 joints -- its arm actions take the arms to its own default (hanging; arm kp 14), so arms that
reach over the worktop sweep down through its edge and top within 0.1-0.5 s (launcher smoke: 90 of 91 A2 starts touched
the worktop; wrists, elbows, hand bases). A pure PD hold of the placed pose instead of the policy does not stand (the
node's HOLD mode with the robot's MOVE_TO_DEFAULT gains: 3 of 8 starts fell over in 6 s, the rest leaned 4-8 deg and the
fingertips came down 7 cm) -- the recorded stances were balanced by the controller, not static.

What A2 does now (`SLICES["A2"]["clean_start"]`; the other slices are untouched):
- `agent.quick_start(..., hold_pose=True)`: the walking policy stands the robot for 12 frames with the 14 arm joints pinned at
  the placed targets (`QUICK_HOLD_ARM_KP` 150 in place of the policy's arm actions), while the reference frames fed to the
  node carry the placed joints and a FROZEN root (the first received root: position and heading; a live session on 11-04
  showed the robot following the operator's body while they set up, so an elbow met the worktop within a second, start after
  start); then the quick start switches to motion tracking itself and the motion policy holds the pose on its own balance,
  without following the operator. The operator's Y (or the auto cycle's; masked from the node) starts a `QUICK_HOLD_FADE_S`
  0.5 s blend of the joints to the received ones, and from then on the received root is re-based onto the frozen one (the
  position and heading offset at the hand-over removed, deltas turned by that heading: `_ref_shift`), so the node sees a
  continuous root and its yaw alignment stays valid. Replay-safe: the frames are logged as fed, the arm pin is in `agent_state`
  (`arm_hold`, restored by `rerecord_holomotion_v14`). No body reference within `QUICK_HOLD_WAIT_S` 1 s: the hold is released
  (walking mode, as before). `--no-start-hold` gives the old behaviour.
- `_start_touches`: the robot's contacts (separation <= 0: touching or inside; the kitchen's 0.1 mm margins no longer count)
  with anything but the floor plane, its own feet (ankle_roll links: floor, mat, pedal) and a plate held in the hand;
  `start_robot()` redraws a new setup while the placed start touches (up to 8 draws); the loop watches the last step's contacts
  every tick while the pose is held (no forward pass; and for `START_SETTLE_S` 1.5 s after a `--no-start-hold` start) and resets
  to a new setup on the first touch (the headset / terminal say what touched what). `--auto-cycle` restarts its countdown on such
  a reset (`AutoCycle.scene_restarted`). Each such reset rebuilds the scene (about a second; the viewer window is recreated with
  it), so a run where most ends get skipped looks like a window opening and closing -- that was the 11-04 session with the
  root still following the operator.
- Sampler margins (A1_VALID): hands >= 8 cm over the bare worktop, >= 18 cm within 5 cm of a plate, >= 6 cm over a fixed prop's
  top, forearm >= 6 cm above the top where it crosses the edge. `slice_start_check.py` reports `touch_after_hold` (its own
  stiff hold) and counts it in its verdict.
- With the recorded A1 ends as the A2 starts (default since 2026-11-04), on the 37 ends with the real controller headless (6 s hold each, the reference root frozen as in the live loop): 32 ends stay free of any contact for the whole hold, the robot drifting at most 3 cm and tilting under 2 deg, the fingertips at most 3.6 cm lower; 4 ends touch the moment they are placed (the torso of two ends standing 41 cm out against the worktop front, an elbow, a wrist) and one after 4.5 s, and the watch skips those for the next end (ends 1, 3, 18, 24, 27; end 33 already at placement). Through the launcher with the auto cycle and the skeleton hands: 2 episodes from ends 19 and 20, no start skipped, both replaying bit-exact, 20 ms ticks. Before the root was frozen the robot followed the operator's body during the hold: the live session of 11-04 reset 10 of 11 starts within a second (an elbow on the worktop; every reset rebuilds the scene and its window), and the headless test with a moving scripted root lost 16 of 37.
- Measured with the real controller headless (synthetic publisher): walking mode as before: 0 of 10 starts clean, wrists / elbows / hand bases on the worktop or a plate 40-100 ms
  after the start (the pelvis sinks 3.5-4.5 cm and pitches 6-15 deg, the fingertips are 13-17 cm lower after 1 s). The hold:
  10 of 10 stood for 6 s (pelvis drift <= 6 cm while the synthetic reference's root moved -- a still operator moves it less --,
  pitch <= 4.5 deg, height -1.2..-1.7 cm, fingertips <= 3.6 cm down), 8 of 10 without any contact, 2 grazes under 0.1 mm (an
  elbow at 1 s, a shoulder link at 3.8 s, against the worktop) that the watch turns into a redraw; the hand-over then moves the
  arms to the reference's (the synthetic human's low arms touch, as any operator's would). Through the launcher with the auto
  cycle and the skeleton hands (gate test knob): 2 episodes recorded, 2 held starts redrawn after a graze, both episodes replay
  bit-exact; 99 columns incl. pico_hands. A PD hold of the pose (the node's HOLD mode, the robot's stand gains) fell in 3 of 8
  starts; the walking policy with the arms pinned fell in 10 of 10 within 2 s.

### Hand tracking for the fingers and a hands-free recording cycle (2026-11-03 night)

`--hand-source skeleton` (default `grip`): the headset's own hand tracking drives the Revo 2 fingers instead of the grip /
trigger poses (docs/HANDOVER_PICO_SKELETON_REVO2_TELEOP.md made first-class). The publisher reads both hands' 26 OpenXR joints
and the isActive flags from the XRoboToolkit SDK every tick and publishes them next to the controller sample (`--hands`, topic
`pico_hands`; scripts/teleop_holomotion_v14.sh adds the flag); the agent retargets them in `get_action`
(`hand_skeleton.py`: 26 -> 21 keypoints -> 6 bone-angle features -> calibration -> 6 motors, pinch blend with PINCH_KNOTS,
10 Hz low-pass, hold on dropout / low quality) and overwrites `left_hand_q` / `right_hand_q`; nothing downstream changes
(hand PD, recorder columns, GripLatch, replays). The recording gets `holomotion.pico_hands` (364 + 2 floats per tick, NaN
before the first sample) and the exact log's info keeps `hand_source` + `hand_skeleton` (sides, min_conf, smooth_hz,
pinch_blend, the calibration arrays), so `rerecord_holomotion_v14` rebuilds the same retarget from the recorded hands. Options:
`--hand-sides left,right`, `--hand-calib-dir` (calib_<side>_pico.json from `scripts/wuji_glove_pilot/bench.py --glove pico
--side <side> --calibrate`; built-in defaults otherwise), `--hand-min-conf 0`, `--hand-smooth-hz 10`. PICO tracks hands only
while the controllers are idle, so the keyboard stands in for their buttons (`pico_keys.py`, moved into the package: r save,
b abandon, n new setup, 0 reset, a / y / s / x / e modes, arrows sticks, h help); with no controller sample the agent serves a
neutral one while hand frames arrive (the quick start wants an operator). The keys' presses go into `holomotion.pico` like
real ones.

`--auto-cycle` (default off; needs `--slice`; user: "after i press space, wait 5 seconds then do the enter motion tracking
mode and recording, after the gate is met and wait time is pass, auto save and reset a new scene, and wait another 5 s to
record, i need to see the count down from the stream"): SPACE on the keyboard arms the cycle; the headset view counts down
`--countdown-s` (5 s, a big digit); the motion-tracking button is pressed for the operator (with `--record-on-motion`, the
default, that starts the recording); the slice's end event + overlap save the episode by itself (`--slice-auto-cut` is forced
on) and the scene resets to a new setup; as soon as the robot stands again the next countdown runs; until `--num-episodes`.
SPACE pauses after the current episode and resumes. `--auto-cycle-max-s N` abandons an episode whose end event has not come
after N s (left menu + B; default 0 = never). Every press is a keyboard-stand-in press (`auto_cycle.py` -> `PicoKeys.
press_buttons`), so the recording and a re-record see ordinary button presses. `--auto-cycle-start` arms it without the SPACE
(headless tests). The headset view shows the cycle's state line above the usual lines. Test knob `KITCHEN_SLICE_GATE_TEST_S=<s>`
forces the slice's end gate at that recording time (headless tests only; the CLI prints it loudly). Headless end-to-end:
scratchpad autocycle_smoke.sh (synthetic publisher + synthetic hands, A2, gate forced at 3 s, 2 episodes, bit-exact replay).

Command for the A2 hand-tracking sessions (put the controllers down; the headset switches to hand tracking):

```bash
scripts/teleop_holomotion_v14.sh --scene kitchen --robot revo2 --record --slice A2 --setup-ranges lv3 --lean-record \
    --save-dir data/teleop_holomotion_v14_kitchen_a2_hands --num-episodes 30 --hand-source skeleton --auto-cycle --countdown-s 5
```

Unverified live (as in the handover): whether the headset's body tracking of the arms stays good with the controllers put
down, and the physical Revo 2 thumb reach. The first live session decides both.

### When the teleop "gets stuck": the headset's body tracking stopped (2026-11-03)

Symptom: the robot holds still, the headset picture is live, nothing crashed. Diagnosis in one look at the publisher log
(`/tmp/holomotion_v14_publisher.log`): `body frames 0.0 Hz (PICO 77.8 Hz), total 38386` repeating with a frozen total, and the
SDK's own `device missing` line just before. The XRoboToolkit service (RoboticsServiceProcess, 127.0.0.1:60061) stopped
delivering body-tracking frames; the controller samples may keep coming (the `PICO x Hz` number is the last body-frame rate,
not live). The sim node holds the robot on a stale reference (max_data_age 0.6 s) by design. Causes seen or plausible: the
headset off the head or asleep, the headset app in the background, a Wi-Fi stall (the headset answered pings in 3-360 ms that
evening while the router took 0.6 ms), or a second XRoboToolkit SDK client attached during the session (that evening
`scripts/wuji_glove_pilot/bench.py --glove pico --viewer`, started from another Claude session, was attached to 60061 the
whole time; a hand-tracking subscription can change the headset's tracking mode).

What was added so this is not silent: the publisher logs `NO BODY TRACKING for N s (controller samples still arrive / stopped
too; last body frame F)` with a timestamp every 5 s and `body tracking RESUMED after N s` (`BodyTrackingWatch`); the CLI prints
the same once on the terminal and shows a red `NO BODY TRACKING from the headset for N s: robot holds` on the headset view
(`ReferenceSubscriber.obs_age`, `REF_LOST_S` 1.5 s); `scripts/teleop_holomotion_v14.sh` warns at start when other clients are
attached to 127.0.0.1:60061 and names them. Recovery: put the headset back on / wake the controllers / bring the app to the
front; the publisher and the CLI resume by themselves when frames return (the robot moves to the new reference, so hold still
for a second). If nothing returns within ~10 s, stop any other SDK client and restart the script: the dataset resumes at the
next episode index. Rule: no second XRoboToolkit client (glove bench, viewers) during a teleop session. An episode recorded
across the stall has a frozen tail; `replay_holomotion_v14 --all` labels it, drop it at export.

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

## Slices of the long-horizon kitchen task (2026-10-30; S2-S6 2026-11-01; kitchen7 ep1 as the demo 2026-11-02)

`--slice S1..S6` (kitchen only) records one slice of the load-dishwasher task as its own episodes (Mode B of
docs/long_horizon_slicing; the scenes and commands: docs/long_horizon_slicing_scenes). Since 2026-11-02 the slices follow
kitchen7 episode 1 (user: "use these as demo slicing, replace the kitchen5 one"): the plate is picked first, the door is
opened with the plate in hand (ep1 passed it to the left hand and pulled with the right), the rack is pulled with the plate
held, the second plate is fetched and loaded, then the rack is pushed in and the door closed. Every setup starts in the
slice's start state, drawn per episode from kitchen7 ep1's own states at the slice boundaries (`scene_setup.SLICE_STARTS`):
a stance box or a walk corridor for the robot, and a preset the kitchen kit puts into the compiled scene after compile
(`kitchen_task.SLICE_PRESET`: dishwasher door angle, upper rack slide, plates standing in the upper rack at ep1's own
resting poses (`RACK_SLOTS`), plates moved on the worktop, bin lid open; qpos only, so the model is the same and the replay
rebuilds the preset from the saved setup). The slice is complete 2 s after its end event, read from the scene state in the
loop (`slices.SliceCut`, advisory: the headset says "COMPLETE: save now (L-menu + A)", no automatic stop; `--slice-auto-cut`
saves by itself). The instruction becomes the slice's, `realformat.success` means the end event was reached, the episode
line gets a `slice` record and the setup carries the `preset`. A plate already standing in the rack when the recording
starts never counts as lifted or loaded (before 2026-11-02 a half-loaded start fired the old S5's gate on frame 0).

Since the same afternoon the slices CHAIN (user: "S2 should be S1 continued work, then the plate is already in the robot's
hand"): seven slices, each starting where the previous one ends, the plate in the hand where ep1 had it. Preset key
`plate_in_hand: {plate, side}` -> `grasp_start.hold_plate` right after `_stand_on_floor`: the holding arm goes to ep1's carry
pose, the eleven finger joints to ep1's grasp, the plate to ep1's pose in the hand-base frame (right hand: ep1 frame 2000 =
40.0 s; left hand: frame 4400 = 88.0 s; the plate centre 15 cm from the base, shifted along the rim for the plate's own
radius) -- ep1's own state, which the controller held for minutes; the quick start holds that pose (hold_real).
`grasp_start.GripLatch` keeps that hand's command at CLOSED_Q until the operator presses the grip or the trigger once (the
recorder and the exact log see the latched command: the replay needs nothing). The one recorded grasp fits 32 of the 50 plate
types in the right hand and 24 in the left (scratchpad grasp_sweep.py: 1 s under a controller-like hold, < 2 cm slip); the
others drop or creep, so a plate-in-hand slice draws its held plate from `grasp_start.GRASPABLE[side]` (plate_b = id + 25 as
always). Traps found on the way: pinning the robot by resetting qpos every step (the kit's settle style) breaks the contact
solution and makes even ep1's recorded grasp creep 9 cm/s, so the check holds the robot with joint PD + a pelvis spring
instead; a "clamp" that pinned the plate and closed the fingers on it pushed the thumb open on rims that sit in the thumb base.

| slice | start | end event | varies per episode |
|---|---|---|---|
| S1 pick up a plate | in front of the plates, one step back from ep1's pick stances (the task's start distance), door closed, rack in, lid closed | a counter plate lifted >= 3 cm for 0.5 s | stance x -0.05..0.05, y -0.75..-0.10, +-15 deg; plate spots +-5 cm; plates |
| S2 carry to the bin and wipe | 0.3-1.4 m before the bin on ep1's carry path (KITCHEN_S2_PATH), plate_a in the right hand, lid closed, door closed | lid open and a plate over the bin for 0.5 s | corridor spot, side, heading, plate_b spot, plates |
| S3 carry to the dishwasher, open the door | 0.3-3.3 m before the hand-over spot on ep1's bin -> door carry (KITCHEN_S3_PATH), plate_a in the right hand, lid open, door closed | door >= 69 deg while a hand touches a plate, hands off the door 0.5 s | corridor spot, side, heading, plate_b spot, plates |
| S4 pull the rack out and load | beside the open door facing it (the door-side box), plate_a in the LEFT hand (after ep1's hand-over), door 70-90 deg, rack in, lid open | a plate newly inside the upper rack, hands off 0.5 s | stance, door, plate_b spot, plates |
| S5 fetch the second plate | 0.3-1.1 m along ep1's dishwasher -> counter walk, door open, rack out, plate_a in the rack | a counter plate lifted >= 3 cm for 0.5 s (plate_a excluded) | corridor spot, side, heading, plate_b spot, door, rack 30-43 cm, plates |
| S6 load the second plate | 0.3-0.9 m along a straight counter -> door path (KITCHEN_S6_PATH), plate_b in the right hand, plate_a in the rack, rack out | a plate newly inside the upper rack, hands off 0.5 s | corridor spot, side, heading, door, rack, plates |
| S7 push the rack in and close | beside the open door, rack out, 1-2 plates in it | racks within 5 cm and door within 5 deg, hands off 0.5 s | stance, door, rack, 1 or 2 plates, plate_b spot |

Check: `scripts/holomotion_v14_checks/slice_start_check.py` (penetrations, preset, the plate in the hand, 1 s hold with the
holding fingers under PD, views), the gate test (scratchpad gate_test_k7.py: every gate off at the start, on in the end state, the
latch), and
the headless smoke per slice (synthetic publisher on 6011, Y at 20 s, left menu + A at 32 s, then `replay_holomotion_v14
--all` must say BIT-EXACT; for the plate-in-hand slices the last frame restored and the plate still in the hand): results on
the page (docs/long_horizon_slicing_scenes) and in data/teleop_holomotion_v14_kitchen_slices_smoke_20261102k7c (..._k7 and
..._k7b = the six-slice set of the morning, before and after the parked-robot settle).
The kitchen5-based sets of 2026-11-01/02 (smoke data ..._20261102 and ..._20261102b, page inputs under
docs/_scan/long_horizon_slicing_scenes/kitchen5_demo/) are superseded. Kit fix 2026-11-02: during the preset's settle the robot
is still in SIMPLE's reset pose (arms forward at worktop height), and an S1 start at x >= 0.13 swept plate_a off its spot with
the right index finger; `kitchen_after_compile` now parks the robot 3 m up for the settle and puts it back. SIMPLE's own reset
steps after the hook still kick a plate within reach of that pose (x >= 0.08: a 1 m/s hop of 3 cm), so no stance box puts the
pelvis closer than 0.42 m to the worktop edge.

    scripts/teleop_holomotion_v14.sh --scene kitchen --robot revo2 --record --slice S3 --setup-ranges lv3 \
        --lean-record --save-dir data/teleop_holomotion_v14_kitchen_s3 --num-episodes 20

The original S1 note (2026-10-30; superseded 2026-11-02: S1 is now the pick at the counter, the corridor below is
`scene_setup.KITCHEN_S1_PATH`, which no slice uses):

`--slice S1` (kitchen only) records the proposal's first slice (docs/long_horizon_slicing) as its own episodes, Mode B:
every setup starts the robot 0.3-1.5 m before the dishwasher on kitchen5 episode 1's approach path (up to 15 cm aside,
facing along it +-15 deg; door closed, rack in, plates on the counter, a new plate type per setup as always). The
operator walks up and opens the door; 2 s after the door is past 69 deg (the task's `door_open` gate) with no hand on
the door for 0.5 s the slice is complete: the headset says "S1 COMPLETE: save now (L-menu + A)" and the operator saves
(user 2026-10-30: no automatic stop or reset). `--slice-auto-cut` restores the automatic save at that point, with the
next setup following. Left menu + B abandons, left menu + both triggers restarts the same setup. The instruction becomes "walk to the dishwasher and open its door" (`KITCHEN_INSTRUCTION`
overrides), `realformat.success` means the slice's end event was reached, and the episode line in `meta/episodes.jsonl`
gets a `slice` record (gate / event / cut frames). Code: `simple.teleop.holomotion_v14.slices` (the cut),
`scene_setup.SLICE_STARTS` (the start). New kit knob `KITCHEN_DOOR_DEG` (0-90, default 0) starts the dishwasher door
open; it is the S2 start preset and the smoke test's way to trigger the S1 cut.

    scripts/teleop_holomotion_v14.sh --scene kitchen --robot revo2 --record --slice S1 --setup-ranges lv3 \
        --lean-record --save-dir data/teleop_holomotion_v14_kitchen_s1 --num-episodes 20

## Atomic slices A1-A5: S1 + S2 as RoboAtom skills with heavily randomized starts (2026-11-02 evening)

User: "slice the task until first cleaning plate into atomic action and create the teleop scene for each atomic scene ...
for atomic action, take https://arxiv.org/pdf/2610.40207 as reference ... the init randomization higher than usual, robot pose
including hand open and close pose when there is nothing in hand, robot location, plates location on the table, plate type";
then "the first atomic action should be move close to the counter", "why there are 3 atomic action scene for a single pick",
"when wipe the plate the robot one foot should be on the pedal so that it drives the lid up".
Page: docs/atomic_slicing_scenes (NAS http://10.40.11.11:8899/atomic_slicing_scenes/), built by docs/_scan/atomic_slicing_scenes/build.py.

RoboAtom (arXiv 2610.40207) labels one segment per dominant skill from a 29-class vocabulary and folds "arm reaching, lifting,
lowering, alignment, and retraction" into the neighbouring manipulation ("emit Pick when grasp changes from not holding to
holding"). Read that way, kitchen7 ep1's first 55.7 s is Move to the counter (0-3.2 s; whole-body locomotion is its own
dominant skill, as the carry is) / Pick (3.2-29.6 s) / Move the plate to the bin (29.6-51.5 s) / Step on the pedal (51.5-51.9 s; a foot skill we add, not RoboAtom's hand Press)
/ Wipe (51.9-55.5 s): five `--slice` ids, one per skill.

| id | skill | start | end event (then 1 s overlap, "save now") |
|---|---|---|---|
| A1 | Move to the counter | box x -0.60..-0.10, y -0.80..0.20, yaw +-40; arms / waist / both hands drawn | a counter plate 0.15-0.55 m ahead within 0.30 m sideways, pelvis < 0.15 m/s, 0.5 s |
| A2 | Pick the plate up | since 2026-11-04 THE RECORDED A1 ENDS EXACTLY (`start_banks/a1_end.json`, 37 episodes; user: "use the same location and pose seed from A1 recorded data"): setup seed k -> end k mod 37, the pelvis x / y / heading, the 29 body joints and the pelvis height as recorded at the A1 gate, both plates at that episode's recorded spots and types, no jitter, no `--random-init-pose` offset on a recorded start; only the hands drawn (resting; the recorded closed hands lock); spawned 0.35 m back; a touching end is skipped by the A2 no-touch machinery (below). `KITCHEN_A2_START=a1` = the 11-03 envelope kind (`at_counter` + `a1_env`: pelvis x 0.26-0.40 less 0.20 x \|sin yaw\|, the picked plate -26..+26 cm aside, yaw 0 +-12, every arm / waist joint uniform in its recorded range, one recorded leg stance, each draw checked on the kinematics, `A1_VALID`), `a1_mix` = the first cut, `v1` = the 11-02 kinds | a plate >= 10 cm above its rest for 0.5 s; milestones hand_on_plate, lift_onset (3 cm) |
| A3 | Move the plate to the bin | box x -0.15..0.05, y -0.85..0.15, yaw +-35; plate A or B in the right or left hand, since 2026-11-05 HELD LEVEL as in A2 ep41 (`grasp_start.GRASPS right_a2e41` / the mirrored left: three pads under the rim, thumb on top, the arm raised 0.45 m ahead), the holding arm +- CARRY_JITTER and the waist drawn until the plate is within 5 deg of flat (`scene_setup.held_plate_tilt_deg`), the grip at ep41's recorded hand command until the operator closes their hand on it, the start HELD by motion tracking with the holding arm pinned until Y (`start_mode hold`, `hold_pin held_arm`); free arm / hand drawn; lid closed; see the section "A3 from the A2 data" below | the bin 0.2-0.65 m ahead within 0.35 m sideways, pelvis < 0.15 m/s, 0.5 s |
| A4 | Step on the pedal | box x 0.55..0.80, y 1.20..1.36, yaw -95..-45; plate in a hand; lid closed | lid >= 1.3 rad for 0.3 s (the foot holding the sprung pedal) |
| A5 | Wipe the plate | `at_pedal` (the tread 0.17-0.23 m ahead, -3..+6 cm aside, yaw -62 +-12) + `pose k7_f2650` (ep1 mid-swipe, 53.0 s) + `foot_on_pedal` left / right (leg IK onto the tread) + plate A or B in the right hand + lid open (pedal pressed) + the free hand drawn | 3 passes of the free hand's nearest fingertip within 16 cm of the plate centre then out past 24 cm, the plate over the open bin (lid up = foot on the pedal) |

A2 from the A1 data (2026-11-03 evening; user: "use the A1 collected data to redesign the A2 scene"): the 37 complete A1 episodes
(data/teleop_holomotion_v14_kitchen_a1, two sessions) end -- state at `slice.event_frame`, the at_counter gate held 0.5 s -- with
the pelvis at x 0.26-0.42 m (6-22 cm from the worktop edge at 0.475; the 11-02 counter kind stood at 0.10-0.26 because the DEFAULT
stance's knees met the drawer fronts at x >= 0.27), y -0.59..-0.04, facing -10..+12 deg, the nearer plate 0.24-0.45 m ahead and up
to 26 cm to either side (31 of 37 between the two plates, both in the gate), the arms held OUT FORWARD nearly straight (shoulder
pitch -0.15..0.12, elbow -0.37..-0.10: fingertips 0.27-0.38 m ahead of the pelvis and 0.13-0.39 m above it, over the worktop edge
at chest height -- the controller-holding posture; the NEAR range's bent arms never occur), waist within +-0.1, legs in the
controller's stance (knee 0.23-0.51, pelvis 0.78 m), speed < 0.1 m/s, 1-2 cm of drift to the cut.
`scripts/holomotion_v14_checks/slice_end_bank.py <level dir>... --out src/simple/teleop/holomotion_v14/start_banks/a1_end.json`
writes the bank incl. `hands_pelvis` by FK (rerun it when more A1 sessions exist); scene_setup `load_bank`, `a1_envelope`
(per-joint min..max of the bank's arms / waist + margins, the recorded leg stances), `_RobotFK` (the robot's own MJCF, loaded
once: fingertip centroid / hand base / elbow per side in the pelvis frame), `A1_VALID` + `_valid_start` (every fingertip, the hand base
and the wrist over the worktop >= over_top_m 0.04 above its top; within radius + plate_near_m 0.05 of a plate >= over_plate_m
0.18 above the top = 15 cm above the rim (11-03 night, user's headset picture of a start with both hands ON the plates: "i dont
want this happen"; 0.15 still looked close; the recorded ends hover a hand low over a plate in 36 of 37 episodes, left out on
purpose; before that: a wrist link 40 mm up went 21 mm into a plate, a fingertip centroid 75 mm up touched one by 4.5 mm) --; an elbow behind the edge less elbow_margin_m 0.07 (the arm is 4-5 cm thick: at 0.04 an elbow / upper arm met the edge by
2-3 mm) or above the top; the forearm (elbow -> wrist) >= forearm_over_top_m 0.045 above the top where it crosses the edge (one crossed 36 mm
up and touched by 3 mm); a hand within 0.05 of a fixed worktop prop (`WORKTOP_PROPS`: bowls, bottle, cartons, from the compiled model) >= prop top + 0.04
(a hand base touched the 19 cm large bowl); hands apart >= 0.10 and on their own side; draws: 10 tries median, 74 at p95, 6 plate
layouts x 100 stances allowed (a layout that defeats 100 stances is redrawn), 0 of 600 ran out; headset grid of 12 starts on
the atomic page, A2 card), start kinds `at_counter` + `a1_env` (DEFAULT: the draw loop, up to 120 tries, start.tries / legs_episode / env;
x_m 0.26-0.40 less x_per_sin_yaw 0.20 x |sin yaw| -- the torso meets the worktop front at 0.417, an elbow at 0.377 / -10 deg),
`at_counter` + `bank_pose` (a1_mix: one recorded body pose, `_hands_clear`), `bank` (+ `jitter`, x capped 0.405), `_bank_pose`;
the body pose goes inline (`pose = {body, source}`; `grasp_start.apply_pose` takes a dict as well as a POSES name), the bank
entry's plate ids override the seed's (bank kind). Hands: NOT from the record -- 22 of 74 A1 hands were closed (operator holding the grip while walking) and 14 of them (curl
0.96) lock after a qpos teleport like the old closed-grip draw (scratchpad bank_hands_open_test.py); every free hand = the resting
draw. The rand map shows the 37 ends as + marks and the bank span as purple bands (`slice_rand_map.py`); the page's A2 card has
the A1-data figure (`docs/_scan/atomic_slicing_scenes/a1_ends_fig.py`).

A3-A5 run with `KITCHEN_PEDAL_RETURN=1` (slices.SLICES[...]["env"], set by the CLI before the kitchen module loads and written
into the recording's knobs): a 3 N m/rad return spring on the pedal, so the lid is up only while a foot presses (as a real
bin). As authored the pedal has no spring: ep1 tapped it (51.5-51.9 s), both feet were on the floor during the swipe and the
lid stayed at 1.45 rad by friction (lid frictionloss 0.08); kitchen4-7 and the S-slices keep that behaviour (knob off).

Randomization (all in the saved setup, replay bit-exact): location as above; both free arms joint by joint
(`grasp_start.ARM_RANGE` full; `_LOW` one step from furniture; `_NEAR` at the counter stance = pitch 0.15..0.70, elbow
0.80..1.40, fingertips <= 0.17 m ahead -- measured with scratchpad arm_reach.py: pitch NEGATIVE = arm forward, the default pose
reaches 0.28 m, pitch -0.2 / elbow 0.2 0.46 m), waist (`WAIST_RANGE`), each free hand a RESTING pose (`draw_hand`, since 11-03: one curl per hand Beta(2,4) x 0.45 of the closure, more toward the little finger,
the thumb half as much; the old HAND_MIX closed grip / pinch kinds are gone: a closed grip set by qpos locks the fingertips in the palm, kitchen_a2 ep0) and held there by `GripLatch` (per-side targets) until the operator squeezes that hand once; the
holding arm +- `CARRY_JITTER` (x0.5 for the left hold grasp); plate spots over the free worktop (A y -0.42..-0.08, B y
-0.68..-0.48, x 0.61-0.76; the picked plate of A2 at x 0.60-0.66) drawn at least both radii + 2 cm apart, a random yaw each
(`plate_spots [x, y, yaw]`, kit); plate A and plate B types drawn independently (`setup.plate_b`, kit `PLATE_ID_B`); which
plate is picked / held and which hand holds it; the Pick's start kind; the Wipe's pedal foot. Suggested next: bin position (kit
must place the bin from `L`), worktop clutter, plate state (upside down / stacked / at the edge), lighting + camera jitter,
door / rack background states, start in motion, a second held object.

Mechanics added: `spawn_back_m` (the robot spawns 0.35 m behind an at-the-plate start for SIMPLE's reset steps, whose arms-
forward pose sweeps a plate from a stance 0.3-0.4 m from the worktop; `_stand_on_floor(shift=)` moves it forward),
`SLICE_STARTS[...]["variants"]` (weighted start specs inside one scene, `start.variant` in the setup), `at_pedal`,
`grasp_start.POSES` / `apply_pose` (whole-body pose + arms / waist / hands), `grasp_start.foot_on_pedal` (damped least-squares IK
of one leg's six joints: the ankle link flat on the pressed tread, 0.1 mm / <= 5 deg), `SliceCut` gates at_counter,
hand_on_plate, at_bin, lid_ge, swipe_passes, and `milestones` (first frame of each, in the episode's slice meta with the RoboAtom
`skill`).
FIX: the bin footprint for "plate over the bin" was taken from body-local geom positions (0.8 m off) -> world positions now
(S2's end event could not fire on a real episode before).
NOT a start: the grasp -> lift boundary (ep1 21.5 s, the plate 3 cm up between two moving hands; `GRASPS["left_pinch"]` +
`POSES["k7_f1077"]` built): placed exactly and held like the controller holds, the plate fell flat within 0.5 s and kept
sliding for 49 / 50 plate types (scratchpad grasp_sweep_pinch2.py) -> the lift onset is a milestone inside the Pick.
Checks: scripts/holomotion_v14_checks/slice_start_check.py --slices A1,...,A5 (10 starts each, controller-like hold; for A5
also the lid still up after the hold), the scratchpad gate test (gate_test_a7.py), the lid-falls test (lid_falls_test.py: with
the knob the lid drops when the robot is lifted off the pedal), headless smokes
data/teleop_holomotion_v14_kitchen_slices_smoke_20261102a7 (5 scenes, bit-exact), smoke_inhand_check.py (A3-A5: the plate still
in the hand at the last frame). UNTESTED LIVE: how the HoloMotion controller takes over the one-foot-on-the-pedal stance when
the operator engages; fallback = start the Wipe with both feet down (A4's end state) and let the operator step on.

Randomization maps (2026-11-03; user: "a location and orientation randomization map so i can tell whether the randomization
is enough, also for the robot pose"): `scripts/holomotion_v14_checks/slice_rand_map.py --n 300 --m 12` draws N starts per
scene (SceneSetups.sample, no physics) and builds M of them as the CLI does, and writes per scene `map_<S>.png` (every start as
a heading arrow on an orthographic top view of the compiled kitchen, the plate spots with their drawn radius and turn, the
spec's box, ep1's stance, a heading rose, the stance cell counts or the target in the pelvis frame), `pose_<S>.png` (each arm /
waist joint's draws against its mechanical range and the spec's band, the fingertip centroid of each hand in the pelvis frame
by forward kinematics, the free hands' kinds), `grid_<S>.jpg` / `poses_<S>.jpg` / `fan_<S>.jpg` (the M built starts from a
camera tied to the pelvis: in the scene, robot-only, and overlaid), `map_all.png` and `coverage.json` (footprints, spans,
joint span fractions, fingertip spreads, hand / held / foot / kind counts) -> docs/_scan/atomic_slicing_scenes/rand, shown in the
page's cards ("Randomization map") and in section 5. Traps: the kit recompiles the model at every env.reset(), so re-read
mjModel / mjData (and every id) after each reset; MuJoCo's orthographic free camera (`cam.orthographic = 1`) takes its view
height from `distance` but the pixel scale is best fitted from a free body placed at known points (segmentation render); the
robot frame of the setups IS the world frame of the compiled kitchen (plate spots and the pelvis land exactly at their setup
values). Findings: A1 / A3 fill their boxes (0.50 / 0.20 m2, +-40 / +-35 deg); A2, A4, A5 are tight because the action's
object fixes the stance (plate rim, pedal, tread under the foot); free arms span 15-35 % of each joint's mechanical range
(no self-collision check on drawn poses); A3's 300 draws came out 172 left / 128 right for the holding hand (a chance deviation
of that seed block: 3000 consecutive seeds give 1492 / 1508).

RoboAtom slicing with a lower-body lane (2026-11-03, user: "we should consider the lowbody key skills, for example A4 should be step not
press ... create a new full ep roboatom slicing for kitchen demo"): RoboAtom's only class below the hands is Move ("physical robot-base/chassis
translation only"); we keep its hand classes for the upper lane and add a lower lane Move / Step (foot on the pedal) / Squat (pelvis >= 6 cm
below standing for 0.5 s) / Turn (>= 45 deg in place, not leading into a walk); whole-body label = the hand skill when the hands do more than
hold, else the lower-body skill. Tool: scripts/holomotion_v14_checks/roboatom_slice_episode.py (reads the annotation scan's events.npz /
events.json / traj.json + caption.json; --move-net 0.30 --turn-deg 45 --squat-drop 0.06) -> segments.json (RoboAtom fields + lane + the other
lane's content), timeline.png, path.png; page docs/_scan/roboatom_k7ep0/build.py -> docs/roboatom_slicing_k7ep0 (NAS
http://10.40.11.11:8899/roboatom_slicing_k7ep0/). kitchen7 ep0: 16 whole-body segments over 180.6 s (+ Idle tail) where the annotation had 18
sub tasks / 27 actions: Move 0-4.8, Pick 4.8-15.7, Move 15.7-33.2, Step 33.2-45.4 (two presses, the lid fell back after the first; the
annotation's 3.3 s "press" + "hold beside the bin" + the carry's first 7.6 s), Move 45.4-56.4, Handover 56.4-61.1, Open 61.1-81.9 (4 stance
shuffles inside), Pull 81.9-87.6, Place 87.6-97.8, Move 97.8-108.3, Pick 108.3-126.2, Move 126.2-136.4 (to the bin, 6 s stop, no press),
Move 136.4-153.6, Place 153.6-158.9, Push 158.9-163.4, Close 163.4-180.6 (Squat 167.9-174.9 inside, pelvis 0.54 m, the door horizontal). No
Turn: all 18 fast turns are parts of walks. A4's kit meta is now skill "Step" / "Step on the bin pedal" (slices.py), the scene unchanged.

First live A1 session + the step-limit trap (2026-11-03 12:50-13:09, data/teleop_holomotion_v14_kitchen_a1, user: "check what happened to
the last run"): 8 episodes saved, all 8 replay BIT-EXACT (read-only `replay_holomotion_v14 <level> --all`). ep0 is a 1-FRAME episode: the
stack started 12:50, the operator pressed record at 13:06 (publisher idle 16 min, 408 headset frames seen), and the recorder's step-limit save
tested the env's `truncated` flag, which counts env steps since the last reset (the sim keeps stepping while standing by) -> true after 10 min
-> the first recorded frame was saved at once. FIXED in src/simple/cli/teleop_holomotion_v14.py: the limit now tests the recording's own
length (exporter.episode_buffer["size"] >= --max-episode-steps); verified by a headless A1 smoke with --max-episode-steps 100 (Y at 20 s:
a 100-frame episode saved, not 1). ep1-7: 4.6-7.1 s each, at_counter gate at 2.8-5.1 s, complete=True / success True, saved every ~15 s by the
operator (left menu + A; overlap 1 s after the event, ep1 saved 0.5 s after the event), starts as drawn (x -0.48..-0.10, y -0.76..+0.18,
yaw -28..+32 deg, hands open / part-closed / closed latched, waist and arms drawn). Loop pacing: 24-30 ms wall per 20 ms tick (1.2-1.5x
slow, 2 reference frames per tick), plus a 0.24-1.7 s hitch between recorded frames 0 and 1 of EVERY episode = the exact log's start
(state copy + sha256 of the whole kitchen model); not fixed. The session ended with CLOSE_CAMERA from the headset (the app closed the
stream; "[TCPServer] Client connection closed by server ... waiting for next connection" is the normal close path), publisher stopped
13:09:23; two "JSON parsing error ... empty input" lines in the publisher log at the end are the app's empty messages while closing.
ep0 must be dropped at export / fill (never deleted from the recording).


### A2 plate-grasp page: which picks kept the plate level (2026-11-05)

User: "a page showing the A2 hand grasping plate pose, only those where the ball is not falling off the plate and the plate is
almost horizontal, highlight the hand grasping pose, extract the pose and count the successful ones". Page `docs/a2_grasp_poses/`
(NAS `fleet_status/a2_grasp_poses/`), builder `docs/_scan/a2_grasp_poses/{analyze.py,build.py,template.html}`, extracted poses in
`docs/a2_grasp_poses/grasp_poses.json`.

Data: `data/teleop_holomotion_v14_kitchen_a2_hands`, 44 episodes of 2026-11-04 (14:22-18:09), all at the A2 gate. Episodes 0-13 were
recorded before `--plate-balls` existed (no balls, nq 130); episodes 14-43 carry the two 10 mm balls (nq 158, lift plate set,
sphere feet). `analyze.py` rebuilds each scene through the replay's `_build_env` (KITCHEN_PLATE_BALLS=2 forced for the early
group), maps the recorded qpos/qvel by joint name and replays the recorded state exactly with mj_forward: plate height and tilt
(angle between the plate z axis and the vertical), hand-plate contacts. The pick = the continuous lift that ends at the recorded
gate (target plate = the one highest at the gate, often NOT the start's plate; rest frame = last 0.5 s still before the gate;
onset = frame after the plate was last below 3 cm). Many starts have a hand on or under the plate (it jumps 3-5 cm and 20-30 deg
in the first 0.2 s, settles, and the operator picks later): 27 of 44 flagged, 13 with more than one lift attempt. For episodes
0-13 the balls are simulated on the recorded motion: set on the plate at the rest frame, settled 0.5 s, then every recorded joint
driven at 200 Hz (lerp between the 50 Hz frames) while only the balls step. Grasping hand = the hand holding the plate at the
gate (at onset the other hand is sometimes still nudging it). Renders at onset and gate with that hand orange, the robot grey,
shadows off (shadow acne), plus the head camera and a top view with the balls.

Counts (both balls on the plate at the end AND the plate tilt never above T during the hold, gate to end): T=5: 1/44, T=10: 8/44
(5 right, 3 left, all in the recorded-balls group: eps 27, 30, 32, 35, 40, 41, 42, 43), T=15: 15/44, T=20: 18/44; over the whole
lift instead of the hold: 3/44 at 10, 12/44 at 15. Both balls stayed on in 25/44 (21/30 of the recorded group; 2 lost them before
the pick). Most failures lift the plate steeply (gate tilt 20-60 deg) with a thumb-index rim pinch; the passing picks mostly use
three or four pads (index/middle/ring + thumb) or a two-finger scoop (ring/pinky under the rim) and come from the end of the
session (eps 40-43 are the cleanest).

Traps: parquet row t = the state BEFORE step t = exact_log physics[t-1] (3e-8); the env reset does not reproduce the recorded
frame 0 (the teleop's start procedure moves things after reset; the replay only works because it sets state0), so never place
anything from the reset pose; `observation.object_poses` "target" is the red bottle, not the plate; the two recording groups need
separate processes (the kitchen module reads its knobs at import).

### A3 from the A2 data: the plate held level as in episode 41 (2026-11-05)

User: "please use a2 ep41 hand pose and arm pose as reference holding, and adjust the a3 30 setup based on that" ... "remember
to include similar randomization rules". Of the 44 A2 picks scored on the plate-grasp page (section above), episode 41 held the
plate flattest with both indicator balls on it (3.2 deg at the gate, 2.9 deg two frames later; index / middle / ring pads under
the rim, the thumb pad on top). A3's plate-in-hand start now uses that grasp instead of kitchen7 ep1's 70-degree carry; the rest
of A3's randomization is the same as before (stance box, free arm in the LOW range, resting free hand, plate spots + yaw, both
plate types, which plate in which hand, lid closed), with two rules added so the plate really starts level and stays in the hand.

**The grasp** (`grasp_start.GRASPS["right_a2e41"]`, `"left_a2e41"` = its mirror image: hand-frame y and the quaternion's x / z,
the arm's roll / yaw signs flipped): read from ep41's exact log at frame 376 (7.52 s): plate centre (0.0958, 0.0062, 0.1383) m in
the right hand base frame, quat (-0.2893, -0.5863, 0.0323, 0.7560); fingers (1.2434, 0.6421, 0.643, 0.5599, 0.6465, 0.6943, 0.8016,
0.7133, 0.8238, 0.6384, 0.7376); arm (-0.5832, -0.8257, 0.4352, 0.1916, -0.8017, -0.3013, -0.2509); waist (-0.036, -0.038, -0.067);
the plate 0.45 m ahead, 0.10 m right, 0.21 m above the pelvis. `r_ref` 0.0914 (lift plate 23, 183 mm across, by `plate_radius`):
other types are shifted along the rim's radial line as before. `cmd` = the skeleton hand's recorded command at that frame,
0.1-0.26 rad beyond the fingers the plate stopped; the latch holds THAT (`latch_target`), not the closed grip (0.75-1.05 rad beyond,
a different grip that turns the plate). The preset key is `plate_in_hand.grasp = "a2e41"` (`_HELD_A2E41`), `grasp_key` ->
"right_a2e41" / "left_a2e41"; old setups (grasp "carry") and the S-slices keep ep1's grasps.

**Level rule** (`scene_setup.held_plate_tilt_deg`, `LEVEL_MAX_DEG` 5, `LEVEL_TRIES` 300, `LEVEL_WAIST_RANGE`): the first draws put
the plate 15-20 deg from flat, because the random waist (pitch up to 0.22 rad) and the arm jitter (RSS ~12 deg on the hand) tilt
whatever the hand holds. For a grasp with `level_deg` the holding arm's jitter and the waist are drawn together and accepted only
when the plate they would hold is within 5 deg of flat on the robot's kinematics alone (`_RobotFK.hand_rot` + the grasp's
rel_quat, legs at the default stance, pelvis level; 1 ms per draw): 30 setups -> all within 5 deg (median 5 tries, max 19), the
placed plate's tilt equals the FK number, no robot-scene contact at placement. The waist band is WAIST_RANGE with the pitch
reaching ep41's own slight lean-back (-0.10..0.22). `plate_in_hand.level_tilt_deg / level_tries` and `waist` are in the setup.

**Held start with the holding arm pinned** (`slices.SLICES["A3"]`: `start_mode "hold"`, `hold_pin "held_arm"`; agent
`quick_start(pin_real_idx=...)`, `arm_pin` in agent_state, restored by rerecord): measured headless with the real controller
(scratchpad a3_hold_diag.py, 8 drawn starts x 6 s, ep41's plate type, synthetic reference):
- walking mode (the old A3 start): 8/8 lost the plate's attitude within the window (80-90 deg, 35-48 cm lower): the walking
  policy swings the raised arm to its hanging default;
- held, no pin: 8/8 kept the plate, 4-13 deg at rest, 14-21 deg for a moment at the switch to motion tracking: the teleop's arm
  PD is 14 N m/rad (recorded `action_kp`) and ep41's own recorded targets sat 33 deg above the actual shoulder (the policy
  overshoots to hold a loaded arm); the hold's reference is the placed joints, so the arm sags from them;
- held, holding arm pinned at QUICK_HOLD_ARM_KP through the hold, faded to the policy over the 0.5 s hand-over: 8/8 kept, 4-11 deg
  at rest (median 5.5), 5-13 deg at most, the plate 2.5-5 cm lower than placed, pelvis drift <= 3 cm, slip in the hand <= 0.4 cm.
  THE DEFAULT. After Y the robot follows the operator's arms: the synthetic operator's arm is nowhere near, so the plate tips
  0.3-0.9 s after Y in these runs (6/8 still in the hand after 2 s). A real operator must hold the plate pose before Y; the
  headset's top line says so during the hold ("plate held level: match the robot's arm, close your hand on the plate ...").
  `--no-start-hold` = walking mode (spills the plate). The CLI's `hold_pose` no longer requires `clean_start` (A2's no-touch
  rule stays A2-only); the held-start recording rule (record from the hand-over) applies to every held slice.

**Grip latch with hand tracking** (`GripLatch.takes`, `LATCH_TAKE_MARGIN` 0.05): the old rule released a latched hand on any
command that was not exactly open. With `--hand-source skeleton` a command is never exactly open, so the plate's hand was
handed over on the first tracked frame and the open tracked hand dropped the plate (the first A3 smoke). Now a held PLATE is
released only when the operator's hand command is closed at least as far as the grasp (hand_curl within 0.05 of the latch
target's, 0.50 for ep41's: the grip button, or a tracked hand closing on the plate; the trigger's pinch does not release it);
a free hand's drawn pose is still replaced by the first non-open command. Replay-safe as before (recorded after the rewrite).

**Graspable types per plate set** (`GRASPABLE["right_a2e41"] / ["left_a2e41"]` = {"lift": [...], "v1": [...]}, `graspable_ids`,
`plate_set_name` from KITCHEN_PLATES): scratchpad a2e41_sweep.py, every type as plate A in each hand, built as the A3 start
is built with ep41's own waist, held 1 s like the controller (legs / waist at the robot's yaml PD gains, the arm stiff at 150
so the grasp is what is tested, the fingers at the latch target; `hold_like_controller(kp_j=array, held_target=)`); kept when
still in the hand, < 2 cm slip and within 15 deg of flat afterwards. Lift set: 48/50 stay in the hand, 26-27 within 2 cm,
16 right [0, 1, 3, 6, 8, 12, 15, 23, 26, 28, 29, 32, 38, 39, 43, 49] / 15 left [0, 1, 6, 10, 13, 15, 23, 26, 28, 29, 32, 38, 39,
43, 49] pass the tilt cut (type 23 ends at 13 deg: the plate pivots on the pads under the weak finger PD, as in the recording,
3 -> 8 deg over ep41's hold). v1 (shallower rims): 8 right / 9 left. The A3 command should carry `--plate-set lift` as A2's did.
`slice_start_check.py` passes the grasp's latch target to its hold now.

**Checks** (2026-11-05): `slice_start_check.py --slices A3 --n 30 --render 3 --seed 8000` on the lift set: 28/30 clean (seed 8042:
a 24 cm plate slid 5.2 cm in the left hand under the 1 s hold, still held; seed 8048: the other plate's spot settled > 2 cm off);
all 30 held plates in the hand after the hold. Rand map A3 regenerated (300 draws fill the box, holding hand 172 L / 128 R,
`rand/rand_A3.png`); headset grid `rand/a3_ego.png` (12 starts: the plate level in front of the chest). Launcher smoke
(synthetic publisher, auto cycle, gate knob 3 s, `--plate-set lift`): 2 episodes recorded, replay BIT-EXACT 2/2 with
`arm_pin` in the agent state; with synthetic hand tracking the first run dropped the plate at the start (the latch rule above),
rerun after the fix: see the atomic page's A3 card for the final numbers. Page: docs/atomic_slicing_scenes (A3 card: the ep41
block, the measurements, the headset grid), NAS atomic_slicing_scenes/ + artifact.

**Command** (the A2 command with the slice changed; 30 setups = 30 consecutive seeds):

    scripts/teleop_holomotion_v14.sh --scene kitchen --robot revo2 --record --slice A3 --setup-ranges lv3 --lean-record \
        --save-dir data/teleop_holomotion_v14_kitchen_a3 --num-episodes 30 --hand-source skeleton --auto-cycle --countdown-s 5 --plate-set lift

**Traps.** The waist draw tilts a held plate as much as the arm does (check the plate, not the arm). The controller-like hold at a
flat kp 150 lets the torso sag 8 deg (the waist carries the upper body): use the robot's gains for legs / waist. The real arm PD
is 14 N m/rad, not the yaml's 40: a static PD hold at 14 is nothing like the policy, which overshoots its targets. `pgrep -f`
with the script's name in the same command line kills the shell (exit 144): use a bracketed pattern and no other mention of the
name. The A2 page's `plate_radius_m` was MuJoCo's bounding-sphere radius (`geom_rbound`), not the half-width: corrected
2026-11-05 from the sweeps' `plate_radius` (aabb) values. Not done: A4 / A5 still start with ep1's carry grasp; A3's location
is still its own box, not the A2 ends; the official-hand variant (`--revo2-hand official`) was not swept with this grasp.

### A2 ep40 / ep41 against kitchen5 ep1: the HOLD hand by hand (2026-11-05)

User: "comparing the accepted 40, 41 eps pickup whole phase including reach, grasp, lift and hold phase ... with the same phase in
kitchen5_ep1_review, show the tactile sensor difference as well as joint value numbers and difference ... two videos start from the
reaching phase and align by the first touch by reading the tactile sensor valid value; under each play window all the hand related
info for left and right and the difference". Page docs/a2_vs_kitchen5_pick (NAS a2_vs_kitchen5_pick/ + artifact), built by
docs/_scan/a2_vs_kitchen5_pick/{extract.py, template.html, build.py}; the A2 episodes got the same review pages as kitchen5
(docs/a2_ep40_review, docs/a2_ep41_review: fill -> chase view -> episode_review_page.py).

- **Pipeline fixes on the way:** `replay --fill` refused the lean A2 recordings because they carry `holomotion.pico_hands` (skeleton
  hands), a column the recorder's feature table does not list: plain source columns are now passed through (`_init_exporter(...,
  extra_features=, probe=)`, `Filler.passthrough`); the filler still refuses a non-empty output, so a second single-episode fill
  needs `--fill-out <level>_epN_filled`; `episode_review_page.py` no longer exits on a single-episode copy (filled index 0 = source
  N; pass the copy's dir as the level dir when it is not `<level>_filled`).
- **Alignment (t = 0):** the grasping hand's first fingertip-pad reading >= 0.5 N that is ON the picked plate (a `*_touch_link` of
  that hand in contact with the plate in the replayed physics): kitchen5 LEFT hand frame 2290 (its right pads first at 2400 and the
  right hand carried the plate on after the lift), A2 ep41 right 209 (a pinky-pad brush; the thumb lands at 242, four pads from 250;
  its pads read 0.5 N from the worktop at 157 already), A2 ep40 right 1136 (its hand rested on the plate from frame 3 and touched it
  at 768-899 and 1010-1095 before the grasp; the anchor is searched from frame 1100; the clip starts 6 s before). Phases by one rule:
  reach = clip start -> touch, grasp = touch -> plate 3 cm up, lift = 3 -> 10 cm, hold = 10 cm -> end (kitchen5: annotation lift
  end 2959 + 1.5 s, the robot already walking). Windows: ep41 0..448, ep40 836..1420, k5 2162..3034; clips cut from the review
  grids at 25 fps (ffmpeg -ss, re-encoded), 854x480 copies for the artifact.
- **Data at 25 Hz:** joints (`states.revo2.*.joint_position`, `action.revo2.*.cmd_position`, deg), grip mode, pads (normal N,
  tangential N, contacts, status: A2 from the filled parquet's `tactile.*`, kitchen5 from its review page's tactile.json), plate
  height / tilt and the hand bodies on the plate from the exact logs (scratchpad pick_phase_geom.py). The page: summary table,
  per pair a synced player (shared aligned time, each clip frozen outside its own range), ribbons with the pad-contact stretches,
  hand tables (both hands, measured + command, pads, plate; R - L column), the A2 - kitchen5 difference table (left-left,
  right-right, grasping hand vs grasping hand), small-multiple charts (pad normal force per fingertip, finger joints, plate).
- **What it shows:** kitchen5's pick was a two-hand rim pinch over 6.3 s of grasp with the grip button (command 84/97 deg, the
  fingers at 74-92 deg around the rim), pad peaks 50 N (left middle) / 34 N (right thumb), the plate at 54 deg at +6.5 s and 87 deg
  max; the A2 picks are one-hand holds from under the rim with the tracked hand (command 0.1-0.26 rad beyond the fingers), four
  pads at 5-31 N, the plate within 3-8 deg, grasp 2.8 s (ep41) / 3.6 s (ep40) from the first pad reading to the lift.
- **Traps:** a pad reads the worktop while reaching (the "valid value" alone is not the plate); the k5 right hand's first pad
  reading over the whole episode is at frame 662 (the dishwasher door), so "first touch" must be searched inside the pick window;
  the review tool's tactile.json is base64 (u16 x 0.1 N normal / tangential, u8 contacts / status, u8 taxels x 0.05 N).

**Hold only (same day, user: "for the comparison only compare the last holding phase period").** The page now compares the hold
alone: t = 0 = the first recorded frame with the picked plate 10 cm above its rest (the A2 gate), the window to the end of the
hold (A2: the recording's end, 1.5 s later; kitchen5: the grasping hand's last frame on the plate, 4.2 s, before the right hand
took it over and carried on). Players = frame strips drawn from one JPEG sprite per episode (`sprites.py`, 4 tiles of 640 x 360
per row, 1.3 / 1.3 / 4.0 MB) instead of seeking video: exact scrubbing, no decoder trouble (headless Chrome wedged on seeks issued
before any frame data and on a seek loop; traced and dropped). Summary table + difference table carry the hold MEANS (pad normal /
tangential force per pad, valid share, joint means, closure, plate tilt). Numbers: A2 ep41 hold 373-448, plate 10 -> 15.4 cm,
tilt mean 6.1 max 7.7 deg, right pads mean 8.6 / 6.0 / 3.4 / 1.6 / 0 N (thumb / index / middle / ring / pinky), closure 40 deg,
command 47 deg; ep40 1346-1420, 10 -> 13.8 cm, tilt 3.6 / 4.6, pads 7.4 / 1.6 / 3.2 / 0.1 / 0, closure 39; kitchen5 2671-2879, 10 ->
11.3 cm, tilt 22.5 / 29.7 deg, LEFT pads 0 / 4.0 / 0 / 3.6 / 4.6 N (the thumb and middle pads never on the plate: the rim sits on
the finger links), closure 81 deg with the grip button (command 84 / 97), its right hand back on the plate at +3.6 s.
The earlier alignment work (first valid pad reading ON the plate: k5 left 2290, ep41 right 209, ep40 right 1136) stays in
extract.py's history / this section for a full-pick comparison if wanted; the fill passthrough and review-tool fixes above hold.

**Right hand vs right hand, drawn (same day, user: "compare when the right hand is holding the plate, not the left; the values
visualisation is hard to read").** kitchen5's window is now its RIGHT-hand hold: t = 0 = frame 2880, the first frame the right hand
holds the plate alone after the left hand's last contact (2879); 4 s shown of a 28.4 s hold (the carry to the bin; `anchor
"handover"`, `hold_len_s 4`, whole-hold stats in `stats_full` / `plate_stats_full`). The numbers are drawn: a hand glyph per episode
(fingers fill with their bend, a tick for the command, fingertip dots grow and darken with the pad force, the plate drawn at its
tilt), the hold means as side-by-side bars, and an A2-minus-kitchen5 difference panel (bars at the scrub time, a diamond for the
window means); the numeric tables are folded under "the numbers". kitchen5 right-hand hold (first 4 s): plate tilt mean 30 deg
(max 48; 65 mean / 90 max over the whole carry, the plate hanging nearly vertical while walking), height falling 11 -> 3 cm, five
pads at 2.3 / 9.0 / 7.6 / 8.3 / 3.8 N (31 N), closure 66 deg with the grip button; A2 ep41 1.5 s: tilt 6.1 / 7.7, four pads
8.6 / 6.0 / 3.4 / 1.6 / 0 N (19.5 N), closure 40; ep40: tilt 3.6 / 4.6, three pads 7.4 / 1.6 / 3.2 / 0.1 / 0 (12.2 N), closure 39.
**All eleven joints (same day, user: "the hand has 11 joint values, why only 6").** The glyph draws each finger as two segments
(proximal below, distal above) and the thumb as three (metacarpal, proximal, distal, from the palm outward), each filled by its own
angle with the command tick; the difference panel has eleven joint rows (+ closure); the window charts one panel per joint; the
summary a chart of the eleven window means with the command tick (kitchen5's grip commands 84 / 97 deg, the fingers sit at 48-79).

### Atomic page made concise + the collected A1 / A2 samples (2026-11-05)

User: "add the current collected A1 and A2 samples to the atomic page, and simplify the html, make it less words". `build.py` was
rewritten with short texts (the verbose version is kept in the scratchpad as build_atomic_verbose_20261105.py and in the git
history of nothing -- docs are untracked): one-paragraph starts, two-column randomization tables, the histories folded into a
dated list, the randomization reading one sentence per scene. New block "Collected samples" on the A1 and A2 cards:
`samples.py <level dir> <tag> [--episodes ..] [--env K=V] [--no-render]` renders the head camera at each episode's end event from
the exact log (320 x 180; separate processes per knob group: A2 episodes 14-43 need KITCHEN_PLATE_BALLS=2) into
docs/atomic_slicing_scenes/samples/<tag>/ and writes samples_<tag>.json (seconds, gate, setup, success); `--no-render` writes the
facts only. A1: 37 complete of 38 (3.8-7.8 s, 2026-11-03); A2: 44 (9-160 s, 2026-11-04), 8 level with the balls on (from the
a2_grasp_poses scoring), the first 2-episode session not counted. Trap: samples.py's ROOT was parents[2] (= docs/) at first, so
the renders landed in docs/docs/ and the json dump failed at the end -- the rows are now written by --no-render from the images.

### Videos on the two pages (2026-11-05, user: "I need to see the grasping videos" / "each ep video")

- **a2_grasp_poses:** `analyze.py --clips` renders, per episode, the grasp close-up (the grasping hand orange, the plate-centred
  free camera, the balls as recorded or simulated) next to the head camera, 25 fps, from 1 s before the hand settled on the plate
  (min(rest, onset) - 50 frames, at most 12 s) to the end of the recording, with the phase (reach / grasp / lift / hold), time,
  plate tilt and rise burnt in: docs/a2_grasp_poses/clips/epNN.mp4 (1280 x 360, crf 26) + clips_small/ (854 x 240, the artifact).
  The success cards show the clip above the stills, the failure table a small player per row. The same run fixed `plate_radius_m`
  to the aabb half-width (geom_rbound was 4 cm too large).
- **atomic_slicing_scenes:** `episode_videos.py <level dir> <tag> [--episodes] [--env]` renders every recorded episode kinematically
  from the exact log (states set + rendered, no physics): the head camera (PinholeView left) next to a chase camera (2.6 m behind-
  right of the pelvis, smoothed heading), 25 fps, time + the slice's end event + milestones burnt in, to
  docs/atomic_slicing_scenes/videos/<tag>/epNNN.mp4 (1280 x 360, crf 26) + videos_<tag>[_part].json. The NAS page embeds them in the
  "Collected samples" tiles (poster = the end-event thumbnail); the artifact copy links to the NAS files (too many MB for the artifact).
  Separate processes per knob group (A2 episodes 14-43: KITCHEN_PLATE_BALLS=2). ~20 ms per frame; A1 37 episodes in ~4 min.

### Plate set v2: wide-rim pasta plates added to the randomization (2026-11-06)

User: "create some new plates reference this one [a photo of a white wide-rim pasta plate: deep well, wide flat rim, foot ring];
add to the V2 randomization." Page: docs/kitchen_plates_v2/ (NAS fleet_status/kitchen_plates_v2/).

**The family.** `make_plates.py --family pasta`: a bowl-shaped well 20-30 mm below the rim corner (a superellipse wall, k 2-2.8,
that steepens into the corner; a flat floor 62-88 mm across so the two indicator balls rest on it), a foot ring under the well
(22-30 % of the radius), a flat rim 38-49 % of the radius wide (43-65 mm) with a rounded edge; round, D 224-265 mm, H 33-46 mm,
rim 4.5-7 mm thick, mostly plain whites (some rim bands / rings), porcelain / stoneware / melamine, all at the 400 g clamp (a real
one is 600-900 g). Collision: 24 sectors x (lower bowl, upper bowl, rim) hulls + the foot cylinder (73 pieces: one wall hull per
sector would fill the bowl to its chord), four sphere feet. The rim underside sits 28-39 mm above the counter. `--id-prefix`,
`--id-start`, `--foot-spheres` are new generator knobs; the dinner family is byte-for-byte unchanged.

**The set.** `scripts/kitchen_plates/make_set_v2.py`: data/kitchen_plates_v2 = the lift set's 50 plates (same ids, same order,
copied) + the first 20 of 40 pasta candidates (seed 23) that pass check_under_rim (fingers 59 mm / thumb 62 mm under, median),
check_pinch (40/40 held through 12 cm, slip 0; the plate hangs at 40 deg in the one-hand rim pinch) and check_rack_fit (40/40
stand, lean 13-21 deg) -- all 40 passed. 70 types, 32 MB, manifest with `family` per plate and `sources`.

**Randomization.** `--plate-set v2` (CLI; `grasp_start.PLATE_SET_NAMES`): setup plate = seed mod 70, plate B independent in the
atomic slices (so a seed lands on a pasta plate 2 times in 7), the S slices +25 mod 70. Types 0-49 = the lift set, so the A2 bank
starts keep their indices; NEW KNOB `KITCHEN_BANK_PLATES=draw|recorded` (scene_setup.sample, recorded with the KITCHEN_* knobs):
a bank start draws its plate TYPES from the loaded set instead of taking the A1 episode's; `--plate-set v2` sets `draw` (else
A2 would never see a pasta plate), lift / v1 keep `recorded`. The recorded location, pose and plate spots stay. Headless A2
launcher smoke (synthetic publisher on 6018, balls, v2): seeds 8105 / 8106 -> pasta_05 + pasta_03 / pasta_06 + pasta_17 at the
recorded A1 spots, the usual clearance nudge, both episodes replay BIT-EXACT (and seeds 8055 / 8056 with lift types, also exact).

**A3 (held start) does not take the pasta plates.** Sweep of ep41's hold over all 20 (scratchpad a2e41_sweep.py, both hands):
the index / middle distal links end 16-20 mm inside the bowl wall (it starts 45-65 mm in from the edge) and the plate tilts
20-70 deg in the hold; slid 3.5-5 cm further out of the hand (new `grasp_start.FAMILY_OUT_M`, `grasp_pose(..., out_m)`,
`plate_family()` from the geom names) the fingers clear the bowl but the thumb no longer clamps the rim and the plate drops.
So `GRASPABLE["*_a2e41"]["v2"]` = the lift lists (A3 draws lift types under v2); FAMILY_OUT_M["pasta"] stays 0.0.

**Tools.** `render_plates.py` (product-photo render + a section SVG per plate; the page's images), `make_set_v2.py`,
docs/_scan/kitchen_plates_v2/{build.py,template.html}, scratchpad kitchen_frame.py (one exact-log frame, head + chase camera).

### A2 two-hand level lifts in the 2026-11-09 session (2026-11-10)

User: "check current A2 recording, find all the horizontal grasp ep with two hand lifting up, show me the video and numbers".
Page: docs/a2_two_hand_lifts/ (NAS fleet_status/a2_two_hand_lifts/). Tool: docs/_scan/a2_two_hand_lifts/analyze.py + build.py.

The session: teleop_holomotion_v14_kitchen_a2_hands episodes 44-65 (headset hand tracking; 65 saved without the gate) and
teleop_holomotion_v14_kitchen_a2_grip 0-9 (controller grip), plate set v2 with the balls. Each exact log is replayed bit for bit
and the contact forces between each hand and the lifted plate are read after every frame. Rule: both hands touch the plate in
>= 80 % of the 0.5 s gate hold, the weaker hand gives >= 20 % of the upward force, and the plate stays within 10 deg of flat.
Result: 2 pass (hands ep 51: left/right 20/80 %, max tilt 6.2 deg; grip ep 8: 25/75 %, 5.5 deg); 3 are two-hand lifts that tilt
10.6-14.3 deg (grip 0 and 2 at about 45/55 %, hands 62 at 22/78 %). In 12 more episodes both hands touch while one carries.

### A3 starts from the A2 press-and-grasp ends (2026-11-10)

User: "use the 28 scenes to set the A3 init pose" (the 28 A2 episodes in which one hand pressed one side of the plate down and
the other grasped the raised side, docs/a2_press_grasp). Page: docs/a3_press_starts/ (artifact 1JWCqPf3tu7uGoFCFEwAes).

**Default A3 start = a recorded A2 end, exactly, in seed order** (`KITCHEN_A3_START=a2_press`; `a2e41` = the 11-05 level hold).
Bank `start_banks/a2_press_end.json` (scripts/holomotion_v14_checks/a2_press_bank.py, one process per recording setup + `--merge`):
the exact log's physics state at `slice.event_frame` (the A2 gate held 0.5 s) read by name -- pelvis x, y, heading AND its
quaternion (the 1-5 deg lean: an upright pelvis put 9 of 28 starts 1-14 mm into the worktop), 29 body joints, both hands'
fingers, the held plate in the right hand base frame + that plate's radius, the operator's hand command at that frame (the
latch target), the other plate's spot (x, y, yaw), both plate types and their plate set. scene_setup `_A3_PRESS` (bank kind with
`held_bank`, `bank_plates="recorded"`: KITCHEN_BANK_PLATES=draw is ignored, the grasp is exact for that plate only), `bank_fits`
(v1 entries need --plate-set v1 -> 13 starts; lift entries also serve v2 -> 15 starts; v2 entries with both types < 50 also serve
lift -> 8). grasp_start: `plate_in_hand.grasp="bank"` carries its grasp inline (`grasp_spec`, `held_latch_target`, no family
slide), `apply_pose` sets a pose dict's `root_quat`. The free (left) hand is drawn resting; the held start (motion tracking +
the pinned holding arm) as before.

Checks: placement 28/28 with no robot contact; slice_start_check 1 s pinned hold 27/28 keep the plate (a2_hands ep 37 drops);
real controller (headless launcher, synthetic publisher whose hand never closes, one episode per start, countdown 2 s and 8 s):
28/28 hold the plate at the hand-over, slip since placement <= 2.6 cm in 27 (ep 37: 7 cm, tilt 12 -> 45 deg), tilt within
~13 deg of the recording; after the hand-over the synthetic arms hang and 5 plates drop (test-publisher artifact). Smoke datasets
data/teleop_holomotion_v14_kitchen_slices_smoke_20261110a3press{,8}_{v1,v2}: 56 episodes, all BIT-EXACT; the 11-05 A3 smoke
(ep41 start) still replays bit-exact.

**Palm-up filter (2026-11-10, later; user: "filter one more time ... only for those right palm is facing upward" + "update A3
scene as well").** The Revo 2 palm faces +x of the hand base (a small finger flex moves the tips toward +x). Bank entries carry
`held.palm_up_deg` (the palm normal's angle from straight up at the end state): 23 of 28 point it 9-56 deg from up (the plate
rests on the hand), 5 point it 114-136 deg (a2_hands ep 0, 4, 5, 37, 66: the hand grips the rim from above, fingers up) -- a
clean gap. The A3 scene now draws only the palm-up ends: `KITCHEN_A3_PALM=up` (default; `any` = all 28). Per plate set: v1 10,
v2 13 (incl. the lift ep 14), lift 6. Start check after the filter: v2 13 distinct, v1 10 distinct, all placed clean, all keep
the plate through the 1 s hold (the weak ep 37 was palm-down). The real-controller smokes above already covered all 23.

### Final A2 dataset: the 30 flattest picks (2026-11-10)

User: "i have recorded 15 more, added previous 23, please find 30 ep which the plate is most flat and save them into a new final
A2 dataset". New: a2_grip_v1 episodes 5-20 (16, 20:56-21:02, grip, v1 plates, balls; 14 press-and-grasp, eps 5 and 17 lifted by the
right hand alone). Pool 39 = those 16 + the 23 palm-up press-and-grasp episodes. Ranked by the largest plate tilt from flat over
the 0.5 s A2 gate hold, from the bit-exact replays (docs/_scan/a2_final/rank_flat.py -> rank.json); kept the 30 smallest: 12.0-41.3
deg (the 9 left out: 42.4-70.8). Composition: a2_grip_v1 17, a2_grip 9, a2_hands 4; all palm up (9-87 deg from up); 29 press-and-
grasp + a2_grip_v1 ep 5 (rank 15, right hand alone).

**Dataset:** data/teleop_holomotion_v14_kitchen_a2_final (lean, 30 episodes, 19,033 frames; selection.json next to the LeRobot root)
by `scripts/holomotion_v14_checks/select_episodes.py --selection docs/_scan/a2_final/selection.json --out <dir>`: episodes
ordered by recording setup (grip_v1 0-16, hands 17-20, grip 21-29), flattest first within each; parquets written with the source
arrow schema + huggingface metadata (pandas made the fixed-size arrays variable lists and LeRobot failed: "Could not infer dtype of
NoneType"); holomotion.pico_hands added as NaN to the grip episodes (the recorder's missing-sample convention); exact logs copied
byte-identical; episodes.jsonl lines + `source` + `selection`. Checks: every column equal to its source (NaN-aware), LeRobot
loads it, replay --all 30/30 BIT-EXACT.

**Replay tool for mixed setups:** the kitchen kit reads its knobs at import, so one process cannot rebuild v1 and v2 plates or
balls on / off. `replay_holomotion_v14` now takes `--episodes 0,3,5-9`, and `--all` / `--episodes` over several recording setups
(env knobs, scene root, robot) run one child process per setup in turn (KITCHEN_* not inherited); `--fill` appends to the same
output (fill_record honours HOLOMOTION_FILL_APPEND). Tested: two episodes across two setups filled into one dataset, both bit-exact.

### A3 starts from the final A2 dataset (2026-11-10 night)

User: "use the final one to create A3 scenes". A3 default now `KITCHEN_A3_START=a2_final`: the same bank start kind (`_A3_FINAL` =
`_A3_PRESS` with bank `a2_final_end`) over the 30 episodes of data/teleop_holomotion_v14_kitchen_a2_final. Bank built by
`a2_press_bank.py --dataset data/teleop_holomotion_v14_kitchen_a2_final --held docs/_scan/a2_final/held.json --episodes 0-16|17-20|21-29
--tag final` (one process per recording setup) + `--merge --out start_banks/a2_final_end.json`; entries carry `source` (the original
dataset / episode) and `selection` (rank, tilt, palm). Per plate set: v1 21 (final eps 0-20), v2 9 (21-29). All palm up; final ep 7
(a2_grip_v1 ep 5) is the one right-hand-alone pick. Older kinds: `a2_press` (the press-and-grasp ends, palm filter) and `a2e41`.
Checks: start checker 30/30 placed clean and held through 1 s (slip <= 3.1 cm); real controller, 8 s countdown, synthetic publisher:
30/30 held at the hand-over, slip < 3 cm in 28 (final ep 1: 4.4 cm, 15 -> 30 deg; ep 10: 3.1 cm); smoke datasets
data/teleop_holomotion_v14_kitchen_slices_smoke_20261110a3final_{v1,v2}: 30/30 BIT-EXACT. Page: artifact 1JWCqPf3tu7uGoFCFEwAes v3.

### A4 starts from the A3 ends (2026-11-10 night)

User: "A3 done, please adjust A4 based on A3 and provide me the cmd to teleop". A4 default now `KITCHEN_A4_START=a3_end` (`v1` = the
old stance box at the bin with ep1's carry grasp, walking mode): the A3 episodes of data/teleop_holomotion_v14_kitchen_a3_final as
they ended (A3 gate: at the bin, standing still 0.5 s), in seed order -- robot, pose, lean, the plate in the right hand, the other
plate, recorded types (v1) -- via the same bank kind (`_A4_A3END`, no spawn-back) and a held start (slices A4 start_mode "hold",
hold_pin "held_arm", only with a3_end). docs/_scan/a3_final_ends/end_state.py: 29 of 30 A3 ends still hold the plate (ep 4 put it
down); the real-controller smoke (8 s countdown) dropped the plate for A3 ep 13 and slid it 9.3 cm for ep 15 (which the 1 s checker
hold drops) -> both left out (excluded_a4.json, held_a4.json) -> start_banks/a3_end.json = 27. Checks: checker 27/27 clean + held
(slip <= 2.5 cm); smoke 27/27 held at the hand-over (slip <= 4.3 cm); 29 smoke episodes BIT-EXACT
(data/teleop_holomotion_v14_kitchen_slices_smoke_20261110a4_v1). Page docs/a4_starts (artifact XLeZhqsFQrpgH7ituujyEW). Command: the
A3 command with --slice A4, --plate-set v1, --save-dir data/teleop_holomotion_v14_kitchen_a4_final.

### A5 starts from the A4 ends (2026-11-10 night)

User: "A4 is done do A5". A5 default now `KITCHEN_A5_START=a4_end` (`v1` = ep1's swipe stance + pose, the foot placed on the pedal by
IK): the A4 episodes of data/teleop_holomotion_v14_kitchen_a4_final as they ended (A4 gate: lid past 1.3 rad for 0.3 s), in seed
order, as the A4 bank start (`_A5_A4END`, preset bin_lid "open") plus the bin joints: the bank builder's new `--scene-joints
k_bin_lid,k_bin_pedal` stores their recorded positions, `_bank_pose` passes them as the pose's `scene_joints` and `apply_pose` sets them
(pedal 0.18 rad = its stop, lid 1.45). In all 31 A4 ends the RIGHT foot holds the pedal (ankle roll or pitch link on the tread:
count both). Held start (slices A5 start_mode "hold", hold_pin "held_arm", only with a4_end). docs/_scan/a4_final_ends/end_state.py:
30 of 31 still hold the plate (ep 12 laid it on the lid); the real-controller smoke dropped it for ep 16 -> start_banks/a4_end.json = 29.
Checks: checker 29/29 held, lid 1.45 after 1 s (the checker counts the foot on the tread as a touch; 3 starts within 1.4 mm of the
worktop / bin wall / tread as recorded); smoke (8 s countdown) 29/29 held, lid 1.40-1.45 at the hand-over; 30 smoke episodes BIT-EXACT
(data/teleop_holomotion_v14_kitchen_slices_smoke_20261110a5_v1). Page docs/a5_starts (artifact 5R8iZ3RK9zv9xi8smeW3d6).
