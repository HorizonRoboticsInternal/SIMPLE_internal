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

The robot's v1.4.1 map:

| PICO | does |
|---|---|
| left stick click | stand up (MOVE_TO_DEFAULT, 3 s to the default pose) |
| right A | policy on, walking mode (also lowers the sim safety band) |
| left stick / right stick x | walk (0.5 m/s fwd, 0.2 m/s side) / turn (0.7 rad/s) |
| right B | motion tracking: the robot follows your body |
| Y or A | back to walking |
| grips | close the Dex3 hands |
| X | zero torque |
| right stick click | emergency stop (2 s damping, then zero torque) |

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

## Quick start (faster collection)

```bash
scripts/teleop_holomotion_v14.sh --scene bottle_bin --backpack-kg 3.2 --record --quick-start
```

After a save, left menu + Y or both triggers, the new scene starts with the robot already standing on the floor at
the policy's default pose, with no leash. The CLI then presses A (walking policy) and, 0.5 s later, B (motion
tracking) for you, through the same code paths as the real buttons. Recording starts as soon as motion tracking is on,
about 1 s after the reset, so you only need left menu + A to save. The scene at launch starts as usual (L3, A, B,
left menu + A), or press left menu + Y to jump straight into a quick-started setup. Episodes recorded this way never
have the leash up, so they replay bit-exactly from the actions alone.

## Headset view

The XRoboToolkit app expects a stereo camera (2 × 16:9 side by side). By default it shows that frame as two pictures;
its right **B** toggles a single view that shows only the left half. `--stream-view` picks what SIMPLE sends:

| view | frame | what you see |
|---|---|---|
| `single` (default) | the left head camera once, centred | one picture in the app's default view |
| `mono` | the left camera in both halves | one picture after B (the app's single view); two identical ones before |
| `stereo` | left \| right | the old dual stream |

B is also the motion-tracking button, so every B press also flips the app's view. With `single`, if the picture looks
cut in half, press B once more; in motion mode B does nothing else. With `--quick-start` you never need B.

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

The ranges are `RANGES` in `src/simple/teleop/holomotion_v14/scene_setup.py`; `--max-distractors`, `--no-randomize`
(the nominal scene). The setup goes into the scene through SIMPLE's own DR (zero-width robot/target regions, the
distractor count, numpy's seed) and the scene module's layout dict; the scene task files are unchanged. Each episode's
setup is in `meta/episodes.jsonl` (`scene_setup`, next to SIMPLE's `environment_config`).

## Recording

LeRobot v2.1 at 50 fps, the decoupled-WBC schema (as the psi0 datasets) plus:
`teleop.navigate_command` (the policy's vx, vy, vyaw, 0), `teleop.latest_obs` (the 65-value reference),
`policy.mode` (0 walking, 1 motion), `policy.raw_action`, `policy.target_real`. `action` is the 43-joint target
(29 body + 2 × 7 Dex3). `observation.object_poses` has fixed slots: the scene's objects, then `distractor_0..N-1`
(NaN when a setup has fewer).

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

`--backpack-kg 3.2` adds HoloMotion's G1 backpack (`backpack_link` on the torso at (−0.095, −0.0005, 0.165), from
their payload assets; no collision, as in their MuJoCo eval) at 3.2 kg, the mass of the v1.4.1 backpack policy, to the
Dex3 G1. `--motion-model-bundle DIR` loads that policy's motion model, `model_22000`. It is not in the repository;
the directory is either the upstream bundle (`config.yaml` + `model_22000.onnx`) or the model folder copied out of the
robot's collection image. The hashes are checked against the upstream lock. The walking model is the public one in
both.

```bash
scripts/teleop_holomotion_v14.sh --scene bottle_bin --backpack-kg 3.2 --motion-model-bundle ~/backpack_model
```

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
