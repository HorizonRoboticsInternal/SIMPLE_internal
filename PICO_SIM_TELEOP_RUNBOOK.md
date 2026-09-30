# PICO sim teleop — runbook

HoloMotion v1.4 controller, G1 + Dex3 (optionally with the 3.2 kg backpack), in the three real-to-sim scenes:
`bottle_bin`, `bowl_sink`, `coffee_cart`. Runs on this workstation (`~/wrk/SIMPLE`) with a PICO 4 / 4 Ultra and two
ankle motion trackers. Verified with scripted input on 2026-09-29; the details behind every flag are in
`docs/holomotion_v14_teleop.md`.

---

## 0. Once per machine (done on this workstation)

- XRoboToolkit PC service installed (`/opt/apps/roboticsservice`), XRoboToolkit app on the headset.
- Policy models in `data/holomotion/v14_models/` (walking + motion tracking).
- The backpack model (`model_22000`, the default) in `data/holomotion/v14_models_backpack_3p2/`, copied from the NAS:
  `rsync -a /mnt/nas28/alan.jiang/holomotion_models/v14_models_backpack_3p2/ data/holomotion/v14_models_backpack_3p2/`, then `sha256sum -c SHA256SUMS` in that folder.
- Publisher env `~/miniconda3/envs/holomotion_teleop` (PICO body → SMPL → robot reference).

## 1. PC checks (every session)

```bash
pgrep -af RoboticsServiceProcess || bash /opt/apps/roboticsservice/runService.sh   # XRoboToolkit service
ss -ltnp | grep -E ':13579|:6001'      # must print nothing: headset video / reference ports free
hostname -I                            # the PC's address for the headset (192.168.50.x)
```

If a port is taken, another sim or replay is holding it: stop that process first.

## 2. Headset (every session)

1. Ankle trackers on and calibrated (Motion Tracker app → Calibrate: stand, then look down at the trackers).
2. Headset on the same network as the PC.
3. XRoboToolkit app → Connect to the PC's address → Status **Working**.
4. Tracking: **Head** + **Controller**, PICO Motion Tracker **Full-body**. **Send** on.
5. Remote Vision: source **Zedmini**. Press **Listen** only after step 3 below.

## 3. Start the sim

```bash
cd ~/wrk/SIMPLE
scripts/teleop_holomotion_v14.sh --scene bottle_bin --record
```

What the defaults give you:

| | default | change with |
|---|---|---|
| scene setup | level-3 ranges, new every episode (below) | `--setup-ranges teleop` (the older ranges, fixed table height) |
| seed | a new random one every launch, printed | `--seed N` |
| start pose | slightly random every episode | `--no-random-init-pose` |
| headset | one picture: the HBVCAM stereo camera's rectified left eye | `--stream-camera fisheye` (both raw fisheye eyes) |
| recorded `ego_view` | the same picture: 1280 × 720, 104.7° × 72.2° | `--record-camera head` (SIMPLE's head camera, 640 × 360) |
| camera tilt | 10° further down than the calibrated mount (56° below horizontal standing) | `--head-tilt-deg 0` (46°), any angle, negative = up |
| robot | G1 + Dex3 + HBVCAM head camera + 3.2 kg backpack, backpack motion model (`model_22000`) | `--motion-model public --backpack-kg 0` |

Other scenes: `--scene bowl_sink`, `--scene coffee_cart`.

**Level-3 setups.** As in the level-3 evaluation sets, every episode draws:

- the robot start: ±10 cm back/forward, ±5 cm sideways;
- the item, inside the level-3 region;
- the table / counter / cart-box height, ±4 cm;
- 3 new distractors.

The bin and the cart stay put. Episodes go to `level-3_pinhole` (`level-3` = the setup ranges, `_pinhole` = the
recorded camera; a dataset holds one image size).

**Camera tilt.** The tilt turns both head cameras, the one the headset shows and SIMPLE's head camera.
- 10° keeps the item in view from the start spot and puts both hands in the picture at the table.
- At 20° the head camera loses the bottle from the start spot.
- The tilt is saved with each episode and the replay rebuilds it.
- 0 is the real robot's calibrated mount; the sim picture then matches the real camera.

Healthy start (about 15 s):

```
[HoloMotion v1.4] policy node ready in ... s
[TCPServer] Ready for connections on 0.0.0.0:13579
[HoloMotion v1.4] setup 0 (seed ...: robot (...), target (...), ...)
[Record] saving to .../data/teleop_holomotion_v14/simple/<env>/level-3_pinhole
```

Then press **Listen** in the app: you see the robot's HBVCAM camera with the status overlay. The body reference runs in
the background; check it with `tail -f /tmp/holomotion_v14_publisher.log` (`body frames ~ Hz` lines).

## 4. First episode

The robot starts standing on the floor, no leash, in walking mode. Recording starts by itself once your PICO
controllers are connected. Press **Y** for motion tracking when you are in position.

With `--no-quick-start` the robot starts hanging on the leash: **left stick click** (stand up, 3 s) → **A** (walking,
the leash lowers) → **Y** (motion tracking) → **left menu + A** (record).

## 5. The collection loop

1. Do the task. The overlay shows `REC 12.3 s` and the task checks (`grasped [x]  at_bin [ ]  placed [ ]`).
2. **Left menu + A** saves. The next random setup loads with the robot standing in walking mode, no leash, and
   recording starts by itself from the (random) start pose. Press **Y** for motion
   tracking again.
3. Repeat. The episode also saves itself when the task check passes (item placed).

If something goes wrong:

| you want to | press | result |
|---|---|---|
| drop this recording, stay in the scene | left menu + B | not saved; left menu + A starts a new recording |
| retry the same setup | both triggers | not saved; the same setup reloads and re-records |
| skip to another setup | left menu + Y | not saved; a new setup loads and records |

## 6. Controls

The sim map (default; `--button-map robot` = the robot's v1.4.1 map, B motion / Y back to walking):

| PICO | does |
|---|---|
| A | walking mode; from motion tracking: back to walking |
| Y | motion tracking: the robot follows your body |
| B | nothing on the robot (the PICO app's view toggle) |
| left stick / right stick x | walk / turn (walking mode) |
| grips | close the Dex3 hands |
| X | zero torque |
| right stick click | emergency stop |
| left stick click | stand up (only with `--no-quick-start`, or after X / emergency stop) |

Sim only (hold the left menu button; nothing reaches the robot while it is held):

| PICO | does |
|---|---|
| left menu + A | start recording / save |
| left menu + B | drop the recording |
| left menu + Y | new random setup |
| both triggers | retry the same setup |

## 7. Headset view

One picture by default (`--stream-view single`): the robot's stereo head camera, left eye, as a flat (pinhole) image,
104.7° × 72.2° at 1280 × 720. The overlay:

```
REC  12.3 s                          (or: starting ... / not recording)
MOTION   saved 3/100   setup 5       (WAIT / STAND / WALK / MOTION)
grasped [x]  at_bin [ ]  placed [ ]
L-menu + A save  + B drop  + Y new setup
```

The app's own right **B** toggles its view, and B does nothing on the robot: if the picture looks cut in half, press
**B**. `--stream-view mono` puts the camera in both halves, `stereo` sends left | right.

## 8. Stop

1. Drop an unfinished episode first (**left menu + B**): Ctrl-C saves a recording that is still running.
2. **Ctrl-C** in the terminal. The publisher stops with it. The session also stops by itself after 100 saved episodes
   (`--num-episodes`).

## 9. After the session

Data: `data/teleop_holomotion_v14/simple/<env>/level-3_pinhole/` — `data/` (parquet), `videos/` (the recorded
HBVCAM picture), `meta/` (task string, random setup per episode), `replay/` (bit-exact logs). The folder name is the
setup ranges (`level-3`; `level-0` with `--setup-ranges teleop`) plus the recorded camera (`_pinhole`, `_fisheye`,
none for `head`). Episodes recorded before 2026-09-30 are in `level-0` (head camera).

```bash
D=data/teleop_holomotion_v14/simple/G1WholebodyBottleBinTeleop-v0/level-3_pinhole
python -m simple.cli.replay_holomotion_v14 $D --all --mode action          # every episode: BIT-EXACT
python -m simple.cli.replay_holomotion_v14 $D --episode 0 --video ep0.mp4  # head | third-person video
```

Quick-started episodes replay from the actions alone. Episodes that started with the leash up also use the logged
leash force.

## 10. Troubleshooting

| symptom | check / fix |
|---|---|
| no picture in the headset | sim running and `[TCPServer] Ready` printed before you pressed Listen; app source Zedmini; same network; port 13579 free |
| two pictures, or half a picture | press B (the app's view toggle; nothing on the robot); keep `--stream-view single` |
| overlay stays on `starting ...`, terminal says `motion tracking not ready` | publisher log shows `body frames` at a steady rate; trackers calibrated, Full-body mode; XRoboToolkit service running |
| terminal says `waiting for the PICO controllers` | controllers asleep, or Send is off in the app |
| robot falls | both triggers: the same setup reloads (the recording is dropped) |
| wrong task string in the data | fixed 2026-09-29 for bottle_bin; `BOTTLE_BIN_INSTRUCTION="..."` overrides |
| replay says `model DIFFERENT` | the scene code changed since recording; record with `--save-model` to keep the compiled model (~125 MB/episode) |
| port 6001 busy | `REFERENCE_PORT=6002 scripts/teleop_holomotion_v14.sh ...` |

## 11. Options

| option | default | what it does |
|---|---|---|
| `--scene` | bottle_bin | bottle_bin, bowl_sink, coffee_cart |
| `--record` | off | save LeRobot episodes + bit-exact logs |
| `--no-quick-start` | quick start on | hang on the leash and stand up by hand instead of starting standing with the policy running |
| `--button-map` | sim | sim: A walk, Y motion, B free; robot: the v1.4.1 map (B motion, Y back to walking) |
| `--no-random-init-pose` | random start pose on | start every episode from the policy's default pose |
| `--init-pose-scale` | 1 | scales those ranges (arms ±0.15 rad, waist ±0.05, legs ±0.03, base ±2 cm / ±3°) |
| `--backpack-kg` | 3.2 | backpack mass on the robot (0 = none) |
| `--motion-model` | backpack | backpack (`model_22000`) or public (v1.4.1 `model_16200`) |
| `--motion-model-bundle` | — | another backpack model folder, hash-checked; overrides `--motion-model` |
| `--seed` | random | first setup's seed; the n-th setup uses seed + n |
| `--max-distractors` | 3 | distractors per setup: 0..N |
| `--no-randomize` | — | the nominal scene every time |
| `--setup-ranges` | lv3 | lv3: level-3 eval ranges (robot ±10 / ±5 cm, level-3 item region, table / counter / cart-box height ±4 cm, 3 distractors), folder `level-3`; teleop: the older ranges (robot ±8 / ±10 cm ±10°, bin/cart moves, 0–3 distractors, fixed table), folder `level-0` |
| `--stream-view` | single | single, mono, stereo |
| `--robot` | stereo | stereo: the G1 with the HBVCAM stereo head camera; stock: SIMPLE's G1 |
| `--stream-camera` | pinhole | pinhole: the stereo camera's left eye, flat; fisheye: both eyes' fisheye images side by side (like the real camera); head: the scene's head camera |
| `--auto-motion` | off | quick start also presses Y (motion tracking) for you |
| `--record-on-motion` | off | each recording starts when you press Y (motion tracking), not at the start of the scene |
| `--record-camera` | auto | what is saved as ego_view: auto = the headset's camera (pinhole → 1280 × 720, folder `_pinhole`; fisheye → the fisheye pair, `_fisheye`); head = SIMPLE's head camera (640 × 360) |
| `--head-tilt-deg` | 10 | point the head cameras (headset and recorded) this many degrees further down than the calibrated mount; 0 = the real mount, negative = up |
| `--stream-port` | 13579 | the port the app connects to (change only for tests) |
| `--num-episodes` | 100 | stop after this many saved episodes |
| `--save-dir` | data/teleop_holomotion_v14 | where the datasets go |
