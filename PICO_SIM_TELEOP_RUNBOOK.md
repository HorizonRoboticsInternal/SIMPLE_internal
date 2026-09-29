# PICO sim teleop — runbook

HoloMotion v1.4 controller, G1 + Dex3 (optionally with the 3.2 kg backpack), in the three real-to-sim scenes:
`bottle_bin`, `bowl_sink`, `coffee_cart`. Runs on this workstation (`~/wrk/SIMPLE`) with a PICO 4 / 4 Ultra and two
ankle motion trackers. Verified with scripted input on 2026-09-29; the details behind every flag are in
`docs/holomotion_v14_teleop.md`.

---

## 0. Once per machine (done on this workstation)

- XRoboToolkit PC service installed (`/opt/apps/roboticsservice`), XRoboToolkit app on the headset.
- Policy models in `data/holomotion/v14_models/` (walking + motion tracking).
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
scripts/teleop_holomotion_v14.sh --scene bottle_bin --backpack-kg 3.2 --record --quick-start
```

Other scenes: `--scene bowl_sink`, `--scene coffee_cart`. Without the backpack: drop `--backpack-kg`.

Healthy start (about 15 s):

```
[HoloMotion v1.4] policy node ready in ... s
[TCPServer] Ready for connections on 0.0.0.0:13579
[HoloMotion v1.4] setup 0 (seed ...: robot (...), target (...), ...)
[Record] saving to .../data/teleop_holomotion_v14/simple/<env>/level-0
```

Then press **Listen** in the app: you see the robot's head camera with the status overlay. The body reference runs in
the background; check it with `tail -f /tmp/holomotion_v14_publisher.log` (`body frames ~ Hz` lines).

## 4. First episode

The scene at launch starts with the robot hanging on the leash. Either:

- **Left menu + Y** (recommended): a new setup comes up quick-started — standing, motion tracking on, recording.
- By hand: **left stick click** (stand up, 3 s) → **A** (walking, the leash lowers) → **B** (motion tracking) →
  **left menu + A** (record).

## 5. The collection loop (`--quick-start`)

1. Do the task. The overlay shows `REC 12.3 s` and the task checks (`grasped [x]  at_bin [ ]  placed [ ]`).
2. **Left menu + A** saves. The next random setup loads with the robot standing, no leash; motion tracking and
   recording start by themselves within about 1 s.
3. Repeat. The episode also saves itself when the task check passes (item placed).

If something goes wrong:

| you want to | press | result |
|---|---|---|
| drop this recording, stay in the scene | left menu + B | not saved; left menu + A starts a new recording |
| retry the same setup | both triggers | not saved; the same setup reloads and re-records |
| skip to another setup | left menu + Y | not saved; a new setup loads and records |

## 6. Controls

The robot's v1.4.1 map:

| PICO | does |
|---|---|
| left stick click | stand up (3 s to the default pose) |
| A | walking mode (lowers the leash) |
| left stick / right stick x | walk / turn |
| B | motion tracking: the robot follows your body |
| Y or A | back to walking |
| grips | close the Dex3 hands |
| X | zero torque |
| right stick click | emergency stop |

Sim only (hold the left menu button; nothing reaches the robot while it is held):

| PICO | does |
|---|---|
| left menu + A | start recording / save |
| left menu + B | drop the recording |
| left menu + Y | new random setup |
| both triggers | retry the same setup |

## 7. Headset view

One picture by default (`--stream-view single`), with the overlay:

```
REC  12.3 s                          (or: starting ... / not recording)
MOTION   saved 3/100   setup 5       (WAIT / STAND / WALK / MOTION)
grasped [x]  at_bin [ ]  placed [ ]
L-menu + A save  + B drop  + Y new setup
```

The app's own right **B** toggles its view. B is also the motion-tracking button, so if the picture looks cut in half,
press **B** once more (in motion mode it does nothing else). `--stream-view mono` puts the camera in both halves,
`stereo` sends left | right.

## 8. Stop

1. Drop an unfinished episode first (**left menu + B**): Ctrl-C saves a recording that is still running.
2. **Ctrl-C** in the terminal. The publisher stops with it. The session also stops by itself after 100 saved episodes
   (`--num-episodes`).

## 9. After the session

Data: `data/teleop_holomotion_v14/simple/<env>/level-0/` — `data/` (parquet), `videos/` (head camera), `meta/` (task
string, random setup per episode), `replay/` (bit-exact logs).

```bash
D=data/teleop_holomotion_v14/simple/G1WholebodyBottleBinTeleop-v0/level-0
python -m simple.cli.replay_holomotion_v14 $D --all --mode action          # every episode: BIT-EXACT
python -m simple.cli.replay_holomotion_v14 $D --episode 0 --video ep0.mp4  # head | third-person video
```

Quick-started episodes replay from the actions alone. Episodes that started with the leash up also use the logged
leash force.

## 10. Troubleshooting

| symptom | check / fix |
|---|---|
| no picture in the headset | sim running and `[TCPServer] Ready` printed before you pressed Listen; app source Zedmini; same network; port 13579 free |
| two pictures, or half a picture | press B once in the app (view toggle); keep `--stream-view single` |
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
| `--quick-start` | off | after each save / retry / new setup: standing, motion tracking, recording |
| `--backpack-kg` | 0 | 3.2 = the v1.4.1 backpack policy's mass |
| `--motion-model-bundle` | public v1.4.1 | the backpack motion model folder (`model_22000`) |
| `--seed` | random | first setup's seed; the n-th setup uses seed + n |
| `--max-distractors` | 3 | distractors per setup: 0..N |
| `--no-randomize` | — | the nominal scene every time |
| `--stream-view` | single | single, mono, stereo |
| `--num-episodes` | 100 | stop after this many saved episodes |
| `--save-dir` | data/teleop_holomotion_v14 | where the datasets go |
