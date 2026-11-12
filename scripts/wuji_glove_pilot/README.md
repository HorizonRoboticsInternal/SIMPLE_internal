# Wuji Glove → BrainCo Revo 2 pilot (2026-10-27)

This pilot drives the Revo 2 hands of the HoloMotion v1.4 sim teleop with a Wuji Glove instead of the PICO grip and trigger.
Everything lives in this folder: no file in `src/` or `scripts/teleop_holomotion_v14.sh` is changed.

## How it works

The glove's `hand_skeleton` stream (21 MediaPipe keypoints, 120 Hz, Ethernet, Python SDK `wuji-sdk`) is reduced to the
Revo 2's 6 motors:

| Revo 2 motor | human signal (angles between bones, independent of the glove frame and the hand side) |
|---|---|
| thumb rotation | palmar abduction of the thumb metacarpal out of the palm plane |
| thumb bend | thumb MCP + IP flexion |
| index, middle, ring, pinky | MCP flexion (spread removed) + PIP + DIP |

- **Mapping.** Each signal maps linearly from its calibrated open / closed value onto the motor's travel. The distal joints follow as on the hand (thumb 1.0 ×, fingers 1.155 ×).
- **Pinch blend (on by default).** When the thumb tip comes within 3.5 cm of a fingertip (the nearest one), that finger keeps following the hand but closes at least to the smallest curl where the Revo 2 thumb can reach its pad, and the thumb goes to the pose that meets the finger at its curl (`PINCH_KNOTS`, one set per hand, from `pinch_poses.py --table`). A full blend is reached at 1.5 cm. The blend never opens a finger the hand has curled further. Pad-on-pad curls: index 0.69–0.98 rad (right) / 0.69–1.08 (left), middle 0.78–1.47; the ring's knots cover 0.70 to the limit, where the thumb touches the ring (tip or side) with full opposition and bend 0.9–1.0. **The thumb cannot reach the pinky in the robot model** (13 mm short at best, thumb fully opposed and bent, pinky at 0.9 rad; confirmed by the physics engine's contact list, with any thumb geom against any pinky geom — `mj_geomDistance` returns spurious 0.0 for some mesh pairs, trust settled contacts only). The pinky pad would need the thumb rotation to reach about 1.9 rad (110°) instead of the model's 1.518 (87°; BrainCo's spec says 90°); extra thumb flex does not help. The model is not above suspicion, though: its widest thumb-to-index opening is 131 mm where BrainCo quotes 100 mm, so the thumb's length / mount / rotation axis differ from the real hand. **What decides it is the real Revo 2**: command thumb rotation and flex to their limits with the pinky at ~60 % and see whether they touch, and measure the real thumb–index opening; if the real hand touches, the model's thumb rotation axis or range gets corrected in a new robot file revision (then the retarget needs no change). Until then a thumb–pinky pinch gets as close as the model allows.
- **Smoothing.** A 10 Hz low-pass filters the targets.
- **Dropouts.** On a glove dropout (no frame, or low confidence), the last target is held, so the hand never springs open.

The glove signals the Revo 2 cannot use are finger spread (it has no abduction joints), independent DIP motion and palm tactile.

## Files

| file | what |
|---|---|
| `wuji_glove.py` | Glove sources: `wuji` (the real glove), `synthetic` (a scripted hand, no hardware needed), `replay:<log>`. Also the append-only glove log. |
| `revo2_retarget.py` | Keypoints → 6 motors → 11 joints, the calibration, and the hold policy (`GloveHand`) |
| `pinch_poses.py` | Finds where the robot's thumb pad meets each finger's pad in the sim (kinematic grid, then the hand PD): one pose per finger, or with `--table` the thumb pose per finger curl → `PINCH_KNOTS` in `revo2_retarget.py` |
| `dexpilot.py` | DexPilot-style vector retargeting (after dex-retargeting / AnyDexRetarget): all six motors solved at once from thumb–finger, finger–finger and knuckle–fingertip vectors with pinch projection, 3 ms per frame over MuJoCo FK. Self-test passes on both hands; **not wired into the pipeline** (opt-in later). It cannot reach where the hand cannot either. |
| `real_revo2_pico.py` | **The real BrainCo Revo 2 hands from the PICO skeleton** (2026-11-04): the same `PicoHandSource` → `Revo2Retargeter` chain as the sim, then the six motors over Modbus RTU (`bc-stark-sdk` 2.0.9, FT4232H on `/dev/ttyUSB*`; right = ttyUSB1 slave 127, left = ttyUSB2 slave 126, both Revo2Touch fw 1.0.22.U). `status` reads state (no motion), `open` opens both hands, `run [--dry-run] [--sides] [--rate 30] [--slew 1500]` follows the operator; hold on dropout, slew-limited, hands opened on exit. SDK finger order `[thumb flex, thumb aux (rotation), index, middle, ring, pinky]`, 0 open … 1000 closed = motor / its limit. Verified so far: detection, open/read/close, and the PICO chain in `--dry-run`; **no real motion yet** (needs the user's go). |
| `pico_xrt_record.py` | `live`: real-time skeleton in a MuJoCo window (body + both hands as capsules at 60 Hz; mouse orbits/zooms, `1` body view, `2` hands view, `--record` saves the stream at exit). `record` what the XRoboToolkit SDK sends (headset pose, 24-joint SMPL body, both 26-joint OpenXR hands + quality flags, controller sample; one sample or N seconds → `xrt.npz` + `meta.json`); `skeleton` draws the body and hand trees from a frame (PNG) or a recording (MP4) and checks that the body's wrists and the hands' wrist joints agree. SDK frame is y-up (drawn z-up); while a hand is not tracked the SDK repeats one pose for all 26 joints, which the tool reports as "no hand data". `--selftest` runs on a synthetic figure. |
| `record_hand_traj.py` | Records a labelled hand trajectory from any source (both hands, a list of poses with countdowns) → `glove_log.bin` + `labels.json`, for calibration and retarget development |
| `pico_hand.py` | Source `pico`: the PICO headset's own hand tracking (XRoboToolkit SDK, 26 OpenXR joints → the 21 keypoints), no glove needed. See "Headset hand tracking instead of the glove". |
| `bench.py` | **Stage 1**: glove → retarget → the sim Revo 2 hand alone (fixed robot, same hand PD and solver as the teleop). It writes a video, metrics and self-checks. |
| `teleop_with_glove.sh` / `.py` | **Stage 2**: the unchanged teleop with the gloved hand's target replaced after the agent computes its command. The other hand keeps the PICO grip and trigger. |
| `_sim_python.sh` | Internal: the "python" that `teleop_with_glove.sh` hands to `scripts/teleop_holomotion_v14.sh` |

## Setup (once, for the real glove)

```bash
uv pip install --python .venv/bin/python wuji-sdk          # the .venv has no pip (done 2026-11-03)
# one glove per USB Ethernet adapter (the layout used since 2026-11-03): plug both in, then
scripts/wuji_glove_pilot/glove_ports.sh      # gives each port a 192.168.1.x address and pins each glove to its own port
ping 192.168.1.100 && ping 192.168.1.101     # left, right
```

Both gloves ship on the same subnet (left `WG1JA…` = 192.168.1.100, right `WG1KA…` = 192.168.1.101, /24), so with one
glove per adapter the kernel needs a host route per glove; `glove_ports.sh` finds which glove is on which port and sets
them (run it again after swapping cables; no sudo). A plain office router did **not** work here (its ports did not bridge
to the PC). The SDK finds the gloves by a network scan; the fallback is the factory address with port **50001** (Wuji's
docs say 50000, the gloves answer on 50001). `WUJI_TELEMETRY_DISABLE=1` switches the SDK's usage reporting off.

The pilot runs as the SDK's current user. Wuji recommends the default user, which uses the built-in hand model.

## Run

```bash
# 0. no glove: the scripted hand, PASS/FAIL checks for flat / fist / oppose / pinch / dropout (about 10 s)
MUJOCO_GL=egl .venv/bin/python scripts/wuji_glove_pilot/bench.py --glove synthetic

# 1. calibrate the operator: 3 poses, 2 s each -> data/wuji_glove_pilot/calib_right.json
.venv/bin/python scripts/wuji_glove_pilot/bench.py --glove wuji --side right --calibrate

# 2. bench with the glove: live viewer, or headless 60 s with video + metrics
.venv/bin/python scripts/wuji_glove_pilot/bench.py --glove wuji --side right --viewer
MUJOCO_GL=egl .venv/bin/python scripts/wuji_glove_pilot/bench.py --glove wuji --side right --seconds 60

# 3. the teleop, one glove: PICO for body and buttons, glove on the right hand
scripts/wuji_glove_pilot/teleop_with_glove.sh --glove wuji --glove-side right -- \
    --scene kitchen --robot revo2 --record --save-dir data/<run>

# 4. the teleop, both hands gloved: the headset tracks the body, the keyboard stands in for the controller buttons
scripts/wuji_glove_pilot/teleop_with_glove.sh --glove-side left,right --keys -- \
    --scene kitchen --robot revo2 --record --slice A2 --setup-ranges lv3 --lean-record \
    --save-dir data/teleop_holomotion_v14_kitchen_a2 --num-episodes 30
```

## Both hands gloved (2026-11-03)

What drives what: the gloves give the **fingers** (the Revo 2 targets of both hands, as in step 3). The **arms and
wrists** still come from the PICO headset's body tracking, exactly as without gloves. The **buttons and sticks** came from
the PICO controllers, which gloved hands can't hold, so `--keys` lets the keyboard of the terminal the teleop runs in
stand in for them (`pico_keys.py`): the keys are laid over the controller sample the agent and the CLI's combos read,
and while no controller sample arrives at all (controllers asleep), a neutral one is supplied so the quick start sees an
operator. The recorded episode is as before: the raw PICO column holds what the agent saw, keys included, so the
replays stay exact.

| key | PICO action | key | PICO action |
|---|---|---|---|
| `r` | record / save (left menu + A) | `a` | walking mode |
| `b` | abandon the episode (left menu + B) | `y` | motion tracking |
| `n` | new random setup (left menu + Y) | `s` | stand up (L3) |
| `0` | reset the same setup (left menu + both triggers) | `x` / `e` | zero torque / emergency stop |
| up / down | walk forward / back (left stick) | left / right | turn (right stick) |
| `,` / `.` | sidestep | `h` | print this list |

A tap holds its buttons for 0.3 s (the combos are edge-detected); an arrow key deflects the stick while held.

**The one thing to check before a session: PICO body tracking with the controllers put down.** The headset's body
tracking normally places the hands from the controllers. Run the probe with the headset on, controllers on the table,
gloves on, and wave the arms:

```bash
~/miniconda3/envs/holomotion_teleop/bin/python scripts/wuji_glove_pilot/pico_hands_probe.py --seconds 60
```

It prints the body-frame rate, the controllers' active flags, the headset's hand-tracking flags and how far each wrist
moved; PASS means the wrists kept moving. If the wrists freeze without the controllers, strap the controllers to the
forearms (they stay tracked and awake, the gloves stay free; the EMF modules want 3 cm from metal) and use the keys for
the buttons.

## Headset hand tracking instead of the glove (2026-11-03)

`--glove pico` (bench) / `WUJI_GLOVE=pico` (teleop) takes the fingers from the PICO headset's hand tracking instead of a
glove: the XRoboToolkit SDK gives 26 OpenXR joints per hand (`get_<side>_hand_tracking_state()`, x y z + quaternion,
metres), `pico_hand.py` keeps the 21 the retarget uses (the palm joint and the four finger metacarpals go), and everything
downstream is unchanged. What differs from the glove:

- **Needs the headset on, hand tracking enabled in the headset's XRoboToolkit app, and the controllers put down**
  (PICO switches to hand tracking when the controllers are idle). Until hand data arrives the SDK returns zeros and the
  source gives no frame: the sim hand stays open.
- **The headset drops off the PC service a few minutes after it is taken off** (the service log
  `~/.local/share/PICOBusinessSuitData/log/<date>.txt`: `reply device find "TestDevice"` on connect, `"TestDevice" disconnect`
  2–10 min after removal; six such sessions on 2026-11-03) and reconnects within seconds of being put on. While it is off
  every client gets zeros for the headset pose, the body and both hands — that is what two failed recorder runs were.
  Several clients at once are fine (a probe received both hands while a viewer was connected). Leaving the SDK open at
  exit aborts the process ("terminate called without an active exception"); the pilot's sources close it.
  The recorder's start-up wait says which case it sees; `--speak` reads the instructions aloud so the headset stays on.
- **Confidence** is the SDK's `isActive` flag (1 = high quality): 1.0, else 0.25. With the default `--min-conf 0.5` a
  low-quality frame is held, not applied; `--min-conf 0` applies every frame.
- **New frame** = the joint array changed (the SDK has no hand timestamp).
- **Own calibration**: `calib_<side>_pico.json` (the headset's hand model is not the glove's), via
  `bench.py --glove pico --side <side> --calibrate`.
- Hands in view of the headset's cameras only; occluded fingers are guessed by the headset.

```bash
.venv/bin/python scripts/wuji_glove_pilot/bench.py --glove pico --side right --viewer --min-conf 0   # first look
.venv/bin/python scripts/wuji_glove_pilot/bench.py --glove pico --side right --calibrate
.venv/bin/python scripts/wuji_glove_pilot/bench.py --glove pico --side both --viewer
scripts/wuji_glove_pilot/teleop_with_glove.sh --glove pico --glove-side left,right --keys -- --scene kitchen --robot revo2 ...
```

`pico_hand.py --selftest` checks the joint mapping, the change detection and the quality flag against a fake SDK.

**Bench outputs** go to `data/wuji_glove_pilot/bench_<time>/`:

- `bench.mp4`: palm and oblique views, the glove skeleton as an orange ghost hand, and motor bars (target and actual).
- `metrics.json`
- `glove_log.bin`: replay it with `--glove replay:<path>`.

**Teleop outputs.** The dataset is written as usual. The glove log goes to `<save-dir>/wuji_glove/`, beside the dataset folder and never inside it.

**Knobs.**

- Bench options: `--no-pinch-blend`, `--smooth-hz`, `--thumb-rot-min` (set it to 1.515 to keep the thumb opposed like the tuned open grip pose), `--min-conf`, `--robot-mjcf`.
- Viewer camera (2026-11-03): the viewer opens zoomed on the shown hand (`--view hand`, 0.30 m, `--view-distance` to change): the palm seen from 20° toward the thumb and 15° toward the fingertips, picked from 12 candidate directions rendered in the open / fist / thumb–index / thumb–middle poses of a recorded session (all five fingers and the thumb's arc visible, the curl readable, the ghost hand beside the robot hand). The look-at follows the hand every tick; the mouse still orbits and zooms. `--view palm | oblique | free`. With `--side both` the viewer shows **both hands raised in front of the chest, palms toward the viewer, fingers up** — the pose of an operator showing both palms to the headset — from in front of the robot (0.62 m), each ghost beside its hand on the outside. The arm pose was solved per arm (Powell over the 7 arm joints: palm normal = forward, fingers = up, hand base 0.32 m ahead / 0.30 m above the pelvis; mirror-symmetric, no body contact; for the left hand the across × along normal points to the *back* of the hand, so its palm sign is flipped in the objective). The bench **pins the arm joints kinematically** every substep (a soft body PD sagged 15 cm); the robot file's red CoM marker is hidden. The hands keep their full physics.
- Teleop: the same settings as the `WUJI_GLOVE_*` environment variables (see `teleop_with_glove.py`).

## Pilot results

| check | result |
|---|---|
| **real right glove** `WG1KA06260803526`, fw 0.11.1, 8 s headless bench (2026-11-03) | found by the SDK scan at 192.168.1.101:50001; 122.7 Hz, 958 frames, 0 held ticks, 0 low-confidence, retarget 0.22 ms; the sim hand on the glove's targets (glove not worn: open hand) |
| **both hands gloved + `--keys`** (synthetic publisher and gloves, headless kitchen, Revo 2, 2026-11-03) | `r` started and `r` saved a 368-frame episode from key presses alone; both hands' `cmd_position` = the glove targets on every frame; the save press is in `holomotion.pico` (bits 68 = left menu + A); physics and action replays **bit-exact** |
| **PICO hand tracking, both hands** (`--glove pico --side both --viewer`, a 197 s session on 2026-11-03, replayed through the bench) | 45–47 Hz per hand, quality flag high throughout, default calibration. A thumb touching the middle finger (2169 ticks) left the robot pads apart with the index-only blend. With `PINCH_KNOTS` (finger follows the hand, thumb meets it): robot thumb–middle pad gap **−0.1 mm median, 89 % of those ticks under 1 mm**, thumb–index 0.5 mm median (60 %); the pinched finger's target equals the hand's own curl on every one of those ticks (one pose per finger had opened it on 97–99 % of them); the nearest finger switched 10 times in 197 s. With the ring knots extended to any thumb–ring contact (same evening): the thumb touches the ring on 48 % of the thumb-on-ring ticks (median 1.0 mm, was 1 % pad-on-pad); thumb–pinky stays at 16 mm (the hand's reach limit). The bench timeline now carries both `gap_mm` (pad to pad) and `touch_mm` (any thumb–finger contact) |
| **PICO pinch recording** `data/wuji_glove_pilot/pico_rec_20261103_212800` (142 s, both hands, each thumb–fingertip pinch held ~3 s; the viewer with `--out` logs the skeleton, the motors, the reached joints and the gaps) | index, middle, ring pinches: robot pads touch (gap −0.1…1.0 mm). Pinky pinches (your thumb 5–8 mm from your pinky): robot at its limits (rotation 1.5, bend 1.0, pinky 0.9 → reached 0.85), gap 14–15 mm, every time. Geometry: the robot thumb pad's whole reachable set stays on the thumb's side of the palm midline and the pinky pad's path on the pinky's side, closest 29 mm centre to centre (pads touch at ~14); your thumb and pinky meet 20 mm on the pinky side of the midline, 50 mm above the palm — a human thumb crosses the palm, this thumb's rotation sweeps it up out of the palm instead. At 137 s PICO collapsed all fingertips onto one point for 0.7 s (a tracking glitch; the hold policy kept the last pose) |
| bench self-checks (flat, fist, oppose, pinch, dropout), synthetic glove (2026-10-27, re-run 2026-11-03 with `PINCH_KNOTS`) | all PASS |
| glove rate / retarget time | 117 Hz received (120 Hz, minus the scripted dropout) / 0.15 ms per frame |
| dropout 0.4 s | target frozen, no jump; held flag set after 0.25 s |
| teleop smoke (kitchen, Revo 2, recording, 906 frames) | recorded `action.revo2.right.cmd_position` = glove target on 906/906 frames; left hand stayed on the PICO pose |
| replay of that episode | **bit-exact** in physics mode and in `--mode action` |

**Findings.**

- **Sim hand lag.** The sim Revo 2 hand trails its target by about 160 ms (thumb rotation) and 240–260 ms (bend and fingers). This comes from the hand PD (`HAND_KP` 2–3, `HAND_KD` 0.3, joint damping 0.2–0.5); the glove path adds less than 20 ms. Grip mode hid it, but continuous glove control will feel slow. Measure the real hand's speed before changing the sim gains.
- **Pinch poses (2026-11-03).** The tuned grip pose `CLOSE2_Q` settles **5.7 mm** short of pad contact in the bench (the P-controller stops short), and it only covered the thumb–index pinch: a thumb touching the middle finger left the robot pads apart. `pinch_poses.py` searched the three motors per finger on the v3 robot file; the found poses settle at 0.0 mm (index), 0.6 mm (middle), 0.4 mm (ring) and 13 mm (pinky, the thumb's reach limit), and the two hands differ by 0.15–0.3 rad (the file's hands are not mirror images), so the tables hold one set per hand. The PICO trigger still drives `CLOSE2_Q` in the plain teleop. A single pose per finger (the first fix) dragged the pinched finger to that pose: a ring curled further than 0.8 rad was opened again, which is what "the skeleton is right but the Revo 2 is not" looked like. `pinch_poses.py --table` then searched the thumb pose per finger *curl*: a fixed thumb pose holds contact across a finger's reachable band (gap ≤ 1 mm) except for a strongly curled middle finger, which needs more thumb bend, hence the 2–4 knots per finger in `PINCH_KNOTS`. **Ring (2026-11-04):** a fine settled map showed pad-on-pad contact is *robust* only for ring curls 0.80–0.90 rad (a dozen thumb settings touch there; outside, isolated settings only — the "sometimes touches" the operator saw). During a ring pinch the ring is now kept in that band (closed to ≥ 0.80, not past 0.95; `PINCH_HI`) with a thumb pose that touches across it (right 1.50/0.875, left 1.45/0.80). On the 21:28 session the thumb is in contact with the ring in 83 % of ring-pinch ticks (86 % at full blend; pad-on-pad 64 %), index 70–78 %, middle 75 %; on the short 19:43 episodes pad-on-pad went from 1 % to 21 %. What remains is the hand's 200 ms lag on the way in.
- **Pinch blend.** Without it, a human pinch leaves the robot pads 37 mm apart (default calibration).
- **Robot file.** The numbers are the same on `g1_rot0_teleop_revo2.xml` and `_v2.xml`.

## Not yet tested / open

- **Real glove, worn.** The right glove streams and drives the bench (2026-11-03), but nobody has worn it yet: the operator calibration (step 1), the live viewer (step 2) and a glove teleop with the headset are still to do. The left glove was not on the network on 2026-11-03 (no answer at 192.168.1.100).
- **PICO body tracking without the controllers** is the open question for a two-glove session: run `pico_hands_probe.py` first (see above). Fallback: controllers strapped to the forearms.
- **Ring / pinky under PICO hand tracking.** In the recorded session, while the thumb sat on the ring fingertip, PICO reported the pinky nearly straight (feature 0.8 rad of a 3.6 fist) although a pinky usually curls with the ring — the headset cannot see a pinky hidden behind the ring, so the robot pinky stayed open. A projection-free MCP measure (angle of the proximal phalanx out of the palm plane) changed the ring / pinky features by < 0.1 rad, so the feature is not the cause. A labelled recording (`record_hand_traj.py`: isolated finger curls, every thumb–fingertip pinch, slow closes) is the way to separate tracking limits from retarget limits and to calibrate each finger's range.
- **Ergonomics.** Wuji asks for at least 3 cm clearance from metal for the EMF sensors. With one glove (step 3) the other hand's controller still gives its buttons; with two gloves use `--keys`.
- **Recording metadata.** `action.revo2.<side>.grip_mode` is the recorder's pose match against the grip poses, so a full index pinch blend used to show as 2 (pinch) while the blend targeted `CLOSE2_Q`; with the searched pinch poses the match may not trigger. The dataset itself doesn't say that a glove was used; the glove log beside it does.
- **Re-recording.** `rerecord_holomotion_v14` re-runs from the PICO input, so it would put the grip poses back on a glove episode. Use the physics or action replay instead.
