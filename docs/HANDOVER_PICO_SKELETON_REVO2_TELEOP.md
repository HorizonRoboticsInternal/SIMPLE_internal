# Handover: PICO hand skeleton → Revo 2 fingers in the HoloMotion v1.4 teleop (as an option)

Written 2026-11-03 for the agent who integrates this into the main teleop. Everything below exists and is verified as a
**pilot that wraps the teleop without editing `src/`** (`scripts/wuji_glove_pilot/`). The job is to make it a first-class
option of the teleop (`scripts/teleop_holomotion_v14.sh`), off by default, with recordings that replay bit-exact.

## 1. What it does today

The PICO headset tracks the operator's hands (26 OpenXR joints per hand through the XRoboToolkit PC service). The pilot
turns each hand into the 6 Revo 2 motor targets every 20 ms and substitutes them for the teleop's grip-button hand poses.
The arms, wrists, body and all buttons stay exactly as in the teleop.

```
XRoboToolkit SDK ──► pico_hand.PicoHandSource ──► 21 MediaPipe keypoints ──► revo2_retarget.Revo2Retargeter ──► 11 joint targets
   (26 joints x (xyz+quat))     26 -> 21 map            (metres, hand frame)        bone angles -> calibration -> 6 motors        cmd.parameters["<side>_hand_q"]
                                                                                    + PINCH_KNOTS (thumb meets the pinched finger)
                                                                                    + 10 Hz low-pass + hold on dropout
```

Run it (both hands, keyboard standing in for the controller buttons, since hand tracking needs the controllers put down):

```bash
scripts/wuji_glove_pilot/teleop_with_glove.sh --glove pico --glove-side left,right --keys -- \
    --scene kitchen --robot revo2 --record --slice A2 --setup-ranges lv3 --lean-record \
    --save-dir data/teleop_holomotion_v14_kitchen_a2 --num-episodes 30
```

Viewer to check the hands alone (no teleop): `.venv/bin/python scripts/wuji_glove_pilot/bench.py --glove pico --side both --viewer --min-conf 0`

## 2. The files

| file (all in `scripts/wuji_glove_pilot/`) | role | keep / move |
|---|---|---|
| `pico_hand.py` | `PicoHandSource(side)`: `latest(t) -> GloveFrame | None`. SDK init once per process (refcount), **must be closed at exit or the process aborts** ("terminate called without an active exception"). Confidence = SDK `isActive` (1 → 1.0, else 0.25). New frame = the 26×7 array changed (no hand timestamp in the SDK). `XR_TO_MP = [1,2,3,4,5,7,8,9,10,12,13,14,15,17,18,19,20,22,23,24,25]`. | move into `src/simple/teleop/holomotion_v14/` |
| `revo2_retarget.py` | `features(kp)` (6 bone-angle scalars, frame- and side-independent), `Calibration` (per-operator open/closed JSON), `Revo2Retargeter(calib, side=...)` → `step(kp, t)` → dict with `q` (11 joints), `motors`, `pinch_w`, `pinch_finger`; `PINCH_KNOTS` per side; `GloveHand` = source + retargeter + hold policy (`min_conf`, `stale_s` 0.25). | move |
| `teleop_with_glove.py` | the hook: wraps `HoloMotionV14Agent.get_action` and overwrites `cmd.parameters[f"{side}_hand_q"]` with the retargeted `q`; wraps `HoloMotionV14Agent.pico` with the keyboard stand-in. Env knobs `WUJI_GLOVE=pico`, `WUJI_GLOVE_SIDES`, `WUJI_GLOVE_KEYS`, `WUJI_GLOVE_MIN_CONF`, `WUJI_GLOVE_SMOOTH_HZ`, `WUJI_GLOVE_PINCH_BLEND`, `WUJI_GLOVE_CALIB_DIR`. | replace by a real option (section 4) |
| `pico_keys.py` | keyboard → controller sample (`r` record/save = left menu + A, `b` abandon, `n` new setup, `0` reset, `a`/`y`/`s`/`x`/`e` modes, arrows = sticks). Needed because PICO switches to hand tracking only while the controllers are idle. | keep as the button source for hand-tracking sessions, or strap the controllers to the forearms |
| `bench.py` | hands-only sim (fixed robot, the teleop's hand PD + solver), viewer, headless video + metrics + `timeline.npz`, `--calibrate`, synthetic self-checks (`--glove synthetic`) | keep as the test rig |
| `record_hand_traj.py` | labelled pose recording (28 poses × 4 s, `--speak` reads prompts aloud so the headset stays on) → `glove_log.bin` + `labels.json` | keep |
| `pinch_poses.py` | sim search for where the thumb pad meets each finger pad (`--table` per finger curl) → `PINCH_KNOTS` | keep |
| `dexpilot.py` | DexPilot-style 6-motor vector solver (dex-retargeting recipe), 3 ms/frame, self-test passes; **not wired in** | optional alternative (`method=` switch) |
| `wuji_glove.py`, `glove_ports.sh` | the Wuji glove source and its network setup — **no longer the target**; `GloveFrame`, `GloveLog`, `make_source`, the synthetic hand live here | keep `GloveFrame`/`GloveLog`/synthetic; the glove parts can go |
| `README.md` | everything measured, in detail | read it once |

Data: `data/wuji_glove_pilot/` — `calib_<side>.json` (Wuji), `calib_<side>_pico.json` (PICO; **none recorded yet, the built-in defaults are in use**), the recorded sessions `bench_20261103_194338` (197 s, both hands, pinches) and `pico_rec_20261103_212800` (142 s, every thumb–fingertip pinch, with `timeline.npz`), replayable with `bench.py --glove replay:<dir>/glove_log.bin`.

## 3. Where the teleop takes the hand targets

- `src/simple/agents/holomotion_v14_agent.py`: `_hand_target(side, grip, trigger)` (line ~208) maps the PICO grip/trigger to `OPEN_Q` / grip poses / `CLOSE2_Q`; `get_action` (line ~390) puts `left_hand_q` / `right_hand_q` into `cmd.parameters`; `pico()` (line ~345) returns the newest controller sample if younger than `PICO_TIMEOUT_SEC` 0.5 s, else `None` — the quick start waits for a sample ("operator"), which is why the keyboard stand-in serves a neutral sample while the controllers sleep.
- `src/simple/cli/teleop_holomotion_v14.py`: `SimControls.poll()` (line ~130) reads `agent.pico()` for the record / save / abandon / reset combos; `GripLatch` (line ~940, `--slice` with `plate_in_hand`) releases once the hand target differs from `OPEN_Q` — the skeleton target differs on the first tick, so it is no blocker.
- Wire: `src/simple/teleop/holomotion_v14/wire.py` — the publisher (`publisher.py`, runs in the `holomotion_teleop` conda env, reads the headset) sends `obs65` (the 65-D reference) and `pico_ctrl` (`pico_payload`: timestamp, button bits, sticks, triggers, grips) on one ZMQ PUB socket; `ReferenceSubscriber` in the sim process keeps `last_pico`.
- Recording: `src/simple/teleop/holomotion_v14/revo2_record.py` writes `action.revo2.<side>.cmd_position` (11), `motor_target` (6) and `grip_mode` (a pose match: 0 open / 1 closed / 2 pinch — with skeleton targets it rarely matches); the CLI writes `holomotion.pico` = the raw controller sample (`RAW_PICO_NAMES`). Replays: `python -m simple.cli.replay_holomotion_v14 <dataset> --all --mode physics|action`; re-recording (`rerecord_holomotion_v14`) re-runs the agent from the recorded PICO input.

## 4. The integration (recommended shape)

1. **Option, off by default.** A CLI flag / config on the teleop, e.g. `--hand-source {grip,skeleton}` (default `grip`), with `--hand-source-sides left,right`, the retarget settings (`min_conf`, `smooth_hz`, `pinch_blend`, calibration dir) and the operator calibration file per side.
2. **Carry the skeleton on the wire, not read it in the sim process.** The pilot reads the SDK directly in the sim process (works: `xrobotoolkit_sdk` imports in `.venv`, `third_party/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64/`). But the recorded raw input (`holomotion.pico`) is what `rerecord_holomotion_v14` replays from, so: add the two 26×7 hand arrays + the two `isActive` flags to `pico_payload` / `pico_sample` (a new topic or the same payload; ~370 floats at the publisher's rate), have the publisher read `xrt.get_{left,right}_hand_tracking_state()` + `get_{left,right}_hand_is_active()` next to `read_pico_control_sample`, and record them (a new raw column, e.g. `holomotion.pico_hands`, float32 (2, 26, 7) + (2,) flags). Then the agent retargets in `get_action` from the sample, exactly as the pilot's `install()` does after the grip map, and a re-record reproduces the hand targets.
3. **Keep `cmd.parameters["<side>_hand_q"]` as the only output.** Nothing downstream changes: the hand PD, the recorder columns, `GripLatch`, replays. `grip_mode` can stay (it is a pose match); add `action.revo2.<side>.pinch_finger`/`pinch_w` only if useful.
4. **Hold policy.** No frame or `isActive` 0 → keep the last target (never spring open; a held object would drop). The pilot's `GloveHand` does this (`stale_s` 0.25).
5. **Buttons while hand tracking.** PICO tracks hands only while the controllers are idle. Either keep `pico_keys.py` (laid over the controller sample inside `pico()`, so `SimControls` and the agent see the same thing) or strap the controllers to the forearms. Whatever is chosen must still write the pressed bits into `holomotion.pico` so replays see the save/abandon presses (the pilot's keys do: verified bits 68 = left menu + A in the recording).
6. **Calibration.** `bench.py --glove pico --side <side> --calibrate` (three poses) → `calib_<side>_pico.json`; or derive open/closed from a `record_hand_traj.py` session. The built-in defaults already work (all pinches in the recordings closed), calibration tightens the finger ranges per operator.

## 5. What is verified

- Headless end-to-end with synthetic publisher + synthetic hands + keys (2026-11-03): `r` started and `r` saved a 368-frame kitchen / Revo 2 episode; both hands' `action.revo2.<side>.cmd_position` equal the retargeted targets on every frame; the key press is in `holomotion.pico`; **physics and action replays bit-exact**.
- PICO stream: 45–47 Hz per hand, `isActive` 1 throughout two sessions; retarget 0.2 ms per frame; the sim hand lags its target 160–260 ms (hand PD, same as with the grip poses).
- Pinches on the operator's recordings: index, middle, ring → robot pads touch (gap ≤ 1 mm). The pinched finger follows the operator's own curl (never opened by the blend), the thumb goes to meet it (`PINCH_KNOTS`).
- `bench.py --glove synthetic`: flat / fist / oppose / pinch / dropout self-checks all PASS.

## 6. Known limits and traps (read before testing)

- **Thumb–pinky pinch cannot close in the robot model**: the thumb pad's whole reachable set stays on the thumb's side of the palm midline, the pinky pad stays on the pinky's side (29 mm centre-to-centre at best; pads touch at ~14); the model's joint ranges equal BrainCo's spec (thumb abduction 0–90°, flex 0–59°). The robot takes the closest pose (thumb fully opposed and bent, pinky ~0.9 rad, 13 mm gap). The model opens thumb-to-index 131 mm where BrainCo quotes 100 mm, so the thumb geometry is suspect — **the physical Revo 2 decides** (not yet tested; the hands were not detected over RS485 on 2026-11-03). If the real thumb reaches, fix the robot file's thumb axis/range; the retarget needs no change.
- **The headset drops off the PC service 2–10 min after it is taken off** and reconnects seconds after it is put on; while off, every client gets zeros (headset pose, body, hands). Service log: `~/.local/share/PICOBusinessSuitData/log/<date>.txt` (`reply device find "TestDevice"` / `"TestDevice" disconnect`). Several SDK clients at once are fine.
- **Close the SDK on every exit path** (`xrt.close()`), including `sys.exit` and Ctrl-C, or the process aborts.
- PICO occlusion: with the thumb on the ring finger it reported the pinky nearly straight (hidden behind the ring); and once (142 s session, t = 137 s) it collapsed all fingertips onto one point for 0.7 s — the hold policy covers it, a min-confidence from `isActive` does not (the flag stayed 1).
- `mj_geomDistance` on mesh pairs can return 0.0 with no contact — use the engine's contact list or PD-settled poses when measuring the hand.
- The two hands of `g1_rot0_teleop_revo2_v3.xml` are not mirror images (pinch poses differ by 0.15–0.3 rad): keep per-side tables.
- The bench's arm pose is pinned kinematically (a soft body PD sagged 15 cm); the teleop's arms are untouched.

## 7. Open items

1. Record the operator calibration (`record_hand_traj.py --glove pico --speak`, headset on throughout) and write `calib_<side>_pico.json`.
2. Test thumb–pinky and the thumb–index opening on the physical Revo 2; revise the robot model if it reaches. (2026-11-04: both hands answer over the FT4232H — right `/dev/ttyUSB1` slave 127, left `/dev/ttyUSB2` slave 126, Revo2Touch fw 1.0.22.U — and `scripts/wuji_glove_pilot/real_revo2_pico.py` drives them from the PICO skeleton with the sim's retarget: `status` / `open` / `run [--dry-run]`; detection, read and the dry run verified, no real motion yet.)
3. Decide buttons for hand-tracking sessions (keys vs controllers on the forearms); the PICO body tracking of the wrists with the controllers put down is also unverified (`pico_hands_probe.py`, headset on).
4. Optional: `dexpilot.py` as `method="dexpilot"` (all six motors solved from fingertip vectors; a flat hand gives thumb bend ~0.7 because the robot's straight thumb points sideways).

## 8. Status 2026-11-03 night: integrated

Done in `src/` as recommended in section 4 (nothing in the pilot was changed; the bench stays the test rig):

- `--hand-source skeleton` on `simple.cli.teleop_holomotion_v14` (default `grip`), with `--hand-sides`, `--hand-calib-dir`,
  `--hand-min-conf` (0), `--hand-smooth-hz` (10). The retarget lives in `src/simple/teleop/holomotion_v14/hand_skeleton.py`
  (`features`, `Calibration`, `Revo2Retargeter`, `PINCH_KNOTS`, `SkeletonHand` = source-less hold policy on (26 x 7, isActive)
  samples); the keyboard stand-in in `src/simple/teleop/holomotion_v14/pico_keys.py` (+ SPACE / p events, programmatic presses).
- The skeleton rides the wire: `publisher.py --hands` publishes both 26 x 7 arrays + isActive flags (`wire.HANDS_TOPIC`,
  `send_pico_hands`, `ReferenceSubscriber.last_hands` / `hands_age`); `scripts/teleop_holomotion_v14.sh` adds `--hands` when the
  CLI gets `--hand-source skeleton`; the synthetic source makes synthetic hands from the pilot's scripted human hand
  (`hand_synth.py`). The agent retargets in `get_action` from `sub.last_hands` and overwrites `<side>_hand_q`; `pico()` serves a
  neutral sample while the controllers sleep and hand frames arrive, and applies `pico_overlay` (the keys).
- Recorded: `holomotion.pico_hands` (float32 366: both arrays flat + the two flags, NaN before the first sample); the exact log's
  info keeps `hand_source` and `hand_skeleton` (sides, min_conf, smooth_hz, pinch_blend, the calibration arrays per side).
  `rerecord_holomotion_v14` rebuilds the agent with that and feeds the hands per tick (`_ReplaySubscriber.set_tick(..., hands)`).
- `--auto-cycle` (user: hands-free sessions): SPACE arms, a countdown on the headset view, the motion-tracking button pressed for
  the operator, `--slice-auto-cut` forced, next countdown when the robot stands again (`auto_cycle.py`).
- Open items 1-4 of section 7 stand (operator calibration, the physical thumb, the arms' body tracking with the controllers down).
