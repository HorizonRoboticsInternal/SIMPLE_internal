# Real G1 teleop + recording — runbook

Verified working 2026-08-21 on the Unitree G1 (then at 192.168.123.164 over the workstation's
`192.168.123.222/24` NIC); the robot's lab-WiFi address (192.168.50.67) works for ssh and the camera server ONLY: the
motor controller's DDS is not reachable there (verified 2026-09-16: no multicast, no unicast
discovery, no low-state), so control needs the wired direct link at 192.168.123.164. `--interface real`
now also resolves to the NIC sharing a /24 with `--robot-ip` when no 192.168.123.x NIC is up. D455 on the robot. Uses the locally built
`decoupled-wbc-deploy:rec` Docker image (Ubuntu 22.04 + Python 3.10 + ROS 2
Humble + lerobot — the host's 24.04/py3.12 cannot run this stack natively).

Simulator teleop is a separate, simpler path — see `SIM_TELEOP.md`.
For the *why* behind every flag and fix, see `REAL_ROBOT_TELEOP.md` and the
bug list at the bottom.

---

## 0. Robot side — camera server

Should already be running; if not:

```bash
ssh unitree@192.168.123.164
cd ~/SIMPLE_deploy
python -m decoupled_wbc.control.sensor.composed_camera --ego-view-camera realsense --port 5555
```

> Use the `-m` module form. Running it as a path
> (`python decoupled_wbc/control/sensor/composed_camera.py`) puts the *script's*
> directory on `sys.path` instead of the current one, so it dies with
> `ModuleNotFoundError: No module named 'decoupled_wbc'` in any shell where
> `PYTHONPATH` was not exported — i.e. every fresh SSH session.

Healthy output: `[RealSense] Intel RealSense D455 ... 640x360@30`, then
`Image sending FPS: 30.00` scrolling. Leave the terminal open.

> `realsense.py` is a file we wrote (missing from the fork AND upstream). If the
> robot copy is ever refreshed, re-copy it:
> `scp third_party/decoupled_wbc/control/sensor/realsense.py unitree@192.168.123.164:~/SIMPLE_deploy/decoupled_wbc/control/sensor/`

### Head pose (aim the ego camera)

The D455 is head-mounted, so the head angle sets what the ego view — and every
recorded frame — actually looks at. Set it before collecting: a tabletop grasp
wants `lookdown`, walking/navigation wants the default straight-ahead.

```bash
cd ~/HoloMotion/deployment/holomotion_pico_ext
./center_head.sh --pose lookdown      # go to that pose and hold
./center_head.sh --list               # list all saved poses
./center_head.sh                      # back to default straight-ahead
```

Do this **before** starting the deploy stack, and keep it fixed for a whole
dataset — changing head pose mid-collection changes the camera extrinsics and
makes episodes visually inconsistent for training.

## 1. Host — start the container

```bash
xhost +local:docker

docker run -it --rm --name g1_deploy \
  --runtime nvidia --gpus all \
  --network=host --ipc=host --privileged --device=/dev \
  -e DISPLAY=$DISPLAY -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /dev/bus/usb:/dev/bus/usb \
  -v /home/Horizon/wrk/SIMPLE/third_party:/home/Horizon/Projects/decoupled_wbc \
  -v /home/Horizon/wrk/SIMPLE/data/real_recordings:/home/Horizon/outputs \
  decoupled-wbc-deploy:rec bash
```

Use the `:rec` tag — `:latest` lacks lerobot and the data exporter will not load.

## 2. In the container — sanity checks

```bash
touch /home/Horizon/outputs/.t && rm /home/Horizon/outputs/.t && echo "outputs writable"
timeout 3 bash -c 'echo > /dev/tcp/192.168.123.164/5555' && echo "camera port open"
```

If "outputs writable" fails: fix ownership on the HOST
(`chown`, never delete the dir — recreating it breaks the live mount), then
restart the container.

## 3. In the container — deploy

```bash
cd /home/Horizon/Projects/decoupled_wbc/decoupled_wbc
python scripts/deploy_g1.py --interface real --robot-ip 192.168.123.164 \
  --body-control-device pico --hand-control-device pico \
  --upper-body-joint-speed 5 \
  --camera-host 192.168.123.164 --camera-port 5555 \
  --data-collection-frequency 30 --root-output-dir /home/Horizon/outputs

tmux attach -t g1_deployment
```

Dataset prompts: enter a task prompt, then **`y` + a name you choose**
(e.g. `g1-real-graspcan`). It creates the dataset if missing and appends on
reuse. Answering `n` hardcodes an unhelpful `{timestamp}-G1-sim` name.

Every flag above matters:

| Flag | Why |
|---|---|
| `--body/hand-control-device pico` | default `dummy` sends a 3-element navigate_cmd → `Pose dimension mismatch: got 32, expected 33` crash |
| `--data-collection-frequency 30` (was 50 until 2026-09-17) | **rows are camera-paced**: the exporter writes one row per D455 frame (blocking ZMQ recv at 30 fps), so the true row rate is ~30 Hz whatever this flag says. It only sets the dataset's `fps` label (timestamps = frame/fps, video container rate). Every dataset recorded before 2026-09-17 said 50 and was relabeled to 30 in place (`scripts/relabel_fps30.py`, backups in `data/real_recordings/_backup_fps50/` and `/mnt/nas28/alan.jiang/_backup_fps50/`). The per-episode log line prints the measured row rate and says MISMATCH if the label is wrong. |
| `--upper-body-joint-speed 5` | default **1000** rad/s vs safety limits of 6 (arm)/80 (hand) rad/s → fast snaps trigger shutdown. NOTE: stock `deploy_g1.py` silently DROPPED this flag when spawning the control loop — we patched it to forward `--upper_body_joint_speed`. Verify the spawned command line shows it. We also raised `HAND_VELOCITY_LIMIT` 50→80 in `joint_safety.py` (measured normal grasp peaks were 60–65 rad/s). |
| `--camera-host 192.168.123.164` | camera is on the robot; default `localhost` → exporter waits forever, saves nothing |
| `--data-collection-frequency 30` | default 20 Hz; 50 matches the sim datasets |
| `--home-after-save` (default ON since 2026-09-16) | **every episode starts from the same pose.** After activation and after every saved (A) or discarded (B) episode the arms glide back to the parked pose (min-jerk, `--home-duration 6`, clip `--home-max-joint-speed 0.5` rad/s) and are **held there** while your arms are ignored (a clutch). The next **A** starts recording at that exact pose and releases the robot into tracking (0.5 s blend). Approach motion becomes part of every episode; you cannot pre-position. Motivation and Psi comparison: `initial_pose_comparison.html`. |
| `--enable-gravity-compensation` (default ON since 2026-09-16, joints = arms) | **arm gravity feed-forward** (= deployment's `ARM_GRAVITY_COMP=on`). Until 2026-09-16 this flag was a no-op on the real robot: `G1Env` stored it but always sent tau = 0, so the arms sagged under gravity (elbow ~+0.09 rad, wrist ~+0.05 at rest) and the parked/init pose never matched the spec exactly. Now `G1Env.queue_action` adds pinocchio's gravity torque g(q) at the commanded pose for the selected groups (arms: 3–4 Nm at the shoulder/elbow in the canonical pose, ~6 Nm with the arms raised), clipped per joint by `--gravity-compensation-max-torque` (10 Nm). Legs/waist untouched. Recordings made with it on have no arm sag: deploy the model with `ARM_GRAVITY_COMP=on` to match. Tests: `tests/test_gravity_feedforward.py` (the G1Env test runs inside the deploy image). |
| `--init-pose` (default `elbows_raised`) | **the init / home pose** the robot boots to, parks at and every recording starts from: every upper-body joint 0 except both elbows −0.66 rad, hands open, height 0.74 (spec: `g1_initial_pose_spec.html`, JSON: `configs/init_pose_elbows_raised.json`). `--init-pose canonical` = the old pose (shoulder roll ±0.2); `--init-pose /path.json` = any pose. **Since 2026-09-16 `--home-after-save` and `--enable-gravity-compensation` are ON by default**: after every save/discard the arms ramp back to this pose with the gravity feed-forward active in every phase (`--no-home-after-save` / `--no-enable-gravity-compensation` to opt out). |
| `--home-randomize` (optional, with `--home-after-save`) | master switch to **vary the home pose each time the robot goes home** (activation, after every save/discard). On its own it changes nothing: the home pose stays canonical (arms hanging, shoulder roll ±0.2, hands open, **0.74 m**). Add `--home-random-arms` to jitter the arm joints (per-joint sigma = spread of the initial arm pose in Psi's real data on nas28 `psi_new_0828/real`, 0.02–0.06 rad, clipped to ±2.5σ and the joint limits; `--home-random-scale 0.5` halves it) and/or `--home-random-height` to draw the base height uniformly in [0.68, 0.74] m (`--home-random-height-min/-max` to change the range). `--home-random-seed N` makes the sequence reproducible, `--home-random-hands` also jitters the hands (σ 0.05). Release is anchored to the drawn pose, so no jump. Unit tests: `tests/test_home_randomize.py`. |

Expected startup: hand calibration (fingers twitch — first physical motion),
both ONNX policies load, and the data pane shows **no**
`Waiting for message ... image False`.

## 4. Operate (PICO controllers — keyboard not required)

1. Let the robot stabilize. Support it or clear space; e-stop hand ready.
2. **Left menu + left trigger** — policy action ON: legs come under RL control.
3. Drive: left joystick = walk/strafe · right joystick = turn · Y/X = height.
4. Match your arms to the robot's pose, then **left menu + right trigger** —
   teleop ON (calibrates instantly from PICO, no countdown — pose first!).
5. Record: **A** = start (`Started recording N`) · perform grasp ·
   **A** = stop+save (`Saved episode and back to idle state`) · **B** = discard.
   With `--home-after-save`: the robot parks at the canonical pose after
   activation and after every A/B; your arms are ignored while parked. Get
   comfortable, then press **A** — recording starts from the parked pose and
   the robot begins following you from that instant (no jump: your hand
   position at the press is the new origin). Second **A** saves and the arms
   glide back to park (`[teleop] parked at the canonical pose`).

tmux: `Alt+arrows` move panes · `Ctrl+b z` zoom · data pane = top right
(recording messages appear THERE; keys go to the focused pane).

## 5. Panic / verify / shutdown

```bash
# PANIC: press `  (backtick) in any pane — kills the whole session

# verify episodes are real (host, separate terminal):
watch -n2 'find /home/Horizon/wrk/SIMPLE/data/real_recordings -name "*.parquet" | wc -l'

# shutdown:
tmux kill-session -t g1_deployment && exit
```

After any `[SAFETY VIOLATION]` the shutdown **latches** — exit the container and
restart from step 1.

---

## Known failure modes (each cost a failed run)

| Symptom | Cause / fix |
|---|---|
| `Pose dimension mismatch: got 32, expected 33` | dummy streamer selected — pass `--body-control-device pico --hand-control-device pico` |
| `[SAFETY VIOLATION] ... hand ... rad/s` on every grasp | joint speed 1000 vs limit 50 — pass `--upper-body-joint-speed 5`; restart container (latched) |
| `SAFETY: Teleop mode timeout after 1.0s` then violation | teleop loop silent >1 s (crashed or not started); check teleop pane first |
| A-press records nothing, no parquet | `Waiting for message ... image False` — camera server not running/reachable on the robot |
| `No module named 'decoupled_wbc.control.sensor.realsense'` | our `realsense.py` missing from the robot copy — re-scp it |
| `No module named 'decoupled_wbc'` (camera server) | ran it as a path in a fresh shell — use the `-m` form from `~/SIMPLE_deploy` (note the near-identical message to the row above: that one is a *missing file*, this one is a *missing sys.path entry*) |
| `Unrecognized options: --egoview_camera, --host` | wrong flags (deploy_g1's own camera launch is buggy); use `--ego-view-camera realsense --port 5555` |
| `PermissionError` / `FileNotFoundError` on outputs | mount dir root-owned or recreated under a live mount — chown on host, restart container |
| `Unrecognized options: --robot-id` / `--manual-control` | those are DataExporterConfig fields, not deploy_g1 flags |
| `Required image 'ego_view_left_mono' ... not found` | exporter's `add_stereo_camera` defaults True (OAK stereo schema) — we patched deploy_g1 to pass `--no-add_stereo_camera` |
| `Pico connected (headset shows tracking) but the teleop gets nothing: no activation, no arm motion` (2026-09-16, three times) | TWO XRoboToolkit PC services on 127.0.0.1:60061 (the port is bound with SO_REUSEPORT, so a second instance starts silently): one started by hand on the host desktop or left over from a crashed teleop, one spawned by the teleop's PicoStreamer. The headset attaches to one, the teleop's client is load-balanced onto the other. Fixed 2026-09-16: `PicoStreamer.run_pico_service` reuses a running service and never starts a second one; `deploy_g1.py` refuses to start when two are listening and tells you what to kill. **Do not start the PC service app by hand**; the deploy starts (or reuses) exactly one. Check on the host: `ss -tlnp \| grep 60061` must show one line. |
| `duplicate (stale) rows: identical state+action to the previous row, in windows of seconds (up to 20 % of an episode on 2026-09-17)` | The exporter kept ONE state message in a consume-once slot, grabbed it BEFORE blocking on the next camera frame, and wrote a row per frame; whenever the subscriber's spin thread delivered late for one camera period the row repeated the previous state/action (the writer-thread change of 08-28 only reduced it). Fixed 2026-09-17: `ROSMsgSubscriber` keeps the last 50 messages with arrival times (`control/utils/state_shelf.py`), the loop waits for the frame FIRST and then takes the freshest state, waiting up to 40 ms for a new one before it would write a duplicate. The per-episode log line now reports stale rows, how often it waited, the max arrival gap vs the max publisher-stamp gap (tells a control-loop pause from a delivery hiccup) and the MEASURED row rate vs the dataset fps (rows are camera-paced, so with a 30 fps camera the true rate is ~30 Hz; if the log says MISMATCH, record with `--data-collection-frequency 30`). Tests: `tests/test_state_shelf.py`. |
| `target_yaw sweeps through ±π for ~1 s mid-turn while the stick is steady; robot wobbles the wrong way` (2026-09-17 ep 4/5/9/12) | The Pico streamer's integrated yaw dial was wrapped to [−π, π] and never reset, so whenever the accumulated dial crossed ±π during a turn it stepped by 2π; the command interpolator (rate-limited to `--upper-body-joint-speed` 5 rad/s) smeared that step into a sweep that the yaw controller followed and the recorder wrote. Fixed 2026-09-17: the dial is continuous (`integrate_yaw_dial`, no wrap), the turn flag + dial are zero-order held through the interpolator (`G1DecoupledWholeBodyPolicy`), and `G1GearWbcPolicy` absorbs any remaining dial jump > π/2 into its offset. Tests: `tests/test_target_yaw_relative.py` (dial crossing π with the robot's rate limit; old wrapping streamer). Affected episodes must be dropped (the robot really turned the wrong way). |
| `Recorded target_yaw = robot heading (−0.3 rad…) at every episode start, no stick input` (2026-09-11/14 sessions) | `G1GearWbcPolicy` computed the relative dial (`target_yaw_rel`) but `G1DecoupledWholeBodyPolicy.get_action` rebuilt its return dict without that key, so the recorder fell back to the absolute re-anchored dial. Fixed 2026-09-16: the wrapper forwards `target_yaw_rel`; the recorder helper lives in `control/utils/navigate_cmd.py`; regression test `tests/test_target_yaw_relative.py` (closed loop, turns the robot, presses A, checks 0). Sessions recorded before 2026-09-16 must have the column zeroed before use (no stick input in them). |
| `Missing features: {'observation.base_vel', 'observation.base_pose'}` | schema declares them but the frame builder never filled them (fork drift) — we patched `run_g1_data_exporter.py` to map `floating_base_pose/_vel` through |
| Joystick does nothing | policy action is OFF — left menu + left trigger (required on real robot; the "don't press it" rule was sim-only) |

## Follow-ups worth doing

- Commit ALL local patches to the `decoupled_wbc` submodule — a submodule update
  erases every one of them and re-breaks the pipeline:
  - `control/sensor/realsense.py` — NEW file (missing from fork and upstream)
  - `scripts/deploy_g1.py` — forwards `--upper_body_joint_speed`; passes
    `--no-add_stereo_camera` to the exporter
  - `control/envs/g1/utils/joint_safety.py` — HAND_VELOCITY_LIMIT 50→80
  - `control/main/teleop/run_g1_data_exporter.py` — fills
    `observation.base_pose`/`base_vel` from `floating_base_pose/_vel`
  - `control/policy/teleop_policy.py`, `run_teleop_policy_loop.py`,
    `configs/configs.py` — `home_after_save` return-to-canonical-pose ramp, `home_randomize` jitter
  - plus the patched deploy Dockerfile (still only in the session scratchpad)
- Report upstream: missing realsense module, dummy-streamer navigate_cmd
  dimension, joint-speed not forwarded + vs safety-limit default mismatch,
  deploy_g1 camera flags, exporter schema vs frame-builder drift.
- Optional: `sudo apt-get install -y libespeak1` in the image for spoken
  recording confirmations (TTS currently falls back to print).
