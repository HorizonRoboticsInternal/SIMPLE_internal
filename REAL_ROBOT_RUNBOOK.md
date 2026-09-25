# Real G1 teleop + recording — runbook

Verified working 2026-08-21 on the Unitree G1 at `192.168.123.164` (workstation
NIC `192.168.123.222/24`, D455 on the robot). Uses the locally built
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
  --data-collection-frequency 50 --root-output-dir /home/Horizon/outputs

tmux attach -t g1_deployment
```

Dataset prompts: enter a task prompt, then **`y` + a name you choose**
(e.g. `g1-real-graspcan`). It creates the dataset if missing and appends on
reuse. Answering `n` hardcodes an unhelpful `{timestamp}-G1-sim` name.

Every flag above matters:

| Flag | Why |
|---|---|
| `--body/hand-control-device pico` | default `dummy` sends a 3-element navigate_cmd → `Pose dimension mismatch: got 32, expected 33` crash |
| `--upper-body-joint-speed 5` | default **1000** rad/s vs safety limits of 6 (arm)/80 (hand) rad/s → fast snaps trigger shutdown. NOTE: stock `deploy_g1.py` silently DROPPED this flag when spawning the control loop — we patched it to forward `--upper_body_joint_speed`. Verify the spawned command line shows it. We also raised `HAND_VELOCITY_LIMIT` 50→80 in `joint_safety.py` (measured normal grasp peaks were 60–65 rad/s). |
| `--camera-host 192.168.123.164` | camera is on the robot; default `localhost` → exporter waits forever, saves nothing |
| `--data-collection-frequency 50` | default 20 Hz; 50 matches the sim datasets |

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
  - plus the patched deploy Dockerfile (still only in the session scratchpad)
- Report upstream: missing realsense module, dummy-streamer navigate_cmd
  dimension, joint-speed not forwarded + vs safety-limit default mismatch,
  deploy_g1 camera flags, exporter schema vs frame-builder drift.
- Optional: `sudo apt-get install -y libespeak1` in the image for spoken
  recording confirmations (TTS currently falls back to print).
