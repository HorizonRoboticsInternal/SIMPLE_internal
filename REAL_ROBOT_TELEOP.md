# Real-robot teleop on the Unitree G1 (decoupled WBC)

Companion docs: `SIM_TELEOP.md` (simulator teleop), `REAL_ROBOT_RUNBOOK.md` (real-robot step-by-step), `SCRIPTED_G1_NOTES.md` (the scripted `*MP-v0` pipeline), `cmd_sweep_report.html` (measured command envelope).


How the PICO controls map to the robot, what the two activation toggles actually
do, and the order to bring things up in. All of this is read from
`third_party/decoupled_wbc` — NVIDIA's GR00T decoupled-WBC stack, vendored into
SIMPLE. **SIMPLE's own CLIs never drive real hardware**; they import this package
as a library and point it at MuJoCo. Real deployment runs this stack directly.

## Launch

```bash
cd third_party/decoupled_wbc
python scripts/deploy_g1.py --interface real --robot-ip 192.168.123.164 \
  --body-control-device pico --hand-control-device pico \
  --upper-body-joint-speed 5 \
  --camera-host 192.168.123.164 --camera-port 5555 \
  --data-collection-frequency 50 --root-output-dir /home/Horizon/outputs
```

(Every one of those flags is load-bearing — see `REAL_ROBOT_RUNBOOK.md` for the
failure each one prevents. There is NO `--dataset-name`/`--task-prompt` flag on
deploy_g1: the exporter asks for the task prompt and dataset interactively —
answer `y` + a name of your choosing to control the dataset name.)

`--interface real` scans local NICs for one on the `192.168.123.x` subnet (Unitree's
robot network) and sets `env_type="real"`. Pass `--interface sim` for the dry run.
Do not pass the robot's own IP — the flag wants *your* interface, not the robot's.

`deploy_g1.py` opens a tmux session `g1_deployment` running the control loop,
teleop policy loop, camera publisher, and data exporter as separate processes that
talk over DDS/ZMQ topics.

## PICO controller map

You do **not** need a keyboard for normal operation. From
`control/teleop/streamers/pico_streamer.py`:

| Input | Function |
|---|---|
| left menu + right trigger | toggle **teleop activation** (keyboard `l`) |
| left menu + left trigger | toggle **policy action** (keyboard `]` / `o`) |
| A | start / stop+save episode (keyboard `c`) |
| B | discard episode (keyboard `x`) |
| left joystick | forward/back + strafe → `lin_vel_x`, `lin_vel_y` |
| right joystick | yaw rate, integrated at 50 Hz into an absolute `target_yaw` |
| Y / X | base height ±1 cm per call, clamped to 0.2–0.74 m |
| triggers / grips | finger closure (Dex3) |

All buttons are edge-detected (`toggle_*_last` comparisons), so holding one does
not retrigger. A PICO press is bridged into the identical keypress by the control
loop — `run_g1_control_loop.py`:

```python
if wbc_goal.get("toggle_data_collection", False):
    dispatcher.handle_key("c")
if wbc_goal.get("toggle_data_abort", False):
    dispatcher.handle_key("x")
```

Two caveats. The A/B recording buttons work out of the box — verified on
hardware; the `manual_control` config flag gates only the sync-sim path, not the
real exporter (an earlier version of this doc claimed otherwise). And the
5-second countdown before activation fires **only for keyboard** activation
(`toggle_activation_by_keyboard`) — activating from the PICO calibrates
immediately, so be in position first.

Keep a keyboard reachable regardless: **`` ` ``  is the e-stop** and kills every
process.

## The two toggles are not "teleop vs model" — they are upper body vs lower body

Neither toggle loads anything. Both policies are constructed at startup; these are
runtime enable switches for the two halves of the decoupled architecture.

### Teleop activation — does the robot follow *your arms*?

Flips `TeleopPolicy.is_active` (`control/policy/teleop_policy.py`).

* **off** (default at startup) — your hand motion is ignored; the upper body holds
  the WBC default pose.
* **on** — `teleop_streamer.calibrate()` snapshots your current pose as the origin,
  then wrist poses are retargeted through IK into `target_upper_body_pose`.

Because calibration happens *at activation*, tracking is relative displacement from
that moment. Activating while your arms are in a different posture than the robot's
causes an immediate jump. Keyboard `k` resets the policy (re-zeroes streamer + IK).

### Policy action — does the **lower-body RL policy** drive the legs?

Flips `use_policy_action` (`control/policy/g1_gear_wbc_policy.py`). This is the
branch that matters:

```python
if self.use_policy_action:
    cmd_q = self.action * action_scale + default_angles   # RL policy commands the legs
else:
    cmd_q = self.observation["q"][lower_body_indices]     # legs hold measured position
```

* **off** (`use_policy_action = False` at init) — the RL policy still runs and its
  observation tensor is still computed, but the output is **discarded**; the legs
  are commanded to hold wherever they currently are.
* **on** — the RL locomotion policy actually drives the 15 lower-body joints,
  balancing and walking per the joystick `navigate_cmd`.

It is a persistent mode flag — the code explicitly does not reset it
(`# NOTE: do not reset use_policy_action — it's a persistent mode flag`).

### Practical ordering

Enable **policy action first** (robot stands and balances under RL control), then
**teleop activation** (arms begin following you). Turning teleop on while the legs
are still holding position means you are moving the arms of a robot that is not
actively balancing.

## Bring-up order in code

1. `resolve_interface()` → `env_type="real"`, `G1Env.use_sim = False`.
2. `G1Env.__init__` calls **`calibrate_hands()`** — left then right Dex3 finger
   zeroing. This is the first physical motion, before anything is armed.
3. `run_g1_control_loop.py` starts the robot-config server and publishers for
   `STATE_TOPIC_NAME` (feeds the exporter), `LOWER_BODY_POLICY_STATUS_TOPIC`, and
   `JOINT_SAFETY_STATUS_TOPIC`; builds the robot model, `G1Env`, and the WBC policy;
   registers `KeyboardEStop` **before** the loop spins.
4. Loop at `control_frequency` (50 Hz): `env.observe()` → `wbc_policy.set_observation()`
   → read goal from `CONTROL_GOAL_TOPIC` → joint commands.
5. `run_teleop_policy_loop.py` is a *separate* process owning the headset; it
   publishes to `CONTROL_GOAL_TOPIC`. A stalled teleop process therefore leaves the
   WBC still balancing the robot.

## How arm following works

Not joint copying — your limb lengths differ from the robot's.

XRoboToolkit provides 4×4 wrist transforms → calibration establishes the origin →
`TeleopRetargetingIK` solves a `BodyIKSolver` over
`ReducedRobotModel.from_active_groups(robot_model, ["upper_body"])`, so IK can never
solve by moving the legs → `target_upper_body_pose` is published. Fingers use
separate per-hand solvers retargeting fingertips onto the 7-DoF Dex3.

Your **legs are not followed**: locomotion comes from the joystick nav command. The
ankle trackers contribute body pose for retargeting, not step-for-step walking.
That separation is the point — arm motion never perturbs the balance policy.

## Recording

`run_g1_data_exporter.py` subscribes to `STATE_TOPIC_NAME`. Episode states cycle
IDLE → RECORDING → NEED_TO_SAVE → IDLE on `c` (or A); `x` (or B) discards. It waits
for both proprio and image messages before writing frames. Output goes to
`outputs/<dataset_name>/`; reusing a name appends.

Saved fields (`data/utils.py::get_dataset_features`):

| Field | Shape |
|---|---|
| `observation.images.ego_view` | RealSense H×W×3 video |
| `observation.state` / `action` | `num_joints` (`state_dim`/`action_dim` default 43) |
| `observation.eef_state` / `action.eef` | 14 — left/right wrist pos + abs quat |
| `observation.img_state_delta` | float32 — camera vs proprio clock skew |
| `teleop.navigate_command` | 4 |
| `teleop.base_height_command` | 1 |
| `observation.base_pose` / `base_vel` | 7 / 6 — filled from `floating_base_pose/_vel` (our patch; stock exporter declared but never wrote them) |

Default `data_collection_frequency` is **20 Hz** — SIMPLE's sim teleop records at
50 Hz, and the schemas differ (sim post-processes to 36-D action / 32-D state with
`hand_joints`/`arm_joints`/`leg_joints` split out). Converting is required before
mixing real and sim data.

## Safety

`deploy_g1.py` shows a checklist and requires confirmation. Verbatim highlights:
sim2sim first with `--interface sim`; validate state reading with the action queue
disabled; **low-gain test with kp 2–5× lower than normal, kd unchanged**; clear
workspace and avoid tables; have a keyboard e-stop, Joycon, or power cutoff within
reach. `` ` `` stops everything, `Ctrl+C` one process, `Ctrl+\` quits tmux.

> Written from source only, with no G1 available to verify against. The vendored
> `decoupled_wbc` may lag upstream GR00T — diff it before trusting gain or safety
> defaults on hardware.
