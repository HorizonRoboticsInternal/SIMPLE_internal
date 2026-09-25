# HoloMotion Teleop in SIMPLE

Teleoperate G1 SIMPLE tasks with the **HoloMotion v1.4 motion-tracking
policy** instead of the decoupled WBC, and record LeRobot-format episodes.

```text
PICO / XRoboToolkit
  -> reference publisher          third_party/holomotion (HoloRetarget, vendored)
  -> ZMQ "reference_qpos"         reference_qpos[36] + hand_joints[14] + buttons
  -> HoloMotionPicoAgent          11-frame reference queue -> 604-D observation
  -> HoloMotion ONNX              KV-cache transformer -> 29 joint targets
  -> ActionCmd("holomotion")      G1Sonic PD torques (policy gains) + Dex3 hand PD
```

Everything needed lives inside this repository — no HoloMotion checkout:

```text
src/simple/teleop/holomotion/     wire format + reference/Dex3/button contracts
src/simple/agents/holomotion_pico_agent.py   reference queue, observation, ONNX
src/simple/cli/teleop_holomotion.py          teleop + LeRobot recording  (teleop-holomotion)
src/simple/cli/holomotion_replay.py          headset-free stream replay  (holomotion-replay)
data/holomotion/models/*.onnx                motion-tracking policy + its train config
data/holomotion/samples/*.npz                recorded reference stream for replay
third_party/holomotion/                      vendored publisher (HoloRetarget + PICO node)
```

The agent is a NumPy port of the observation and inference protocol used by the
on-robot Orin deployment (`actor_observation.py`, `policy_runtime.py`), verified
against the canonical Warp kernel to fp32 rounding. The SIMPLE side needs only
`onnxruntime` and `pyzmq`.

> `data/` is gitignored, so the ONNX and sample recording are present in this
> working copy but do not travel through git. After a fresh clone, copy them in
> or pass `--onnx-path`.

## Task / environment

| Gym id | Task uid | Controller |
|---|---|---|
| `simple/G1WholebodyLocomotionPickBetweenTablesHoloMotionTeleop-v0` | `g1_wholebody_locomotion_pick_between_tables_holomotion_teleop` | HoloMotion |
| `simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0` | `g1_wholebody_locomotion_pick_between_tables_teleop` | decoupled WBC |

The HoloMotion task subclasses the decoupled-WBC teleop task: same scene
(`hssd:scene6`), objects, domain randomization and success criteria; only the
controller differs. The controller is selected by the CLI/agent, so any other
`g1_sonic` task can be driven the same way by passing its gym id.

## Requirements

* **SIMPLE side** (agent, inference, recording): the SIMPLE venv (`uv sync`).
  `onnxruntime` on CPU is enough — inference is ~0.3 ms/step.
* **Publisher side** (only for live PICO teleop): a separate environment with
  Newton/Warp + CUDA and the XRoboToolkit SDK. See
  [`third_party/holomotion/README.md`](../third_party/holomotion/README.md).
  Not needed for replay-driven runs.

## Run

Terminal 1 — reference stream. With a PICO headset:

```bash
third_party/holomotion/run_publisher.sh            # bind tcp://*:6001 at 50 Hz
```

or, with no headset, replay a recording (pure SIMPLE venv):

```bash
holomotion-replay --hand-demo                      # bundled sample, looping
holomotion-replay path/to/recording.npz            # any --save-reference-path dump
```

Terminal 2 — SIMPLE:

```bash
teleop-holomotion                                  # defaults to the PickBetweenTables task
teleop-holomotion simple/G1WholebodyLocomotionPickBetweenTablesHoloMotionTeleop-v0 \
    --record --save-dir data/teleop_holomotion --num-episodes 20
```

The ONNX is picked up from `data/holomotion/models` automatically; override
with `--onnx-path` or `HOLOMOTION_MOTION_ONNX`. Other options: `--reference-uri`
(default `tcp://127.0.0.1:6001`), `--headless`, `--no-use-gpu`, `--dr-level`,
`--no-pico-stream` (disable the ego-camera feed to the headset on port 13579).

To watch the reference stream itself in MuJoCo:

```bash
python third_party/holomotion/deployment/holomotion_teleop/holomotion_teleop_mjviewer.py \
    --uri tcp://127.0.0.1:6001 --model g1_comp        # needs the publisher env
```

### PICO controls

```text
right axis click           drop robot / release the elastic band
left grip + right grip     reset environment
left / right trigger       close Dex3 hands (mapped by the publisher)
left menu + A              start recording, or save the episode being recorded
left menu + X              abandon the episode being recorded
```

### Episode flow

1. After reset the robot PD-holds the policy's default pose. Without
   `--record` it hangs on the elastic band until you click the right stick;
   with `--record` the band is skipped.
2. Once the 11-frame reference window is filled (~0.25 s) and the robot has
   settled, motion tracking starts: the reference yaw is aligned to the
   robot's heading at that moment (same as motion-mode entry on the real G1)
   and the KV cache is reset. Stand still, facing forward, when tracking
   engages.
3. With `--record`, an episode starts automatically when tracking is active
   and ends on task success, `left menu + A`, or Ctrl-C (saved). Resets
   discard the in-progress episode.

The robot tracks the operator with a fixed ~200 ms lag (the 10-frame future
horizon the policy was trained with).

## Dataset schema

LeRobot format via `Gr00tDataExporter` (50 fps). The decoupled-WBC keys are
kept for tooling compatibility; decoupled-only commands are zero-filled.

| Key | Shape | Notes |
|---|---|---|
| `observation.images.ego_view` | video | head stereo left |
| `observation.state` | (43,) | body 29 + left hand 7 + right hand 7 (`WHOLE_BODY_JOINTS` order) |
| `action` | (43,) | policy joint targets (29) + Dex3 targets (14) |
| `observation.base_pose` / `observation.base_vel` | (7,) / (6,) | floating base |
| `observation.object_poses` | (7·N,) | pos + quat per scene object |
| `teleop.reference_qpos` | (36,) | reference frame the policy tracked this step |
| `teleop.reference_window` | (11·36,) | current + 10 future reference frames |
| `teleop.hand_joints` | (14,) | published Dex3 command (thumb/index/middle order) |
| `policy.raw_action` | (29,) | raw ONNX output (ONNX joint order in `names`) |
| `policy.obs` | (604,) | exact policy observation (replay / determinism checks) |
| `observation.eef_state`, `action.eef`, `teleop.navigate_command`, `teleop.base_height_command`, `observation.torso_rpy_command` | — | zero-filled (decoupled-WBC only) |

`teleop.reference_window` lets the 11-frame × 79-D "HoloMotion action"
used by the Isaac Lab pipeline be reconstructed offline
(`dof_pos, dof_vel, root_pos, root_rot, hand_joints` per frame).

## Implementation notes

- `G1Sonic.apply_action` gained a `"holomotion"` command: PD with per-joint
  `kp`/`kd` from the ONNX metadata (MJCF order), hand PD shared with the
  decoupled path, optional elastic-band force while hanging.
- Joint orders: HoloRetarget's 29-dof order equals the g1_sonic MJCF body
  order; the ONNX uses Isaac Lab breadth-first order (mapped via metadata
  `joint_names`); Dex3 fingers are thumb/index/middle in the stream and
  thumb/middle/index in the MJCF (`DEX3_NATURAL_TO_MJCF`).
- Control runs at 50 Hz (physics 200 Hz, decimation 4), matching the
  policy's training/deployment clock.
