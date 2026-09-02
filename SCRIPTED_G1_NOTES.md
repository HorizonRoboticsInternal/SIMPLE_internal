# Scripted G1 data generation in SIMPLE — orientation notes

Written for another Claude picking this repo up cold. Covers what the scripted
(motion-planning) pipeline is, how it is wired, what it can produce, and the
four traps that will otherwise cost you hours.

Repo: `physical-superintelligence-lab/SIMPLE` (SIMulation-based Policy Learning
and Evaluation). Paper: arXiv 2606.08278.

---

## 1. The word "scripted" does not appear in the code

The paper calls it "scripted policies"; the code calls it motion planning. The
mapping you need:

| Paper term | Env IDs | CLI | Agent |
|---|---|---|---|
| Automated Motion Planning ("scripted") | `*MP-v0` | `simple.cli.datagen` | `MotionPlannerAgent` |
| Low-Latency VR Teleoperation | `*Teleop-v0` | `simple.cli.teleop_decoupled_wbc` | `PicoDecoupledAgent` |

There is **no real-world data collection code here** — the repo is
simulation-only. The paper's real-robot numbers come from teleoperating a
physical G1 elsewhere. The `unitree_sdk2py` DDS channels in
`src/simple/envs/sonic_loco_manip.py` speak the real robot's protocol but are
bridged into MuJoCo.

## 2. Controllers — three of them, do not mix them up

- **AMO** (`src/simple/robots/policy/AMO_Policy.py`, `amo_jit.pt`) — used by the
  **scripted** G1 tasks. RL policy driving 15 lower-body joints (hips, knees,
  ankles, waist); arms follow CuRobo joint targets directly. Reached via
  `ActionCmd("loco_command")` on the `G1Wholebody` robot.
- **decoupled WBC** (`third_party/decoupled_wbc`, config `g1_29dof_gear_wbc.yaml`)
  — used by the **teleop** tasks via `G1Sonic`, `ActionCmd("decoupled_wbc")`,
  PD position control at 50 Hz control / 200 Hz sim.
- **SONIC** (`ActionCmd("wbc_torque")`) — present but effectively unused. The
  README still lists "Integrate SONIC whole-body controller" as an open TODO,
  and the code explicitly bypasses it (`sonic_config` VERSION `sonic_model12` is
  incompatible with `get_wbc_policy`).

`G1Wholebody`/`G1`/`G1Inspire*` contain **zero** references to `sonic` or
`decoupled_wbc`. Confirm with grep before believing anything else.

Careful: the paper's appendix calls AMO "a decoupled architecture", so
"decoupled" describes both AMO and the `decoupled_wbc` package. They are
different code.

## 3. How one scripted episode is produced

```
task.decompose()            -> list[SubtaskSpec]      # the "script"
MotionPlannerAgent.synthesize()
    GraspObjectSpec  -> BoDex/GraspNet cached grasp -> CuRobo plan
    Walk/Turn/Stand  -> loco_command (AMO)
    Lift/Lower/...   -> more CuRobo plans
  -> flattens everything into a deque of ActionCmd
env.step(agent.get_action(...)) until terminated/truncated
LerobotRecorder.step() -> saves ONLY if reward > 0.9
```

Key files: `src/simple/agents/mp.py` (the planner agent, ~600 lines, the heart),
`src/simple/datagen/subtask_spec.py` (the primitive vocabulary),
`src/simple/mp/curobo.py` (CuRobo wrapper), `src/simple/envs/lerobot.py` (recorder).

Primitive vocabulary: `OpenGripperSpec`, `CloseGripperSpec`, `GraspObjectSpec`,
`LiftSpec`, `LowerSpec`, `RetreatSpec`, `MoveEEFToPoseSpec`, `PhaseBreakSpec`,
and for humanoids `StandSpec`, `WalkSpec`, `TurnSpec`, `HeightAdjustSpec`.

A minimal `decompose()` (`franka_tabletop_grasp_mp.py`) is four specs: open,
approach, close, lift. A G1 one uses `grasp_type="bodex"`, `hand_uid="dex3_right"`
and `lock_links` to freeze the waist/other arm during planning.

Grasps are **offline**: objects are dropped in MuJoCo to enumerate stable poses,
BoDex synthesizes dexterous grasps per pose, all cached under
`data/assets/graspnet/dex_grasp/<hand>/<obj>/`. Nothing is synthesized at runtime.

**`--num-episodes` counts successes, not attempts.** `LerobotRecorder.step` calls
`save_episode()` only when `reward > 0.9` and deletes the frame buffer otherwise.

## 4. Capability surface

72 registered `*MP-v0` tasks = 40 base + 32 `Variant*` re-parameterizations.

By robot: **G1Wholebody 59**, G1 3, G1Inspire 3, Aloha 3, Vega 2, Franka 2.
By hand: **60 use the Dex3 hand**; the other 11 are parallel-gripper or
grasp-free (Franka, Aloha, Vega, the 3 G1Inspire, `G1WholebodyLocomotionMP`,
`G1WholebodySitMP`).

Capability families in the G1 whole-body suite (tags overlap):
locomotion-and-pick-between-tables (14 variants, the largest family), `XMove`
(17), `Turn` (12), `Bend` (15), `Handover` (12), pick-and-place (11), tabletop
grasp (11), `Sit` (1). Composites exist, e.g.
`G1WholebodyTurnXMoveAndBendHandoverMP-v0`.

Output is LeRobot v2.1: `data/`(parquet) + `videos/chunk-000/<camera>/episode_*.mp4`
+ `meta/info.json`. G1 records **7 cameras** (head stereo L/R, front stereo L/R,
side, wrist, wrist_left); arm robots record 5 (no head stereo). `images/` is
always empty — LeRobot deletes PNGs after encoding to mp4, and the parquet holds
no image columns.

`scripts/postprocess_psi0.py` converts to the Psi-0 training format: 36-D action
= 14 hand + 14 arm + 3 torso rpy + 1 height + 4 nav (`vx, vy, vyaw, target_yaw`),
32-D state. Note this is the **controller-agnostic high-level interface** — both
AMO and decoupled WBC consume it, so post-processed MP and teleop datasets are
byte-compatible and indistinguishable by schema. Do not try to infer the
controller from a released dataset.

## 5. Four traps

**(a) The G1 MJCF ships the wrong actuator type — this silently produces zero data.**
Upstream `data/robots/g1/g1_29dof_with_dex3.xml` declares all 14 Dex3 hand joints
as `<motor>` (torque). The pipeline writes joint *position* targets, so MuJoCo
applies them as torques, fingers never close, nothing lifts, and every episode
ends `Motion plan exhausted before episode end`. Measured: **0 successes in 168
attempts**. Fix — convert to position actuators and stiffen the waist:

```xml
<position name="right_hand_thumb_0_joint" kp="20" kv="1" forcerange="-5 5" ctrlrange="-3 3"/>
<!-- waist_roll kp 140->300 kv 24->40 ; waist_pitch kp 140->500 kv 24->50 -->
```

With the patch: `RC=0`, 2/2 episodes, zero failures. Upstream's original is kept
at `data/robots/g1/g1_29dof_with_dex3.xml.upstream-orig`.

**(b) `--plan-batch-size` means the opposite thing per grasp type.**
Parallel-gripper tasks (Franka/Aloha/Vega) need `40` — the CLI default of 1 hands
CuRobo one grasp candidate and every attempt fails IK. BoDex/Dex3 tasks require
`1`, because `batch_plan_for_approach_bodex`'s `batch_size > 1` branch in
`src/simple/mp/curobo.py` is a literal `...` stub, so `trajs` is never assigned
(`local variable 'trajs' referenced before assignment`).

**(c) Policy weights are Git LFS.** `amo_jit.pt`, `adapter_jit.pt`,
`adapter_norm_stats.pt` are ~130-byte pointer stubs if you cloned with
`GIT_LFS_SKIP_SMUDGE=1` or without git-lfs. Every G1 whole-body task dies on load.
Without git-lfs, fetch via
`https://media.githubusercontent.com/media/physical-superintelligence-lab/SIMPLE/main/<path>`.

**(d) Some tasks are simply broken upstream.** e.g. `G1TabletopPickNPlaceMP-v0`
declares `eef_pose` and `mujoco` in its `observation_space` but `reset()` returns
neither, so gymnasium's checker raises before step 1 (`rc=1` immediately).

Non-issues you can ignore: the `Warp CUDA error: cuDeviceGetUuid` lines at
startup, and `EGLError: EGL_NOT_INITIALIZED` in the renderer destructor at
teardown.

## 6. Running it

```bash
export MUJOCO_GL=egl OMNI_KIT_ACCEPT_EULA=YES
python -m simple.cli.datagen simple/G1TabletopGraspMP-v0 \
  --sim-mode mujoco --headless --no-webrtc \
  --target-object graspnet1b:21 --num-episodes 2 \
  --plan-batch-size 1 --save-dir data/out
```

`--sim-mode` decides the pixels: `mujoco` = untextured primitive render (fine for
verifying the pipeline, useless for training); `mujoco_isaac` = photorealistic
(HSSD room, vMaterials, textured objects) **in the same single pass**.
`TabletopGraspEnv._render_frame` returns Isaac's frame when Isaac is loaded and
MuJoCo's otherwise. Scripted data therefore **never needs the replay stage** —
that is teleop-only, because Isaac rendering is disabled during VR capture for
latency. First `mujoco_isaac` launch pulls ~1.5 GB of assets and spends minutes
on Isaac boot / RTX shaders.

`LeRobotDataset.create` does `mkdir(exist_ok=False)`, so delete `--save-dir` first
or the run aborts with `FileExistsError`.

Assets: only `robots_g1.zip` (134 MB) + `assets_graspnet.zip` (374 MB) from the
`USC-PSI-Lab/SIMPLE` HF **dataset** repo are needed for G1 tabletop work — not
the 2.8 GB `assets.zip`. `scripts/download_data.sh` references `robots.zip` and
`scenes.zip`, which do not exist in that repo.

CuRobo is a hard import in `src/simple/robots/mixin.py`, so it must build even
for MuJoCo-only runs.

## 7. Expected yield

`G1TabletopGraspMP-v0` with the MJCF patch runs ~**64%** success (measured 1254
lifts ≥4 cm out of 1950 attempts). The paper's 58.9 demos/hr for motion planning
is an aggregate across task families; locomotion and turn-then-reach tasks are
markedly harder. Budget many attempts per saved episode, and remember failures
never reach disk.

Helper scripts added alongside these notes: `run_scripted_datagen_example.sh`
(minimal Franka example, `SIM=mujoco_isaac` for real pixels),
`run_dex3_all_tasks.sh` (sweep all 60 Dex3 tasks), and `make_video_viewer.py`
(builds `video_viewer.html` to browse results episode by episode; shows only
`head_stereo_left` + `front_stereo_left` by default).

## 8. Measured sweep: all 60 Dex3 tasks, 2 episodes each

Run with the patched MJCF, `--plan-batch-size 1`, `--sim-mode mujoco`, 900 s cap
per task. Result: **34 of 60 produced video** — 61 episodes, 415 videos, 6.3 GB.

| Outcome | Tasks |
|---|---|
| 2 episodes | 27 |
| 1 episode | 7 |
| Timed out at 900 s, 0 episodes | 23 |
| Errored at startup | 3 |

**Known-good (2 episodes)** — at least one per capability family, so use these as
your reference when something else misbehaves:
`G1TabletopGraspMP`, `G1WholebodyTabletopGraspMP` + `Variant1/2/3`,
`G1WholebodyTabletopHandoverMP`, `G1WholebodyLocomotionPickBetweenTablesMP` +
`Variant1/2/9/10/12`, `G1WholebodyBendPickAndPlaceOnSofaMP` + `Variant1/2`,
`G1WholebodyBendPickVariant1MP`, `G1WholebodyPickNPlaceVariant1MP`,
`G1WholebodyPickAndBendPlaceMP`, `G1WholebodyXMoveAndPickMP` + `Variant1/3`,
`G1WholebodyXMoveAndPickNPlaceMP`, `G1WholebodyXMoveBendPickMP`,
`G1WholebodyYMoveBendPickMP`, `G1WholebodyTurnPickVariant1MP`,
`G1WholebodyXMoveAndHandoverVariant1MP`,
`G1WholebodyTurnXMoveAndHandoverVariant1MP`.

**Timeouts** are concentrated in the multi-stage composites — most
`LocomotionPickBetweenTables` variants and the `Turn…Move…Bend…` chains. These
are not broken, just low-yield: every stage must succeed within one episode.
Raise the timeout or ask for `--num-episodes 1` to recover many of them.

**Startup failures — three separate upstream bugs:**

| Task | Error |
|---|---|
| `G1TabletopPickNPlaceMP-v0` | `observation_space` declares `eef_pose`/`mujoco`, `reset()` returns neither |
| `G1WholebodyBendPickMP-v0` | `AssertionError: only supports render/physics step parity for g1 wholebody tasks (also follow AMO)` |
| `G1WholebodyTurnYMoveAndPickMP-v0` | `KeyError: 'wxyz'` |

`G1WholebodyBendPickMP-v0` has a **released dataset** on HuggingFace, so it
worked at some point — that one is a regression, not a never-finished task.
