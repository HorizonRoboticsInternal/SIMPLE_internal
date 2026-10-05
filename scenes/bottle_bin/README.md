# Tabletop bottle / trash-bin scene (g1comp, MuJoCo)

Standalone MuJoCo scene of the new pick-and-drop setup, built 2026-09-16 the same way as `../tabletop_box/`
(same robot model, head servo setting and D455 camera), plus the same scene as a SIMPLE task for the sonic
(decoupled-WBC) teleop stack and an action-replay tool for real recordings. Not rendered in Isaac yet.

    MUJOCO_GL=glfw DISPLAY=:1 ~/wrk/SIMPLE/.venv/bin/python build_scene.py [--robot-side right] [--cover shelf] [--tag _x]

Writes `scene.xml` (self-contained apart from the robot meshes, keyframe `default_pose`), `assets/*.obj`
(generated bottle and bin meshes), `layout.json` (every computed position, camera pose, contacts) and
`renders/*.png`: `overview`, `front_left`, `front_right`, `side`, `bottle_closeup`, `bin_closeup`,
`ego_d455` (the robot's head camera) and `plan` (orthographic top view with the dimensions drawn on it).
`index.html` (from `make_report.py`) shows all of them.

World frame: robot pelvis at the origin facing +x (the table), z up, the robot's right = -y.
"Left" / "right" below are the robot's left / right.

| item | value | note |
|---|---|---|
| table | 2.07 m across (facing the robot) x 0.60 m deep, top 0.72 m, 3 cm slab, four 4 cm legs | user spec |
| cover board | along the far edge, standing on the slab, top edge 0.92 m above the floor (20 cm lip, 1.8 cm thick) | assumed vertical; `--cover shelf --cover-depth D` = horizontal board at 0.92 m over the far D m |
| bottle | 500 ml PET: 6.5 cm dia x 21.2 cm, 0.5 kg (full; the recorded bottle has water in it and a 0.1 kg bottle tips under the replayed fingers), upright | `--bottle-mass`; generated mesh `assets/bottle_500ml_*.obj` |
| bottle position | axis 56.4 cm from the LEFT edge, 9 cm from the near edge (0.4 cm right of the robot's centre line) | `--bottle-side right` = from the right edge; `--bottle-ref near` = 9 cm to the bottle's near surface |
| robot | g1comp (29 DOF + Dex3 + pan/tilt D455), facing the table | holomotion `g1_comp_45dof.xml` |
| robot placement | front of the feet 3 cm from the near edge (pelvis 15.5 cm; spec 17 cm; 7 cm on 2026-09-16, 3 cm on 2026-09-17: replays of session 02-25-56 grasp 7/10 there vs 5/10 at 7 cm), pelvis 56 cm from the LEFT edge (the table runs 1.51 m to the robot's right) | `--robot-to-edge 0.17` = the spec; `--robot-side right`, `--robot-ref pelvis` |
| trash bin | 36 x 24 x 39 cm, black, 4 cm corner radius, 8 mm wall, open top; turned 90 deg clockwise (user, 2026-09-16): the 36 cm side points toward the table, the 24 cm side runs along the table edge | `--bin-yaw` (default -90); generated mesh `assets/bin_36x24x39.obj`; collision = box segments (group 5) |
| bin position | on the floor 0.84 m to the robot's right and 0.25 m behind the pelvis start = fitted (2026-09-17) to 14 release points of session 02-25-56 replays (two passes at feet 3 cm: x -0.35..-0.13, y -0.90..-0.78, all inside with 1.8 cm to spare); 0.41 m out from the near edge, 0.67 m from the table's right end | `--bin-xy -0.25 -0.84` (default); `--bin-xy 0 -1.40` = the original table-end spot |
| pose | pose 3 (default since 2026-09-16): every upper-body joint 0 except the elbows at -0.66, hands open, waist 0, pelvis 0.74 m, legs solved for flat feet = the `elbows_raised` init pose of the 2026-09-17 recordings and of decoupled_wbc | `--pose 2` = arms hanging, `--pose 1` = Psi0 client pose |
| colours | Dex3 hands black, floor plain light grey (user, 2026-09-16) | `HAND_RGBA`, `FLOOR_RGBA`; the replay applies the same to SIMPLE's model |
| head | pan 0, tilt 10 deg down (114 ticks); D455 axis 57.7 deg below horizontal, lens 1.23 m above the floor | as tabletop_box |
| ego camera | 90 deg HFOV, 16:9 (58.7 deg VFOV), rendered 1280x720 (SIMPLE uses 640x360) | |

Checks in `layout.json`: no contacts except feet-floor (hands 15 cm above the slab, 7 cm past the near edge);
the bottle is straight ahead of the robot, in the head camera's view just beyond the hands.

## SIMPLE task + action replay (MuJoCo, sonic teleop stack)

`bottle_bin_task.py` registers the same layout (numbers read from `layout.json`) as task `g1_wholebody_bottle_bin_teleop`,
gym id `simple/G1WholebodyBottleBinTeleop-v0` (SonicLocoManipEnv, robot g1_sonic): SIMPLE's table primitive 0.60 x 2.07 x
0.03 at top 0.72, generated legs + cover board + bin as static objects (no free joint, coloured; engine patch in the file),
the bottle as the target (free), D455 head camera 640 x 360 / 90 deg / servo 10 deg like the box task (probe: 57.7 deg down).
`BOTTLE_BIN_TARGET=graspnet1b:66` swaps the generated bottle for a GraspNet object in its standing stable pose.

`replay_in_scene.py` = SIMPLE's `replay_real_in_mujoco.py` with the scene kept: the recorded upper-body targets, navigation
command and base height of a real episode drive PicoDecoupledAgent tick by tick; the lower-body RL policy walks by itself.

    cd sim/bottle_bin && MUJOCO_GL=egl ~/wrk/SIMPLE/.venv/bin/python replay_in_scene.py --probe        # replay/probe/{probe.json,third_person.png,head_stereo_left.png}
    MUJOCO_GL=egl ~/wrk/SIMPLE/.venv/bin/python replay_in_scene.py --session 2026-09-17-00-16-29-G1-sim --episodes 12
    -> replay/replay_<session>_ep12.mp4 (third-person | sim D455 | real D455), trace_ep12.json, index.html

Recorded start poses (2026-09-17 session, 15 episodes) vs the elbows_raised target: every episode's first state and first
action are within 0.12 rad (elbows -0.56..-0.72, the right index finger is 0.08 rad off in every episode); the decoupled_wbc
default init pose is the same preset, so the replay warm-up pose matches the recordings.
Replay sweep (2026-09-16 evening, `replay/`): with SIMPLE's 0.1 kg default bottle the fingers tip the bottle at 2.2 s at 17, 12 and
7 cm alike (the sim wrist follows the real wrist path); with a 0.5 kg bottle at 7 cm the grasp closes and lifts 22 cm. The
recordings release the bottle ~0.87 m to the right / 0.2-0.4 m behind the start (commanded-velocity integration; base velocity is
not recorded), so the bin moved there (`--bin-xy -0.32 -0.87`) and episodes 6 and 12 both end with the bottle in the bin.
Flags of replay_in_scene.py: `--robot-to-edge`, `--bottle-mass`, `--bin-xy X Y`, `--tag`; outputs are tagged per run.

Session 2026-09-17-02-25-56 (30 fps; rows are resampled onto the 50 Hz ticks): episodes 0,11,...,99 replayed at feet 7 cm
(5/10 in the bin) and 3 cm (7/10; 66, 77, 99 fail to grasp: their real bottle stood 0.29-0.36 m ahead of the pelvis vs the
scene's 0.245). The real bottle position varied per episode (+-4 cm), so no single distance grasps all ten;
`replay/bottle_from_closure_02-25-56.json` holds each episode's estimated real bottle position for `--bottle-xy`.
`replay/bin_fit.py --tag <tag>` fits the bin centre to the release points of a pass.

## Page and kit

## 2026-09-23: layout refitted on all 97 kept episodes; task gates

- **Can** at pelvis-frame (0.270, -0.004) = 11.5 cm from the near edge (`BOTTLE_FROM_FRONT` 0.115): the middle of a plateau
  (25.5-28.5 cm ahead) where the replays carry the can off the table in 81-84 of 97 episodes, vs 78 at 24.5 cm. It must stay on the
  centre line: the start pose's open thumbs sit at (0.264, +-0.061, 0.888), so a can >~5 mm off the line at 23-30 cm spawns inside a
  thumb (`replay/sweep/spawn_map.json`).
- **Bin** at (-0.36, -1.53), 36 cm side toward the table — 1.53 m to the robot's right, 36 cm behind the pelvis start, just past the
  table's right end (`BIN_XY_DEFAULT`). Two earlier fits ((-0.25, -0.84) on 09-17 and (-0.26, -0.80) on the morning of 09-23) scored
  73 and 78 "in the bin" but only by stopping the robot: it walked into the bin in 97/97 episodes and stopped 0.65 m short of its
  free walk. The bin has to be where a *free* walk ends: replayed with no bin, the robot walks 0.90-1.31 m, turns 77-110 deg (the
  recorded headings, tracked to +-3 deg) and lets go at a median (-0.39, -1.40). 61 bin positions were scored with all 97 episodes
  each, by cans in the bin whose walk the bin did not shorten; (-0.36, -1.53) is on the plateau with the fewest interrupted walks.
  Result: 37/97 in the bin, walk untouched in 87, 10 cut short by > 6 cm. The rest is operator variability (they steered by eye
  to wherever the bin was relative to that episode's start); a per-episode bin at the free-walk release point would catch ~all 83
  carried cans.
- **Task gates** (`bottle_bin_task.py`, in `info["task_progress"]`): `grasped` (right hand on the bottle, off the table, upright,
  held 0.5 s) -> `at_bin` (walked >= 0.5 m, pelvis within 0.8 m of the bin, still for 0.4 s) -> `placed` (axis inside the inner
  opening, below the rim, hand off). Reward 0.3 / 0.6 / 1.0; `check_success` = reward >= 0.9 = placed. `robot_touched_bin` is a
  diagnostic. Over the 97: grasped 77, at_bin 77, placed 36.
- **Tools**: `sweep_replay.py` (headless batch replay, ~8x real time, 10 workers; `--fast-obs`; per-job `bottle_xy` / `bin_xy` /
  `bin_yaw`; records gates, robot-bin contact, feet and base paths), `replay/sweep/run_shards.sh`, `fit_layout.py`,
  `make_fit_plots.py`, `replay/sweep/make_summary.py` -> `summary_20260923.json` read by `make_report.py`, `publish_to_nas.sh`.
  Trap: the WBC's arm interpolation *and* its 1 s teleop safety timeout run on `time.monotonic()`; the sweep installs one virtual
  50 Hz clock in every loaded decoupled_wbc / gear_sonic / simple.agents module (`VirtualClock.install`), otherwise a fresh
  machine (uptime < the clock's start) makes the WBC inject a safe goal every tick and the robot never walks.
- Session 02-25-56 is real-robot teleop despite the `-G1-sim` name: `observation.base_pose` x/y and `base_vel` are zeros; only the
  heading quaternion is real.

## Levels 0-3 in Isaac (2026-09-24)

The same level design as SIMPLE's benchmark sets, rendered in Isaac in the plain HSSD room `scene2` (no living-room furniture; `../make_levels.py`; the Isaac renderer needs the
`Meshes/visual` + `Meshes/collision` layout in the generated `assets/*.usda`, which `_write_usda` now writes):

| level | what changes from the base scene | knob |
|---|---|---|
| 0 | 3 GraspNet distractors on the table and a new table + ground material (object and furniture shader params pinned: a random metallic bowl rendered black) | `BOTTLE_BIN_NUM_DISTRACTORS` |
| 1 | + new lighting | |
| 2 | + the bottle moved inside ±3 × ±8 cm | `BOTTLE_BIN_TARGET_JITTER` / region env var, see `../make_levels.py` |
| 3 | + alternative layout: 10 scenes, each with a different robot start (up to 10 cm back/forward, 5 cm sideways) AND a different table height (±4 cm; SIMPLE's DR manager shifts the table, the table legs follow through per-height mesh variants `<asset>_p031`/`_m040`) | `ROBOT_BACK`, `ROBOT_LEFT`, `LEVEL3_EPISODES` in `../make_levels.py`; the start is stored in each scene's state, so eval needs no env var |

```
python sim/make_levels.py bottle_bin --levels 0 1 2 3 --episodes 10 --out data/evals_scenes
    -> data/evals_scenes/G1WholebodyBottleBinTeleop-v0/dr-level-<n>/  (LeRobot set as data/evals_new20; frames/ep*.png = Isaac head camera AFTER the same stabilisation the eval runs, i.e. the controller's start pose;
       meta/scene_env.json = the env vars used, re-applied by eval_scene.py)
python sim/eval_scene.py bottle_bin simple/G1WholebodyBottleBinTeleop-v0 psi0_decoupled_wbc train --data-format lerobot \
    --data-dir data/evals_scenes/G1WholebodyBottleBinTeleop-v0/dr-level-3 --port 21000 --headless --num-episodes 20
```

Isaac head camera: the kit's D455 pose is a MuJoCo-side patch, so the task sets `isaac_cameras_follow_mujoco = True` and SIMPLE's Isaac engine copies the MuJoCo camera's world pose every step (`_sync_cameras_from_mujoco`); without it the Isaac view is narrower and higher than the verified MuJoCo/real view.

Isaac-rendered replay (`../replay_isaac.py`): runs this kit's `replay_in_scene.py` unchanged inside the SIMPLE level-N scene in
`mujoco_isaac` mode (the env is reset to one scene of `data/evals_scenes`, so the distractors, table material and lighting are
the set's; the middle video panel is then the Isaac head camera). The wall-clock-paced replays get bowl_sink's VirtualClock
installed from outside and ticked per `env.step`, so the controller runs deterministically at 50 Hz despite Isaac's slow steps.
```
python sim/replay_isaac.py bottle_bin --session 2026-09-17-02-25-56-G1-sim --episode 10 --scene 0   # drives sweep_replay.py (the tool behind the verified successes)      # -> replay/isaac_level0_ep<N>.mp4
```

Isaac third person (final recipe, 2026-09-24): two passes per episode plus a separate composer, because nothing can run after a kit
replay inside the Isaac process (SimulationApp shutdown segfaults). Pass A `--no-third` writes the kit video (middle panel = Isaac head
camera). Pass B `--hide-shell` keeps the kit's own free-camera pose (it stands outside the HSSD room, so the room's walls / ceilings /
openings prims are hidden) and saves an Isaac third-person + head JPEG every control tick to `replay/third_bottle_bin_ep10/`. The composer
then swaps the left panel in; the panels are frame-synchronous by construction: kit frame k (written every 2nd tick after the settle)
is tick S + 2k with S = M - 2N (M captured ticks, N kit frames), no image matching.
```
python sim/replay_isaac.py bottle_bin --session 2026-09-17-02-25-56-G1-sim --episode 10 --scene 0 --no-third                                   # pass A -> replay/isaac_level0_ep10.mp4
python sim/replay_isaac.py bottle_bin --session 2026-09-17-02-25-56-G1-sim --episode 10 --scene 0 --hide-shell --out sim/bottle_bin/replay/passB_ep10.mp4   # pass B -> replay/third_bottle_bin_ep10/f*.jpg h*.jpg
python sim/compose_third.py sim/bottle_bin/replay/isaac_level0_ep10.mp4 sim/bottle_bin/replay/third_bottle_bin_ep10 sim/bottle_bin/replay/isaac_level0_ep10_3p.mp4 sim/bottle_bin/replay/isaac_level0_ep10_3p.jpg 0
```
Level-3 heights: `SIMPLE_LEVEL3_TABLE_DZ` (set by make_levels.py, 0.04) is the half-range of the per-scene table offset; the kit reads the actual table top at reset and loads the matching furniture variant.
