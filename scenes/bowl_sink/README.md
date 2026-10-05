# Bowl-to-sink kitchen scene (g1comp, MuJoCo) + SIMPLE task + real-run replay

World frame: pelvis at the origin at the start pose, x toward the bowl counter, z up, the robot's right = −y.

| item | value | source / knob |
|---|---|---|
| bowl counter | 2.60 m long (faces the robot) × 0.63 m deep, top **0.86 m**, 3 cm slab on a solid cabinet | drawing "260 x 63 x 86" |
| back unit | 2.60 × 0.41 × 1.07 m against the wall behind it — the only 1.07 m unit | drawing "262 x 41 x 107" |
| sink counter | **2.08** × 0.41 m, worktop **0.86 m = the bowl counter**, **starting at the bowl counter's front-edge line** and running toward the robot's rear | drawing "266 x 41 x 107": user 2026-09-23 corrected the length to 208 and the height to 86 (the 107 is the back unit). `--sink-start front-edge` (default; `wall` = the old layout) |
| sink basin | 0.54 (along) × 0.33 (across, 4 cm rims) × 0.10 m deep, **3 cm rounded corners** | drawing "41 x 54 x 10" + the 4 cm arrow; corners per the user, 2026-09-23; `--basin-radius` |
| basin position | far edge **127 cm from the sink counter's start** (= the bowl counter's front-edge line), so the basin spans 73–127 cm along the counter | the drawing's 127, user-confirmed 2026-09-23 ("the sink right edge from the left table edge"); `--sink-ref far-edge --sink-along 1.27` (default); the 63 was withdrawn by the user. `--sink-ref end --sink-from-end` is still available |
| robot start | pelvis **2.04 m** from the counter's LEFT end, front of the feet **0.38 m** from the front edge (2026-09-24: moved back twice at the user's request so the walk-in does not run the knees into the cabinet; the recording walks the pelvis +0.25 m in its first 3 s) | `--robot-from-left`, `--robot-to-edge`. 0.28 ≈ the measured 0.30 (monocular + leg odometry); 1.85 is where the bowl sat along the counter in the 2026-09-18 session (the drawing's 139 is the 09-21 deploy setup) |
| bowl | 15 cm diameter, 8 cm deep, 4 mm wall, hollow, green, 0.12 kg, **at the counter's edge**: centre 6 cm beyond it, so the 4.5 cm base sits 1.5 cm inside (4 s of physics: 0 drift, 0.02° tilt), **straight ahead** of the robot | user 2026-09-23; `--bowl-from-edge 0.06`, `--bowl-xy`, `--bowl-mass` |
| pose / head | elbows −0.66, all else 0; pan 0, tilt 10°; D455 90 × 58.7° | as bottle_bin |

## Tuning against the 55 teleop episodes (2026-09-23) — read the correction first

**Deterministic mapping and verification (2026-09-24, supersedes the rows below for the start pose).** All replays
now run on a virtual clock (`REPLAY_FAST=1`: no pacing, no video, identical results run to run), one replay per
episode at each walking gain (`replay/run_missing.sh` with `TAG`/`GAIN`/`RTE`/`WORKERS`, traces
`replay/trace_ep*_f{100,130,150,170,190}.json`), the start pose fitted offline from the sim hand at the recorded
closure/release rows (`replay/fit_placement.py`, `replay/closure_rows_0918.json`), then the fitted scene replayed for
real (`trace_ep*_v150.json`, gain 1.5, feet 0.25, from-left 2.04; `trace_ep*_v130.json`, gain 1.3, feet 0.27,
from-left 1.94):

| gain | fitted best start (feet from edge, from left end) | completes predicted | live gates in the 2026-09-23 scene: grasped / placed |
|---|---|---|---|
| 1.0 | 0.27, 2.00 | 8 | 22 / 4 |
| 1.3 | 0.23, 1.96 | 14 | 24 / 9 |
| **1.5** | **0.25, 2.04** (built) | **18** | 24 / 9 |
| 1.7 | 0.24, 2.15 | 12 | 26 / 7 |
| 1.9 | 0.24, 2.15 | 16 | 28 / 10 |

Verification, every episode in one fixed scene: gain 1.3 at 0.27/1.94 → moved to bowl 54, grasped 22, at sink 22,
**placed 8** (of 54); gain 1.5 at the fitted 0.25/2.04 → 55 / 28 / 28 / **placed 7** (of 55). The fit is not
predictive (18 promised, 7 delivered): every start pose and gain tried lands 6–10 of 55, set by how far each episode's
recorded arm path is from this bowl, not by where the robot stands. The scene stays at 0.25/2.04
(`build_scene.py --robot-to-edge 0.38 --robot-from-left 2.04 --bowl-from-edge 0.06`; the 0.30 layout is kept in `_backup_0930/`). At 0.38 only long-reach episodes still grasp (screened 2026-09-24: 42, 41, 7 place the bowl; 19 does not).


**Correction (later on 2026-09-23).** The first pass below fitted the scene to a base path integrated from the
*commanded* velocity. Three independent methods (monocular back-projection of the bowl in the first frame, per-footfall
leg odometry, and an adversarial re-check) show that integral overstates the walk ~2.4× — the robot achieves about
half of what it is commanded. The bowl is 0.57 m ahead of the start pelvis (not 0.92), so the measured start distance
is the original 0.30 m, and the `--nav-gain 1.7` was calibrated against the same biased reference: at gain **1.0** the
sim tracks the commands about as poorly as the real robot and grasps 28/55 with the bowl at each episode's measured
spot (vs 13/55 at 1.7). What the drawing got wrong for THIS session is only the bowl's place along the counter:
~1.85 m from the left end, not 1.39 (the sink is then 0.96 m to its right). Per-episode data: `replay/batch_mono_plan.json`
(monocular bowl), `replay/batch_mono.json` (sim hand at the recorded closure/opening rows, gain 1.0).

| scene (gain 1.0, all 55 episodes, no per-episode help) | grasps | in the basin |
|---|---|---|
| drawing as-is (bowl 1.39 from the left end) | 9 | 0 |
| bowl 1.85 from the left end, 15 cm in from the edge, robot 0.19 | 20 | **15** (verified) |
| bowl 1.85 from the left end, **at the edge**, robot 0.28, sink counter 2.08 from the front-edge line, basin 127 from its start | see `replay/batch_edge.jsonl` | |

The rows below are the superseded first pass, kept for the record.


`replay/fit_points_0918.json` holds, per episode, where the bowl was and where the hand opened, both in that
episode's own start frame (the base path is integrated from the commanded velocity, because the robot's position is
never recorded). Medians over 55 episodes: bowl 0.92 m ahead and 0.03 m right; counter top 0.85 m; turn −90°; the
drop 0.79 m behind and 1.30 m right of the bowl, hand 0.95 m above the floor.

`replay/run_fit_parallel.sh` replays every episode (6+ workers, one process each) and records where the **simulated**
hand is at the recorded closure and opening rows. That trajectory does not depend on what the scene contains, so one
pass fixes both the bowl and the basin:

| scene | along | across | grasps | drops in the basin | both |
|---|---|---|---|---|---|
| drawing, deploy-run start | 1.27 | 1.415 | 5 | 5 | 2 |
| drawing, data start distance | 1.27 | 1.415 | 6 | 32 | 4 |
| fitted to the recordings alone | 0.85 | 1.30 | 6 | 3 | 0 |
| **fitted to the replays (default, verified)** | **1.06** | **1.37** | **15** | **33** | **12** |
| bowl pinned per episode | 1.07 | 1.33 | — | 31 | — |

Out of 55; the default row is the verified run (`replay/run_verify.sh`: all 55 episodes in this one scene, no
per-episode help). Flat optima: 1.05–1.20 m along, 1.30–1.40 m across; re-optimising every placement against the
verified grasps gains at most 3 episodes and moves the scene by 1–3 cm, inside the measurement error. The fitted lateral puts the robot 1.405 m from the
counter's left end against the drawing's 1.39 — a 1.5 cm agreement by an independent route. The 127 is the one number
still open: the replays want 1.06, the recordings alone want 0.85, the drawing says 1.27; the simulated carry runs
~19 cm longer than the real one, which is what pulls the replay's answer back toward the drawing.

**The walk has to be scaled.** The simulated base covers only ~59 % of the commanded displacement and turns ~0.25 rad
further, so the hand arrives 0.14–0.25 m short of the bowl and nothing is picked up. `--nav-gain 1.7` lines them up
(calibrated on episode 37). `--walk-stretch` moves the base as far but still misses the grasp: the arm trajectory is
indexed by row, not by time. The remaining limit is the start pose — the robot was parked differently in every
episode, so the closure point is spread over 31 cm and one bowl position can only serve about a third of them.

## Task gates (2026-09-23): four checks, strictly in order

`bowl_sink_gates.py` — one class used by both the SIMPLE task (live, every `env.step`, driving reward and success)
and `replay_in_scene.py` (live during a replay; `gates` in every summary and batch line; `--tag` traces can also be
re-scored offline with `BowlSinkGates.from_trace`).  Each gate latches with the time it was met and can only latch
after the previous one; `info["task_progress"]` carries the flags and diagnostics.

| # | gate | passes when | knobs (`GateCfg`) |
|---|---|---|---|
| 1 | **at_bowl** — moved to the bowl and stopped | walked ≥ 5 cm since the start, bowl within 0.60 m of the pelvis and within 60° of the heading, base still (< 0.15 m/s) for 0.4 s, before any grasp | `min_walk_bowl`, `at_bowl_radius`, `at_bowl_bearing`, `still_speed`, `hold_s` |
| 2 | **grasped** — picked the bowl up | right-hand geoms in contact with the bowl and the bowl ≥ 5 cm above its start height, held together for 0.3 s (a knock does not count) | `grasp_lift`, `grasp_hold_s` |
| 3 | **at_basin** — moved to the sink and stopped | walked ≥ 20 cm since the grasp, basin centre within 0.80 m of the pelvis and within 75° of the heading, base still for 0.4 s | `min_walk_basin`, `at_basin_radius`, `at_basin_bearing` |
| 4 | **placed** — dropped/placed the bowl in the basin | hand no longer touching the bowl, bowl centre inside the basin footprint shrunk by the bowl's 4.5 cm base radius, and below the rim (0.86 m). `settled` is added once it also rests there for 0.4 s | `basin_len`, `basin_across`, `bowl_base_r`, `sink_h` |

Reward = 0.25 per gate passed, 1.0 once placed; `check_success` = placed (reward ≥ `success_criteria`).  Diagnostics:
`max_lift`, `tilt_at_grasp`, `min_pelvis_to_bowl`, `min_pelvis_to_basin`, `bowl_end_minus_basin`, `hand_on_bowl_s`,
`stalled_at` (the first gate not passed).  In a replay, contact comes from the MuJoCo contact list; when re-scoring an
old trace, "hand on the bowl" = palm or index tip within 13 cm of the bowl centre.

Funnel on the verified batch of 2026-09-23 (`replay/trace_ep*_final.json`, 55 episodes, bowl 15 cm in from the edge):
at_bowl **55** → grasped **17** → at_basin **17** → placed **12** (settled 7).  Every grasp reached the sink; five drops
missed the basin.  The old single `in_sink` flag agrees with `placed` on 52 of 55 (it accepted a bowl still held over the
basin, or resting on the rim).

## Files

`bowl_sink_task.py` reads `layout.json` and honours `BOWL_SINK_ROBOT_TO_EDGE`, `BOWL_SINK_BOWL_XY="x,y"`,
`BOWL_SINK_SINK_ALONG`, `BOWL_SINK_SINK_REF`, `BOWL_SINK_BOWL_MASS`, `BOWL_SINK_INSTRUCTION`.
`replay_in_scene.py` takes `--session psi0/BowlToSink_0918` (or `--client-log` for a holobrain deployment run),
`--episodes`, `--robot-to-edge`, `--bowl-xy X Y`, `--sink-along`, `--nav-gain`, `--walk-stretch`, `--tag`.

## Levels 0-3 in Isaac (2026-09-24)

The same level design as SIMPLE's benchmark sets, rendered in Isaac in the plain HSSD room `scene2` (no living-room furniture; `../make_levels.py`; the Isaac renderer needs the
`Meshes/visual` + `Meshes/collision` layout in the generated `assets/*.usda`, which `_write_usda` now writes):

| level | what changes from the base scene | knob |
|---|---|---|
| 0 | 3 GraspNet distractors on the table and a new table + ground material (object and furniture shader params pinned: a random metallic bowl rendered black) | `BOWL_SINK_NUM_DISTRACTORS` |
| 1 | + new lighting | |
| 2 | + the bowl moved ±8 cm along the edge (no jitter toward the edge: it stays on the edge line) | `BOWL_SINK_TARGET_JITTER` / region env var, see `../make_levels.py` |
| 3 | + alternative layout: 10 scenes, each with a different robot start (up to 10 cm back/forward, 5 cm sideways) AND a different table height (±4 cm; SIMPLE's DR manager shifts the table, the counter cabinet and sink counter (basin and gate heights follow) follow through per-height mesh variants `<asset>_p031`/`_m040`) | `ROBOT_BACK`, `ROBOT_LEFT`, `LEVEL3_EPISODES` in `../make_levels.py`; the start is stored in each scene's state, so eval needs no env var |

```
python scenes/make_levels.py bowl_sink --levels 0 1 2 3 --episodes 10 --out data/evals_scenes
    -> data/evals_scenes/G1WholebodyBowlSinkTeleop-v0/dr-level-<n>/  (LeRobot set as data/evals_new20; frames/ep*.png = Isaac head camera AFTER the same stabilisation the eval runs, i.e. the controller's start pose;
       meta/scene_env.json = the env vars used, re-applied by eval_scene.py)
python scenes/eval_scene.py bowl_sink simple/G1WholebodyBowlSinkTeleop-v0 psi0_decoupled_wbc train --data-format lerobot \
    --data-dir data/evals_scenes/G1WholebodyBowlSinkTeleop-v0/dr-level-3 --port 21000 --headless --num-episodes 20
```

Isaac head camera: the kit's D455 pose is a MuJoCo-side patch, so the task sets `isaac_cameras_follow_mujoco = True` and SIMPLE's Isaac engine copies the MuJoCo camera's world pose every step (`_sync_cameras_from_mujoco`); without it the Isaac view is narrower and higher than the verified MuJoCo/real view.

Isaac-rendered replay (`../replay_isaac.py`): runs this kit's `replay_in_scene.py` unchanged inside the SIMPLE level-N scene in
`mujoco_isaac` mode (the env is reset to one scene of `data/evals_scenes`, so the distractors, table material and lighting are
the set's; the middle video panel is then the Isaac head camera). The wall-clock-paced replays get bowl_sink's VirtualClock
installed from outside and ticked per `env.step`, so the controller runs deterministically at 50 Hz despite Isaac's slow steps.
```
python scenes/replay_isaac.py bowl_sink --episode 7 --nav-gain 1.5 --scene 0      # -> replay/isaac_level0_ep<N>.mp4
```

Isaac third person (final recipe, 2026-09-24): two passes per episode plus a separate composer, because nothing can run after a kit
replay inside the Isaac process (SimulationApp shutdown segfaults). Pass A `--no-third` writes the kit video (middle panel = Isaac head
camera). Pass B `--hide-shell` keeps the kit's own free-camera pose (it stands outside the HSSD room, so the room's walls / ceilings /
openings prims are hidden) and saves an Isaac third-person + head JPEG every control tick to `replay/third_bowl_sink_ep7/`. The composer
then swaps the left panel in; the panels are frame-synchronous by construction: kit frame k (written every 2nd tick after the settle)
is tick S + 2k with S = M - 2N (M captured ticks, N kit frames), no image matching.
```
python scenes/replay_isaac.py bowl_sink --episode 7 --nav-gain 1.5 --scene 0 --no-third                                   # pass A -> replay/isaac_level0_ep7.mp4
python scenes/replay_isaac.py bowl_sink --episode 7 --nav-gain 1.5 --scene 0 --hide-shell --out sim/bowl_sink/replay/passB_ep7.mp4   # pass B -> replay/third_bowl_sink_ep7/f*.jpg h*.jpg
python scenes/compose_third.py sim/bowl_sink/replay/isaac_level0_ep7.mp4 sim/bowl_sink/replay/third_bowl_sink_ep7 sim/bowl_sink/replay/isaac_level0_ep7_3p.mp4 sim/bowl_sink/replay/isaac_level0_ep7_3p.jpg 0
```
Level-3 heights: `SIMPLE_LEVEL3_TABLE_DZ` (set by make_levels.py, 0.04) is the half-range of the per-scene table offset; the kit reads the actual table top at reset and loads the matching furniture variant.
