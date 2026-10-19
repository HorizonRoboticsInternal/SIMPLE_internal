# Action replay of the real G1 recordings → sim datasets

How the real teleop recordings (bottle → bin, bowl → sink, coffee cart) are replayed in simulation and turned into the
`data/replay_datasets_phys/<kit>` datasets. Everything here runs from the SIMPLE checkout root with its `.venv`.

```
real recording (psi0 LeRobot)          scene fit per episode            record + render               dataset
───────────────────────────    →   ───────────────────────────   →   ─────────────────────────   →   ────────────────
36-D action per row, 30 fps        where the hand closes ->          A. MuJoCo action replay,        build_dataset.py
hands, arms, torso rpy,            object; where it opens ->         full state saved per step       to_psi0.py
height, nav command                container; walk must be free     B. Isaac render of that
                                   (sweep_replay / fit_batch)        state, frame by frame
                                                                     (phys_render.py)
```

Two different replays are involved, and the names matter:

| | what drives the sim | used for |
|---|---|---|
| **action replay** | the recorded psi0 action, row by row at 50 Hz, through the decoupled-WBC agent (`PicoDecoupledAgent`): upper-body joint targets, torso rpy, base height, navigation command; the lower-body RL policy walks by itself; MuJoCo physics decides what happens | fitting the scene, validating success, recording the state |
| **state replay** | the MuJoCo state recorded by the validated action replay, forced into the Isaac scene before every step (`--state-in`); no physics runs during rendering | rendering the dataset frames (the images show exactly the validated run) |

SIMPLE's upstream `scripts/run_replay_wbc.sh` / `replay_wbc.py` ("npz replay": MuJoCo state trace → Isaac) is the same two-phase
idea for the 64-D WBC teleop datasets only; it does not read the psi0 recordings and is not used here.

## 1. Inputs

| input | where | notes |
|---|---|---|
| real recordings, psi0 LeRobot v2.1 | `data/real_recordings/psi0/ChipCanToTrash_0225_keep97` (97 eps), `psi0/BowlToSink_0918` (55), `psi0/CartCoffeeCup_0919` (97) | 30 fps, `action` 36-D = `[0:7] left hand, [7:14] right hand, [14:21] left arm, [21:28] right arm, [28:31] torso rpy, [31] height, [32:36] vx vy flag target_yaw`; `observation.images.egocentric` video. Not in git: copy from the workstation (`/home/Horizon/wrk/SIMPLE/data/real_recordings/psi0/`). The bucket LMDBs (`simple_teleop/psi_g1_chipcantotrash_0225_keep97_lmdb`, …) are exports **of** these sets; nothing here reads LMDB. |
| raw session (bottle only, optional) | `data/real_recordings/2026-09-17-02-25-56-G1-sim` (102 eps, has `observation.state`) | the keep97 set is 97 of these episodes in the psi0 layout; the tools accept both formats |
| scene layouts + assets | `data/real_scenes/<kit>/{layout.json, assets/}` (git LFS) | `git lfs checkout data/real_scenes src/simple/robots` after cloning |
| robot | `data/robots/g1_comp/g1_comp_45dof.xml` (tracked) + `data/robots/g1comp/head_meshes/{head_servo_link,xl330_link,d455_link}.STL` | the head STLs are not in git: copy from `/mnt/nas28/alan.jiang/samebg_sim_datasets_20261012/robot/head_meshes/` or point `SIMPLE_G1COMP_ASSET_DIR` at them (see `docs/g1comp_real_scene_robot.md`) |
| SIMPLE assets | `data/robots/g1_sonic`, `data/scenes` (HSSD), `data/assets`, `data/vMaterials_2` | `bash data/_download.sh` (or the lazy download on first use) |
| scene sets | `data/evals_scenes/<env>/dr-level-0` (tracked) | `replay_isaac.py` reads `meta/scene_env.json` of the level even with `--no-state` |
| fits | `scenes/<kit>/replay/validated_episodes.json` (tracked; bottle 91, bowl 54, coffee 52 episodes) | the per-episode scene parameters + the reference base path of the validated run; built from the full fit / sweep / trace data (see §3) |

Environment: `MUJOCO_GL=egl`, `OMNI_KIT_ACCEPT_EULA=YES` (both set by the tools), optional `SIMPLE_DATA_DIR` (data root, default `data/`),
`PHYS_OUT` (dataset root, default `data/replay_datasets_phys`), `REPLAY_LOG_DIR` (logs, default `data/replay_logs`).
`scenes/<kit>/replay/` is git-ignored except the shipped files: every run writes its results there.

## 2. Stage A — action replay in MuJoCo

`scenes/bottle_bin/sweep_replay.py` (bottle) and `scenes/<kit>/replay_in_scene.py` (bowl, coffee; driven through
`scenes/replay_isaac.py --sim-mode mujoco`) do the same thing: the 30-fps rows are resampled onto the 50 Hz control ticks, the
hand / arm / waist targets and the navigation command of each tick go to the agent, the lower body walks. Before row 0 the robot
settles (150 ticks) and holds the recording's first upper-body pose (100 ticks) so the arm is where the recording starts.

```bash
# bottle: one episode with an explicit placement (metres in the robot's start frame, x toward the table, right = -y)
python scenes/bottle_bin/sweep_replay.py --session psi0/ChipCanToTrash_0225_keep97 --tag demo --episodes 2 \
    --bottle-xy 0.2551 -0.0097 --bin-xy -0.4543 -1.2918 --bin-yaw -108.5 --video-dir data/replay_logs/demo
#   -> scenes/bottle_bin/replay/sweep/demo/ep002.json   (grasp_ok, in_bin, release, base_path, bottle_end, ... )
#      data/replay_logs/demo/replay_psi0_ChipCanToTrash_0225_keep97_ep002_demo.mp4  (third person | sim head cam | real head cam)
# bottle: many episodes from a jobs file (one {ep, bottle_xy, bin_xy, bin_yaw} per episode), several processes
python scenes/bottle_bin/sweep_replay.py --session psi0/ChipCanToTrash_0225_keep97 --tag bottlevid2 \
    --jobs scenes/bottle_bin/replay/jobs_bottlevid2.json --shard 0/4 --fast-obs      # repeat with 1/4 .. 3/4

# bowl / coffee: one episode, scene knobs through the kit's env vars
BOWL_SINK_ROBOT_TO_EDGE=1.132 BOWL_SINK_BOWL_XY=1.332,0.205 BOWL_SINK_SINK_ALONG=1.154 BOWL_SINK_ROBOT_DY=1.25 \
python scenes/replay_isaac.py bowl_sink --session psi0/BowlToSink_0918 --episode 0 --nav-gain 1.5 \
    --sim-mode mujoco --no-state --no-third --summary-out /tmp/ep0.json --trace-out /tmp/ep0_trace.json
# bowl / coffee: many episodes from a fits json ({"<ep>": {params}}), N processes
python scenes/fit_batch.py bowl_sink --fits my_fits.json --tag fit1 --jobs 8      # -> scenes/bowl_sink/replay/fitbatch_fit1.jsonl
```

Success per kit (what `phys_render.success()` checks): bottle `grasp_ok and in_bin`; bowl gate `placed`; coffee
`task_success and on_table and pre_grasp.ok and max_lift_m >= 0.04`. Keep at most ~10 replay processes on one machine (the
DDS domain overflows beyond ~12).

## 3. Stage B — per-episode scene fit and validation

The replayed hand path is open loop, so the scene is fitted to each recording: the object goes where the hand closes, the
container where the hand opens, and the walk must be free of furniture. Rules: physical contact only (no kinematic attach,
grasp assist, pins — the options exist in the code for diagnostics and are off in every validated run), the object untouched
until 1 s before the recorded closure, the base path within 0.10 m of a free-walk / carry twin (same actions, furniture far away),
leg contact with furniture ≤ 25 ticks (bottle) / ≤ 50 (bowl).

Bottle (`scenes/bottle_bin/replay/`):

```bash
# 1. free walk: table 6 m ahead, bin 21 m away -> where the hand closes / opens with nothing in the way
python scenes/bottle_bin/sweep_replay.py --session psi0/ChipCanToTrash_0225_keep97 --tag freewalk --jobs scenes/bottle_bin/replay/jobs_freewalk.json --fast-obs
# 2. placement from the free walk (bottle = palm + (0.098, 0.094) at closure; bin = palm + rot(heading, (0.20, 0.13)) at release, >= 0.40 m ahead)
python scenes/bottle_bin/replay/freewalk_fit_bottle.py --out scenes/bottle_bin/replay/jobs_fwb0.json
# 3. replay the jobs, and their carry twins (same job, bin 21 m away)
python scenes/bottle_bin/sweep_replay.py --session psi0/ChipCanToTrash_0225_keep97 --tag fwb0  --jobs scenes/bottle_bin/replay/jobs_fwb0.json --fast-obs
python scenes/bottle_bin/sweep_replay.py --session psi0/ChipCanToTrash_0225_keep97 --tag fwb0t --jobs scenes/bottle_bin/replay/jobs_fwb0.json --fast-obs --bin-xy -15 15
# 4. validity -> valid_fwb0.json; nudge the failures (+-1.5 cm) and repeat; collect the best valid job per episode
python scenes/bottle_bin/replay/valid_bottle_twin.py fwb0 fwb0t
python scenes/bottle_bin/replay/collect_fw_bottle.py fwb0 fwb1 ...   # -> jobs_bottlevid.json
```

Bowl / coffee: `scenes/bowl_sink/replay/freewalk_fit.py`, `scenes/coffee_cart/replay/freewalk_table.py` build the fits from the free
walk; `scenes/fit_batch.py` replays a fits json; `scenes/fit_next.py <kit> --from <tag> --to <next>` moves the container under
where the object landed / the object under where the hand closed; `walk_gap*.py` check the walk against the free twin
(`walkgap_<tag>.json`). Coffee's cup is grasped closed-loop (`cup_closed_loop.py`, `cup_close_row` / `cup_open_row` in the params).

The outcome of all rounds is shipped as `scenes/<kit>/replay/validated_episodes.json` = `{ep: {params, ref_path}}` (bottle also
keeps the raw `valid_bottlevid*.json` + `sweep/bottlevid*/ep*.json`). `phys_render.validated(kit)` reads it.

## 4. Stage C — record the state, render in Isaac, build the episode

```bash
python scenes/phys_render.py bottle_bin --jobs 3                 # all validated episodes, 3 at a time
python scenes/phys_render.py bottle_bin --jobs 1 --episodes 2 46  # a few
```

Per episode (`scenes/phys_render.py`):

1. **record** — `replay_isaac.py <kit> --episode N --seed 1000+N --no-third --no-state --sim-mode mujoco --state-out
   data/replay_datasets_phys/_states/<kit>_epNNN.npy` with the fitted placement (`BOTTLE_BIN_JOB_*` / `BOWL_SINK_*` / `COFFEE_CART_*`
   env from the params): the full MuJoCo integration state before every `env.step`. Kept only if the task succeeds again and
   the base path stays within 0.03 m of the validated run (`rec_ok`).
2. **render** — the same command in `mujoco_isaac` mode with `--state-in <npy> --export data/replay_datasets_phys/<kit>/ep_NNN`:
   every step starts from the recorded state (`forced == steps` in `replay_meta.json`), the Isaac head camera (D455 on the
   g1comp head, 640 × 360) is exported from the first recorded row on (the settle is not exported), seed 1000+N picks the table /
   floor materials and the lights, the light level comes from `scenes/<kit>/replay/light_levels.json` (70 % bright / 30 % dim).
3. **build** — `scenes/build_dataset.py <ep dir>`: `frames.npz` + `images/*.jpg` → `ep_NNN/dataset` (SIMPLE teleop LeRobot
   schema, 50 fps) via the Gr00t exporter; `video.mp4` = the kit video with the Isaac head camera in the middle panel.

Outputs: `data/replay_datasets_phys/<kit>/ep_NNN/{dataset/, frames.npz, video.mp4, replay_meta.json}`,
`data/replay_datasets_phys/_states/<kit>_epNNN.npy` (float64 `(T, 566)`: time, qpos 57, qvel 55, act, warm start, ctrl, …),
`data/replay_datasets_phys/<kit>/results.jsonl` (one line per episode: `rec_success, rec_path_dev, rec_ok, frames, forced, steps,
isaac_success, built, ok, wall_s`), logs in `data/replay_logs/phys_render_<kit>/`. An episode counts when `ok` is true.
On the workstation this produced bottle 90 / bowl 50 / coffee 52 episodes (`replay_datasets_phys/<kit>`); the recorded states are
also in the NAS-28 bundle `/mnt/nas28/alan.jiang/samebg_sim_datasets_20261012/_states/`. Re-rendering from existing states
(same motion, another look) does not need stage A again: `scenes/render_fix.py KIT EP ...` and the same-background variant
`SAMEBG_SEED=1000 SAMEBG_LIGHT=bright python scenes/render_samebg.py bottle_bin 2 46` (→ `data/replay_datasets_phys_samebg`).

A re-run regenerates an episode that is physically equivalent, not identical: today even the tree that made the datasets
reproduces 10 of 22 validated bottle jobs bit-exactly (the others within a few mm, same outcome), and a fresh record of episode 2
differs from the delivered one by up to 0.08 rad in the joints late in the episode with the same 1250 frames and result. To
reproduce the delivered datasets exactly, render from the delivered states (`render_fix.py` / `render_samebg.py`) instead of
re-recording. State width: the delivered `_states` are 566 wide (recorded on the stock G1, 2026-09-29); recordings made with
the g1comp robot are 584 (its three head bodies); `--state-in` maps either onto the current model.

Dataset frame `k` of an episode is recorded state row `k + 202` (150 settle + 100 pre-pose ticks, minus the rows before the
export gate opens): use `replay_meta.json` (`frames`, `steps`) when aligning.

## 5. Stage D — psi0-format datasets

```bash
python scenes/to_psi0.py bottle_bin                                   # -> data/replay_datasets_phys/psi0/ChipCanToTrash_0225_sim
python scenes/to_psi0.py bottle_bin --phys-root data/replay_datasets_phys_samebg --out-name ChipCanToTrash_0225_sim_samebg --light-level bright
```

Converts every built `ep_NNN/dataset` into the psi0 LeRobot schema of the real sets (`observation.hand_joints` 14,
`arm_joints` 14, `leg_joints` 15, `prev_torso_rpy`, `prev_height`, `states`, `action` 36-D in the real order conventions,
`observation.images.egocentric`), keeping 50 fps, one dataset per task; `meta/episodes.jsonl` keeps the source episode and
the scene parameters. The LMDB training exports on the bucket were made from these datasets outside this repo.

## 6. Replay-time physics pins (`scenes/replay_physics.py`)

The fits and the datasets were made with the SIMPLE working tree of 2026-09-25..29. This branch's eval-side defaults changed
afterwards, and with them the validated placements do not reproduce (bottle `bottlevid2`, 22 jobs: 4 outcomes flip, base paths
off by up to 14 cm). Every replay tool therefore imports `scenes/replay_physics.py` before `gym.make`, which pins:

| pin | branch default (eval) | replay value | source of the change |
|---|---|---|---|
| arm controller init pose (`G1Sonic.controller_cfg.{left,right}_arm_cfg.init_qpos`) | raised arms `[0.5, 0, ±0.2, 0.55, 0, 0, 0]` | zeros | upstream merge 2026-10-06 |
| `g1_sonic.STABILIZE_VEL_THRESHOLD` | 1e-3 | 1e-4 | upstream merge 2026-10-06 |
| task `initial_joint_positions` (fixed level-3 arm/hand pose written at reset) | set for the three real-scene tasks | `{}` → plain first physics step | `03818db` 2026-10-08 |
| `PicoDecoupledAgent` elastic-band start (hang, lower, land, then WBC) | commented out upstream 2026-08-27 | restored in `src/simple/agents/pico_decoupled_agent.py` | this commit |

With the pins the branch reproduces today's local tree exactly (same 22/22 outcomes, 20/22 base paths within 3 cm, 10/22
bit-identical; the remaining mm-level differences exist between any two runs of the local tree since 2026-09-29 as well).
Evaluation does not import the module and keeps the branch defaults. If a future change moves these numbers again, re-run
`sweep_replay.py --jobs scenes/bottle_bin/replay/jobs_bottlevid2.json --tag check` and compare with `replay/sweep/bottlevid2/`.

## 7. Checks

- A single job reproduces bit-exactly in a fresh process (`sweep_replay.py --self-check`); the physics is chaotic across
  placements (1 cm can flip a grasp), so re-fit, do not "nudge" a validated job.
- `phys_render` step 1 re-runs the validated placement and compares the base path (`rec_path_dev` ≤ 0.03 m); a larger
  deviation means the scene or robot changed (table height, asset geometry, controller) — fix that before rendering.
- `replay_meta.json`: `forced == steps` and `state_in_problem` empty, else the render is not the validated run.
- Review pages used for the delivered sets: http://10.40.11.11:8899/replay_phys/ (data check), /replay_review_physical/.

## 8. Traps

- The three kits share module names (`build_scene`, `replay_in_scene`): import one kit per process (`replay_isaac.py` does).
- Isaac: the kit's `env.close()` segfaults the process after an export; the tools save first and `os._exit`. One retry per
  episode is built into `phys_render.py`.
- Real rows are 30 fps (`relabel_fps30.py` history); the replays resample onto 50 Hz ticks; 1 tick = 20 ms, 1 row = 33 ms.
- psi0 left-hand order: the psi0 **action** orders the left hand thumb, middle, index while the state is thumb, index, middle;
  `REPLAY_PSI0_LEFT_HAND=legacy` restores the old (swapped) mapping that the coffee recordings were rendered with.
- A fresh worktree / clone has LFS pointers instead of scene assets (`git lfs checkout`); a missing
  `data/robots/g1comp/head_meshes` fails at robot construction.
- `scenes/<kit>/replay/` and `data/replay_*` fill with run output; both are git-ignored. New shipped fit files need `git add -f`.

## 9. File map

| file | role |
|---|---|
| `scenes/bottle_bin/sweep_replay.py` | bottle action replay: batch, headless, jobs / shards, twin runs, state recording hooks |
| `scenes/<kit>/replay_in_scene.py` | bowl / coffee action replay (also the paced single-episode tool with video) |
| `scenes/replay_isaac.py` | runs a kit replay inside a SIMPLE scene set, MuJoCo or MuJoCo + Isaac; `--state-out / --state-in / --export / --seed` |
| `scenes/fit_batch.py`, `scenes/fit_next.py`, `scenes/<kit>/replay/*.py` | fitting rounds and validation per kit |
| `scenes/phys_render.py` | record + render + build, the dataset driver; `validated()` = the shipped fits |
| `scenes/replay_physics.py` | the replay-time robot / reset pins (§6), imported by every kit replay tool |
| `scenes/build_dataset.py`, `scenes/merge_lerobot.py`, `scenes/merge_clean.py`, `scenes/repair_meta.py` | episode datasets, merging, meta repair |
| `scenes/to_psi0.py`, `scenes/patch_psi0.py` | psi0-format datasets and in-place label patches |
| `scenes/render_fix.py`, `scenes/render_samebg.py` | re-render from recorded states (per episode / same background) |
| `scenes/export_batch.py`, `scenes/collect_best.py`, `scenes/isaac_refit.py` | the earlier (pre-state-replay) Isaac export path, kept for the fit history |
| `scenes/<kit>/replay/validated_episodes.json`, `light_levels.json` | shipped inputs of stage C |
