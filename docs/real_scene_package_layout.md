# Real scene runtime layout

The three new tasks follow the flat layout used by the maintained SIMPLE
bucket snapshot:

- `src/simple/tasks/g1_wholebody_bottle_bin_teleop.py`
- `src/simple/tasks/g1_wholebody_bowl_sink_teleop.py`
- `src/simple/tasks/g1_wholebody_coffee_cart_teleop.py`

`tasks/__init__.py` imports their registered task classes. `envs/__init__.py`
registers their existing Gym IDs with `SonicLocoManipEnv` and a task UID, just
like the existing teleoperation tasks. Geometry helpers are flat task modules.
There is no scene task subpackage or extra scene environment class.

Read-only layouts and fixed scene assets live under
`${SIMPLE_DATA_DIR:-data}/real_scenes/<scene>/`; G1-comp robot assets live under
`data/robots/g1_comp`. These directories must be published separately with the
shared data bucket before cluster use. The standard evaluator applies
`scene_env.json` and refreshes the selected task before constructing its env.
Top-level scene builders/evaluation conveniences and asset symlinks remain;
duplicate task/gate entrypoints are removed. Tools import packaged modules directly.

HoloMotion runtime dependencies are under `simple/teleop/holomotion_v14` and
camera calibration/runtime resources under `simple/resources/hbvcam_stereo`.
Original third-party paths remain compatibility symlinks; original notices
are preserved. No new bucket snapshot has been published by this local change.

Generated OBJ/USD meshes are written to a local per-process cache under
`SIMPLE_SCENE_CACHE_DIR` (or `XDG_CACHE_HOME`, falling back to the temporary
directory). Cache paths include the scene and a digest of its layout/knobs.
Level-3 variants remain separate by height offset. Shared resources are only
read, so `real_scenes/` can be linked from the bucket. Missing fixed resources
raise an error instead of generating files in the shared source directory.

HoloMotion also writes generated robot XML and model-bundle symlinks to this
local cache. Cached robot XML uses absolute paths to the fixed stock meshes,
so moving it out of `data/robots` does not require copying or modifying them.
Do not set `SIMPLE_SCENE_CACHE_DIR` to a bucket path.
