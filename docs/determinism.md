# Reproducible evaluation controls

These controls are opt-in. With no new environment variables, evaluator seeds,
controller timing, mesh order and renderer behavior retain their existing defaults.

| Variable | Behavior |
|---|---|
| `SIMPLE_DETERMINISTIC_RUNTIME=1` | Seed each episode from base seed plus dataset episode index; forward episode metadata; use MuJoCo time in WBC policy calls; sort collision meshes. Requires `SIMPLE_EVAL_SEED`. |
| `SIMPLE_EVAL_SEED=0` | Base seed. Without deterministic mode, retains the previous behavior of passing this same reset seed to each episode. |
| `SIMPLE_SORT_COLLISION_MESHES=1` | Enable sorted collision meshes independently. |
| `SIMPLE_SYNCHRONIZED_RENDERING=1` | Update Isaac before synchronizing; forward articulation kinematics and publish transforms; render without advancing time after synchronization. |
| `SIMPLE_ISAAC_DETERMINISTIC=1` | Enable synchronized rendering, fixed-sample path tracing and renderer initialization/reset warmup. This does not itself enable simulator/model RNG controls. |
| `SIMPLE_ISAAC_SPP=16` | Samples per pixel, integer 1–32; used only in path-tracing mode. |

Use all controls with a compatible PSI model server:

```bash
export SIMPLE_DETERMINISTIC_RUNTIME=1
export SIMPLE_EVAL_SEED=0
export SIMPLE_ISAAC_DETERMINISTIC=1
export SIMPLE_ISAAC_SPP=16
export PYTHONHASHSEED=0
# Start the server with --request-seeding --seed 0, then run the evaluator.
```

Both `cli/eval.py` and `cli/eval_decoupled_wbc.py` forward the dataset episode index.
The PSI adapters already forward it and their executed-action step index in HTTP
request history. The model server must independently implement request seeding;
setting the simulator environment alone cannot seed a separate server process.

WBC reset, stabilization and rollout use actual `robot.mjData.time`, including
stabilization time. A scoped adapter redirects imported clocks in decoupled-WBC
policy modules during those calls. It never modifies the global Python time
module. Outside enabled controller calls, wall clocks remain real. No installed
WBC dependency source is edited. Other WBC baselines receive the same clock
integration, but their model/RNG behavior is not covered by the PSI protocol.

Path tracing disables light/result caching, adaptive sampling, denoising,
auto-exposure and temporal post-processing. The tonemapper is preserved. Settings
are logged and checked by readback; registration alone does not prove a renderer
build honors a setting. Reset discards one initial episode setup, restores global
Python/NumPy/Torch RNG state, then rebuilds using the same explicit seed/options.
Warmup uses 60 captures for the discarded setup and 8 for real resets, checking
that MuJoCo integration state is unchanged. Stage loading is checked with a
180-second timeout; this does not expose all internal renderer state.

## Evidence and limits

The earlier job-local implementation completed three TabletopGrasp greedy runs
at seed 0 with 8/10 success each. Both initial head-camera images and qpos/qvel
matched exactly across all ten episodes. Earlier Handover probes verified WBC
delay invariance and simulation-time timeouts; sorted-model replay matched across
processes. The native implementation here adds runtime switches and extends the
existing GitHub evaluator seed handling. It has CPU regression checks but has not
yet been revalidated on Isaac GPUs in this checkout. Full-rollout bitwise equality
and determinism across driver/runtime versions remain unproven.

Path tracing changes appearance and cost: keep results separate from RTX real-time
benchmarks. A finite total sample cap can stop rendering an unchanged view; equal
static frames alone do not prove repeatability after pose changes/resets.

Run CPU checks with `python -m pytest tests/test_determinism.py`. Full simulator
validation requires the existing SIMPLE assets, Isaac runtime and a supported GPU.
Do not run the previous bucket text-patching scripts over this native implementation.
