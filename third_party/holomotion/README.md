# Vendored HoloMotion publisher

Files copied from HoloMotion **v1.4.1** so SIMPLE can run HoloMotion teleop
without a separate HoloMotion checkout.

```text
holoretarget/                                 HoloRetarget package + robot assets
  assets/unitree_g1/g1_mocap_29dof.xml        29-dof mocap MJCF (+ meshes)
  assets/unitree_g1/g1_comp_45dof.xml         composite: Dex3 hands + pan/tilt head
  assets/target_configs/smplx_to_g1.json      IK target table
deployment/holomotion_teleop/
  holomotion_teleop_node.py                   PICO -> HoloRetarget -> ZMQ publisher
  holomotion_teleop_mjviewer.py               MuJoCo viewer for the reference stream
  setup_holomotion_teleop_x86_ubuntu2204.sh   publisher environment bootstrap
run_publisher.sh                              launcher that sets PYTHONPATH
```

Only the *publisher* lives here. The SIMPLE side (agent, observation builder,
ONNX inference, recording) needs none of it — see
`src/simple/teleop/holomotion/` and `docs/holomotion_teleop.md`.

## Environment

The publisher does **not** run in the SIMPLE venv. HoloRetarget solves IK with
Newton/Warp on a CUDA GPU, and PICO input needs the XRoboToolkit SDK:

* `newton==1.0.0`, `warp-lang==1.12.0` (NVIDIA index), `trimesh`
* `numpy`, `pyzmq`, `mujoco`
* `xrobotoolkit_sdk` (built from the XRoboToolkit PC service SDK)
* a CUDA-capable GPU

Bootstrap a conda env (`holomotion_teleop` by default):

```bash
bash deployment/holomotion_teleop/setup_holomotion_teleop_x86_ubuntu2204.sh
```

The script also `pip install -e`'s a HoloMotion checkout, which does not exist
in this vendored copy — that step is unnecessary here because `run_publisher.sh`
puts this directory on `PYTHONPATH` so `import holoretarget` resolves. If the
script fails at that step, the environment is still usable.

Installing these packages into the SIMPLE venv is not recommended: it would
upgrade `warp-lang` from the version SIMPLE pins.

## Run

```bash
./run_publisher.sh                                   # bind tcp://*:6001 at 50 Hz
HOLOMOTION_PY=/path/to/python ./run_publisher.sh --hz 50
```

Then, in the SIMPLE venv:

```bash
teleop-holomotion simple/G1WholebodyLocomotionPickBetweenTablesHoloMotionTeleop-v0 --record
```

Without a headset, use the SIMPLE-native replay instead of the publisher:

```bash
holomotion-replay --hand-demo
```

## Keeping in sync

These files are a copy, not a submodule. When updating from upstream
HoloMotion, re-copy `holoretarget/` and `deployment/holomotion_teleop/*.py`,
then re-apply the one local change: `holomotion_teleop_mjviewer.py` inserts
this directory on `sys.path` so `holoretarget` imports in the vendored layout.
Wire-format or schema changes must also be mirrored in
`src/simple/teleop/holomotion/protocol.py`.
