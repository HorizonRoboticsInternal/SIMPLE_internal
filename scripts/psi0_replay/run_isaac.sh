#!/usr/bin/env bash
# Isaac (mujoco_isaac) replay with the environment the psi0 eval launchers use.
# usage: scripts/psi0_replay/run_isaac.sh <replay_psi0.py args...>
set -e
cd "$(dirname "$0")/../.."
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export SIMPLE_DETERMINISTIC_RUNTIME=${SIMPLE_DETERMINISTIC_RUNTIME:-1} SIMPLE_EVAL_SEED=${SIMPLE_EVAL_SEED:-0}
exec env -u LD_LIBRARY_PATH .venv/bin/python scripts/psi0_replay/replay_psi0.py --sim-mode mujoco_isaac "$@"
