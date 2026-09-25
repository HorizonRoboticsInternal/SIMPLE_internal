#!/usr/bin/env bash
# Run the vendored HoloMotion reference publisher (PICO -> HoloRetarget -> ZMQ).
#
# The publisher needs its own environment (Newton/Warp + CUDA and the
# XRoboToolkit SDK), not the SIMPLE venv. Point HOLOMOTION_PY at that
# interpreter, or set HOLOMOTION_TELEOP_CONDA_ENV to a conda env name.
#
#   ./run_publisher.sh                       # defaults: bind tcp://*:6001 at 50 Hz
#   ./run_publisher.sh --hz 50 --hand-source trigger --head-source headset
#
# Any extra arguments are forwarded to holomotion_teleop_node.py.
set -euo pipefail

VENDOR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE="$VENDOR_ROOT/deployment/holomotion_teleop/holomotion_teleop_node.py"
ENV_NAME="${HOLOMOTION_TELEOP_CONDA_ENV:-holomotion_teleop}"

resolve_python() {
  if [[ -n "${HOLOMOTION_PY:-}" ]]; then
    echo "$HOLOMOTION_PY"
    return
  fi
  for base in "${CONDA_PREFIX:-}/envs" "$HOME/miniconda3/envs" "$HOME/anaconda3/envs" "$HOME/micromamba/envs"; do
    if [[ -x "$base/$ENV_NAME/bin/python" ]]; then
      echo "$base/$ENV_NAME/bin/python"
      return
    fi
  done
  echo ""
}

PY="$(resolve_python)"
if [[ -z "$PY" ]]; then
  cat >&2 <<EOF
[run_publisher] No publisher interpreter found.

Create one (Newton/Warp + CUDA, XRoboToolkit SDK):
  bash $VENDOR_ROOT/deployment/holomotion_teleop/setup_holomotion_teleop_x86_ubuntu2204.sh
or point at an existing interpreter:
  HOLOMOTION_PY=/path/to/python $0 "\$@"
EOF
  exit 1
fi

# Vendored layout: holoretarget lives next to deployment/
export PYTHONPATH="$VENDOR_ROOT${PYTHONPATH:+:$PYTHONPATH}"

ARGS=("$@")
if [[ ${#ARGS[@]} -eq 0 ]]; then
  ARGS=(--robot-zmq-uri "tcp://*:6001" --robot-zmq-mode bind --hz 50)
fi

echo "[run_publisher] python : $PY"
echo "[run_publisher] node   : $NODE"
echo "[run_publisher] args   : ${ARGS[*]}"
exec "$PY" "$NODE" "${ARGS[@]}"
