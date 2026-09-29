#!/usr/bin/env bash
# Sim teleop with the HoloMotion v1.4 controller (teleop-collection branch) in one command:
# starts the reference publisher (holomotion_teleop conda env) in the background and the SIMPLE teleop CLI in the
# foreground; the publisher is stopped when the CLI exits.
#
#   scripts/teleop_holomotion_v14.sh --scene bottle_bin --record          # PICO headset + controllers
#   PUBLISHER_ARGS="--source synthetic --press 1:L3 --press 5:A --stick 8:12:left_y=0.5" \
#       scripts/teleop_holomotion_v14.sh --scene bowl_sink --headless               # no headset: scripted input
#
# Env: PUBLISHER_ARGS (default "--source pico"), HOLOMOTION_TELEOP_PY (publisher python), SIMPLE_PY (SIMPLE python),
#      REFERENCE_PORT (default 6001). Everything else is passed to `python -m simple.cli.teleop_holomotion_v14`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUB_PY="${HOLOMOTION_TELEOP_PY:-$HOME/miniconda3/envs/holomotion_teleop/bin/python}"
SIM_PY="${SIMPLE_PY:-$ROOT/.venv/bin/python}"
PORT="${REFERENCE_PORT:-6001}"
PUB_ARGS="${PUBLISHER_ARGS:---source pico}"
LOG="${TMPDIR:-/tmp}/holomotion_v14_publisher.log"

[[ -x "$PUB_PY" ]] || { echo "publisher python not found: $PUB_PY (set HOLOMOTION_TELEOP_PY)" >&2; exit 1; }
[[ -x "$SIM_PY" ]] || { echo "SIMPLE python not found: $SIM_PY (set SIMPLE_PY)" >&2; exit 1; }

# shellcheck disable=SC2086
"$PUB_PY" -u "$ROOT/src/simple/teleop/holomotion_v14/publisher.py" --uri "tcp://*:$PORT" $PUB_ARGS > "$LOG" 2>&1 &
PUB_PID=$!
trap 'kill $PUB_PID 2>/dev/null || true' EXIT INT TERM
echo "[teleop_holomotion_v14] publisher pid $PUB_PID ($PUB_ARGS), log $LOG"

cd "$ROOT"
"$SIM_PY" -u -m simple.cli.teleop_holomotion_v14 --reference-uri "tcp://127.0.0.1:$PORT" "$@"
