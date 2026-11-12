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
# 2026-11-03: --hand-source skeleton needs the headset's hand tracking on the wire (publisher --hands); --auto-cycle / skeleton read the
# keyboard (cbreak mode): restore the terminal when the CLI ends (it exits with os._exit)
if [[ " $* " == *" --hand-source skeleton "* || " $* " == *" --hand-source=skeleton "* ]] && [[ "$PUB_ARGS" != *--hands* ]]; then
  PUB_ARGS="$PUB_ARGS --hands"
fi
if [[ " $* " == *" --hand-source skeleton "* || " $* " == *" --auto-cycle "* ]] && [[ -t 0 ]]; then
  saved_tty="$(stty -g)"
fi

[[ -x "$PUB_PY" ]] || { echo "publisher python not found: $PUB_PY (set HOLOMOTION_TELEOP_PY)" >&2; exit 1; }
[[ -x "$SIM_PY" ]] || { echo "SIMPLE python not found: $SIM_PY (set SIMPLE_PY)" >&2; exit 1; }

# 2026-11-03: a second client of the XRoboToolkit SDK service (127.0.0.1:60061; e.g. scripts/wuji_glove_pilot/bench.py --glove pico)
# attached during a session can change the headset's tracking mode; when the body tracking stops the teleop holds. Say so up front.
if [[ "$PUB_ARGS" == *pico* ]]; then
  # Inspect sockets directly: lsof can hang on an unavailable NFS mount before checking TCP.
  # This advisory check must never block teleop startup, even if socket inspection stalls.
  # (|| true: under set -e -o pipefail an empty grep would otherwise end the script here -- it did, 2026-11-03 22:00)
  OTHER_PIDS="$(timeout --kill-after=1s 2s ss -Htnp state established '( dport = :60061 )' 2>/dev/null |
                   grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u | tr '\n' ' ' || true)"
  if [[ -n "${OTHER_PIDS// /}" ]]; then
    echo "[teleop_holomotion_v14] WARNING: other XRoboToolkit SDK clients are attached to 127.0.0.1:60061 (PIDs: ${OTHER_PIDS}):" >&2
    for p in $OTHER_PIDS; do echo "    $p  $(ps -o args= -p "$p" 2>/dev/null | cut -c1-120)" >&2; done
    echo "    a second client (the glove bench with --glove pico, a viewer) can stop the headset's body tracking; stop it before teleoperating." >&2
  fi
fi

# shellcheck disable=SC2086
"$PUB_PY" -u "$ROOT/src/simple/teleop/holomotion_v14/publisher.py" --uri "tcp://*:$PORT" $PUB_ARGS > "$LOG" 2>&1 &
PUB_PID=$!
trap 'kill $PUB_PID 2>/dev/null || true; [[ -n "${saved_tty:-}" ]] && stty "$saved_tty" 2>/dev/null || true' EXIT INT TERM
echo "[teleop_holomotion_v14] publisher pid $PUB_PID ($PUB_ARGS), log $LOG"

cd "$ROOT"
"$SIM_PY" -u -m simple.cli.teleop_holomotion_v14 --reference-uri "tcp://127.0.0.1:$PORT" "$@"
