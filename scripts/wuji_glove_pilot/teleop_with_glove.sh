#!/usr/bin/env bash
# Wuji Glove pilot, stage 2: scripts/teleop_holomotion_v14.sh (reference publisher + teleop CLI, both unchanged) with the
# gloved hand's Revo 2 target taken from the glove. The other hand keeps the PICO grip / trigger poses.
#
#   scripts/wuji_glove_pilot/teleop_with_glove.sh [--glove wuji|synthetic|replay:<log>] [--glove-side right|left|left,right] [--keys] -- <teleop args>
#   scripts/wuji_glove_pilot/teleop_with_glove.sh --glove wuji --glove-side right -- --scene kitchen --robot revo2 --record --save-dir data/<run>
#   scripts/wuji_glove_pilot/teleop_with_glove.sh --glove-side left,right --keys -- --scene kitchen --robot revo2 --record ...
#
# Defaults: --glove wuji, --glove-side right. --keys: the keyboard in this terminal stands in for the PICO controllers'
# buttons and sticks (both hands gloved; see pico_keys.py). Everything after "--" goes to the teleop CLI; the publisher's own
# settings come from the variables scripts/teleop_holomotion_v14.sh reads (PUBLISHER_ARGS, REFERENCE_PORT, ...). More glove
# settings: the WUJI_GLOVE_* variables listed in teleop_with_glove.py.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export WUJI_GLOVE="${WUJI_GLOVE:-wuji}" WUJI_GLOVE_SIDES="${WUJI_GLOVE_SIDES:-right}" WUJI_GLOVE_KEYS="${WUJI_GLOVE_KEYS:-0}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --glove) WUJI_GLOVE="$2"; shift 2 ;;
    --glove-side|--glove-sides) WUJI_GLOVE_SIDES="$2"; shift 2 ;;
    --keys) WUJI_GLOVE_KEYS=1; shift ;;
    --) shift; break ;;
    *) break ;;
  esac
done
export WUJI_GLOVE_REAL_PY="${SIMPLE_PY:-$ROOT/.venv/bin/python}"
echo "[teleop_with_glove] glove $WUJI_GLOVE on $WUJI_GLOVE_SIDES$([[ "$WUJI_GLOVE_KEYS" == 1 ]] && echo ', keyboard for the PICO buttons')"
if [[ "$WUJI_GLOVE_KEYS" == 1 && -t 0 ]]; then        # --keys puts the terminal in cbreak mode and the CLI ends with os._exit: restore it here
  saved_tty="$(stty -g)"; trap 'stty "$saved_tty" 2>/dev/null || true' EXIT
fi
SIMPLE_PY="$HERE/_sim_python.sh" "$ROOT/scripts/teleop_holomotion_v14.sh" "$@"
