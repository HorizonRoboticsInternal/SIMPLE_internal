#!/usr/bin/env bash
# Stand-in for the SIMPLE python of scripts/teleop_holomotion_v14.sh (teleop_with_glove.sh points SIMPLE_PY here): runs
# teleop_with_glove.py, which wraps the agent's hand command and then starts the unchanged teleop CLI with the same arguments.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${WUJI_GLOVE_REAL_PY:?run scripts/wuji_glove_pilot/teleop_with_glove.sh}" -u "$HERE/teleop_with_glove.py" "$@"
