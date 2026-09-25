#!/usr/bin/env bash
# Screen recorded episodes in a kit's level scene (MuJoCo physics, no Isaac) to find ones that succeed there:
#   screen_episodes.sh <kit> <jobs> <extra wrapper args...> -- <episodes...>
# Results: levels_logs/screen_<kit>_ep<N>.log (grep task_success / in_bin / gates)
set -u
kit=$1; jobs=$2; shift 2; extra=(); while [ "$1" != "--" ]; do extra+=("$1"); shift; done; shift
cd "$(dirname "$0")/.."; export MUJOCO_GL=egl OMNI_KIT_ACCEPT_EULA=YES
L=scenes/levels_logs; W=scenes/replay_isaac.py; mkdir -p $L
run() { .venv/bin/python $W $kit --episode $1 --sim-mode mujoco --no-third "${extra[@]}" > $L/screen_${kit}_ep$1.log 2>&1; echo "ep $1 done: $(grep -a -o '"task_success": [a-z]*\|"placed": [a-z]*\|"in_bin": [a-z]*' $L/screen_${kit}_ep$1.log | tail -2 | tr '\n' ' ')"; }
n=0
for ep in "$@"; do run $ep & n=$((n+1)); if [ $n -ge $jobs ]; then wait -n; n=$((n-1)); fi; done; wait
echo "SCREEN DONE $kit"
