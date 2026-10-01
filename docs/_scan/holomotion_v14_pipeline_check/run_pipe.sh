#!/usr/bin/env bash
# The HoloMotion v1.4 pipeline check: the three HoloBrain deploy presets in the loop of simple.cli.eval_holomotion_v14, one
# task after another (server -> bridge -> eval -> bit-exact replay check), then the repeat check, then checks.json.
#   bash run_pipe.sh <pipe dir>      env: TASKS="bottle_bin bowl_sink coffee_cart" EPISODES=5 STEPS=1500 SIM_MODE=simple REPEAT=1
# Rendering: SIM_MODE=simple = SIMPLE's standard Isaac rendering (the default of eval_holomotion_v14); mujoco = the old views.
set -uo pipefail
PIPE=$(mkdir -p "${1:?pipe dir}" && cd "$1" && pwd)
SIMPLE=$HOME/wrk/SIMPLE; PKG=$HOME/wrk/robot_orchard_deploy/holobrain_g1_deploy; PY=$SIMPLE/.venv/bin/python
EPISODES=${EPISODES:-5}; STEPS=${STEPS:-1500}; SIM_MODE=${SIM_MODE:-simple}; REPEAT=${REPEAT:-1}
declare -A PRESET=([bottle_bin]=chipcan_nativec9 [bowl_sink]=bowltosink_c9 [coffee_cart]=cart_c19)
declare -A SPORT=([bottle_bin]=8014 [bowl_sink]=8015 [coffee_cart]=8016)
declare -A BPORT=([bottle_bin]=21000 [bowl_sink]=21001 [coffee_cart]=21002)
export PATH="$HOME/wrk/simple_validate/.ffbin:$PATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMNI_KIT_ACCEPT_EULA=Y SIMPLE_DISABLE_TUI=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HOLOBRAIN_G1_DEPLOY=$PKG SIMPLE_DETERMINISTIC_RUNTIME=1
cd "$SIMPLE"
wait_http() { local url=$1 pid=$2 n=${3:-300}; for _ in $(seq 1 "$n"); do sleep 2; kill -0 "$pid" 2>/dev/null || return 1; curl -s -m 2 "$url" >/dev/null 2>&1 && return 0; done; return 1; }
stop_group() { local pid=$1; kill -0 "$pid" 2>/dev/null || return 0; kill -TERM -- "-$pid" 2>/dev/null; for _ in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || return 0; sleep 1; done; kill -KILL -- "-$pid" 2>/dev/null; }
# the model server's python is not in serve.sh's process group and ignores TERM: kill whatever still listens on the port
stop_port() { local port=$1; for _ in 1 2 3; do local pids; pids=$(ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p"$" {print $NF}' | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); [ -z "$pids" ] && return 0; kill -KILL $pids 2>/dev/null; sleep 1; done; }
echo "##### pipe started $(date)  tasks=${TASKS:-bottle_bin bowl_sink coffee_cart} episodes=$EPISODES steps=$STEPS sim-mode=$SIM_MODE -> $PIPE"
for task in ${TASKS:-bottle_bin bowl_sink coffee_cart}; do
    preset=${PRESET[$task]}; sp=${SPORT[$task]}; bp=${BPORT[$task]}
    echo "=== $(date '+%F %T') TASK START $task ($preset) server :$sp bridge :$bp"
    if curl -s -m 2 "http://localhost:$sp/health" >/dev/null 2>&1; then echo "   something already serves :$sp; stop it first"; continue; fi
    (cd "$PKG" && env PORT=$sp setsid nohup bash scripts/serve.sh "$preset" --steps 8 --replan 15 > "$PIPE/server_$task.log" 2>&1 &
     echo $! > "$PIPE/server_$task.pid")
    spid=$(cat "$PIPE/server_$task.pid")
    wait_http "http://localhost:$sp/health" "$spid" 300 || { echo "   server :$sp not up"; tail -5 "$PIPE/server_$task.log"; stop_group "$spid"; continue; }
    curl -s -m 3 "http://localhost:$sp/health" > "$PIPE/health_$task.json"
    setsid nohup "$PY" scripts/holomotion_v14_vla_bridge.py --upstream-port "$sp" --port "$bp" --preset "$preset" \
        --log "$PIPE/bridge_$task.jsonl" > "$PIPE/bridge_$task.log" 2>&1 &
    bpid=$!; echo $bpid > "$PIPE/bridge_$task.pid"
    wait_http "http://localhost:$bp/info" "$bpid" 60 || { echo "   bridge :$bp not up"; tail -5 "$PIPE/bridge_$task.log"; stop_group "$bpid"; stop_group "$spid"; continue; }
    "$PY" -u -m simple.cli.eval_holomotion_v14 --scene "$task" --port "$bp" --image-size 640x360 --num-episodes "$EPISODES" \
        --max-episode-steps "$STEPS" --eval-dir "$PIPE/eval_$task" --sim-mode "$SIM_MODE" > "$PIPE/eval_$task.log" 2>&1
    rc=$?
    run=$(ls -d "$PIPE/eval_$task"/vla/*/dr-level-3 2>/dev/null | head -1)
    echo "   eval rc=$rc run=$run"
    if [ -n "$run" ]; then
        "$PY" -u -m simple.cli.replay_holomotion_v14 "$run" --all --mode action > "$PIPE/replay_$task.log" 2>&1
        echo "   replay: $(grep -h 'episodes bit-exact' "$PIPE/replay_$task.log" | tail -1)"
    fi
    if [ "$REPEAT" = 1 ] && [ "$task" = bottle_bin ]; then        # the same scene twice through the live model
        for ab in a b; do
            "$PY" -u -m simple.cli.eval_holomotion_v14 --scene "$task" --port "$bp" --image-size 640x360 --num-episodes 1 \
                --max-episode-steps 500 --eval-dir "$PIPE/repeat_$ab" --sim-mode "$SIM_MODE" --no-save-video > "$PIPE/repeat_$ab.log" 2>&1
        done
        echo "   repeat runs done"
    fi
    stop_group "$bpid"; stop_group "$spid"; stop_port "$bp"; stop_port "$sp"
    echo "=== $(date '+%F %T') TASK DONE $task rc=$rc (ports :$sp :$bp freed: $(ss -ltn 2>/dev/null | grep -cE ":$sp |:$bp " | sed 's/^0$/yes/'))"
done
"$PY" "$SIMPLE/docs/_scan/holomotion_v14_pipeline_check/make_checks.py" "$PIPE" && echo "   checks.json written"
echo "##### pipe finished $(date)"
