#!/usr/bin/env bash
# launch_sims.sh -- launch N SimRoki simulator instances on consecutive ports.
#
# Usage:
#   ./launch_sims.sh [NUM_INSTANCES] [BASE_PORT]
#
# Defaults: 10 instances starting at port 8080.
# Each instance receives SIMROKI_PORT=<port> via the environment.
# A companion script `kill_sims.sh` is generated alongside this file.

set -euo pipefail

NUM=${1:-10}
BASE_PORT=${2:-8080}
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
BINARY="$PROJECT_DIR/target/release/native_app"
PIDFILE_DIR="$PROJECT_DIR/RL/.sim_pids"
LOG_DIR="$PROJECT_DIR/RL/.sim_logs"

# ---------- build release binary if needed ----------
if [[ ! -f "$BINARY" ]]; then
    echo "[launch] Release binary not found. Building ..."
    (cd "$PROJECT_DIR" && cargo build --release -p native_app)
fi

# ---------- prepare dirs ----------
mkdir -p "$PIDFILE_DIR" "$LOG_DIR"

# ---------- launch instances ----------
PIDS=()
for i in $(seq 0 $(( NUM - 1 ))); do
    PORT=$(( BASE_PORT + i ))
    LOG="$LOG_DIR/sim_${PORT}.log"

    echo "[launch] Starting sim on port $PORT ..."
    SIMROKI_PORT=$PORT "$BINARY" > "$LOG" 2>&1 &
    PID=$!
    echo "$PID" > "$PIDFILE_DIR/sim_${PORT}.pid"
    PIDS+=("$PID")
    echo "[launch]   PID=$PID  log=$LOG"
done

# ---------- wait for all instances to be ready ----------
echo ""
echo "[launch] Waiting for all $NUM instances to become healthy ..."

MAX_WAIT=60   # seconds
for i in $(seq 0 $(( NUM - 1 ))); do
    PORT=$(( BASE_PORT + i ))
    URL="http://127.0.0.1:${PORT}/health"
    ELAPSED=0

    while true; do
        if curl -sf "$URL" > /dev/null 2>&1; then
            echo "[launch] Port $PORT ready."
            break
        fi

        if (( ELAPSED >= MAX_WAIT )); then
            echo "[launch] ERROR: port $PORT did not become healthy within ${MAX_WAIT}s."
            echo "[launch] Check log: $LOG_DIR/sim_${PORT}.log"
            exit 1
        fi

        sleep 1
        ELAPSED=$(( ELAPSED + 1 ))
    done
done

echo ""
echo "[launch] All $NUM simulators running (ports ${BASE_PORT}..$(( BASE_PORT + NUM - 1 )))."
echo "[launch] To stop them: bash $PROJECT_DIR/RL/kill_sims.sh"

# ---------- generate kill script ----------
KILL_SCRIPT="$PROJECT_DIR/RL/kill_sims.sh"
cat > "$KILL_SCRIPT" << 'KILLEOF'
#!/usr/bin/env bash
# kill_sims.sh -- stop all SimRoki simulator instances launched by launch_sims.sh
set -euo pipefail

PIDFILE_DIR="$(cd "$(dirname "$0")" && pwd)/.sim_pids"
if [[ ! -d "$PIDFILE_DIR" ]]; then
    echo "[kill] No pid directory found. Nothing to do."
    exit 0
fi

KILLED=0
for pidfile in "$PIDFILE_DIR"/sim_*.pid; do
    [[ -f "$pidfile" ]] || continue
    PID=$(cat "$pidfile")
    PORT=$(basename "$pidfile" .pid | sed 's/sim_//')
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID" 2>/dev/null && echo "[kill] Stopped sim port=$PORT pid=$PID"
        KILLED=$(( KILLED + 1 ))
    else
        echo "[kill] Sim port=$PORT pid=$PID already stopped."
    fi
    rm -f "$pidfile"
done

echo "[kill] Done. Stopped $KILLED instance(s)."
KILLEOF
chmod +x "$KILL_SCRIPT"
