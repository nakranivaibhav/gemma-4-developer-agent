#!/usr/bin/env bash
# Run a task list against a REMOTE model server reached through an SSH tunnel; sandboxes stay on this machine.
# usage: run_remote.sh <run_id> <tasks_at_once> <task_list_file> <server_json> [agent_dir] [local_port]
# Saves the server's /metrics before and after next to the results. The ledger pin comes from <server_json>.
set -u
cd "$(dirname "$0")/../../.."
RUN=$1; N=$2; LIST=$3; SJ=$4; AGENT=${5:-agents/out8k}; PORT=${6:-8001}
OUT=runs/$RUN; mkdir -p "$OUT"
curl -s -m 20 127.0.0.1:$PORT/v1/models >/dev/null || { echo "REMOTE_SERVER_DOWN port $PORT"; exit 3; }
curl -s -m 20 127.0.0.1:$PORT/metrics > "$OUT/metrics_before.txt"
T0=$(date +%s); echo "RUN_START $RUN n=$N $(date -u +%Y-%m-%dT%H:%M:%SZ)"
MODEL_PORT=$PORT SERVER_JSON=$SJ RUN_ID=$RUN EVAL_TIER=${EVAL_TIER:-smoke} EVAL_NOTE="${EVAL_NOTE:-remote server run}" CONCURRENCY=$N \
  envs/harness/bin/python -I kits/swe-agent/run/run_tasks.py "$AGENT" "$OUT" "$LIST"
echo "RUN_RC $?"
curl -s -m 20 127.0.0.1:$PORT/metrics > "$OUT/metrics_after.txt"
echo "RUN_DONE $RUN seconds=$(( $(date +%s) - T0 )) $(date -u +%Y-%m-%dT%H:%M:%SZ)"
