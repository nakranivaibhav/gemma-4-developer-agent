#!/usr/bin/env bash
# hist.sh against a REMOTE model server (a rented pod reached through an SSH tunnel); sandboxes stay on this machine.
# N samples of one agent over a task list, side by side, each its own run_tasks process with run id <prefix>_<tag><k>
# and CONCURRENCY = TOTAL / N. No local server is started or stopped.
# usage: bash kits/swe-agent/experiments/hist_remote.sh <agent_dir> <prefix> <task_list> <N> <TOTAL> <tunnel_port> <proxy_port>
#   tunnel_port  local end of the SSH tunnel to the pod's vLLM (e.g. ssh -N -L 8011:127.0.0.1:8000 root@pod -p ...)
#   proxy_port   local port for the request-capture proxy (full request/response pairs for training)
# env: MODEL_DIR (the model path the pod serves under, same absolute path as here), SERVER_JSON (server note pinned into
#      the ledger), SAMPLE_TAG (default p), RUN_LABEL, KAGGLE_CLOCK / KAGGLE_CLOCK_SPEED / KAGGLE_CLOCK_CAP_S (the clock).
# Launch it detached under the watchdog. A relaunch with the same prefix resumes (finished tasks are skipped).
set -u
cd "$(dirname "$0")/../../.."
AGENT=$1; PREFIX=$2; TASKS=$3; N=$4; TOTAL=$5; TPORT=$6; PXPORT=$7
PER=$(( TOTAL / N ))
curl -s -m 20 127.0.0.1:$TPORT/v1/models >/dev/null || { echo "REMOTE_SERVER_DOWN port $TPORT"; exit 3; }
echo "remote server up on :$TPORT $(date -u +%H:%MZ), $N samples x $(envs/harness/bin/python -I kits/swe-agent/run/manifest.py ids "$TASKS" | wc -l) tasks, $PER at once each"
OUT0=runs/${PREFIX}; mkdir -p "$OUT0"
curl -s -m 20 127.0.0.1:$TPORT/metrics > "$OUT0/metrics_before.txt"
LOG_PROXY_FULL="$OUT0/requests.jsonl.gz" envs/harness/bin/python -I kits/swe-agent/experiments/log_proxy.py "$OUT0/proxy_settings.jsonl" $PXPORT $TPORT > "$OUT0/proxy.log" 2>&1 &
PXPID=$!
trap 'kill $PXPID $MPID 2>/dev/null' EXIT
for i in $(seq 1 20); do curl -s -m 2 127.0.0.1:$PXPORT/v1/models >/dev/null && break; sleep 1; done
PORT=$PXPORT; curl -s -m 5 127.0.0.1:$PXPORT/v1/models >/dev/null || { echo "PROXY_FAILED, running direct"; PORT=$TPORT; }
# server counters every 60 s (running/waiting requests, KV cache use, generated tokens): the concurrency check
( while sleep 60; do echo "$(date -u +%H:%M:%SZ) $(curl -s -m 10 127.0.0.1:$TPORT/metrics | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|gpu_cache_usage_perc|generation_tokens_total|num_preemptions_total)' | awk '{printf "%s=%s ", $1, $2}')"; done > "$OUT0/server_metrics.log" ) &
MPID=$!
PIDS=()
for k in $(seq 1 $N); do
  RUN=${PREFIX}_${SAMPLE_TAG:-p}$k; OUT=runs/$RUN; mkdir -p "$OUT"
  ln -sfn "../${PREFIX}/requests.jsonl.gz" "$OUT/requests.jsonl.gz"  # the runner's heartbeat watches this file
  echo "SAMPLE_START $RUN $(date -u +%H:%MZ)"
  MODEL_PORT=$PORT RUN_ID=$RUN EVAL_TIER=hist EVAL_NOTE="${RUN_LABEL:+$RUN_LABEL; }remote sample $k of $N: $AGENT on $TASKS, $PER at once ($TOTAL total)" CONCURRENCY=$PER \
    envs/harness/bin/python -I kits/swe-agent/run/run_tasks.py "$AGENT" "$OUT" $TASKS > "$OUT/run.log" 2>&1 &
  PIDS+=($!)
  sleep 20  # stagger the starts so the sandboxes do not all build in the same second
done
tail -q -F --pid=$$ runs/${PREFIX}_${SAMPLE_TAG:-p}*/run.log 2>/dev/null | grep --line-buffered -E '^(HEARTBEAT|ALL_DONE|Traceback)|^RESULT' | awk '{print substr($0, 1, 160); fflush()}' &  # awk+fflush, not cut: cut block-buffers into a file, so the watchdog saw silence for ~25 lines at a time
TAILPID=$!
RC=0; for p in "${PIDS[@]}"; do wait $p || RC=$?; done
kill $TAILPID 2>/dev/null
curl -s -m 20 127.0.0.1:$TPORT/metrics > "$OUT0/metrics_after.txt"
echo "HIST_DONE $PREFIX rc=$RC $(date -u +%H:%MZ)"
exit $RC
