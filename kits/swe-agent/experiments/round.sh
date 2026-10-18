#!/usr/bin/env bash
# One round of the prompt loop: two full passes of one agent folder over a task list, back to back, repaired sandbox
# (run_tasks.py default), 3 at a time, serve_fast. Starts the server if it is not up. No early stop: pass 2 always runs.
# CONCURRENCY (default 3) and MODEL_DIR / MODEL_EXTRA (see serve_fast.sh) pass through to the server and the runner.
# usage: bash kits/swe-agent/experiments/round.sh <agent_dir> <run_prefix> [tasks=tune] [tier=tune_pass]
#   -> runs/<run_prefix>_p1/ and runs/<run_prefix>_p2/, ledger run_ids <run_prefix>_p1 and <run_prefix>_p2
# Launch it detached through tools/run_with_watchdog.py (see CLAUDE.md "Long runs"), never directly in a session.
set -u
cd "$(dirname "$0")/../../.."
AGENT=$1; PREFIX=$2; TASKS=${3:-tune}; TIER=${4:-tune_pass}
[ -f "$AGENT/agent.yaml" ] || { echo "NO_AGENT $AGENT"; exit 2; }
if ! curl -s -m 2 127.0.0.1:8000/v1/models >/dev/null; then
  echo "server down, starting serve_fast $(date -u +%H:%MZ)"
  setsid nohup bash kits/swe-agent/serve/serve_fast.sh "$AGENT" > "runs/logs/serve_${PREFIX}.log" 2>&1 < /dev/null &
  SPID=$!
  for i in $(seq 1 120); do curl -s -m 2 127.0.0.1:8000/v1/models >/dev/null && break
    kill -0 $SPID 2>/dev/null || { echo "SERVER_DIED see runs/logs/serve_${PREFIX}.log"; break; }  # e.g. CUDA OOM at startup
    sleep 10; echo "waiting for server ($i)"; done
  curl -s -m 2 127.0.0.1:8000/v1/models >/dev/null || { echo "SERVER_FAILED"; exit 2; }
fi
echo "server up $(date -u +%H:%MZ)"
RC_ALL=0
for PASS in ${PASSES:-1 2}; do  # PASSES=1: one pass only (parallel sweep, 2026-10-16)
  RUN=${PREFIX}_p${PASS}; OUT=runs/$RUN
  mkdir -p "$OUT"
  [ -f "$OUT/metrics_before.txt" ] || curl -s localhost:8000/metrics > "$OUT/metrics_before.txt"  # first launch only (a relaunch resumes)
  echo "PASS_START $RUN $(date -u +%H:%MZ)"
  # a relaunch with the same prefix resumes: run_tasks.py skips tasks the run id already has a ledger row for
  # CAPTURE=1 (default): requests go through kits/swe-agent/experiments/log_proxy.py, which keeps every chat request and response
  # in full in $OUT/requests.jsonl.gz (exact prompt/completion pairs for training). Falls back to direct if it fails.
  PORT=8000; PXPID=
  if [ "${CAPTURE:-1}" = 1 ]; then
    LOG_PROXY_FULL="$OUT/requests.jsonl.gz" envs/harness/bin/python -I kits/swe-agent/experiments/log_proxy.py "$OUT/proxy_settings.jsonl" 8001 8000 \
      > "$OUT/proxy.log" 2>&1 &
    PXPID=$!
    for i in $(seq 1 20); do curl -s -m 2 127.0.0.1:8001/v1/models >/dev/null && break; sleep 1; done
    if curl -s -m 2 127.0.0.1:8001/v1/models >/dev/null; then PORT=8001; echo "CAPTURE $OUT/requests.jsonl.gz"
    else echo "PROXY_FAILED, running direct"; kill $PXPID 2>/dev/null; PXPID=; fi
  fi
  MODEL_PORT=$PORT RUN_ID=$RUN EVAL_TIER=$TIER EVAL_NOTE="pass $PASS of 2: $AGENT on $TASKS, serve_fast, ${CONCURRENCY:-3} at once" CONCURRENCY=${CONCURRENCY:-3} \
    envs/harness/bin/python -I kits/swe-agent/run/run_tasks.py "$AGENT" "$OUT" $TASKS
  RC=$?
  [ -n "$PXPID" ] && kill $PXPID 2>/dev/null
  curl -s localhost:8000/metrics > "$OUT/metrics_after.txt"
  echo "PASS_DONE $RUN rc=$RC $(date -u +%H:%MZ)"
  [ $RC -ne 0 ] && RC_ALL=$RC
done
echo "ROUND_DONE $PREFIX rc=$RC_ALL $(date -u +%H:%MZ)"
exit $RC_ALL
