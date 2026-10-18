#!/usr/bin/env bash
# Per-task pass-count histogram: N independent samples of one agent over a task list, all running side by side
# against ONE model server so the GPU stays fed (no per-run tail). Each sample is its own run_tasks process with
# its own run id <prefix>_s<k> and CONCURRENCY = TOTAL / N. ADK does not forward a seed to the server, so the
# samples are independent random draws at the agent's temperature.
# usage: bash kits/swe-agent/run/hist_runs.sh <agent_dir> <prefix> <task_list> [N=8] [TOTAL=128]
# env: MODEL_DIR, SPEC ("" = plain decoding, unset = ngram), GPU_UTIL. Launch it detached under the watchdog.
set -u
cd "$(dirname "$0")/../../.."
AGENT=$1; PREFIX=$2; TASKS=$3; N=${4:-8}; TOTAL=${5:-128}
PER=$(( TOTAL / N ))
# stop any leftover server first (a stranded one hijacks port 8000)
sp=$(ps -eo pid,args | grep '[v]llm.entrypoints' | awk '{print $1}')
[ -n "$sp" ] && { echo "CLEAN stopping vLLM $sp"; kill -INT $sp; for i in $(seq 60); do ps -p $sp >/dev/null 2>&1 || break; sleep 2; done; }
MAX_NUM_SEQS=$TOTAL setsid nohup bash kits/swe-agent/serve/serve_fast.sh "$AGENT" > "runs/logs/serve_${PREFIX}.log" 2>&1 < /dev/null &
SPID=$!
trap 'kill -INT $(ps -eo pid,args | grep "[v]llm.entrypoints" | awk "{print \$1}") 2>/dev/null; kill $PXPID $CPU_PID 2>/dev/null' EXIT
for i in $(seq 1 120); do curl -s -m 2 127.0.0.1:8000/v1/models >/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "SERVER_DIED see runs/logs/serve_${PREFIX}.log"; exit 2; }; sleep 5; done
curl -s -m 2 127.0.0.1:8000/v1/models >/dev/null || { echo "SERVER_FAILED"; exit 2; }
grep -E 'Maximum concurrency' "runs/logs/serve_${PREFIX}.log" | cut -c40-160
echo "server up $(date -u +%H:%MZ), $N samples x $(envs/harness/bin/python -I kits/swe-agent/run/manifest.py ids "$TASKS" | wc -l) tasks, $PER at once each"
OUT0=runs/${PREFIX}; mkdir -p "$OUT0"
curl -s localhost:8000/metrics > "$OUT0/metrics_before.txt"
# one capture proxy for all samples (full request/response pairs for training)
LOG_PROXY_FULL="$OUT0/requests.jsonl.gz" envs/harness/bin/python -I kits/swe-agent/run/log_proxy.py "$OUT0/proxy_settings.jsonl" 8001 8000 > "$OUT0/proxy.log" 2>&1 &
PXPID=$!
for i in $(seq 1 20); do curl -s -m 2 127.0.0.1:8001/v1/models >/dev/null && break; sleep 1; done
PORT=8001; curl -s -m 2 127.0.0.1:8001/v1/models >/dev/null || { echo "PROXY_FAILED, running direct"; PORT=8000; }
# system log every 30 s: CPU busy %, load, RAM GB, GPU util %, GPU MiB, containers
SYS=$OUT0/sys.csv; echo "utc,cpu_busy_pct,load1,mem_used_gb,gpu_util_pct,gpu_mem_mib,containers" > "$SYS"
( read -r _ a b c d e f g _ < /proc/stat; t0=$((a+b+c+d+e+f+g)); i0=$((d+e))
  while sleep 30; do read -r _ a b c d e f g _ < /proc/stat; t1=$((a+b+c+d+e+f+g)); i1=$((d+e))
    echo "$(date -u +%H:%M:%SZ),$(( 100 * ((t1-t0) - (i1-i0)) / (t1-t0) )),$(cut -d' ' -f1 /proc/loadavg),$(free -m | awk '/Mem:/{printf "%.1f", $3/1024}'),$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '),$(timeout 10 docker ps -q | wc -l)" >> "$SYS"
    t0=$t1; i0=$i1; done ) &
CPU_PID=$!
PIDS=()
for k in $(seq 1 $N); do
  RUN=${PREFIX}_s$k; OUT=runs/$RUN; mkdir -p "$OUT"
  # the runner's heartbeat watches <run dir>/requests.jsonl.gz: point it at the shared capture
  ln -sfn "../${PREFIX}/requests.jsonl.gz" "$OUT/requests.jsonl.gz"
  # resume: skip tasks this run id already has a ledger row for
  TODO=$(envs/harness/bin/python -I - "$RUN" "$TASKS" <<'PY'
import json, sys
sys.path.insert(0, "kits/swe-agent/run")
import manifest
run, tasks = sys.argv[1], manifest.ids(sys.argv[2])
done = {json.loads(l)["id"] for l in open("results/task_runs.jsonl") if f'"run_id": "{run}"' in l}
print(" ".join(t for t in tasks if t not in done))
PY
)
  echo "SAMPLE_START $RUN todo=$(echo $TODO | wc -w) $(date -u +%H:%MZ)"
  [ -z "$TODO" ] && continue
  MODEL_PORT=$PORT RUN_ID=$RUN EVAL_TIER=hist EVAL_NOTE="sample $k of $N: $AGENT on $TASKS, $PER at once ($TOTAL total)" CONCURRENCY=$PER \
    envs/harness/bin/python -I kits/swe-agent/run/run_tasks.py "$AGENT" "$OUT" $TODO > "$OUT/run.log" 2>&1 &
  PIDS+=($!)
  sleep 20  # stagger the starts so 128 sandboxes do not all build in the same second
done
# relay every runner's heartbeat and results into this log, so the watchdog sees one live log
tail -q -F runs/${PREFIX}_s*/run.log 2>/dev/null | grep --line-buffered -E '^(HEARTBEAT|ALL_DONE|Traceback)|^RESULT' | cut -c1-160 &
TAILPID=$!
RC=0; for p in "${PIDS[@]}"; do wait $p || RC=$?; done
kill $TAILPID 2>/dev/null
curl -s localhost:8000/metrics > "$OUT0/metrics_after.txt"
echo "HIST_DONE $PREFIX rc=$RC $(date -u +%H:%MZ)"
exit $RC
