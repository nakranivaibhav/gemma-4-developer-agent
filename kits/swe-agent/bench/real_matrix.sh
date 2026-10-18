#!/usr/bin/env bash
# Real tasks in parallel under several server configurations: start the local server, run the same task list
# at each concurrency, stop the server. usage: [ROUND=1|2] bash kits/swe-agent/bench/real_matrix.sh <task_id> [...]
# Results: runs/real_<name>_c<N>/, server logs: runs/logs/bench/real_<name>_serve.log
# A marker /tmp/gemma-4-developer-agent_real_<name>_c<N>.done is touched after each run.
set -u
cd "$(dirname "$0")/../../.."
PY=envs/harness/bin/python
TASKS=("$@")
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

run_cfg() {  # name, gpu_util, extra vLLM flags, concurrency levels
  local name=$1 util=$2 extra=$3 levels=$4 pid i n up=0
  echo "=== $name util=$util extra=[$extra] $(date -u +%H:%MZ)"
  MAX_NUM_SEQS=8 GPU_UTIL=$util VLLM_EXTRA="$extra" $PY -I kits/swe-agent/serve/serve_model.py data/external/model agents/sample \
    > runs/logs/bench/real_${name}_serve.log 2>&1 &
  pid=$!
  for i in $(seq 1 90); do
    kill -0 $pid 2>/dev/null || break
    curl -s -m 2 localhost:8000/v1/models >/dev/null 2>&1 && { up=1; break; }
    sleep 5
  done
  if [ $up = 1 ]; then
    for n in $levels; do
      kill -0 $pid 2>/dev/null || { echo "$name SERVER_DIED before c$n"; break; }
      echo "--- $name concurrency=$n $(date -u +%H:%MZ)"
      bash kits/swe-agent/run/run_parallel.sh $n real_${name}_c$n "${TASKS[@]}"
      touch /tmp/gemma-4-developer-agent_real_${name}_c$n.done
    done
  else
    echo "$name FAILED_TO_START"
    grep -E 'Error|OOM|out of memory|not supported' runs/logs/bench/real_${name}_serve.log | tail -4 | cut -c1-300
  fi
  kill -INT $pid 2>/dev/null
  for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done
  kill -KILL $pid 2>/dev/null
  for i in $(seq 1 30); do [ "$(gpu_used)" -lt 2000 ] && break; sleep 2; done
  docker ps -q --filter ancestor=swebench-sandbox:latest | xargs -r docker rm -f >/dev/null 2>&1
  echo "$name stopped gpu_used_mib=$(gpu_used)"
}

NGRAM="--speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}'"
if [ "${ROUND:-1}" = 1 ]; then    # do real tasks keep up with the simulated gain?
  run_cfg u094    0.94 ""       "3 4"
  run_cfg ngram88 0.88 "$NGRAM" "2"
else                              # push each setup further, then the one-at-a-time reference on the same tasks
  run_cfg ngram88 0.88 "$NGRAM" "3 4"
  run_cfg u094    0.94 ""       "6 1"
fi
echo "REAL_MATRIX_DONE $(date -u +%H:%MZ)"
