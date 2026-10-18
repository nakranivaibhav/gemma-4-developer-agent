#!/usr/bin/env bash
# Speed matrix: for each server configuration, start the local server, run bench_multi.py at several
# agent counts, stop the server. usage: bash kits/swe-agent/bench/bench_matrix.sh <round1|round2>
# Results: runs/bench/<name>.json, server logs: runs/logs/bench/<name>_serve.log
set -u
cd "$(dirname "$0")/../../.."
PY=envs/harness/bin/python
ROUND="${1:-round1}"
gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1; }

run_cfg() {  # name, gpu_util, extra vLLM flags, agent counts, extra environment (optional)
  local name=$1 util=$2 extra=$3 levels=$4 envs=${5:-} pid i up=0
  echo "=== $name util=$util extra=[$extra] $(date -u +%H:%MZ)"
  env $envs MAX_NUM_SEQS=${MAX_NUM_SEQS:-8} GPU_UTIL=$util VLLM_EXTRA="$extra ${MODEL_EXTRA:-}" $PY -I kits/swe-agent/serve/serve_model.py "${MODEL_DIR:-data/external/model}" "${BENCH_AGENT:-agents/sample}" \
    > runs/logs/bench/${name}_serve.log 2>&1 &
  pid=$!
  for i in $(seq 1 90); do
    kill -0 $pid 2>/dev/null || break
    curl -s -m 2 localhost:8000/v1/models >/dev/null 2>&1 && { up=1; break; }
    sleep 5
  done
  if [ $up = 1 ]; then
    grep -E 'Available KV cache memory|Maximum concurrency' runs/logs/bench/${name}_serve.log | cut -c40-160
    echo "$name gpu_used_mib=$(gpu_used)"
    $PY -I kits/swe-agent/bench/bench_multi.py "${SERVED:-main_lora}" runs/bench/${name}.json $name $levels
  else
    echo "$name FAILED_TO_START"
    grep -E 'Error|OOM|out of memory|not supported' runs/logs/bench/${name}_serve.log | tail -4 | cut -c1-300
  fi
  kill -INT $pid 2>/dev/null
  for i in $(seq 1 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done
  kill -KILL $pid 2>/dev/null
  for i in $(seq 1 30); do [ "$(gpu_used)" -lt 2000 ] && break; sleep 2; done
  echo "$name stopped gpu_used_mib=$(gpu_used)"
}

NGRAM="--speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}'"
if [ "$ROUND" = round1 ]; then  # which lever matters at all
  run_cfg base        0.88 "" "1 2 4 6 8"
  run_cfg util094     0.94 "" "1 2 4 6 8"
  run_cfg offload     0.88 "--kv-offloading-size 40" "1 2 4 6 8"
  run_cfg throughput  0.88 "--performance-mode throughput" "1 2 4 6 8"
  run_cfg fp8kv       0.88 "--kv-cache-dtype fp8" "1 2 4 6 8"
  run_cfg ngram       0.88 "$NGRAM" "1 2 4 6 8"
elif [ "$ROUND" = variant ]; then  # another model size: caller sets MODEL_DIR, MODEL_EXTRA, BENCH_AGENT, SERVED, TAG
  # 24 added 2026-10-15: CPU was 11% busy at 8 tasks, so the GPU is the limit to find
  LV=${VARIANT_LEVELS:-"1 2 4 8 12 16 24"}
  export MAX_NUM_SEQS=${LV##* }
  run_cfg ${TAG}_plain ${GPU_UTIL:-0.88} "" "$LV"
  run_cfg ${TAG}_ngram ${GPU_UTIL:-0.88} "$NGRAM" "$LV"
else  # round 2: more room for the cache without changing the numbers the model computes
  run_cfg u094_n3       0.94 "" "3"
  run_cfg u096          0.96 "" "3 4"
  run_cfg u094_nomm     0.94 "--limit-mm-per-prompt '{\"image\":0,\"audio\":0}'" "3 4"
  run_cfg u094_batch8k  0.94 "--max-num-batched-tokens 8192" "3 4"
  run_cfg u094_cpuspill 0.94 "--kv-offloading-size 40 --no-disable-hybrid-kv-cache-manager" "4 6 8" "VLLM_USE_SIMPLE_KV_OFFLOAD=1"
  run_cfg u094_ngram    0.94 "$NGRAM" "1 2 3 4"
fi
echo "MATRIX_DONE $(date -u +%H:%MZ)"
