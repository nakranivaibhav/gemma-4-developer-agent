#!/usr/bin/env bash
# Time + profile decode against the live server at the shapes we run. Two server launches: plain, then ngram speculative.
# usage: bash kits/swe-agent/bench/kprof/run_all.sh        results in runs/kprof/<cfg>/{timing.json,trace/}
set -u
cd "$(dirname "$0")/../../../.."
PY=envs/harness/bin/python
NGRAM="--speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}'"
PROMPT=8000

serve() {  # serve <name> <extra flags> <gpu_util>
  TRACE=$PWD/runs/kprof/_trace_$1; mkdir -p $TRACE
  PROF="--profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$TRACE\"}'"
  MAX_NUM_SEQS=8 GPU_UTIL=$3 VLLM_EXTRA="$PROF $2" \
    $PY -I kits/swe-agent/serve/serve_model.py data/external/model agents/sample > runs/logs/kprof/serve_$1.log 2>&1 &
  SPID=$!
  for i in $(seq 90); do curl -sf localhost:8000/v1/models >/dev/null && break; sleep 5; done
  curl -sf localhost:8000/v1/models >/dev/null || { echo "SERVER_FAILED $1"; return 1; }
  BASE=$(curl -s localhost:8000/v1/models | $PY -I -c "import json,sys;print([m['id'] for m in json.load(sys.stdin)['data'] if 'lora' not in m['id']][0])")
  echo "server $1 up, base model id: $BASE"
}
stop() {
  kill -INT $SPID 2>/dev/null; sleep 20; kill -KILL $SPID 2>/dev/null
  for i in $(seq 30); do [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ] && break; sleep 5; done
  echo "server stopped gpu_used_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
}
cfg() {  # cfg <name> <num_seqs> <prompt_tokens> <model> [ngram]
  echo "--- $1 $(date -u +%H:%MZ)"
  OUT=runs/kprof/$1; mkdir -p $OUT/trace
  $PY -I kits/swe-agent/bench/kprof/profile_server.py $OUT $2 $3 $4 ${5:-} 2>&1 | grep -aE 'TIMING|PROFILE_DONE|Traceback|Error' | cut -c1-300
  mv $TRACE/* $OUT/trace/ 2>/dev/null
  $PY -I kits/swe-agent/bench/kprof/summarize_trace.py $OUT
}

# plain at util 0.94 and ngram at 0.88, as in the real-task runs; 3 x 8000 tokens does not fit the cache at once, so 3-seq runs use 4000
serve plain "" 0.94 && { cfg b1_base 1 8000 $BASE; cfg b1_lora 1 8000 main_lora; cfg b2_lora 2 8000 main_lora; cfg b3_lora 3 4000 main_lora; }
stop
serve ngram "$NGRAM" 0.88 && { cfg b1_lora_ngram 1 8000 main_lora ngram; cfg b3_lora_ngram 3 4000 main_lora ngram; }
stop
echo "KPROF_DONE $(date -u +%H:%MZ)"
