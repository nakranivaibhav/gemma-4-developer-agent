#!/usr/bin/env bash
# Run variant_pipeline.sh for several model sizes back to back, after an earlier run's marker appears.
# usage: bash kits/swe-agent/bench/variant_chain.sh <wait_marker> <tag>:<expected_bytes (all *.safetensors)>[:<task_list>[:<run_prefix>[:<cfg>/<agents>]]] ...
#   task_list   default tune (a manifest selector)
#   run_prefix  default v_<tag>_lean (its pipeline log is runs/<run_prefix>_pipeline.log)
#   cfg/agents  e.g. ngram/8: skip the speed matrix and use this setting (already measured for this size)
# Each step runs under tools/run_with_watchdog.py with its own log and marker (/tmp/gemma_<run_prefix>.done).
# Launch this script itself setsid-detached; it only waits and sequences.
set -u
cd "$(dirname "$0")/../../.."
R=$(cd ../.. && pwd)
WAIT=$1; shift
echo "CHAIN_WAIT $WAIT $(date -u +%H:%MZ)"
until [ -f "$WAIT" ]; do sleep 30; done
sleep 60  # let the previous vLLM server release the GPU
# A step killed by its watchdog leaves its vLLM server (own session) and task sandboxes behind; the next step's
# server then cannot start and its bench hits the old model (404s, 2026-10-15 18:38Z). Clear both first.
clean_gpu() {
  local sp; sp=$(ps -eo pid,args | grep '[v]llm.entrypoints' | awk '{print $1}')
  if [ -n "$sp" ]; then echo "CLEAN stopping leftover vLLM $sp"; kill -INT $sp
    for i in $(seq 1 60); do ps -p $sp >/dev/null 2>&1 || break; sleep 2; done; kill -9 $sp 2>/dev/null; fi
  # timeout: container 5f8fac7cbb18 (2026-10-16) ignores rm/kill and hung the chain; skip any that stays stuck
  local c; c=$(timeout 20 docker ps -q --filter ancestor=swebench-sandbox-local:latest)
  [ -n "$c" ] && echo "CLEAN removing $(echo $c | wc -w) leftover sandboxes" && { timeout 60 docker rm -f $c >/dev/null || echo "CLEAN rm timed out, left: $(timeout 20 docker ps -q --filter ancestor=swebench-sandbox-local:latest | tr '\n' ' ')"; }
  sleep 10
}
for spec in "$@"; do
  clean_gpu
  IFS=: read -r tag bytes tasks prefix pick <<< "$spec"
  dir=data/external/models/gemma-4-${tag}-it-qat-w4a16-ct
  [ "$tag" = qwen ] && dir=data/external/models/qwen3.8-27b-nvfp4  # reference model, see variant_pipeline.sh
  prefix=${prefix:-v_${tag}_lean}
  log=runs/${prefix}_pipeline.log; done_m=/tmp/gemma_${prefix}.done; rm -f "$done_m"
  echo "CHAIN_START $tag $prefix ${tasks:-tune} ${pick:-matrix} $(date -u +%H:%MZ)"
  RUN_PREFIX=$prefix PICK_FORCE=${pick/\// } uv run --project "$R" "$R/tools/run_with_watchdog.py" --log "$log" --marker "$done_m" \
    --idle-seconds 1800 -- bash kits/swe-agent/bench/variant_pipeline.sh "$tag" "$dir" "$bytes" "" "${tasks:-tune}"
  echo "CHAIN_DONE $tag rc=$? $(date -u +%H:%MZ)"
  sleep 60
done
echo "CHAIN_ALL_DONE $(date -u +%H:%MZ)"
