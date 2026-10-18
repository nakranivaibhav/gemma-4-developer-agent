#!/usr/bin/env bash
# Benchmark another Gemma 4 size end to end, unattended:
#   1. wait until the model download has finished (all *.safetensors adding up to the expected size, no .kaggle-partial left)
#   2. speed matrix (bench_matrix.sh variant): plain vs ngram speculative decoding, 1-16 simulated agents at once
#   3. pick the setting: the fewest agents within 3% of the best turns/min (no errors), at most MAX_PAR
#   4. two passes of the lean agent for this size over the tune tasks at that setting (run_round.sh)
# usage: bash kits/swe-agent/bench/variant_pipeline.sh <tag> <model_dir> <expected_safetensors_bytes> [model_extra] [task_list]
# env: RUN_PREFIX (default v_<tag>_lean), PICK_FORCE="<plain|ngram> <agents>" skips the speed matrix (setting already measured)
# Launch it detached through tools/run_with_watchdog.py (CLAUDE.md "Long runs").
set -u
cd "$(dirname "$0")/../../.."
TAG=$1; MODEL_DIR=$(realpath -m "$2"); WANT=$3; export MODEL_EXTRA=${4:-${MODEL_EXTRA:-}}; TASKS=${5:-tune}
PREFIX=${RUN_PREFIX:-v_${TAG}_lean}
# the 12B checkpoint declares an architecture this vLLM does not register; serve it as the text-only class
[ "$TAG" = 12b ] && [ -z "$MODEL_EXTRA" ] && export MODEL_EXTRA="--hf-overrides '{\"architectures\":[\"Gemma4ForCausalLM\"],\"model_type\":\"gemma4\"}'"
# qwen: reference model, not submittable (Gemma-only comp). Parsers and FP8 KV cache from the nvidia/Qwen3.8-27B-NVFP4 card;
# text only (no vision tower in memory); no LoRA. 21.9 GB of weights leave ~5 GB on this card for KV and the linear-attention state.
if [ "$TAG" = qwen ]; then
  export TOOL_PARSER=qwen3_coder REASONING_PARSER=qwen3 ENABLE_LORA=0
  # vLLM 0.19.1 cannot load this checkpoint (no lm_head.input_scale); envs/qwen has 0.31.0 (2026-10-16). No CUDA toolkit
  # here, so keep FlashInfer (needs nvcc to compile) out of both attention and sampling.
  export VLLM_PY=$PWD/envs/qwen/bin/python VLLM_USE_FLASHINFER_SAMPLER=0
  export VARIANT_LEVELS=${VARIANT_LEVELS:-"1 2 4 8 12 16"}  # 105k KV tokens at 0.84 (about 3 full 32k contexts)
  [ -z "$MODEL_EXTRA" ] && export MODEL_EXTRA="--language-model-only --kv-cache-dtype fp8_e4m3 --attention-backend TRITON_ATTN"
fi
# CPU measured in the 12B round (8 tasks at once, 275 min): 11% busy on average, 36% peak of 12 cores, so up to 24;
# beyond that, test bursts would start to slow the 300 s command limit. The round still logs CPU every minute.
MAX_PAR=${MAX_PAR:-24}
# 0.88 left too little room on this 32 GB GPU (desktop holds ~2.2 GB): 12B ngram bench and 12B run server OOMed at startup, 2026-10-15
export GPU_UTIL=${GPU_UTIL:-0.84}
export MODEL_DIR TAG BENCH_AGENT=agents/lean_${TAG} SERVED=$MODEL_DIR
mkdir -p runs/logs/bench runs/bench

echo "WAIT_DOWNLOAD $TAG $(date -u +%H:%MZ)"
i=0
until [ -f "$MODEL_DIR/config.json" ] && [ "$(cat "$MODEL_DIR"/*.safetensors 2>/dev/null | wc -c)" = "$WANT" ] \
      && ! ls "$MODEL_DIR"/*.kaggle-partial >/dev/null 2>&1; do
  sleep 30; i=$((i + 1)); [ $((i % 20)) = 0 ] && echo "still downloading $(du -sh "$MODEL_DIR" | cut -f1) $(date -u +%H:%MZ)"
done
sleep 60  # let the unpacker finish writing the small files
echo "DOWNLOAD_OK $TAG $(date -u +%H:%MZ)"

if [ -n "${PICK_FORCE:-}" ]; then
  PICK="$PICK_FORCE forced forced"
else
echo "MATRIX_START $TAG $(date -u +%H:%MZ)"
bash kits/swe-agent/bench/bench_matrix.sh variant
PICK=$(envs/harness/bin/python -I - "$TAG" "$MAX_PAR" <<'EOF'
import json, sys
tag, cap = sys.argv[1], int(sys.argv[2])
rows = []
for cfg in ("plain", "ngram"):
    try:
        d = json.load(open(f"runs/bench/{tag}_{cfg}.json"))
    except OSError:
        print(f"BENCH_MISSING {cfg}", file=sys.stderr)
        continue
    rows += [(cfg, l["agents"], l["turns_per_min"]) for l in d["levels"] if not l["errors"] and l["agents"] <= cap]
best = max(t for *_, t in rows)
cfg, n, t = min((r for r in rows if r[2] >= 0.97 * best), key=lambda r: (r[1], -r[2]))
print(cfg, n, t, best)
EOF
)
fi
[ -n "$PICK" ] || { echo "PICK_FAILED"; exit 3; }
read CFG PAR TPM BEST <<< "$PICK"
echo "PICK $TAG config=$CFG concurrency=$PAR turns_per_min=$TPM best=$BEST $(date -u +%H:%MZ)"

[ "$CFG" = plain ] && export SPEC="" || unset SPEC
export MAX_NUM_SEQS=$(( PAR > 8 ? PAR : 8 )) CONCURRENCY=$PAR
# CPU log for the round, one row a minute: whole-machine busy %, 1-min load, running task containers
CPU=runs/${PREFIX}_cpu.csv; echo "utc,cpu_busy_pct,load1,containers" > "$CPU"
( read -r _ a b c d e f g _ < /proc/stat; t0=$((a+b+c+d+e+f+g)); i0=$((d+e))
  while sleep 60; do
    read -r _ a b c d e f g _ < /proc/stat; t1=$((a+b+c+d+e+f+g)); i1=$((d+e))
    echo "$(date -u +%H:%MZ),$(( 100 * ((t1-t0) - (i1-i0)) / (t1-t0) )),$(cut -d' ' -f1 /proc/loadavg),$(docker ps -q | wc -l)" >> "$CPU"
    t0=$t1; i0=$i1
  done ) &
CPU_PID=$!
bash kits/swe-agent/run/run_round.sh "agents/lean_${TAG}" "$PREFIX" "$TASKS"
RC=$?
kill $CPU_PID 2>/dev/null
SP=$(ps -eo pid,args | grep '[v]llm.entrypoints' | awk '{print $1}')
[ -n "$SP" ] && kill -INT $SP
echo "PIPELINE_DONE $TAG rc=$RC $(date -u +%H:%MZ)"
exit $RC
