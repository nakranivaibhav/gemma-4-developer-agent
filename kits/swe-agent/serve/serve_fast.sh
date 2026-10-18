#!/usr/bin/env bash
# The measured-fastest local server setup: ngram speculative decoding at GPU memory 0.88. Run tasks 3 at a time against it.
# Measured on 12 real tasks: 19.8 tool calls/min and 38.3 tasks/hour, against 6.8 and 13.0 one at a time on the plain server.
# 4 at a time overflows the 7.11 GiB cache at this setting and is slower (12.1 calls/min).
# usage: bash kits/swe-agent/serve/serve_fast.sh [agent_dir]      then: make run (kits/swe-agent/Makefile)
set -u
cd "$(dirname "$0")/../../.."
NGRAM="--speculative-config '{\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}'"
# MODEL_DIR / MODEL_EXTRA serve another variant, e.g. the 12B: MODEL_DIR=data/external/models/gemma-4-12b-it-qat-w4a16-ct
# MODEL_EXTRA="--hf-overrides '{\"architectures\":[\"Gemma4ForCausalLM\"],\"model_type\":\"gemma4\"}'" (run_tasks.py reads the same MODEL_DIR)
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8} GPU_UTIL=${GPU_UTIL:-0.88} VLLM_EXTRA="${SPEC-$NGRAM} ${MODEL_EXTRA:-}" exec envs/harness/bin/python -I kits/swe-agent/serve/serve_model.py "${MODEL_DIR:-data/external/model}" "${1:-agents/sample}"
