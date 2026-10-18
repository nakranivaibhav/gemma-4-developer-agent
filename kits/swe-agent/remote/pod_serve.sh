#!/usr/bin/env bash
# Runs ON the rented pod: start vLLM with the local server's flags (kits/swe-agent/serve/serve_model.py --print), listening on the pod's own
# 127.0.0.1 only. Reach it from the local machine through an SSH tunnel.
# usage: LOCAL_REPO=<repo path on your machine> GPU_UTIL=0.92 MAX_NUM_SEQS=16 [TP=1] [NUM_SPEC=4] [PORT=8000] [GPUS=0,1] [TAG=x] bash pod_serve.sh
#   TP       tensor-parallel size (the model split across TP GPUs, as on Kaggle's 4x L4 with TP=4)
#   NUM_SPEC ngram speculative tokens per step (0 = speculation off)
#   GPUS     CUDA_VISIBLE_DEVICES for this server (one server per GPU: run it once per GPU with its own PORT)
#   TAG      log name suffix: /root/vllm<TAG>.log
#   EXTRA    more vllm flags, e.g. --disable-custom-all-reduce (NCCL_* env passes through too)
set -u
source /root/container_env.sh 2>/dev/null
cd /root  # python -m puts the current folder on the import path; never start from inside the vllm package
P=${LOCAL_REPO:?set LOCAL_REPO to the absolute path of this repo on your own machine: the harness asks the server for the model by that path}
SPEC=()
if [ "${NUM_SPEC:-4}" != "0" ]; then
  SPEC=(--speculative-config "{\"method\":\"ngram\",\"num_speculative_tokens\":${NUM_SPEC:-4},\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}")
fi
[ -n "${GPUS:-}" ] && export CUDA_VISIBLE_DEVICES=$GPUS
nohup python3 -m vllm.entrypoints.openai.api_server --model $P/data/external/model --host 127.0.0.1 --port ${PORT:-8000} \
  --max-model-len 32768 --dtype bfloat16 --gpu-memory-utilization ${GPU_UTIL:-0.92} --enable-auto-tool-choice \
  --tool-call-parser gemma4 --reasoning-parser gemma4 \
  --reasoning-config '{"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}' \
  --default-chat-template-kwargs '{"enable_thinking": true}' --no-scheduler-reserve-full-isl \
  --enable-lora --max-loras 2 --max-lora-rank 8 --lora-modules main_lora=$P/agents/out8k/adapters/main_lora tool_lora=$P/agents/out8k/adapters/tool_lora \
  --tensor-parallel-size ${TP:-1} --max-num-seqs ${MAX_NUM_SEQS:-16} "${SPEC[@]}" ${EXTRA:-} \
  > /root/vllm${TAG:-}.log 2>&1 < /dev/null &
echo "vllm pid $! port ${PORT:-8000} tp ${TP:-1} spec ${NUM_SPEC:-4} gpus ${GPUS:-all}"
