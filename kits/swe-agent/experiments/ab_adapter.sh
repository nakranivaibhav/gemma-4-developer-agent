#!/usr/bin/env bash
# Base vs adapter on one server setup: both arms are served with the ADAPTER agent's folder (LoRA enabled, its
# adapter loaded), the base arm just never asks for it. N samples per arm via hist.sh, base arm first.
# usage: bash kits/swe-agent/experiments/ab_adapter.sh <base_agent> <adapter_agent> <prefix> <task_list> [N=4] [TOTAL=128]
# env: MODEL_DIR is required (serve_fast.sh would otherwise load the 31B), plus the hist.sh knobs. Runs <prefix>_base_p1..pN then <prefix>_sft_p1..pN.
# Launch it detached under the watchdog.
set -u
cd "$(dirname "$0")/../../.."
BASE=$1; SFT=$2; PREFIX=$3; TASKS=$4; N=${5:-4}; TOTAL=${6:-128}
[ -n "${MODEL_DIR:-}" ] || { echo "set MODEL_DIR"; exit 2; }
[ -f "$SFT/adapters/main_lora/adapter_model.safetensors" ] || { echo "NO_ADAPTER in $SFT"; exit 2; }
SERVE_AGENT=$SFT SAMPLE_TAG=p RUN_LABEL="ab=$PREFIX arm=base" bash kits/swe-agent/experiments/hist.sh "$BASE" "${PREFIX}_base" "$TASKS" "$N" "$TOTAL" || exit $?
SERVE_AGENT=$SFT SAMPLE_TAG=p RUN_LABEL="ab=$PREFIX arm=sft" bash kits/swe-agent/experiments/hist.sh "$SFT" "${PREFIX}_sft" "$TASKS" "$N" "$TOTAL"
