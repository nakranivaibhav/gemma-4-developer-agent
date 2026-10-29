#!/usr/bin/env bash
# Overnight chain for SFT pool v2 on the local GPU (2026-10-23). One GPU, so the steps run in order:
#   1. train E2B on runs/sft/e2b_v2/packed.jsonl (recipe of v1: training/configs/sft_e2b_v2.yaml)
#   2. copy the adapter into agents/r09_e2b_8tool_sft_v2/adapters/main_lora
#   3. A/B base vs v2 on one server loaded with the v2 adapter: 123 usable tasks x 3 seeds, 96 at once
#   4. the v1 adapter again under the same server settings, so v1 and v2 are compared like for like
# Plain decoding (SPEC=""), as on Kaggle; the Kaggle clock is on by default (run_tasks.py).
# Launch detached under tools/run_with_watchdog.py.
set -u
cd "$(dirname "$0")/.."
E2B=data/external/models/gemma-4-e2b-it-qat-w4a16-ct
echo "STEP train $(date -u +%H:%MZ)"
uv run training/sft_train.py --config training/configs/sft_e2b_v2.yaml || { echo "TRAIN_FAILED"; exit 2; }
A=runs/sft/e2b_v2_train/adapter
[ -f "$A/adapter_model.safetensors" ] || { echo "NO_ADAPTER $A"; exit 2; }
cp "$A/adapter_config.json" "$A/adapter_model.safetensors" agents/r09_e2b_8tool_sft_v2/adapters/main_lora/
echo "STEP adapter copied $(date -u +%H:%MZ)"
echo "STEP ab base vs v2 $(date -u +%H:%MZ)"
MODEL_DIR=$E2B SPEC="" bash kits/swe-agent/experiments/ab_adapter.sh \
  agents/r09_e2b_8tool agents/r09_e2b_8tool_sft_v2 ab2_e2b_v2 usable 3 96 || { echo "AB_FAILED"; exit 3; }
echo "STEP v1 rerun $(date -u +%H:%MZ)"
MODEL_DIR=$E2B SPEC="" SERVE_AGENT=agents/r09_e2b_8tool_sft SAMPLE_TAG=p RUN_LABEL="ab=ab2_e2b_v1 arm=sft" \
  bash kits/swe-agent/experiments/hist.sh agents/r09_e2b_8tool_sft ab2_e2b_v1_sft usable 3 96 || { echo "V1_FAILED"; exit 4; }
echo "CHAIN_DONE $(date -u +%H:%MZ)"
