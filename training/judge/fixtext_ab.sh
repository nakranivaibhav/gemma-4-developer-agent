#!/usr/bin/env bash
# Outcome check for the repairs: base E2B on our 20 repaired fastapi tasks, repaired text vs original text, 4 tries each,
# plain decoding, Kaggle clock on (default). Same agent, same server settings, run back to back.
set -u
cd "$(dirname "$0")/../.."
E2B=data/external/models/gemma-4-e2b-it-qat-w4a16-ct
IDS=tasks/fixed_pilot_kaggle.ids
TASKS_JSONL=data/tasks_fixed_pilot.jsonl MODEL_DIR=$E2B SPEC="" RUN_LABEL="ab=fixtext arm=fixed" \
  bash kits/swe-agent/experiments/hist.sh agents/r09_e2b_8tool fixtext_e2b_fixed $IDS 4 40 || { echo FIXED_ARM_FAILED; exit 2; }
MODEL_DIR=$E2B SPEC="" RUN_LABEL="ab=fixtext arm=original" \
  bash kits/swe-agent/experiments/hist.sh agents/r09_e2b_8tool fixtext_e2b_orig $IDS 4 40 || { echo ORIG_ARM_FAILED; exit 3; }
echo FIXTEXT_AB_DONE
