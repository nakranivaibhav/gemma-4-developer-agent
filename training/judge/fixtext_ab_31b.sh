#!/usr/bin/env bash
# Repairs outcome check with the 31B (2026-10-25): agents/r09_31b_8tool (31B + main_lora, 8 tools) on the 20 repaired
# fastapi tasks, repaired text vs original text, 2 tries each, plain decoding, Kaggle clock on (31B+adapter profile).
# Both arms on the same server settings, back to back. Primary read: solved at all (fairness); clock read alongside.
set -u
cd "$(dirname "$0")/../.."
IDS=tasks/fixed_pilot_kaggle.ids
TASKS_JSONL=data/tasks_fixed_pilot.jsonl SPEC="" RUN_LABEL="ab=fixtext31b arm=fixed" \
  bash kits/swe-agent/experiments/hist.sh agents/r09_31b_8tool fixtext_31b_fixed $IDS 2 4 || { echo FIXED_ARM_FAILED; exit 2; }
SPEC="" RUN_LABEL="ab=fixtext31b arm=original" \
  bash kits/swe-agent/experiments/hist.sh agents/r09_31b_8tool fixtext_31b_orig $IDS 2 4 || { echo ORIG_ARM_FAILED; exit 3; }
echo FIXTEXT31B_DONE
