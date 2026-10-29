#!/usr/bin/env bash
# 31B on the 20 repaired fastapi tasks with the MINIMAL notes (v2), 2 tries each; compare with the v1 notes (18/40)
# and the original text (0/40, dg_31b_r08). Same agent and settings as fixtext_ab_31b.sh.
set -u
cd "$(dirname "$0")/../.."
TASKS_JSONL=data/tasks_fixed_pilot_v2.jsonl SPEC="" RUN_LABEL="ab=fixtext31b arm=fixed_v2" \
  bash kits/swe-agent/experiments/hist.sh agents/r09_31b_8tool fixtext_31b_fixedv2 tasks/fixed_pilot_kaggle_v2.ids 2 4 || { echo FAILED; exit 2; }
echo FIXTEXT31B_V2_DONE
