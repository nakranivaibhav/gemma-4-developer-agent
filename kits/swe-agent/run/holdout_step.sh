#!/usr/bin/env bash
# The holdout step for a candidate that passed the tune screen (P >= 0.90): two candidate passes over the holdout tasks
# plus one more baseline pass, so both sides have two passes there. Order: candidate, baseline, candidate (guards against drift).
# Same settings as a tune pass: repaired sandbox, serve_fast already up on 127.0.0.1:8000, 3 at once. No early stop.
# usage: bash kits/swe-agent/run/holdout_step.sh <candidate_agent_dir> <candidate_prefix> <baseline_agent_dir> <baseline_run_id>
#   -> ledger run_ids <candidate_prefix>_p1, <baseline_run_id>, <candidate_prefix>_p2, tier holdout
# Launch it detached through tools/run_with_watchdog.py (see CLAUDE.md "Long runs"), never directly in a session.
set -u
cd "$(dirname "$0")/../../.."
CAND=$1; PREFIX=$2; BASE=$3; BASE_RUN=$4; TASKS=holdout
for A in "$CAND" "$BASE"; do [ -f "$A/agent.yaml" ] || { echo "NO_AGENT $A"; exit 2; }; done
curl -s -m 5 127.0.0.1:8000/v1/models >/dev/null || { echo "SERVER_DOWN"; exit 2; }
RC_ALL=0
one_pass() {  # <agent_dir> <run_id> <note>
  local OUT=runs/$2; mkdir -p "$OUT"
  curl -s localhost:8000/metrics > "$OUT/metrics_before.txt"
  echo "PASS_START $2 $(date -u +%H:%MZ)"
  RUN_ID=$2 EVAL_TIER=holdout EVAL_NOTE="$3" CONCURRENCY=3 envs/harness/bin/python -I kits/swe-agent/run/run_tasks.py "$1" "$OUT" "$TASKS"
  local RC=$?
  curl -s localhost:8000/metrics > "$OUT/metrics_after.txt"
  echo "PASS_DONE $2 rc=$RC $(date -u +%H:%MZ)"
  [ $RC -ne 0 ] && RC_ALL=$RC
}
one_pass "$CAND" "${PREFIX}_p1" "holdout step, candidate pass 1 of 2: $CAND on $TASKS, serve_fast, 3 at once"
one_pass "$BASE" "$BASE_RUN"    "holdout step, extra baseline pass: $BASE on $TASKS, serve_fast, 3 at once"
one_pass "$CAND" "${PREFIX}_p2" "holdout step, candidate pass 2 of 2: $CAND on $TASKS, serve_fast, 3 at once"
echo "HOLDOUT_DONE $PREFIX rc=$RC_ALL $(date -u +%H:%MZ)"
exit $RC_ALL
