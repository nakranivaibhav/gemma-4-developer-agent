#!/usr/bin/env bash
# Finish one pass that was cut short: run only the tasks of a task list that have no ledger row yet for <run_id>,
# into the same run folder, with the same runner settings as kits/swe-agent/run/run_round.sh (repaired sandbox, 3 at a time).
# usage: bash kits/swe-agent/run/resume_pass.sh <agent_dir> <run_id> [tasks=tune] [tier=tune_pass]
# Launch it detached through tools/run_with_watchdog.py with --idle-seconds 1800: a task may print nothing for 20+ minutes.
set -u
cd "$(dirname "$0")/../../.."
AGENT=$1; RUN=$2; TASKS=${3:-tune}; TIER=${4:-tune_pass}
[ -f "$AGENT/agent.yaml" ] || { echo "NO_AGENT $AGENT"; exit 2; }
curl -s -m 2 127.0.0.1:8000/v1/models >/dev/null || { echo "SERVER_DOWN"; exit 2; }
MISSING=$(envs/harness/bin/python -I -c "
import json, sys
sys.path.insert(0, 'kits/swe-agent/run')
import manifest
done = {json.loads(l).get('id') for l in open('results/task_runs.jsonl') if json.loads(l).get('run_id') == sys.argv[1]}
print(' '.join(t for t in manifest.ids(sys.argv[2]) if t not in done))" "$RUN" "$TASKS")
N=$(echo $MISSING | wc -w)
echo "RESUME $RUN missing=$N $(date -u +%H:%MZ)"
[ "$N" -eq 0 ] && { echo "RESUME_DONE $RUN rc=0 nothing missing"; exit 0; }
OUT=runs/$RUN
[ -f "$OUT/settings.json" ] && cp "$OUT/settings.json" "$OUT/settings_before_resume.json"
RUN_ID=$RUN EVAL_TIER=$TIER EVAL_NOTE="resume of $RUN: $N tasks left after a watchdog stop, $AGENT on $TASKS, serve_fast, 3 at once" CONCURRENCY=3 \
  envs/harness/bin/python -I kits/swe-agent/run/run_tasks.py "$AGENT" "$OUT" $MISSING
RC=$?
curl -s localhost:8000/metrics > "$OUT/metrics_after.txt"
echo "RESUME_DONE $RUN rc=$RC $(date -u +%H:%MZ)"
exit $RC
