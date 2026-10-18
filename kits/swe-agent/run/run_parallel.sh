#!/usr/bin/env bash
# Real tasks N at a time against the already-running local server, with server counters saved before and after.
# usage: bash kits/swe-agent/run/run_parallel.sh <concurrency> <results_name> <task_id> [...]
set -u
cd "$(dirname "$0")/../../.."
N=$1; NAME=$2; shift 2
OUT=runs/$NAME; mkdir -p $OUT
curl -s localhost:8000/metrics > $OUT/metrics_before.txt
CONCURRENCY=$N envs/harness/bin/python -I kits/swe-agent/run/run_tasks.py agents/sample $OUT "$@"
curl -s localhost:8000/metrics > $OUT/metrics_after.txt
echo "PARALLEL_DONE $(date -u +%H:%MZ)"
