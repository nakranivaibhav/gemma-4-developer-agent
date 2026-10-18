#!/usr/bin/env bash
# Generic probe: run_agent_probe.sh <agent_dir_name> <results_name> <task_id>...   Needs the local vLLM server up.
set -u
cd "$(dirname "$0")/../../.."
agent=$1; out=$2; shift 2
envs/harness/bin/python -I kits/swe-agent/run/run_tasks.py "agents/$agent" "runs/$out" "$@"
