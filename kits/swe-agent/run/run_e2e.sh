#!/usr/bin/env bash
# End-to-end probe: sample agent on 3 public tasks, then the skill agent on 1. Needs the local vLLM server up.
set -u
cd "$(dirname "$0")/../../.."
PY=envs/harness/bin/python
$PY -I kits/swe-agent/run/run_tasks.py agents/sample runs/sample_agent "$@"
$PY -I kits/swe-agent/run/run_tasks.py agents/skill runs/skill_agent requests_7505
