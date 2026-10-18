#!/usr/bin/env bash
# Sub-agent probe: sample agent with a mandatory first call to code_analyzer_agent. Needs the local vLLM server up.
set -u
cd "$(dirname "$0")/../../.."
envs/harness/bin/python -I kits/swe-agent/run/run_tasks.py agents/subagent_probe runs/subagent_probe "$@"
