#!/usr/bin/env bash
# Round 3 on the answer key (one run) alongside the 100-task SWE-rebench pilot, same prompt.
set -u
cd "$(dirname "$0")/../.."
envs/judge/bin/python training/judge/run_judge.py --run judge_v3_r1 --ids-file /home/vaibhav/data/gemma4/judge/key76.ids --concurrency 4 &
envs/judge/bin/python training/judge/run_judge.py --run pilot100_v3 --ids-file /home/vaibhav/data/gemma4/judge/pilot100.ids --concurrency 4 &
wait
python3 -I training/judge/score_judge.py judge_v3_r1 --vs judge_v2_r1
echo CALIB_PILOT_DONE
