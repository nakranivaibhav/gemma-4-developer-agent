#!/usr/bin/env bash
# One calibration round: the judge twice over the answer-key tasks (self-agreement), then the scores.
# usage: bash training/judge/calib_round.sh <tag> [concurrency]   e.g. v1  ->  runs judge_v1_r1 + judge_v1_r2
set -u
cd "$(dirname "$0")/../.."
TAG=$1; C=${2:-4}; IDS=/home/vaibhav/data/gemma4/judge/key76.ids
envs/judge/bin/python training/judge/run_judge.py --run judge_${TAG}_r1 --ids-file $IDS --concurrency $C &
envs/judge/bin/python training/judge/run_judge.py --run judge_${TAG}_r2 --ids-file $IDS --concurrency $C &
wait
python3 -I training/judge/score_judge.py judge_${TAG}_r1 --vs judge_${TAG}_r2
python3 -I training/judge/score_judge.py judge_${TAG}_r2
echo CALIB_DONE $TAG
