#!/usr/bin/env bash
# Fixer pilot: repair judge-flagged tasks (our Kaggle key tasks + 50 SWE-rebench), then re-judge the fixed packets blind.
set -u
cd "$(dirname "$0")/../.."
J=/home/vaibhav/data/gemma4/judge; PY=envs/judge/bin/python
$PY training/judge/run_fixer.py --run fix_pilot_kaggle --verdicts judge_v3_r1 --ids-file $J/fixpilot_kaggle.ids --concurrency 3 &
$PY training/judge/run_fixer.py --run fix_pilot_swe --verdicts all_v3 --ids-file $J/fixpilot_swe.ids --concurrency 3 &
wait
for r in fix_pilot_kaggle fix_pilot_swe; do
  python3 -c "
import json,glob
ids=[json.load(open(f))['id'] for f in glob.glob('runs/judge/$r/*.json') if (json.load(open(f)).get('fix') or {}).get('fixable')=='yes']
open('$J/$r.fixable.ids','w').write('\n'.join(ids)+'\n'); print('$r fixable', len(ids))"
  $PY training/judge/run_judge.py --run ${r}_rejudge --packets $J/packets_fixed --ids-file $J/$r.fixable.ids --concurrency 3
done
echo FIX_PILOT_DONE
