#!/usr/bin/env bash
# Fixer pilot v2 (minimal addenda): fixer with rule 0 -> trim pass -> leak check; re-judge the trimmed packets blind;
# tasks whose trimmed note fails the re-judge get re-judged with the untrimmed note (fallback).
set -u
cd "$(dirname "$0")/../.."
J=/home/vaibhav/data/gemma4/judge; PY=envs/judge/bin/python; FX=$J/packets_fixed2
$PY training/judge/run_fixer.py --run fix2_pilot_kaggle --verdicts judge_v3_r1 --ids-file $J/fixpilot_kaggle.ids --fixed $FX --concurrency 3 &
$PY training/judge/run_fixer.py --run fix2_pilot_swe --verdicts all_v3 --ids-file $J/fixpilot_swe.ids --fixed $FX --concurrency 3 &
wait
for r in fix2_pilot_kaggle fix2_pilot_swe; do
  python3 -c "
import json,glob
ids=[o['id'] for o in (json.load(open(f)) for f in glob.glob('runs/judge/$r/*.json')) if o.get('final_addendum')]
open('$J/$r.fixed.ids','w').write('\n'.join(ids)+'\n'); print('$r with a note', len(ids))"
  $PY training/judge/run_judge.py --run ${r}_rejudge --packets $FX --ids-file $J/$r.fixed.ids --concurrency 3
  python3 -c "
import json,glob
bad=[o['id'] for o in (json.load(open(f)) for f in glob.glob('runs/judge/${r}_rejudge/*.json') if not f.endswith('run.json')) if (o.get('verdict') or {}).get('R7',{}).get('answer')!='yes']
open('$J/$r.fallback.ids','w').write('\n'.join(bad)+'\n'); print('$r trimmed note failed re-judge', len(bad))"
  [ -s $J/$r.fallback.ids ] && $PY training/judge/run_judge.py --run ${r}_rejudge_full --packets ${FX}_full --ids-file $J/$r.fallback.ids --concurrency 3
done
echo FIX_PILOT2_DONE
