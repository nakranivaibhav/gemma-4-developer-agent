#!/usr/bin/env bash
# Repair every flagged training-pool task (minimal notes, fixer v2 + trim + leak check), then re-judge the trimmed
# packets blind, with the untrimmed note as fallback for tasks that fail. Resume-safe at every step.
set -u
cd "$(dirname "$0")/../.."
J=/home/vaibhav/data/gemma4/judge; PY=envs/judge/bin/python; FX=$J/packets_fixed2; R=fix2_heldout
$PY training/judge/run_fixer.py --run $R --verdicts all_v3 --ids-file $J/fix_heldout.ids --fixed $FX --concurrency 6
python3 -c "
import json,glob
ids=[o['id'] for o in (json.load(open(f)) for f in glob.glob('runs/judge/$R/*.json')) if o.get('final_addendum')]
open('$J/$R.fixed.ids','w').write('\n'.join(ids)+'\n'); print('$R with a note', len(ids))"
$PY training/judge/run_judge.py --run ${R}_rejudge --packets $FX --ids-file $J/$R.fixed.ids --concurrency 6
python3 -c "
import json,glob
bad=[o['id'] for o in (json.load(open(f)) for f in glob.glob('runs/judge/${R}_rejudge/*.json') if not f.endswith('run.json')) if (o.get('verdict') or {}).get('R7',{}).get('answer')!='yes']
open('$J/$R.fallback.ids','w').write('\n'.join(bad)+'\n'); print('$R trimmed note failed re-judge', len(bad))"
[ -s $J/$R.fallback.ids ] && $PY training/judge/run_judge.py --run ${R}_rejudge_full --packets ${FX}_full --ids-file $J/$R.fallback.ids --concurrency 6
echo FIX_ALL_DONE
