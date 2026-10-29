#!/usr/bin/env bash
# Judge every outside task as its packet appears: repeat {list built packets -> judge the ones without a verdict}
# until the packet build has ended and a pass finds nothing new. Resume-safe (run_judge.py skips judged tasks).
set -u
cd "$(dirname "$0")/../.."
RUN=${1:-all_v3}; C=${2:-8}
P=/home/vaibhav/data/gemma4/judge/packets; IDS=/home/vaibhav/data/gemma4/judge/all.ids
L=/home/vaibhav/.claude/jobs/5a3b355f/tmp/all_built.ids
while :; do
  build_done=0; [ -f /tmp/judge_packets_all.done ] && build_done=1
  grep -Fxf <(ls "$P") "$IDS" > "$L"
  todo=$(python3 -c "
import json,os,sys
d='runs/judge/$RUN'; n=0
for i in open('$L'):
    f=os.path.join(d,i.strip()+'.json')
    if not (os.path.exists(f) and json.load(open(f)).get('verdict')): n+=1
print(n)")
  echo "PASS $(date -u +%H:%MZ) built=$(wc -l < $L) todo=$todo build_done=$build_done"
  if [ "$todo" = 0 ]; then
    [ $build_done = 1 ] && break
    sleep 300; continue
  fi
  envs/judge/bin/python training/judge/run_judge.py --run "$RUN" --ids-file "$L" --concurrency "$C"
done
echo ALL_JUDGED $RUN
