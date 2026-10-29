#!/usr/bin/env bash
# Continuous E2B screen of approved outside tasks, batch after batch, never leaving the GPU idle (2026-10-25).
# Each batch runs pass_pipeline.sh: its pull + graphs + gold/empty check overlap the previous batch's E2B phase,
# and its E2B starts the moment the previous batch ends. At most two batches of images on disk at once.
# Picks via manifest.py next: approved train tasks not yet in a batch, repaired ones first once re-judged
# (until then clean ones). When nothing is approved and unclaimed, waits 30 min (the fixer adds more), retries.
# usage: [REMOTE=revideo MIN_FREE_GB=120] bash kits/swe-agent/outside/screen_loop.sh <first wait marker> [batch size 40]
#        stop: touch /tmp/screen_loop.stop. REMOTE (env) is passed to every batch: containers run on that host.
set -u
cd "$(dirname "$0")/../../.."
PREV=$1; B=${2:-40}; PY=envs/judge/bin/python
k=$(ls data/outside/screen[0-9]*.ids 2>/dev/null | sed -E 's/.*screen0*([0-9]+)\.ids/\1/' | sort -n | tail -1); k=${k:-0}
while [ ! -f /tmp/screen_loop.stop ]; do
  free_gb() { if [ -n "${REMOTE:-}" ]; then ssh -o BatchMode=yes "$REMOTE" "df --output=avail -BG / | tail -1" | tr -dc 0-9
              else df --output=avail -BG /home/vaibhav/data | tail -1 | tr -dc 0-9; fi; }
  while [ "$(free_gb)" -lt "${MIN_FREE_GB:-250}" ]; do  # disk guard (on REMOTE when set)
    echo "$(date -u +%H:%MZ) disk below ${MIN_FREE_GB:-250} GB free; waiting"; sleep 600; done
  Q=data/outside/queue.txt   # named batches to run first (e.g. backfills); one name per line, removed when launched
  if [ -s $Q ]; then
    N=$(head -1 $Q); sed -i 1d $Q; out="QUEUED $N ($(wc -l < data/outside/$N.ids) tasks)"
  else
    k=$((k + 1)); N=$(printf 'screen%03d' $k)
    until out=$($PY -I kits/swe-agent/outside/manifest.py next "$B" "$N" | tail -1) && [[ $out == NEXT\ [0-9]* ]]; do
      echo "$(date -u +%H:%MZ) nothing approved and unclaimed; retry in 30 min"; sleep 1800
      [ -f /tmp/screen_loop.stop ] && exit 0
    done
  fi
  echo "$(date -u +%H:%MZ) $out"
  rm -f /tmp/$N.done
  WAIT_FOR=$PREV setsid nohup uv run tools/run_with_watchdog.py --log runs/logs/$N.log --marker /tmp/$N.done \
    --idle-seconds 3600 -- bash kits/swe-agent/outside/pass_pipeline.sh $N > /dev/null 2>&1 < /dev/null &
  echo "$(date -u +%H:%MZ) LAUNCH $N (E2B after $PREV)"
  until [ -f "$PREV" ]; do sleep 60; done   # previous batch finished: $N now holds the GPU; prepare the next
  $PY -I kits/swe-agent/outside/manifest.py build | tail -2
  PREV=/tmp/$N.done
done
echo SCREEN_LOOP_STOPPED
