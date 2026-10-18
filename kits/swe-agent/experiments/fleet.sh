#!/usr/bin/env bash
# Start one GPU's workers for a fleet run (kits/swe-agent/run/fleet.py): one task-runner process per pass (run id), all pulling
# whole tasks from the shared queue. Output of all four goes to stdout, each line prefixed with its run id.
# usage: bash kits/swe-agent/experiments/fleet.sh <fleet_dir> <gpu> <server_json> <port> [port2 port3 port4]
#   one port: every pass talks to the same server; four ports: pass i talks to server i (one model copy per GPU)
# Ends when every worker has ended: the GPU's target went to 0 (drain), or the whole queue is done.
set -u
cd "$(dirname "$0")/../../.."
FD=$1; GPU=$2; SJ=$3; shift 3; PORTS=("$@")
RUNS=($(envs/harness/bin/python -I -c "import json,sys; print(' '.join(json.load(open(sys.argv[1]+'/manifest.json'))['runs']))" "$FD"))
echo "FLEET_LAUNCH gpu=$GPU ports=${PORTS[*]} runs=${RUNS[*]} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
for i in "${!RUNS[@]}"; do
  RUN=${RUNS[$i]}; PORT=${PORTS[$(( i < ${#PORTS[@]} ? i : 0 ))]}
  AGENT=$(envs/harness/bin/python -I -c "import json,sys; print(json.load(open(sys.argv[1]+'/manifest.json'))['agents'][sys.argv[2]])" "$FD" "$RUN")
  OUT=runs/$RUN/$GPU; mkdir -p "$OUT"
  ( FLEET_DIR=$FD FLEET_GPU=$GPU MODEL_PORT=$PORT SERVER_JSON=$SJ RUN_ID=$RUN EVAL_TIER=fleet \
      EVAL_NOTE="fleet run $(basename "$FD"): $AGENT on $GPU port $PORT" \
      envs/harness/bin/python -I kits/swe-agent/run/run_tasks.py "$AGENT" "$OUT" 2>&1 | sed -u "s/^/[$RUN] /" ) &
  sleep 2
done
wait
echo "FLEET_GPU_DONE gpu=$GPU $(date -u +%Y-%m-%dT%H:%M:%SZ)"
