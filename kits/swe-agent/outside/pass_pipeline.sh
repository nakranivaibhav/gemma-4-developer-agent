#!/usr/bin/env bash
# One timed forward pass over outside tasks: pull -> graphs -> gold/empty check -> E2B (1 try per valid task) -> remove
# images. Prints a TIME line per phase so we can size the full screen. (2026-10-25, first run: data/outside/pass50.*)
# usage: bash kits/swe-agent/outside/pass_pipeline.sh <name>     reads data/outside/<name>.{ids,jsonl}
# env: TRIES (E2B tries per task, 4; GPU waits on pulls anyway), PULL_PAR (4), GOLD_PAR (6), TOTAL (E2B tries at once, 24), WAIT_FOR (a marker file to wait on before the GPU
#      phase, e.g. another run's .done), KEEP_IMAGES=1 to skip the teardown.
#      REMOTE=<ssh host>: run the task containers on that host's Docker through an SSH socket tunnel (pulls, checks and
#      E2B sandboxes all go there; the model still serves from this machine's GPU). 2026-10-26: REMOTE=revideo.
set -u
cd "$(dirname "$0")/../../.."
N=$1; D=data/outside; OUT=runs/outside/$N; mkdir -p "$OUT"
E2B=data/external/models/gemma-4-e2b-it-qat-w4a16-ct
t() { date +%s; }
if [ -n "${REMOTE:-}" ]; then
  SOCK=/home/vaibhav/data/${REMOTE}-docker.sock
  if ! DOCKER_HOST=unix://$SOCK docker info > /dev/null 2>&1; then
    rm -f "$SOCK"
    setsid nohup ssh -N -o BatchMode=yes -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      -o StreamLocalBindUnlink=yes -L "$SOCK":/var/run/docker.sock "$REMOTE" > /dev/null 2>&1 < /dev/null &
    sleep 5
  fi
  export DOCKER_HOST=unix://$SOCK
  disk() { ssh -o BatchMode=yes "$REMOTE" "df -h / | tail -1" | awk -v h="$REMOTE" '{print h": "$3" used, "$4" free"}'; }
else
  disk() { df -h /home/vaibhav/data | tail -1 | awk '{print $3" used, "$4" free"}'; }
fi
phase() { echo "TIME $1 $(( $(t) - $2 )) s $(date -u +%H:%MZ)"; }

envs/judge/bin/python -I -c "import json; [print(json.loads(l)['docker_image']) for l in open('$D/$N.jsonl')]" | sort -u > "$OUT/images.txt"
echo "PASS $N: $(wc -l < $D/$N.ids) tasks, $(wc -l < $OUT/images.txt) images"

s=$(t)
xargs -P "${PULL_PAR:-4}" -I{} bash -c 'a=$(date +%s); if docker pull -q {} >/dev/null 2>&1 || { sleep 60; docker pull -q {} >/dev/null 2>&1; } || { sleep 180; docker pull -q {} >/dev/null 2>&1; }; then echo "PULL ok $(( $(date +%s)-a ))s {}"; else echo "PULL FAILED {}"; fi' < "$OUT/images.txt" | tee "$OUT/pulls.txt"
phase pull $s
# keep the shared base layers (OS + conda, ~850 MB) between batches: pin one image under a second name
if ! docker image inspect gemma/base-anchor:pinned > /dev/null 2>&1; then
  a=$(grep -m1 -oP 'PULL ok \S+ \K\S+' "$OUT/pulls.txt"); [ -n "$a" ] && docker tag "$a" gemma/base-anchor:pinned && echo "ANCHOR $a"
fi
echo "DISK $(disk)"

s=$(t); envs/harness/bin/python -I kits/swe-agent/outside/build_graphs.py "$D/$N.jsonl" > "$OUT/graphs.log" 2>&1; phase graphs $s

s=$(t)
GOLD_RUNS=1 PAR="${GOLD_PAR:-6}" envs/harness/bin/python -I kits/swe-agent/outside/gold_check.py "$D/$N.jsonl" "$OUT/gold" > "$OUT/gold.log" 2>&1
phase gold_check $s
envs/judge/bin/python -I - "$OUT/gold/gold.jsonl" "$D/$N.ids" "$OUT/valid.ids" <<'EOF'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(dict)
for r in rows:
    by[r["id"]][r["kind"]] = r
ids = [l.strip() for l in open(sys.argv[2]) if l.strip()]
valid = [i for i in ids if by[i].get("gold", {}).get("resolved") and by[i].get("empty", {}).get("resolved") is False]
bad = collections.Counter("no gold row" if "gold" not in by[i] else "gold fails" if not by[i]["gold"].get("resolved")
                          else "empty passes" if by[i].get("empty", {}).get("resolved") else "other" for i in ids if i not in valid)
open(sys.argv[3], "w").write("\n".join(valid) + "\n")
print(f"VALID {len(valid)}/{len(ids)} invalid: {dict(bad)}; gold test seconds median "
      f"{sorted(r['seconds'] for r in rows if r['kind']=='gold' and r.get('seconds'))[len(valid)//2] if valid else '-'}")
EOF

if [ -n "${WAIT_FOR:-}" ]; then s=$(t); echo "waiting for $WAIT_FOR (GPU)"; until [ -f "$WAIT_FOR" ]; do sleep 30; done; phase wait_gpu $s; fi
# the GPU is shared with other sessions: wait until vLLM can get its memory (0.88 x 32 GB), else it fails at startup
gpu_free() { [ -e /tmp/gpu_hold ] && { echo 0; return; }   # touch /tmp/gpu_hold = someone else needs the GPU
             nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
s=$(t); while [ "$(gpu_free)" -lt "${NEED_GPU_MB:-28500}" ]; do echo "GPU busy ($(gpu_free) MB free); waiting"; sleep 120; done
[ "$(( $(t) - s ))" -gt 0 ] && phase wait_gpu_free $s
s=$(t)
TASKS_JSONL="$D/$N.jsonl" LEDGER=results/outside_task_runs.jsonl MODEL_DIR=$E2B SPEC="" RUN_LABEL="pass=$N" \
  bash kits/swe-agent/experiments/hist.sh agents/r09_e2b_8tool "${N}_e2b" "$OUT/valid.ids" "${TRIES:-4}" "${TOTAL:-24}"
phase e2b $s
if grep -q 'Engine core initialization failed' runs/logs/serve_${N}_e2b.log 2>/dev/null; then   # lost a race for the GPU
  echo "E2B server failed to start; retrying once when the GPU is free"
  while [ "$(gpu_free)" -lt "${NEED_GPU_MB:-28500}" ]; do sleep 120; done
  s=$(t)
  TASKS_JSONL="$D/$N.jsonl" LEDGER=results/outside_task_runs.jsonl MODEL_DIR=$E2B SPEC="" RUN_LABEL="pass=$N" \
    bash kits/swe-agent/experiments/hist.sh agents/r09_e2b_8tool "${N}_e2b" "$OUT/valid.ids" "${TRIES:-4}" "${TOTAL:-24}"
  phase e2b_retry $s
fi

if [ -z "${KEEP_IMAGES:-}" ]; then
  s=$(t); xargs -r docker rmi -f < "$OUT/images.txt" > /dev/null 2>&1; docker image prune -f > /dev/null 2>&1; phase teardown $s
  echo "DISK $(disk)"
fi
echo PASS_DONE $N
