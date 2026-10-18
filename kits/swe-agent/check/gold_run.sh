#!/usr/bin/env bash
# Sharded real-harness gold check. usage: [SHARDS=6] [PAR=4] [NO_PATCH=1] [WHEELS_DIR=..] [KAGGLE_SANDBOX_DIR=..] [TMPDIR=..] \
#   bash kits/swe-agent/check/gold_run.sh <out_dir> [task ids | --prefix X | --only-failed prev.jsonl]
# One process serializes its container work, so the parallelism comes from several processes.
set -u; cd "$(dirname "$0")/../../.."
out=$1; shift; n=${SHARDS:-6}; rm -rf "$out"; mkdir -p "$out"
for i in $(seq 0 $((n-1))); do
  SHARD=$i/$n PAR=${PAR:-4} envs/harness/bin/python -I kits/swe-agent/check/gold_verify.py "$out" "$@" > "$out/shard_$i.log" 2>&1 &
done
wait
envs/harness/bin/python -I - "$out" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1] + "/gold.jsonl")]
by = collections.defaultdict(lambda: [0, 0])
for r in rows:
    k = r["id"].rsplit("_", 1)[0]; by[k][1] += 1; by[k][0] += r["resolved"]
print("GOLD_TOTAL", sys.argv[1].split("/")[-1], f"{sum(r['resolved'] for r in rows)}/{len(rows)}", " ".join(f"{k}={a}/{b}" for k, (a, b) in sorted(by.items())), flush=True)
PY
