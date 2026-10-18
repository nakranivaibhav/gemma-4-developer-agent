#!/usr/bin/env bash
# Max-parallel sweep for E2B and E4B (training rollouts, not Kaggle; user 2026-10-16).
# Per model: speed matrix plain vs ngram at VARIANT_LEVELS agents, pick the fewest agents within 3% of best,
# then ONE real pass over all 123 tasks at that level. A system log runs alongside the whole sweep.
# The 1-24 matrix from 2026-10-15 is kept in runs/bench/sweep1_baseline/.
# Output: runs/par_sweep_sys.csv (every 10 s), runs/par_sweep.log, ledger run_ids v_<tag>_par_p1
set -u
cd "$(dirname "$0")/../../.."
SYS=runs/par_sweep_sys.csv
echo "utc,cpu_busy_pct,load1,mem_used_gb,gpu_util_pct,gpu_mem_mib,containers" > "$SYS"
( read -r _ a b c d e f g _ < /proc/stat; t0=$((a+b+c+d+e+f+g)); i0=$((d+e))
  while sleep 10; do
    read -r _ a b c d e f g _ < /proc/stat; t1=$((a+b+c+d+e+f+g)); i1=$((d+e))
    g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    m=$(free -m | awk '/Mem:/{printf "%.1f", $3/1024}')
    echo "$(date -u +%H:%M:%SZ),$(( 100 * ((t1-t0) - (i1-i0)) / (t1-t0) )),$(cut -d' ' -f1 /proc/loadavg),$m,$g,$(timeout 10 docker ps -q | wc -l)" >> "$SYS"
    t0=$t1; i0=$i1
  done ) &
SYS_PID=$!
trap 'kill $SYS_PID 2>/dev/null' EXIT
export MAX_PAR=128 VARIANT_LEVELS="16 24 32 48 64 96 128" PASSES=1
touch /tmp/gemma_go.marker
bash kits/swe-agent/bench/variant_chain.sh /tmp/gemma_go.marker \
  e2b:8316306646:usable:v_e2b_par \
  e4b:11513861828:usable:v_e4b_par
echo "SWEEP_DONE $(date -u +%H:%MZ)"
