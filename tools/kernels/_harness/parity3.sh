#!/usr/bin/env bash
# parity3.sh PY KDIR SHAPES OUTDIR [cand_module incumbent_module oracle_module]
# Runs parity.py for candidate / incumbent / oracle on the same shapes, then refrel.py. Serial (one GPU process).
set -u
PY=$1; K=$2; SH=$3; O=$4; C=${5:-test_parity}; I=${6:-parity_control_ref_fp32}; OR=${7:-parity_control_oracle}
H=$(dirname "$0")
for pair in "$C:parity" "$I:parity_control_ref_fp32" "$OR:parity_control_oracle"; do
  m=${pair%%:*}; d=${pair##*:}
  "$PY" "$H/parity.py" --kernel "$K/$m.py" --shapes "$SH" --dtype ${DT:-float32} --device cuda --out "$O/$d" 2>&1 | grep -E "PARITY_RESULT|Error|Traceback"
done
"$PY" "$H/refrel.py" --cand "$O/parity/parity.json" --incumbent "$O/parity_control_ref_fp32/parity.json" \
  --oracle "$O/parity_control_oracle/parity.json" --out "$O/refrel.json"
