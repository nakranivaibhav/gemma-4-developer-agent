#!/usr/bin/env bash
# CPU self-test of fastgate.py (Amendment 6 Tier C) on the toy target. Never touches the GPU.
# PY defaults to `uv run python`; override with PY=/path/to/python.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
H="$REPO/tools/kernels/_harness"; T="$H/tests"
PY="${PY:-uv run python}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
cd "$REPO"
expect() {  # expect <name> <exit code> <substring of last line> <patch> [extra args]
  local name="$1" rc_want="$2" want="$3" patch="$4"; shift 4
  local rc=0
  $PY "$H/fastgate.py" --target "$T/toy_target.py" --patch "$patch" --steps 40 --eval-batches 4 \
    --device cpu --out "$OUT/$name" "$@" >"$OUT/$name.log" 2>&1 || rc=$?
  local last; last="$(tail -n 1 "$OUT/$name.log")"; echo "$last"
  if [[ $rc -ne $rc_want || "$last" != *"$want"* ]]; then
    echo "FAIL $name (exit $rc, want $rc_want / '$want')"; cat "$OUT/$name.log"; exit 1; fi
  [[ -s "$OUT/$name/fastgate.json" && -s "$OUT/$name/fastgate.csv" ]] || [[ "$want" == FASTGATE_MEMCHECK* ]] \
    || { echo "FAIL $name: missing outputs"; exit 1; }
}
# Deterministic CPU reference: identity must be exactly equal => PASS; a numerically different patch => INDETERMINATE.
expect identity 0 "FASTGATE_RESULT passed=True" "$T/toy_patch_identity.py"
expect fused_gelu 2 "FASTGATE_RESULT passed=INDETERMINATE" "$T/toy_patch_fused_gelu.py"
# A patch that renames parameters (wraps layer 0) falls back to order alignment and still passes exactly.
cat > "$OUT/rename_patch.py" <<'PY'
from torch import nn
def apply(model):
    model[0] = nn.Sequential(model[0]); return model
PY
expect rename 0 "FASTGATE_RESULT passed=True" "$OUT/rename_patch.py"
grep -q "ORDER alignment" "$OUT/rename.log" || { echo "FAIL rename: no order-alignment warning"; exit 1; }
expect memcheck 0 "FASTGATE_MEMCHECK" "$T/toy_patch_identity.py" --memory-check
echo "FASTGATE_CPU_TESTS_PASSED"
