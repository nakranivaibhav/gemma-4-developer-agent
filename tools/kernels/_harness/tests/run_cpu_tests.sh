#!/usr/bin/env bash
# CPU self-test of the kernel harness. Run from anywhere; exits non-zero on the first failure.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
H="$REPO/tools/kernels/_harness"
T="$H/tests"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
cd "$REPO"
export CUDA_VISIBLE_DEVICES=""   # keep every test off the GPU

check() {  # check <name> <required substring> <command...>
  local name="$1" want="$2"; shift 2
  local log="$OUT/$name.log"
  local rc=0
  "$@" >"$log" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then echo "FAIL $name (exit $rc)"; cat "$log"; exit 1; fi
  local last; last="$(tail -n 1 "$log")"
  echo "$last"
  if [[ "$last" != *"$want"* ]]; then echo "FAIL $name: last line lacks '$want'"; cat "$log"; exit 1; fi
}

check profile_dry "PROFILE_RESULT passed=True" \
  uv run python "$H/kprofile.py" --target "$T/toy_target.py" --dry-run
check profile_dry_patched "PROFILE_RESULT passed=True" \
  uv run python "$H/kprofile.py" --target "$T/toy_target.py" --patch "$T/toy_patch_fused_gelu.py" --dry-run
check profile_cpu "PROFILE_RESULT step_ms=" \
  uv run python "$H/kprofile.py" --target "$T/toy_target.py" --steps 3 --warmup 1 --device cpu --out "$OUT/profile"
for f in profile_table.txt profile_ops.json trace.json summary.json; do
  [[ -s "$OUT/profile/$f" ]] || { echo "FAIL profile_cpu: missing $f"; exit 1; }
done
check parity "PARITY_RESULT passed=True" \
  uv run python "$H/parity.py" --kernel "$T/toy_kernel_gelu.py" --device cpu --dtype float32 \
    --shapes '[[1], [7, 13], {"shape": [4, 33, 5], "fill": "zeros"}]' --out "$OUT/parity"
check drift_identity "DRIFT_RESULT passed=True" \
  uv run python "$H/drift.py" --target "$T/toy_target.py" --patch "$T/toy_patch_identity.py" \
    --steps 20 --seed 0 --device cpu --out "$OUT/drift_identity"
check drift_fused_gelu "DRIFT_RESULT passed=True" \
  uv run python "$H/drift.py" --target "$T/toy_target.py" --patch "$T/toy_patch_fused_gelu.py" \
    --steps 20 --seed 0 --device cpu --out "$OUT/drift_gelu"
check bench_import "bench_import_ok" \
  uv run python -c "import sys; sys.path.insert(0, '$REPO/tools/kernels'); import _harness.bench as b; assert callable(b.bench_fn) and callable(b.gpu_is_idle); print('bench_import_ok')"
check bench_help "bench.py" bash -c "uv run python '$H/bench.py' --help >/dev/null && echo bench.py help ok"
# Negative control: a wrong candidate must FAIL parity (exit 1, passed=False).
cat > "$OUT/bad_kernel.py" <<'PY'
import torch, torch.nn.functional as F
def candidate(x): return F.gelu(x, approximate="tanh")
def reference(x): return F.gelu(x)
def make_inputs(s, device, dtype): return [torch.randn(s, dtype=dtype).to(device).requires_grad_(True)]
PY
if uv run python "$H/parity.py" --kernel "$OUT/bad_kernel.py" --device cpu --shapes '[[64]]' --out "$OUT/bad" >"$OUT/bad.log" 2>&1; then
  echo "FAIL negative control: tanh-GELU passed parity"; cat "$OUT/bad.log"; exit 1
fi
tail -n 1 "$OUT/bad.log"
grep -q "PARITY_RESULT passed=False" "$OUT/bad.log" || { echo "FAIL negative control output"; exit 1; }
echo "ALL_CPU_TESTS_PASSED"
