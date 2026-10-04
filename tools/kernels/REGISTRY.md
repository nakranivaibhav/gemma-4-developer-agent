<!-- CONTRACT: one row per kernel; update a row's status in place (candidate -> adopted | null).
     share/speedups come from kprofile.py / bench.py on an idle GPU; parity from parity.py;
     drift min cos from drift.py (50 steps). Dates UTC via `date -u +%Y-%m-%dT%H:%MZ`.
     `adopted` only after a full training run with the kernel scores within the noise floor of its reference. -->
# Kernel registry

Columns:
- **kernel** — folder name under `tools/kernels/`.
- **op replaced** — the reference op(s) the patch swaps out.
- **dtype** — dtype the kernel runs at (fp32 / bf16 / fp16).
- **shapes seen** — the profiled shapes it was checked at (largest first).
- **share of step** — the replaced op's share of step time from `kprofile.py`.
- **op speedup** — microbench median ratio, reference / kernel.
- **step speedup** — `bench.py` step median ratio, reference / patched.
- **parity** — pass / fail from `parity.py` at the README tolerances.
- **drift min cos** — minimum per-step grad cosine from `drift.py`.
- **status** — candidate / adopted / null.
- **first target** — the target file it was first measured on.
- **date UTC** — last status change.

| kernel | op replaced | dtype | shapes seen | share of step | op speedup | step speedup | parity | drift min cos | status (candidate/adopted/null) | first target | date UTC |
|---|---|---|---|---|---|---|---|---|---|---|---|
