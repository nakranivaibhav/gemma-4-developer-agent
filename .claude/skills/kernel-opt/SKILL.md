---
name: kernel-opt
description: Orchestrator checklist for running the kernel-optimizer agent on an agreed training run — build the target file, confirm a GPU window, invoke the agent with ONLY the target path, review the hand-back line by line, re-run the gates yourself, adopt or file the null. Use after a round's refined.md is agreed and before the long run launches, or when the human asks for a custom kernel / training speed-up.
argument-hint: "<target file> [--gpu-minutes N] [--hours N]"
---

# /kernel-opt — one kernel pass on one training target

You (the orchestrator) own the target, the GPU window, the review and the
adoption. The agent owns the discovery. Do not tell it what the hot spot is.

## 0. Pitfalls first
- The agent's numbers are only valid on an **idle GPU** (< 1200 MiB used). A
  training run beside it corrupts every timing and can stretch that run's
  epochs past its watchdog. Book the window; never co-tenant.
- A patch module that edits a file a live queue re-reads is a launch fault.
  The agent must only produce `tools/kernels/<name>/` and a patch module; the
  wiring into a trainer happens in the node's own `src/` copy, by you.
- Subagent code gets your own line-by-line review before any run uses it —
  index math, boundary masks, dtype casts in the backward, the shapes it
  never saw. Then re-run parity and drift yourself; do not trust the report.
- The drift gate is three levels (unit / 50-step / full-run CV within noise).
  The agent proves the first two. The third is yours: the first arm shipping
  the kernel is scored paired against its reference and must sit inside the
  noise floor. Only then mark the kernel `adopted` in the registry.

## 1. Build the target file
Write `comps/<slug>/lines/kernels/<run>/target.py` following
`tools/kernels/_harness/target.py`: `build_target()` returns the model
factory, a batch factory that yields REAL data at the REAL batch size and
dtype, and the exact `step()` the trainer runs (forward + loss). State the
interpreter in the header comment. Smoke it once on CPU or a tiny GPU slice:
`uv run tools/kernels/_harness/kprofile.py --target <file> --dry-run`.

## 2. Confirm the window and invoke
```
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # must be < 1200 MiB
```
Invoke `Agent(subagent_type="kernel-optimizer", prompt=...)` with only: the
target path, the results directory, the time box, and the GPU-minutes budget.
No profile hints, no op names, no architecture description.

## 3. Review the hand-back
Read `results/report.md` and every file under `tools/kernels/<name>/`. Check:
plan entry predates kernel code (timestamps), tolerances match the README
table, parity ran at the largest profiled shape, drift used identical init +
batches, bench used warmup and medians. Re-run `parity.py` and `drift.py`
yourself. Any red ⇒ send the agent back once with the exact failing line, or
file the null.

## 4. Adopt or file
- Adopt-candidate: wire the patch into the node's `src/` copy, journal a
  `NOTE kernel_opt …` line naming kernel, share, step speedup, drift cos, and
  set the registry status to `candidate`. After the first full run passes the
  paired noise-floor check, flip it to `adopted`.
- Null: journal the NOTE with the measured share and the reason, registry
  status `null`, reopen-if written in the report.
