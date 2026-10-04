# tools/kernels — kernel-optimisation harness

A small, model-agnostic harness for replacing one hot op in a PyTorch training
step with a faster kernel, and proving the replacement changes nothing that
matters. It knows about models only through a **target file** and about kernels
only through a **patch file**. Nothing here is specific to a competition or repo.

## The loop

1. **Profile.** `kprofile.py` runs warmup + N profiled steps (forward, loss,
   backward, `zero_grad`) and N separate unprofiled steps for wall time. Read
   `gpu_busy_fraction` first: below 0.85 the GPU is waiting on the input
   pipeline or host, and a faster kernel buys little. Verdict `input_bound`
   means fix the feed, not the math.
2. **Pick.** Rank ops by *share of step × achievable fraction*. Each op in
   `profile_ops.json` carries `amdahl_max_speedup = 1/(1-share)`: the whole-step
   speedup if that op cost zero. Classify the op as memory- or compute-bound from
   bytes moved vs FLOPs at the recorded shapes. Unfused elementwise chains,
   norm + activation pairs, copy/cat glue and fp32 GEMMs off tensor cores are the
   usual memory-bound wins. Write the plan (op, share, Amdahl bound, expected
   gain, tolerances) in `<name>/results/plan.md` before writing kernel code.
3. **Write.** Triton first; raw CUDA only when Triton cannot express the op
   (say why). Wrap it in a `torch.autograd.Function` with an explicit backward.
4. **Parity.** `parity.py` compares the candidate at its dtype against the
   reference evaluated in **fp64**, forward and every input gradient, at the real
   shapes and the awkward ones (batch 1, odd sizes, all-zero input, the largest
   profiled shape). Two candidate runs must be bit-identical. Tolerances are
   fixed by dtype and are not loosened to pass:

   | dtype    | rtol  | atol  |
   |----------|-------|-------|
   | float32  | 1e-5  | 1e-6  |
   | bfloat16 | 1e-2  | 1e-3  |
   | float16  | 5e-3  | 1e-4  |

   Pass = `|cand - ref| <= atol + rtol*|ref|` elementwise (torch.allclose
   semantics), no NaN/Inf in the candidate, and deterministic. `worst_rel` is
   reported only where the rtol term governs (`|ref| > atol/rtol`).
5. **Bench.** `bench.bench_fn` for the op alone (CUDA events, warmup, median and
   p90), then `bench.py --target … --patch …` for the whole step. The step median
   is the number that counts. Bench only on an idle GPU (< 1200 MiB used); the
   CLI aborts otherwise.
6. **Drift.** `drift.py` trains the reference and a patched deep copy from the
   same weights on the same batches for 50 steps. Gates: per-step gradient cosine
   ≥ 0.999 and each step's relative loss difference inside the seed-noise band
   (the same quantity for the reference vs the reference from init seed S+1).
   Any failure means revert, not tune.
7. **Adopt.** A kernel becomes a *candidate* at ≥ 3 % step-time gain with parity
   and drift green. It becomes *adopted* only after the first full training run
   that uses it scores within the noise floor of its reference run. Otherwise it
   is filed as *null* with the measured numbers and a reopen-if condition.

## Target protocol (`_harness/target.py`)

A target file defines `build_target() -> Target`:

```python
import torch
from torch import nn
from _harness.target import Target      # tools/kernels must be on sys.path

def build_model(seed):                   # fresh CPU model, deterministic init
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 1))

def make_batch(device, seed):            # any pytree of tensors, deterministic in seed
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(8, 16, generator=g).to(device),
            "y": torch.randn(8, 1, generator=g).to(device)}

def step(model, batch):                  # forward + loss only, NO backward
    return nn.functional.mse_loss(model(batch["x"]), batch["y"])

def build_target():
    return Target("my_model", build_model, make_batch, step, dtype_policy="bf16_autocast")
```

`dtype_policy` is `fp32`, `bf16_autocast` or `fp16_autocast`; the harness wraps
`step` in the matching autocast and runs backward outside it. For real work the
batch factory must yield real data at the real batch size, and `step` must be
the exact forward + loss the trainer runs. Helpers: `load_target`,
`autocast_ctx`, `flat_grads`, `tree_map`, `tree_to`.

## Patch protocol (`_harness/kpatch.py`)

A patch file defines `apply(model) -> model` that swaps modules or ops for the
kernel versions. It must not change the parameter count or re-initialise
parameters (drift asserts both). The reference code is never edited.

```python
from mykernels.fused_norm_act import FusedNormAct
def apply(model):
    for name, m in model.named_children(): ...   # replace the target modules in place
    return model
```

## Folder layout

```
tools/kernels/
  README.md  REGISTRY.md
  _harness/  target.py kpatch.py kprofile.py parity.py bench.py drift.py tests/
  <name>/    kernel.py        the Triton/CUDA op + autograd.Function
             reference.py     the op it replaces, in plain torch
             test_parity.py   exposes candidate / reference / make_inputs
             apply_patch.py   apply(model) swap (loaded by path; never named patch.py)
             results/         plan.md, profile/, parity.json, bench.json, drift.json, report.md
```

Wiring a kernel into a particular trainer happens outside this tree, in that
project's own copy of its source.

## CLI lines

Run from the repo root; every CLI prints one `*_RESULT` line last.

```bash
uv run python tools/kernels/_harness/kprofile.py --target T.py --dry-run
uv run python tools/kernels/_harness/kprofile.py --target T.py --steps 25 --warmup 5 --out R/profile [--patch P.py]
uv run python tools/kernels/_harness/parity.py  --kernel tools/kernels/<name>/test_parity.py \
    --shapes '[[1,64],[37,64],[4096,64]]' --dtype bfloat16 --device cuda --out R/parity
uv run python tools/kernels/_harness/bench.py   --target T.py --patch P.py --steps 50 --warmup 10 --out R/bench
uv run python tools/kernels/_harness/drift.py   --target T.py --patch P.py --steps 50 --seed 0 --out R/drift \
    [--lr 1e-3 --optimizer adamw --min-cos 0.999 --band-mult 1.0 --criterion same_step --device cuda]
bash tools/kernels/_harness/tests/run_cpu_tests.sh     # harness self-test, CPU only
```

Result lines: `PROFILE_RESULT step_ms= gpu_busy= peak_gb= verdict= out=` ·
`PARITY_RESULT passed= n_shapes= worst_rel= deterministic=` ·
`BENCH_RESULT ref_ms= patched_ms= speedup= peak_ref_gb= peak_patched_gb=` (or
`BENCH_ABORT gpu_busy=<MiB>`) · `DRIFT_RESULT passed= min_cos= max_rel_dloss= noise_band= steps=`.

Notes on reading them:
- `kprofile.py` on CPU reports `gpu_busy=na` and verdict `cpu_only_no_gpu_verdict`;
  op shares there are CPU self time and do not transfer to the GPU.
- Op share is taken against the unprofiled step time; profiler overhead can make
  a share slightly off, so treat shares under 1 % as noise.
- Drift `same_step` compares each step's patched drift against that same step's
  seed-noise drift; `pooled` compares the maxima. Both are in `drift.json`.
- Drift reseeds the global RNG to the batch seed before every arm's step, so
  dropout masks match between arms.

## Registry

`REGISTRY.md` gets one row per kernel, updated in place as its status moves
candidate → adopted or → null. Row format:

```
| fused_gn_gelu | GroupNorm+GELU | bf16 | (64,256,32,32) | 11.2 % | 2.41x | 1.071x | pass | 0.99994 | candidate | toy_target | 2026-09-25T12:00Z |
```

Dates are UTC from `date -u +%Y-%m-%dT%H:%MZ`.

## Amendments (2026-09-25, from the first real target — biohub n85)

**Reference-relative parity.** The fixed tolerance table is the first gate, not the only one. When it fails,
run two controls through the same parity CLI: the *incumbent* (the reference op in its production dtype, e.g.
fp32 torch) and the *oracle* (fp64 math with the reference's own I/O dtype). If the oracle also fails, the
absolute gate is unmeetable at that reduction size and parity is judged **reference-relative**: the candidate's
`max_abs` error vs fp64 must be <= the incumbent's on every (shape x tensor) cell, with determinism unchanged.
Record all three runs. A candidate that is less accurate than the incumbent anywhere fails, whatever the table says.

**Nondeterministic references.** Before trusting `drift.py`, run it once with the identity patch
(`_harness/tests/toy_patch_identity.py`). If the identity run fails the cosine bar, the reference itself is not
run-to-run deterministic (atomics, data-dependent stages) and the trajectory gate is void for that target. Use the
same-weights instrument instead: at identical weights and batch, grad cosine ref-vs-patched must sit at the
ref-vs-ref floor (|difference| within 1e-6 of the floor) on >= 8 batches spanning the regimes the run visits.

**Two regimes.** Profile and bench at random init AND from a trained checkpoint of the same architecture when the
step's cost depends on model outputs (detection counts, sparsity). Quote the trained-regime step speedup as the
planning number; the init number is transient.

**Mega-kernels.** A single persistent kernel over a whole sub-graph (a small per-pair or per-token MLP, a
norm + attention + residual block) is a candidate class of its own: the win is that the intermediate
activations never reach HBM in either direction, with recompute in registers/SMEM in the backward.
Pre-register with the bytes-avoided arithmetic. If dropout must be generated in-kernel, parity is judged
with dropout OFF (exact eval-mode match to the reference chain) plus a train-mode check of the kept fraction
and loss/grad statistics; the same-weights cosine gate then runs in eval mode and the plan says so.

### Amendment 5 (2026-09-26, orchestrator) — forward-changing ops upstream of a discrete stage
A kernel that changes FORWARD rounding of a block feeding a discrete selection (peak detection, matching, argmax)
cannot be judged by the whole-step same-weights cosine: a bf16-ULP change flips selections and the gradient graph
changes shape (measured on n85: node sets differed on 16/16 frames, cosine 0.9999x vs floor 0.9999992, while a
torch-only re-layout control failed identically). Gating instruments instead: (a) a CONTINUOUS loss that stops
before the discrete stage (e.g. detection loss on the encoder), same weights, cosine vs the ref-vs-ref floor;
(b) an fp32 anchor (autocast off, same batch) — the kernel passes when its gradient is at least as close to the
anchor as the reference's is, within the ref-vs-ref resolution, on ≥ 8 batches in the TRAINED regime. Report the
init regime too; plan on trained. A pass here makes the kernel a candidate whose adoption is a training-run A/B.
Reference run: comps/biohub-cell-tracking-during-development/lines/kernels/n85/results/orch_verify_p2.py.

### Amendment 6 (2026-09-26, human directive) — FAST adoption: no dedicated full-run A/B for kernels
The full-run twin (node_0091) cost 3.65 GPU-h and resolved only one reseed unit (0.011 adj), so it could catch a
gross defect and nothing finer. Adoption is now decided in ≤ 30 GPU-min by three tiers, all on the real trainer step:
- **Tier A — exactness (seconds).** Forward bit-exact vs the reference at every profiled shape (`check_fwd_exact`
  style, 12/12 shapes, losses identical on ≥ 8 batches) AND backward within the reference's own run-to-run band ⇒
  **ADOPT directly.** Nothing a training run could add is resolvable by a training run.
- **Tier B — same weights (minutes).** Whole-step grad cosine at the ref-vs-ref floor + node sets identical
  (Amendment "Nondeterministic references"), or Amendment 5's continuous-loss + fp32-anchor pair for a
  forward-changing op upstream of the discrete stage.
- **Tier C — short-horizon divergence from a TRAINED checkpoint (`_harness/fastgate.py`, ~10–20 min).** Three
  arms from the recipe's `ckpt_ep020.pth` (or the target's trained shim) with the recipe optimiser (AdamW, lr 1e-3)
  on identical batches for K ≥ 200 steps: `ref`, `ref2` (the same reference again — its distance from `ref` IS the
  nondeterminism band), `patched`. Read per step: |loss_p − loss_r| vs |loss_r2 − loss_r|; at the end: parameter
  distance ‖w_p − w_r‖ vs ‖w_r2 − w_r‖ (global and per top-level module), and on ≥ 8 fixed eval batches the
  detection loss + peak count + best-recall proxy of the three arms. PASS = patched within `band_mult` (default 2)
  of the ref2 band on every read, no NaN. drift.py's "different init seed" band is NOT the band here: it is
  orders of magnitude too loose (it measures a different model, not the same model's nondeterminism).
- **Rule.** Tier A pass ⇒ adopted. Else Tier B pass AND Tier C pass ⇒ adopted. Any fail ⇒ null or iterate. The
  first REAL training node that ships an adopted composition gets its result read against the reseed floor as a
  free by-product (that node's own gate) — never a dedicated twin. Every candidate on the board (pair_l1_gelu,
  up_cat3d, node_gather, sdpa_t2, bn_relu3d 8d-1) is re-judged under this rule; node_0091's single-twin read is
  demoted to descriptive (it was inside one seed unit for the trio and could not separate sdpa_t2 from noise).
**Amendment 6 addendum — calibration and false-fail handling (2026-09-26).** With a nondeterministic reference the
patched arm is an exchangeable third draw, so every gated ratio is one random draw over another: on a toy target
the IDENTITY patch failed 1 run in 10 at `band_mult 2` (eval-loss ratio 0.63–2.24 across seeds). Therefore:
(1) run `fastgate.py` with `_harness/tests/toy_patch_identity.py` on the real target first and record its ratios;
(2) a single-conjunct fail triggers ONE rerun at `--seed S+1`; a fail on both runs is a FAIL, a pass on the rerun is
a PASS with both outputs kept; (3) `--memory-check` before any real run (three arms + optimiser state; the stock arm
sets the peak). CLI:
```
uv run python tools/kernels/_harness/fastgate.py --target T.py --patch P.py --out DIR [--steps 200] [--seed 0] \
  [--lr 1e-3] [--optimizer adamw|sgd] [--clip-grad 0] [--band-mult 2.0] [--dist-every 25] [--eval-batches 8] \
  [--ckpt SD.pth] [--device auto] [--memory-check]
  -> DIR/fastgate.{json,csv}; last line FASTGATE_RESULT passed=<True|False|INDETERMINATE> ...; exit 0/1/2.
```
**Measured calibration on the real stock target (2026-09-26, `lines/kernels/n85/fastgate/stock/`).** Identity: per-step
exceed count 65/200, d_pat/d_band 0.95, one conjunct (eval_det_loss) at 2.14x the band ⇒ FAIL on a single run, as
predicted. Across every arm incl. identity, parameter distance ratios sat at 0.94–1.09 (and patched-vs-ref2 at
1.01–1.06): the distance read is the robust one; the loss and 8-batch eval reads are heavy-tailed because of the
discrete detection/matching stage and are what produce single-conjunct fails. Next revision: gate on distance +
eval with ≥ 16 batches, report loss spikes descriptively; and a window script must wait on the chain MARKER, not on
GPU memory alone (a multi-stage node frees the card between stages — fastgate fired in such a gap and collided).
