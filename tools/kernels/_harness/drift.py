"""Short-horizon training drift: patched model vs the untouched reference.

Builds the reference at seed S, deep-copies it and applies the patch to the copy
(same parameter count; identical initial values for parameters that keep their
name). Trains both on identical batches ``make_batch(device, S*10_000+i)`` with
separately instantiated optimisers of the same type and hyperparameters. The
global RNG is reseeded to the same value before every model's step so dropout
and other in-graph randomness match.

Noise band: the reference trained from a different init seed (S+1) on the same
batches; its per-step ``|loss_noise - loss_ref| / |loss_ref|`` is the scale of
change that a harmless perturbation produces.

Pass = min grad cosine >= --min-cos AND no NaN AND rel-dloss inside the band.
The band test is ``--criterion same_step`` (default, primary): at every step i,
``rel_patched[i] <= band_mult * rel_noise[i]``; or ``pooled``:
``max(rel_patched) <= band_mult * max(rel_noise)``. Both are always reported.
Last stdout line: ``DRIFT_RESULT passed=… min_cos=… max_rel_dloss=… noise_band=… steps=…``.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _bootstrap  # noqa: E402
    _bootstrap.fix_path()

import torch  # noqa: E402

from _harness.kpatch import assert_same_initial_values, assert_same_param_count, load_patch  # noqa: E402
from _harness.target import (Target, flat_grads, load_target, named_flat_grads, resolve_device,  # noqa: E402
                             train_step)

BATCH_SEED_STRIDE = 10_000  # batch seed for step i is seed*STRIDE + i


def _make_opt(kind: str, model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """Fresh optimiser of the requested type (same hyperparameters for every arm)."""
    if kind == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    raise ValueError(f"unknown optimizer {kind!r}")


def _grad_vectors(ref: torch.nn.Module, pat: torch.nn.Module) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Aligned flat grads: by parameter name when the name sets match, else by parameter order."""
    ref_named, pat_named = named_flat_grads(ref), named_flat_grads(pat)
    if set(ref_named) == set(pat_named):
        names = list(ref_named)  # reference order defines the concatenation order for both
        return (torch.cat([ref_named[n] for n in names]) if names else torch.zeros(0),
                torch.cat([pat_named[n] for n in names]) if names else torch.zeros(0), "by_name")
    return flat_grads(ref), flat_grads(pat), "by_order"


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity in fp64; 1.0 when both are all-zero, NaN if either is non-finite."""
    a64, b64 = a.double(), b.double()
    if not (torch.isfinite(a64).all() and torch.isfinite(b64).all()):
        return float("nan")
    na, nb = a64.norm(), b64.norm()
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float((a64 @ b64) / (na * nb))


def _step(target: Target, model: torch.nn.Module, opt: torch.optim.Optimizer, device: torch.device,
          batch_seed: int) -> float:
    """Reseed, fetch the batch, forward+backward (grads left on params). Returns the loss."""
    torch.manual_seed(batch_seed)  # identical in-graph randomness (dropout etc.) for every arm
    batch = target.make_batch(device, batch_seed)
    opt.zero_grad(set_to_none=True)
    return float(train_step(target, model, batch, device))


def run(args: argparse.Namespace) -> int:
    """Drift run; see module docstring."""
    device = resolve_device(args.device)
    target = load_target(args.target)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = target.build_model(args.seed)
    pat = load_patch(args.patch).apply(copy.deepcopy(ref))
    assert_same_param_count(ref, pat)
    n_same = assert_same_initial_values(ref, pat)
    noise = target.build_model(args.seed + 1)
    models = {"ref": ref.to(device).train(), "patched": pat.to(device).train(), "noise": noise.to(device).train()}
    opts = {k: _make_opt(args.optimizer, m, args.lr) for k, m in models.items()}

    rows: list[dict[str, Any]] = []
    align = "by_name"
    for i in range(args.steps):
        bseed = args.seed * BATCH_SEED_STRIDE + i
        losses = {k: _step(target, models[k], opts[k], device, bseed) for k in models}
        g_ref, g_pat, align = _grad_vectors(models["ref"], models["patched"])  # after backward, before step
        if g_ref.numel() != g_pat.numel():
            raise AssertionError(f"grad vectors differ in length: {g_ref.numel()} vs {g_pat.numel()}")
        cos = _cosine(g_ref, g_pat)
        for o in opts.values():
            o.step()
        denom = abs(losses["ref"]) if losses["ref"] != 0 else 1e-30  # guard an exactly-zero reference loss
        rows.append({"step": i, "loss_ref": losses["ref"], "loss_patched": losses["patched"],
                     "loss_noise": losses["noise"],
                     "rel_dloss_patched": abs(losses["patched"] - losses["ref"]) / denom,
                     "rel_dloss_noise": abs(losses["noise"] - losses["ref"]) / denom,
                     "grad_cos": cos})

    def finite(v: float) -> bool:
        return math.isfinite(v)

    any_nan = any(not (finite(r["loss_ref"]) and finite(r["loss_patched"]) and finite(r["grad_cos"])) for r in rows)
    min_cos = min(r["grad_cos"] for r in rows) if not any_nan else float("nan")
    max_rel = max(r["rel_dloss_patched"] for r in rows)
    noise_band = max(r["rel_dloss_noise"] for r in rows)
    # Same-step test: each step's patched drift must sit inside that step's seed-noise drift.
    same_step_fail = [r["step"] for r in rows if not r["rel_dloss_patched"] <= args.band_mult * r["rel_dloss_noise"]]
    pooled_ok = max_rel <= args.band_mult * noise_band
    band_ok = not same_step_fail if args.criterion == "same_step" else pooled_ok
    cos_ok = (not any_nan) and min_cos >= args.min_cos
    passed = bool(cos_ok and band_ok and not any_nan)

    with open(out / "drift.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {"target": target.name, "patch": args.patch, "device": str(device), "dtype_policy": target.dtype_policy,
               "seed": args.seed, "noise_seed": args.seed + 1, "steps": args.steps, "optimizer": args.optimizer,
               "lr": args.lr, "min_cos_bar": args.min_cos, "band_mult": args.band_mult, "criterion": args.criterion,
               "grad_alignment": align, "params_compared_for_init": n_same,
               "passed": passed, "cos_ok": cos_ok, "any_nan": any_nan, "min_cos": min_cos,
               "max_rel_dloss": max_rel, "noise_band": noise_band,
               "same_step_pass": not same_step_fail, "same_step_fail_steps": same_step_fail,
               "pooled_pass": pooled_ok, "rows": rows}
    (out / "drift.json").write_text(json.dumps(summary, indent=1))
    for r in rows[:: max(1, args.steps // 10)]:
        print(f"step {r['step']:4d} loss_ref={r['loss_ref']:.6g} rel_patched={r['rel_dloss_patched']:.3e} "
              f"rel_noise={r['rel_dloss_noise']:.3e} cos={r['grad_cos']:.6f}")
    print(f"same_step_pass={not same_step_fail} (fails at {same_step_fail[:10]}) pooled_pass={pooled_ok} "
          f"align={align}")
    print(f"DRIFT_RESULT passed={passed} min_cos={min_cos:.6f} max_rel_dloss={max_rel:.3e} "
          f"noise_band={noise_band:.3e} steps={args.steps} criterion={args.criterion}")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="target file defining build_target()")
    ap.add_argument("--patch", required=True, help="patch file defining apply(model)")
    ap.add_argument("--steps", type=int, default=50, help="training steps")
    ap.add_argument("--seed", type=int, default=0, help="init seed S (noise arm uses S+1)")
    ap.add_argument("--out", required=True, help="output dir for drift.json / drift.csv")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate (all arms)")
    ap.add_argument("--optimizer", default="adamw", choices=["sgd", "adamw"], help="optimizer type (all arms)")
    ap.add_argument("--min-cos", type=float, default=0.999, help="minimum per-step grad cosine")
    ap.add_argument("--band-mult", type=float, default=1.0, help="multiplier on the seed-noise band")
    ap.add_argument("--criterion", default="same_step", choices=["same_step", "pooled"], help="band test used for pass")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    args = ap.parse_args(argv)
    if args.steps < 1:
        ap.error("--steps must be >= 1")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
