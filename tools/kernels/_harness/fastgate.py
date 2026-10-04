"""Tier C fast gate (README Amendment 6): short-horizon training divergence of a patched model vs its OWN
run-to-run nondeterminism band.

Three arms from the SAME initial weights (``target.build_model(seed)``, then ``--ckpt`` loaded strict if given):
  ref     the reference
  ref2    ``deepcopy(ref)`` -- identical weights, NOT a different seed; its distance from ``ref`` IS the band
  patched ``patch.apply(deepcopy(ref))``
Each arm gets a fresh optimiser of the same type/hparams. Step i: ``batch = make_batch(device, seed*10_000+i)``
(built ONCE, shared by the three arms, checksummed so an in-place mutation by ``step`` is caught); per arm:
``torch.manual_seed(seed*10_000+i)``, ``zero_grad``, forward+loss under the dtype policy, backward, optional
grad clip, ``opt.step()``, grads freed, ``empty_cache``.

Reads: per step loss of every arm, ``rel_pat = |l_pat-l_ref|/|l_ref|`` and ``rel_band = |l_ref2-l_ref|/|l_ref|``;
every ``--dist-every`` steps (and at the end) the fp64 parameter distance ``||w_pat-w_ref||`` and
``||w_ref2-w_ref||`` globally and per top-level module (first dotted component of the parameter name); after
training, ``--eval-batches`` fixed batches ``make_batch(device, 777_000+j)`` in eval mode under no_grad: the step
loss and, when the model exposes ``.encode`` returning ``(unet_out, det_logits)``, the count of detection logits
> 0 (peak-count proxy) and -- only if the step's module globals expose ``_tr.compute_detection_loss`` and
``DET_NEG_WEIGHT`` and the batch carries ``coords``/``masks`` -- the detection loss.

PASS = every conjunct holds (each reported by name):
  finite          no NaN/inf loss in any arm, and final parameters finite
  pooled_loss     max rel_pat <= band_mult * max(max rel_band, 1e-7)
  dist_global     d_pat <= band_mult * max(d_band, 1e-12 * ||w_ref||)
  dist_module     the same rule per top-level module (violators listed)
  eval_loss       mean|e_pat-e_ref| <= band_mult * max(mean|e_ref2-e_ref|, 1e-7)
  eval_det_loss   (when available) same rule as eval_loss
  eval_peaks      (when available) mean|n_pat-n_ref| <= band_mult * max(mean|n_ref2-n_ref|, 1.0)
Degenerate band: if d_band == 0 and rel_band == 0 at every step the reference is deterministic and the band says
nothing. Then the patched arm must be EXACTLY equal (d_pat == 0 and rel_pat == 0 everywhere) for PASS; otherwise
the verdict is INDETERMINATE (raw distances reported), never a silent pass. Non-finite values always FAIL.

Outputs ``OUT/fastgate.json`` and ``OUT/fastgate.csv``. Exit 0 pass / 1 fail / 2 indeterminate. Last stdout line:
``FASTGATE_RESULT passed=<True|False|INDETERMINATE> steps=K max_rel_pat=… max_rel_band=… d_pat=… d_band=…
eval_dloss_pat=… eval_dloss_band=… violators=<modules|none>``.
``--memory-check`` builds the arms, runs 2 steps, prints ``FASTGATE_MEMCHECK peak_gb=…`` and exits.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _bootstrap  # noqa: E402
    _bootstrap.fix_path()

import torch  # noqa: E402

from _harness.kpatch import assert_same_initial_values, assert_same_param_count, load_patch  # noqa: E402
from _harness.target import Target, autocast_ctx, load_target, resolve_device, tree_leaves  # noqa: E402

BATCH_SEED_STRIDE = 10_000   # train batch seed for step i is seed*STRIDE + i (same as drift.py)
EVAL_SEED_BASE = 777_000     # eval batch j is make_batch(device, EVAL_SEED_BASE + j)
REL_FLOOR = 1e-7             # floor on the relative-loss band and the eval-loss band
DIST_REL_FLOOR = 1e-12       # floor on the distance band, relative to ||w_ref||
PEAK_FLOOR = 1.0             # floor on the peak-count band: one logit-voxel per batch
EPS = 1e-30
ARMS = ("ref", "ref2", "patched")


# ----------------------------------------------------------------------------- building
def _make_opt(kind: str, model: torch.nn.Module, lr: float) -> torch.optim.Optimizer:
    """Fresh optimiser of the requested type (identical hyperparameters for every arm)."""
    if kind == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr)
    raise ValueError(f"unknown optimizer {kind!r}")


def _load_ckpt(path: str) -> dict[str, torch.Tensor]:
    """Load a state_dict (or a dict wrapping one under state_dict/model_state_dict/model) on CPU."""
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and not all(isinstance(v, torch.Tensor) for v in obj.values()):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(obj.get(key), dict):
                return obj[key]
        raise ValueError(f"{path}: not a state_dict and no state_dict/model_state_dict/model key")
    return obj


def _storage_ptrs(model: torch.nn.Module) -> set[int]:
    return {p.data_ptr() for p in model.parameters()}


def build_arms(target: Target, patch_path: str, seed: int, ckpt: str | None,
               device: torch.device) -> tuple[dict[str, torch.nn.Module], dict[str, Any]]:
    """Build ref / ref2 / patched from one set of weights; assert identity of init and no shared storage."""
    ref = target.build_model(seed)
    info: dict[str, Any] = {"ckpt": ckpt}
    if ckpt:
        res = ref.load_state_dict(_load_ckpt(ckpt), strict=True)
        info["ckpt_missing"], info["ckpt_unexpected"] = list(res.missing_keys), list(res.unexpected_keys)
    # Both copies are taken from ref BEFORE the patch touches anything: apply() may mutate in place.
    ref2 = copy.deepcopy(ref)
    pat = load_patch(patch_path).apply(copy.deepcopy(ref))
    if pat is ref or pat is ref2:
        raise AssertionError("patch.apply returned a model object aliased to another arm")
    assert_same_param_count(ref, ref2)
    assert_same_param_count(ref, pat)
    info["params_compared_ref2"] = assert_same_initial_values(ref, ref2)
    info["params_compared_patched"] = assert_same_initial_values(ref, pat)
    models = {"ref": ref, "ref2": ref2, "patched": pat}
    for m in models.values():
        m.to(device).train()
    ptrs = {k: _storage_ptrs(m) for k, m in models.items()}
    for a in ARMS:
        for b in ARMS:
            if a < b and ptrs[a] & ptrs[b]:
                raise AssertionError(f"arms {a} and {b} share parameter storage after .to(device)")
    info["param_count"] = sum(p.numel() for p in ref.parameters())
    return models, info


# ----------------------------------------------------------------------------- alignment and distance
def aligned_params(ref: torch.nn.Module, other: torch.nn.Module) -> tuple[list[tuple[str, torch.Tensor, torch.Tensor]],
                                                                         str]:
    """(name, p_ref, p_other) triples aligned by name when the name sets match, else by order (ref's names)."""
    rn, on = dict(ref.named_parameters()), dict(other.named_parameters())
    if set(rn) == set(on):
        return [(n, rn[n], on[n]) for n in rn], "by_name"
    rl, ol = list(ref.named_parameters()), list(other.named_parameters())
    if len(rl) != len(ol) or any(a.numel() != b.numel() for (_, a), (_, b) in zip(rl, ol)):
        raise AssertionError("parameter names differ AND order alignment fails (count or per-param numel mismatch)")
    return [(n, a, b) for (n, a), (_, b) in zip(rl, ol)], "by_order"


def _module_of(name: str) -> str:
    return name.split(".", 1)[0]


@torch.no_grad()
def param_distance(pairs: list[tuple[str, torch.Tensor, torch.Tensor]]) -> tuple[float, dict[str, float]]:
    """Global and per-top-level-module L2 distance in fp64, accumulated per parameter (no full flat copy)."""
    per_mod: dict[str, float] = {}
    for name, a, b in pairs:
        d = (a.detach().reshape(-1).double() - b.detach().reshape(-1).double()).pow(2).sum().item()
        per_mod[_module_of(name)] = per_mod.get(_module_of(name), 0.0) + d
    total = sum(per_mod.values())
    return math.sqrt(total), {k: math.sqrt(v) for k, v in per_mod.items()}


@torch.no_grad()
def param_norm(pairs: list[tuple[str, torch.Tensor, torch.Tensor]]) -> tuple[float, dict[str, float]]:
    """||w_ref|| globally and per top-level module (fp64)."""
    per_mod: dict[str, float] = {}
    for name, a, _ in pairs:
        per_mod[_module_of(name)] = per_mod.get(_module_of(name), 0.0) + a.detach().double().pow(2).sum().item()
    return math.sqrt(sum(per_mod.values())), {k: math.sqrt(v) for k, v in per_mod.items()}


def params_finite(model: torch.nn.Module) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in model.parameters())


# ----------------------------------------------------------------------------- stepping
def _batch_checksum(batch: Any) -> list[float]:
    """Cheap fingerprint of every tensor leaf, to detect an in-place mutation of the shared batch by step()."""
    return [float(t.detach().double().sum()) if t.numel() else 0.0 for t in tree_leaves(batch)]


def _free(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()


def train_one(target: Target, model: torch.nn.Module, opt: torch.optim.Optimizer, batch: Any,
              rng_seed: int, device: torch.device, clip: float) -> float:
    """One arm, one step: reseed, zero_grad, forward+loss under the policy, backward, [clip], step, free grads.

    Mirrors target.train_step (autocast around forward+loss only, backward outside) with the optimizer step and
    optional clipping added, since train_step itself stops after backward.
    """
    torch.manual_seed(rng_seed)
    opt.zero_grad(set_to_none=True)
    with autocast_ctx(target.dtype_policy, device):
        loss = target.step(model, batch)
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise TypeError(f"target.step must return a scalar tensor, got {type(loss)}")
    loss.backward()
    if clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    opt.step()
    val = float(loss.detach())
    del loss
    opt.zero_grad(set_to_none=True)  # free this arm's grads before the next arm's forward
    _free(device)
    return val


class _EncodeRecorder:
    """Temporarily wrap ``model.encode`` so the eval step's OWN forward (its own autocast/ctx) is recorded."""

    _MISSING = object()

    def __init__(self, model: torch.nn.Module) -> None:
        self.model, self.out = model, None
        self.active = callable(getattr(model, "encode", None))

    def __enter__(self) -> "_EncodeRecorder":
        if self.active:
            self.saved = self.model.__dict__.get("encode", self._MISSING)  # a patch may have set an instance attr
            bound = self.model.encode

            def rec(*a: Any, **k: Any) -> Any:
                out = bound(*a, **k)
                self.out = out
                return out
            self.model.encode = rec  # type: ignore[method-assign]
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.active:
            if self.saved is self._MISSING:
                del self.model.encode
            else:
                self.model.encode = self.saved  # type: ignore[method-assign]


def _det_logits_list(enc_out: Any) -> list[torch.Tensor] | None:
    if not (isinstance(enc_out, (tuple, list)) and len(enc_out) == 2):
        return None
    dl = enc_out[1]
    if isinstance(dl, torch.Tensor):
        return [dl]
    if isinstance(dl, (list, tuple)) and all(isinstance(t, torch.Tensor) for t in dl):
        return list(dl)
    return None


def _det_loss(target: Target, det_logits: list[torch.Tensor], batch: Any) -> float | None:
    """Detection loss via the step's own trainer module, only when every ingredient is present."""
    g = getattr(target.step, "__globals__", {})
    tr, dnw = g.get("_tr"), g.get("DET_NEG_WEIGHT")
    fn = getattr(tr, "compute_detection_loss", None)
    if fn is None or dnw is None or not isinstance(batch, dict) or "coords" not in batch or "masks" not in batch:
        return None
    W = len(det_logits)
    if batch["coords"].dim() < 2 or batch["coords"].shape[1] != W:
        return None
    return float(sum(fn(det_logits[i].float(), batch["coords"][:, i], batch["masks"][:, i], dnw)
                     for i in range(W)) / W)


@torch.no_grad()
def evaluate(target: Target, models: dict[str, torch.nn.Module], device: torch.device,
             n_batches: int) -> dict[str, Any]:
    """Eval-mode step loss (+ peak proxy, + det loss when available) per arm on fixed batches; modes restored."""
    modes = {k: m.training for k, m in models.items()}
    rows: list[dict[str, Any]] = []
    try:
        for m in models.values():
            m.eval()
        for j in range(n_batches):
            bseed = EVAL_SEED_BASE + j
            batch = target.make_batch(device, bseed)
            row: dict[str, Any] = {"eval_batch": j, "batch_seed": bseed}
            for k, m in models.items():
                torch.manual_seed(bseed)
                with _EncodeRecorder(m) as rec, autocast_ctx(target.dtype_policy, device):
                    loss = target.step(m, batch)
                row[f"loss_{k}"] = float(loss)
                dl = _det_logits_list(rec.out) if rec.active else None
                if dl is not None:
                    row[f"peaks_{k}"] = int(sum(int((d > 0).sum()) for d in dl))
                    dloss = _det_loss(target, dl, batch)
                    if dloss is not None:
                        row[f"det_loss_{k}"] = dloss
                del loss, rec
                _free(device)
            rows.append(row)
    finally:
        for k, m in models.items():
            m.train(modes[k])
    return {"rows": rows}


# ----------------------------------------------------------------------------- gate
def _mean_abs(rows: list[dict[str, Any]], key: str, a: str, b: str) -> float | None:
    if not rows or any(f"{key}_{a}" not in r or f"{key}_{b}" not in r for r in rows):
        return None
    return sum(abs(r[f"{key}_{a}"] - r[f"{key}_{b}"]) for r in rows) / len(rows)


def _fmt(v: Any) -> str:
    return f"{v:.3e}" if isinstance(v, float) else str(v)


def run(args: argparse.Namespace) -> int:
    """Fast gate; see module docstring."""
    t0 = time.time()
    device = resolve_device(args.device)
    target = load_target(args.target)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    models, build_info = build_arms(target, args.patch, args.seed, args.ckpt, device)
    opts = {k: _make_opt(args.optimizer, m, args.lr) for k, m in models.items()}
    pairs_pat, align_pat = aligned_params(models["ref"], models["patched"])
    pairs_band, align_band = aligned_params(models["ref"], models["ref2"])
    pairs_x, _ = aligned_params(models["ref2"], models["patched"])  # descriptive: the exchangeable third pair
    if align_pat != "by_name":
        print("WARNING: patched parameter names differ from ref; distances use ORDER alignment "
              "(per-module keys are ref's names)", flush=True)
    # kpatch only compares same-named params; re-check every aligned pair so a renamed param cannot hide a diff.
    for name, a, b in pairs_pat + pairs_band:
        if a.shape != b.shape and a.numel() == b.numel():
            print(f"WARNING: {name}: shape {tuple(a.shape)} vs {tuple(b.shape)} (same numel; compared flat)")
        if not torch.equal(a.detach().reshape(-1), b.detach().reshape(-1)):
            raise AssertionError(f"initial values differ for aligned param {name} ({align_pat})")

    steps = 2 if args.memory_check else args.steps
    rows: list[dict[str, Any]] = []
    dists: list[dict[str, Any]] = []
    nan_step: int | None = None

    def record_dist(n_updates: int) -> None:
        dp, dpm = param_distance(pairs_pat)
        db, dbm = param_distance(pairs_band)
        dx, _ = param_distance(pairs_x)
        dists.append({"updates": n_updates, "d_pat": dp, "d_band": db, "d_pat_vs_ref2": dx,
                      "d_pat_module": dpm, "d_band_module": dbm})

    for i in range(steps):
        bseed = args.seed * BATCH_SEED_STRIDE + i
        batch = target.make_batch(device, bseed)
        cks = _batch_checksum(batch)
        losses: dict[str, float] = {}
        for k in ARMS:
            losses[k] = train_one(target, models[k], opts[k], batch, bseed, device, args.clip_grad)
            cks2 = _batch_checksum(batch)
            if cks2 != cks:
                raise AssertionError(f"step {i}: target.step mutated the shared batch in place (arm {k}); "
                                     "fastgate shares one batch across arms and cannot run this target as-is")
        del batch
        den = max(abs(losses["ref"]), EPS)
        rows.append({"step": i, "loss_ref": losses["ref"], "loss_ref2": losses["ref2"],
                     "loss_pat": losses["patched"],
                     "rel_pat": abs(losses["patched"] - losses["ref"]) / den,
                     "rel_band": abs(losses["ref2"] - losses["ref"]) / den})
        n_upd = i + 1
        if not all(math.isfinite(v) for v in losses.values()):
            nan_step = i
            record_dist(n_upd)
            print(f"non-finite loss at step {i}: {losses}; stopping early", flush=True)
            break
        if (not args.memory_check) and (n_upd % args.dist_every == 0 or n_upd == steps):
            record_dist(n_upd)
        if i % max(1, steps // 10) == 0:
            print(f"step {i:4d} loss_ref={losses['ref']:.6g} rel_pat={rows[-1]['rel_pat']:.3e} "
                  f"rel_band={rows[-1]['rel_band']:.3e}", flush=True)

    peak_gb = (torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else None
    if args.memory_check:
        print(f"memory-check: arms built, {steps} steps run, params/arm={build_info['param_count']}", flush=True)
        print(f"FASTGATE_MEMCHECK peak_gb={peak_gb if peak_gb is None else round(peak_gb, 3)} device={device}")
        return 0

    ev = evaluate(target, models, device, args.eval_batches) if nan_step is None else {"rows": []}
    erows = ev["rows"]

    # ---- conjuncts
    finite = (nan_step is None and all(params_finite(m) for m in models.values())
              and all(math.isfinite(r[f"loss_{k}"]) for r in erows for k in ARMS))
    max_rel_pat = max(r["rel_pat"] for r in rows)
    max_rel_band = max(r["rel_band"] for r in rows)
    final = dists[-1]
    d_pat, d_band = final["d_pat"], final["d_band"]
    w_norm, w_norm_mod = param_norm(pairs_pat)
    bm = args.band_mult
    conj: dict[str, dict[str, Any]] = {}
    conj["finite"] = {"pass": finite, "nan_step": nan_step}
    bar = bm * max(max_rel_band, REL_FLOOR)
    conj["pooled_loss"] = {"pass": max_rel_pat <= bar, "value": max_rel_pat, "band": max_rel_band, "bar": bar}
    bar = bm * max(d_band, DIST_REL_FLOOR * w_norm)
    conj["dist_global"] = {"pass": d_pat <= bar, "value": d_pat, "band": d_band, "bar": bar, "w_ref_norm": w_norm}
    violators, per_mod = [], {}
    for mod, dpm in final["d_pat_module"].items():
        dbm = final["d_band_module"].get(mod, 0.0)
        mbar = bm * max(dbm, DIST_REL_FLOOR * w_norm_mod.get(mod, 0.0))
        per_mod[mod] = {"d_pat": dpm, "d_band": dbm, "bar": mbar, "w_ref_norm": w_norm_mod.get(mod, 0.0),
                        "pass": dpm <= mbar}
        if dpm > mbar:
            violators.append(mod)
    conj["dist_module"] = {"pass": not violators, "violators": violators, "modules": per_mod}
    e_pat, e_band = _mean_abs(erows, "loss", "patched", "ref"), _mean_abs(erows, "loss", "ref2", "ref")
    if e_pat is not None:
        bar = bm * max(e_band, REL_FLOOR)
        conj["eval_loss"] = {"pass": e_pat <= bar, "value": e_pat, "band": e_band, "bar": bar}
    else:
        conj["eval_loss"] = {"pass": False, "reason": "no eval rows (non-finite training or --eval-batches 0)"}
    dl_pat, dl_band = _mean_abs(erows, "det_loss", "patched", "ref"), _mean_abs(erows, "det_loss", "ref2", "ref")
    if dl_pat is not None:
        bar = bm * max(dl_band, REL_FLOOR)
        conj["eval_det_loss"] = {"pass": dl_pat <= bar, "value": dl_pat, "band": dl_band, "bar": bar}
    pk_pat, pk_band = _mean_abs(erows, "peaks", "patched", "ref"), _mean_abs(erows, "peaks", "ref2", "ref")
    if pk_pat is not None:
        bar = bm * max(pk_band, PEAK_FLOOR)
        conj["eval_peaks"] = {"pass": pk_pat <= bar, "value": pk_pat, "band": pk_band, "bar": bar}

    all_pass = all(c["pass"] for c in conj.values())
    degenerate = all(d["d_band"] == 0.0 for d in dists) and all(r["rel_band"] == 0.0 for r in rows)
    exact = (all(d["d_pat"] == 0.0 for d in dists) and all(r["rel_pat"] == 0.0 for r in rows)
             and (e_pat in (None, 0.0)) and (dl_pat in (None, 0.0)) and (pk_pat in (None, 0.0)))
    if not finite:
        verdict = "False"
    elif degenerate:
        verdict = "True" if exact else "INDETERMINATE"
    else:
        verdict = "True" if all_pass else "False"
    if degenerate:
        print(f"DEGENERATE BAND: reference is deterministic (d_band=0, rel_band=0 at every step); "
              f"patched exact={exact} -> verdict {verdict}", flush=True)

    # ---- outputs
    with open(out / "fastgate.csv", "w", newline="") as fh:
        dist_at = {d["updates"]: d for d in dists}
        w = csv.writer(fh)
        w.writerow(["step", "loss_ref", "loss_ref2", "loss_pat", "rel_pat", "rel_band", "d_pat", "d_band"])
        for r in rows:
            d = dist_at.get(r["step"] + 1)
            w.writerow([r["step"], r["loss_ref"], r["loss_ref2"], r["loss_pat"], r["rel_pat"], r["rel_band"],
                        "" if d is None else d["d_pat"], "" if d is None else d["d_band"]])
    dev_name = (torch.cuda.get_device_name(device) if device.type == "cuda"
                else (platform.processor() or platform.machine()))
    summary = {
        "verdict": verdict, "degenerate_band": degenerate, "patched_exact": exact, "conjuncts": conj,
        "target": target.name, "dtype_policy": target.dtype_policy, "patch": args.patch, "args": vars(args),
        "torch_version": torch.__version__, "device": str(device), "device_name": dev_name,
        "cuda_max_memory_allocated_gb": peak_gb, "build": build_info,
        "alignment": {"patched": align_pat, "ref2": align_band},
        "batch_seed_rule": f"train: seed*{BATCH_SEED_STRIDE}+i ; eval: {EVAL_SEED_BASE}+j ; "
                           "torch.manual_seed(batch seed) before every arm's step",
        "steps_run": len(rows), "rows": rows, "distances": dists, "eval": erows,
        "summary": {"max_rel_pat": max_rel_pat, "max_rel_band": max_rel_band, "d_pat": d_pat, "d_band": d_band,
                    "eval_dloss_pat": e_pat, "eval_dloss_band": e_band,
                    "eval_dloss_pat_vs_ref2": _mean_abs(erows, "loss", "patched", "ref2"),
                    "d_pat_vs_ref2": final["d_pat_vs_ref2"], "eval_ddet_pat": dl_pat,
                    "eval_ddet_band": dl_band, "eval_dpeaks_pat": pk_pat, "eval_dpeaks_band": pk_band,
                    "same_step_exceed_count": sum(1 for r in rows if r["rel_pat"] > bm * r["rel_band"])},
        "wall_s": round(time.time() - t0, 2),
    }
    (out / "fastgate.json").write_text(json.dumps(summary, indent=1, default=str))

    for name, c in conj.items():
        extra = {k: v for k, v in c.items() if k not in ("pass", "modules")}
        print(f"conjunct {name:14s} pass={c['pass']} " + " ".join(f"{k}={_fmt(v)}" for k, v in extra.items()))
    print(f"descriptive (exchangeable pair, not gated): d_pat_vs_ref2={final['d_pat_vs_ref2']:.3e} "
          f"eval_dloss_pat_vs_ref2={_fmt(summary['summary']['eval_dloss_pat_vs_ref2'])}")
    print(f"peak_gb={peak_gb} align={align_pat} wall_s={summary['wall_s']} out={out}")
    print(f"FASTGATE_RESULT passed={verdict} steps={len(rows)} max_rel_pat={max_rel_pat:.3e} "
          f"max_rel_band={max_rel_band:.3e} d_pat={d_pat:.3e} d_band={d_band:.3e} "
          f"eval_dloss_pat={_fmt(e_pat)} eval_dloss_band={_fmt(e_band)} "
          f"violators={','.join(violators) if violators else 'none'}")
    return {"True": 0, "False": 1}.get(verdict, 2)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="target file defining build_target()")
    ap.add_argument("--patch", required=True, help="patch file defining apply(model)")
    ap.add_argument("--out", required=True, help="output dir for fastgate.json / fastgate.csv")
    ap.add_argument("--steps", type=int, default=200, help="training steps K")
    ap.add_argument("--seed", type=int, default=0, help="init seed S (all three arms share it)")
    ap.add_argument("--lr", type=float, default=1e-3, help="learning rate (all arms)")
    ap.add_argument("--optimizer", default="adamw", choices=["sgd", "adamw"], help="optimizer type (all arms)")
    ap.add_argument("--clip-grad", type=float, default=0.0,
                    help="max grad norm per arm before opt.step (0 = off; the n85 recipe uses 1.0)")
    ap.add_argument("--band-mult", type=float, default=2.0, help="multiplier on the ref2 nondeterminism band")
    ap.add_argument("--dist-every", type=int, default=25, help="parameter-distance cadence in steps")
    ap.add_argument("--eval-batches", type=int, default=8, help="fixed eval batches after training")
    ap.add_argument("--ckpt", default=None, help="state_dict loaded strict into ref after build_model(seed)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--memory-check", action="store_true", help="build arms, run 2 steps, print peak GB, exit")
    args = ap.parse_args(argv)
    if args.steps < 1 or args.dist_every < 1 or args.eval_batches < 0:
        ap.error("--steps and --dist-every must be >= 1, --eval-batches >= 0")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
