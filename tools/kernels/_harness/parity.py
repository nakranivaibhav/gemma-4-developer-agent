"""Op-level parity: candidate at its dtype vs the reference evaluated in fp64.

Library: :func:`check_parity`. CLI: ``--kernel FILE --shapes JSON`` where the
kernel module exposes ``candidate(*inputs)``, ``reference(*inputs)`` and
``make_inputs(shape_spec, device, dtype) -> list[Tensor]``.

Pass = forward AND every input gradient satisfy ``|c - r| <= atol + rtol*|r|``
elementwise (torch.allclose semantics, reference in fp64), no NaN/Inf in the
candidate, AND two candidate runs are bit-identical.
Last stdout line: ``PARITY_RESULT passed=… n_shapes=… worst_rel=… deterministic=…``.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _bootstrap  # noqa: E402
    _bootstrap.fix_path()

import torch  # noqa: E402

TOL: dict[torch.dtype, dict[str, float]] = {
    torch.float32: dict(rtol=1e-5, atol=1e-6),
    torch.bfloat16: dict(rtol=1e-2, atol=1e-3),
    torch.float16: dict(rtol=5e-3, atol=1e-4),
}
REL_EPS = 1e-30  # denominator guard; max_rel is only taken where |ref| > atol/rtol
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class TensorStats:
    """Comparison of one candidate tensor against its fp64 reference."""

    name: str
    shape: list[int]
    max_abs: float
    max_rel: float
    frac_outside_tol: float
    cand_nan: int
    cand_inf: int
    ref_nonfinite: int
    passed: bool


@dataclass
class ParityReport:
    """Result of :func:`check_parity` for one set of inputs."""

    dtype: str
    tol: dict[str, float]
    tensors: list[TensorStats] = field(default_factory=list)
    deterministic: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """All tensors within tolerance and candidate deterministic."""
        return bool(self.tensors) and all(t.passed for t in self.tensors) and self.deterministic

    @property
    def worst_rel(self) -> float:
        """Largest max_rel over all compared tensors (NaN if any tensor had a NaN)."""
        vals = [t.max_rel for t in self.tensors]
        return float("nan") if not vals or any(v != v for v in vals) else max(vals)

    @property
    def worst_abs(self) -> float:
        """Largest max_abs over all compared tensors (NaN if any tensor had a NaN)."""
        vals = [t.max_abs for t in self.tensors]
        return float("nan") if not vals or any(v != v for v in vals) else max(vals)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable view."""
        return {"dtype": self.dtype, "tol": self.tol, "passed": self.passed, "deterministic": self.deterministic,
                "worst_abs": self.worst_abs, "worst_rel": self.worst_rel,
                "tensors": [asdict(t) for t in self.tensors], "notes": self.notes}

    def summary(self) -> str:
        """One-line human summary."""
        bad = [t.name for t in self.tensors if not t.passed]
        return (f"passed={self.passed} dtype={self.dtype} worst_abs={self.worst_abs:.3e} "
                f"worst_rel={self.worst_rel:.3e} deterministic={self.deterministic} failing={bad or '-'}")


def _as_list(out: Any) -> list[torch.Tensor]:
    """Normalise an op output (tensor or tuple/list of tensors) to a list of tensors."""
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)) and all(isinstance(o, torch.Tensor) for o in out):
        return list(out)
    raise TypeError(f"op must return a tensor or a tuple of tensors, got {type(out)}")


def _prep(inputs: Sequence[torch.Tensor], dtype: torch.dtype | None) -> list[torch.Tensor]:
    """Fresh leaf copies of inputs; floating ones cast to ``dtype`` (None keeps dtype); requires_grad preserved."""
    out = []
    for x in inputs:
        y = x.detach().clone()
        if dtype is not None and y.is_floating_point():
            y = y.to(dtype)
        y.requires_grad_(x.requires_grad and y.is_floating_point())
        out.append(y)
    return out


def _run(fn: Callable[..., Any], inputs: Sequence[torch.Tensor], dtype: torch.dtype | None,
         grad_outputs64: list[torch.Tensor] | None, check_grad: bool) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run ``fn`` on prepared inputs; return (outputs, grads w.r.t. inputs that require grad)."""
    xs = _prep(inputs, dtype)
    outs = _as_list(fn(*xs))
    grads: list[torch.Tensor] = []
    wrt = [x for x in xs if x.requires_grad]
    if check_grad and wrt:
        assert grad_outputs64 is not None
        # Same fp64 grad_output for both runs, cast to each output's own dtype.
        gos = [g.to(o.dtype) for g, o in zip(grad_outputs64, outs)]
        grads = list(torch.autograd.grad(outs, wrt, grad_outputs=gos, allow_unused=True))
        grads = [torch.zeros_like(x) if g is None else g for g, x in zip(grads, wrt)]
    return [o.detach() for o in outs], [g.detach() for g in grads]


def _compare(name: str, cand: torch.Tensor, ref: torch.Tensor, tol: dict[str, float]) -> TensorStats:
    """allclose-style comparison of candidate (any dtype) against fp64 reference."""
    if cand.shape != ref.shape:
        return TensorStats(name, list(cand.shape), float("inf"), float("inf"), 1.0, 0, 0, 0, False)
    c = cand.to(torch.float64)
    r = ref.to(torch.float64)
    n_nan = int(torch.isnan(c).sum())
    n_inf = int(torch.isinf(c).sum())
    ref_nonfinite = int((~torch.isfinite(r)).sum())
    diff = (c - r).abs()
    bound = tol["atol"] + tol["rtol"] * r.abs()
    outside = ~(diff <= bound)  # NaN compares False, so NaN counts as outside
    numel = max(c.numel(), 1)
    max_abs = float(diff.max()) if c.numel() else 0.0
    # max_rel is taken only where the rtol term dominates the bound (|ref| > atol/rtol); below that the
    # pass test is governed by atol and a relative error is not meaningful.
    sig = r.abs() > tol["atol"] / tol["rtol"]
    max_rel = float((diff[sig] / (r.abs()[sig] + REL_EPS)).max()) if bool(sig.any()) else 0.0
    if n_nan:  # a NaN in the candidate must not hide behind max() ignoring it
        max_rel = max_abs = float("nan")
    passed = n_nan == 0 and n_inf == 0 and not bool(outside.any())
    return TensorStats(name, list(cand.shape), max_abs, max_rel, float(outside.sum()) / numel,
                       n_nan, n_inf, ref_nonfinite, passed)


def check_parity(candidate: Callable[..., Any], reference: Callable[..., Any], inputs: list[torch.Tensor], *,
                 dtype: torch.dtype, tol: dict[str, float] | None = None, check_grad: bool = True,
                 grad_output: list[torch.Tensor] | torch.Tensor | None = None, seed: int = 0) -> ParityReport:
    """Compare ``candidate`` at ``dtype`` with ``reference`` evaluated in fp64.

    Args:
        candidate: the op under test.
        reference: the op it replaces (run on float64 copies of ``inputs``).
        inputs: input tensors; floating ones with ``requires_grad`` get gradients checked.
        dtype: dtype the candidate runs at (floating inputs are cast to it).
        tol: ``dict(rtol, atol)``; defaults to ``TOL[dtype]``.
        check_grad: also compare input gradients.
        grad_output: upstream gradient(s); default is a seeded standard normal per output.
        seed: seed for the default grad_output.
    """
    tol = dict(tol if tol is not None else TOL[dtype])
    report = ParityReport(dtype=str(dtype).replace("torch.", ""), tol=tol)

    ref_outs, _ = _run(reference, inputs, torch.float64, None, check_grad=False)
    if check_grad:
        if grad_output is None:
            gen = torch.Generator(device="cpu").manual_seed(seed)
            # Generate on CPU for device-independent values, then move next to each output.
            gos64 = [torch.randn(o.shape, generator=gen, dtype=torch.float64).to(o.device) for o in ref_outs]
        else:
            gos64 = [g.detach().to(torch.float64) for g in _as_list(grad_output)]
    else:
        gos64 = None
    ref_outs, ref_grads = _run(reference, inputs, torch.float64, gos64, check_grad)
    cand_outs, cand_grads = _run(candidate, inputs, dtype, gos64, check_grad)
    cand_outs2, cand_grads2 = _run(candidate, inputs, dtype, gos64, check_grad)

    if len(cand_outs) != len(ref_outs):
        report.notes.append(f"output count differs: candidate={len(cand_outs)} reference={len(ref_outs)}")
        return report
    for i, (c, r) in enumerate(zip(cand_outs, ref_outs)):
        report.tensors.append(_compare(f"out[{i}]", c, r, tol))
    for i, (c, r) in enumerate(zip(cand_grads, ref_grads)):
        report.tensors.append(_compare(f"grad_in[{i}]", c, r, tol))
    if check_grad and not cand_grads and any(x.requires_grad for x in inputs):
        report.notes.append("check_grad requested but no gradient was produced")
    pairs = list(zip(cand_outs + cand_grads, cand_outs2 + cand_grads2))
    report.deterministic = all(torch.equal(a, b) for a, b in pairs)
    return report


def _load_kernel(path: str) -> Any:
    """Import a kernel module and check it exposes candidate/reference/make_inputs."""
    from _harness.target import _import_file
    mod = _import_file(path, "kernel")
    for attr in ("candidate", "reference", "make_inputs"):
        if not callable(getattr(mod, attr, None)):
            raise AttributeError(f"{path} must define callable '{attr}'")
    return mod


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", required=True, help="module with candidate, reference, make_inputs")
    ap.add_argument("--shapes", required=True, help="JSON list of shape specs, inline or a path to a .json file")
    ap.add_argument("--dtype", default="float32", choices=sorted(DTYPES), help="candidate dtype")
    ap.add_argument("--device", default="cpu", help="cpu | cuda | cuda:N")
    ap.add_argument("--out", default=None, help="output dir for parity.json (default: <kernel dir>/results)")
    ap.add_argument("--seed", type=int, default=0, help="seed for inputs (seed+i per shape) and grad_output")
    ap.add_argument("--no-grad", action="store_true", help="forward only")
    args = ap.parse_args(argv)

    from _harness.target import resolve_device
    device = resolve_device(args.device)
    dtype = DTYPES[args.dtype]
    shapes_src = Path(args.shapes)
    shapes = json.loads(shapes_src.read_text() if shapes_src.suffix == ".json" and shapes_src.is_file() else args.shapes)
    if not isinstance(shapes, list) or not shapes:
        raise ValueError("--shapes must be a non-empty JSON list")
    mod = _load_kernel(args.kernel)
    out = Path(args.out) if args.out else Path(args.kernel).resolve().parent / "results"
    out.mkdir(parents=True, exist_ok=True)

    results = []
    for i, spec in enumerate(shapes):
        torch.manual_seed(args.seed + i)  # make_inputs may draw from the global RNG; seed it per shape
        inputs = mod.make_inputs(spec, device, dtype)
        rep = check_parity(mod.candidate, mod.reference, inputs, dtype=dtype, check_grad=not args.no_grad,
                           seed=args.seed + i)
        print(f"shape={json.dumps(spec)} {rep.summary()}")
        results.append({"shape_spec": spec, **rep.to_dict()})
    passed = all(r["passed"] for r in results)
    deterministic = all(r["deterministic"] for r in results)
    rels = [r["worst_rel"] for r in results]
    worst_rel = float("nan") if any(v != v for v in rels) else max(rels)  # v != v is the NaN test
    (out / "parity.json").write_text(json.dumps({"kernel": str(Path(args.kernel).resolve()), "dtype": args.dtype,
                                                 "device": str(device), "tol": TOL[dtype], "passed": passed,
                                                 "results": results}, indent=1))
    print(f"PARITY_RESULT passed={passed} n_shapes={len(results)} worst_rel={worst_rel:.3e} "
          f"deterministic={deterministic} out={out / 'parity.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
