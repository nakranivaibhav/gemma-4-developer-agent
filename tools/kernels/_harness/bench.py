"""CUDA-event timing: an op microbench helper and a whole-step bench CLI.

Library: :func:`bench_fn` (CUDA only; raises without CUDA), :func:`gpu_is_idle`.
CLI (step mode): ``--target FILE [--patch FILE] --steps N --warmup W --out DIR``
times forward+loss+backward+zero_grad per step for the reference model and, if
given, the patched model (same seed, same batch). Refuses to run on a GPU that
another process is using unless ``--allow-busy`` (numbers from a shared GPU are
not quotable).
Last stdout line: ``BENCH_RESULT ref_ms=… patched_ms=… speedup=… peak_ref_gb=… peak_patched_gb=…``
(or ``BENCH_ABORT gpu_busy=<MiB>``).
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _bootstrap  # noqa: E402
    _bootstrap.fix_path()

import torch  # noqa: E402

IDLE_LIMIT_MIB = 1200  # above this, another process owns the GPU


def _quantile(xs: list[float], q: float) -> float:
    """Nearest-rank quantile of a non-empty list (q in [0, 1])."""
    s = sorted(xs)
    # Index of the q-quantile by nearest rank: ceil(q*n)-1, clamped into [0, n-1].
    idx = min(len(s) - 1, max(0, int(-(-q * len(s) // 1)) - 1))
    return s[idx]


def bench_fn(fn: Callable[[], Any], *, warmup: int = 20, iters: int = 100) -> dict[str, float]:
    """Time ``fn()`` with one CUDA event pair per iteration.

    Returns dict(median_ms, p90_ms, min_ms, mean_ms). Raises RuntimeError without CUDA.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("bench_fn needs CUDA; CPU timings are not a substitute")
    if iters < 1:
        raise ValueError("iters must be >= 1")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()  # every event must have completed before elapsed_time is read
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    return {"median_ms": statistics.median(times), "p90_ms": _quantile(times, 0.90),
            "min_ms": min(times), "mean_ms": statistics.fmean(times), "iters": iters, "warmup": warmup}


def gpu_mem_used_mib() -> int:
    """Max memory.used (MiB) over visible GPUs via nvidia-smi. Raises if nvidia-smi fails."""
    res = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True, timeout=30)
    vals = [int(v.strip()) for v in res.stdout.strip().splitlines() if v.strip()]
    if not vals:
        raise RuntimeError(f"nvidia-smi returned no memory lines: {res.stdout!r}")
    return max(vals)


def gpu_is_idle(limit_mib: int = IDLE_LIMIT_MIB) -> bool:
    """True iff every visible GPU has <= ``limit_mib`` MiB in use."""
    return gpu_mem_used_mib() <= limit_mib


def _step_bench(target: Any, patch_path: str | None, seed: int, device: torch.device,
                warmup: int, steps: int) -> dict[str, Any]:
    """Bench one arm: build, one fixed batch, time train steps; report timing + peak memory."""
    from _harness.kpatch import load_patch
    from _harness.target import train_step
    model = target.build_model(seed)
    if patch_path:
        model = load_patch(patch_path).apply(model)
    model = model.to(device).train()
    batch = target.make_batch(device, seed)

    def one() -> None:
        train_step(target, model, batch, device)
        model.zero_grad(set_to_none=True)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    stats = bench_fn(one, warmup=warmup, iters=steps)
    stats["peak_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1024**3
    del model, batch
    gc.collect()
    torch.cuda.empty_cache()
    return stats


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="target file defining build_target()")
    ap.add_argument("--patch", default=None, help="patch file defining apply(model)")
    ap.add_argument("--steps", type=int, default=50, help="timed steps per arm")
    ap.add_argument("--warmup", type=int, default=10, help="warmup steps per arm")
    ap.add_argument("--out", required=True, help="output dir for bench.json")
    ap.add_argument("--seed", type=int, default=0, help="model + batch seed (same for both arms)")
    ap.add_argument("--device", default="cuda", help="cuda | cuda:N")
    ap.add_argument("--allow-busy", action="store_true", help="run even if the GPU is in use (numbers not quotable)")
    ap.add_argument("--idle-limit-mib", type=int, default=IDLE_LIMIT_MIB, help="busy threshold for the idle check")
    args = ap.parse_args(argv)

    used = gpu_mem_used_mib()  # checked before this process creates a CUDA context
    if used > args.idle_limit_mib and not args.allow_busy:
        print(f"BENCH_ABORT gpu_busy={used}")
        return 2
    if not torch.cuda.is_available():
        raise RuntimeError("bench CLI needs CUDA")
    from _harness.target import load_target
    device = torch.device(args.device)
    target = load_target(args.target)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ref = _step_bench(target, None, args.seed, device, args.warmup, args.steps)
    print(f"ref     median={ref['median_ms']:.3f} ms p90={ref['p90_ms']:.3f} peak={ref['peak_mem_gb']:.3f} GB")
    pat = None
    if args.patch:
        pat = _step_bench(target, args.patch, args.seed, device, args.warmup, args.steps)
        print(f"patched median={pat['median_ms']:.3f} ms p90={pat['p90_ms']:.3f} peak={pat['peak_mem_gb']:.3f} GB")
    speedup = ref["median_ms"] / pat["median_ms"] if pat else None
    rec = {"target": target.name, "patch": args.patch, "device": str(device), "dtype_policy": target.dtype_policy,
           "gpu_mem_used_mib_before": used, "allow_busy": args.allow_busy, "quotable": used <= args.idle_limit_mib,
           "reference": ref, "patched": pat, "speedup": speedup}
    (out / "bench.json").write_text(json.dumps(rec, indent=1))
    fmt = lambda v, f: f"{v:{f}}" if v is not None else "na"  # noqa: E731
    print(f"BENCH_RESULT ref_ms={ref['median_ms']:.3f} patched_ms={fmt(pat and pat['median_ms'], '.3f')} "
          f"speedup={fmt(speedup, '.4f')} peak_ref_gb={ref['peak_mem_gb']:.3f} "
          f"peak_patched_gb={fmt(pat and pat['peak_mem_gb'], '.3f')} quotable={rec['quotable']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
