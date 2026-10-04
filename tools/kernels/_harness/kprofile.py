"""Profile one training step of a target (optionally patched).

Runs W warmup steps, then N steps under ``torch.profiler`` (forward, loss,
backward, ``zero_grad``), then N *unprofiled* timed steps for the wall-clock
numbers (the profiler adds overhead, so step time is never read from it).

Outputs in --out: profile_table.txt, profile_ops.json, trace.json, summary.json.
Last stdout line: ``PROFILE_RESULT step_ms=… gpu_busy=… peak_gb=… verdict=… out=DIR``.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _bootstrap  # noqa: E402
    _bootstrap.fix_path()

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile as torch_profile  # noqa: E402

from _harness.kpatch import load_patch  # noqa: E402
from _harness.target import (Target, load_target, resolve_device, train_step,  # noqa: E402
                             tree_leaves)

BUSY_THRESHOLD = 0.85  # below this kernel-time / wall-time fraction the GPU is waiting on something else


def _self_device_us(evt: Any) -> float:
    """Self device (CUDA) time in microseconds for a key_averages row, across torch versions."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        val = getattr(evt, attr, None)
        if val is not None:
            return float(val)
    return 0.0


def _build(target: Target, patch_path: str | None, seed: int, device: torch.device) -> torch.nn.Module:
    """Build the model at ``seed``, apply the patch if given, move to device, train mode."""
    model = target.build_model(seed)
    if patch_path:
        model = load_patch(patch_path).apply(model)
    return model.to(device).train()


def _sync(device: torch.device) -> None:
    """Synchronise the device so host timers see finished work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_steps(target: Target, model: torch.nn.Module, batch: Any, device: torch.device, n: int) -> list[float]:
    """Wall time (ms) of ``n`` unprofiled steps: CUDA events on GPU, perf_counter on CPU."""
    times: list[float] = []
    for _ in range(n):
        if device.type == "cuda":
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            train_step(target, model, batch, device)
            model.zero_grad(set_to_none=True)
            end.record()
            torch.cuda.synchronize(device)  # must finish before elapsed_time is valid
            times.append(start.elapsed_time(end))
        else:
            t0 = time.perf_counter()
            train_step(target, model, batch, device)
            model.zero_grad(set_to_none=True)
            times.append((time.perf_counter() - t0) * 1e3)
    return times


def dry_run(target: Target, patch_path: str | None, seed: int) -> int:
    """Build model, one batch, one step on CPU; print shapes; return exit code."""
    device = torch.device("cpu")
    model = _build(target, patch_path, seed, device)
    batch = target.make_batch(device, seed)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"target={target.name} policy={target.dtype_policy} params={n_params}")
    for i, leaf in enumerate(tree_leaves(batch)):
        print(f"  batch[{i}] shape={tuple(leaf.shape)} dtype={leaf.dtype}")
    loss = train_step(target, model, batch, device)
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    finite = bool(torch.isfinite(loss).item())
    print(f"  loss={loss.item():.6g} finite={finite} params_with_grad={n_grad}/{len(list(model.parameters()))}")
    print(f"PROFILE_RESULT passed={finite} dry_run=True loss={loss.item():.6g} params={n_params}")
    return 0 if finite else 1


def run(args: argparse.Namespace) -> int:
    """Full profile run; see module docstring."""
    target = load_target(args.target)
    if args.dry_run:
        return dry_run(target, args.patch, args.seed)
    device = resolve_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    model = _build(target, args.patch, args.seed, device)
    batch = target.make_batch(device, args.seed)

    for _ in range(args.warmup):
        train_step(target, model, batch, device)
        model.zero_grad(set_to_none=True)
    _sync(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with torch_profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        for _ in range(args.steps):
            train_step(target, model, batch, device)
            model.zero_grad(set_to_none=True)
        _sync(device)
    prof.export_chrome_trace(str(out / "trace.json"))

    # Unprofiled timing + peak memory measured on these steps only.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = _timed_steps(target, model, batch, device, args.steps)
    step_ms = statistics.median(times)
    peak_gb = torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None

    sort_key = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    avgs = prof.key_averages(group_by_input_shape=True)
    (out / "profile_table.txt").write_text(avgs.table(sort_by=sort_key, row_limit=60))

    # Kernel time per step = sum of self device time over all rows / profiled steps.
    # Summing SELF time avoids double counting parent ops that include child kernels.
    rows = []
    total_self_dev_us = 0.0
    for evt in avgs:
        self_us = _self_device_us(evt)
        total_self_dev_us += self_us
        rows.append((evt, self_us))
    kernel_ms_per_step = total_self_dev_us / 1e3 / max(args.steps, 1)

    # Merge rows by op name (shapes were grouped separately above) for the per-op json.
    by_name: dict[str, dict[str, Any]] = {}
    for evt, self_us in rows:
        d = by_name.setdefault(evt.key, {"name": evt.key, "self_cuda_us": 0.0, "self_cpu_us": 0.0,
                                          "calls": 0, "shapes": []})
        d["self_cuda_us"] += self_us
        d["self_cpu_us"] += float(evt.self_cpu_time_total)
        d["calls"] += int(evt.count)
        if evt.input_shapes and evt.input_shapes not in d["shapes"]:
            d["shapes"].append(evt.input_shapes)
    ops = []
    for d in by_name.values():
        # Per-step self time of this op; its share is taken against the unprofiled wall step.
        per_step_ms = (d["self_cuda_us"] if device.type == "cuda" else d["self_cpu_us"]) / 1e3 / max(args.steps, 1)
        pct = per_step_ms / step_ms if step_ms > 0 else 0.0
        pct = min(pct, 0.999999)  # profiler-vs-unprofiled mismatch can push >1; cap so Amdahl stays finite
        ops.append({
            "name": d["name"],
            "self_cuda_ms": d["self_cuda_us"] / 1e3 / max(args.steps, 1),
            "self_cpu_ms": d["self_cpu_us"] / 1e3 / max(args.steps, 1),
            "calls": d["calls"] // max(args.steps, 1),
            "pct_of_step": pct,
            "shapes": d["shapes"],
            "amdahl_max_speedup": 1.0 / (1.0 - pct),
        })
    ops.sort(key=lambda o: o["pct_of_step"], reverse=True)
    (out / "profile_ops.json").write_text(json.dumps(ops, indent=1))

    if device.type == "cuda":
        busy = kernel_ms_per_step / step_ms if step_ms > 0 else 0.0
        verdict = "input_bound" if busy < BUSY_THRESHOLD else "gpu_bound"
    else:
        busy, verdict = None, "cpu_only_no_gpu_verdict"
    summary = {
        "target": target.name, "patch": args.patch, "device": str(device), "dtype_policy": target.dtype_policy,
        "warmup": args.warmup, "steps": args.steps, "seed": args.seed,
        "step_ms_median": step_ms, "step_ms_all": times,
        "kernel_ms_per_step": kernel_ms_per_step if device.type == "cuda" else None,
        "gpu_busy_fraction": busy, "peak_mem_gb": peak_gb, "verdict": verdict,
        "busy_threshold": BUSY_THRESHOLD,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    print(f"{'op':60s} {'self_ms/step':>12s} {'pct':>7s} {'amdahl':>7s} {'calls':>6s}")
    for o in ops[:15]:
        ms = o["self_cuda_ms"] if device.type == "cuda" else o["self_cpu_ms"]
        print(f"{o['name'][:60]:60s} {ms:12.3f} {100*o['pct_of_step']:6.2f}% {o['amdahl_max_speedup']:7.3f} {o['calls']:6d}")
    busy_s = f"{busy:.3f}" if busy is not None else "na"
    peak_s = f"{peak_gb:.3f}" if peak_gb is not None else "na"
    print(f"PROFILE_RESULT step_ms={step_ms:.3f} gpu_busy={busy_s} peak_gb={peak_s} verdict={verdict} out={out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="target file defining build_target()")
    ap.add_argument("--patch", default=None, help="optional patch file defining apply(model)")
    ap.add_argument("--steps", type=int, default=25, help="profiled steps (also the number of timed steps)")
    ap.add_argument("--warmup", type=int, default=5, help="warmup steps before profiling")
    ap.add_argument("--out", default=None, help="output directory (required unless --dry-run)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    ap.add_argument("--seed", type=int, default=0, help="model + batch seed")
    ap.add_argument("--dry-run", action="store_true", help="one CPU step, print shapes, exit")
    args = ap.parse_args(argv)
    if not args.dry_run and not args.out:
        ap.error("--out is required unless --dry-run")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
