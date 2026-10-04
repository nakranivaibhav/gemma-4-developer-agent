"""Toy target for harness tests: Linear -> GroupNorm -> GELU -> Linear on random regression data."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # tools/kernels, so '_harness' imports as a package
from _harness.target import Target  # noqa: E402

IN, HIDDEN, OUT, GROUPS, BATCH = 16, 32, 1, 4, 8  # tiny on purpose; HIDDEN must divide by GROUPS


def build_model(seed: int) -> nn.Module:
    """Deterministic init without disturbing the caller's global RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return nn.Sequential(nn.Linear(IN, HIDDEN), nn.GroupNorm(GROUPS, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, OUT))


def make_batch(device: torch.device, seed: int) -> dict[str, torch.Tensor]:
    """x ~ N(0,1) of shape (BATCH, IN); y = sin(sum(x)) of shape (BATCH, OUT). Drawn on CPU, then moved."""
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(BATCH, IN, generator=gen)
    y = torch.sin(x.sum(dim=1, keepdim=True))
    return {"x": x.to(device), "y": y.to(device)}


def step(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """MSE loss; forward only."""
    return nn.functional.mse_loss(model(batch["x"]).float(), batch["y"])


def build_target() -> Target:
    """Target protocol entry point."""
    return Target(name="toy_mlp_gn_gelu", build_model=build_model, make_batch=make_batch, step=step,
                  dtype_policy="fp32", notes="harness self-test target")
