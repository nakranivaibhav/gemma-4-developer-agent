"""Kernel module for the parity CLI: erf-form GELU (candidate) vs F.gelu (reference)."""
from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

INV_SQRT2 = 1.0 / math.sqrt(2.0)


def candidate(x: torch.Tensor) -> torch.Tensor:
    """Pure-torch exact GELU."""
    return 0.5 * x * (1.0 + torch.erf(x * INV_SQRT2))


def reference(x: torch.Tensor) -> torch.Tensor:
    """The op being replaced."""
    return F.gelu(x)


def make_inputs(shape_spec: Any, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
    """shape_spec is a list of ints, or {"shape": [...], "fill": "zeros"} for an all-zero input."""
    if isinstance(shape_spec, dict):
        shape, fill = shape_spec["shape"], shape_spec.get("fill", "randn")
    else:
        shape, fill = shape_spec, "randn"
    x = torch.zeros(shape, dtype=dtype) if fill == "zeros" else 3.0 * torch.randn(shape, dtype=dtype)
    return [x.to(device).requires_grad_(True)]
