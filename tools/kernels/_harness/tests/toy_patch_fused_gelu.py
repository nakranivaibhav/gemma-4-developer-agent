"""Replace every nn.GELU (exact erf form) with a mathematically identical pure-torch module.

Not bit-identical to the ATen kernel, which is the point: it exercises parity and
drift plumbing with real (tiny) floating-point differences and no Triton.
"""
from __future__ import annotations

import math

import torch
from torch import nn

INV_SQRT2 = 1.0 / math.sqrt(2.0)


class ErfGELU(nn.Module):
    """gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1.0 + torch.erf(x * INV_SQRT2))


def _swap(module: nn.Module) -> None:
    """Recursively replace exact-form nn.GELU children in place."""
    for name, child in module.named_children():
        if isinstance(child, nn.GELU):
            if child.approximate != "none":
                raise ValueError(f"{name}: tanh-approximate GELU is not the op this patch replaces")
            setattr(module, name, ErfGELU())
        else:
            _swap(child)


def apply(model: nn.Module) -> nn.Module:
    """Swap GELUs and return the same model object."""
    _swap(model)
    return model
