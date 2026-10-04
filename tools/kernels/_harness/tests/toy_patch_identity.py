"""Identity patch: the control arm; returns the model unchanged."""
from __future__ import annotations

import torch


def apply(model: torch.nn.Module) -> torch.nn.Module:
    """Return the model unchanged."""
    return model
