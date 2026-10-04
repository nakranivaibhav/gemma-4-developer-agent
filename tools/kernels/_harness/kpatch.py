"""Patch protocol: a patch file defines ``apply(model) -> model`` and swaps ops for kernel versions.

The reference model code is never edited; a patch mutates (or wraps) a model
instance after it is built.
"""
from __future__ import annotations

from types import ModuleType

import torch

from _harness.target import _import_file


def load_patch(path: str) -> ModuleType:
    """Import a patch file by path; it must expose a callable ``apply(model) -> model``."""
    module = _import_file(path, "patch")
    fn = getattr(module, "apply", None)
    if fn is None or not callable(fn):
        raise AttributeError(f"{path} must define a callable apply(model) -> model")
    return module


def identity_patch(model: torch.nn.Module) -> torch.nn.Module:
    """The null patch: return the model unchanged (a control arm for the harness)."""
    return model


def param_count(model: torch.nn.Module) -> int:
    """Total number of parameter elements."""
    return sum(p.numel() for p in model.parameters())


def assert_same_param_count(ref: torch.nn.Module, patched: torch.nn.Module) -> None:
    """Raise AssertionError unless both models hold the same number of parameter elements."""
    n_ref, n_pat = param_count(ref), param_count(patched)
    if n_ref != n_pat:
        raise AssertionError(f"patch changed the parameter count: ref={n_ref} patched={n_pat}")


def assert_same_initial_values(ref: torch.nn.Module, patched: torch.nn.Module) -> int:
    """For parameters present by name in both, require identical shape and values. Returns the number compared."""
    ref_p = dict(ref.named_parameters())
    n = 0
    for name, p in patched.named_parameters():
        if name not in ref_p:
            continue
        r = ref_p[name]
        if r.shape != p.shape:
            raise AssertionError(f"param {name}: shape {tuple(r.shape)} vs patched {tuple(p.shape)}")
        if not torch.equal(r.detach().cpu(), p.detach().cpu()):
            raise AssertionError(f"param {name}: initial values differ after patch")
        n += 1
    return n
