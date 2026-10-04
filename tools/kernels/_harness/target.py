"""Target protocol: the only thing the harness knows about a model.

A *target file* is any Python file defining ``build_target() -> Target``. The
harness never imports a trainer or a repository layout; everything it needs
comes through these five fields.
"""
from __future__ import annotations

import contextlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, ContextManager

import torch

DTYPE_POLICIES = ("fp32", "bf16_autocast", "fp16_autocast")


@dataclass
class Target:
    """What the harness needs to run one training step of a reference model.

    Attributes:
        name: short identifier used in reports.
        build_model: seed -> fresh model on CPU with deterministic init.
        make_batch: (device, seed) -> one batch (any pytree of tensors), deterministic in seed.
        step: (model, batch) -> scalar loss. Forward + loss only; NO backward inside.
        dtype_policy: 'fp32' | 'bf16_autocast' | 'fp16_autocast'.
        notes: free text (interpreter, real batch size, caveats).
    """

    name: str
    build_model: Callable[[int], torch.nn.Module]
    make_batch: Callable[[torch.device, int], Any]
    step: Callable[[torch.nn.Module, Any], torch.Tensor]
    dtype_policy: str
    notes: str = ""

    def __post_init__(self) -> None:
        if self.dtype_policy not in DTYPE_POLICIES:
            raise ValueError(f"dtype_policy={self.dtype_policy!r}; expected one of {DTYPE_POLICIES}")


def _import_file(path: str, prefix: str) -> Any:
    """Import a Python file by path under a unique module name and return the module."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"{prefix} file not found: {p}")
    mod_name = f"_kharness_{prefix}_{p.stem}_{abs(hash(str(p)))}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module  # needed so dataclasses/pickle inside the file can resolve the module
    spec.loader.exec_module(module)
    return module


def load_target(path: str) -> Target:
    """Import ``path`` and return the :class:`Target` produced by its ``build_target()``."""
    module = _import_file(path, "target")
    fn = getattr(module, "build_target", None)
    if fn is None or not callable(fn):
        raise AttributeError(f"{path} must define a callable build_target() -> Target")
    tgt = fn()
    # Duck-type check rather than isinstance: the target file may import this module under another name.
    for field in ("name", "build_model", "make_batch", "step", "dtype_policy"):
        if not hasattr(tgt, field):
            raise TypeError(f"build_target() in {path} returned an object without '{field}'")
    if tgt.dtype_policy not in DTYPE_POLICIES:
        raise ValueError(f"{path}: dtype_policy={tgt.dtype_policy!r}; expected one of {DTYPE_POLICIES}")
    return tgt


def autocast_ctx(policy: str, device: torch.device | str) -> ContextManager[Any]:
    """Return the autocast context for a dtype policy (a nullcontext for fp32)."""
    dev = torch.device(device)
    if policy == "fp32":
        return contextlib.nullcontext()
    if policy == "bf16_autocast":
        return torch.autocast(device_type=dev.type, dtype=torch.bfloat16)
    if policy == "fp16_autocast":
        return torch.autocast(device_type=dev.type, dtype=torch.float16)
    raise ValueError(f"unknown dtype policy {policy!r}; expected one of {DTYPE_POLICIES}")


def flat_grads(model: torch.nn.Module) -> torch.Tensor:
    """Concatenate every parameter's ``.grad`` (zeros where None) in parameter order, as fp32 on CPU-or-param device."""
    parts: list[torch.Tensor] = []
    for p in model.parameters():
        if p.grad is None:
            parts.append(torch.zeros(p.numel(), dtype=torch.float32, device=p.device))
        else:
            parts.append(p.grad.detach().reshape(-1).float())
    if not parts:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(parts)


def named_flat_grads(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Per-parameter flattened fp32 grads keyed by parameter name (zeros where None)."""
    out: dict[str, torch.Tensor] = {}
    for name, p in model.named_parameters():
        g = p.grad
        out[name] = (torch.zeros(p.numel(), dtype=torch.float32, device=p.device) if g is None
                     else g.detach().reshape(-1).float())
    return out


def tree_map(fn: Callable[[torch.Tensor], Any], tree: Any) -> Any:
    """Apply ``fn`` to every tensor leaf of a nested list/tuple/dict; other leaves pass through."""
    if isinstance(tree, torch.Tensor):
        return fn(tree)
    if isinstance(tree, dict):
        return type(tree)((k, tree_map(fn, v)) for k, v in tree.items())
    if isinstance(tree, tuple) and hasattr(tree, "_fields"):  # namedtuple: rebuild positionally
        return type(tree)(*(tree_map(fn, v) for v in tree))
    if isinstance(tree, (list, tuple)):
        return type(tree)(tree_map(fn, v) for v in tree)
    return tree


def tree_to(batch: Any, device: torch.device | str) -> Any:
    """Move every tensor in a nested batch to ``device``."""
    dev = torch.device(device)
    return tree_map(lambda t: t.to(dev), batch)


def tree_leaves(tree: Any) -> list[torch.Tensor]:
    """Return the tensor leaves of a nested batch in traversal order."""
    leaves: list[torch.Tensor] = []
    tree_map(lambda t: leaves.append(t), tree)
    return leaves


def resolve_device(name: str | None) -> torch.device:
    """'auto'/None -> cuda if available else cpu; otherwise ``torch.device(name)``."""
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    return dev


def train_step(target: Target, model: torch.nn.Module, batch: Any, device: torch.device) -> torch.Tensor:
    """One step as the harness defines it: forward+loss under the policy, backward outside autocast.

    Returns the detached loss. Grads are left on the parameters (callers zero them).
    """
    with autocast_ctx(target.dtype_policy, device):
        loss = target.step(model, batch)
    if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
        raise TypeError(f"target.step must return a scalar tensor, got {type(loss)} "
                        f"with shape {getattr(loss, 'shape', None)}")
    loss.backward()
    return loss.detach()
