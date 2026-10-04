#!/usr/bin/env python
"""Run an UNMODIFIED trainer script with a kernel patch applied to its model right after construction.

    python tools/kernels/train_with_patch.py --trainer <trainer.py> --patch <patch.py> [--model-class NAME] \
        -- <trainer args...>

The trainer file is imported by path under its own stem as module name (so patches that look up
sys.modules[model.__module__] see it), its model class is wrapped in a subclass whose __init__ calls the
original __init__ and then patch.apply(self), and the trainer's main() runs with sys.argv = [trainer, args...].
Nothing on disk is edited. The patch protocol is tools/kernels/_harness/kpatch.py (apply(model) -> model);
patches that re-class the model in place (model.__class__ = ...) work because __init__ runs on the instance.
Prints ONE line before main(): KPATCH_ACTIVE trainer=<md5> patch=<path> patch_md5=<md5> model_class=<name>
and, after the first construction, KPATCH_APPLIED classes=<sorted set of re-classed module type names>.
Everything else (checkpoints, logs, RESUME) is the trainer's own: state_dict keys are unchanged by a patch
that adds no parameters (asserted).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path


def _md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def _import(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod            # register BEFORE exec (dataclasses / patches look the module up)
    spec.loader.exec_module(mod)
    return mod


def wrap(trainer: Path, patch_path: Path, model_class: str):
    """Import trainer + patch, replace trainer.<model_class> by the self-patching subclass; return the trainer module."""
    patch = _import(patch_path, "kpatch_" + patch_path.stem)
    assert callable(getattr(patch, "apply", None)), f"{patch_path} has no apply(model)"
    tr = _import(trainer, trainer.stem)            # module name = file stem (e.g. train_unet_transformer)
    Orig = getattr(tr, model_class)

    class Patched(Orig):                            # noqa: D101
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            before = sum(p.numel() for p in self.parameters())
            keys = list(self.state_dict().keys())
            patch.apply(self)
            after = sum(p.numel() for p in self.parameters())
            assert before == after, f"patch changed the parameter count {before} -> {after}"
            assert keys == list(self.state_dict().keys()), "patch changed state_dict keys"
            # re-classed modules are subclasses named <Base><Suffix> created by the patches
            classes = sorted({type(m).__name__ for m in self.modules()
                              if not type(m).__module__.startswith("torch.")
                              and type(m).__name__ != type(m).__mro__[1].__name__
                              and type(m).__name__.startswith(type(m).__mro__[1].__name__)})
            print(f"KPATCH_APPLIED classes={classes}", flush=True)

    Patched.__name__ = Orig.__name__                # patches that check the class NAME still match
    Patched.__qualname__ = Orig.__qualname__
    setattr(tr, model_class, Patched)
    return tr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trainer", required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--model-class", default="UNetNodeTransformer")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="trainer args after --")
    a = ap.parse_args()
    rest = a.rest[1:] if a.rest[:1] == ["--"] else a.rest
    trainer = Path(a.trainer).resolve()
    patch_path = Path(a.patch).resolve()
    print(f"KPATCH_ACTIVE trainer={_md5(trainer)} patch={patch_path} patch_md5={_md5(patch_path)} "
          f"model_class={a.model_class}", flush=True)
    tr = wrap(trainer, patch_path, a.model_class)
    sys.argv = [str(trainer)] + rest
    tr.main()


if __name__ == "__main__":
    main()
