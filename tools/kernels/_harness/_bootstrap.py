"""Make ``_harness`` importable as a package when a CLI is run as a plain script.

Running ``python tools/kernels/_harness/kprofile.py`` puts ``_harness/`` itself at
``sys.path[0]``, so any file in it would shadow a stdlib or third-party module of
the same name (this is why the profiler is ``kprofile.py`` and the patch helpers
``kpatch.py``: ``profile`` is stdlib and is imported by ``cProfile``, which
torch._dynamo pulls in; ``patch`` is a common third-party name). As a second
guard, every CLI calls :func:`fix_path` before importing siblings: it drops the
``_harness`` directory from ``sys.path`` and puts its parent (``tools/kernels``)
there instead, so siblings are imported as ``_harness.<name>``.
"""
from __future__ import annotations

import sys
from pathlib import Path


def fix_path() -> None:
    """Replace the ``_harness`` dir on ``sys.path`` with its parent directory."""
    here = Path(__file__).resolve().parent
    # Remove every spelling of the harness dir that the interpreter may have inserted.
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != here]
    parent = str(here.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
