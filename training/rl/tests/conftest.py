"""Shared fixtures. CPU tests use synthetic token ids and fake ledger rows only (no task text in the repo).
GPU tests carry @pytest.mark.gpu and skip unless RL_GPU_TESTS=1 and enough GPU memory is free."""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before any torch import
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

NL1, NL2, NL3, CH = 107, 108, 109, 101          # "\n", "\n\n", "\n\n\n", <channel|> (synthetic ids)
NEWLINES = frozenset({NL1, NL2, NL3})


def gpu_free_mb() -> int:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
        return int(out.split()[0])
    except Exception:
        return 0


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs the GPU (set RL_GPU_TESTS=1)")


def pytest_collection_modifyitems(config, items):
    need = int(os.environ.get("RL_GPU_TESTS_MB", "14000"))
    if os.environ.get("RL_GPU_TESTS") == "1" and gpu_free_mb() >= need:
        return
    skip = pytest.mark.skip(reason=f"GPU test: set RL_GPU_TESTS=1 and free >= {need} MB")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


def call(prompt, completion, logprobs=None, utc=0.0, finish="tool_calls"):
    return {"prompt_ids": list(prompt), "completion_ids": list(completion), "utc": utc, "finish_reason": finish,
            "logprobs": [-0.1 * (i + 1) for i in range(len(completion))] if logprobs is None else logprobs}


def ledger_row(run_id, task, resolved_kaggle=False, cut=False, harness_error=None, sha="aaaa", **kw):
    r = {"id": task, "run_id": run_id, "resolved": bool(resolved_kaggle), "resolved_kaggle": resolved_kaggle,
         "kaggle_cut": cut, "harness_error": harness_error,
         "settings": {"file_sha": {"adapters/main_lora/adapter_model.safetensors": sha}}}
    r.update(kw)
    return r
