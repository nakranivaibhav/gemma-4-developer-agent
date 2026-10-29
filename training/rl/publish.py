"""Make the agent folder a step rolls out with: a copy of the base agent whose adapters/main_lora holds the
step's policy adapter. hist.sh serves that folder (SERVE_AGENT) and run_tasks.py pins every file's sha into each
ledger row, which is what the on-policy guard compares against.

v1 restarts the vLLM server each step (hist.sh starts and stops it; ~1-2 min against ~25 min of rollouts) so the
GPU is free for the trainer in between. Runtime adapter loading (VLLM_ALLOW_RUNTIME_LORA_UPDATING +
/v1/load_lora_adapter) and sleep mode exist in vLLM 0.19.1 and are the next optimisation, not used yet.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")


def sha16(path: str | Path) -> str:
    """Same digest run_tasks.py pins: sha256 of the file, first 16 hex chars."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def make_agent(base_agent: str | Path, adapter_dir: str | Path, dest: str | Path) -> str:
    base_agent, adapter_dir, dest = Path(base_agent), Path(adapter_dir), Path(dest)
    for f in ADAPTER_FILES:
        if not (adapter_dir / f).is_file():
            raise FileNotFoundError(adapter_dir / f)
    if not (base_agent / "agent.yaml").is_file():
        raise FileNotFoundError(base_agent / "agent.yaml")
    tmp = dest.with_name(dest.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(base_agent, tmp, ignore=shutil.ignore_patterns("adapters", "__pycache__"))
    lora = tmp / "adapters" / "main_lora"
    lora.mkdir(parents=True)
    for f in ADAPTER_FILES:
        shutil.copy2(adapter_dir / f, lora / f)
    sha = sha16(lora / "adapter_model.safetensors")
    if sha != sha16(adapter_dir / "adapter_model.safetensors"):
        raise IOError(f"adapter copy differs from {adapter_dir}")
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)                           # the folder appears complete or not at all
    return sha
