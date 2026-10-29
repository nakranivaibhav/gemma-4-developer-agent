"""Share the GPU with the other pipelines on this machine (the outside-task screen loop).

Contract (kits/swe-agent/outside/pass_pipeline.sh): while /tmp/gpu_hold exists, a screen batch does not start its
vLLM server. hist.sh kills ANY vLLM on port 8000 when it starts, so before a rollout we must also be sure no other
server is up or starting: hold first, then wait until no vLLM process is seen twice in a row and memory is free.
"""
from __future__ import annotations

import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

HOLD = Path("/tmp/gpu_hold")


def free_mb() -> int:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"])
    return int(out.split()[0])


def vllm_running() -> bool:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    return any("vllm.entrypoints" in line for line in out.splitlines())


def wait_free(need_mb: int, poll_s: int = 60, log=print, free=free_mb, running=vllm_running, sleep=time.sleep):
    quiet = 0
    while True:
        busy = running()
        mb = free()
        quiet = quiet + 1 if (not busy and mb >= need_mb) else 0
        if quiet >= 2:
            return
        log(f"GPU_WAIT free={mb}MB vllm={'up' if busy else 'down'} need={need_mb}MB")
        sleep(poll_s)


@contextmanager
def hold(enabled: bool = True):
    """Touch /tmp/gpu_hold for the duration; remove it only if we created it."""
    mine = enabled and not HOLD.exists()
    if mine:
        HOLD.write_text(f"training/rl pid {os.getpid()}\n")
    try:
        yield
    finally:
        if mine and HOLD.exists() and HOLD.read_text().startswith(f"training/rl pid {os.getpid()}"):
            HOLD.unlink()
