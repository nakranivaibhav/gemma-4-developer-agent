"""Start the competition model under vLLM with the starter notebook's settings (tp=1 locally).

Usage: envs/harness/bin/python -I kits/swe-agent/serve/serve_model.py <model_dir> <agent_dir> [--print]
Execs the vLLM server in place, so the caller owns one process.
"""
import time
import json
import os
import shlex
import sys
from pathlib import Path

import torch
from adk_submission import VllmConfig, VllmServer, discover_adapters
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

model_dir, agent_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
cfg = VllmConfig(
    model=str(model_dir),
    port=8000,
    host="127.0.0.1",
    # TOOL_PARSER / REASONING_PARSER / ENABLE_LORA=0: serve a non-Gemma reference model (Qwen: qwen3_coder / qwen3)
    tool_call_parser=os.environ.get("TOOL_PARSER", "gemma4"),
    reasoning_parser=os.environ.get("REASONING_PARSER", "gemma4"),
    max_model_len=32768,
    dtype="bfloat16" if torch.cuda.is_bf16_supported() else "auto",
    # 0.90 + the default 256 request slots ran out of memory on one 32 GB card (sampler warm-up), see logs
    gpu_memory_utilization=float(os.environ.get("GPU_UTIL", "0.88")),
    # README 3.1: the scoring server defaults thinking ON for requests that do not say either way
    default_chat_template_kwargs={"enable_thinking": True},
    enable_auto_tool_choice=True,
    enable_lora=os.environ.get("ENABLE_LORA", "1") == "1",
    max_loras=8,
    max_lora_rank=128,
    tensor_parallel_size=1,
    startup_timeout=60 * 20,
    # VLLM_EXTRA: more server flags for speed experiments, e.g. "--kv-offloading-size 40"
    extra_args=["--max-num-seqs", os.environ.get("MAX_NUM_SEQS", "4")] + shlex.split(os.environ.get("VLLM_EXTRA", "")),
)
server = VllmServer(cfg, adapter_manifest=adapters)
cmd, env = server.build_cmd(), server.build_env()
# VLLM_PY: run the server from another env's python (envs/qwen has vLLM 0.31.0: 0.19.1 cannot load the Qwen3.8
# NVFP4 checkpoint's quantized lm_head, "no parameter lm_head.input_scale", 2026-10-15). Gemma runs never set it.
if os.environ.get("VLLM_PY"):
    cmd[0] = os.environ["VLLM_PY"]
if "--print" not in sys.argv:
    # leave a note of this server's settings for the run ledger (kits/swe-agent/run/pin.py reads it)
    root = Path(__file__).resolve().parents[3]  # the repo root
    note = root / "runs" / "logs" / "current_server.json"  # same path whatever the cwd
    note.parent.mkdir(parents=True, exist_ok=True)
    shown = Path(os.path.abspath(sys.argv[1]))  # as given, symlinks not followed: recorded relative to the repo root
    note.write_text(json.dumps({
        "model_dir": str(shown.relative_to(root) if shown.is_relative_to(root) else shown), "gpu_memory_utilization": cfg.gpu_memory_utilization, "max_num_seqs": os.environ.get("MAX_NUM_SEQS", "4"),
        "extra_flags": os.environ.get("VLLM_EXTRA", ""), "started_utc": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
    }, indent=1))
print("CMD", " ".join(cmd), flush=True)
if "--print" not in sys.argv:
    os.execve(cmd[0], cmd, env)
