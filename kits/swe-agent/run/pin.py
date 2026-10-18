"""Pin the settings of an evaluation run: everything that could change an agent's behaviour, in one flat dict.

usage: python -I kits/swe-agent/run/pin.py <agent_dir>          (prints the dict; run_tasks.py embeds it in every ledger line)

What goes in: the agent's model/adapter/tools, the sampling numbers, the per-task budgets, a sha256 + size of every
prompt and config file, the server settings (from runs/logs/current_server.json, written by serve_model.py),
and library versions. The agent folder itself is copied next to the run's results as the full pin.
"""
import hashlib
import json
import os
import shutil
import sys
from importlib import metadata
from pathlib import Path

import yaml


class _Loader(yaml.SafeLoader):
    pass


def _include(loader, node):  # the agent yaml uses !include for prompts and sampling; keep the path, not the text
    return {"!include": loader.construct_scalar(node)}


_Loader.add_constructor("!include", _include)
TEXT = (".yaml", ".yml", ".md", ".json", ".txt")
WEIGHTS = (".safetensors", ".bin", ".pt")


def _load(path):
    return yaml.load(path.read_text(), Loader=_Loader) or {}


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


ROOT = Path(__file__).resolve().parents[3]  # the repo root
# SERVER_JSON: point a run at another server note, e.g. a rented remote server reached through a tunnel
SERVER_JSON = Path(os.environ.get("SERVER_JSON") or ROOT / "runs" / "logs" / "current_server.json")


def rel(path):
    """A path as stored in the ledger: relative to the repo root when it is inside it, so rows read the same on any machine.
    Symlinks are not followed: data/ or envs/ may be links to another disk and still count as inside the repo."""
    path = Path(os.path.abspath(path))
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def pin_settings(agent_dir, server_json=SERVER_JSON):
    agent_dir = Path(agent_dir)
    agent = _load(agent_dir / "agent.yaml")
    sampling = _load(agent_dir / "configs" / "sampling.yaml") if (agent_dir / "configs" / "sampling.yaml").exists() else {}
    ev = (_load(agent_dir / "eval_config.yaml") if (agent_dir / "eval_config.yaml").exists() else {}).get("evaluation", {})
    subs = []
    for t in agent.get("tools", []):
        if isinstance(t, dict) and "agent_tool" in t:
            sub = _load(agent_dir / t["agent_tool"]["config_path"])
            subs.append({"name": sub.get("name"), "adapter": sub.get("adapter"), "tools": sub.get("tools", [])})
    files = {}
    for p in sorted(agent_dir.rglob("*")):
        if p.is_file() and p.suffix in TEXT + WEIGHTS:
            files[str(p.relative_to(agent_dir))] = {"sha": _sha(p), "bytes": p.stat().st_size}
    server = {}
    if os.path.exists(server_json):
        server = json.load(open(server_json))
    return {
        "agent_name": agent.get("name"),
        "model": agent.get("model"),
        "adapter": agent.get("adapter"),
        "tools": [t if isinstance(t, str) else "agent_tool:" + t["agent_tool"]["config_path"] for t in agent.get("tools", [])],
        "sub_agents": subs,
        "temperature": sampling.get("temperature"),
        "top_p": sampling.get("top_p"),
        "max_output_tokens": sampling.get("max_output_tokens"),
        "thinking_budget": (sampling.get("thinking_config") or {}).get("thinking_budget"),
        "include_thoughts": (sampling.get("thinking_config") or {}).get("include_thoughts"),
        "budget_max_tool_calls": ev.get("max_tool_calls"),
        "budget_max_time_minutes": ev.get("max_time_minutes"),
        "budget_max_turns": ev.get("max_turns"),
        "budget_command_timeout_s": ev.get("timeout_seconds"),
        "files": files,
        "server": server,
        "versions": {k: metadata.version(k) for k in ("vllm", "swegemma", "litellm", "adk-submission") if _has(k)},
    }


def _has(pkg):
    try:
        metadata.version(pkg)
        return True
    except metadata.PackageNotFoundError:
        return False


def snapshot_agent(agent_dir, dest):
    """Copy the agent folder (configs, prompts, adapter configs; weight files are hashed in the pin, not copied)."""
    shutil.copytree(agent_dir, dest, ignore=shutil.ignore_patterns(*("*" + s for s in WEIGHTS), "__pycache__"), dirs_exist_ok=True)


if __name__ == "__main__":
    print(json.dumps(pin_settings(sys.argv[1]), indent=1))
