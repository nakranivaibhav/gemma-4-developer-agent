"""Rollout phase: N tries of every task with one agent folder, through hist.sh (the same server/runner/clock setup
as every other measurement), with the TITO proxy and rollout tags on, into the RL run's own ledger.

The sandboxes run wherever DOCKER_HOST points (rollout.docker_host; e.g. an SSH-tunnelled socket to a sandbox VM),
the model server and the trainer on this machine's GPU.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import ROOT

sys.path.insert(0, str(ROOT / "kits/swe-agent/run"))


@dataclass
class RolloutRun:
    prefix: str
    run_ids: list[str]
    capture: Path
    ledger: Path
    expected: int


def task_ids(selector: str) -> list[str]:
    import manifest
    return manifest.ids(selector)


def check_train_only(ids: list[str]) -> None:
    import manifest
    split = {r["id"]: r["split"] for r in manifest.rows()}
    bad = [t for t in ids if split.get(t) != "tune"]
    if bad:
        raise ValueError(f"RL trains on tune tasks only; not tune: {bad[:5]}")


def ledger_count(ledger: Path, run_ids: list[str]) -> int:
    if not ledger.exists():
        return 0
    want, n = set(run_ids), 0
    for line in open(ledger):
        if json.loads(line).get("run_id") in want:
            n += 1
    return n


def hist_env(cfg, agent_dir: Path, ledger: Path, label: str) -> dict:
    env = dict(os.environ)
    env.update(MODEL_DIR=cfg.model.model_dir, LEDGER=str(ledger), LOG_PROXY_TITO="1", ROLLOUT_TAG="1",
               SAMPLE_TAG="p", SERVE_AGENT=str(agent_dir), RUN_LABEL=label)
    if cfg.rollout.spec == "ngram":
        env.pop("SPEC", None)                  # serve_fast.sh default: ngram speculative decoding
    else:
        env["SPEC"] = cfg.rollout.spec         # "" = plain decoding
    if cfg.rollout.docker_host:
        env["DOCKER_HOST"] = cfg.rollout.docker_host
    return env


def remove_leftover_sandboxes(run_ids: list[str], env: dict, runner=subprocess.run) -> int:
    """Containers of these run ids still around (a killed attempt): remove them before a relaunch."""
    ids = []
    for rid in run_ids:
        out = runner(["docker", "ps", "-aq", "--filter", f"label=swegemma_run={rid}"], env=env, capture_output=True,
                     text=True, check=False).stdout or ""
        ids += out.split()
    if ids:
        runner(["docker", "rm", "-f", *ids], env=env, capture_output=True, check=False)
    return len(ids)


def run(cfg, agent_dir: Path, prefix: str, ids: list[str], tries: int, ledger: Path, task_file: Path,
        label: str, attempts: int = 2, runner=subprocess.run) -> RolloutRun:
    task_file.write_text("\n".join(ids) + "\n")
    run_ids = [f"{prefix}_p{k}" for k in range(1, tries + 1)]
    expected = len(ids) * tries
    total = min(cfg.rollout.concurrency, expected)
    total -= total % tries
    total = max(total, tries)
    for attempt in range(attempts):            # a relaunch with the same run ids resumes (finished tasks are skipped)
        if ledger_count(ledger, run_ids) >= expected:
            break
        env = hist_env(cfg, agent_dir, ledger, label)
        remove_leftover_sandboxes(run_ids, env, runner)
        runner(["bash", "kits/swe-agent/experiments/hist.sh", str(agent_dir), prefix, str(task_file), str(tries),
                str(total)], cwd=ROOT, env=env, check=False)
    return RolloutRun(prefix, run_ids, ROOT / "runs" / prefix / "requests.jsonl.gz", ledger, expected)
