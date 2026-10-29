"""Join the TITO capture (one row per model call, tagged with its try) with the run ledger (one row per try).

A try is identified by rollout_id = "<run_id>/<task_id>": run_tasks.py sends it as the X-Rollout-Id header on
every model call when ROLLOUT_TAG=1, and the capture proxy records it. Calls of one try are sequential, so their
capture time orders them.

Strict on purpose: an untagged call, two ledger rows for one try, or a try served by another adapter than the one
the batch claims (the on-policy guard) raise instead of being skipped.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from . import reward

ADAPTER_KEY = "adapters/main_lora/adapter_model.safetensors"   # run_tasks.py pins sha256[:16] of every agent file


@dataclass
class Rollout:
    rollout_id: str
    task: str
    run_id: str
    outcome: reward.Outcome
    calls: list[dict] = field(default_factory=list)   # tito records in call order
    exclude: str | None = None                          # why the try is not used; None = used

    @property
    def reward(self) -> float | None:
        return None if self.exclude else self.outcome.reward


def read_capture(path: str | Path) -> tuple[dict[str, list[dict]], dict]:
    """rollout_id -> [{"utc", "status", "prompt_ids", ...} | {"error"}], plus counts."""
    by_id: dict[str, list[dict]] = defaultdict(list)
    stats = {"calls": 0, "untagged": 0}
    with gzip.open(path, "rt") as fh:
        for line in fh:
            row = json.loads(line)
            stats["calls"] += 1
            rid = row.get("rollout_id")
            if not rid:
                stats["untagged"] += 1
                continue
            tito = row.get("tito") or {"error": f"no tito record (status {row.get('status')})"}
            by_id[rid].append({"utc": row["utc"], "status": row.get("status"), **tito})
    for calls in by_id.values():
        calls.sort(key=lambda c: c["utc"])
    return dict(by_id), stats


def read_ledger(path: str | Path, run_ids: set[str]) -> list[dict]:
    rows = []
    with open(path) as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("run_id") in run_ids:
                rows.append(r)
    return rows


def adapter_sha(row: dict) -> str | None:
    return ((row.get("settings") or {}).get("file_sha") or {}).get(ADAPTER_KEY)


def build(ledger_rows: list[dict], capture: dict[str, list[dict]], policy_sha: str,
          allow_untagged: int = 0, capture_stats: dict | None = None) -> list[Rollout]:
    if capture_stats and capture_stats.get("untagged", 0) > allow_untagged:
        raise ValueError(f"{capture_stats['untagged']} captured calls carry no rollout id (ROLLOUT_TAG off?)")
    seen, out = set(), []
    for r in ledger_rows:
        rid = f"{r['run_id']}/{r['id']}"
        if rid in seen:
            raise ValueError(f"two ledger rows for {rid}")
        seen.add(rid)
        sha = adapter_sha(r)
        if sha != policy_sha:
            raise ValueError(f"on-policy guard: {rid} was served adapter {sha}, the batch policy is {policy_sha}")
        oc = reward.classify(r)
        ro = Rollout(rid, r["id"], r["run_id"], oc, capture.get(rid, []))
        if oc.excluded:
            ro.exclude = oc.kind
        elif not ro.calls:
            ro.exclude = "no_calls"
        elif any("error" in c for c in ro.calls):
            ro.exclude = "tito_missing"
        elif any(c["prompt_ids"] == ro.calls[0]["prompt_ids"] for c in ro.calls[1:]):
            ro.exclude = "restarted"          # a resumed rollout re-ran this try: two attempts share the id
        out.append(ro)
    orphans = set(capture) - seen
    if orphans:
        raise ValueError(f"captured calls for {len(orphans)} tries with no ledger row, e.g. {sorted(orphans)[:3]}")
    return out
