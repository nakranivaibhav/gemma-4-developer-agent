"""Per-task solve rates from every local run: the starting pool for training tasks.

For each public task: repo, split (holdout = never train), and per model family the number of clean attempts,
solves and solve rate. RL learns from tasks the model solves sometimes (0 < rate < 1); tasks never solved
by any model need better data (a stronger model's trajectory or a hint) before they can teach anything.
Attempts with a harness error are left out (they say nothing about the model).

usage: task_pool.py [--out runs/sft/task_pool.csv]
"""
import argparse
import csv
import json
from collections import defaultdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kits" / "swe-agent" / "run"))  # kit helpers
import manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # the repo root
FAMILIES = ["31b", "12b", "e4b", "e2b", "qwen"]


def family(model):
    m = (model or "").lower()
    return next((f for f in FAMILIES if f in m), "other")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/sft/task_pool.csv")
    a = ap.parse_args()
    tasks = {json.loads(line)["instance_id"]: json.loads(line)["repo"] for line in open(ROOT / "data" / "tasks.jsonl")}
    holdout = manifest.holdout()
    latest = {}
    for line in open(ROOT / "results" / "task_runs.jsonl"):
        r = json.loads(line)
        latest[(r.get("run_id"), r["id"])] = r
    att, sol = defaultdict(int), defaultdict(int)
    for (rid, tid), r in latest.items():
        if r.get("harness_error") not in (None, "None", ""):
            continue
        s = r.get("settings") if isinstance(r.get("settings"), dict) else {}
        f = family(s.get("model"))
        att[(tid, f)] += 1
        sol[(tid, f)] += str(r.get("resolved")).lower() == "true"
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["task_id", "repo", "split"] + [f"{f}_{k}" for f in FAMILIES for k in ("tries", "solved", "rate")] + ["any_solved"]
    counts = defaultdict(int)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for tid in sorted(tasks):
            split = "holdout" if tid in holdout else "train"
            row = [tid, tasks[tid], split]
            for f in FAMILIES:
                n, k = att[(tid, f)], sol[(tid, f)]
                row += [n, k, f"{k / n:.2f}" if n else ""]
            anyk = sum(sol[(tid, f)] for f in FAMILIES) > 0
            row.append(int(anyk))
            w.writerow(row)
            if split == "train":
                n, k = att[(tid, "31b")], sol[(tid, "31b")]
                band = "untried" if not n else "never" if k == 0 else "always" if k == n else "sometimes"
                counts[band] += 1
                counts["train_any_model_solved"] += anyk
    print(f"wrote {a.out}: {len(tasks)} tasks, {len(holdout)} holdout")
    print("train tasks by 31B outcome:", dict(counts))


if __name__ == "__main__":
    main()
