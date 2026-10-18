"""The screen queue: the tune tasks (split=tune in tasks/manifest.csv: every usable public task except the rich repo,
which is the holdout) split into strata by repo x number of files the real fix changed (1 / 2 / 3+), each stratum
shuffled once with a fixed seed. Files-changed counts come from runs/eda_dashboard/tasks.csv (analysis/eda_dashboard_data.py).

usage: python -I kits/swe-agent/run/queue.py build                 -> tasks/screen_queue.json (strata + a cursor per stratum)
       python -I kits/swe-agent/run/queue.py draw <n> <run_id>     -> prints n task ids, advances the cursors, logs results/draws.jsonl

A draw takes n tasks in proportion to the strata sizes (largest remainder), so a screen looks like the tune set in
miniature. Each stratum is walked in order from its cursor, so successive draws take fresh tasks until the stratum
is used up, then it wraps around (reshuffled). A task can be looked up in any draw by run_id.
"""
import csv
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest  # noqa: E402

HOLDOUT_REPO = "rich"
SEED = 20261010
QUEUE = manifest.ROOT / "tasks" / "screen_queue.json"
DRAWS = manifest.ROOT / "results" / "draws.jsonl"
TASKS = manifest.ROOT / "runs" / "eda_dashboard" / "tasks.csv"


def bucket(n):
    return "1" if n <= 1 else "2" if n == 2 else "3+"


def build():
    strata = {}
    tune, holdout, excluded = set(manifest.ids("tune")), manifest.holdout(), set(manifest.ids("excluded"))
    for r in csv.DictReader(open(TASKS)):
        if r["id"] not in tune:
            continue
        repo = r["id"].rsplit("_", 1)[0]
        strata.setdefault(f"{repo}/{bucket(int(r['files_changed']))}", []).append(r["id"])
    rng = random.Random(SEED)
    out = {}
    for k in sorted(strata):
        ids = sorted(strata[k])
        rng.shuffle(ids)
        out[k] = {"tasks": ids, "cursor": 0, "wraps": 0}
    QUEUE.parent.mkdir(exist_ok=True)
    QUEUE.write_text(json.dumps({"seed": SEED, "holdout_repo": HOLDOUT_REPO, "excluded": sorted(excluded), "strata": out}, indent=1))
    for k, v in out.items():
        print(f"{k:14s} {len(v['tasks'])}")
    print("total", sum(len(v["tasks"]) for v in out.values()), "| holdout", len(holdout), "| excluded", len(excluded))


def allocate(sizes, n):
    total = sum(sizes.values())
    exact = {k: n * s / total for k, s in sizes.items()}
    alloc = {k: int(e) for k, e in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - alloc[k], reverse=True)[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def draw(n, run_id):
    q = json.loads(QUEUE.read_text())
    alloc = allocate({k: len(v["tasks"]) for k, v in q["strata"].items()}, n)
    picked = {}
    for k, want in alloc.items():
        s = q["strata"][k]
        got = []
        while len(got) < want:
            if s["cursor"] >= len(s["tasks"]):
                rng = random.Random(q["seed"] + s["wraps"] + 1)
                rng.shuffle(s["tasks"])
                s["cursor"], s["wraps"] = 0, s["wraps"] + 1
            got.append(s["tasks"][s["cursor"]])
            s["cursor"] += 1
        picked[k] = got
    QUEUE.write_text(json.dumps(q, indent=1))
    rec = {"run_id": run_id, "utc": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), "n": n, "by_stratum": picked,
           "cursors": {k: v["cursor"] for k, v in q["strata"].items()}}
    with open(DRAWS, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(" ".join(t for k in picked for t in picked[k]))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build()
    else:
        draw(int(sys.argv[2]), sys.argv[3])
