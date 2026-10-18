"""Per-task pass counts over N samples of one agent, and the histogram of those counts.

usage: envs/harness/bin/python -I analysis/pass_hist.py <run_prefix> [--tasks usable]
Reads ledger rows with run_id <run_prefix>_s1 .. _sN (kits/swe-agent/experiments/hist.sh). A sample that has no row for a task
(run cut short) leaves that task with fewer tries; the table shows tries so this is visible.
Bands: never (0 passes), mixed (some but not all), always (every try). Mixed tasks are the ones RL can learn from:
a group of rollouts on them has both rewards.
Writes runs/sft/pass_hist_<run_prefix>.csv: task_id, split, tries, passes, rate, band.
"""
import argparse
import collections
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kits" / "swe-agent" / "run"))  # kit helpers
import manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # the repo root


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prefix")
    ap.add_argument("--tasks", default="usable")
    a = ap.parse_args()
    tasks = manifest.ids(a.tasks)
    holdout = manifest.holdout()
    pat = re.compile(rf"^{re.escape(a.prefix)}_s(\d+)$")
    rows = {}
    for r in map(json.loads, open(ROOT / "results" / "task_runs.jsonl")):
        m = pat.match(str(r.get("run_id")))
        if m:
            rows[(int(m.group(1)), r["id"])] = r  # a rerun in the same sample replaces the earlier row
    samples = sorted({k for k, _ in rows})
    tries, passes, errors = collections.Counter(), collections.Counter(), collections.Counter()
    for (k, tid), r in rows.items():
        tries[tid] += 1
        passes[tid] += r.get("resolved") in (True, "True", 1)
        errors[tid] += bool(r.get("harness_error"))
    out = []
    for t in tasks:
        n, p = tries[t], passes[t]
        band = "untried" if n == 0 else "never" if p == 0 else "always" if p == n else "mixed"
        out.append({"task_id": t, "split": "holdout" if t in holdout else "train", "tries": n, "passes": p,
                    "rate": round(p / n, 3) if n else "", "band": band, "harness_errors": errors[t]})
    dest = ROOT / "runs" / "sft" / f"pass_hist_{a.prefix}.csv"
    with open(dest, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    full = max(tries.values(), default=0)
    print(f"{a.prefix}: samples {samples}, {len(rows)} task runs, {len(tasks)} tasks, most tries per task {full}")
    print(f"tasks with fewer than {full} tries: {sum(1 for o in out if o['tries'] < full)}")
    for split in ("train", "holdout", "all"):
        sel = [o for o in out if split == "all" or o["split"] == split]
        hist = collections.Counter(o["passes"] for o in sel if o["tries"])
        bands = collections.Counter(o["band"] for o in sel)
        print(f"\n{split} ({len(sel)} tasks): never {bands['never']}, mixed {bands['mixed']}, always {bands['always']}"
              + (f", untried {bands['untried']}" if bands["untried"] else ""))
        print("  passes: " + "  ".join(f"{k}:{hist[k]}" for k in range(full + 1)))
    mixed = sorted((o for o in out if o["band"] == "mixed"), key=lambda o: (-o["rate"], o["task_id"]))
    print("\nmixed tasks: " + ", ".join(f"{o['task_id']} {o['passes']}/{o['tries']} ({o['split']})" for o in mixed))
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
