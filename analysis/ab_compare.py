"""Paired A/B of two multi-sample runs (kits/swe-agent/run/hist_runs.sh prefixes) over the same tasks.

usage: envs/harness/bin/python -I analysis/ab_compare.py <prefix_A> <prefix_B> [--tasks usable] [--split train]
Per task: passes out of N in A and in B. Reports total passes per arm, solved-per-try mean, tasks solved at least
once, mixed-band counts, tasks better/worse/same, and a paired bootstrap over tasks (B=10000) of the per-task
pass-rate difference B - A: its mean, 90% interval and P(B > A).
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kits" / "swe-agent" / "run"))  # kit helpers
import manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # the repo root


def load(prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_s(\d+)$")
    rows = {}
    for r in map(json.loads, open(ROOT / "results" / "task_runs.jsonl")):
        m = pat.match(str(r.get("run_id")))
        if m:
            rows[(int(m.group(1)), r["id"])] = r.get("resolved") in (True, "True", 1)
    tries, passes, per_sample = collections.Counter(), collections.Counter(), collections.Counter()
    for (k, t), ok in rows.items():
        tries[t] += 1
        passes[t] += ok
        per_sample[k] += ok
    return tries, passes, per_sample


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tasks", default="usable")
    ap.add_argument("--split", choices=["all", "train", "holdout"], default="all")
    x = ap.parse_args()
    holdout = manifest.holdout()
    tasks = [t for t in manifest.ids(x.tasks)
             if x.split == "all" or (t in holdout) == (x.split == "holdout")]
    (ta, pa, sa), (tb, pb, sb) = load(x.a), load(x.b)
    both = [t for t in tasks if ta[t] and tb[t]]
    print(f"{x.split}: {len(tasks)} tasks, {len(both)} tried in both arms")
    for name, tr, ps, smp in ((x.a, ta, pa, sa), (x.b, tb, pb, sb)):
        band = collections.Counter("never" if ps[t] == 0 else "always" if ps[t] == tr[t] else "mixed" for t in both)
        print(f"  {name}: passes {sum(ps[t] for t in both)} / {sum(tr[t] for t in both)} runs | per sample "
              f"{[smp[k] for k in sorted(smp)]} | ever {sum(ps[t] > 0 for t in both)} | never {band['never']} "
              f"mixed {band['mixed']} always {band['always']}")
    d = {t: pb[t] / tb[t] - pa[t] / ta[t] for t in both}
    print(f"  per task: B better {sum(v > 0 for v in d.values())}, worse {sum(v < 0 for v in d.values())}, "
          f"same {sum(v == 0 for v in d.values())}")
    moved = sorted((t for t in both if d[t]), key=lambda t: d[t])
    print("  changed tasks (A -> B passes): " + ", ".join(f"{t} {pa[t]}->{pb[t]}" for t in moved))
    rng = random.Random(0)
    vals = list(d.values())
    boots = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(10000))
    mean = sum(vals) / len(vals)
    print(f"  mean pass-rate change B - A per task: {mean:+.4f} (90% interval {boots[500]:+.4f} to {boots[9500]:+.4f}), "
          f"P(B > A) = {sum(b > 0 for b in boots) / len(boots):.3f}")


if __name__ == "__main__":
    main()
