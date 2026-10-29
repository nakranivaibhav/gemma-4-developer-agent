"""Paired A/B of two multi-sample runs (kits/swe-agent/experiments/hist.sh prefixes) over the same tasks.

usage: envs/harness/bin/python -I analysis/ab_compare.py <prefix_A> <prefix_B> [--tasks usable] [--split train]
Per task: passes out of N in A and in B. Reports total passes per arm, solved-per-try mean, tasks solved at least
once, mixed-band counts, tasks better/worse/same, and a paired bootstrap over tasks (B=10000) of the per-task
pass-rate difference B - A: its mean, 90% interval and P(B > A).

--metric kaggle (default, the standard since 2026-10-23): a try passes only if it resolved AND stayed within the
Kaggle clock's cap (330 Kaggle-seconds). Runs made with the clock on carry resolved_kaggle; older runs are replayed
from their trace with the same profile the clock would pick for that agent (kaggle_clock.auto_profile). A try whose
trace is missing counts by its plain result and is reported. --metric resolved = the old plain pass count.
"""
import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kits" / "swe-agent" / "run"))  # kit helpers
import kaggle_clock  # noqa: E402
import kaggle_time  # noqa: E402
import manifest  # noqa: E402

PROFILES = kaggle_time.load_profiles()

ROOT = Path(__file__).resolve().parents[1]  # the repo root


STATS = collections.Counter()  # how each try was scored, printed at the end


def passed(r, metric, cap_s=330.0):
    ok = r.get("resolved") in (True, "True", 1)
    if metric == "resolved" or not ok:
        return ok
    if "resolved_kaggle" in r:  # run made with the clock on
        STATS["clock"] += 1
        return bool(r["resolved_kaggle"])
    s = r.get("settings") or {}
    auto = kaggle_clock.auto_profile(s.get("model"), bool(s.get("adapter")))
    trace = ROOT / r["trace"] if r.get("trace") else None
    if auto is None or trace is None or not trace.exists():
        STATS["no_replay"] += 1
        return ok
    name, speed = auto
    calls, tools = kaggle_time.trace_calls(trace)
    secs = kaggle_time.task_seconds(PROFILES[name], calls, tools, speed)
    STATS["replayed"] += 1
    STATS["replayed_cut"] += secs > cap_s
    return secs <= cap_s


def load(prefix, metric):
    pat = re.compile(rf"^{re.escape(prefix)}_[sp](\d+)$")  # hist.sh samples: _s1.. or _p1.. (SAMPLE_TAG)
    rows = {}
    for r in map(json.loads, open(ROOT / "results" / "task_runs.jsonl")):
        m = pat.match(str(r.get("run_id")))
        if m:
            rows[(int(m.group(1)), r["id"])] = passed(r, metric)
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
    ap.add_argument("--metric", choices=["kaggle", "resolved"], default="kaggle")
    x = ap.parse_args()
    holdout = manifest.holdout()
    tasks = [t for t in manifest.ids(x.tasks)
             if x.split == "all" or (t in holdout) == (x.split == "holdout")]
    (ta, pa, sa), (tb, pb, sb) = load(x.a, x.metric), load(x.b, x.metric)
    both = [t for t in tasks if ta[t] and tb[t]]
    print(f"{x.split}: {len(tasks)} tasks, {len(both)} tried in both arms | metric {x.metric}"
          + (f" (resolved tries scored: {dict(STATS)})" if x.metric == "kaggle" else ""))
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
