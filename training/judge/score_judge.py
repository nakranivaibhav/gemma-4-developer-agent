"""Score a judge run against the answer key (tasks/judge_answer_key.jsonl), and two runs against each other.

usage: python3 -I training/judge/score_judge.py <run> [--vs <run2>] [--misses]
Per rubric item: hits (key yes, judge yes), misses (key yes, judge no), false alarms (key no, judge yes), clean
passes (key no, judge no). Key labels "unsure" or null are left out of that item's count. --misses prints every miss
and false alarm with the judge's evidence, for the prompt-revision step. --vs prints per-item agreement between the
two runs (self-consistency).
"""
import argparse
import glob
import json
import os

ITEMS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_run(run):
    out = {}
    for f in glob.glob(os.path.join(ROOT, "runs/judge", run, "*.json")):
        if f.endswith("run.json"):
            continue
        o = json.load(open(f))
        if o.get("verdict"):
            out[o["id"]] = o
    return out


ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--vs")
ap.add_argument("--misses", action="store_true")
a = ap.parse_args()

key = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(ROOT, "tasks/judge_answer_key.jsonl"))}
run = load_run(a.run)
errs = [f for f in glob.glob(os.path.join(ROOT, "runs/judge", a.run, "*.json"))
        if not f.endswith("run.json") and not json.load(open(f)).get("verdict")]
cost = sum((o.get("cost_usd") or 0) for o in run.values())
print(f"{a.run}: {len(run)} verdicts, {len(errs)} without a verdict, ${cost:.2f} total, "
      f"${cost / max(len(run), 1):.3f}/task")
print(f"{'item':4s} {'hits':>5s} {'miss':>5s} {'false':>6s} {'clean':>6s}   recall  false-alarm rate")
rows = []
for it in ITEMS:
    h = m = fa = c = 0
    for i, o in run.items():
        k = (key.get(i) or {}).get(it)
        if k not in ("yes", "no"):
            continue
        j = o["verdict"][it]["answer"]
        if it == "R7":  # R7 asks "solvable?": the defect is a "no"
            k, j = ("yes" if k == "no" else "no"), ("yes" if j == "no" else "no")
        h += k == "yes" and j == "yes"
        m += k == "yes" and j == "no"
        fa += k == "no" and j == "yes"
        c += k == "no" and j == "no"
        if a.misses and k != j:
            rows.append((it, i, "MISS" if k == "yes" else "FALSE", o["verdict"][it]["evidence"][:300]))
    rec = f"{h / (h + m):.2f}" if h + m else "  - "
    far = f"{fa / (fa + c):.2f}" if fa + c else "  - "
    print(f"{it:4s} {h:5d} {m:5d} {fa:6d} {c:6d}   {rec:>6s}  {far:>6s}" + ("   (R7 counted as 'unsolvable')" if it == "R7" else ""))
for r in rows:
    print(f"  {r[0]} {r[2]:5s} {r[1]}: {r[3]}")
if a.vs:
    other = load_run(a.vs)
    both = sorted(set(run) & set(other))
    print(f"self-agreement {a.run} vs {a.vs} on {len(both)} tasks:")
    for it in ITEMS:
        same = sum(run[i]["verdict"][it]["answer"] == other[i]["verdict"][it]["answer"] for i in both)
        print(f"  {it}: {same}/{len(both)} = {same / max(len(both), 1):.2f}")
