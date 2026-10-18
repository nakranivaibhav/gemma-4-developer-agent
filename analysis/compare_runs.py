"""Compare two agents task by task, each given as one or more passes over the same tasks.

usage: envs/harness/bin/python -I analysis/compare_runs.py <base_run>[,<base_run>...] <cand_run>[,<cand_run>...]
  e.g. analysis/compare_runs.py tune_out8k_p1,tune_out8k_p2 r01_habits_p1,r01_habits_p2

A task's score for one agent is the share of that agent's passes that resolved it (0, 0.5 or 1 with two passes).
The verdict number is a paired bootstrap over tasks (B=5000, fixed seed): P(candidate mean > base mean).
Promotion screen agreed for this comp: P >= 0.90. Only tasks present in every listed run are compared.
Also prints the split by tag (the tags column of tasks/manifest.csv), by repo, and how the runs behaved (calls, rejected calls, budget).
"""
import json
import random
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kits" / "swe-agent" / "run"))  # kit helpers
import manifest  # noqa: E402
from tag_tasks import load_tags  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # the repo root
B = 5000


def load(run_ids):
    rows = {r: {} for r in run_ids}
    for line in open(ROOT / "results" / "task_runs.jsonl"):
        r = json.loads(line)
        if r.get("run_id") in rows and r.get("id"):
            rows[r["run_id"]][r["id"]] = r  # the last line per run + task wins
    return rows


def score(rows, ids):
    return {i: st.mean(bool(rows[r][i].get("resolved")) for r in rows) for i in ids}


def behaviour(rows, ids):
    out = {}
    for name, d in rows.items():
        s = [d[i].get("trace_stats") or {} for i in ids]
        cap = [(d[i].get("settings") or {}).get("budget_max_tool_calls") or 60 for i in ids]
        calls = [x.get("budget_counted_calls") or 0 for x in s]
        out[name] = {
            "solved": sum(bool(d[i].get("resolved")) for i in ids),
            "calls_mean": round(st.mean(calls), 1),
            "at_call_limit": sum(c >= k for c, k in zip(calls, cap)),
            "rejected_calls": sum(x.get("rejected_calls") or 0 for x in s),
            "tasks_with_rejected": sum(1 for x in s if x.get("rejected_calls")),
            "empty_patch": sum(1 for i in ids if not d[i].get("patch_chars")),
            "harness_error": sum(1 for i in ids if d[i].get("harness_error") or d[i].get("error")),
            "agent_hours": round(sum(d[i].get("agent_seconds") or 0 for i in ids) / 3600, 1),
            "max_prompt": max((x.get("max_prompt_tokens") or 0) for x in s),
        }
    return out


def main(base_ids, cand_ids):
    base, cand = load(base_ids), load(cand_ids)
    ids = sorted(set.intersection(*(set(d) for d in list(base.values()) + list(cand.values()))))
    for name, d in list(base.items()) + list(cand.items()):
        print(f"{name}: {len(d)} tasks in ledger")
    if not ids:
        print("no task is present in every run")
        return
    sb, sc = score(base, ids), score(cand, ids)
    n = len(ids)
    mb, mc = sum(sb.values()), sum(sc.values())
    print(f"\ncompared on {n} tasks")
    print(f"base      {mb:.1f}/{n} = {100 * mb / n:.1f}%   ({', '.join(base_ids)})")
    print(f"candidate {mc:.1f}/{n} = {100 * mc / n:.1f}%   ({', '.join(cand_ids)})")
    diff = [sc[i] - sb[i] for i in ids]
    rng = random.Random(0)
    boots = sorted(sum(rng.choices(diff, k=n)) for _ in range(B))
    p = sum(b > 0 for b in boots) / B
    print(f"difference {mc - mb:+.1f} tasks ({100 * (mc - mb) / n:+.1f} points); 90% interval {boots[int(.05 * B)]:+.1f} to {boots[int(.95 * B)]:+.1f} tasks")
    print(f"P(candidate > base) = {p:.3f}   (screen: >= 0.90)")
    up = sorted(i for i in ids if sc[i] > sb[i])
    down = sorted(i for i in ids if sc[i] < sb[i])
    print(f"\nbetter on {len(up)} tasks, worse on {len(down)}, same on {n - len(up) - len(down)}")
    print("  better:", ", ".join(f"{i}({sb[i]:g}->{sc[i]:g})" for i in up))
    print("  worse: ", ", ".join(f"{i}({sb[i]:g}->{sc[i]:g})" for i in down))
    tags = load_tags()
    groups = {"untagged": [i for i in ids if i not in tags], "tagged": [i for i in ids if i in tags]}
    for i in ids:
        groups.setdefault("repo " + re.sub(r"_\d+$", "", i), []).append(i)
    print()
    for g, sel in groups.items():
        if sel:
            print(f"  {g:14s} base {sum(sb[i] for i in sel):5.1f}/{len(sel):<3d} candidate {sum(sc[i] for i in sel):5.1f}/{len(sel)}")
    print("\nhow each pass behaved:")
    for name, v in {**behaviour(base, ids), **behaviour(cand, ids)}.items():
        print(f"  {name}: " + "  ".join(f"{k}={x}" for k, x in v.items()))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1].split(","), sys.argv[2].split(","))
