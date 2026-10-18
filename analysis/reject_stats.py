"""Rejected tool calls (a mandatory argument "not present"): what the broken call looked like, and what it cost.

usage: envs/harness/bin/python -I analysis/reject_stats.py <run_id> [<run_id> ...]
Reads saved traces + the ledger. A "streak" is a run of rejected calls with no accepted call in between.
"""
import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repo root
LONG = ("new_string", "old_string", "content", "command")


def calls_of(run, tid):
    out = []
    for s in json.load(open(ROOT / "runs" / run / "traces" / f"trace_{tid}.json")).get("steps", []):
        for tc in s.get("tool_calls") or []:
            obs = s.get("observation")
            raw = obs.get("content") if isinstance(obs, dict) else obs
            try:
                o = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                o = {}
            o = o if isinstance(o, dict) else {}
            msg = str(o.get("error_message") or o.get("error") or "")
            out.append({"name": tc.get("function_name"), "args": tc.get("arguments") or {}, "rejected": "mandatory input" in msg,
                        "error": bool(msg) or o.get("status") == "error"})
    return out


def shape(args):
    """How the first long string of a rejected call ended: the character just before `,<next key>:` swallowed into it."""
    for k, v in args.items():
        m = re.search(r"(.{0,25})(.),(old_string|new_string|filepath|content|allow_multiple):$", str(v), re.S)
        if m:
            return k, m.group(2), m.group(1)[-25:]
    return None, None, None


def main(runs):
    rows = collections.defaultdict(dict)
    for line in open(ROOT / "results" / "task_runs.jsonl"):
        r = json.loads(line)
        if r.get("run_id") in runs and r.get("id"):
            rows[r["run_id"]][r["id"]] = r
    enders, firsts, after, okay_len, bad_len = collections.Counter(), [], collections.Counter(), [], []
    per_task = collections.defaultdict(list)  # task -> [(had rejection, solved)]
    for run in runs:
        for tid, row in rows[run].items():
            cs = calls_of(run, tid)
            per_task[tid].append((any(c["rejected"] for c in cs), bool(row.get("resolved"))))
            i = 0
            while i < len(cs):
                c = cs[i]
                if c["name"] in ("edit_file", "write_file") and not c["rejected"] and not c["error"]:
                    okay_len.append(max(len(str(c["args"].get(k, ""))) for k in ("new_string", "content")))
                if not c["rejected"]:
                    i += 1
                    continue
                j = i
                while j < len(cs) and cs[j]["rejected"]:
                    j += 1
                key, ch, tail = shape(c["args"])
                enders[(c["name"], key, ch)] += 1
                same = sum(1 for k in range(i + 1, j) if cs[k]["args"] == c["args"])
                nxt = cs[j] if j < len(cs) else None
                kind = "nothing (trace ends)" if nxt is None else nxt["name"]
                if nxt and nxt["name"] in ("edit_file", "write_file"):
                    size = max(len(str(nxt["args"].get(k, ""))) for k in ("new_string", "content"))
                    kind += " short(<200)" if size < 200 else " long"
                after[kind] += 1
                if key:
                    bad_len.append(len(str(c["args"][key])))
                firsts.append((run, tid, j - i, same, c["name"], key, ch, tail))
                i = j
    print(f"runs: {', '.join(runs)}")
    print(f"streaks: {len(firsts)}; rejected calls in them: {sum(f[2] for f in firsts)}; streak length median {st.median(f[2] for f in firsts)}, "
          f"max {max(f[2] for f in firsts)}; streaks of 1: {sum(f[2] == 1 for f in firsts)}")
    print("first call of each streak, by (tool, argument that swallowed the next key, character it ended with):")
    for k, v in enders.most_common(8):
        print(f"   {v:3d}  {k}")
    print(f"length of that swallowed string: median {st.median(bad_len):.0f} chars (accepted edit/write calls: median {st.median(okay_len):.0f}, "
          f"p75 {sorted(okay_len)[int(.75 * len(okay_len))]}, p90 {sorted(okay_len)[int(.9 * len(okay_len))]})")
    for lo, hi in ((0, 200), (200, 500), (500, 1500), (1500, 10 ** 9)):
        b, g = sum(lo <= x < hi for x in bad_len), sum(lo <= x < hi for x in okay_len)
        print(f"   string of {lo}-{hi if hi < 10 ** 9 else 'up'} chars: {b} broken first calls vs {g} accepted calls  ({100 * b / max(1, b + g):.1f}% broken)")
    print("first accepted call after a streak:", dict(after.most_common()))
    print("text just before the wrong ending (10 samples):")
    for f in firsts[:10]:
        print(f"   {f[1]:15s} streak {f[2]:3d} (exact repeats {f[3]:3d}) {f[4]}.{f[5]} ends {f[6]!r} after {f[7]!r}")
    mixed = {t: v for t, v in per_task.items() if any(a for a, _ in v) and not all(a for a, _ in v)}
    with_r = [s for v in mixed.values() for a, s in v if a]
    without = [s for v in mixed.values() for a, s in v if not a]
    print(f"\ncost check, same tasks only ({len(mixed)} tasks that had a rejection in some passes but not others):")
    if mixed:  # needs 2+ runs over the same tasks; with one run every task is all-or-nothing
        print(f"   passes WITH a rejection:    solved {sum(with_r)}/{len(with_r)} = {100 * st.mean(with_r):.0f}%")
        print(f"   passes WITHOUT a rejection: solved {sum(without)}/{len(without)} = {100 * st.mean(without):.0f}%")
    always = sum(1 for v in per_task.values() if all(a for a, _ in v))
    print(f"   tasks with a rejection in every pass: {always}; never: {sum(1 for v in per_task.values() if not any(a for a, _ in v))}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
