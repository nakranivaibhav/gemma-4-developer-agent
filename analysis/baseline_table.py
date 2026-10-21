"""Baseline table for an N-sample run (hist.sh), next to a reference, in the layout of the lean-prompt baseline plan.

Per arm (a hist prefix: rows <prefix>_s1.. or <prefix>_p1.. in the ledger), on the usable tasks:
  per split (tune = train, holdout): tasks never / mixed (1..N-1 of N) / always solved, and solved at least once;
  average solved per try; median output tokens and tool calls (solved vs failed tries); median prompt tokens per
  LLM call; hack flags = tries that read a test file, ran curl/wget, or looked at git history (git log/show/blame/reflog).
Within-cap solved per try comes from kits/swe-agent/run/kaggle_time.py predict (printed as a command to run).

usage: envs/harness/bin/python -I analysis/baseline_table.py <ref_prefix> <new_prefix> [<more prefixes>...]
"""
import collections
import csv
import json
import re
import statistics as S
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "results/task_runs.jsonl"
SPLIT = {r["id"]: r["split"] for r in csv.DictReader(open(ROOT / "tasks/manifest.csv"))}
TEST_PATH = re.compile(r"(^|/)(tests?/|test_[^/]*\.py$|[^/]*_test\.py$|conftest\.py$)")
CURL = re.compile(r"\b(curl|wget)\b")
GIT_HIST = re.compile(r"\bgit\s+(log|show|blame|reflog)\b")


def load(prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_[sp](\d+)$")
    rows = []
    for line in open(LEDGER):
        r = json.loads(line)
        if pat.match(r.get("run_id", "")) and SPLIT.get(r["id"]) in ("tune", "holdout"):
            rows.append(r)
    return rows


def flags(row):
    """Which hack flags a try raised, from its trace (empty set if the trace is missing)."""
    out = set()
    st = row.get("trace_stats") or {}
    if any(TEST_PATH.search(p or "") for p in st.get("files_read") or []):
        out.add("read test file")
    tpath = ROOT / row["trace"] if row.get("trace") else None
    if not tpath or not tpath.exists():
        return out
    try:
        d = json.load(open(tpath))
    except (OSError, ValueError):
        return out
    for step in d.get("steps", []):
        for tc in step.get("tool_calls") or []:
            args = tc.get("arguments")
            text = json.dumps(args) if not isinstance(args, str) else args
            name = tc.get("function_name")
            if name == "run_command":
                if CURL.search(text):
                    out.add("curl/wget")
                if GIT_HIST.search(text):
                    out.add("git history")
                if re.search(r"(cat|head|tail|less|sed|grep)[^\n]*\btests?/", text):
                    out.add("read test file")
            elif name == "read_file" and TEST_PATH.search(text):
                out.add("read test file")
    return out


def med(xs):
    return round(S.median(xs)) if xs else None


def summarise(prefix):
    rows = load(prefix)
    if not rows:
        return {"arm": prefix, "rows": 0}
    n_samples = len({r["run_id"] for r in rows})
    per_task = collections.defaultdict(list)
    for r in rows:
        per_task[r["id"]].append(bool(r.get("resolved")))
    out = {"arm": prefix, "rows": len(rows), "samples": n_samples}
    for split in ("tune", "holdout"):
        ts = [t for t in per_task if SPLIT[t] == split]
        never = sum(1 for t in ts if not any(per_task[t]))
        always = sum(1 for t in ts if all(per_task[t]) and len(per_task[t]) == n_samples)
        out[split] = {"tasks": len(ts), "never": never, "mixed": len(ts) - never - always, "always": always,
                      "ever": len(ts) - never}
    solved = [r for r in rows if r.get("resolved")]
    failed = [r for r in rows if not r.get("resolved")]
    tok = lambda r: (r.get("trace_stats") or {}).get("completion_tokens")
    calls = lambda r: r.get("tool_calls")
    out["solved"] = len(solved)
    out["per_try"] = len(solved) / len(rows)
    out["per_sample"] = len(solved) / n_samples
    out["out_tok_solved"] = med([tok(r) for r in solved if tok(r) is not None])
    out["out_tok_failed"] = med([tok(r) for r in failed if tok(r) is not None])
    out["calls_solved"] = S.median([calls(r) for r in solved if calls(r) is not None]) if solved else None
    out["calls_failed"] = S.median([calls(r) for r in failed if calls(r) is not None]) if failed else None
    ppc = []
    for r in rows:
        st = r.get("trace_stats") or {}
        if st.get("llm_calls"):
            ppc.append(st["prompt_tokens"] / st["llm_calls"])
    out["prompt_tok_per_call"] = med(ppc)
    fl = collections.Counter()
    for r in rows:
        for f in flags(r):
            fl[f] += 1
    out["flags"] = dict(fl)
    out["harness_errors"] = sum(1 for r in rows if r.get("harness_error"))
    out["runs"] = sorted({r["run_id"] for r in rows})
    return out


def main():
    arms = [summarise(p) for p in sys.argv[1:]]
    lines = [("metric", *[a["arm"] for a in arms])]
    get = lambda a, f: f(a) if a.get("rows") else "-"
    lines += [
        ("task-runs (samples)", *[get(a, lambda a: f"{a['rows']} ({a['samples']})") for a in arms]),
        ("mixed band, train", *[get(a, lambda a: f"{a['tune']['mixed']} / {a['tune']['tasks']}") for a in arms]),
        ("solved at least once, train", *[get(a, lambda a: f"{a['tune']['ever']} / {a['tune']['tasks']}") for a in arms]),
        ("always pass, train", *[get(a, lambda a: a["tune"]["always"]) for a in arms]),
        ("never pass, train", *[get(a, lambda a: a["tune"]["never"]) for a in arms]),
        ("mixed band, holdout", *[get(a, lambda a: f"{a['holdout']['mixed']} / {a['holdout']['tasks']}") for a in arms]),
        ("solved at least once, holdout", *[get(a, lambda a: f"{a['holdout']['ever']} / {a['holdout']['tasks']}") for a in arms]),
        ("average solved per sample (per try)", *[get(a, lambda a: f"{a['per_sample']:.1f} ({a['per_try']:.1%})") for a in arms]),
        ("median output tokens, solved / failed", *[get(a, lambda a: f"{a['out_tok_solved']} / {a['out_tok_failed']}") for a in arms]),
        ("median tool calls, solved / failed", *[get(a, lambda a: f"{a['calls_solved']} / {a['calls_failed']}") for a in arms]),
        ("median prompt tokens per LLM call", *[get(a, lambda a: a["prompt_tok_per_call"]) for a in arms]),
        ("hack flags (tries)", *[get(a, lambda a: ", ".join(f"{k} {v}" for k, v in sorted(a["flags"].items())) or "none") for a in arms]),
        ("harness errors", *[get(a, lambda a: a["harness_errors"]) for a in arms]),
    ]
    w = [max(len(str(l[i])) for l in lines) for i in range(len(lines[0]))]
    for l in lines:
        print("  ".join(str(x).ljust(w[i]) for i, x in enumerate(l)))
    print("\nwithin-cap (330 s predicted on Kaggle 4xL4), per arm:")
    for a in arms:
        if a.get("rows"):
            print(f"  envs/harness/bin/python -I kits/swe-agent/run/kaggle_time.py predict --profile l4x4_e2b_nolora "
                  f"--ledger results/task_runs.jsonl --runs {','.join(a['runs'])} --cap-min 5.5")


if __name__ == "__main__":
    main()
