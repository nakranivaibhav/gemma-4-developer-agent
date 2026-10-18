"""How the agent behaved in each run, read from the saved traces: one line of process numbers per run.

usage: envs/harness/bin/python -I analysis/round_stats.py <run_id> [<run_id> ...]
Numbers are per task unless marked total. "source file" = not a test, not a scratch/repro script at the repo root.
"""
import collections
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repo root
SCRATCH = re.compile(r"(^|/)(tests?/|test_[^/]*$|[^/]*_test\.py$|conftest\.py$)|^(repro|reproduce|debug|verify|check|scratch|tmp|demo|example)[^/]*$|^[^/]+\.(py|txt|html|md|json|sh)$")


def is_source(path):
    return bool(path) and not SCRATCH.search(path.replace("/workspace/", "").lstrip("./"))


def task_stats(run, tid):
    steps = json.load(open(ROOT / "runs" / run / "traces" / f"trace_{tid}.json")).get("steps", [])
    n = 0
    s = collections.Counter()
    first_edit = None
    streak = best = 0
    missing = collections.Counter()
    for step in steps:
        for tc in step.get("tool_calls") or []:
            name, a = tc.get("function_name"), tc.get("arguments") or {}
            obs = step.get("observation")
            raw = obs.get("content") if isinstance(obs, dict) else obs
            try:
                o = json.loads(raw) if isinstance(raw, str) else (raw or {})
            except Exception:
                o = {}
            o = o if isinstance(o, dict) else {}
            msg = str(o.get("error_message") or o.get("error") or "")
            rejected = "mandatory input" in msg
            if rejected:
                s["rejected"] += 1
                streak += 1
                best = max(best, streak)
                missing[(name, msg.split("not present:")[-1].split("You could")[0].strip().replace("\n", "+"))] += 1
                continue
            streak = 0
            if name in ("get_status", "submit_patch"):
                s[name] += 1
                continue
            n += 1
            ok = not (o.get("status") == "error" or msg)
            if name in ("edit_file", "write_file"):
                fp = str(a.get("filepath", ""))
                if ok and is_source(fp):
                    s["src_edits"] += 1
                    first_edit = first_edit or n
                    if a.get("allow_multiple"):
                        s["allow_multiple"] += 1
                elif ok:
                    s["scratch_writes"] += 1
                else:
                    s["edit_failed"] += 1
            elif name == "run_command":
                cmd = str(a.get("command", ""))
                if "pytest" in cmd:
                    s["pytest"] += 1
                elif re.search(r"python3?\s+(-c|\S+\.py)", cmd):
                    s["own_script"] += 1
                if re.search(r"\b(grep|rg)\b", cmd):
                    s["grep_after_edit" if first_edit else "grep_before_edit"] += 1
                if re.search(r"\bsed -i\b", cmd):
                    s["sed_i"] += 1
                if re.search(r"\brm\b", cmd):
                    s["rm"] += 1
            elif name == "read_file":
                s["read"] += 1
            elif name in ("get_code_neighbors", "search_similar_code", "get_code_subgraph"):
                s["graph_tools"] += 1
            elif name == "code_analyzer":
                s["sub_agent"] += 1
    s["calls"] = n
    return s, first_edit, best, missing


def main(runs):
    for run in runs:
        rows = {}
        for line in open(ROOT / "results" / "task_runs.jsonl"):
            r = json.loads(line)
            if r.get("run_id") == run:
                rows[r["id"]] = r
        per = {tid: task_stats(run, tid) for tid in rows}
        miss = collections.Counter()
        for v in per.values():
            miss.update(v[3])
        print(f"\n{run}: solved {sum(bool(r.get('resolved')) for r in rows.values())}/{len(rows)}")
        for label, sel in (("solved", [t for t in rows if rows[t].get("resolved")]), ("failed", [t for t in rows if not rows[t].get("resolved")])):
            if not sel:
                continue
            mean = lambda k: st.mean(per[t][0][k] for t in sel)
            fe = [per[t][1] for t in sel if per[t][1]]
            patches = [(ROOT / rows[t]["patch"]).read_text(errors="replace") if (ROOT / rows[t].get("patch", "x")).exists() else "" for t in sel]
            junk = sum(1 for p in patches if any(not is_source(f) for f in re.findall(r"^\+\+\+ b/(\S+)", p, re.M)))
            dbg = sum(1 for p in patches if re.search(r"^\+.*\b(DEBUG|print\(.*(debug|DEBUG))", p, re.M))
            print(f"  {label:6s} n={len(sel):2d} calls {mean('calls'):.1f} | read {mean('read'):.1f} grep-before-edit {mean('grep_before_edit'):.1f} "
                  f"grep-after-edit {mean('grep_after_edit'):.1f} graph {mean('graph_tools'):.1f} sub-agent {mean('sub_agent'):.2f} | first source edit: median call "
                  f"{st.median(fe) if fe else '-'} (none in {len(sel) - len(fe)}) | source edits {mean('src_edits'):.1f} scratch writes {mean('scratch_writes'):.1f} "
                  f"failed edits {mean('edit_failed'):.2f} | pytest {mean('pytest'):.1f} own scripts {mean('own_script'):.1f} | sed -i {mean('sed_i'):.2f} "
                  f"allow_multiple {mean('allow_multiple'):.2f} rm {mean('rm'):.2f} | patch has non-source files {junk}/{len(sel)} debug lines {dbg}/{len(sel)}")
        streaks = sorted(((per[t][2], t, bool(rows[t].get("resolved"))) for t in rows if per[t][2]), reverse=True)
        print(f"  rejected calls total {sum(per[t][0]['rejected'] for t in rows)} in {len(streaks)} tasks; streaks of 3+: {sum(1 for x in streaks if x[0] >= 3)}; "
              f"longest: {[(x[1], x[0], 'solved' if x[2] else 'failed') for x in streaks[:5]]}")
        print(f"  rejected by tool + missing argument: {dict(miss.most_common(6))}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
