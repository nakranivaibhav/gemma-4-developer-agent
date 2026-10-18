"""usage: python -I analysis/fail_tools.py <run_id> — tool-error kinds, where calls go, and whether names the hidden tests need are in the task text."""
import json, re, sys, collections, statistics
from pathlib import Path
C = Path(__file__).resolve().parents[1]  # the repo root
RUN = sys.argv[1]
tasks = {}
for l in open(C / "data/tasks.jsonl"):
    t = json.loads(l); tasks[t["instance_id"]] = t
led = {}
for l in open(C / "results/task_runs.jsonl"):
    r = json.loads(l)
    if r.get("run_id") == RUN: led[r["id"]] = r
def cmd_kind(c):
    c = c.strip()
    if re.search(r"\bpytest\b", c): return "pytest"
    if re.match(r"(cd \S+ && )?(PYTHONPATH=\S+ )?python3? ", c): return "python script/-c"
    w = re.split(r"[\s|;&]+", c)[0]
    return w if w in ("grep", "sed", "cat", "ls", "find", "git", "head", "tail", "rm", "pip", "echo", "wc", "awk") else "other"
per = {}
errs = {True: collections.Counter(), False: collections.Counter()}
ex = collections.defaultdict(list)
for i, r in led.items():
    steps = json.load(open(C / r["trace"]))["steps"]
    k = collections.Counter(); seen_edit_ok = None; n = 0; after = 0; rm = 0
    for s in steps:
        for c in s.get("tool_calls") or []:
            n += 1; name = c["function_name"]; a = c.get("arguments") or {}
            try: o = json.loads(s["observation"]["content"])
            except Exception: o = {}
            if not isinstance(o, dict): o = {}
            bad = o.get("status") == "error" or o.get("error_type") or o.get("error")
            if name == "run_command":
                ck = cmd_kind(str(a.get("command", ""))); k["cmd:" + ck] += 1
                if ck == "rm": rm += 1
            else: k[name] += 1
            if name in ("edit_file", "write_file") and not bad and seen_edit_ok is None and not re.search(r"repro|reproduc|debug|inspect|check_|verify|compare|test_", str(a.get("filepath", "")).split("/")[-1]):
                seen_edit_ok = n
            if bad:
                et = str(o.get("error_type") or ("error" if o.get("error") else "status=error"))
                if name == "run_command": et = "command exit!=0/" + et
                errs[bool(r["resolved"])][f"{name}: {et}"] += 1
                if len(ex[f"{name}: {et}"]) < 2: ex[f"{name}: {et}"].append((i, str(o.get("error_message") or o.get("error") or o.get("stderr") or "")[:230].replace("\n", " ")))
    per[i] = dict(res=bool(r["resolved"]), k=k, n=n, first_src_edit=seen_edit_ok)
for res in (True, False):
    rows = [p for p in per.values() if p["res"] == res]; tot = sum(p["n"] for p in rows)
    kk = collections.Counter()
    for p in rows: kk.update(p["k"])
    print(("RESOLVED" if res else "FAILED  "), f"n={len(rows)} calls/task {tot/len(rows):.1f} |", ", ".join(f"{a} {b/len(rows):.1f}" for a, b in kk.most_common(14)))
    print("   errors/task:", ", ".join(f"{a} {b/len(rows):.2f}" for a, b in errs[res].most_common(9)))
    fe = [p["first_src_edit"] for p in rows]
    print("   first successful source edit at call: median", statistics.median(x for x in fe if x), "| never:", sum(1 for x in fe if not x))
print()
for k, v in ex.items():
    for i, m in v[:1]: print(f"EX {k} [{i}] {m}")
print()
# names the hidden tests need: is the missing name in the task text?
for i, r in sorted(led.items()):
    if r["resolved"]: continue
    out = (C / r["test_output"]).read_text(errors="replace")
    ps = tasks[i]["problem_statement"]
    names = set()
    for pat in (r"cannot import name '(\w+)'", r"No module named '([\w.]+)'", r"unexpected keyword argument '(\w+)'", r"has no attribute '(\w+)'"):
        names.update(re.findall(pat, out))
    if names:
        print(f"{i}: " + ", ".join(f"{n} -> {'IN task text' if n.split('.')[-1] in ps else 'NOT in task text'}" for n in sorted(names)[:5]), f"| task text {len(ps)} chars")
