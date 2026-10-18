"""fail_digest.py <comp_dir> <run_id> [full] — per-task signals for one run, from the ledger, traces, patches, test outputs."""
import json, re, sys, collections, statistics
from pathlib import Path
C, RUN = Path(sys.argv[1]), sys.argv[2]
tasks = {}
for l in open(C / "data/tasks.jsonl"):
    t = json.loads(l); tasks[t["instance_id"]] = t
led = {}
for l in open(C / "results/task_runs.jsonl"):
    r = json.loads(l)
    if r.get("run_id") == RUN: led[r["id"]] = r
files = lambda d: sorted(set(re.findall(r"^\+\+\+ b/(\S+)", d or "", re.M)))
added = lambda d: sum(1 for x in (d or "").split("\n") if x.startswith("+") and not x.startswith("+++"))
def sig(r):
    t = tasks[r["id"]]; st = r["trace_stats"]
    patch = (C / r["patch"]).read_text(errors="replace") if (C / r["patch"]).exists() else ""
    out = (C / r["test_output"]).read_text(errors="replace") if (C / r["test_output"]).exists() else ""
    gf, mf = files(t["patch"]), files(patch)
    gsrc = [f for f in gf if not f.startswith(("docs", "docs_src")) and not f.endswith((".md", ".rst"))]
    steps = json.load(open(C / r["trace"]))["steps"]
    acts = [s for s in steps if s.get("tool_calls")]
    first_edit = next((i for i, s in enumerate(acts) if s["tool_calls"][0]["function_name"] in ("edit_file", "write_file")), None)
    py = []
    for s in acts:
        c = s["tool_calls"][0]
        if c["function_name"] == "run_command" and re.search(r"pytest|python -m unittest", str(c["arguments"].get("command", ""))):
            try: o = json.loads(s["observation"]["content"])
            except Exception: o = {}
            ec = o.get("exit_code", (o.get("details") or {}).get("exit_code") if isinstance(o.get("details"), dict) else None)
            py.append(ec)
    summ = re.findall(r"^(?:FAILED|ERROR) .*", out, re.M)
    e = re.findall(r"^E\s+(.*)", out, re.M)
    return dict(id=r["id"], res=bool(r["resolved"]), gold_files=len(gf), gold_src=gsrc, model_files=mf,
                hit=len([f for f in gsrc if f in mf]), nsrc=len(gsrc), extra=[f for f in mf if f not in gf],
                gold_add=added(t["patch"]), model_add=added(patch), calls=st.get("budget_counted_calls"), ended=st.get("ended_with"),
                first_edit=first_edit, nacts=len(acts), pytest=py, edit_fail=st.get("files_edit_failed_only"), malformed=st.get("malformed_calls"),
                rejected=st.get("rejected_calls"), tool_err=st.get("tool_errors"), repeated=st.get("repeated_calls"),
                tests=(r.get("tests_passed"), r.get("tests_failed"), r.get("tests_errors")), exit=r.get("test_exit_code"),
                summ=summ, e=e, test_files=files(t["test_patch"]), mins=round(r["agent_seconds"] / 60, 1), herr=r.get("harness_error"))
S = [sig(r) for r in led.values()]
def agg(name, rows):
    if not rows: return
    med = lambda k: statistics.median(x[k] for x in rows if x[k] is not None)
    print(f"{name}: n={len(rows)} | calls med {med('calls')} | first edit at action med {med('first_edit')} | "
          f"never ran pytest {sum(1 for x in rows if not x['pytest'])} | last pytest run exit 0: {sum(1 for x in rows if x['pytest'] and x['pytest'][-1] == 0)} | "
          f"touched every gold source file {sum(1 for x in rows if x['nsrc'] and x['hit'] == x['nsrc'])} | touched none {sum(1 for x in rows if x['nsrc'] and x['hit'] == 0)} | "
          f"model added lines med {med('model_add')} vs gold {med('gold_add')} | malformed med {med('malformed')} | tool errors med {med('tool_err')} | patch has extra files {sum(1 for x in rows if x['extra'])}")
agg("RESOLVED", [x for x in S if x["res"]]); agg("FAILED  ", [x for x in S if not x["res"]])
if len(sys.argv) > 3:
    for x in sorted((x for x in S if not x["res"]), key=lambda x: x["id"]):
        print(f"\n## {x['id']} | tests p/f/e {x['tests']} exit {x['exit']} | calls {x['calls']} ended {x['ended']} {x['mins']}min | gold +{x['gold_add']} in {x['gold_files']} files, model +{x['model_add']}")
        print(f"   gold src {x['gold_src'][:6]}{'…' if len(x['gold_src']) > 6 else ''} | model {x['model_files'][:6]} | hit {x['hit']}/{x['nsrc']}")
        print(f"   first edit at action {x['first_edit']}/{x['nacts']} | pytest exits {x['pytest'][-8:]} | malformed {x['malformed']} rejected {x['rejected']} tool_err {x['tool_err']} repeated {x['repeated']} edit-failed-only {x['edit_fail']}")
        for l in x["summ"][:3]: print("   " + l[:170])
        if len(x["summ"]) > 3: print(f"   … {len(x['summ'])} summary lines")
        if x["e"]: print("   E " + x["e"][0][:170])
        if x["herr"]: print("   HARNESS " + str(x["herr"])[:120])
