"""Outside-task manifest: one row per SWE-rebench task with its state at every stage, DERIVED from the run artifacts
(never edited by hand, so it cannot drift; rebuild it any time).

usage:
  envs/judge/bin/python -I kits/swe-agent/outside/manifest.py build            -> tasks/outside_manifest.csv (+ summary)
  envs/judge/bin/python -I kits/swe-agent/outside/manifest.py next N NAME [--prefer repaired|clean]
        -> data/outside/NAME.ids + NAME.jsonl: the next N approved training tasks not yet in any batch
           (data/outside/*.ids), exported with their repaired text where they have one

Stages and where they are read from:
  judge     runs/judge/all_v3/            clean (solvable as written) / flagged (+ which rubric items)
  repair    runs/judge/fix2_*/ + *_rejudge*/  repaired (minimal note passes the blind re-judge; else the full note
            passes) / leak / unfixable / rejudge_fail / pending (flagged, not repaired yet) / env (R6, not repaired)
  approved  clean, or repaired: the task is usable as training text
  split     heldout (docs/outside_tasks_judge.md repos) / train / duplicate (the same task under a second id,
            DUPLICATES below: never approved, so it is never trained on or tested twice)
  gold      runs/outside/*/gold/gold.jsonl (latest run per task): valid / gold_fails / empty_passes / unchecked
            + ref_fix_test / empty_patch_test: pass / fail / not_run (each test on its own; timed-out checks ignored)
  graph     runs/outside/*/graph_validation*.jsonl: ok / missing / failed checks (G1 G2 G3)
Model results (E2B or any other) are NOT task properties and stay out of this file: they live in the run ledger
results/outside_task_runs.jsonl (one row per try, with the model and settings); the build prints a summary of them.
"""
import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
J = "/home/vaibhav/data/gemma4/judge"
OUT_CSV = os.path.join(ROOT, "tasks/outside_manifest.csv")
ADDENDA = os.path.join(ROOT, "data/outside/approved_addenda.json")
# The same task listed under two ids (a repo that moved organisations): duplicate id -> the id we keep.
DUPLICATES = {"jmcgeheeiv__pyfakefs-1286": "pytest-dev__pyfakefs-1286"}   # repo moved to pytest-dev; same PR, same commit
PARQUETS = ["/home/vaibhav/data/gemma4/outside/swe-rebench/data/filtered-00000-of-00001.parquet",
            *sorted(glob.glob("/home/vaibhav/data/gemma4/outside/swe-rebench-leaderboard/data/20*.parquet"))]


def jload(pattern):
    out = {}
    for f in glob.glob(pattern):
        if f.endswith("run.json"):
            continue
        try:
            o = json.load(open(f))
        except ValueError:
            continue
        out[o["id"]] = o
    return out


def r7(o):
    return ((o or {}).get("verdict") or {}).get("R7", {}).get("answer")


def build():
    v = pd.read_parquet(f"{J}/all_v3_verdicts.parquet").set_index("id")
    held = {l.strip() for l in open(f"{J}/heldout_repos.txt") if l.strip()}
    src = pd.concat([pd.read_parquet(p, columns=["instance_id", "docker_image", "install_config"]) for p in PARQUETS])
    src = src.drop_duplicates("instance_id").set_index("instance_id")
    fixes, rj, rjf = {}, {}, {}
    for run in sorted(glob.glob(os.path.join(ROOT, "runs/judge/fix2_*"))):
        name = os.path.basename(run)
        if name.endswith("_rejudge_full"):
            rjf.update(jload(f"{run}/*.json"))
        elif name.endswith("_rejudge"):
            rj.update(jload(f"{run}/*.json"))
        elif "kaggle" not in name:
            fixes.update(jload(f"{run}/*.json"))
    gold = defaultdict(dict)
    slow = defaultdict(Counter)  # (id, kind) -> Counter of slow failing verdicts by (exit, f2p, p2p)
    for f in sorted(glob.glob(os.path.join(ROOT, "runs/outside/*/gold/gold.jsonl")), key=os.path.getmtime):
        for l in open(f):
            r = json.loads(l)
            if r.get("exit") == -1 and "Timeout" in str(r.get("error") or ""):
                continue  # the check timed out on an overloaded Docker host: says nothing about the task
            if r.get("seconds", 0) > 240 and ((r["kind"] == "gold") != bool(r.get("resolved"))):
                # a failing verdict from a check that ran > 240 s: overloaded host (2026-10-27), not trusted alone.
                # Trusted once two such checks give the same exit code and test counts (2026-10-29: 40 tasks failed
                # identically 2-3 times, on a busy and on a quiet revideo; they are slow suites, not load artifacts)
                sig = (r.get("exit"), r.get("f2p"), r.get("p2p"))
                slow[(r["id"], r["kind"])][sig] += 1
                if slow[(r["id"], r["kind"])][sig] < 2:
                    continue
            gold[r["id"]][r["kind"]] = r
    graphv = {}  # kits/swe-agent/outside/validate_graphs.py output (G1 files, G2 base commit, G3 no fix leak)
    for f in sorted(glob.glob(os.path.join(ROOT, "runs/outside/*/graph_validation*.jsonl")), key=os.path.getmtime):
        for l in open(f):
            r = json.loads(l)
            graphv[r["id"]] = r
    e2b = defaultdict(Counter)
    led = os.path.join(ROOT, "results/outside_task_runs.jsonl")
    if os.path.exists(led):
        for l in open(led):
            r = json.loads(l)
            if "e2b" not in ((r.get("settings") or {}).get("model") or "").lower():
                continue
            if str(r.get("error") or "").startswith(("INVALID_TASK", "PULL_FAILED", "GRAPH_FAILED")):
                continue  # stream mode: no agent try happened (see kits/swe-agent/outside/outside.py)
            e2b[r["id"]]["tries"] += 1
            e2b[r["id"]]["solved"] += bool(r.get("resolved"))
            e2b[r["id"]]["solved_clock"] += bool(r.get("resolved_kaggle"))
    addenda, rows = {}, []
    for i, t in v.iterrows():
        flags = "".join(k[1] for k in ["R1", "R2", "R3", "R4", "R5", "R6"] if t[k])
        if t.ok:
            judge, repair = "clean", ""
        else:
            judge = "flagged"
            fx = fixes.get(i)
            if t.R6:
                repair = "env"
            elif fx is None:
                repair = "pending"
            elif not fx.get("final_addendum"):
                repair = "unfixable"
            elif (fx.get("leak") or {}).get("leaks_implementation") == "yes":
                repair = "leak"
            elif r7(rj.get(i)) == "yes":
                repair, addenda[i] = "repaired", fx["final_addendum"].strip()
            elif r7(rjf.get(i)) == "yes":
                repair, addenda[i] = "repaired_full", fx["fix"]["addendum"].strip()
            elif i in rj:
                repair = "rejudge_fail"
            else:
                repair = "rejudge_pending"
        approved = judge == "clean" or repair in ("repaired", "repaired_full")
        approved = approved and i not in DUPLICATES
        g = gold.get(i, {})
        if "gold" not in g:
            gs = "unchecked"
        elif not g["gold"].get("resolved"):
            gs = "gold_fails"
        elif g.get("empty", {}).get("resolved"):
            gs = "empty_passes"
        else:
            gs = "valid"
        ref = "pass" if g.get("gold", {}).get("resolved") else ("fail" if "gold" in g else "not_run")
        emp = "fail" if "empty" in g and not g["empty"].get("resolved") else ("pass" if "empty" in g else "not_run")
        gv = graphv.get(i)
        gr = "missing" if gv is None else ("ok" if gv.get("ok") else ";".join(p[:2] for p in gv["problems"]))
        s = src.loc[i] if i in src.index else None
        cfg = (s["install_config"] if s is not None else None) or {}
        rows.append(dict(id=i, repo=t.repo, split="duplicate" if i in DUPLICATES else "heldout" if t.repo in held else "train",
                         created=str(t.created.date()), judge=judge, flags=flags, repair=repair,
                         approved=int(approved), python=str(cfg.get("python", "")),
                         image=s["docker_image"] if s is not None else "", gold=gs, ref_fix_test=ref, empty_patch_test=emp, graph=gr))
    os.makedirs(os.path.dirname(ADDENDA), exist_ok=True)
    json.dump(addenda, open(ADDENDA, "w"))
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    d = pd.DataFrame(rows)
    tr = d[d.split == "train"]
    print(f"manifest {len(d)} tasks -> {OUT_CSV}")
    print("judge:", dict(Counter(d.judge)), "| repair:", dict(Counter(d[d.judge == "flagged"].repair)))
    print(f"train approved {tr.approved.sum()} (clean {(tr.judge == 'clean').sum()}, repaired "
          f"{tr.repair.isin(['repaired', 'repaired_full']).sum()}) | gold:", dict(Counter(tr[tr.approved == 1].gold)))
    sc = {i: c for i, c in e2b.items() if c["tries"]}  # summary from the ledger only, not written to the manifest
    print(f"e2b (ledger) screened {len(sc)} tasks, {sum(c['tries'] for c in sc.values())} tries, solved "
          f"{sum(c['solved'] for c in sc.values())} (clock {sum(c['solved_clock'] for c in sc.values())}), "
          f"tasks solved at least once {sum(c['solved'] > 0 for c in sc.values())}")
    return d


def nxt(n, name, prefer):
    d = build()
    claimed = set()  # every id already handed to a batch (data/outside/*.ids), checked or not yet
    for f in glob.glob(os.path.join(ROOT, "data/outside/*.ids")):
        claimed |= {l.strip() for l in open(f) if l.strip()}
    d = d[(d.split == "train") & (d.approved.astype(int) == 1) & (d.gold == "unchecked") & ~d.id.isin(claimed)]
    d = d.assign(rep=d.repair.isin(["repaired", "repaired_full"]))
    d = d.sort_values("rep", ascending=(prefer != "repaired"), kind="stable")
    ids = list(d.id.head(n))
    if not ids:
        print("NEXT none")
        return
    base = os.path.join(ROOT, "data/outside", name)
    open(base + ".ids", "w").write("\n".join(ids) + "\n")
    subprocess.run([sys.executable, "-I", os.path.join(ROOT, "kits/swe-agent/outside/export_tasks.py"),
                    "--ids", base + ".ids", "--out", base + ".jsonl"], check=True, stdout=subprocess.DEVNULL)
    add = json.load(open(ADDENDA)) if os.path.exists(ADDENDA) else {}
    rows = [json.loads(l) for l in open(base + ".jsonl")]
    with open(base + ".jsonl", "w") as f:
        for r in rows:
            if r["instance_id"] in add:
                r["problem_statement"] = r["problem_statement"].rstrip() + "\n\n" + add[r["instance_id"]]
                r["repaired"] = True
            f.write(json.dumps(r) + "\n")
    print(f"NEXT {len(ids)} tasks -> {base}.ids/.jsonl ({sum(1 for r in rows if r.get('repaired'))} repaired)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "next"])
    ap.add_argument("n", nargs="?", type=int)
    ap.add_argument("name", nargs="?")
    ap.add_argument("--prefer", default="repaired", choices=["repaired", "clean"])
    a = ap.parse_args()
    build() if a.cmd == "build" else nxt(a.n, a.name, a.prefer)
