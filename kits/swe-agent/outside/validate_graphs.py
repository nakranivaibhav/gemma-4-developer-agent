"""Check every outside task's code graph before we trust (or share) it (2026-10-26).

usage: envs/harness/bin/python -I kits/swe-agent/outside/validate_graphs.py <tasks.jsonl> [<out.jsonl>] [--procs 8]

Per task (graph stem = <repo short>_<base commit>, shared by tasks on the same commit):
  G1 files   the graph .json exists, parses, has >= 1 node (the random .npz is not checked: no tool reads it)
  G2 base    the repo the graph was built from (the judge packet's repo/) is the task's BASE commit: the reference fix
             applies cleanly (git apply --check), is not already applied (reverse check fails), and the test patch
             applies cleanly. A graph built from a later commit would carry the fix in its node texts.
  G3 leak    no line the reference fix ADDS (>= 25 chars, not also a removed line) appears in a node text unless that
             exact line already exists somewhere in the base repo
A task passes when G1-G3 all hold. Output: one JSON line per task + a summary line; exit 0 always (it reports).
"""
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import outside  # noqa: E402

PACKETS = Path("/home/vaibhav/data/gemma4/judge/packets")


def added_lines(patch: str):
    plus, minus = set(), set()
    for ln in patch.splitlines():
        if ln.startswith("+++") or ln.startswith("---"):
            continue
        if ln.startswith("+"):
            plus.add(ln[1:].strip())
        elif ln.startswith("-"):
            minus.add(ln[1:].strip())
    return {x for x in plus - minus if len(x) >= 25}


def git_check(repo: Path, patch: str, reverse=False) -> bool:
    if not patch.strip():
        return True
    with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as f:
        f.write(patch)
    try:
        args = ["git", "apply", "--check", "--whitespace=nowarn"] + (["-R"] if reverse else []) + [f.name]
        return subprocess.run(args, cwd=repo, capture_output=True, text=True).returncode == 0
    finally:
        os.unlink(f.name)


def in_repo(repo: Path, line: str) -> bool:
    return subprocess.run(["grep", "-rqF", "--include=*.py", "--", line, "."], cwd=repo).returncode == 0


def check(row):
    tid, stem = row["instance_id"], outside.graph_stem(row)
    gpath, epath = outside.GRAPH_DIR / f"{stem}.json", outside.EMB_DIR / f"{stem}.npz"
    rec = {"id": tid, "stem": stem, "problems": []}
    # G1
    try:
        g = json.loads(gpath.read_text())
        ids = [n["id"] for n in g["nodes"]]
        rec["nodes"], rec["edges"] = len(ids), len(g["edges"])
        if not ids:
            rec["problems"].append("G1 empty graph")
    except Exception as e:  # noqa: BLE001
        rec["problems"].append(f"G1 unreadable: {type(e).__name__}: {str(e)[:120]}")
        return rec
    # G2
    repo = PACKETS / tid / "repo"
    if not repo.is_dir():
        rec["problems"].append("G2 no packet repo (graph built from the image's /testbed); base not verifiable")
    else:
        if not git_check(repo, row["patch"]):
            rec["problems"].append("G2 reference fix does not apply to the graph's repo")
        elif git_check(repo, row["patch"], reverse=True):
            rec["problems"].append("G2 reference fix already applied in the graph's repo (later commit)")
        if not git_check(repo, row.get("test_patch") or ""):
            rec["problems"].append("G2 test patch does not apply to the graph's repo")
    # G3
    adds = added_lines(row["patch"])
    if adds:
        texts = "\n".join(n.get("text") or "" for n in g["nodes"])
        hits = [a for a in adds if a in texts and not (repo.is_dir() and in_repo(repo, a))]
        if hits:
            rec["problems"].append(f"G3 {len(hits)} fix lines in node texts, e.g. {hits[0][:80]!r}")
    rec["ok"] = not rec["problems"]
    return rec


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    procs = int(sys.argv[sys.argv.index("--procs") + 1]) if "--procs" in sys.argv else 8
    specs = outside.load_specs(args[0])
    out = Path(args[1]) if len(args) > 1 else None
    rows = list(specs.values())
    recs = []
    with mp.Pool(procs) as pool:
        for r in pool.imap(check, rows, chunksize=8):   # imap, not map: a progress line keeps a watchdog fed
            recs.append(r)
            if len(recs) % 200 == 0:
                print(f"validated {len(recs)}/{len(rows)}", file=sys.stderr, flush=True)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r) + "\n" for r in recs))
    bad = [r for r in recs if not r.get("ok")]
    kinds = {}
    for r in bad:
        for p in r["problems"]:
            kinds[p[:2] + " " + p.split(" ", 2)[1]] = kinds.get(p[:2] + " " + p.split(" ", 2)[1], 0) + 1
    print(json.dumps({"tasks": len(recs), "ok": len(recs) - len(bad), "bad": len(bad), "by_check": kinds,
                      "graphs": len({r["stem"] for r in recs})}))


if __name__ == "__main__":
    main()
