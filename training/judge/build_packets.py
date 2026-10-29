"""Build one read-only packet per task for the Opus task-quality judge (docs/judge_rubric.md).

A packet is a folder the judge works in (its only tools are Read / Grep / Glob, rooted there):
    TASK.md                 issue text + hints, and how the required tests are defined
    REFERENCE_FIX.diff      the task's reference fix
    TEST_CHANGES.diff       the task's test changes
    REQUIRED_TESTS.txt      the FAIL_TO_PASS list when the task has one
    repo/                   the repository at the base commit, with no .git (so no later history leaks in)

usage:
  python training/judge/build_packets.py kaggle --ids fastapi_15661,fastapi_14609 [--out DIR]
  python training/judge/build_packets.py swerebench --ids-file F [--out DIR] [--parquet P]

Kaggle tasks come from data/tasks.jsonl + data/snapshots/<id>.tgz. SWE-rebench tasks come from the parquet rows; the
repo is fetched from GitHub at base_commit (shallow, one commit). Packets live on the data SSD by default.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path("/home/vaibhav/data/gemma4/judge/packets")
PARQUET = Path("/home/vaibhav/data/gemma4/outside/swe-rebench/data/filtered-00000-of-00001.parquet")

TASK_MD = """# Task {id}

Repository: {repo} at commit {commit}. The repository is in `repo/` (no git history).

## Issue
{issue}

## Hints
{hints}

## Required tests
{required}
"""


def write_common(d, t, required):
    d.mkdir(parents=True, exist_ok=True)
    (d / "TASK.md").write_text(TASK_MD.format(
        id=t["instance_id"], repo=t["repo"], commit=t["base_commit"],
        issue=(t.get("problem_statement") or "").strip() or "(empty)",
        hints=(t.get("hints_text") or "").strip() or "(none)", required=required))
    (d / "REFERENCE_FIX.diff").write_text(t["patch"])
    (d / "TEST_CHANGES.diff").write_text(t["test_patch"])


def strip_git(repo):
    g = repo / ".git"
    if g.is_dir():
        shutil.rmtree(g)
    elif g.exists():
        g.unlink()


def build_kaggle(ids, out):
    rows = {json.loads(l)["instance_id"]: json.loads(l) for l in open(ROOT / "data/tasks.jsonl")}
    for i in ids:
        t, d = rows[i], out / i
        if (d / "repo").exists():
            print("skip (exists)", i)
            continue
        write_common(d, t, "Every test function that TEST_CHANGES.diff adds or modifies. (This task has no separate "
                           "FAIL_TO_PASS list; the scorer reads the test functions from the test changes.)")
        tmp = d / "_extract"
        tmp.mkdir()
        with tarfile.open(ROOT / f"data/snapshots/{i}.tgz") as tf:
            tf.extractall(tmp, filter="data")
        kids = [p for p in tmp.iterdir()]
        src = kids[0] if len(kids) == 1 and kids[0].is_dir() else tmp
        shutil.move(str(src), str(d / "repo"))
        shutil.rmtree(tmp, ignore_errors=True)
        strip_git(d / "repo")
        print("built", i)


def git(*a, cwd):
    subprocess.run(["git", *a], cwd=cwd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)


def build_swerebench(ids, out, parquet, workers=1):
    import glob as _glob
    from concurrent.futures import ThreadPoolExecutor
    import pandas as pd
    files = [f for p in str(parquet).split(",") for f in sorted(_glob.glob(p))]
    df = pd.concat([pd.read_parquet(f) for f in files]).drop_duplicates("instance_id").set_index("instance_id")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda i: build_one_swerebench(i, df.loc[i], out), ids))


def build_one_swerebench(i, r, out):
        d = out / i
        if (d / "repo").exists():
            print("skip (exists)", i, flush=True)
            return
        t = dict(r, instance_id=i)
        f2p = list(r["FAIL_TO_PASS"]) if r["FAIL_TO_PASS"] is not None else []
        write_common(d, t, "The tests listed in REQUIRED_TESTS.txt (FAIL_TO_PASS: they fail before the fix and must "
                           "pass after it).")
        (d / "REQUIRED_TESTS.txt").write_text("\n".join(f2p) + "\n")
        repo = d / "repo"
        repo.mkdir()
        try:
            git("init", "-q", cwd=repo)
            git("remote", "add", "origin", f"https://github.com/{r['repo']}.git", cwd=repo)
            git("fetch", "-q", "--depth", "1", "origin", r["base_commit"], cwd=repo)
            git("checkout", "-q", "FETCH_HEAD", cwd=repo)
        except subprocess.CalledProcessError as e:
            print("FAILED", i, (e.stderr or b"").decode()[-200:], file=sys.stderr, flush=True)
            shutil.rmtree(d, ignore_errors=True)
            return
        except subprocess.TimeoutExpired:
            print("FAILED", i, "timeout", file=sys.stderr, flush=True)
            shutil.rmtree(d, ignore_errors=True)
            return
        strip_git(repo)
        print("built", i, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["kaggle", "swerebench"])
    ap.add_argument("--ids", default="")
    ap.add_argument("--ids-file")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--parquet", default=str(PARQUET), help="file, glob, or comma-separated list")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    ids = [x for x in a.ids.split(",") if x]
    if a.ids_file:
        ids += [l.strip() for l in open(a.ids_file) if l.strip()]
    out = Path(a.out)
    if a.source == "kaggle":
        build_kaggle(ids, out)
    else:
        build_swerebench(ids, out, a.parquet, a.workers)
