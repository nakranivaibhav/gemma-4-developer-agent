"""Write outside (SWE-rebench) tasks in our tasks.jsonl format, plus what outside.py needs to run and score them.

usage: envs/judge/bin/python -I kits/swe-agent/outside/export_tasks.py --out <tasks.jsonl> [--parquet P ...] \
           (<instance_id> ... | --ids <file of ids>)
Default sources: the SWE-rebench filtered set, its test split and every leaderboard month under
/home/vaibhav/data/gemma4/outside/; the first source holding an id wins. Unknown ids are an error.

Each row: the data/tasks.jsonl fields (instance_id, repo, base_commit, patch, test_patch, problem_statement,
hints_text, created_at) + FAIL_TO_PASS / PASS_TO_PASS (lists) + docker_image + test_cmd + env_vars
(+ environment_setup_commit, version, source). A row with docker_image is what switches run_tasks.py to outside mode.
To use another issue text (e.g. a repaired one), replace problem_statement in the written rows; nothing else reads it.
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

OUT_ROOT = Path(os.environ.get("OUTSIDE_DATA", "/home/vaibhav/data/gemma4/outside"))
DEFAULT = [str(OUT_ROOT / "swe-rebench/data/filtered-00000-of-00001.parquet"),
           *sorted(glob.glob(str(OUT_ROOT / "swe-rebench/data/test-*.parquet"))),
           *sorted(glob.glob(str(OUT_ROOT / "swe-rebench-leaderboard/data/20*.parquet")))]


def as_list(v):
    return [str(x) for x in (list(v) if v is not None else [])]


def row_out(r, source):
    ic = r["install_config"] or {}
    env_vars = {k: v for k, v in (ic.get("env_vars") or {}).items() if v is not None}
    return {
        "instance_id": r["instance_id"], "repo": r["repo"], "base_commit": r["base_commit"],
        "patch": r["patch"], "test_patch": r["test_patch"], "problem_statement": r["problem_statement"],
        "hints_text": r.get("hints_text") or "", "created_at": str(r.get("created_at") or ""),
        "FAIL_TO_PASS": as_list(r["FAIL_TO_PASS"]), "PASS_TO_PASS": as_list(r["PASS_TO_PASS"]),
        "docker_image": r["docker_image"], "test_cmd": ic.get("test_cmd") or "", "env_vars": env_vars,
        "environment_setup_commit": r.get("environment_setup_commit") or "", "version": str(r.get("version") or ""),
        "source": Path(source).name,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids", nargs="*")
    ap.add_argument("--ids", dest="ids_file", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parquet", action="append", help="source parquet (repeatable); default: all SWE-rebench sets")
    a = ap.parse_args()
    want = list(a.ids) + (a.ids_file.read_text().split() if a.ids_file else [])
    if not want:
        sys.exit("no ids given")
    # A task can sit in several sources; the swe-rebench test-* copies of some carry no docker_image (NaN), while a
    # leaderboard copy has one (2026-10-29: 18 pass-1 tasks crashed on the NaN image). Prefer a row with an image.
    found, imageless = {}, {}
    for src in a.parquet or DEFAULT:
        left = set(want) - set(found)
        if not left:
            break
        df = pd.read_parquet(src)
        for r in df[df.instance_id.isin(left)].to_dict(orient="records"):
            row = row_out(r, src)
            if isinstance(row["docker_image"], str) and row["docker_image"]:
                found.setdefault(r["instance_id"], row)
            else:
                imageless.setdefault(r["instance_id"], row)
    no_image = [i for i in imageless if i not in found]
    if no_image:
        print(f"warning: no docker_image in any source (cannot run): {no_image}", file=sys.stderr)
    found.update({i: imageless[i] for i in no_image})
    missing = [i for i in want if i not in found]
    if missing:
        sys.exit(f"not found in any source: {missing[:10]}{' ...' if len(missing) > 10 else ''}")
    bad = [i for i in want if not found[i]["test_cmd"] or not found[i]["FAIL_TO_PASS"]]
    if bad:
        print(f"warning: no test_cmd or empty FAIL_TO_PASS (cannot be scored): {bad}", file=sys.stderr)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fh:
        for i in dict.fromkeys(want):
            fh.write(json.dumps(found[i]) + "\n")
    print(f"wrote {len(set(want))} tasks to {a.out}")


if __name__ == "__main__":
    main()
