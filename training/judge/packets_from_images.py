"""Build judge packets from a task's Docker image when its GitHub repo or commit is gone (2026-10-24: 29 of 7,136).

usage: envs/judge/bin/python training/judge/packets_from_images.py --ids-file F [--workers 3] [--keep-image]

SWE-rebench images hold the repo at /testbed at the base commit. Pull the image, copy /testbed out of a stopped
container (never run), drop .git, then remove the container (and the image unless --keep-image). The TASK.md / diff
files are written the same way as build_packets.py.
"""
import argparse
import glob
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from build_packets import OUT, strip_git, write_common  # noqa: E402

PARQUETS = ["/home/vaibhav/data/gemma4/outside/swe-rebench/data/filtered-00000-of-00001.parquet",
            *sorted(glob.glob("/home/vaibhav/data/gemma4/outside/swe-rebench-leaderboard/data/20*.parquet"))]


def sh(*a, timeout=3600):
    return subprocess.run(list(a), check=True, capture_output=True, text=True, timeout=timeout).stdout.strip()


def one(i, r, keep):
    d = OUT / i
    if (d / "repo").exists():
        print("skip (exists)", i, flush=True)
        return
    img = r["docker_image"]
    try:
        sh("docker", "pull", "-q", img)
        cid = sh("docker", "create", img)
        try:
            write_common(d, dict(r, instance_id=i), "The tests listed in REQUIRED_TESTS.txt (FAIL_TO_PASS: they fail "
                                                    "before the fix and must pass after it).")
            f2p = list(r["FAIL_TO_PASS"]) if r["FAIL_TO_PASS"] is not None else []
            (d / "REQUIRED_TESTS.txt").write_text("\n".join(f2p) + "\n")
            sh("docker", "cp", f"{cid}:/testbed", str(d / "repo"))
        finally:
            sh("docker", "rm", cid)
        strip_git(d / "repo")
        if not keep:
            sh("docker", "rmi", img)
        print("built", i, flush=True)
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        print("FAILED", i, str(getattr(e, "stderr", "") or e)[-200:], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--keep-image", action="store_true")
    a = ap.parse_args()
    df = pd.concat([pd.read_parquet(p) for p in PARQUETS]).drop_duplicates("instance_id").set_index("instance_id")
    ids = [l.strip() for l in open(a.ids_file) if l.strip()]
    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(lambda i: one(i, df.loc[i], a.keep_image), ids))
