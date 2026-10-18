"""The task manifest: tasks/manifest.csv, one row per task. It says what each task is and where its setup lives,
never how our agent did on it (that is the run ledger, results/task_runs.jsonl).

Columns
  id               task id, e.g. fastapi_15661
  source           where the task comes from: kaggle-public (later swe-gym, swe-smith, ...)
  repo, base_commit  the code the task starts from
  split            tune | holdout | excluded.  holdout is the test set: never trained on, never tuned on
  lists            named subsets, space-separated (smoke6, screen_01, ...)
  snapshot         the repo files, relative to the repo root (data/snapshots/<id>.tgz)
  graph, embedding the files behind the code-graph tools (empty when the task has none)
  sandbox          the Docker image the task runs in
  origin           where `make setup` downloads the task data from
  gold_passes      yes when the real fix passes the hidden tests in the sandbox
  empty_fails      yes when an empty patch fails them (otherwise the task measures nothing)
  excluded_reason  why the task is out, for split=excluded
  tags             labels read from the task text only (kits/swe-agent/run/tag_tasks.py), space-separated

usage: python kits/swe-agent/run/manifest.py ids <selector>    task ids, one per line
       python kits/swe-agent/run/manifest.py check             do the setup files of every usable task exist?
       python kits/swe-agent/run/manifest.py summary           counts per source / split / list

A selector is one of: a split (tune, holdout, excluded), `train` (= tune; training reads only this), `usable`
(= not excluded), `all`, a list name (smoke6, ...), a path to a file of ids, or several of these joined by `+`.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repo root
MANIFEST = ROOT / "tasks" / "manifest.csv"
COLUMNS = ["id", "source", "repo", "base_commit", "split", "lists", "snapshot", "graph", "embedding", "sandbox",
           "origin", "gold_passes", "empty_fails", "excluded_reason", "tags"]
SPLITS = ("tune", "holdout", "excluded")


def rows():
    with open(MANIFEST, newline="") as fh:
        return list(csv.DictReader(fh))


def write(all_rows):
    with open(MANIFEST, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in all_rows:
            w.writerow({c: r.get(c, "") for c in COLUMNS})


def ids(selector):
    """Task ids for a selector, in manifest order (file selectors keep the file's order)."""
    out = []
    for part in str(selector).split("+"):
        path = Path(part)
        if path.suffix and path.is_file():  # a plain file of ids
            out += path.read_text().split()
            continue
        rs = rows()
        if part == "all":
            sel = rs
        elif part == "usable":
            sel = [r for r in rs if r["split"] != "excluded"]
        elif part == "train":
            sel = [r for r in rs if r["split"] == "tune"]
        elif part in SPLITS:
            sel = [r for r in rs if r["split"] == part]
        else:
            sel = [r for r in rs if part in r["lists"].split()]
            if not sel:
                sys.exit(f"manifest: no split, list or file called {part!r}")
        out += [r["id"] for r in sel]
    return list(dict.fromkeys(out))  # de-duplicated, order kept


def holdout():
    return set(ids("holdout"))


def tags():
    """{task id: [tag, ...]} for tagged tasks only."""
    return {r["id"]: r["tags"].split() for r in rows() if r["tags"].strip()}


def check():
    bad = 0
    for r in rows():
        if r["split"] == "excluded":
            continue
        for col in ("snapshot", "graph", "embedding"):
            if r[col] and not (ROOT / r[col]).exists():
                print(f"MISSING {r['id']} {col} {r[col]}")
                bad += 1
    print(f"{'OK' if not bad else f'{bad} missing'}: {len(rows())} rows checked")
    return bad


def summary():
    rs = rows()
    print(f"{len(rs)} tasks")
    for col in ("source", "split"):
        print(f"  {col}: " + ", ".join(f"{k}={v}" for k, v in Counter(r[col] for r in rs).most_common()))
    lists = Counter(n for r in rs for n in r["lists"].split())
    print("  lists: " + ", ".join(f"{k}={v}" for k, v in sorted(lists.items())))
    tagged = Counter(t for r in rs for t in r["tags"].split())
    print("  tags: " + ", ".join(f"{k}={v}" for k, v in tagged.items()))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "ids":
        print("\n".join(ids(sys.argv[2])))
    elif cmd == "check":
        sys.exit(1 if check() else 0)
    else:
        summary()
