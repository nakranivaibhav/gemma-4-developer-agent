"""Write a gold check's results into tasks/manifest.csv (columns gold_passes, empty_fails).

usage: envs/harness/bin/python -I kits/swe-agent/check/gold_to_manifest.py <gold_out_dir> <nopatch_out_dir>
Both dirs come from gold_run.sh: the first with the real fix, the second with NO_PATCH=1. Tasks missing from a run keep
their old value. The split column is never changed here: moving a task in or out of `excluded` is a human decision.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run"))
import manifest  # noqa: E402


def resolved(out_dir):
    return {r["id"]: bool(r["resolved"]) for r in map(json.loads, open(Path(out_dir) / "gold.jsonl"))}


def main():
    gold, empty = resolved(sys.argv[1]), resolved(sys.argv[2])
    rows = manifest.rows()
    changed = []
    for r in rows:
        new = dict(r)
        if r["id"] in gold:
            new["gold_passes"] = "yes" if gold[r["id"]] else "no"
        if r["id"] in empty:
            new["empty_fails"] = "no" if empty[r["id"]] else "yes"
        if (new["gold_passes"], new["empty_fails"]) != (r["gold_passes"], r["empty_fails"]):
            changed.append(f"{r['id']}: gold_passes {r['gold_passes']}->{new['gold_passes']}, "
                           f"empty_fails {r['empty_fails']}->{new['empty_fails']} (split {r['split']})")
        r.update(new)
    manifest.write(rows)
    print(f"gold: {sum(gold.values())}/{len(gold)} pass with the real fix | empty patch: {sum(empty.values())}/{len(empty)} pass")
    print("changed rows:" if changed else "no row changed", *changed, sep="\n  ")


if __name__ == "__main__":
    main()
