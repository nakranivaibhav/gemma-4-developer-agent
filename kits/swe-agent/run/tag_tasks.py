"""Task tags: labels put on tasks by rules that read only the task itself, never a run result.

usage: envs/harness/bin/python -I kits/swe-agent/run/tag_tasks.py build              -> the `tags` column of tasks/manifest.csv (usable tasks)
       envs/harness/bin/python -I kits/swe-agent/run/tag_tasks.py summary <run_id>   -> resolved counts: all / untagged / tagged

A tagged task stays in every run. The tag only lets a run report a second figure next to the full one.

Tag `needs_unstated_name`: the hidden tests use a name that the real fix introduces and the task text never mentions.
  - introduced = a `def` / `class` the real fix adds (and does not also remove), or the module name of a file it creates
  - used       = the name appears as a whole word in the lines the test patch adds, and those lines do not define it themselves
  - unstated   = the name does not occur anywhere in the task text
Dunder names, names of 3 characters or fewer and `test*` names are ignored. The rule is rough on purpose: it does not see
new arguments or attributes, and it does not see tests that pin an exact message.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manifest  # noqa: E402

ROOT = manifest.ROOT
TAG = "needs_unstated_name"
DEF = r"^\s*(?:async\s+)?(?:def|class)\s+(\w+)"


def _lines(diff, sign):
    return "\n".join(l[1:] for l in (diff or "").split("\n") if l.startswith(sign) and not l.startswith(sign * 3))


def unstated_names(task):
    gold, tests, text = task["patch"], _lines(task["test_patch"], "+"), task["problem_statement"]
    new = set(re.findall(DEF, _lines(gold, "+"), re.M)) - set(re.findall(DEF, _lines(gold, "-"), re.M))
    for m in re.finditer(r"^diff --git a/(\S+) b/\S+\nnew file mode", gold, re.M):
        new.add(m.group(1).rsplit("/", 1)[-1].rsplit(".", 1)[0])
    own = set(re.findall(DEF, tests, re.M))  # names the tests define for themselves
    return sorted(n for n in new - own
                  if len(n) > 3 and not n.startswith(("__", "test")) and re.search(rf"\b{re.escape(n)}\b", tests) and n not in text)


def load_tags():
    """{task id: [tag, ...]} for tagged tasks, from the manifest."""
    return manifest.tags()


def split_line(rows, tags=None):
    """One line with the resolved count for all tasks, the untagged ones and the tagged ones. rows: [(task id, resolved)]."""
    tags = load_tags() if tags is None else tags
    def part(sel):
        return f"{sum(bool(r) for _, r in sel)}/{len(sel)}"
    return (f"all={part(rows)} untagged={part([x for x in rows if x[0] not in tags])} "
            f"tagged={part([x for x in rows if x[0] in tags])}")


def build():
    rows = manifest.rows()
    pool = {r["id"]: r["split"] for r in rows if r["split"] != "excluded"}
    found = {}
    for line in open(ROOT / "data" / "tasks.jsonl"):
        t = json.loads(line)
        if t["instance_id"] in pool:
            names = unstated_names(t)
            if names:
                found[t["instance_id"]] = names
    for r in rows:  # rewrite only this rule's tag, keep any other tags
        other = [x for x in r["tags"].split() if x != TAG]
        r["tags"] = " ".join(other + ([TAG] if r["id"] in found else []))
    manifest.write(rows)
    for name in ("tune", "holdout"):
        hit = sorted(t for t in found if pool[t] == name)
        print(f"{name}: {len(hit)} of {sum(v == name for v in pool.values())} tagged")
        for t in hit:
            print(f"   {t}: {', '.join(found[t][:5])}")


def summary(run_id):
    latest = {}
    for line in open(ROOT / "results" / "task_runs.jsonl"):
        r = json.loads(line)
        if r.get("run_id") == run_id and r.get("id"):
            latest[r["id"]] = bool(r.get("resolved"))
    print(f"{run_id}: {split_line(sorted(latest.items()))}")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "summary":
        summary(sys.argv[2])
    else:
        build()
