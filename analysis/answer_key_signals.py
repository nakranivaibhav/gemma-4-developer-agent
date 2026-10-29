"""Outcome-blind signals per task, to draft the judge's answer key (2026-10-24).

usage: python3 analysis/answer_key_signals.py [--tasks data/tasks.jsonl] [--skip-prefix rich_] > out.jsonl

For each task, from the issue text, the reference fix (patch) and the test changes (test_patch) only:
  new_names     names the fix adds (def / class / keyword argument / new file stem) that the added test lines use
                and that neither the issue text nor the hints contain
  new_strings   string literals of 12+ characters in the added test lines that the fix also adds and that the
                issue text does not contain (an exact message or format the tests check)
  skip_markers  added test lines with skip / skipif / needs_* markers (a required test that may always skip)
  env_markers   added test lines touching dates, clocks, the network, platform or Python version checks
Solve rates are attached separately (they are outcome data, used only to pick candidate clean tasks).
"""
import argparse
import json
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", default="data/tasks.jsonl")
ap.add_argument("--skip-prefix", default="rich_")
a = ap.parse_args()


def added(diff):
    return [l[1:] for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]


def new_files(diff):
    # a file the diff creates: "--- /dev/null" followed by "+++ b/path"
    out, prev = [], ""
    for l in diff.splitlines():
        if l.startswith("+++ ") and prev.startswith("--- /dev/null"):
            out.append(l[4:].removeprefix("b/"))
        prev = l
    return out


DEF = re.compile(r"^\s*(?:async\s+)?(?:def|class)\s+([A-Za-z_]\w*)")
KW = re.compile(r"[(,]\s*([a-z_]\w*)\s*(?::[^=,)]*)?=")  # keyword parameters in a def line
STR = re.compile(r"""(?:"([^"\\\n]{12,})"|'([^'\\\n]{12,})')""")
SKIP = re.compile(r"skip|needs_|xfail", re.I)
ENV = re.compile(r"datetime|date\.today|time\.time|freezegun|socket|requests\.get|urlopen|httpbin|sys\.version|platform\.|os\.name|random\.", re.I)

for line in open(a.tasks):
    t = json.loads(line)
    tid = t["instance_id"]
    if a.skip_prefix and tid.startswith(a.skip_prefix):
        continue
    text = (t.get("problem_statement") or "") + "\n" + (t.get("hints_text") or "")
    fix_add, test_add = added(t["patch"]), added(t["test_patch"])
    test_blob = "\n".join(test_add)
    names = set()
    for l in fix_add:
        m = DEF.match(l)
        if m:
            names.add(m.group(1))
            if re.match(r"^\s*(?:async\s+)?def", l):
                names.update(k for k in KW.findall(l) if k not in ("self", "cls"))
    for f in new_files(t["patch"]):
        stem = f.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if stem and stem != "__init__":
            names.add(stem)
    new_names = sorted(n for n in names
                       if n not in text and not n.startswith("_") and re.search(r"\b%s\b" % re.escape(n), test_blob))
    fix_strings = {s for l in fix_add for g in STR.findall(l) for s in g if s}
    test_strings = {s for l in test_add for g in STR.findall(l) for s in g if s}
    new_strings = sorted(s for s in test_strings & fix_strings if s not in text)
    print(json.dumps({
        "id": tid,
        "new_names": new_names,
        "new_strings": new_strings[:8],
        "skip_markers": [l.strip()[:120] for l in test_add if SKIP.search(l)][:5],
        "env_markers": [l.strip()[:120] for l in test_add if ENV.search(l)][:5],
        "issue_chars": len(t.get("problem_statement") or ""),
        "fix_files": len(re.findall(r"^\+\+\+ ", t["patch"], re.M)),
    }))
