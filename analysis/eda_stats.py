"""Per-task facts for the public tasks: statement size, gold patch shape, test patch shape, graph size.

usage: envs/harness/bin/python -I analysis/eda_stats.py <data_dir> <out.json>
Reads tasks.jsonl, graphs/, snapshots/ and the model tokenizer; prints only a short summary.
"""
import collections
import json
import os
import re
import statistics
import sys

data_dir, out_path = sys.argv[1], sys.argv[2]
tasks = [json.loads(line) for line in open(os.path.join(data_dir, "tasks.jsonl"))]

try:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(os.path.join(data_dir, "external", "model"))
except Exception as exc:  # token counts are optional
    print("tokenizer unavailable:", repr(exc)[:120])
    tok = None

TEST_RE = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_test\.py$|(^|/)conftest\.py$")
DOC_RE = re.compile(r"\.(md|rst|txt|mdx)$|(^|/)docs?/")


def kind(path):
    if TEST_RE.search(path):
        return "test"
    if path.startswith("docs_src/"):
        return "doc example code"
    if path.endswith(".py") or path.endswith(".pyi"):
        return "library code"
    if DOC_RE.search(path):
        return "docs"
    return "other"


def parse_patch(text):
    files, cur = [], None
    for line in text.splitlines():
        if line.startswith("--- "):
            old = line[4:].strip()
            cur = {"old": old, "path": None, "added": 0, "removed": 0, "hunks": 0, "new": old == "/dev/null",
                   "added_lines": []}
            files.append(cur)
        elif line.startswith("+++ ") and cur is not None and cur["path"] is None:
            new = line[4:].strip()
            src = new if new != "/dev/null" else cur["old"]
            cur["path"] = re.sub(r"^[ab]/", "", src)
            cur["deleted"] = new == "/dev/null"
        elif cur is None or cur["path"] is None:
            continue
        elif line.startswith("@@"):
            cur["hunks"] += 1
            if line.startswith("@@ -0,0 "):  # these patches write new files as "--- a/<path>", never /dev/null
                cur["new"] = True
        elif line.startswith("+"):
            cur["added"] += 1
            cur["added_lines"].append(line[1:])
        elif line.startswith("-"):
            cur["removed"] += 1
    return [f for f in files if f["path"]]


graph_cache = {}


def graph_info(repo_short, sha):
    key = f"{repo_short}_{sha}"
    if key in graph_cache:
        return graph_cache[key]
    path = os.path.join(data_dir, "graphs", key + ".json")
    info = None
    if os.path.exists(path):
        g = json.load(open(path))
        prefixes = set()
        for node in g["nodes"]:
            parts = node["id"].split(".")
            for i in range(1, len(parts)):
                prefixes.add(".".join(parts[:i]))
        pairs = {tuple(sorted((e["source"], e["target"]))) for e in g["edges"] if e["source"] != e["target"]}
        info = {
            "nodes": len(g["nodes"]),
            "links": len(pairs),
            "edge_types": dict(collections.Counter(e.get("type", "?") for e in g["edges"])),
            "text_chars": sum(len(n.get("text") or "") for n in g["nodes"]),
            "prefixes": prefixes,
        }
    graph_cache[key] = info
    return info


def in_graph(path, prefixes):
    parts = re.sub(r"\.pyi?$", "", path).split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    for start in range(len(parts)):
        tail = parts[start:]
        if len(tail) < 2 and len(parts) > 1:
            break
        if ".".join(tail) in prefixes:
            return True
    return False


rows = []
for t in tasks:
    repo_short = t["instance_id"].rsplit("_", 1)[0]
    stmt = t["problem_statement"] or ""
    gold = parse_patch(t["patch"] or "")
    tests = parse_patch(t["test_patch"] or "")
    g = graph_info(repo_short, t["base_commit"])
    code_files = [f for f in gold if kind(f["path"]) in ("library code", "doc example code") and not f["new"]]
    covered = [f for f in code_files if g and in_graph(f["path"], g["prefixes"])]
    snap = os.path.join(data_dir, "snapshots", t["instance_id"] + ".tgz")
    new_tests = sum(
        1 for f in tests for line in f["added_lines"] if re.match(r"\s*(async\s+)?def test_\w+", line)
    )
    rows.append({
        "id": t["instance_id"],
        "repo": repo_short,
        "created": t["created_at"][:10],
        "stmt_chars": len(stmt),
        "stmt_tokens": len(tok(stmt)["input_ids"]) if tok else None,
        "has_hints": bool((t.get("hints_text") or "").strip()),
        "hints_chars": len(t.get("hints_text") or ""),
        "has_code_block": "```" in stmt,
        "has_traceback": "Traceback (most recent call last)" in stmt,
        "names_gold_file": any(os.path.basename(f["path"]) in stmt for f in gold if not f["new"]),
        "gold_files": len(gold),
        "gold_kinds": dict(collections.Counter(kind(f["path"]) for f in gold)),
        "gold_new_files": sum(f["new"] for f in gold),
        "gold_added": sum(f["added"] for f in gold),
        "gold_removed": sum(f["removed"] for f in gold),
        "gold_hunks": sum(f["hunks"] for f in gold),
        "gold_code_files": len(code_files),
        "gold_code_files_in_graph": len(covered),
        "gold_paths": [f["path"] for f in gold][:12],
        "test_files": len(tests),
        "test_new_files": sum(f["new"] for f in tests),
        "test_added": sum(f["added"] for f in tests),
        "test_new_funcs": new_tests,
        "test_paths": [f["path"] for f in tests][:8],
        "graph": bool(g),
        "graph_nodes": g["nodes"] if g else None,
        "graph_links": g["links"] if g else None,
        "graph_text_chars": g["text_chars"] if g else None,
        "snapshot_mb": round(os.path.getsize(snap) / 1e6, 1) if os.path.exists(snap) else None,
    })

edge_types = collections.Counter()
for info in graph_cache.values():
    if info:
        edge_types.update(info["edge_types"])

out = {
    "n_tasks": len(rows),
    "n_graph_files": len([f for f in os.listdir(os.path.join(data_dir, "graphs")) if f.endswith(".json")]),
    "n_unique_commits": len({(r["repo"], t["base_commit"]) for r, t in zip(rows, tasks)}),
    "edge_types": dict(edge_types),
    "tasks": rows,
}
json.dump(out, open(out_path, "w"))


def q(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


print("tasks", len(rows), "| graph files", out["n_graph_files"], "| unique commits", out["n_unique_commits"])
print("edge types", dict(edge_types))
for name in ("stmt_tokens", "gold_files", "gold_added", "gold_removed", "test_files", "test_new_funcs", "graph_nodes"):
    vals = [r[name] for r in rows if r[name] is not None]
    print(f"{name:16s} min {min(vals)} median {statistics.median(vals)} p90 {q(vals, 0.9)} max {max(vals)}")
