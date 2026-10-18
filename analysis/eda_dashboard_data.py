"""Write the small CSV files behind the data-tour page from eda_stats.json and the data folder.

usage: python -I analysis/eda_dashboard_data.py <data_dir> <eda_stats.json> <out_dir>
"""
import csv
import json
import os
import sys

data_dir, stats_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
stats = json.load(open(stats_path))


def write(name, columns, rows):
    with open(os.path.join(out_dir, name), "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for row in rows:
            writer.writerow([row[c] for c in columns])
    print(name, len(rows), "rows")


# one row per public task
cols = ["id", "repo", "created", "stmt_tokens", "has_code_block", "names_changed_file", "files_changed",
        "new_files", "lines_changed", "lines_added", "lines_removed", "test_files", "new_test_files",
        "new_tests", "graph_symbols", "graph_links", "existing_code_files", "existing_code_files_in_graph",
        "snapshot_mb"]
rows = []
for r in stats["tasks"]:
    rows.append({
        "id": r["id"], "repo": r["repo"], "created": r["created"], "stmt_tokens": r["stmt_tokens"],
        "has_code_block": int(r["has_code_block"]), "names_changed_file": int(r["names_gold_file"]),
        "files_changed": r["gold_files"], "new_files": r["gold_new_files"],
        "lines_changed": r["gold_added"] + r["gold_removed"], "lines_added": r["gold_added"],
        "lines_removed": r["gold_removed"], "test_files": r["test_files"], "new_test_files": r["test_new_files"],
        "new_tests": r["test_new_funcs"], "graph_symbols": r["graph_nodes"], "graph_links": r["graph_links"],
        "existing_code_files": r["gold_code_files"], "existing_code_files_in_graph": r["gold_code_files_in_graph"],
        "snapshot_mb": r["snapshot_mb"],
    })
write("tasks.csv", cols, rows)


# what the download holds
def folder(path):
    full = os.path.join(data_dir, path)
    if os.path.isfile(full):
        return 1, os.path.getsize(full)
    n = size = 0
    for root, _, files in os.walk(full):
        for f in files:
            n += 1
            size += os.path.getsize(os.path.join(root, f))
    return n, size


items = [
    ("tasks.jsonl", "tasks.jsonl", "The public tasks, one per line: the issue text, the starting commit, the real fix and the scoring tests."),
    ("snapshots", "snapshots/", "One archive per task: the whole repository as it stood just before the fix. Unpacked into the sandbox."),
    ("graphs", "graphs/", "One file per starting commit: every function, method and class in the repository and which ones are linked."),
    ("embeddings", "embeddings/", "One file per starting commit: a number vector for each symbol in the graph, used by the search tool."),
    ("wheels", "wheels/", "Python packages the sandbox installs from, since it has no internet."),
    ("sample_submission", "sample_submission/", "A working example agent: prompts, settings, one helper agent and two placeholder adapters."),
    ("docker", "docker/", "The recipe for the sandbox image."),
    ("sandbox", "sandbox/", "The script that installs a repository inside the sandbox."),
    ("HARNESS_README.md", "HARNESS_README.md", "The organisers' technical guide to how a task is run and scored."),
    ("external/model", "model (separate download)", "The one allowed model, Gemma 4 31B at 4-bit."),
]
inv = []
for path, label, what in items:
    n, size = folder(path)
    inv.append({"item": label, "files": n, "size_mb": round(size / 1e6, 1), "what_it_is": what})
write("download.csv", ["item", "files", "size_mb", "what_it_is"], inv)

# one task record, field by field
tasks = [json.loads(line) for line in open(os.path.join(data_dir, "tasks.jsonl"))]
one = next(t for t in tasks if t["instance_id"] == "requests_7505")
meaning = {
    "instance_id": ("No", "The task's name: repository plus pull-request number."),
    "repo": ("Yes", "Which repository the task belongs to."),
    "base_commit": ("No", "The exact version of the code the agent starts from."),
    "problem_statement": ("Yes", "The issue text. This is the whole brief the agent gets."),
    "hints_text": ("Yes, when present", "Extra discussion from the issue. Empty on every public task."),
    "patch": ("No", "The real fix the maintainers merged. For us to study; scoring never reads it."),
    "test_patch": ("No", "The tests that decide the score. Added only after the agent has finished."),
    "created_at": ("No", "When the pull request was opened."),
}
fields = []
for key, (shown, what) in meaning.items():
    value = one[key] or ""
    fields.append({"field": key, "shown_to_agent": shown, "chars_in_example": len(value), "what_it_is": what})
write("task_record.csv", ["field", "shown_to_agent", "chars_in_example", "what_it_is"], fields)
