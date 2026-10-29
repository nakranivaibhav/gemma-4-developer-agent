"""Build the code graph (and the embedding file the harness gates on) for outside tasks.

usage: envs/harness/bin/python -I kits/swe-agent/outside/build_graphs.py <tasks.jsonl> [instance_id ...] [--force]
For each task: the repo at its base commit (the judge packet's repo/ folder; if there is none, /testbed copied out of
the task's pulled image) -> graph/code_graph.py -> <OUTSIDE_GRAPH_DIR>/<repo short>_<base commit>.json, then
graph/random_embeddings.py -> <OUTSIDE_EMB_DIR>/<same stem>.npz. Defaults: data/outside/graphs, data/outside/embeddings
(outside.py points the per-task harness config there; data/graphs stays the 127 official files that
graph/graph_match.py compares against).

Why an .npz at all: search_similar_code is off, but the harness shows the "Code Intelligence Tools" block in the task
prompt only when BOTH files exist (agent_runner.build_agent_prompt), as it does on every Kaggle task. Random vectors
keep the outside prompt identical in shape; no tool we use reads them.
Each graph is then loaded with the harness's own loader and one get_neighbor query is run, so a file that the tools
cannot use fails here. Prints one line per task: nodes, edges, source.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import outside  # noqa: E402

ROOT = outside.ROOT
PACKETS = Path(os.environ.get("OUTSIDE_PACKETS", "/home/vaibhav/data/gemma4/judge/packets"))  # repo/ at the base commit
PY = sys.executable


def repo_source(row, tmp: Path) -> tuple[Path, str]:
    pk = PACKETS / row["instance_id"] / "repo"
    if pk.is_dir():
        return pk, "packet"
    name = f"outside_graph_{row['instance_id'].lower()}"[:60]
    subprocess.run(["docker", "create", "--name", name, row["docker_image"], "true"], check=True, capture_output=True)
    try:
        subprocess.run(["docker", "cp", f"{name}:/testbed", str(tmp / "testbed")], check=True, capture_output=True)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    shutil.rmtree(tmp / "testbed" / ".git", ignore_errors=True)
    return tmp / "testbed", "image"


def check(path: Path, row) -> tuple[int, int, str]:
    from swegemma.graph import graph_utils

    g = graph_utils.load_graph_from_json(str(path))
    # one real neighbour query through the harness's own lookup, as get_code_neighbors makes it
    from swegemma import graph as sg

    node = next((n for n in g.nodes if g.degree(n) > 0), next(iter(g.nodes), None))
    probe = ""
    if node is not None:
        nb = sg.get_neighbor(node=f"{row['repo']};{node}", max_neighbors=5, sstable_path=str(outside.GRAPH_DIR),
                             embeddings_sstable_path=str(outside.EMB_DIR), use_local_storage=True, repo_name=row["repo"],
                             graph_dir=str(outside.GRAPH_DIR), embeddings_dir=str(outside.EMB_DIR),
                             base_commit=row["base_commit"])
        nb = nb if isinstance(nb, list) else nb.get(f"{row['repo']};{node}", [])
        probe = f"{node} -> {len(nb)} neighbours"
    return g.number_of_nodes(), g.number_of_edges(), probe


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    force = "--force" in sys.argv
    if not args:
        sys.exit(__doc__)
    specs = outside.load_specs(args[0])
    ids = args[1:] or list(specs)
    outside.GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    outside.EMB_DIR.mkdir(parents=True, exist_ok=True)
    for tid in ids:
        row = specs[tid]
        stem = outside.graph_stem(row)
        gpath, epath = outside.GRAPH_DIR / f"{stem}.json", outside.EMB_DIR / f"{stem}.npz"
        src = "cached"
        if force or not gpath.exists():
            with tempfile.TemporaryDirectory(prefix="outside_graph_") as tmp:
                repo, src = repo_source(row, Path(tmp))
                r = subprocess.run([PY, "-I", str(ROOT / "graph" / "code_graph.py"), str(repo), stem,
                                    "--out", str(outside.GRAPH_DIR)], capture_output=True, text=True)
                if r.returncode != 0:
                    print(f"GRAPH_FAIL {tid}: {r.stderr[-400:]}", flush=True)
                    continue
        if force or not epath.exists():
            r = subprocess.run([PY, "-I", str(ROOT / "graph" / "random_embeddings.py"), str(gpath), "--out", str(outside.EMB_DIR)],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"EMB_FAIL {tid}: {r.stderr[-400:]}", flush=True)
                continue
        nodes, edges, probe = check(gpath, row)
        print(json.dumps({"id": tid, "graph": str(gpath.relative_to(ROOT)) if gpath.is_relative_to(ROOT) else str(gpath),
                          "nodes": nodes, "edges": edges, "source": src, "probe": probe}), flush=True)


if __name__ == "__main__":
    main()
