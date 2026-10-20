"""Write random node embeddings for official-format code graphs, so the harness turns on the graph tools for an
outside repo.

Why random: Google's embeddings cannot be rebuilt (method unpublished; no public encoder comes close, docs/embeddings.md),
and on the 123 Kaggle tasks random vectors cost about 6 passes per 984 runs against Google's, not significant (pooled
2 rounds each, E2B). A stand-in that is MORE useful than Google's vectors could teach the agent in RL to trust
search_similar_code's nearest-neighbour results more than Kaggle rewards; random vectors cannot.

Format, as data/embeddings/<repo>_<commit>.npz: one float32[256] per node id. Each vector is seeded from the node's id
and text, so a node that keeps its code keeps its vector across commits of a repo (as the official vectors do).

usage: python -I graph/random_embeddings.py <graph.json> [<graph.json> ...] --out <dir>
writes <dir>/<graph file stem>.npz (the harness looks for <repo>_<commit>.npz next to <repo>_<commit>.json's name).
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

DIM = 256


def node_vector(node_id, text):
    seed = int.from_bytes(hashlib.sha256(f"{node_id}\0{text}".encode()).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("graphs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for g in a.graphs:
        nodes = json.loads(g.read_text())["nodes"]
        vecs = {n["id"]: node_vector(n["id"], n.get("text") or "") for n in nodes}
        np.savez_compressed(a.out / f"{g.stem}.npz", **vecs)
        print(f"{g.name}: {len(vecs)} nodes -> {a.out / (g.stem + '.npz')}")


if __name__ == "__main__":
    main()
