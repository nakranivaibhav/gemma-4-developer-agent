<!-- code_graph_builder.md contract
WHAT: how our code-graph builder (graph/code_graph.py) reproduces the official graphs, how well it matches, and what is
still open (embeddings). Numbers come from graph/graph_match.py -> results/graph_match.csv. Update in place when the
builder or the check changes; date each update.
-->
# Code graph builder, as of 2026-10-17

`graph/code_graph.py` builds a code graph for any repo in the same format as the official `data/graphs/*.json`. It
rebuilds all 127 official graphs with the **same nodes, the same node texts and the same edges**. The only thing
that differs is the order of nodes and edges in the file, which is random in the official files too.

## Result

`envs/harness/bin/python -I graph/graph_match.py <scratch_dir>` rebuilds every graph from its task snapshot and
compares it with the official file (48 s on 10 cores, CPU only).

| | official | ours | same |
|---|---|---|---|
| graphs | 127 | 127 | 127 exact |
| node ids | 367,985 | 367,985 | all |
| node texts | 367,985 | 367,985 | all |
| edge pairs | 226,247 | 226,247 | all, 0 extra, 0 missing |
| node order (given the official file order) | | | 127 of 127 |
| byte-identical file | | | 0 of 127 (edge order differs, see below) |

Per graph: `results/graph_match.csv`. By repo: fastapi 67/67, rich 47/47, requests 12/12, httpx 1/1.

Loaded with the harness's own loader (`swegemma.graph.graph_utils.load_graph_from_json`), our requests graph gives
the same `resolve_node_name` and `get_neighbor` answers as the official one on the symbols tried (Session.request,
to_native_string, HTTPAdapter).

## How the official graphs are built (what the builder copies)

Full rules with examples are in the docstring of `graph/code_graph.py`. In short:

- **Nodes:** one per `def` and `class` in every `.py` file, tests and docs included. `async def` is never a node.
  Ids are dotted module paths: the path from the nearest "source root" (the repo root, or the folder that holds a
  top-level package, such as `src/` or fastapi's `docs_src/`).
- **Node text:** functions keep their exact source. Classes are a stub: header, docstring (or the word `None`), then
  each method's signature with body `...`.
- **Edges:** one type, `calls`, stored in both directions. A class is linked to its methods. A function is linked
  to what it calls, to exception classes it catches (`except X`), and a class to its base classes. Names are
  resolved through the file's imports, read top to bottom.

Quirks of the official builder that we copy on purpose, so the tools behave the same on outside repos:

- **Class prefix lost:** a class defined inside a method resets the "current class", so every later method in that
  class loses its class prefix and its link to the class (`tests.test_requests.test_x`, not `...TestRequests.test_x`).
- **Forward references:** a function called before it is defined in the same file is not linked.
- **Wrong package level:** relative imports in an `__init__.py` resolve one package level too high, so they do not
  link.
- **Fuzzy fallback:** if `pkg.Name` is not a node, the call links to the single node named `Name` under that
  prefix (`requests.Request` → `requests.models.Request`). With several candidates (`requests.get`) there is no link.
- **No self-calls:** `self.method()` is never linked.

## Order is random in the official files

- **File order:** the order in which files appear changes from graph to graph with no pattern (hashed directory
  order on the builder's machine). Within a file, definitions are in source order.
- **Edge order:** fits no single file order (the best order breaks 319 of 3,714 ordering constraints on
  requests_0b401c76). It most likely comes from iterating a Python set.
- **Why it matters a little:** `get_code_neighbors` lists at most 50 neighbours, in edge order. For a node with more
  than 50, which 50 the model sees is already arbitrary on Kaggle. Ours uses a seeded shuffle per repo.

## Use on a new repo

```bash
envs/harness/bin/python -I graph/code_graph.py <repo_checkout> <repo>_<commit> --out data/graphs
```

The harness then needs an embedding file of the same name in `data/embeddings/` (over 100 bytes) before it shows
the graph tools to the agent.

## Open: embeddings

The graph is done; the embeddings are not. The official vectors come from a Google-internal model (see
`outside_data.md`), one 256-number vector per node id. Next step: build stand-in embeddings and check them against
the 127 official files on the tasks we have, using the measures the forum used (rank agreement with the official
vectors' nearest neighbours, and top-10 hit rate for the changed function).
