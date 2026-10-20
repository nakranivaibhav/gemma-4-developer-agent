<!-- embeddings.md contract
WHAT: what the official node embeddings (data/embeddings/*.npz) are, how hard we tried to rebuild them, what they are
worth to the agent (paired E2B runs), and the decision for outside repos. Numbers come from results/task_runs.jsonl
(run ids below) and the journal PROBE lines of 2026-10-18 to 2026-10-20. Update in place if new information arrives
(a host answer on forum #748801, or an RL eval that points at the vectors); date each update.
-->
# Node embeddings, as of 2026-10-20

**Closed. Decision: keep all three graph tools; for outside repos, fill the embedding files with random vectors
(`graph/random_embeddings.py`). Do not try to rebuild Google's vectors.** Reopen if the hosts publish the method, or if
an RL checkpoint behaves differently with official and random vectors on the Kaggle tasks.

## What the official vectors are

From the 127 official files (367,985 vectors, 10,362 distinct node texts):

- **A graph network over raw code text.** A node's vector depends on its own text and its neighbours' text. Isolated
  nodes with the same text always get identical vectors (44,350 of 44,350 pairs, across modules); the same text with
  different neighbours gets different vectors. Nothing repo-wide: unchanged nodes keep identical vectors across commits.
- **Raw text in, long context.** One added blank line moves a vector. Texts sharing their first 51,031 characters get
  equal vectors, texts that differ at 13,792 do not: a limit of about 2.5k to 9k Gemma tokens.
- **Nearly collapsed.** About 5 effective dimensions; random pairs have cosine 0.77 on average. Near-identical code
  can land far apart.
- **Source unknown.** The harness paths point to an internal Google pipeline (`func_node_embeddings` in
  `swegemma/graph/embedding_utils.py`). No host answer on forum #748801 (a participant reports Gemma embedding models
  stuck near a 50% match).

## Rebuild attempt

- **No public encoder matches.** About 30 models (EmbeddingGemma, CodeGemma, Gemma 2/3, Qwen3-Embedding, Qwen coder,
  UniXcoder, CodeBERT, CodeT5+ 256-d, E5, BGE, nomic, CodeRankEmbed, MiniLM) explain R2 0.28 to 0.42 of the isolated
  nodes' vectors under a linear map, the same as character TF-IDF (0.40); an MLP reaches 0.55 (EmbeddingGemma).
- **Best learned imitation:** EmbeddingGemma features into a 4-layer GIN (PyTorch Geometric) trained with a ranking
  loss to copy the official similarity order. On a repo it never saw, its similarity order matches Google's at rank
  correlation 0.36 (3 seeds; leave-one-repo-out over fastapi, rich, requests+httpx; untuned 0.12). Replaying the
  agents' real searches, it returns 18% of the same results as the official vectors (random vectors: 0.1%).

## What the vectors are worth to the agent

`search_similar_code(query, k)` has two routes, chosen by the harness from the query (`get_similar_nodes`): a query
that names a graph node goes to nearest neighbours by vector; anything else (words, code, unknown names) goes to a
keyword score over node names and code (+3 per word in the name, +1 in the code, +0.5 if not a test), which uses no
vectors and which our graph builder reproduces exactly. In replayed agent searches 59% took the keyword route.

E2B agent at temperature 0.7, 123 usable tasks x 8 tries per round, same server settings throughout; passes per 984
runs, pooled over the rounds shown:

| arm | agent folder, run ids | rounds | passes per 984 |
|---|---|---|---|
| all 3 graph tools, official vectors | `e2b_t07`: `hist_e2b_t07`, `_r2` | 2 | **75** |
| all 3 graph tools, random vectors | `e2b_t07` + `EMBEDDINGS_DIR`: `hist_e2b_t07_randvec`, `_r2` | 2 | **69** |
| without `search_similar_code` | `e2b_t07_nosimilar`: `hist_e2b_t07_nosimilar`, `_r2` | 2 | **62** |
| no graph tools | `e2b_t07_nograph`: `hist_e2b_t07_nograph` | 1 | **54** |

- Random vs official: mean per-task change -0.006 (90% interval -0.016 to +0.004), not significant.
- Random vs without the tool: +7 per 984, borderline (P = 0.07). Without vs with the tool: -13, significant (P = 0.02).
- The same agent run twice scored 72 and 78: one round carries about 6 passes of noise.
- A prompt rule asking for plain-word queries only (`e2b_t07_kwsearch`, to force the keyword route) did not change
  how E2B queries (64% single names before and after) and tripled its searches; it cannot remove the vector route.

## Decision for outside repos (RL)

- The tool is worth keeping (+13 per 984); at most about 6 of that may come from Google's vectors.
- A trained stand-in risks being *more* useful than Google's vectors on some searches, which RL would turn into
  over-trust that Kaggle does not reward. Random vectors cannot cause that, and cost little on Kaggle.
- So: build the graph with `graph/code_graph.py`, write random vectors with
  `python -I graph/random_embeddings.py <graph.json> --out <dir>`, and point the harness at that folder
  (`EMBEDDINGS_DIR=<dir>` for `kits/swe-agent/run/run_tasks.py`). Gate RL checkpoints on the Kaggle tasks with the
  official vectors, as every eval already does.
