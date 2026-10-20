<!-- outside_data.md contract
WHAT: the outside SWE data we can train on (task sets and solved-run sets), what it takes to run those tasks in our
harness, and the plan for the first outside run. Full source table with links: docs/training_research.md section 1-2.
Update in place as steps land; date each update. Mark anything not yet checked as "unverified".
-->
# Outside training data, as of 2026-10-20

Our own 77 training tasks give E2B only 8–13 mixed tasks (`pass_histogram_e2b.md`), so the RL pool has to come from
outside task sets. None of them has been downloaded or run yet. Research done 2026-10-15; sources and links in
`training_research.md`.

## Task sets (can be run, have tests that check the fix)

| dataset | tasks | repos | how the environment ships | notes |
|---|---|---|---|---|
| SWE-Gym | 2,438 (Lite: 230) | 11: pandas 737, MONAI 374, moto 343, mypy 257, dvc 225, dask 145, modin 107, pydantic 83, conan 75, hydra 66, bokeh 26 | one Docker image per task (`xingyaoww/sweb.eval.x86_64.*`) | real issues; big repos with heavy test suites |
| SWE-smith-py | 50,908 | 131 | one Docker image per repo, shared by all its tasks | synthetic bugs (injected by an LLM), not real issues |
| SWE-rebench | 21,336 + 6,542 filtered | 3,400+ | about 7,500 prebuilt images on Docker Hub (`swerebench`) | real issues 2014–2025, many small libraries |
| SWE-rebench-V2 | about 32k | – | – | not inspected |
| Scale-SWE | 20,181 | 5,200 | one image per task | – |
| R2E-Gym | 4,578 – 8,101 | 13 | one image per repo commit, 300–500 MB each | no FAIL_TO_PASS column; tests mapped by expected status |
| OpenSWE | 45k, about 9k after quality filter | 12.8k | Dockerfiles only | – |

## Solved-run sets from other models (for SFT)

| dataset | size | made by |
|---|---|---|
| nebius SWE-rebench OpenHands | 67k runs, 32k solved | Qwen3-Coder-480B |
| SWE-smith trajectories | 76k runs | Claude 3.7 Sonnet |
| Scale-SWE-Distilled | 71k runs | DeepSeek v3.2 |
| nvidia SWE-Hero | 34k runs | – |
| SWE-Gym OpenHands | about 6k runs | – |
| R2E-Gym SFT | 3.2k runs | Claude 3.5 Sonnet |

All of these use other tools and prompts than ours, so they need converting to our tool calls and cutting to 32k
tokens. Runs made by our own agent on outside tasks need neither.

## Overlap with our holdout

Our holdout is fastapi, rich and requests tasks. Drop any outside task whose repo matches
`fastapi|rich|requests|httpx` (forks included), or the local score stops being a fair test. SWE-bench train has 28
fastapi tasks and SWE-bench test has 44 requests tasks. SWE-Gym and SWE-smith are clean by repo name. SWE-rebench,
V2, Scale-SWE and R2E-Gym are unverified: filter them locally on the rows we pull.

## What "adapter" means here

Our harness runs every task in **one shared image** (`swebench-sandbox-local`, about 470 MB). A task is two things:

1. a row in `data/tasks.jsonl`: `instance_id, repo, base_commit, patch, test_patch, problem_statement, hints_text,
   created_at`;
2. a git snapshot of the repo at `base_commit`: `data/snapshots/<id>.tgz`, about 160 MB each.

The harness unpacks the snapshot into `/workspace`, installs the repo offline from cached wheels (the sandbox has no
network), runs the agent, then checks the patch in a fresh container with pytest.

SWE-Gym and the others ship each task as **its own Docker image** with the repo and all its dependencies already
installed (for example pandas built with its C extensions). Our shared image does not have those dependencies, and
it cannot download them because the sandbox is offline. So the "adapter" is the glue between the two formats:

- **Task rows:** convert each SWE-Gym row into our `tasks.jsonl` format. Easy: SWE-Gym uses the same SWE-bench
  fields, plus FAIL_TO_PASS and PASS_TO_PASS lists.
- **Environment:** make the harness start its agent container and its check container from the task's own image
  instead of `swebench-sandbox-local`, with the repo at `/workspace`. This is the real work. SWE-bench-style images keep
  the repo at `/testbed` inside a conda env (SWE-bench convention, unverified for these images), and our harness
  setup steps assume its own image. How much has to change
  is unverified; the smoke run finds out.

The 5-task smoke run: convert 5 SWE-Gym tasks, run their real fix (the gold patch) through our checker and see it
pass, run an empty patch and see it fail, then run E2B on them once. That proves tasks, environment and checking all
work before a large run.

## Code graphs and embeddings for outside tasks

Every public task comes with a code graph (`data/graphs/<repo>_<commit>.json`: AST symbols plus `calls` edges) and
node embeddings (`data/embeddings/<repo>_<commit>.npz`: one 256-number vector per symbol). When both exist, the
harness adds a "Code Intelligence Tools" section to the task prompt and turns on `search_similar_code`,
`get_code_neighbors` and `get_code_subgraph`. Outside tasks have neither, so without them the prompt and the tool set
differ from what the model sees on Kaggle.

How much our agents use these tools (from the run traces, 2026-10-17):

| runs | share of tool calls that are graph tools | runs that used one | solved runs that used one |
|---|---|---|---|
| E2B, temperature 0.7 (984 runs) | 7.9% (mostly `search_similar_code`, 848 calls) | 378 of 984 | 15 of 72 |
| E4B (243 runs) | 1.7% | 18 | 1 of 35 |
| 31B + adapter (247 runs) | 0.3% | 17 | 8 of 121 |
| Qwen 27B (122 runs) | 0.4% | 12 | 3 of 60 |

- **Graphs: done 2026-10-17.** `graph/code_graph.py` rebuilds all 127 official graphs exactly (same nodes, texts and
  edges; only the file order differs, which is random in the official files too). Details: `code_graph_builder.md`.
- **Embeddings: closed 2026-10-20, random vectors.** The originals cannot be rebuilt, and on the Kaggle tasks random
  vectors score close to them (69 vs 75 passes per 984, not significant). `graph/random_embeddings.py` writes them for
  any graph; the runner takes the folder via `EMBEDDINGS_DIR`. Details: `embeddings.md`.
- **Cost:** CPU only. SWE-Gym needs one graph per task commit (2,438); SWE-smith-py one per repo (about 131).

### What the forum says (checked 2026-10-17)

Full text of 215 of the 216 topics read (2026-10-17); 24 of them mention the graphs or graph tools. The ones that matter:

| topic | who | finding |
|---|---|---|
| [#748801](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748801) (2026-10-13) | participant | Asks how the embeddings were made so others can build them for new repos. **No host answer** as of 2026-10-17. |
| [#745998](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745998) (2026-10-05) | host side | Private (hidden) tasks have code graphs too. Async functions are missing from all graphs and will not be added; no private task touches an async function. |
| [#742911](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/742911) (2026-09-24) | participant + host | Graphs hold no async functions (0 of 305), only one edge type (`calls`), no module nodes. Empty embedding files in the download were fixed 2026-09-26. |
| [#745731](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745731) (2026-10-04) | participant | Audit of all 127 graphs: every edge stored both ways, class-to-member edges also typed `calls`, no `self.m()` calls. Embeddings track graph distance more than code text (Spearman −0.28 to −0.34 with graph distance, 0.06 to 0.18 with TF-IDF), so they look graph-based. TF-IDF over def names finds a changed def in the top 10 on 52% of tasks; the embeddings on 9%. Public notebook with a graph builder (Apache 2.0), but it builds a fuller graph than the official one. |
| [#744040](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744040) (2026-09-28) | participants | Embeddings look near-collapsed: random pairs have mean cosine 0.78 to 0.90, 8–19% of pairs ≥ 0.99. |
| [#748323](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/748323) (2026-10-11) | participant | The graph tools returned something on 82, 11, 270, 270 of the same 314 calls across four harness builds. Since the 8 Oct build (swegemma 0.2.10, which we run) they answer most calls. |
| [#746250](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746250) (2026-10-06) | participant | Adding the graph tools gave 50/129 against 55/129 without them (older build, one run each). |
| [#747696](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/747696) (2026-10-09) | participant | Kaggle public LB, base Gemma model (size not stated in the excerpt): removing the analyzer's 3 graph tools dropped the score from 0.13 to 0.08. Same direction as our A/B. |
| [#745220](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/745220) (2026-10-02) | participant | `search_similar_code` matches the exact symbol name, not keywords as the prompt says (`RequestEncodingMixin.path_url` finds results, `path_url` finds none). `get_code_neighbors` also needs the exact full name. |
| [#744577](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/744577) (2026-09-30) | participant + host | `search_similar_code` returned full node source with no size limit (fastapi.applications.FastAPI is about 130k characters), which overflowed the 32k context. The host said it would cut it to `max_stdout_chars` like other tools (our 0.2.10 has the cap: `swegemma/tools/graph.py` applies `max_stdout_chars` in `search_similar_code`). |
| [#746235](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/746235) (2026-10-06) | participant | Leaving the 3 graph tools out of `agent.yaml` is allowed (answered by another participant, not a host). |
| [#743751](https://www.kaggle.com/competitions/gemma-4-developer-agent/discussion/743751) (2026-09-26) | participant | Asks how the graph and embeddings were built and what similarity means. No answers. |

### What this means for outside tasks

- **Graphs:** buildable to match. The format and its gaps are documented above, so a builder must copy the gaps too
  (no async, `calls` only, both directions, no `self` calls) to look like what Kaggle serves.
- **Embeddings:** the method is unknown and the hosts have not answered. Measured 2026-10-18 to 2026-10-20 (correcting
  the earlier guess above): they are a graph network over the raw code text of a node and its neighbours, and no
  public encoder comes close. Use random vectors (`embeddings.md`).
- **Option to skip graphs entirely:** the harness adds the "Code Intelligence Tools" prompt section only for graph
  tools the agent lists in its `agent.yaml` (`_has_tool` in `refs/harness_src/swegemma/harness/agent_runner.py`). If
  our agent drops the 3 graph tools, the prompt is the same with or without graph files, and outside tasks need no
  graphs. **Tested 2026-10-17 and it costs passes (next section), so we keep the graph tools.**

### Graph-tools A/B, 2026-10-17

E2B, temperature 0.7, all 123 tasks × 8 tries in each arm, same server, load and settings. Arm B is
`agents/e2b_t07_nograph/`: the 3 graph tools removed from `agent.yaml` and from the `code_analyzer` sub-agent
(its prompt now names `read_file` only). The captured requests confirm arm B had no "Code Intelligence Tools" section.

| | with graph tools (`hist_e2b_t07`) | without (`hist_e2b_t07_nograph`) |
|---|---|---|
| passes (of 984 runs) | 72 | 54 |
| solved per try | 12, 8, 13, 11, 6, 8, 8, 6 | 6, 6, 5, 7, 6, 8, 10, 6 |
| tasks solved at least once | 24 | 18 |
| mixed tasks (training / holdout) | 21 (13 / 8) | 16 (10 / 6) |
| always solved | 3 | 2 |
| passes, training / holdout | 57 / 15 | 45 / 9 |
| harness errors | 12 | 10 |
| tool calls | 10,615 (971 graph) | 9,496 |

- **Per task:** without graph tools, 4 tasks did better, 14 worse, 105 the same. Biggest drops: fastapi_14301 7→2,
  requests_6629 3→1, rich_3905 3→1, requests_7309 6→4. Biggest gain: requests_6589 2→5.
- **Paired bootstrap over tasks (10,000 resamples):** mean per-task pass rate changes by −0.018 (90% interval −0.033
  to −0.005) without the tools; P(without > with) = 0.007. The holdout moves the same way (15 → 9 passes, no task
  better).
- **Cause unverified:** only 15 of the 72 solved runs in arm A called a graph tool, so the extra calls alone do not
  explain 18 more passes. The tool list and the sub-agent prompt also changed. A single run per arm, so the size of
  the gap is less certain than its sign.
- **Script:** `envs/harness/bin/python -I analysis/ab_compare.py hist_e2b_t07 hist_e2b_t07_nograph [--split train]`.

## Constraints for a large outside run

- **Disk:** 122 GB free of 915 GB (2026-10-17). SWE-bench-style per-task images for repos like pandas are several GB
  each (unverified for SWE-Gym; measure in the smoke run). 2,438 images will not fit at once, so a large run must
  pull, run and delete images in batches.
- **Docker Hub pull limits:** thousands of image pulls will hit Docker Hub rate limits; a logged-in account raises
  them. Exact current limits unverified.
- **CPU:** our 5090 machine was already at 84% CPU with 128 small tasks. pandas, mypy and dask test suites are much
  heavier than fastapi or rich, so fewer tasks will fit at once.
- **Time:** E2B did 984 runs in 68 min on our tasks (about 870 per hour). SWE-Gym × 8 tries = 19,504 runs, at
  least 22 hours at that rate, probably more with heavier tests.
- **Difficulty:** E2B solves 7% of our tasks. SWE-Gym repos are larger and harder, so most tasks will likely land in
  the never-passed band for E2B (unverified).

## Plan (step 1 done, rest proposed)

1. **Decide on graph tools: done 2026-10-17, keep them** (72 vs 54 passes, above). So outside tasks need graphs
   and embeddings. Graph builder done the same day (127/127 exact, `code_graph_builder.md`); embeddings settled
   2026-10-20 as random vectors (`graph/random_embeddings.py`, `embeddings.md`). The
   adapter checks of step 2 (gold patch passes, empty patch fails) do not need either; the E2B try does.
2. **Smoke run, 5 tasks** from SWE-Gym (adapter, graphs, gold patch passes, empty patch fails, one E2B try).
3. **Same smoke run on SWE-smith-py.** One image per repo means about 131 images for 51k tasks, so disk and pull
   limits stop mattering, and synthetic bugs are often easier, which suits a small model.
4. **Screen about 200 tasks from each set** with E2B, 8 tries each (about 2 hours each), and count mixed tasks.
5. **Pick the pool** with the most mixed tasks per GPU hour and run it at full size.
6. **SWE-rebench filtered** (6.5k, real issues, many small libraries close to our task shape) is the next set to try
   if the first two give few mixed tasks.
