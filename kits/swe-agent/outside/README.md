<!-- kits/swe-agent/outside/README.md contract
WHAT: how outside (SWE-rebench) tasks run in our harness, each in its own docker image: the pieces, the commands, the
decisions and the known limits. Update in place when the adapter changes; date each update.
-->
# Outside tasks (SWE-rebench), as of 2026-10-25

Our harness normally runs every task in one shared image (`swebench-sandbox-local`) from a repo snapshot plus the
wheel/setup hook. A SWE-rebench task instead comes with its own image that already holds the repo at the base commit
(`/testbed`) and a conda env named `testbed` with the dependencies installed. This folder lets `run_tasks.py` run
those tasks in their own images. A task row that carries `docker_image` switches it on. Rows without one (all our
Kaggle tasks) go down the old path, unchanged.

| file | what it does |
|---|---|
| `export_tasks.py` | parquet rows -> our tasks.jsonl format + `FAIL_TO_PASS`, `PASS_TO_PASS`, `docker_image`, `test_cmd`, `env_vars` |
| `outside.py` | the adapter: image swap, workspace setup, verification, the hooks `run_tasks.py` installs |
| `gold_check.py` | reference fix (N runs) must resolve, empty patch must not; one JSONL line per run |
| `build_graphs.py` | code graph (+ random-vector .npz) for each task's repo at its base commit |
| `agent_smoke.py` | model-free test of the agent path: container, tools, prompt, submit, verify |

## Commands

```bash
# 1. export (envs/judge has pandas)
envs/judge/bin/python -I kits/swe-agent/outside/export_tasks.py --ids data/outside/pilot6.ids --out data/outside/pilot6.jsonl
# 2. pull the images (explicit; nothing here pulls on its own). Docker Hub limits anonymous pulls.
envs/judge/bin/python -I -c "import json; [print(json.loads(l)['docker_image']) for l in open('data/outside/pilot6.jsonl')]" | xargs -n1 docker pull -q
# 3. graphs -> data/outside/graphs + data/outside/embeddings
envs/harness/bin/python -I kits/swe-agent/outside/build_graphs.py data/outside/pilot6.jsonl
# 4. gold x2 + empty
GOLD_RUNS=2 PAR=3 envs/harness/bin/python -I kits/swe-agent/outside/gold_check.py data/outside/pilot6.jsonl runs/outside/gold_pilot6
# 5. model-free agent-path smoke (+ a Kaggle-task regression)
KAGGLE_CHECK=requests_7505 envs/harness/bin/python -I kits/swe-agent/outside/agent_smoke.py data/outside/pilot6.jsonl runs/outside/smoke_pilot6
# 6. a real agent run (needs the model server up): the usual runner, pointed at the outside tasks file
TASKS_JSONL=data/outside/pilot6.jsonl RUN_ID=<run id> LEDGER=results/outside_task_runs.jsonl \
  envs/harness/bin/python -I kits/swe-agent/run/run_tasks.py agents/r09_e2b_8tool runs/<run id> data/outside/pilot6.ids
```
`run_tasks.py` takes the task ids as a file (`*.ids`, one id per line); outside ids are not in `tasks/manifest.csv`.
Leave out `LEDGER` to write to `results/task_runs.jsonl` as usual. Outside rows then carry `image` and `outside`
(F2P / P2P counts, `strict`).

## How a task runs

**Containers.** Both the agent's and the verifier's containers start from the task's image. The swap is made in
docker-py's `containers.run`, keyed by a context variable set while the task runs. Everything else stays the
harness's: network `gemma-tarpit`, 4 GB, 2 CPUs, `sleep infinity`, working dir `/workspace`.

**Workspace: move, then symlink back.** `/testbed` is moved to `/workspace` and `/testbed` becomes a symlink to
`/workspace`. The harness's file tools need `/workspace` to be a real folder: `copy_from` refuses any path whose
`readlink -f` leaves `/workspace`, so pointing a `/workspace` symlink at `/testbed` breaks `read_file`. A copy would
leave the env's editable install pointing at the old tree, so the agent's edits would not be imported. With the move,
the `/testbed` paths the image baked in still resolve. The move is a rename on overlayfs and takes milliseconds.

**Dangling installs.** Every image tried (6 of 6) was built with the repo at `/<name>` and moved to `/testbed`
later, so the env's egg-link, `.pth` or `__editable__` finder points at `/<name>`, which no longer exists. Without a
fix, `import pynmea2` fails and gold fails. The setup links each such top-level folder to `/workspace` when the rest
of the path exists there (`links` in the gold rows).

**Python env.** `<conda>/envs/testbed/bin` goes first on the container's `PATH`, with `CONDA_PREFIX` and the task's
`env_vars`, so the agent's `python` / `pytest` are the task's. The harness's own `python3 -S -c` helper (the patch
applier) needs Python >= 3.9, so in the verifier it runs with a >= 3.9 Python found in the image
(`/opt/conda/bin/python3`). Git identity variables are set too, so the harness's baseline commit always succeeds.

**Skipped for these tasks:** snapshot extraction, wheel injection, `sandbox/setup.py`, the editable install, and the
`pytest.ini` / `conftest.py` the harness writes. The image already has the env, and the repo's own pytest config must
stay as SWE-rebench ran it.

**No future commits.** The images ship `origin/main` past the base commit, which can hold the fix itself. In the agent
container only, the remotes and every ref that is not an ancestor of HEAD are deleted, the reflogs are expired and
the objects pruned (`git gc --prune=now`; `OUTSIDE_GIT_GC=0` skips the prune). Tags in the history stay. The
untracked `issue.md` the images leave in the repo and at `/` is removed.

**Graphs.** Graphs are built from the judge packet's `repo/`, or from `/testbed` copied out of the image when there
is no packet; both give the same nodes on pynmea2. They are written to `data/outside/graphs/<repo>_<commit>.json`, not
`data/graphs/`: `graph/graph_match.py` compares every file in `data/graphs/` against a Kaggle snapshot. The
per-task harness config points `graph_dir` / `embeddings_dir` at the outside folders. The `.npz` holds random vectors
(`graph/random_embeddings.py`). `search_similar_code` is off, but the harness shows the "Code Intelligence Tools" block
in the prompt only when both files exist, as it does on every Kaggle task.

## Verification, and how it differs from the Kaggle scorer

`verify_outside` works in a fresh container of the task's image:
1. Strip protected files from the agent's patch with the harness's own `strip_protected_files_from_patch` (tests,
   `conftest.py`, `pytest.ini`, `pyproject.toml`, `setup.cfg`, `tox.ini`, `.pth`, ...), as the Kaggle scorer does.
   Apply it with the harness's applier.
2. Reset the test_patch's files to the base commit, then apply the test_patch.
3. Run the task's own `test_cmd` on the test_patch's files (SWE-bench's file rule; rows whose `test_cmd` already names
   the files run as written).
4. Parse the `-rA` summary (SWE-bench's `parse_log_pytest_v2`).
5. `resolved` = every `FAIL_TO_PASS` and every `PASS_TO_PASS` test is PASSED or XFAIL. A required test that is
   missing or skipped counts as not passed, which is stricter than SWE-bench (SWE-bench ignores skipped ones).

Differences from the Kaggle criterion (exit 0, zero failures or errors in the whole run, every required test passed):
- **The whole run need not be clean.** The Kaggle rule is reported as `strict` (resolved, exit 0, no FAILED or
  ERROR line anywhere). 878 of the 6,542 filtered tasks have `FAIL_TO_FAIL` tests, so they can never be `strict`,
  even with the gold fix.
- **Test ids match exactly**, from the `-rA` summary. Kaggle uses JUnit XML with fuzzy name matching.
- **The command is the task's**, with the repo's own pytest config. Kaggle uses `python3 -s -m pytest ... -p no:anyio`
  under a hermetic `pytest.ini`.
- **The test timeout is 1800 s** (`OUTSIDE_VERIFY_TIMEOUT`, SWE-bench's value). Kaggle uses 300 s.
- **Gold patches are stripped too.** A reference fix that edits `pyproject.toml` or `setup.cfg` loses those hunks,
  as an agent's patch would. `patch_stripped` in the gold rows flags it.

## Pilot, 2026-10-25 (`runs/outside/gold_pilot6`, `runs/outside/smoke_pilot6`)

| task | image on disk (unique) | gold x2 | empty | graph nodes | smoke |
|---|---|---|---|---|---|
| jd__tenacity-479 | 4.63 GB (2.47) | 2/2 | fails | 438 | ok |
| Knio__pynmea2-124 | 5.12 GB (4.45) | 2/2 | fails | 263 | ok |
| kjd__idna-83 | 5.01 GB (4.35) | 2/2 | fails | 165 | ok |
| mherrmann__gitignore_parser-17 | 5.13 GB (4.47) | 2/2 | fails | 16 | ok |
| allo-media__text2num-77 | 5.12 GB (4.45) | 2/2 | fails | 186 | ok |
| agronholm__exceptiongroup-69 | 4.65 GB (2.49) | 2/2 | **passes**: unsound | 161 | import not from repo |

exceptiongroup-69 is broken in its image: the env also has a released `exceptiongroup` 1.2.0 (a pytest dependency,
which already contains the fix) that shadows the repo's `src/`. The empty patch therefore resolves. The gold/empty
check exists to drop such tasks, so run it before using any task.

## At scale

- **Pulls.** Each image is about 1.4 GB compressed and took 45 to 135 s to pull here. Anonymous Docker Hub pulls are
  rate-limited; `docker login` raises the limit. Pull once, then keep the image.
- **Disk.** Each image is 4.6 to 5.1 GB on disk, 2.5 to 4.5 GB of it unique (the shared base layers are 0.67 or
  2.2 GB). The 760 GB free holds about 170 images: remove an image once its task's runs are done.
- **Bad tasks.** Expect some unsound tasks (shadowed packages, as above). Gold-check every task before using it.
- **Python < 3.9.** 1,867 of the 6,542 filtered tasks use a Python < 3.9 env. Only 3.9 images were tried. The helper
  Python is probed per image, and an image without one fails loudly at the first container.
- **Test time.** The pilot's tests take 1 to 4 s. Larger repos are slower, and the limit is 1800 s. The
  leaderboard rows ask for 16 GB on 10 of 860 tasks; we keep the harness's 4 GB.
- **Agent setup.** About 0.5 s per container here, plus a one-time image probe of about 1 s per image per process.
  `git gc` grows with the repo's history.

## Remote Docker host (revideo), 2026-10-26
The screen can run its task containers on another machine's Docker while the harness and the model stay here:
`REMOTE=revideo` on `pass_pipeline.sh` / `screen_loop.sh` opens an SSH unix-socket tunnel
(`/home/vaibhav/data/revideo-docker.sock` -> revideo's `/var/run/docker.sock`) and sets `DOCKER_HOST` to it, so
pulls, checks and agent sandboxes all happen on revideo. Nothing is installed or copied there.

What lives on revideo, and for how long:
- task images: about two batches at a time, deleted at each batch's teardown; the loop pauses below `MIN_FREE_GB`
  free on revideo (150).
- task containers: removed by the harness when each task ends.
- lasting: the offline network `gemma-tarpit` and the pinned base image `gemma/base-anchor:pinned`.

**Clean it regularly** (revideo is a shared VM; keep it uncluttered). After a run, or whenever the screen is stopped:
```bash
bash kits/swe-agent/outside/remote_cleanup.sh revideo              # everything of ours: images, base pin, network
bash kits/swe-agent/outside/remote_cleanup.sh revideo --keep-base  # between runs: keep base pin + network
```
It removes only our objects (`swerebench/*`, `gemma/base-anchor:pinned`, `gemma-tarpit`, dangling layers), refuses
while one of our task containers is running (`--force` overrides), and prints revideo's free disk before and after.
