# swe-agent kit

Everything needed to run a coding agent on SWE-style tasks on one GPU machine, the way the
[Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent) scorer does:
the agent works inside an offline Docker sandbox, then the hidden tests run on its patch.

## What you need

- Linux, an NVIDIA GPU (24 GB+ for E2B/E4B/12B; the 31B needs about 20 GiB of weights plus cache, it ran on a 32 GB RTX 5090)
- Docker, `uv`, `make`
- A Kaggle account with the competition rules accepted, and `KAGGLE_USERNAME` / `KAGGLE_KEY` set

## Commands (run from the repo root)

```bash
make setup                                  # data, model (SIZE=e2b), Python env, sandbox image: first time only
make serve SIZE=e2b                         # start vLLM in the background (log: runs/logs/server.log)
make run AGENT=agents/e2b_t07 TASKS=smoke6  # run an agent on tasks
make gold-check                             # is every task solvable at all? updates tasks/manifest.csv
make down                                   # stop the server and leftover sandboxes
```

`TASKS` is a selector from `tasks/manifest.csv`: a split (`tune`, `holdout`), `usable` (everything not excluded),
a named list (`smoke6`, `screen_01`, ...), or task ids. Each run writes `runs/<run>/` (patches, test output, traces,
the exact settings and a copy of the agent folder) and appends one row per task to `results/task_runs.jsonl`.

Useful knobs: `SIZE=e2b|e4b|12b|31b`, `NGRAM=0` (plain decoding instead of n-gram speculation), `CONCURRENCY=8`,
`RUN=<name>`, `NOTE="..."`, `GPU_UTIL=0.84` (if the desktop holds part of the GPU).

## What is in here

| folder | what it does |
|---|---|
| `sandbox/` | the Docker image each task runs in. `Dockerfile` adds 31 test wheels (`wheels_extra.txt`, fetched by `fetch_wheels.py`) on top of the competition image; `repair.py` installs, per repo, what that repo's own tests need. `build_hook.py` joins the organiser's `setup.py` with `repair.py` (the harness copies one file into each container). With it the real fix passes 125 of 129 public tasks, against 43 in the image as shipped. |
| `serve/` | `serve_model.py` starts vLLM with the starter notebook's settings; `serve_fast.sh` adds the measured-fastest options. |
| `run/` | `run_tasks.py` (the runner) and its helpers: `manifest.py` (task selectors), `pin.py` (settings pinned per run), `trace_stats.py`, `tag_tasks.py`, `kaggle_time.py` / `kaggle_clock.py` (Kaggle-time model and optional per-task cut), `fleet.py` (several GPUs on one queue), `log_proxy.py` (captures every request for training data). Shell drivers: `run_round.sh` (two passes), `hist_runs.sh` (N samples side by side), `holdout_step.sh`, `resume_pass.sh`. |
| `check/` | `gold_verify.py` / `gold_run.sh` run the harness's own verifier with the real fix or an empty patch; `env_census.py` and `tool_samples.py` are earlier environment probes. |
| `bench/` | server speed tests (`bench_multi.py`, `bench_matrix.sh`, `real_matrix.sh`), model-size pipelines (`variant_pipeline.sh`), decode profiling (`kprof/`), and the Kaggle speed-test notebook builder. |
| `remote/` | the same on a rented GPU pod: `pod_setup.sh`, `pod_serve.sh` (set `LOCAL_REPO`), `run_remote.sh` through an SSH tunnel. |
| `requirements.txt` | the exact harness environment (250 packages); 41 come from the competition wheelhouse, including its vLLM 0.19.1. |

## Notes

- The model runs on the host, not in Docker: GPU access from containers needs the NVIDIA container toolkit, which
  this setup does not assume. Only the task sandboxes are containers.
- Long runs (`run_round.sh`, `hist_runs.sh`) are meant to be started detached under `tools/run_with_watchdog.py`.
- The harness is the competition's `swegemma` 0.2.10. Two places in `run_tasks.py` wrap its internals (sandbox CPU
  shares / fleet labels, and the optional Kaggle clock); both are commented where they happen.
