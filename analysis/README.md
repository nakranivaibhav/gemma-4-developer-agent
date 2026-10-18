# analysis

Read-only scripts over the run ledger (`results/task_runs.jsonl`) and the saved run outputs (`runs/`). None of them
runs a model. Run them from the repo root with `envs/harness/bin/python -I analysis/<script> ...`; each file's
docstring has its usage.

| question | script |
|---|---|
| Is agent B better than agent A? (two passes each, the prompt-loop screen, P >= 0.90) | `compare_runs.py` |
| Is B better than A? (N samples each, from `kits/swe-agent/experiments/hist.sh`) | `ab_compare.py` |
| How often is each task solved over N samples? (never / mixed / always) | `pass_hist.py` |
| How do the Gemma 4 sizes compare, counting only what fits in Kaggle's time? | `model_report.py` |
| How did the agent behave in a run? (calls, edits, tests run, endings) | `round_stats.py` |
| What went wrong on the failed tasks of one run? | `fail_digest.py`, `fail_tools.py` |
| Which tool calls were rejected, and what did they cost? | `reject_stats.py` |
| Which tasks are worth training on? (solve rate per task and model) | `task_pool.py` |
| How fast was each GPU in a multi-GPU fleet run? | `fleet_speed.py` |
| What do the tasks look like? (statement, fix and test sizes, graph sizes) | `eda_stats.py`, then `eda_dashboard_data.py` |
| Browse runs step by step | `trace_app.py` (static site), `trace_summary.py` (one trace in the terminal) |
