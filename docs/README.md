<!-- docs/README.md contract
WHAT: the index of this competition's write-ups, plus the handful of facts every other doc assumes. One line per doc.
A new write-up goes in this folder and gets a line here. The journal (../journal.md) stays the event history;
these files are the current picture and are updated in place.
-->
# Gemma 4 developer agent: docs, as of 2026-10-20

Focus now: training only (curate SWE data, then RL). Kaggle submissions are parked.

## Docs

| file | what it holds |
|---|---|
| [competition.md](competition.md) | The competition in one page: metric, scorer, allowed models, submission format, deadlines |
| [bench_report.md](bench_report.md) | Speed tests and solve counts for every model on the RTX 5090, in plain language |
| [measured_numbers.md](measured_numbers.md) | Output-token budgets per model, fixed run settings, where the sampling config lives, Kaggle speeds |
| [pass_histogram_e2b.md](pass_histogram_e2b.md) | E2B pass counts over 8 tries per task: how many tasks are in the mixed band |
| [outside_data.md](outside_data.md) | Outside task sets and solved-run sets, the adapter we need, graph-tools A/B result, plan for the first outside run |
| [outside_tasks_judge.md](outside_tasks_judge.md) | task-quality judge on 7,137 SWE-rebench tasks: 60.5% solvable as written, reasons, clean pool, held-out repos, repair pilot (2026-10-25) |
| [code_graph_builder.md](code_graph_builder.md) | Our code-graph builder: rebuilds all 127 official graphs exactly (nodes, texts, edges) |
| [embeddings.md](embeddings.md) | The official node embeddings: what they are, why we cannot rebuild them, what they are worth to the agent; decision: random vectors for outside repos |
| [training_prep.md](training_prep.md) | Local training data on hand, the tools that make it, open training steps |
| [training_research.md](training_research.md) | Full research notes with links: datasets, Gemma 4 LoRA stack, RL frameworks |
| [autopilot/](autopilot/README.md) | The Claude Code autopilot that drives the work: playbook, design, decision cards |

## Facts the docs assume

- **Tasks:** 129 public tasks (fastapi 67, rich 48, requests 13, httpx 1). 46 are holdout (`split=holdout` in
  `tasks/manifest.csv`, all rich) and never go into training. 6 are unscoreable and left out of every run (`split=excluded`, reason in `excluded_reason`):
  4 where even the real fix fails, 2 where an empty patch passes. Runs use the other 123 (selector `usable`):
  77 training + 46 holdout.
- **Models allowed on Kaggle:** Gemma 4 31B, 12B, E4B, E2B, one per submission. Qwen 27B is a local reference only.
- **Kaggle scoring:** one task at a time on 4× L4, about 120 tasks in 12 h, so about 6 min per task.
- **Time formula:** task seconds ≈ output tokens ÷ decode speed + 1.9 s per tool call + new prompt tokens ÷ prefill
  speed + 30 s setup (profiles in `kits/swe-agent/run/kaggle_time_profiles.json`).
- **Run ledger:** every task run is one row in `results/task_runs.jsonl`, with its settings pinned in the row.
- **Machine:** RTX 5090 32 GB, 12 CPU cores, 93 GB RAM, 122 GB free disk (2026-10-17), no CUDA compiler.
- [Sandbox hosting options](sandbox_hosting_options.md) — parked: ephemeral big VM vs Modal vs Cloudflare for task containers (2026-10-27)
