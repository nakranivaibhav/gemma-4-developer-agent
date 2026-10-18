# The competition

[Google - The Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent) on Kaggle:
build a coding agent that fixes real Python repositories, scored by how many hidden tasks it resolves.

| | |
|---|---|
| metric | Resolution Rate: share of hidden tasks whose target tests all pass after the agent's patch (higher is better) |
| hidden test set | about 120 tasks from private repositories, split evenly public / private leaderboard |
| scorer | 4x NVIDIA L4, tasks run one at a time, 12 hours for everything (about 6 minutes per task) |
| models allowed | Gemma 4 31B, 12B, E4B or E2B (4-bit QAT checkpoints), one per submission |
| submission | `submission.zip`: declarative agent config + optional LoRA adapters, no Python entry point |
| public data | 129 tasks from 4 public repos (`tasks/manifest.csv`), repo snapshots, code graphs + embeddings, wheels, Docker files |
| deadlines | entry and team merger 2026-11-25, final submission 2026-12-02 (23:59 UTC) |

daily_submission_limit: 1

## Details (written at the start, 2026-10-09)

We do not predict rows — we **build a coding agent**. The agent is handed a real
Python repository plus a bug report / feature request (SWE-bench style), works in
an offline sandbox with 9 fixed tools (shell, read/edit/write file, 3 code-graph
tools, status, submit_patch), and must produce a git patch. Score = **Resolution
Rate**: the fraction of hidden tasks whose target tests all pass after our patch
is applied to a fresh copy of the repo (higher is better, range 0–1).

A submission is a `submission.zip` holding **declarative config only** — an
`agent.yaml` (+ optional sub-agents, prompts, skills, `eval_config.yaml`) and
optional **LoRA adapters** (`.safetensors`, rank ≤128, ≤8 adapters). No Python
entrypoint is executed. Kaggle runs it on 4× L4 GPUs with the one allowed base
model, `gemma-4-31b-it-qat-w4a16-ct` (4-bit Gemma 4 31B, 32,768-token context),
producing `submission.parquet` `[id, prediction]` where prediction is the patch.
Total unpacked size < 3 GiB. Sandboxes are air-gapped, 4 GiB RAM / 2 vCPU.
Per-task defaults: 60 min, 100 tool calls, 500 turns (overridable in
`eval_config.yaml`). Edits to test files / pytest config are discarded at grading.

Levers therefore are: prompts, agent topology (sub-agents, loops), sampling /
thinking settings, budgets, skills, and fine-tuned LoRA adapters. Public data:
`tasks.jsonl` (training tasks), repo `snapshots/`, pre-built `graphs/` +
`embeddings/`, `wheels/`, docker files, and a `sample_submission/` directory.
Source of the mechanics: `data/HARNESS_README.md`. Limit of 1 submission/day and
the deadline come from the Kaggle API (maxDailySubmissions, deadline); team
merger / entry deadline 2026-11-25. A sibling "paper track" comp exists
(gemma-4-developer-agent-paper) — out of scope here.

From the competition pages (saved in `refs/pages/`): the hidden test set is
**about 120 tasks, split evenly public/private, curated from PRIVATE repositories**
(not the four public ones). The whole run has **12 hours for all tasks**, sandbox
setup included, patch validation excluded. Custom tools cannot be registered —
only the 9 harness tools plus sub-agents via `agent_tool` — but **skills may ship
Python/bash scripts** run in the sandbox via `run_skill_script` (time debited
from the budget) and resources read via `load_skill_resource`. External data and
models are allowed if reasonably accessible and low cost (rules §6). Two final
submissions are selected for judging.

## Settled since

- **Model sizes:** since 2026-10-11 four Gemma 4 sizes are allowed (31B, 12B, E4B, E2B), one per submission
  (host post in forum thread #748287). The 12B serves about 1.9x faster than the 31B on the scorer's GPUs.
- **Task concurrency on Kaggle:** the host confirmed the hidden tasks run one at a time, about 120 tasks in 12 h;
  running over the time limit errors the whole submission.
- **Local serving:** the exact 31B checkpoint serves on one RTX 5090 (32 GB) with vLLM 0.19.1: about 20 GiB of
  weights, 32k context, `--max-num-seqs` 4 to 16 (`kits/swe-agent/serve/serve_model.py`).
