# Gemma 4 Developer Agent

Our working repo for the Kaggle competition
[Google - The Gemma 4 Developer Agent](https://www.kaggle.com/competitions/gemma-4-developer-agent): a coding
agent built on Gemma 4 that gets a real Python repository plus an issue, works in an offline sandbox, and must
produce a patch that makes the hidden tests pass. The competition is summarised in [docs/competition.md](docs/competition.md).

Everything here is shared as it is: the agents we ran, the local evaluation setup, every task-run result, the
research log, and the Claude Code autopilot that drove the work. Shared publicly under the competition rules
(code shared with all participants, Apache-2.0).

## Results so far (local, repaired sandbox, 123 usable public tasks)

| model | tasks solved per run | notes |
|---|---|---|
| Gemma 4 31B | 60.5 of 123 (49%) | the lean agent, average of 2 runs |
| Gemma 4 12B | 31 (25%) | |
| Gemma 4 E4B | 18 (15%) | |
| Gemma 4 E2B | 7.3 (6%) | |
| Qwen3.8 27B (reference, not allowed) | 60 (49%) | |

- **Code-graph tools help.** E2B, 123 tasks x 8 samples: 72 passes with the graph tools, 54 without
  (P(no-graph better) = 0.007). [docs/outside_data.md](docs/outside_data.md)
- **The official code graphs can be rebuilt.** `graph/code_graph.py` reproduces all 127 official graphs exactly:
  367,985 nodes, every text, 226,247 edges. [docs/code_graph_builder.md](docs/code_graph_builder.md)
- **The shipped sandbox under-counts.** In the competition image as shipped the real fix passes 43 of 129 public
  tasks; in our repaired image, 125. [kits/swe-agent/README.md](kits/swe-agent/README.md)

All write-ups: [docs/README.md](docs/README.md). Next step: curate SWE training data, then RL.

## Where everything is

```
agents/          every agent we ran, as submitted: agent.yaml + prompts + configs (agents/README.md)
tasks/           manifest.csv: one row per task (split, named lists, where its setup files live)
results/         task_runs.jsonl: one row per task-run with pinned settings; small summaries
kits/swe-agent/  the run machinery: Makefile, sandbox image, model server, runner, gold check, benchmarks
graph/           code-graph builder + exact-match check against the official graphs
analysis/        A/B tests, pass histograms, trace and failure stats, Kaggle-time estimates
training/        SFT data collection from solved runs
docs/            write-ups; docs/autopilot/ explains the autopilot
journal.md       the research log: one line per event, append-only
CLAUDE.md        the autopilot's operating rules (.claude/ and .codex/ hold its skills and worker agents)
tools/           the autopilot's scripts: journal search, Kaggle API wrapper, run watchdog, stop hook
```

Not in git (made locally by `make setup`): `data/` (competition data and model weights), `envs/` (Python
environments), `runs/` (full run outputs: traces, patches, test logs).

## Quick start

Needs Linux, an NVIDIA GPU, Docker, `uv`, `make`, and a Kaggle account that has accepted the competition rules
(`KAGGLE_USERNAME` / `KAGGLE_KEY` set).

```bash
make setup                                   # data, E2B weights, Python env, sandbox image
make serve SIZE=e2b                          # model server in the background
make run AGENT=agents/e2b_t07 TASKS=smoke6   # 6 tasks; results in runs/ and results/task_runs.jsonl
make down
```

Details and options: [kits/swe-agent/README.md](kits/swe-agent/README.md).

## How the work is driven

The work runs inside [Claude Code](https://claude.com/claude-code) with an autopilot set up from the "grandmaster"
template: `CLAUDE.md` holds the rules, `.claude/skills/` the stage procedures (`start`, `eda`, `validate`, `baseline`,
`experiment`, `progress`, ...), and `.claude/agents/` the worker roles. Every decision and measurement is a line in
`journal.md`. See [docs/autopilot/README.md](docs/autopilot/README.md).

## License

Apache License 2.0, see [LICENSE](LICENSE). The competition data, the model weights and the competition's own
harness (`swegemma`, `adk-submission`) are not part of this repo and keep their own terms.
