# Agents

Every agent folder we ran, byte-for-byte as it ran. An agent folder is exactly what the competition accepts as a
submission: `agent.yaml` (model, tools, sub-agents), `prompts/`, `configs/sampling.yaml`, `eval_config.yaml`, and
optional LoRA `adapters/` (here always the starter's no-op placeholder: `lora_B` is all zeros).

Run one with `make run AGENT=agents/<name> TASKS=<selector>` (see `kits/swe-agent/README.md`). Every ledger row in
`results/task_runs.jsonl` pins the sha256 of each file of the agent it ran, so a result can always be traced back to
one of these folders.

**Read the numbers with care:** "solved" pools every task-run of that agent in the ledger, over different task lists
(tune 77, usable 123, holdout 46), models and dates. It is not a ranking. Paired comparisons on the same tasks are in
`docs/` (for example `docs/pass_histogram_e2b.md` and `docs/bench_report.md`).

## Gemma 4 31B: prompt rounds (tune tasks unless noted)

| agent | what it is | task-runs | solved |
|---|---|---|---|
| `sample` | the competition's starter agent, unchanged (max output 16,384 tokens) | 49 | 11 (environment checks) |
| `out8k` | `sample` with max output tokens 8,192: the first full baseline | 523* | 239 (46%)* |
| `base_l100` | identical files to `out8k`; the "base" arm of the 4-pass fleet run on all 123 tasks | *shared with `out8k`* | |
| `edit`, `pace`, `sibling` | `out8k` + one prompt section each: editing habits / paced budget plan / check sibling places | not run alone | |
| `habits` | `edit` + `pace` + `sibling` together (round 1) | 154 | 74 (48%) |
| `r02_plain` | system prompt rewritten in plain language (round 2) | 154 | 73 (47%) |
| `r03_lean` | system prompt cut to about half (round 3) | 453* | 225 (50%)* |
| `lean_l100` | identical files to `r03_lean`; the "lean" arm of the fleet run | *shared with `r03_lean`* | |
| `r04_lean` | lean + a bundle aimed at fewer tokens per task, no helper sub-agent (round 4) | 155 | 75 (48%) |
| `r05_prompt` | lean with only round 4's system prompt: attribution for round 4 (round 5) | 154 | 62 (40%) |
| `r06_garble` | lean + a recovery rule for garbled edits (round 6, stopped early) | 36 | 20 (56%) |

\* `out8k` + `base_l100` and `r03_lean` + `lean_l100` have identical files, so their rows cannot be told apart by hash.

## Other Gemma 4 sizes and a reference model (the lean agent, model name switched)

| agent | what it is | task-runs | solved |
|---|---|---|---|
| `lean_12b` | lean on Gemma 4 12B (placeholder adapters removed: they do not load on the 12B) | 246 | 62 (25%) |
| `lean_e4b` | lean on Gemma 4 E4B | 369 | 54 (15%) |
| `lean_e2b` | lean on Gemma 4 E2B | 365 | 22 (6%) |
| `lean_qwen` | lean on Qwen3.8-27B NVFP4, reference only (not allowed in the competition) | 123 | 60 (49%) |

## E2B experiments

| agent | what it is | task-runs | solved |
|---|---|---|---|
| `e2b_t07` | `lean_e2b` at temperature 0.7: 8 samples per task for a pass-count histogram | 984 | 72 (7%) |
| `e2b_t07_nograph` | `e2b_t07` without the 3 code-graph tools: the graph-tools A/B | 984 | 54 (5%) |

The A/B is paired on the same 123 tasks x 8 samples: with graph tools 72 passes, without 54, P(no-graph better) = 0.007.

## Harness probes (a few tasks each, to learn how the harness behaves)

| agent | what it probed |
|---|---|
| `skill`, `skill_probe` | skills: which tools they add, which resource paths load, where scripts run |
| `subagent_probe` | a mandatory first call to the `code_analyzer` sub-agent |
| `subagent_transfer` | the analyzer as a hand-over sub-agent instead of a tool |
