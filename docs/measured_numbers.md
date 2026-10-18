<!-- measured_numbers.md contract
WHAT: output-token budgets, fixed run settings, where the sampling config lives, and other measured numbers.
Copied from the Claude Doc https://claude.ai/code/artifact/ec1a753a-d38b-44b9-a009-503e3bde77b8 on 2026-10-17;
this file is the repo copy. Update in place when a number changes; the journal keeps the history.
-->
# Gemma 4 agent: measured numbers and fixed settings, as of 2026-10-17

## Summary

E4B and E2B use about 2.8k output tokens on a solved task, far under their 6-minute budget. The 31B needs about
8.2k, right at its budget of 6.3k–9.0k. The 12B uses about 53k, five times its budget.

| model | output tokens a 6-min Kaggle task can hold | output tokens a solved run uses (median) | tool calls in a solved run (median) | solved (average, out of 123) |
|---|---|---|---|---|
| 31B + adapter | 6.3k – 9.0k | 8.2k | 28 | 60.5 |
| Qwen 27B (reference, not allowed on Kaggle) | not measured | 17.0k | 27 | 60 |
| 12B | 13.6k – 19.3k | 53.4k | 49 | 31 |
| E4B | 18.8k – 26.6k | 2.8k | 5 | 18 |
| E2B | 27.0k – 38.1k | 2.9k | 5.5 | 7.3 |

The budget range runs from 65 tool calls (low end) to 20 tool calls (high end), because every tool call costs about
1.9 s of the 330 s. The 12B, E4B and E2B budgets are without an adapter; an adapter cost the 31B about 12%.

## Output tokens per task

Failed runs write two to three times more output than solved runs, for every model. Output tokens = thinking +
reply + tool-call text, summed over the whole task, from the run traces.

| model | solved runs | output, solved (median) | output, solved (middle half) | peak context, solved (median) | failed runs | output, failed (median) | tool calls, failed (median) |
|---|---|---|---|---|---|---|---|
| 31B + adapter | 121 | 8,178 | 5,279 – 12,446 | 14,918 | 125 | 18,773 | 48 |
| Qwen 27B | 60 | 16,979 | 7,347 – 38,074 | 17,282 | 62 | 55,884 | 91 |
| 12B | 62 | 53,402 | 31,394 – 75,946 | 22,160 | 183 | 50,988 | 55 |
| E4B | 54 | 2,790 | 1,970 – 3,739 | 8,414 | 312 | 5,446 | 10 |
| E2B | 22 | 2,880 | 1,611 – 3,670 | 7,736 | 342 | 6,981 | 11 |

Solved runs that fit the 6-minute cap:

- **31B:** 63 of 121, using 5,448 tokens and 18 tool calls (median).
- **12B:** 10 of 62, using 11,652 tokens and 18.5 tool calls.
- **E4B and E2B:** all of them.

Runs used: 31B `l100_lean_p1`, `l100_lean_p2` · 12B `v_12b_lean_p1/p2`, `v_12b_lean_rest_p1/p2` · E4B and E2B
`v_e4b_lean_p1/p2`, `v_e4b_par_p1`, `v_e2b_lean_p1/p2`, `v_e2b_par_p1` · Qwen `v_qwen_lean_p1`. All on the 123 tasks,
with the same lean prompts.

## Fixed settings

Every Gemma model runs with the same sampling and the same limits. Only Qwen uses its own model card's sampling.

| setting | Gemma 31B / 12B / E4B / E2B | Qwen 27B |
|---|---|---|
| temperature | 0.2 | 1.0 |
| top_p | 0.95 | 0.95 |
| top_k | not set | 20 |
| max output tokens per reply | 8,192 | 8,192 |
| thinking budget per reply | 4,096 (thoughts kept) | 4,096 |
| context length | 32,768 | 32,768 |
| max tool calls per task | 100 | 100 |
| max time per task | 60 min | 60 min |
| max turns per task | 500 | 500 |
| timeout per sandbox command | 300 s | 300 s |

Exception: the pass-count run (see `pass_histogram_e2b.md`) used `agents/e2b_t07/`, the E2B agent with
temperature 0.7.

Kaggle limits we plan against:

- **Time per task:** 6 min, scored one task at a time on 4× L4. We cut locally at 330 s of predicted Kaggle time and
  count a cut task as unsolved.
- **Whole submission:** 12 h for about 120 tasks.
- **Kaggle time formula:** task seconds ≈ output tokens ÷ decode speed + 1.9 s per tool call + new prompt tokens ÷
  prefill speed + 30 s setup.

## Where the sampling config lives

Each agent folder holds its own sampling file, `configs/sampling.yaml`. The agent's `agent.yaml` pulls it in with
`generate_content_config: !include configs/sampling.yaml`. All paths are under `agents/`.

| model | agent folder | sampling file | limits file |
|---|---|---|---|
| 31B + adapter | `lean_l100_agent/` | `lean_l100_agent/configs/sampling.yaml` | `lean_l100_agent/eval_config.yaml` |
| 12B | `lean_12b_agent/` | `lean_12b_agent/configs/sampling.yaml` | `lean_12b_agent/eval_config.yaml` |
| E4B | `lean_e4b_agent/` | `lean_e4b_agent/configs/sampling.yaml` | `lean_e4b_agent/eval_config.yaml` |
| E2B | `lean_e2b_agent/` | `lean_e2b_agent/configs/sampling.yaml` | `lean_e2b_agent/eval_config.yaml` |
| Qwen 27B | `lean_qwen_agent/` | `lean_qwen_agent/configs/sampling.yaml` | `lean_qwen_agent/eval_config.yaml` |

The Gemma sampling file in full:

```yaml
temperature: 0.2
top_p: 0.95
max_output_tokens: 8192
thinking_config:
  thinking_budget: 4096
  include_thoughts: true
```

Other settings live elsewhere:

- **Server:** context length, GPU memory share and tool parser are set in `kits/swe-agent/serve/serve_model.py`.
- **Kaggle speed profiles:** in `kits/swe-agent/run/kaggle_time_profiles.json`, used by `kits/swe-agent/run/kaggle_time.py`.
- **Per-run copy:** every run pins its exact settings into each ledger row of `results/task_runs.jsonl`, under the
  `settings` field.
- **Seed:** ADK's LiteLLM path does not forward a seed to the server, so repeated runs are independent random draws.

## Other measured details

### Speed on Kaggle 4× L4, one task at a time

The 31B with an adapter writes about 30 tokens/s, so the 6-minute cap is tight. The E2B writes about 4× faster.
Measured 2026-10-15, from `kits/swe-agent/run/kaggle_time_profiles.json`.

| model | output tokens/s at 1k context | output tokens/s at 30k context | prompt reading (tokens/s) |
|---|---|---|---|
| 31B + adapter | 33.6 | 29.0 | 1,260 |
| 31B, no adapter | 38.3 | 32.4 | 1,296 |
| 12B | 79.6 | 59.9 | 2,708 |
| E4B | 113.2 | 85.5 | 4,820 |
| E2B | 171.1 | 119.4 | 8,206 |

### Solved within the 6-minute cap (two runs each)

| model | solved at any speed | solved within the cap |
|---|---|---|
| 31B + adapter | 64, 57 | 30, 33 |
| 12B | 34, 28 | 5, 5 |
| E4B | 18, 17 | 18, 17 |
| E2B | 8, 7 | 8, 7 |

### Local RTX 5090: tasks at once

| model | max tasks at once | replies per minute there | replies per minute, 1 task |
|---|---|---|---|
| E2B | 128 or more | 863 | 22 |
| E4B | 64 | 483 | 21 |
| 12B | 16 or more | 163 | 21 |
| Qwen 27B | about 8 | 40 | 10 |
| 31B | about 4 to 8 | 32 – 38 | 9 |

More tasks at once does not change how many get solved, but each task takes longer. At peak, CPU hit 99% of 12 cores
and RAM 21 GB of 93; the GPU was 50–82% busy on average.

### Training data on hand

- **Passing runs:** 979 in total; 806 are on training-side tasks, 173 on holdout tasks we keep out.
- **Coverage:** the 806 cover only 49 of 83 training tasks.
- **Converted:** 748 are in `runs/sft/solved_v0.jsonl`. Most of the rest are Qwen runs, waiting on a converter fix
  for its tool-call format.
- **Since then:** the E2B pass-count run added 984 more runs with full request capture, 72 of them solved (see
  `pass_histogram_e2b.md`).
