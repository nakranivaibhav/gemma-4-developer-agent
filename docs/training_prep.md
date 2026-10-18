<!-- training_prep.md contract
WHAT: the starting point for training work (SFT data, then RL): what data exists locally, the tools that make it,
what is still open. Facts with numbers and paths; outside research lives in docs/training_research.md (not
repeated here). Update in place when a number changes; date every update.
-->
# Training prep, as of 2026-10-15

## Local data we already have

**Task pool:** 129 public tasks: fastapi 67, rich 48, requests 13, httpx 1. 46 are holdout (`split=holdout` in `tasks/manifest.csv`) and
never go into training. That leaves 83 train-side tasks. Per-task solve rates for every model are in
`runs/sft/task_pool.csv`, built by `analysis/task_pool.py`.

Train-side tasks by 31B outcome, over all local runs so far:

| 31B outcome | tasks | use |
|---|---|---|
| solved sometimes | 37 | RL signal (reward varies) and SFT data |
| always solved | 10 | SFT data, little RL signal |
| never solved | 31 | need a stronger model's trajectory or a hint first |
| not tried | 5 | left out of every run list so far (not in the `usable` selector) |

**Solved trajectories:** `runs/sft/solved_v0.jsonl` is built by `training/collect_sft.py`. Each row is OpenAI chat
format: system prompt, task prompt, then assistant turns carrying `reasoning_content` and `tool_calls`, with one tool
message per call. Sub-agent steps are dropped. Each row carries `out_tokens`, `tool_calls`, `kaggle_s`,
`within_cap` (330 s), `peak_context_tokens` and `rendered_tokens`.

| model | rows | distinct tasks | within cap: rows / tasks |
|---|---|---|---|
| 31B | 663 | 47 | 375 / 43 |
| 12B, partial | 41 | 30 | 6 / 5 |

The 12B, E4B, E2B and Qwen rows grow as the running chain finishes. Re-run both scripts afterwards.

**Format check:** `training/sft_render_check.py` renders each row through the 31B chat template using the harness's
own tool schemas (`runs/sft/tools_schema.json`, compiled offline from the lean agent).

- Runs under about 20 tool calls render to the same token count the model really saw: ratio 0.99–1.00.
- Longer runs render 1.1–5.8× longer, because the harness compacted their history mid-run.
- So a compacted run's full transcript is not what the model saw. 157 rows exceed 32,768 tokens.
- **223 of the 375 within-cap 31B rows are exact as stored.** Use those first.

**Exact data from now on:** `kits/swe-agent/run/run_round.sh` now sends every run through `kits/swe-agent/run/log_proxy.py` with full
capture (`CAPTURE=1` by default). Every chat request and response is kept in `runs/<run>/requests.jsonl.gz`.
These are exact prompt/completion pairs, compaction included, and they cover sub-agent calls too. Capture starts
with the E4B step of the chain running tonight. The Qwen runs will be the first exact teacher data for distillation.

## Constraints that shape training (from the harness and the research)
- **Adapter size:** the archive limit is 3 GiB. For the 31B, a rank-128 LoRA on all 7 projections is about
  2 GB; rank 256 is about 4 GB and does not fit. Allowed vLLM ranks are 1, 8, 16, 32, 64, 128, 256, 320, 512.
- **Context:** the model limit is 32,768 tokens, prompt plus output, so train at 16k–32k sequence length.
- **Loss mask:** the Gemma 4 template puts tool results inside the model's own turn and has no generation tags.
  Library "assistant only" loss masks will be wrong; write our own mask (train on calls and answers, not tool output).
- **Base weights:** the scorer serves the int4 QAT checkpoint. A LoRA trained elsewhere must be checked on that
  exact serving path (vLLM 0.19.1, int4 base plus LoRA) before any real training. Details and options are in
  docs/training_research.md section 3.
- **Contamination:** our local holdout has fastapi, rich and requests tasks. Drop
  `fastapi|rich|requests|httpx` from any outside pool so the local measurement stays clean. SWE-bench train
  has 28 fastapi tasks, and SWE-bench test has 44 requests tasks.

## Open for tomorrow
1. **Outside tasks into our sandbox.** Our harness runs tasks from `data/snapshots/*.tgz` in
   `swebench-sandbox-local`. SWE-Gym and SWE-rebench ship one Docker image per task. Find the smallest adapter
   between them, then do a 5-task smoke run.
2. **LoRA serving smoke test.** Train a rank-8 LoRA for a few steps on `solved_v0` exact rows, serve it on the
   int4 base exactly like the scorer does, and check that outputs change and nothing errors.
3. **First SFT set.** Use exact within-cap rows plus the captured E4B, E2B and Qwen requests. Prefer short solves,
   since the 6-minute cap rewards them.
4. **RL pool.** Start from the 37 sometimes-solved tasks plus decontaminated outside tasks. The research suggests
   the Modal SWE-Gym pattern as the template: GRPO, LoRA rank 32, 8 attempts per task.
