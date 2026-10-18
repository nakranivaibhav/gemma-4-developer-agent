<!-- bench_report.md contract
WHAT: one-page, plain-language summary of the local speed tests and the all-model task runs on the RTX 5090,
for planning training rollouts. Numbers come from runs/bench/*.json, results/task_runs.jsonl and
runs/par_sweep_sys.csv. Update in place when a new test lands; the journal keeps the history.
-->
# Model speed and solve report: RTX 5090, as of 2026-10-17

## Summary

| model | solved (average of its runs, out of 123) | max tasks at once |
|---|---|---|
| 31B + adapter | 60.5 (49%) | about 4 to 8 |
| Qwen 27B | 60 (49%) | about 8 |
| 12B | 31 (25%) | 16 or more |
| E4B | 18 (15%) | 64 |
| E2B | 7.3 (6%) | 128 or more |

## Two kinds of tests

1. **Speed test.** Fake coding conversations, no real tasks. It measures how many model replies per minute
   the GPU can produce when N conversations run at the same time. One reply = one "turn".
2. **Real run.** The model works on our 123 real coding tasks, each in its own sandbox, and we check whether
   the tests pass. This gives the solve count.

## 1. Speed test: how many tasks the GPU can run at once

| model | replies/min with 1 task running | best replies/min | number of tasks at once for the best |
|---|---|---|---|
| E2B | 22 | 863 | **128 or more** (still rising) |
| E4B | 21 | 483 | **64** (slows down above this) |
| 12B | 21 | 163 | **16 or more** (highest tested) |
| 31B | 9 to 16 | 32 to 38 | **about 4 to 8** |
| Qwen 27B | 10 to 15 | 40 | **about 8** |

## 2. Real runs: how many of the 123 tasks each model solves

| model | solved (one number per run) |
|---|---|
| 31B + adapter | 64, 57 |
| Qwen 27B | 60 |
| 12B | 34, 28 |
| E4B | 18, 17, 19 |
| E2B | 8, 7, 7 |

## Time per task (the middle task, in seconds)

- **Model time:** the time the model spent working on the task (thinking, replying, using tools).
- **Total time:** start to finish, including sandbox setup, running the tests and cleanup.

| model | tasks at once | model time | total time |
|---|---|---|---|
| E4B | 24 | 155 | 160 |
| E4B | 64 | 527 | 753 |
| E2B | 24 | 141 | 151 |
| E2B | 128 | 1038 | 1043 |

More tasks at once means each task waits longer for the GPU, so each one takes longer. Total output per hour still
goes up, but by less than the number of tasks.

## Machine load during the big real runs

- CPU (12 cores): about 23% busy on average, 99% at short peaks. No task failed because of it.
- RAM: at most 21 GB of 93.
- GPU busy: 82% for E2B at 128, 50% for E4B at 64. The GPU waits while sandboxes run tests.

## E2B pass counts over 8 tries (temperature 0.7)

9.0 solved per try on average (7.3%); 21 tasks mixed (13 training, 8 holdout). Full result: `pass_histogram_e2b.md`.

## Open issues

- **Stuck sandbox containers** (5f8fac7cbb18, 4f0186ef7be6) were cleared by the machine restart on 2026-10-17.
- **Qwen solves missing from the training set:** the converter took only 4 of its 33 training-side solves because of
  its tool-call format. The full conversations are saved in runs/v_qwen_lean_p1/requests.jsonl.gz.
- **Qwen setup:** it runs from envs/qwen (vLLM 0.31.0) with Triton attention and the FlashInfer sampler off,
  because there is no CUDA compiler on this machine.
