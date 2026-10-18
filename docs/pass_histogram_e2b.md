<!-- pass_histogram_e2b.md contract
WHAT: per-task pass counts for E2B over 8 independent tries, to size the "mixed" band that RL can learn from.
Numbers come from results/task_runs.jsonl rows hist_e2b_t07_s1..s8, via analysis/pass_hist.py. Add a section per new
model or setting; never change a measured number without a journal CORRECT line.
-->
# E2B pass counts over 8 tries, 2026-10-17

E2B at temperature 0.7 passes 21 of 123 tasks on some tries but not all. Only 13 of those are training tasks, and
only 8 of those pass on 2 to 7 tries. That is too few for RL on our own tasks alone.

## Setup

- **Agent:** `agents/e2b_t07/`, the lean E2B agent with temperature 0.7 and top_p 0.95 (normally 0.2).
- **Tasks:** selector `usable` in `tasks/manifest.csv`: 77 training + 46 holdout. The 6 unscoreable tasks (`split=excluded`)
  are left out.
- **Tries:** 8 independent tries per task, run ids `hist_e2b_t07_s1` to `hist_e2b_t07_s8`. There is no real seed: ADK
  does not forward one, so each try is a fresh random draw.
- **Load:** 8 runners × 16 tasks = 128 at once on one vLLM server (RTX 5090), from 07:42 to 08:50 UTC (68 min).
- **Script:** `bash kits/swe-agent/experiments/hist.sh agents/e2b_t07 hist_e2b_t07 usable 8 128`, then
  `envs/harness/bin/python -I analysis/pass_hist.py hist_e2b_t07`.

## Result

| passes out of 8 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| training tasks (77) | 61 | 5 | 3 | 3 | 0 | 0 | 1 | 1 | 3 |
| holdout tasks (46) | 38 | 4 | 1 | 3 | 0 | 0 | 0 | 0 | 0 |

| band | training | holdout | all |
|---|---|---|---|
| never passed | 61 | 38 | 99 |
| mixed (some tries pass) | 13 | 8 | 21 |
| always passed | 3 | 0 | 3 |

- **Solved per try:** 12, 8, 13, 11, 6, 8, 8, 6 = 72 in total, 9.0 on average (7.3%). The earlier runs at
  temperature 0.2 solved 8, 7, 7 (7.3 on average).
- **Solved at least once:** 24 of 123 tasks.

## Mixed tasks

| task | passes | split |
|---|---|---|
| fastapi_14301 | 7/8 | training |
| requests_7309 | 6/8 | training |
| requests_6629 | 3/8 | training |
| requests_7205 | 3/8 | training |
| requests_7433 | 3/8 | training |
| fastapi_14360 | 2/8 | training |
| requests_6589 | 2/8 | training |
| requests_7315 | 2/8 | training |
| fastapi_14459, fastapi_14492, fastapi_14616, fastapi_9425, requests_6644 | 1/8 each | training |
| rich_3454, rich_3518, rich_3905 | 3/8 each | holdout |
| rich_3006 | 2/8 | holdout |
| rich_2725, rich_2943, rich_3894, rich_4077 | 1/8 each | holdout |

A task that passes 1 of 8 gives RL a learning signal only some of the time: with 8 rollouts per step, a group often
has no pass at all.

## Errors and load

- **Harness errors:** 12 of 984 runs. All are the model's own failures, counted as fails: 9 ended without calling
  `submit_patch`, 3 filled the 32k context.
- **Machine:** CPU 84% busy on average (99% peak), GPU 88% busy on average, RAM 37 GB at most, up to 128 containers.
  All 128 slots stayed full the whole run, so CPU is close to being the limit at this load.

## Compared with the earlier E2B runs (temperature 0.2)

Computed from the ledger. The earlier runs are `v_e2b_lean_p1`, `v_e2b_lean_p2` and `v_e2b_par_p1` (3 tries;
`v_e2b_par_p1` has 119 of 123 tasks, 4 were cut). "All 11" pools those 3 tries with the 8 new ones.

| | tries | solved at least once | mixed | always | never |
|---|---|---|---|---|---|
| temperature 0.2, training (77) | 3 | 10 | 7 | 3 | 67 |
| temperature 0.2, holdout (46) | 3 | 2 | 2 | 0 | 44 |
| temperature 0.7, training (77) | 8 | 16 | 13 | 3 | 61 |
| temperature 0.7, holdout (46) | 8 | 8 | 8 | 0 | 38 |
| all 11, training (77) | 11 | 17 | 15 | 2 | 60 |
| all 11, holdout (46) | 11 | 8 | 8 | 0 | 38 |

All 123 tasks: solved at least once 12 (3 tries at 0.2), 24 (8 tries at 0.7), 25 (all 11). 11 tasks are solved in
both sets; only fastapi_15589 was solved at 0.2 and never at 0.7. The bands depend on the number of tries: more tries
move tasks from "never" and "always" into "mixed".

## Without the graph tools

Same 123 tasks × 8 tries with the 3 graph tools removed (`hist_e2b_t07_nograph`): 54 passes instead of 72, 18 tasks
solved at least once instead of 24, 16 mixed (10 training, 6 holdout) instead of 21. Full comparison:
`outside_data.md`, section "Graph-tools A/B". Per-task table: `runs/sft/pass_hist_hist_e2b_t07_nograph.csv`.

## Files

- Per-task table: `runs/sft/pass_hist_hist_e2b_t07.csv` (task_id, split, tries, passes, rate, band, harness_errors).
- Every model request and reply: `runs/hist_e2b_t07/requests.jsonl.gz` (147 MB).
- Machine log every 30 s: `runs/hist_e2b_t07/sys.csv`. Runner logs: `runs/hist_e2b_t07_s<k>/run.log`.
