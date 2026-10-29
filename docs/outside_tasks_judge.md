<!-- outside_tasks_judge.md contract
WHAT: results of the task-quality judge on outside tasks (SWE-rebench) and on our own Kaggle tasks: how the judge was
built and checked, what it found, the clean pool, the held-out proposal, and the repair (fixer) step. Update in place
as steps land; date each update. Numbers come from runs/judge/ and journal.md; never edit a number without its source.
-->
# Task-quality judge on outside tasks (2026-10-25, repairs updated 2026-10-26)

## What was done
- **Data** (on the SSD, `/home/vaibhav/data/gemma4/outside/`): SWE-rebench filtered (6,542 tasks with prebuilt
  images, 1,790 repos, 2016 to 2025-04), the SWE-rebench monthly fresh sets (860 tasks, 2025-01 to 2026-03), and
  Nebius's 67k OpenHands runs on SWE-rebench. 7,137 unique tasks (rich excluded).
- **Judge**: Opus 5.5 at medium effort through the Claude Agent SDK, read-only tools on a packet per task (issue,
  reference fix, test changes, repo at the base commit, no git history). Binary rubric in `judge_rubric.md`:
  R1 unstated name, R2 exact text, R3 tests beyond the issue, R4 unclear issue, R5 outside facts, R6 environment,
  R7 solvable as written. Code `training/judge/`.
- **Calibration** on an answer key of 76 of our fastapi / requests / httpx tasks (`tasks/judge_answer_key.jsonl`;
  38 read in depth with failed attempts, 38 clean by solve rate). Three prompt rounds; final (v3): unsolvable tasks
  found 24/26, no clean task flagged, self-agreement 0.96-1.00 per item. Caveat: 9 key labels were revised after
  round 1 (each logged in its row).
- **Outside check** on a 100-task pilot: tasks the judge calls solvable were solved by Nebius's 480B model in 75% of
  tries on average (median 97%); tasks it flags, 19% (median 0%).

## What it found (all 7,137 tasks)
| | tasks | share |
|---|---|---|
| solvable as written | 4,316 | 60.5% |
| unsolvable as written | 2,821 | 39.5% |

Reasons among the flagged tasks (a task can have several): tests check more than the issue asks 1,909 (68%), exact
text or format the issue never gives 1,031 (37%), unclear issue 831 (29%), unstated name 705 (25%), outside facts 200
(7%), environment (network, dates, versions) 68 (2%). Rates vary a lot by repo: conan 53/73 flagged, pyflakes 0/16.

Our own Kaggle tasks show the same defects: 27 of the 38 hardest fastapi train tasks are unsolvable as written.

## Pools (proposal, not yet frozen)
- **Held-out repos (never trained on)**: sympy, astropy, litellm, moto, schemathesis, nested-pandas, keras, PyPSA,
  xarray, PSyclone: 112 tasks, 99 created after Gemma 4's data cutoff (January 2025), 74 of them clean. Plus rich (46).
  Chosen from the fresh monthly sets: at least 4 clean post-cutoff tasks, at least 60% clean, and mostly post-cutoff
  (sqlglot, jsonargparse and tox qualified but have many pre-cutoff tasks, so they stay in training).
  Files: `/home/vaibhav/data/gemma4/judge/heldout_repos.txt`, `heldout_clean_v1.ids`.
- **Training pool**: 7,025 tasks outside those repos; 4,232 clean over 1,494 repos (`train_clean_v1.ids`);
  2,725 flagged without an environment problem are repair candidates.

## Repairs (fixer pilot)
An Opus fixer writes a short "Requirements:" addendum (what the tests need, never how), a second blind call checks it
does not give away the implementation, and the frozen judge re-judges the repaired task.
- Our 24 judge-flagged fastapi tasks: 21 fixable (3 large designs refused), 20 pass the re-judge, 0 leaks.
- 50 flagged SWE-rebench tasks: 50 fixable, 3 caught by the leak check, 47 repaired. Median addendum: 2 bullets.
- Outcome check with base E2B on the 20 repaired fastapi tasks (4 tries each): repaired 1/80, original 0/80.
  Inconclusive (E2B too weak on these tasks). A 31B check is the next step before repairing all 2,725.
- 31B check on the 20 repaired fastapi tasks (2 tries each): original text 0/40, first notes 18/40, minimal notes 16/39
  (14 within the clock). The repairs work; the minimal notes keep about the same benefit with half the text.
- **Minimal notes (2026-10-26)**: the fixer adds a fact only if a solver could not infer it from the issue and the repo
  AND a fix that follows the issue would fail without it; a blind trim pass then keeps a line only with the test line
  that needs it. Full run on the 2,675 flagged training tasks: 2,618 notes (57 unfixable), 120 caught by the leak check,
  2,533 pass the blind re-judge. Notes: 1 line 57%, 2 lines 26%, 3 lines 10%, 4+ 7%.
- **Training pool now**: 6,765 approved tasks (4,232 clean + 2,533 repaired). Per-task state (judge, repair, image
  check, E2B screen) in `tasks/outside_manifest.csv`, rebuilt by `kits/swe-agent/outside/manifest.py build`.
- Open question: Kaggle's hidden tasks have unrepaired text; training on repaired text should be A/B-tested against
  dropping those tasks.
