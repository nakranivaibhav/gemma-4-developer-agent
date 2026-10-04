# Decision Cards — spec + worked examples

A Decision Card is the one thing the human ever has to read. Every stage ends
with one: a short, plain-language readout of what you found, what you propose,
and what it costs — then you either **wait** or **proceed**, decided by the
autonomy dial. Write for a smart non-specialist. No jargon, no acronyms left
unexplained, no in-chat thumbnails as proof — report numbers and file paths and
let the human open files at full resolution.

This doc is the reference. It (1) shows three filled-in cards for the **House
Prices** competition
(`kaggle.com/competitions/house-prices-advanced-regression-techniques`), and
(2) documents the per-field reasoning behind a good card. The canonical card
*format* lives in CLAUDE.md's "Decision Card format" — this doc never re-pastes it.

---

## The card format

The canonical format block lives in CLAUDE.md "Decision Card format"; below is only
the per-field reasoning that goes beyond it.

Field rules:
- **`<stage>`** is one of the six gates, in order: `understand · toolkit · eda ·
  validation · experiment_plan · submit`.
- **What's going on** — one sentence a non-specialist gets on the first read.
- **Found / propose** — 2–4 bullets. Facts you found, or the thing you want to
  do next. Numbers belong here, with units and direction ("lower is better").
- **Why** — one line of justification, in plain words.
- **Cost** — always three parts: wall-clock time · compute · submissions used out
  of the daily limit (spec.md `daily_submission_limit`). If a card spends no
  submission, say `0 submissions`.
- **Your call** — the four standing options. `[Tell me more]` means "expand this
  card with detail / file paths," it is not a separate decision.
- **Autonomy line** — the current mode and whether this card is *waiting* for the
  human or *proceeding* on its own (see the dial table at the end).

The card is a readout, not a place to do work. The work already happened; its
artifacts are on disk (`spec.md`, `eda.md`, `validation.md`, `graph.md`,
`submissions.md`, the node folders). The card points at them.

The budget in **Cost** is derived, never typed from memory (limit from spec.md):
```bash
lim=$(grep -oP 'daily_submission_limit:\s*\K\d+' comps/<slug>/spec.md)
uv run tools/kaggle_io.py budget --ledger comps/<slug>/submissions.md --limit "$lim"
# -> 2026-06-05  2/5 used  (3 remaining, resets 00:00 UTC)
```
and "today" for that line comes from `date -u +%F`, never your own sense of the
date — a competition spans days and gets resumed.

---

## Example A — UNDERSTAND card (House Prices)

The `understand` gate. The goal here is to read the competition's own words back
in plain language and confirm we got the metric right *before* anything is built
— a wrong reading of the metric poisons every later number. Built by
`/kaggle-start <url>` from the pulled overview + `sample_submission.csv`; the
facts are frozen in `comps/house-prices.../spec.md`.

```
📋 understand
What's going on:   This is a contest to guess the sale price of houses in Ames,
                   Iowa from ~80 facts about each house (size, year built,
                   neighborhood, garage, and so on).
Found / propose:   • Goal: predict one dollar number — the sale price — for each
                     house in the test file.
                   • How they grade us: they compare our guesses to the real
                     prices after taking the logarithm of both, then measure the
                     typical gap (this is "RMSE on log SalePrice"). LOWER is
                     better; a perfect score is 0.
                   • Taking the log means a $20k miss on a $500k house counts the
                     same as a $4k miss on a $100k house — percentage error, not
                     dollar error. So we'll train and validate on log-price too.
                   • What we hand in: a CSV with two columns, Id and SalePrice,
                     one row per test house — ~1459 rows, matching
                     data/sample_submission.csv exactly.
                   • Budget: ~5 submissions per day. No hard deadline (this is an
                     evergreen practice competition).
                   • Where the bar sits: a guess-the-average baseline scores
                     ~0.42; a tidy gradient-boosted model lands ~0.125; the top
                     of the public board is ~0.0 (heavily blended). Our first
                     real target is to beat ~0.13.
Why:               If we optimize dollar error instead of log error, we'd chase
                   expensive houses and score worse on their metric.
Cost:              ~2 min reading · no compute · 0 submissions.
Your call:         [Approve] [Change something] [Skip] [Tell me more]
Autonomy: interactive — waiting
```

Notes on what makes this card correct:
- The metric is stated **with direction** ("LOWER is better") and **in words**
  ("typical gap after taking logs"), not just "RMSLE."
- The submission shape (`Id,SalePrice`, ~1459 rows) matches what
  `validate_submission.py` will later enforce against `sample_submission.csv`.
- "Where the bar sits" gives the human a yardstick so the later CV numbers mean
  something. Those reference points are public-knowledge ballparks, labeled as
  such — not promises.
- `understand` is a human gate in every mode except `full_auto`, so the card is
  **waiting**.

---

## Example B — TOOLKIT card (House Prices)

The `toolkit` gate. We propose the **families** we'll grow the solution tree
from. Per the tree rules in `CLAUDE.md`, the family choice **seeds the root
branches** — branch A and branch B are genuinely different approaches, not two
tweaks of one. Keeping ≥2 families alive is a standing rule, so we name two
deliberately-different starting points up front.

```
📋 toolkit
What's going on:   Picking the kinds of model we'll try first. We'll grow two
                   different "families" side by side so we're never stuck betting
                   on one idea.
Found / propose:   • Branch A — gradient-boosted decision trees. These are the
                     default winner on spreadsheet-style data like this (mixed
                     numbers + categories, missing values, no images/text).
                     Concretely: LightGBM first, with XGBoost and CatBoost as
                     siblings, plus plain scikit-learn for the data prep. Needs
                     `uv add lightgbm xgboost catboost scikit-learn`.
                   • Branch B (deliberately different) — a regularized linear
                     model: scikit-learn's ElasticNet on log-price, after one-hot
                     encoding the categories and scaling the numbers. It makes
                     very different mistakes from trees, which makes a later blend
                     of A + B stronger than either alone. Needs only
                     scikit-learn (already a dep).
                   • Both branches train and score on log(SalePrice) to match the
                     official metric exactly.
Why:               Trees usually win this dataset; a linear model is the cheapest
                   "structurally different" second family to keep alive, and the
                   two disagree in useful ways.
Cost:              ~1 min to add libraries · no training yet · 0 submissions.
Your call:         [Approve] [Change something] [Skip] [Tell me more]
Autonomy: auto_except_submit — proceeding
```

Notes:
- Branch A and Branch B are **different families**, exactly as the toolkit gate
  is supposed to seed (`draft` operators off the root). LightGBM / XGBoost /
  CatBoost are siblings *within* A (they're all boosted trees), so swapping among
  them later is an `improve`, while jumping A→B is a `draft`.
- Branch B is chosen for *complementarity*, stated in plain words ("makes
  different mistakes") — that's the reason a blend helps, not jargon.
- Library installs are shown as `uv add …`, never a global pin (hard rule #1).
- Shown here in `auto_except_submit`, so `toolkit` is **proceeding** without a
  pause — the card is still emitted for the record, but the loop continues.

---

## Example C — SUBMIT card (House Prices)

The `submit` gate. This is the only irreversible, rate-limited, public action, so
it stays a human gate in every mode except `full_auto`. The card's job is to show
that (1) the file is valid, (2) this node actually beats what we last submitted
by more than fold-noise, and (3) we can afford the slot.

```
📋 submit
What's going on:   One of our models clearly beat the last one we sent in. Asking
                   before I spend one of today's 5 submissions on it.
Found / propose:   • Candidate: node_0012 (LightGBM + target-encoded
                     neighborhood). Local cross-validation: 0.1208 ± 0.0017
                     (RMSE on log-price, lower is better).
                   • Last thing we submitted: node_0007 at 0.1271 local. So this
                     is a 0.0063 improvement — about 3.7× the ±0.0017 fold-noise,
                     so it's real, not luck.
                   • File data/.../node_0012/submission.csv passed the format
                     check: columns Id,SalePrice, 1459 rows, no blanks, ids match
                     sample_submission.csv.
                   • Leakage self-check: clean (fit-inside-fold, target,
                     id/order all pass).
Why:               It clears local CV by well over fold-noise and the file is
                   valid and leak-clean — the bar for spending a slot.
Cost:              ~10s to upload + poll for the public score · no training ·
                   1 of 5 submissions today (currently 2/5 used → would be 3/5).
Your call:         [Approve] [Change something] [Skip] [Tell me more]
Autonomy: auto_except_submit — waiting
```

Behind the card, before it's shown, these already ran (and their outputs are what
the bullets quote):
```bash
# format gate — never spend a slot on a malformed file
uv run tools/validate_submission.py \
    --submission comps/house-prices.../nodes/node_0012/submission.csv \
    --sample    comps/house-prices.../data/sample_submission.csv \
    --id Id

# budget gate — derived from the ledger, resets 00:00 UTC
uv run tools/kaggle_io.py budget --ledger comps/house-prices.../submissions.md
```
On `[Approve]`, the actual send + async poll:
```bash
uv run tools/kaggle_io.py submit house-prices-advanced-regression-techniques \
    --file comps/house-prices.../nodes/node_0012/submission.csv \
    --message "node_0012 lgbm+tgt-enc cv=0.1208"
# then append one UTC-timestamped row to submissions.md and poll:
uv run tools/kaggle_io.py submissions house-prices-advanced-regression-techniques
```

Notes:
- **CV vs last submitted**, not CV vs the public board, is the decision rule
  (trust the CV; the public LB is a noisy slice — `CLAUDE.md` rule #6). A slot is
  only spent on a node that beats the *last submitted CV* by more than fold-noise.
- The improvement is quoted in **fold-noise units** ("3.7× the ±0.0017"), which
  is how a non-specialist can tell a real gain from luck.
- The Cost line names the exact slot math (`2/5 → 3/5`), pulled from the budget
  reader, not remembered.
- A server-rejected submission does **not** burn the quota, so on a 4xx the slot
  is not counted and we can safely retry after fixing.

---

## Autonomy dial — behavior at each gate

The three modes, what each pauses at, and how the human flips the dial by voice are
all defined in CLAUDE.md's "Autonomy dial" table — see there. The dial decides
whether each card above **waits** for the human or is emitted **for the record** and
**proceeds**.
