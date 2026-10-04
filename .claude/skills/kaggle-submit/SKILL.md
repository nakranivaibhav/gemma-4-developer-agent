---
name: kaggle-submit
description: "Spend a submission slot. Handles BOTH competition types: a plain CSV upload, and the notebooks-only kind where Kaggle reruns your notebook against a hidden test set with internet disabled — ship fitted state as a Dataset, feature-engineer + infer on the test set in-kernel, never retrain. Use when a valid node is ready to go to the leaderboard, or a stage reaches the `submit` gate."
argument-hint: <slug> <node_id>   e.g. titanic node_0007
allowed-tools: Bash, Read, Write, Edit
---

# /kaggle-submit — spend one slot, correctly for the competition's type

Resolve `<slug>` and `<node_id>` from args (no node id ⇒ the champion). All paths
repo-relative. `DATE=$(date -u +%Y-%m-%dT%H:%MZ)` — never type a date.

## 0 · The reflex checklist (do it, don't narrate it)

- Node is `status: valid` with non-null `cv` — a node that hasn't cleared the leakage
  self-checks can never be submitted (hard rule 3).
- `uv run tools/validate_submission.py --submission … --sample … --id "$(grep -E '^id_col:' comps/<slug>/spec.md | awk '{print $2}')"` — a malformed CSV wastes a slot.
- Budget, derived not stored:
  `uv run tools/kaggle_io.py budget --ledger comps/<slug>/journal.md --limit "$(grep -oP 'daily_submission_limit:\s*\K\d+' comps/<slug>/spec.md)"`. Zero remaining ⇒ block, say when 00:00 UTC frees the next.
- CV gate: beat the last `SUBMIT` line's `cv=` by more than 2·sem, in the metric's
  improving direction. Never spend a slot to A/B on the LB (hard rule 6). Carve-outs,
  named in the card: final-ensemble submits, and a human-directed `PROBE`.
- Gate the human per `control.md` (`interactive` / `auto_except_submit` wait at the
  SUBMIT card + `.waiting-on-human`; `full_auto` proceeds). Surface a cv-too-good jump
  before spending, regardless of mode.

## 1 · Which kind of competition is this? Decide BEFORE building anything

| kind | how you submit |
|---|---|
| **file** | upload `submission.csv` — `uv run tools/kaggle_io.py submit <slug> --file … --message …` |
| **notebooks-only** | §2 — a CSV upload can NEVER succeed, on any client version |

A `400 FAILED_PRECONDITION` on a file upload IS the notebooks-only signal
(`tools/kaggle_io.py classify-error` maps it to `notebooks_only`). It burns no quota.
`spec.md` should record the kind at kaggle-start; if it doesn't, one rejected upload
tells you for free.

## 2 · Notebooks-only: ship STATE, never predictions

**The thing that breaks naive attempts:** Kaggle reruns your notebook against a
**hidden test set** that is larger/smaller/different from the public slice. So stored
per-row predictions are worthless — the hidden rows are new. What ships fine is trained
**state**. Only feature engineering and inference must happen in-kernel.

**Never retrain in the kernel.** It is the difference between a ~40-minute rerun and a
timeout, and a refit inside the kernel is unverifiable — you cannot see its numbers.

### 2a · Export the fitted state to a private Dataset
Every model, imputer, encoder, scaler, meta, plus the feature/clean modules the pickles
need, into one dir with `dataset-metadata.json` (`{title, id: "<user>/<name>", licenses}`):
```bash
uv run --no-sync python -m kaggle datasets create  -p <node>/dataset --dir-mode zip   # first time
uv run --no-sync python -m kaggle datasets version -p <node>/dataset -m "<what>" --dir-mode zip
uv run --no-sync python -m kaggle datasets status <user>/<name>                        # -> ready
```
Attach several datasets when state splits across nodes — locate each **by content**
in-kernel (a marker filename), never by a hard-coded mount path.

### 2b · One serve core, embedded verbatim
Put the whole inference path in ONE module (`serve_core<NNNN>.py`): load artifacts →
per-test-well features → predict → write `submission.csv`. Embed it into the notebook
**base64** (raw-string embedding breaks on backslashes) and write it out before import,
so the bytes Kaggle runs are the bytes you verified. Assert loudly at load: feature-list
agreement, model feature counts, round counts, no `.fit`/`.train` anywhere.

### 2c · Verify locally BEFORE pushing — and know your noise floor
Run the serve core against the local test slice and diff **layer by layer**, tightest
first. The lesson that cost this repo a round: an end-to-end diff can be legitimately
loose (an unseeded RNG in feature-building, non-bit-reproducible GPU fits), so a loose
end-to-end number proves nothing on its own — **isolate the deterministic channels and
require them bit-exact**, then check the end-to-end residual sits at the *already
established* floor rather than assuming it's fine.
- feed cached/stateless inputs ⇒ each deterministic channel must reproduce its
  reference to ~1e-12 (or exactly 0.0);
- compare the end-to-end residual against a PRIOR verified run's residual, not against zero.

### 2d · The kernel contract
`kernel-metadata.json`: `is_private: true`, `competition_sources: ["<slug>"]`,
`enable_internet: false`, `enable_gpu` only if inference needs it, `dataset_sources: [...]`.
The notebook must:
- resolve the input tree by walking `/kaggle/input` (mounts move);
- wrap every stage so a failure prints a traceback and the run continues;
- **always emit a valid `submission.csv`** — per-item try/except plus an unconditional
  final fallback stage. A hidden-rerun exception yields an empty score and a dead slot;
- print a heartbeat with a live ETA, and the per-item cost.

Project the hidden cost from the public pass: measure s/item locally, apply the
measured local→Kaggle penalty (this repo has seen 2.7×–4.8× on 4 cores), multiply by the
estimated hidden item count.

### 2e · Push, read the public pass, THEN submit
```bash
set -a && . ./.env && set +a && : "${KAGGLE_KEY:=$KAGGLE_TOKEN}" && export KAGGLE_KEY
uv run <node>/kernel/build_<node>_kernel.py                  # regenerate from the serve core
uv run --no-sync python -m kaggle kernels push   -p <node>/kernel
uv run --no-sync python -m kaggle kernels status <user>/<kernel-slug>   # RUNNING -> COMPLETE
uv run --no-sync python -m kaggle kernels output <user>/<kernel-slug> -p <node>/kernel_output
```
**Read the log before spending the slot**: every stage `ok`, row count == sample
submission, 0 missing, 0 fallback, sane value range, the shipped state actually mounted.
Then submit the KERNEL — `kernel_version` MUST be explicit (`None` returns a 403 that is
*not* the rules/verification 403; it costs no quota):
```bash
uv run --no-sync python - <<'PY'
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi(); api.authenticate()
print(api.competition_submit_code(file_name="submission.csv", message="<node cv=…>",
      competition="<slug>", kernel="<user>/<kernel-slug>", kernel_version=<N>))
PY
```

**Hard rule 9 still binds:** the kernel is a REGISTERED node (op `improve`, family
`serving`, parents = the node whose function it serves), with `cv`/`sem`/`folds`
inherited and no new metric. Register it before it exists on Kaggle.

## 3 · Mark it, after the fact (artifact-then-mark)

Only once the submit returned exit 0 / a ref:
```bash
printf '%s  SUBMIT %s cv=%s lb=%s — %s\n' "$DATE" "$node" "$cv" "${lb:-pending}" "$note" >> comps/<slug>/journal.md
```
Scoring is async and a notebooks-only rerun takes as long as the rerun takes — poll
`competitions submissions` detached with a marker-waiter, spaced, never tight.
**Match the status on the row carrying YOUR ref** — a listing-wide `grep COMPLETE`
matches some older submission's row and fires the marker instantly (this bug has fired
here once). When the score lands, append (never edit) the backfill and set `node.md`'s
`lb`, then re-render:
```bash
printf '%s  LB %s lb=%s — async score landed\n' "$DATE" "$node" "$lb" >> comps/<slug>/journal.md
uv run tools/render_state.py comps/<slug>
```
Log a CV↔LB gap as a **diagnostic** — never an auto-demote (hard rule 6). Submitting
does not re-rank the champion; `PROMOTE` lines do.

## Errors worth classifying, not retrying
`uv run tools/kaggle_io.py classify-error --text "<stderr>"` →
`notebooks_only` (see §2) · `rules_not_accepted` **403 = accept rules / phone-verify in
the browser, NOT bad creds** — the #1 misdiagnosis · `auth` = env vars · `rate_limited`
429 = already backed off. A server-rejected submission does **not** burn quota.

## SUBMIT Decision Card
```
📋 submit
What's going on:   Node <id> (<one-line change>) beats the last submitted CV — spending a slot.
Found / propose:   • CV <cv> ± <sem> vs last submitted <last_cv> (Δ <d>, > 2·sem)
                   • <used>/<lim> used today, <remaining> left (resets 00:00 UTC)
                   • <file validates | kernel public pass clean: stages ok, N rows, 0 fallback>
                   • <deadline> — <days_left> days left
Why:               CV cleared the fold-noise band; the LB is the OOD check, not the selector.
Cost:              <~mins> · <no compute | one kernel rerun> · 1 of the daily <lim>
Your call:         [Approve] [Change something] [Skip] [Tell me more]
Autonomy: <mode> — <waiting | proceeding>
```
