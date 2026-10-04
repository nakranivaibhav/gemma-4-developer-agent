# grandmaster

A **human-in-the-loop, markdown-driven Kaggle solver that runs inside Claude
Code.** You paste a competition link; Claude takes it from understanding the
problem all the way to submissions — pausing at the right moments for you and
grinding autonomously in between.

There is almost no application here. The "program" is **markdown**: a standing
playbook ([`CLAUDE.md`](CLAUDE.md)), per-stage **skills**, parallel **subagents**,
and one **workflow** for the proposer↔critic refinement loop. The only real code is a thin
`tools/` of reusable `uv run` scripts (folds, Kaggle I/O, submission validation);
leakage is the `kaggle-developer` subagent's own fast self-check, not a tool.
Everything competition-specific — the data pipeline, the
features, the models — is **bootstrapped per competition** into `comps/<slug>/`,
never pinned globally.

[`CLAUDE.md`](CLAUDE.md) is the full playbook (autonomy dial, Decision Card
format, stage flow, graph semantics, leakage discipline, resume model, budget
rules). This README is the practical front door.

---

## 1. One-time setup

```bash
uv sync                 # install the tools/ deps (pandas, numpy, sklearn)
uv add kaggle           # the Kaggle CLI, used by tools/kaggle_io.py

cp .env.example .env     # then open .env and paste in your Kaggle creds
```

Your **`.env`** (git-ignored, never committed) holds two values from
kaggle.com → **Settings → "Create New Token"**:

```
KAGGLE_USERNAME=your-handle
KAGGLE_KEY=your-api-key
```

Load it before running — export into your shell, or pass it to `uv`:

```bash
export $(grep -v '^#' .env | xargs)                    # load creds into the shell, OR
uv run --env-file .env tools/kaggle_io.py --selftest   # load per-command
```
`--selftest` checks auth handling + the budget reader.

### Two human gates that cannot be automated

Per competition, you (the human) must do these in a browser — Claude will surface
them and **stop**, it will not retry around them:

1. **Accept the competition rules** on the competition page.
2. **Phone-verify** your Kaggle account (required for GPU/internet on kernels).

If either is missing, downloads and submits return **403** — which means "rules
not accepted / unverified," *not* bad credentials (the #1 misdiagnosis).
`uv run tools/kaggle_io.py classify-error --text "<error>"` maps it for you.

---

## 2. Quickstart

Inside a Claude Code session in this repo, **just paste the competition link** —
Claude reads [`CLAUDE.md`](CLAUDE.md) and drives the whole pipeline itself (start →
eda → validate → baseline → experiment → final), pausing only at the gated
**Decision Cards** per the autonomy dial. You do **not** need to type the
`/kaggle-*` commands; they're listed here only because you can also run or re-run a
single stage by hand.

```
/kaggle-start <competition-url-or-slug>   # stage 0 — bootstrap comps/<slug>/, spec.md, download data
/kaggle-eda                               # stage 1 — understand + clean the data
/kaggle-validate                          # stage 2 — freeze the CV (folds.json) + holdout
/kaggle-baseline                          # stage 3 — dumb baseline → first submission → champion/
/kaggle-experiment                        # stage 4 — the experiment loop: propose (proposer↔critic) → build EVERY proposal → gate → decide
```

Helpers, any time:

```
/kaggle-submit     # budget-gated submit + async poll for the public score
/kaggle-status     # plain-language readout of where everything stands (read-only)
/kaggle-study      # learn a top solution: a 3D page + a stage-by-stage chat walk + an agreed lessons log
```

### Autonomy dial — flip it by voice

Stored in `comps/<slug>/config.md`; change it any time by just saying so.

| mode | pauses at | use when |
|---|---|---|
| `interactive` (default) | **every** gate | new comp, learning the data |
| `auto_except_submit` | only `understand` + `submit` | the experiment grind |
| `full_auto` | nothing | walk away |

Say **"go auto"** / **"ask me before submitting"** / **"pause"** and Claude
updates `config.md`. `understand` and `submit` stay human except in `full_auto`:
a misread metric poisons everything downstream, and a real submission is the only
irreversible, rate-limited, public action.

---

## 3. Repo layout

```
grandmaster/
  CLAUDE.md                       # the full operating playbook (read this for the rules)
  README.md                       # this file
  pyproject.toml  uv.lock         # tools/ deps only (modelling libs added per-comp)
  tools/                          # the thin reusable uv-run scripts (the only real code)
    make_folds.py                 #   leak-correct CV scheme → folds.json (group/temporal safety baked into the split)
    kaggle_io.py                  #   download / submit / submissions / budget / classify-error
    validate_submission.py        #   shape/columns check vs sample_submission
                                  #   (leakage has no tool — kaggle-developer self-checks each node, see §4)
  .claude/
    skills/                       # the per-stage procedures (one folder per slash command)
      kaggle-start  kaggle-eda  kaggle-validate  kaggle-baseline
      kaggle-experiment  kaggle-submit  kaggle-status  kaggle-io  kaggle-leakage
      kaggle-study
    agents/                       # parallel workers (fresh context, can't pause)
      kaggle-proposer.md          #   proposes the next experiments; writes node records once confirmed
      kaggle-proposal-reviewer.md #   critiques the proposals before any code is written
      kaggle-developer.md         #   builds AND self-gates one node in isolation (fast leak self-checks, voids on leak)
      kaggle-eda-explorer.md      #   fans out EDA probes
                                  # (the proposer↔critic refinement loop runs through disk
                                  #  contracts in comps/<slug>/rounds/ — no workflow script)
  comps/                          # one folder per competition (data gitignored)
    <slug>/ …                     # see below
```

### Per-competition layout — `comps/<slug>/`

Everything is markdown except the data and the frozen fold indices.

```
comps/<slug>/
  progress.md      # MACRO resume: setup checklist + stage checkboxes + derived date/budget/deadline header
  spec.md          # the contract — prose + a fenced machine block of key fields (metric, target, id, deadline)
  config.md        # autonomy mode
  eda.md           # findings + cleaning rationale (prose, no checkboxes)
  validation.md    # the frozen CV scheme + why it matches the official metric
  folds.json       # frozen fold indices (split-seed only)
  graph.md         # THE MAP: a Mermaid DAG of all nodes + a description table linking to each node.md
  data.md          # DATA LINEAGE: engineered feature-sets (raw→base→fs_*) + which nodes consume each
  journal.md       # append-only, one timestamped line per node
  rounds/          # the propose↔critic disk loop: round_NNNN/iter_N/{proposals.md, review.md, VERDICT}
  submissions.md   # append-only, UTC-timestamped ledger — the source of truth for budget
  champion/        # the best valid node's src/ + submission.csv + README (byte-copied, never symlinked)
  nodes/node_NNNN/ # node.md (THE node record: plan + metrics + gate booleans), src/, train.log,
                   #   submission.csv (leakage gate booleans live in node.md, written by the developer's self-check)
  data/            # downloaded + unzipped (gitignored)
```

---

## 4. Safety rails (in one paragraph)

The CV is **frozen once** at `/kaggle-validate` (`uv run tools/make_folds.py`
picks the leak-correct scheme — time/group/stratified/plain — with an inviolable
holdout) and never refit across folds; every transform is fit **inside the train
fold only**. Leakage is gated by the `kaggle-developer` subagent's own fast
self-checks on each node it builds (target/id not among the features, a quick
single-feature-vs-target correlation sweep, reading its own fold loop to confirm
fit-inside-fold, then OOF coverage / no-NaN / distribution / cv-too-good sanity —
seconds, never a training run; group & temporal safety come upstream from the frozen
folds) — a **gate, not a warning**: a node that leaks is **void**, no matter how good
its CV. Claude **trusts a well-built CV over the
public LB** — the LB is a small noisy slice, so a CV↔LB gap is surfaced as a
diagnostic, never auto-acted. The submission **budget and deadline are derived
from UTC timestamps** in `submissions.md` at read time (against the per-comp daily
limit recorded in spec.md, resets 00:00 UTC) so they can't drift across a resume;
dates always come from `date -u`, never memory. The whole run is **resumable**: macro state in `progress.md` checkboxes,
micro per-node state in the converged `node.md` record's **`stage`** field
(`proposed → built → scored → reviewed → decided → submitted`, advanced only after
its artifact exists — artifact-then-mark) — on restart, resume at the first
unchecked stage, rebuild the search frontier from the `graph.md` map, and continue
the in-progress node from its `stage`.

### Running unattended

By default Claude asks before each command. For a hands-off run, create a
**local, git-ignored** `.claude/settings.local.json` in your clone — it is specific
to this one repo and is never committed:

```jsonc
{ "permissions": { "defaultMode": "bypassPermissions" } }
```

That skips prompts the safe way: scoped to this repo only, not your whole machine.
The repo itself ships **no** permission config — you opt in locally.

---

**For the full rules — the Decision Card format, graph operators
(draft/improve/debug/combine), the complete leakage suite, the marker-file pattern
for long trainings, and the subagent/workflow split — read [`CLAUDE.md`](CLAUDE.md).**
