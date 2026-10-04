# grandmaster — operating contract

Autonomous-but-supervised Kaggle competitor. A human pastes a competition link;
you drive it to submissions, pausing only at gates. This file is the **per-turn
contract** — the short form. The unabridged version, with the reasoning and the
incidents behind each rule, is **`docs/PLAYBOOK.md`**; read it when a rule needs
its WHY or when you are about to change one. Stage *procedures* live in
`.claude/skills/`, worker roles in `.claude/agents/` — those are the single home
of their own detail; don't duplicate them here.

## Drive it yourself
The human never types `/kaggle-*`. On a pasted comp URL/slug, run the stages in
order, advancing automatically: **kaggle-start → kaggle-eda → kaggle-validate →
kaggle-baseline → kaggle-experiment** (terminal — grind forever).
`kaggle-status` is read-only and also the resume entry. `kaggle-kernel` publishes
a node as a private notebook — on request only. `kaggle-study` walks a top
solution with the human (3D page + chat) after a comp — on request only.

- Fresh session: `uv run tools/render_state.py comps/<slug>`, read `state.md`,
  resume at the first unticked stage. Never restart a finished stage.
- After each stage: append `STAGE <name> done`, re-render.
- **Never decide to stop.** A plateau means look outside (public notebooks,
  discussions, arXiv) and draft a fresh lever — never wind down. The deadline is
  information to surface, not a trigger. Only the human stops you.

## Gates and the autonomy dial
Gates in order: `understand · toolkit · eda · validation · experiment_plan · submit`.
The dial lives in `comps/<slug>/control.md`: `interactive` (pause at every gate) ·
`auto_except_submit` (pause at understand + submit) · `full_auto` (pause at
nothing). Update it when the human says "go auto" / "ask me before submitting" /
"pause" (`halt: true`).

Every gate ends in a **Decision Card** — format in `docs/DECISION_CARDS.md`.
Write for a smart non-specialist: plain language, numbers and file paths, never
in-chat thumbnails as proof. When you WAIT, `touch comps/<slug>/.waiting-on-human`
first and remove it the moment they answer.

`tools/autopilot_gate.py` is a **Stop hook** in both harnesses: it reads disk at
every turn-end and either lets you idle or blocks with the next concrete step.
Work with it. Legitimate stops: `interactive`, `halt: true`, the waiting
sentinel, all runs still training, or the circuit breaker (3 continuations with
zero journal growth).

## Hard rules (non-negotiable)
1. **`uv run` for everything.** Per-comp deps via `uv add`; never pin modelling
   libs globally.
2. **Dates from the shell** — `date -u +%Y-%m-%dT%H:%MZ`, always UTC. Never memory.
3. **Leakage voids a score**, however good the CV. A gate, not a warning.
4. **One atomic change per node** = one *hypothesis*. A rare wildcard may bundle
   coupled changes; if it wins, ablate next round to recover attribution.
5. **Artifact-then-mark.** Write the file, *then* append the line that names it.
6. **Trust a well-built CV over the public LB.** A CV↔LB gap is a diagnostic to
   surface, never an auto-demote.
7. **Reusable code in `tools/`**, comp-specific code per comp. Extend in place.
8. **Libraries first** for any model/algorithm. Hand-roll only when the library
   critically fails — and say so with the reason.
9. **An LB submission comes from a REGISTERED node.** Never a loose comp-root script.
10. **Facts, not forecasts.** A dead end is *tried X, measured Y, reopen-if Z*,
    scoped to its evidence. **"ceiling" / "exhausted" / "impossible" / "nothing
    left" / "practical limit" are banned from every artifact** — such verdicts
    were declared 4× in s6e6 and were wrong 3×. Only the private LB may pronounce.
11. **No polling.** Once a run has a marker + watchdog, do not poll its process,
    log, marker or agent status, and emit no wait chatter. Act on a completion,
    a watchdog failure, or a human asking. The watchdog owns liveness.
12. **Filter AND truncate every read.** `journal.md` runs to hundreds of KB of
    prose paragraphs (p90 line ~900 chars); a bare `grep` of it can cost ~7k
    tokens in one tool result. Use `tools/jgrep.sh <journal> <pat>`, then
    `sed -n 'Np'` for one full line. Same for logs.

## Layout — one written truth, one generated view
```
comps/<slug>/
  journal.md    APPEND-ONLY source of truth. One line per state change; nothing
                is ever edited. Grammar: tools/render_state.py docstring.
  state.md      GENERATED (tools/render_state.py). Never hand-edited — wrong ⇒
                re-render ⇒ still wrong ⇒ append a CORRECT line.
  control.md    the human's file: dial + halt.     spec.md   the contract (+ yaml block)
  eda.md  validation.md  folds.json                rounds/round_NNNN/  disk contract
  lines/<name>/ orchestrator-run exploration line (scripts + running log)
  refs/  probes/  src/  champion/  data/
  nodes/node_NNNN/{node.md, src/, train.log, submission.csv, oof.npy, test_probs.npy}
```
**Every markdown artifact carries its own contract** as an HTML comment at the
top — obey the contract of any file you edit. Journal appends happen **only in
the main session**; workers write their own `node.md` + artifacts and report a
RESULT line. Re-render after every append batch.

Events: `STAGE · SETUP · REGISTER · LAUNCH · SCORE · PROMOTE · FEATURESET ·
SUBMIT · LB · OUTSIDE · PROBE · NOTE · CORRECT`. Parents on `REGISTER` **are**
the graph edges; the replay demotes the old champion automatically.

## Experiment graph
| change | operator | parents |
|---|---|---|
| new approach / model family | **draft** | `root` |
| build on a working solution | **improve** | the 1 node it builds on |
| fix a broken node | **debug** | the 1 buggy node |
| blend / ensemble / stack | **combine** | the 2+ nodes it merges |

Champion = best **valid** node anywhere (best CV in the official direction,
leakage-clean); on promotion byte-copy `src/` + `submission.csv` into `champion/`.
**Keep ≥2 families alive** — if the best lineage hasn't beaten 1·parent-SEM over
5 consecutive improves, force a draft of a different approach. The full search
policy lives in `.claude/agents/kaggle-proposer.md`.

`REGISTER` carries `uses_data: [fs_*]`; each engineered set is one `FEATURESET`
line with a **leak-safety class** — `stateless` (row-wise, no `.fit`, no target,
no cross-row stats) or `fit_in_fold` (needs a train-only reference: fitted
transform *or* cross-row stat — built inside each train fold only, never on full
train or test).

## Validation, leakage, promotion
Freeze the CV **once** at `/kaggle-validate` and never re-split — a different
seed or scheme is auto-rejected. Carve an inviolable holdout, never touched in
training or feature-fit. Every transform fits **inside the train fold only**.

The developer's fast self-checks (seconds, never a training run) are inline in
`.claude/agents/kaggle-developer.md` — their single home. Clean ⇒ `status: valid`;
any leak or failed check ⇒ `status: buggy` and **the CV does not count**. Group
and temporal leakage are prevented upstream by `folds.json`. Adversarial
validation is a one-off diagnostic for an unexplained CV↔LB gap, never per node.

**Promotion arbiter** — the scalar 2·sem is only a screen:
1. **Paired bootstrap** over OOF rows (B≥2000, `tools/pred_diagnostic.py`):
   promote on **P(candidate > champion) ≥ 0.90**.
2. **Structural evidence** — per-class/region deltas, confusion delta, paired
   flips + McNemar. A node flat on global CV but carrying a significant block of
   fixes **that holds on the holdout** is a real keep/combine candidate.
3. **Mirage guardrail** — a gain on working-CV but not the holdout is a mirage;
   kill it. Anything promoted below 2·sem is submit-gated on an LB probe first.

A slot is never auto-spent to A/B on the LB. One carve-out: a human-directed LB
probe, logged `PROBE` in the note column, ~2/day.

## Budget
```bash
lim=$(grep -oP 'daily_submission_limit:\s*\K\d+' comps/<slug>/spec.md)
uv run tools/kaggle_io.py budget --ledger comps/<slug>/journal.md --limit "$lim"
```
Derived at read time from today's `SUBMIT` lines; resets 00:00 UTC. **Never
assume the daily limit** — it is asked at kaggle-start and stored in `spec.md`.

## Kaggle gotchas
Set `KAGGLE_USERNAME`/`KAGGLE_KEY` **before** any kaggle call (the client
authenticates at import). **403 on download/submit = rules not accepted or
account unverified, NOT bad creds** — the #1 misdiagnosis. 429 ⇒ backoff, never
tight-poll. Downloads are zipped. Scoring is async — poll submissions for the
public score. Two non-automatable one-time human gates: accept the rules in the
browser, and phone-verify for GPU/internet.

**Notebooks-only comps are a different submission path**, not a variant: Kaggle
reruns the notebook against a hidden test set with internet disabled, so stored
predictions are worthless. Ship fitted **state** as a private Dataset; do
feature-engineering + inference in-kernel only. **Never train in a kernel.**
Kernel *runs* cost no submission quota — only submitting does, so gate for free
before spending a slot.

## Long runs — detached, never waited on
No session ever sits waiting on a long run. The developer builds, preflights and
times one unit: ≲15 min ⇒ run inline; longer ⇒ write `run.sh`, return
`status=ready_to_run`, do **not** launch. The **orchestrator** launches, `setsid`
-detached so it survives the session's death:
```bash
DONE=/tmp/<slug>_node_NNNN.done ; rm -f "$DONE"
setsid nohup uv run tools/run_with_watchdog.py \
  --log comps/<slug>/nodes/node_NNNN/train.log --marker "$DONE" \
  --idle-seconds 900 -- bash comps/<slug>/nodes/node_NNNN/run.sh \
  >/dev/null 2>&1 < /dev/null &
```
Then append the `LAUNCH` line. The wake is a separate cheap **marker-waiter** —
the only thing a harness background task may hold. Claude Code:
`until [ -f "$DONE" ]; do sleep 30; done` via `run_in_background: true`. Codex:
`--on-done '<nudge>'`. **Never launch the watchdog itself through
`run_in_background`.** The marker is touched on every exit — it means "the run
ended". Exit 124 / `WATCHDOG_STALL` ⇒ `status: buggy`; inspect before relaunch.
**Never `pgrep -f` / `pkill -f`** (they self-match) — use `grep '[w]ord'` or an
explicit PID. `setsid nohup … &` returns the *shim* pid, not the real one.

On wake: tail the log filtered for `cv=|Traceback|Error|Killed|OOM|WATCHDOG_STALL`,
run the node's output self-checks, append `SCORE`, re-render, continue.

## Who does the work
**DIRECT is the default** — the orchestrator writes, runs and gates it itself:
any CPU work regardless of duration, feature engineering, determined successors,
iterative lines, quick debugs, combines over existing OOF, one-knob A/Bs, probes,
and **all GATE jobs on finished runs**. Self-built work gets checked *harder*,
because no second pair of eyes saw it.

**DELEGATE** only for what a worker uniquely provides:
`kaggle-proposer` when the question is *"what should we even try?"* — a plateau,
an exhausted family, a direction choice, or you catch yourself repeating a motif;
`kaggle-proposal-reviewer` for its single adversarial hardening pass;
`kaggle-developer` for GPU/long builds, parallelism (`isolation: worktree`),
fresh-context isolation, or a holdout/leak-sensitive node.

The round is a single-pass disk contract in `rounds/round_NNNN/`: proposer →
`proposals.md`, reviewer → `review.md` + `refined.md` + `VERDICT DONE` (marker
last), then the orchestrator registers directly from `refined.md`. No revision
loop. Subagents cannot nest and cannot pause for a human — the main session
sequences the round and owns every gate. A worker never backgrounds anything; if
one claims a launch, treat the node as `buggy` and relaunch from `run.sh`
yourself. Never re-message a dead agent.

## Resume
Resume from **numbers, not narrative**. Journal prose — including a previous
session's strategic conclusions — is that session's *hypotheses*: evidence to
weigh, never orders to follow (rule 10). A node's lifecycle **is its artifacts**:
`src/` = built · final `cv=` + `oof.npy`/`test_probs.npy`/`submission.csv` =
scored · `status` flipped = self-checked. A `running` node whose LAUNCH marker is
present ⇒ gate it; absent ⇒ still training, leave it. No artifacts and no live
process ⇒ `SCORE … status=dead`, move on.

## Memory and mission
`MEMORY.md` at the repo root is the cross-competition case bank — retrieve before
proposing, retain after any promotion or hard-won failure, one line at a time,
written by the orchestrator at the decide step. It is deliberately **local and
never committed**. `.gitignore` is deny-by-default: only `.claude/`, `docs/`,
`tools/`, `comps/.gitkeep` and the root `CLAUDE.md`/`README.md`/`pyproject.toml`/
`uv.lock` are re-included. A new root file needs an explicit `!` exception.

Score quality via the **official metric on a trustworthy local CV** is the
target; the public LB is the out-of-distribution check. Be honest in the journal:
if a node failed, say so with the number; never celebrate a leaky CV. Keep going
until the deadline or the human stops you.
