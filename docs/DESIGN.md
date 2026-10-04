# grandmaster — design & rationale

> Operating rules live in CLAUDE.md — this file is only the *why* (research lineage + rationale). Where this file used to restate rules, it now points.

This is the *why* behind the playbook. The standing contract is `CLAUDE.md`; the
procedures are the skills in `.claude/skills/`; the workers are the subagents in
`.claude/agents/`; the proposer↔critic refinement loop runs through disk contracts
in `comps/<slug>/rounds/`; the reusable code is in `tools/`. This file explains the research lineage we
borrowed from, why we run it as markdown inside Claude Code, the graph/gate/
validation/resume models, the mapping from our sibling neural-ring-detector loop,
and the honest expectations.

---

## 1 · Research lineage — what we took, and from where

Five papers shaped this system. We took one proven lever from each and left the
rest.

**AutoKaggle (arXiv:2410.20424)** — a multi-agent *phase pipeline*
(understand → EDA → engineering → model → submit) with a reader/planner/
developer/reviewer cast. Its two empirically-load-bearing parts are (a) a
**validated tools library** the agents call instead of re-deriving boilerplate,
and (b) **per-phase unit tests**: the paper reports its *completion* rate rising
from **0.10 → 0.85** once each phase had to pass its own tests before the next
began. We took both. Our `tools/` (make_folds, validate_submission, kaggle_io)
*is* the validated library; "every node — including data-cleaning and feature nodes
— clears the unit-test + leakage self-check before its CV counts" (CLAUDE.md) is the
per-phase-test discipline applied at node granularity (the leakage check is the
developer's fast self-gate, §5, not a standing tool). We did
**not** keep the rigid linear phase chain — see AIDE.

**AIDE (arXiv:2502.13138)** — a *single-agent best-first **solution-tree
search***. Each node is a complete solution; three operators grow the tree —
**draft** (a fresh approach), **improve** (one change on a working node), **debug**
(repair a broken one) — and the search greedily expands the best valid node.
The decisive idea: **history lives in the tree, not the context window**. The
agent never has to re-summarise its past; the tree *is* the memory, and a node's
parent edge records exactly what was tried. On MLE-bench, AIDE earned **~4× more
medals than the best linear (phase-pipeline) agent**. We took the **operators +
tree-as-memory** (§3), generalised from AIDE's tree to a **DAG** so a `combine` node
can merge several parents. We did **not** keep AIDE's *greedy best-first
expansion*: instead a `kaggle-proposer` agent proposes a small batch each round and
the orchestrator builds **every** confirmed proposal — simpler to supervise, and the
proposer (not a search controller) carries the policy for what to try next.

**MLE-bench (arXiv:2410.07095)** — the **medal yardstick**: 75 real Kaggle
competitions, scored against the actual private leaderboards, reporting
any-medal pass rates. Numbers worth internalising for expectation-setting:
o1-preview + AIDE scored **~16.9% any-medal pass@1, rising to ~34.1% pass@8**;
**Claude 3.5 Sonnet + AIDE ~7.6%**. The cost is real — a full sweep is roughly
**~1800 GPU-hours ≈ $3k per seed**. The honest reading: *autonomy medals on a
minority of competitions.* this is precisely why grandmaster is **human-in-the-loop**
— we spend the human's judgement at the few irreversible gates (metric reading,
real submissions) where being wrong is catastrophic, and let the machine grind
the reversible middle. We treat the official metric on a trustworthy local CV as
the target and the public LB as the out-of-distribution check (CLAUDE.md Mission).

**Agent K (arXiv:2411.03562)** — an **RL-free** agent with a **nested,
cross-competition memory** that accumulates structured experience and reaches
**Kaggle-grandmaster-equivalent** performance without policy-gradient training.
Lesson taken: durable memory > clever weights. Within a comp, our memory is the
append-only `journal.md` + `submissions.md` ledger + the `graph.md` node records; across
comps it is the shared, in-place-extended `tools/` library and these docs. We do
*not* yet carry a cross-comp case bank — that is the natural next increment, and
it would slot into the toolkit gate.

**DS-Agent (arXiv:2402.17453)** — **case-based reasoning**: *retrieve a relevant
past solution before proposing*. We honour the spirit at the toolkit/experiment-
plan gates: the first drafts seed from known-good recipes for the task type
(GBDT-on-tabular, Darts-for-time-series) rather than inventing from scratch, and
the library/family choice at the toolkit gate seeds the root branches. A full
retrieval bank is future work; for now the "cases" are the recipes encoded in the
skills and the operator's parents rule.

**One-line summary of the borrow:** *AutoKaggle's tests + tools, run over AIDE's
search (generalised to a DAG), measured by MLE-bench's medal yardstick, with Agent-K-style durable memory
and a DS-Agent retrieve-before-propose instinct — all wrapped in a human-gated
shell.*

---

## 2 · Why markdown-driven, inside Claude Code

The whole system is files a fresh Claude session can read and act on — no bespoke
runtime. Four Claude Code primitives map cleanly onto four roles:

| primitive | role here | example |
|---|---|---|
| **skill** (`.claude/skills/*/SKILL.md`) | a **procedure** the main session runs at a gate | `/kaggle-validate` freezes the CV |
| **subagent** (`.claude/agents/*.md`) | an **isolated worker** with fresh context | `kaggle-developer` builds one node |
| **CLAUDE.md** | the **standing rules** every actor obeys | leakage voids a score; one atomic change/node |
| **disk contract** (`comps/<slug>/rounds/round_NNNN/iter_N/`) | a **deterministic agent loop through files** | the propose↔critic refinement: proposer writes `proposals.md`, reviewer writes `review.md` + a one-word `VERDICT` marker; the orchestrator sequences spawns and reads only `VERDICT` |

The loop used to be a Claude-only `Workflow` script (`propose-loop.js`); moving it
to disk contracts made it resumable mid-refinement, auditable after the fact, and
portable to any agent runtime with subagents (e.g. Codex) — the orchestration is
just "spawn, then stat a marker file", the same pattern as the long-training
`DONE` markers.

**The key constraint that shapes everything: subagents cannot pause for a human —
only the main session can** (CLAUDE.md, "Operating mode").
A subagent runs to completion and returns.
So the architecture is **"gate the ends, auto the middle"**:

- Every **gate** (`understand · toolkit · eda · validation · experiment_plan ·
  submit`) lives in a **skill in the main session**, because only there can we
  render a Decision Card and *wait*.
- The **experiment grind** — propose → register → build every proposal → gate →
  decide — has no inherent need for a human mid-step, so it is delegated to
  subagents (the propose↔critic pair exchanging through the round dir's disk
  contract), sequenced by the main session as orchestrator.

That is why `understand` and `submit` stay human except in `full_auto`: a wrong
reading of the metric poisons every downstream CV, and a real submission is the
one irreversible, rate-limited, public action. The orchestrator reflects this — in
`auto_except_submit` it asks the human before spending a slot; in `full_auto` it
submits within budget.

### 2.5 · Event-sourced state + the Stop-hook driver (2026-07 refactor)

Two structural problems showed up in the first live comp (rogii, Jul 2026),
diagnosed from the session logs:

1. **Multi-file state drifted.** One node event needed coordinated edits in ~5
   places (node.md frontmatter, graph.md's Mermaid + table + header, data.md's
   consumed-by, progress.md's header) and the rules spent paragraphs policing
   agreement. Fix: **event-sourcing.** `journal.md` became the single written
   source of truth — one appended, structured line per event (grammar:
   `tools/render_state.py` docstring) — and everything readable is a *printout*:
   `tools/render_state.py` replays the journal into a generated `state.md`
   (header, checklists, Mermaid, node table, lineage, ledger). Append-only truth
   cannot drift; "exactly one champion" went from an invariant to police to an
   arithmetic consequence of replay. `control.md` (dial + halt) is the one file
   the human writes. graph/progress/data/submissions/outside .md files are
   retired.

2. **The human was the scheduler.** Sessions idled for 10+ hours waiting for a
   "?" — nothing woke the agent when a training finished or a round closed, and
   developer agents that waited on multi-hour runs died and orphaned them
   ("zombie" nodes). Fix, three parts: (a) `tools/autopilot_gate.py` registered
   as a **Stop lifecycle hook in both harnesses** (Claude Code and Codex ≥0.1xx
   share the hook schema) — at every turn-end it replays the journal and either
   blocks the stop with the next concrete step or lets the session idle
   legitimately (interactive dial, halt flag, `.waiting-on-human` sentinel,
   runs still in flight, or a 3-strike no-progress breaker); (b) **wake on run
   completion** — Claude Code launches watchdogged runs via harness-tracked
   background Bash (re-invokes on exit), Codex adds `--on-done` to the watchdog
   to nudge the idle session; (c) **sessions never wait on long runs** — the
   developer stops at `ready_to_run` + `run.sh`, the orchestrator launches
   detached and spawns a GATE job on the completion wake, eliminating the
   zombie class by construction.

---

## 3 · The graph model

The experiment graph (`graph.md`) is the search and the memory at once (the AIDE
idea, arXiv:2502.13138, generalised from a tree to a **DAG**). **Every node is one
atomic change** so every CV delta is attributable, and it **attaches to the
deepest ancestor(s) whose work it keeps** — most nodes have one parent, but a
`combine` node merges several. The four operators (draft / improve / debug /
combine) and their parent rules are defined in CLAUDE.md's "Experiment graph"
section — see there for the table.

- The **toolkit gate seeds the root drafts**: "use Darts" and "LightGBM on lag
  features" are two drafts off `root` — two families, not two experiments inside
  one branch (this is where DS-Agent's retrieve-before-propose lands).
- The **champion is the best *valid* node anywhere in the graph** — best CV under
  the official direction, leakage-clean. On promotion, byte-copy its `src/` +
  `submission.csv` into `champion/` (cp, never symlink); a rejected node leaves
  `champion/` untouched.
- **Keep ≥2 families alive.** If the best lineage hasn't beaten CV by more than
  one parent-SEM over **5 consecutive improves**, force a **draft** of a
  structurally different family — *pivot the architecture, don't keep tuning*.
  This is the anti-local-minimum rule; tuning that stalls is the signal to draft.
- Status, not checkboxes. Each node's `node.md` carries a `status` field (`proposed ·
  running · buggy · valid · champion · dead`) and a `stage` field (the lifecycle);
  `graph.md` is the Mermaid DAG + table that maps them. **The statuses are the
  search frontier**, so a restart rebuilds the frontier by reading `graph.md`.

**Search policy (operator selection each round).** The full policy lives in its
single home, `.claude/agents/kaggle-proposer.md` — that agent applies it to draft N
proposals each round, the `kaggle-proposal-reviewer` critiques them (the
refinement runs through the `rounds/` disk contract, max 3 iterations,
orchestrator-enforced), and the orchestrator builds **every**
confirmed proposal; there is no best-first frontier-expansion controller that picks
which to expand.

**Data lineage (`data.md`).** `graph.md` is *experiment* lineage (node → parent);
`data.md` is its *data* companion — the engineered feature-sets (`raw → base → fs_*`)
and which nodes consume each, with every node linking back via its `uses_data` field.
The load-bearing part is the per-set **leak-safety class**: `stateless` (row-wise, no
fit — reusable as-is) vs `fit_in_fold` (a fitted transform or a **cross-row** stat —
target-encode, scaler, kNN density, group aggregate — built inside the train fold
only). This is not just bookkeeping: a *label-free* cross-row feature fit on the
whole train leaks even though it never touches the label — a static check can't see
it — so the `fit_in_fold` leak-safety class is precisely what flags it. The proposer
reuses a feature-set before re-engineering one and the developer obeys the class,
building the reference from train-fold rows only.

---

## 4 · Human-in-the-loop gates, the autonomy dial, Decision Cards

Each stage ends with a **Decision Card** — a plain-language readout (what's going
on / found or propose / why / cost in time·compute·submissions-out-of-5 / your
call), written for a smart non-specialist. We never show in-chat thumbnails as
proof; we report numbers + file paths and let the human open files at full
resolution (a lesson carried straight from the neural-ring-detector memory —
in-chat thumbnails hide 10-px offsets).

The **autonomy dial** (stored in `config.md`, flipped any time by voice) decides
which gates actually pause — its three modes and what each pauses at are tabulated
in CLAUDE.md's "Autonomy dial" table.

`understand` and `submit` are the two gates kept human except in `full_auto`,
for the reasons in §2 (metric-poisoning, irreversibility). The dial is the single
control surface; the rest of the system reads it and behaves accordingly.

---

## 5 · Validation & leakage discipline

A leaky CV is worse than no CV — it *confidently* misleads. So validation is
frozen once and policed hard.

- **Freeze the CV once** (`/kaggle-validate` → `tools/make_folds.py`) and never
  refit across folds. `make_folds.py` picks the leak-correct scheme from the spec:
  `TimeSeriesSplit` if a time column, else `GroupKFold` if a group key, else
  `StratifiedKFold` for classification, else `KFold`. **The seed controls only the
  split.** Output is `folds.json` (positional integer indices). Carve an inviolable
  holdout never touched in training or feature-fit.
- **Fit inside the fold.** Every transform (scaler/encoder/imputer/target-encoder/
  selector) is fit on the train fold only — `make_folds.py` writes the indices and
  the node's harness loops them.
- **The developer self-gates leakage — there is no standing scanner tool.** There is
  no `tools/leakage_scan.py`; instead `kaggle-developer` runs a handful of fast
  data/output checks (seconds, never a training run) on the node it just built:
  target and id absent from the feature list; a quick single-feature-vs-target
  correlation sweep on a sample (nothing implausibly perfect); reading its own fold
  loop to confirm every transform / cross-row stat is fit inside each train fold
  only; then OOF coverage, no-NaN, and distribution sanity, and a cv-too-good
  judgment versus the parent. A leak still **VOIDs** the node's CV — the score does
  not count.
- **Group and temporal leakage are prevented upstream, not re-scanned per node.**
  `tools/make_folds.py` picks the group/time-correct split and the frozen
  `folds.json` keeps a group key from straddling folds and keeps lags/rolling
  past-only — so the developer's per-node self-checks don't have to re-derive them.
- **CV↔LB is a diagnostic, not a trigger.** A gap is *logged and surfaced*, never
  auto-acted; chasing the public LB causes private shake-up. **Trust a well-built
  CV over the public LB** (CLAUDE.md Hard rule 6). A submission slot only goes to
  a node that beats the last submitted CV by more than fold-noise — never to A/B
  on the LB.
- **Adversarial validation was dropped as a standing gate** — it is available
  only as a one-off diagnostic when a big unexplained CV↔LB gap appears, not run
  every node. The CV-too-good tripwire covers the routine "implausible jump" case
  and flags it for human eyes before a slot is spent.

This is AutoKaggle's per-phase-test lever (arXiv:2410.20424) applied at node
granularity: a feature that "improves CV" but fails fit-inside-fold is **buggy,
not good**, and its CV does not count.

---

## 6 · The resume model

A competition spans days and gets resumed, so state must survive a cold start
with zero in-context memory — which is the whole reason state lives in artifacts on
disk rather than the context window. The mechanics (the two resume surfaces, the
artifact-then-mark rule, and the per-node `stage` lifecycle) are specified in
CLAUDE.md's "Resume model" section — see there for the rule text.

**Why derived, never a stored counter.** Budget and deadline are *computed* from the
append-only `submissions.md` ledger at read time (the exact formulae are in CLAUDE.md
"Budget & deadline") precisely so they can't drift across a resume — a mutable
counter would. The deadline is surfaced when it shrinks but never triggers a wind-
down: there is no "finish" stage, the loop runs until the human stops it.

The canonical restart path is `/kaggle-status`: read `progress.md` → find the
in-progress stage → if experiments, read the `graph.md` map to rebuild the
frontier → open any `running` node's `node.md` and resume from its `stage`.

---

## 7 · Mapping from the sibling neural-ring-detector loop

grandmaster is the same autonomous-loop shape as the sibling neural-ring-detector
"AutoResearch" agent, retargeted from "best ring model" to "best Kaggle solution."
The concepts map one-to-one — which is why the operating discipline feels familiar:

| neural-ring-detector | grandmaster analogue |
|---|---|
| champion / challenger (best model so far vs the run) | **champion node** = best valid node in the graph; each new node is the challenger, A/B'd against its parent |
| per-experiment file isolation (`experiments/exp_NNN/src/`, self-contained) | per-node isolation (`comps/<slug>/nodes/node_NNNN/src/`, copy-parent-then-one-change); reusable code in `tools/` is extended in place, never forked |
| marker-file auto-polling (`[ -f $DONE ]`, never `pgrep -f` which self-matches) | identical marker-file pattern for long local trainings (`DONE=/tmp/<slug>_node_NNNN.done`; wait on `[ -f "$DONE" ]`), event-driven, no timers |
| `journal.md` (append-only, one dense line per experiment) | `journal.md` (append-only, one UTC line per node) + `submissions.md` ledger |
| `history.json` (champion record + experiments list) | the `graph.md` map + per-node `node.md` records + `champion/` byte-copy (the durable, resumable record) |
| visual_score ≥ 9.0 LLM-judge gate (Claude Vision rates eval cards) | **official-metric CV gate** + the human Decision-Card gate; the LLM-judge role is replaced by a *numeric* gate (the metric) plus *human* sign-off at submit, because Kaggle gives a real metric where ring-fit only had perceptual quality |
| "numeric metric is a proxy; visual quality is primary" | "well-built local CV is the target; public LB is the OOD check" — CV is the trustworthy signal, LB is the proxy that can mislead |
| `exp_NNN` ids, `experiments/` tree | `node_NNNN` ids, `graph.md` DAG with draft/improve/debug/combine operators |
| "silence is a bug — add prints and re-run" | same: a node that returns nothing/hangs gets log visibility, not a guess; `train.log` is tailed filtered for `cv=|Traceback|Error|Killed|OOM` |

The one substantive difference: ring-detection had **no ground-truth metric**, so
its gate was an LLM vision judge scoring perceptual ring-fit. Kaggle *does* have a
ground-truth metric, so the automatable gate is the numeric CV and the human is
reserved for the two judgement calls a number can't make — reading the metric
right (`understand`) and spending an irreversible public slot (`submit`).

---

## 8 · Known risks & honest expectations

- **Autonomy medals on a minority.** Per MLE-bench (arXiv:2410.07095), strong
  agent+scaffold combos clear any-medal on roughly **8–34%** of comps
  (Claude 3.5 Sonnet + AIDE ~7.6%; o1-preview + AIDE ~16.9% pass@1 → ~34.1%
  pass@8). Expect *competent, often-unplaced* runs, not routine gold. Set this
  expectation in the first Decision Card.
- **Compute & money are real.** A full bench sweep is ~1800 GPU-h ≈ $3k/seed
  (arXiv:2410.07095); a single comp is far less, but a stalled tree can burn time
  for nothing. The ≥2-families / pivot-after-5-stalled-improves rule is the
  circuit-breaker; the deadline-derived header is the hard stop.
- **The CV can still lie.** The developer's self-checks catch the common modes, but
  a novel leak (a sneaky group key, a time-ordered id) can slip a fast check. When CV
  looks too good, the cv-too-good tripwire — a developer judgment recorded in
  `node.md` versus the parent — forces human eyes before a slot is spent.
- **Public-LB temptation.** The single most common self-inflicted loss is chasing
  the public LB into a private shake-up. The discipline (trust CV, log-don't-act
  on the gap, slot only for a CV win) is a guardrail, not a guarantee — a human in
  `interactive` mode is the real safety net.
- **Pessimism contagion.** A fresh session's behavior is set by the artifacts it
  wakes up into, not by the rulebook — terminal verdicts ("ceiling reached", "all
  levers exhausted") written into the resume surfaces made later sessions give up
  early. In s6e6 such verdicts were declared four times and were wrong three: an
  under-built base, a ported public recipe, and a full-pool shrinkage stack each
  broke a declared "ceiling". The mitigation is structural, not exhortative: every
  comp artifact carries its format contract at its top (stamped by `kaggle-start`;
  narrative lives only in `journal.md`); dead ends are recorded as *scoped
  closures* — tried X, measured Y, reopen-if Z — with verdict vocabulary banned
  (hard rule 10); resumers read numbers as state and journal prose as the previous
  session's hypotheses; and `/kaggle-status` audits the contracts on every resume.
- **Two non-automatable Kaggle gates** (accept rules in the browser, phone-verify
  for GPU/internet) block downloads/submits and *cannot be retried around* — a
  403 means "rules not accepted / unverified," **not** bad creds (the #1
  misdiagnosis; `tools/kaggle_io.py classify-error` maps it). Surface them; don't
  loop.
- **No cross-comp case bank yet.** Agent K's nested memory (arXiv:2411.03562) and
  DS-Agent's retrieval (arXiv:2402.17453) are only partially realised — the
  "cases" are the recipes baked into skills and the in-place `tools/` library, not
  a queryable bank. That is the clearest next increment.
- **Single-agent search, supervised.** We deliberately run AIDE's search
  (arXiv:2502.13138, generalised to a DAG) as one human-gated search rather than an unbounded swarm.
  This trades raw throughput for attributable deltas and a human able to veto at
  the two gates that matter — the right trade when a wrong submission is public
  and rate-limited.
