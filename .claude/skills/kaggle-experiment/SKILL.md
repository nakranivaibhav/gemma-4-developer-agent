---
name: kaggle-experiment
description: "Stage 4 — the experiment loop: DIRECT-path inline nodes by default (the orchestrator writes, runs, gates, registers); on plateaus a single-pass delegated round (proposer ideas → reviewer's hardening pass → orchestrator registers from refined.md → build → gate → decide). Use when there's a frozen CV + baseline champion and the human says \"run experiments\" / \"/kaggle-experiment\" / \"improve the model\" / \"go auto\"."
argument-hint: "[interactive|auto] [--n-proposals N]"
allowed-tools: Bash, Read, Write, Edit, Agent, Skill
---

# kaggle-experiment — grind direct, delegate for perspective

You are the **orchestrator**. The DIRECT path is the default: you write the
code, run it, gate it, register it, and append every journal line yourself —
for any CPU work regardless of duration, including feature engineering (the
FEATURESET discipline, leak-safety classes and self-checks apply unchanged).
Delegate only for what a worker uniquely provides:

| worker | invoke for |
|---|---|
| **kaggle-proposer** | a FRESH PERSPECTIVE — plateau, family exhausted, choosing between directions, or you notice yourself repeating a motif. Returns free-form ideas, not specs. |
| **kaggle-proposal-reviewer** | ONE critical pass over the proposer's ideas — verifies numbers at source, simulates bars, then writes `refined.md`, the buildable version. No revision loop. |
| **kaggle-developer** | GPU/long builds · parallel nodes (`isolation: worktree`) · fresh-context isolation after your own failed variant · holdout/leak-sensitive nodes needing a second pair of eyes |

Read `CLAUDE.md` for the standing contract; this skill is the procedure.
**Self-built work gets checked harder, not softer** — writing the code yourself
never licenses skipping a gate a developer would have passed.

## 0 · Orient (every entry)
- `<slug>` from `comps/` (or the arg). `DATE=$(date -u +%Y-%m-%dT%H:%MZ)` — never type a date.
- Read `control.md` → mode. `auto_except_submit`/`full_auto` ⇒ **AUTO**; `interactive` ⇒ **MANUAL**.
- `uv run tools/render_state.py comps/<slug>` then read `state.md` (header, champion, node table, data lineage) + `spec.md`'s yaml machine block + the `journal.md` tail. Confirm `folds.json` + `champion/` exist (else run `/kaggle-validate` + `/kaggle-baseline` first).
- **Resume:** if a node is `running`, check its LAUNCH marker: present ⇒ the run ended, GATE it now (§4); absent ⇒ still training, leave it. Otherwise a node's artifacts are its lifecycle — continue at the **first missing one** (`src/` = built · `train.log` final `cv=` + `oof.npy`/`test_probs.npy`/`submission.csv` = scored · a journal `SCORE` line = self-checked/decided). A `running` node with no artifacts and no live process ⇒ append `SCORE <id> status=dead`, move on. A round dir with `proposals.md` but no `VERDICT` ⇒ spawn the reviewer; with `refined.md` + `VERDICT` ⇒ register (§2).
- **Work from disk, not recollection:** at the start of EVERY round, re-render and re-read `state.md` + the `journal.md` tail — never your memory of earlier rounds.
- **Numbers over narrative:** journal prose is *hypothesis, not state*; a closure binds only at its stated scope until its reopen-if triggers (hard rule 10).

## 1 · DIRECT path (the default loop)
While the next step is determined by the last measurement, just do it:
- register the node inline (§2's template), build it, run it (≲15 min inline;
  longer ⇒ write `run.sh` and launch `setsid`-detached per CLAUDE.md "Long local
  trainings"), gate it (§4's checks, run by you), append `SCORE`, decide (§6).
- probes stay probes (one `PROBE` line each, `probes/` dir, never nodes).
- an exploration line may own `lines/<name>/` with its own scripts + log.

Go delegated the moment the question becomes *"what should we even try?"*
rather than *"what does this measure?"* — or when you catch yourself proposing
the third variant of the same motif.

## 1b · DELEGATED round (single pass, no iterations)
1. **Bootstrap:** next round number from `ls comps/<slug>/rounds/` (max NNNN + 1,
   zero-padded; derived, never stored), `mkdir -p` the dir.
2. Spawn **kaggle-proposer** with `slug` + `round_dir` + `n_proposals` (a
   ceiling, default 3) + optionally the specific question you're stuck on. It
   writes `<round_dir>/proposals.md` (ideas: hypothesis + evidence + rough cost)
   and any propose-time probe scripts beside it.
3. Spawn **kaggle-proposal-reviewer** with `slug` + `round_dir`. It writes
   `review.md` (the critique), `refined.md` (the hardened, buildable specs +
   one-line DROPs + build order), then `VERDICT` (`DONE`) last.
4. Read `refined.md` and **register every spec in it** (§2). The reviewer
   prunes — you don't. A missing/malformed file = a failed subagent — respawn
   that role once, then stop and tell the human.
- **AUTO:** straight to §2/§3 in the reviewer's build order.
- **MANUAL:** render the Proposal Card from `refined.md`, `touch
  comps/<slug>/.waiting-on-human`, **wait**; register the accepted subset.

## 2 · REGISTER — you write the nodes (main session is the only journal writer)
For each spec (from `refined.md`, or your own DIRECT-path successor): reserve
the next zero-padded id (max in `state.md` + 1, re-render first if stale),
`mkdir -p comps/<slug>/nodes/node_NNNN/src`, write `node.md` — frontmatter
(`id · desc ≤8 words · op · parents · family · uses_data · status: proposed`,
`cv/sem/folds/lb` null) + a free-form plan body handing a fresh-context
developer the ONE atomic change, the hypothesis, the target, the gates with
directions, and every reference worth READING (never which files/functions to
write). A NEW feature-set's full recipe lives in the producing node's plan.
Then append, artifact-then-mark, `$DATE` from the shell:
- `$DATE  FEATURESET fs_<name> class=<stateless|fit_in_fold> from=<base|fs_x> producer=node_NNNN — <what>` (if new)
- `$DATE  REGISTER node_NNNN op=<op> parents=[…] family=<f> well=<well> uses_data=[…] round=<round|inline> — <desc>`
For a delegated round also append ONE `ROUND-OPEN round_NNNN nodes=[…] — <one-line rationale each>`. Re-render and check no RENDER ERRORS.

## 3 · BUILD — inline by default, developers for their niche
DIRECT-path nodes: build in the main session. Delegated GPU/parallel/isolation
nodes: spawn **kaggle-developer** (BUILD job) with the node dir (`node.md` IS
the spec), `spec.md`, `folds.json`, `parent_src`, metric+direction, and the
parent per-fold scores. Parallel independent nodes ⇒ one message, worktrees;
GPU nodes ⇒ serialize.
- **Short run (≲15 min projected):** run inline (either path), self-gate, score.
- **Long run:** the builder (you or the developer) writes
  `nodes/node_NNNN/run.sh` and STOPS (`ready_to_run`). YOU launch it
  `setsid`-detached through the watchdog (never via background-mode Bash); the
  wake is a marker-waiter (`until [ -f "$DONE" ]; do sleep 30; done`,
  background-mode Bash) or Codex `--on-done`. Append the `LAUNCH` line. While
  it trains: hard rule 11 — no polling, no chatter.

## 4 · GATE — on each run's completion wake (inline by default)
Marker present ⇒ tail `train.log` filtered for
`cv=|Traceback|Error|Killed|OOM|WATCHDOG_STALL`, then run the gate YOURSELF:
the node's output self-checks (OOF coverage — every train row exactly once, no
NaN; prediction-distribution sanity; submission schema via
`tools/validate_submission.py`; cv-too-good judgment vs parent), fill
`node.md`'s result fields, append `SCORE`. Spawn a developer GATE job only when
the node is holdout/leak-sensitive or the output checks themselves need fresh
eyes. Exit 124 / `WATCHDOG_STALL` / traceback ⇒ `status=buggy`; any
error-severity leak ⇒ `buggy` with `LEAK:` in the note (the CV does NOT count).

**Report contract (delegated builds):** every developer report ends with one
`RESULT` line (`RESULT node=… cv=… sem=… folds=[…] status=… runtime=… note=…`).
Carry ONLY that line into the round's state; on a mismatch with `node.md`,
trust `node.md` + the artifact and say so.

## 5 · SCORE — append the event
`$DATE  SCORE node_NNNN status=<s> cv=<f> sem=<f> folds=[…] — <note>`, re-render.
A `buggy` node's CV does not count.

## 6 · DECIDE — promote rule + the historian pass
**Promote rule (math, not judgment — canonical gate in CLAUDE.md "Budget &
deadline"):** leak-clean CV win beyond 2·sem promotes directly; anything closer
arbitrates via `tools/pred_diagnostic.py` paired bootstrap
**P(candidate > champion) ≥ 0.90** (report the bootstrap's own control p95 at
the same n where the statistic is known not to concentrate); a
McNemar-significant fix-block that holds on the holdout is keep/combine
material at flat CV. **Mirage guardrail:** a working-CV gain that fails the
holdout is killed. On promote: byte-copy `src/` + `submission.csv` →
`champion/`, update `champion/README`, append `PROMOTE` (the replay demotes the
old champion). On reject: touch nothing.

**The decide writes — ONE pass before anything else starts:**
1. `journal.md` — `NOTE`/`PROBE` lines worth keeping (scoped closures only;
   banned vocabulary stays banned), then `ROUND-CLOSE` for delegated rounds.
2. re-render; check no RENDER ERRORS and the expected champion.
3. `MEMORY.md` — write-on-event, one line, never batched later.

## 7 · SUBMIT (gated)
Submit only a node whose CV beats the **last submitted CV** by more than
fold-noise (2·sem) — never spend a slot to A/B on the LB. Validate + budget:
```bash
lim=$(grep -oP 'daily_submission_limit:\s*\K\d+' comps/<slug>/spec.md)
uv run tools/kaggle_io.py budget --ledger comps/<slug>/journal.md --limit "$lim"
```
- **MANUAL / `auto_except_submit`:** SUBMIT Decision Card, sentinel, **wait**;
  then `/kaggle-submit <slug> node_NNNN`.
- **`full_auto` + budget:** `/kaggle-submit <slug> node_NNNN`.

## Proposal Card (manual gate, from refined.md)
```
📋 experiment plan · <n> specs for <slug>
What's going on:   <one plain sentence on where the search stands>
Specs:             1. <op> <desc> — <why> (vs <parent> cv <x>)
                   2. …
Dropped:           <k> ideas, each with the number that killed it
Cost:              <~mins · cpu/gpu each>
Your call:         [Approve all] [Accept some] [Redirect] [Tell me more]
Autonomy: <mode> — waiting
```

## Invariants
- Build every spec in `refined.md` — the reviewer prunes, you don't.
- One atomic change per node; every CV delta is attributable.
- Leakage voids the score; a leaky node never promotes.
- Trust CV over the LB; a CV↔LB gap is a diagnostic, not an auto-demote.
- Artifact-then-mark; dates from `date -u`; all scripts via `uv run`.
- The main session is the only journal writer.
