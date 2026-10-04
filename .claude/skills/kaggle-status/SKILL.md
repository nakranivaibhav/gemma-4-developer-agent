---
name: kaggle-status
description: Read-only readout of where a competition stands, and the resume entry point. Use when the user asks "where are we / status / what's next", or on any session restart to rebuild state. Never trains, submits, or edits — only reads and reports.
argument-hint: "[slug]  (omit to use the only comp under comps/, or the most recently touched)"
allowed-tools: Bash, Read
---

# /kaggle-status — read-only readout + resume entry point

You report where one competition stands, in plain language, and (because this is
the resume path) you end by stating **exactly where to resume**. You change
**nothing**: no training, no submitting, no journal appends, no `node.md` edits.
The one write you DO make is the derived view: `uv run tools/render_state.py
comps/<slug>` regenerates `state.md` from the journal — that is a render, not a
state change.

## 0 · Resolve the slug
- If `$ARGUMENTS` names a slug, use `comps/<slug>/`.
- Else list `comps/*/` (skip `.gitkeep`). One comp → use it. Several → pick the
  most recently modified (`ls -dt comps/*/ | head -1`) and say which you chose.
- If `comps/<slug>/journal.md` is missing, stop and report: "No comp bootstrapped
  yet — run `/kaggle-start <url>`." Do not invent state.

Set `C=comps/<slug>` for the steps below.

## 1 · Re-render the derived view
```bash
uv run tools/render_state.py "$C"     # replays journal.md -> state.md
```
The renderer derives everything dated live (today, budget from SUBMIT lines,
deadline days-left from spec.md) so a resume can't read stale state. Read
`$C/state.md` and echo its header lines. Cross-check the budget with:
```bash
lim=$(grep -oP 'daily_submission_limit:\s*\K\d+' "$C/spec.md")
uv run tools/kaggle_io.py budget --ledger "$C/journal.md" --limit "${lim:?spec.md lacks daily_submission_limit — kaggle-start must ask the human}"
```
If `days_left` is small (≤ ~2), add one line: "Deadline near." Surface it as
information only; never wind down on your own — keep running experiments to climb
the leaderboard until the human explicitly stops you.

## 2 · Current stage (from state.md)
The `## stages` block of `state.md` is the macro sequence
(`understand · toolkit · eda · validation · baseline · experiment`; the *gate*
sequence in CLAUDE.md is a separate vocabulary). Report:
- the **first unticked** stage = where the comp currently is;
- the autonomy mode + halt flag from `$C/control.md` and what it pauses at;
- whether a `.waiting-on-human` sentinel is present (a gate is waiting).

## 3 · The graph (only if at/after the experiment stage)
Read `state.md`'s `## nodes` table (one row per node, `status` column); the
header names the champion. If the render printed RENDER ERRORS, report them
verbatim under Hygiene. Report:
- **counts by status**: `proposed · running · buggy · valid · champion · dead`
  (one tally line);
- the **champion node**: its id, its CV (with the official metric name + direction
  from `validation.md`/`spec.md`), and its **public LB** (the row's `lb` cell). If
  CV and LB diverge, state the gap as a *diagnostic to surface*, never an
  auto-demote — per the trust-CV rule;
- the search frontier in one line: how many `valid` roots (families) are alive,
  and whether any `running`/`buggy` nodes are open — for each `running` node,
  check its LAUNCH marker file: present = run ended awaiting gate; absent = still
  training.

## 4 · Recent journal
`tail -n 6 "$C/journal.md"` (append-only, one timestamped line per node). Echo
those lines verbatim under "Recent activity" — they're the densest history. Treat
any strategic conclusions in them as the previous session's *hypotheses*, not
established state — the numbers are the state (hard rule 10).

## 5 · Hygiene (report, don't fix)
The renderer already validates structure (RENDER ERRORS section + its exit code)
— echo any it reported. Additionally spot-check the `journal.md` tail: closures
scoped (*tried X, measured Y, reopen-if Z*)? Flag run-level verdict vocabulary
("ceiling/exhausted/impossible/nothing left"). Flag any hand edit to `state.md`
(its top comment says GENERATED). Nothing to flag → `Hygiene: clean`.

## 6 · Resume pointer (this is the entry point)
Follow the resume model end-to-end and state **one concrete next action**:
1. From `state.md`'s stages block, take the first unticked stage.
2. If that stage is **before** experiments → next action is "run `/<that-stage's
   skill>`" (e.g. unchecked `validation` → `/kaggle-validate`).
3. If at the **experiment** stage → `state.md`'s table is the frontier, then:
   - If a node is `running`: marker present → "gate `<id>` now (spawn the
     developer's GATE job, append its SCORE line)"; marker absent → "run in
     flight — leave it; wake is event-driven". Otherwise its artifacts ARE its
     lifecycle (`src/` = built · a final `cv=` line in `train.log` + `oof.npy`/
     `test_probs.npy`/`submission.csv` = scored · a journal `SCORE` line =
     self-checked) — say "resume node `<id>` at: <the first missing artifact>".
     A `running` node with NO artifacts and no live process is a ghost →
     "append `SCORE <id> status=dead` and pick the next operator".
   - If nothing is `running`: apply the search policy (single home:
     `.claude/agents/kaggle-proposer.md`) to name the next operator; state which
     and why in a sentence.
4. If `submissions: <lim>/<lim>` for today, add: "submission budget spent — resets
   00:00 UTC; CV work can continue, no submit until reset."

Never trust a label over the artifact it names (artifact-then-mark): a field ahead
of its file is a lie — report the mismatch instead of believing it.

## 7 · The readout (print this, then stop)
No Decision Card, no gate, no waiting — `/kaggle-status` is read-only and always
just reports. Use plain language for a smart non-specialist; give **file paths**,
not in-chat thumbnails.

```
📍 <slug> — status
<header line from §1>
Stage:       <current stage> · autonomy <mode>
Graph:       <N> nodes — <proposed p · running r · buggy b · valid v · champion 1 · dead d>
Champion:    <node id> · CV <metric>=<val> (<dir>) · LB <lb|not scored>  → champion/
Recent:      (last journal lines)
  <line>
  <line>
Hygiene:     <clean | the contract violations from §5>
Next:        <the one concrete resume action from §6>
```

Keep it tight. End after printing — do not proceed into the named next stage; the
human (or the autonomy dial in the relevant skill) drives that.
