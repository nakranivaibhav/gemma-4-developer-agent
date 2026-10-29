You are reviewing a coding task before it is used to train and test coding agents. You are NOT fixing the task.
Your job: decide whether the task is fair, by the rubric below, and back every "yes" with evidence.

Your working folder holds one task:
- `TASK.md`: the issue text, hints, and how the required tests are defined. Read it first.
- `REFERENCE_FIX.diff`: the fix the task's authors made.
- `TEST_CHANGES.diff`: the test changes; the required tests are defined in TASK.md.
- `REQUIRED_TESTS.txt` (when present): the required tests by name.
- `repo/`: the repository at the base commit, before the fix. Use it to check what a developer could infer from the
  existing code (naming conventions, existing messages, sibling functions, existing xfail tests, docs).

How to work:
1. Read TASK.md, then TEST_CHANGES.diff and REFERENCE_FIX.diff in full.
2. List, for yourself, everything the required tests depend on: names they import or call, routes, files, exact strings
   or structures they compare, behaviours they assert, skip conditions, versions, clocks, network.
3. For each dependency ask: would a competent developer who has read only the issue (and hints) and the repo produce
   exactly this? Search the repo (Grep/Glob) before deciding something is not inferable.
4. Then answer the rubric.

{RUBRIC}

Decision rules:
- Be strict on evidence, not on suspicion. "yes" needs the exact required-test line and what the issue says or omits.
- A hard task can still be fair. Difficulty alone is never a "yes" on R1-R6.
- A name or string counts as unstated only if a reasonable developer could have chosen a different one that the test
  would reject. If the repo's own pattern pins it down, it is inferable (answer no).
- R7 is no exactly when at least one of R1-R6 is a yes that would make a correct-by-the-issue fix fail.
- R4: a short issue (even a title) is clear enough when it names the symptom or the faulty behaviour and a developer
  can reproduce or locate it in the repo. But it is R4 yes when the most natural reading of the issue leads to a fix
  the tests reject (for example the title says to allow or remove something that a required test still expects).
  When the only gap is a name the tests need, answer R1; when it is exact text or structure, answer R2. Do not also
  answer R4 for that same gap.
- R6: when the tests load certificate or key files, look at their validity dates (the PEM text often states them in a
  comment, or the test's own fixture says how they were made).
- R5: a link to an outside issue, PR or discussion is not R5 by itself. It is R5 only when the tests need a fact that
  exists only behind that link or elsewhere outside the repo.
- R6: also check test data that can expire or depends on today's date: certificates, tokens, hard-coded dates, or
  "days ago" arithmetic. Today is {TODAY}. A certificate or token in the test data that has expired is R6 yes.

Answer with the JSON object only, matching the schema you were given.
