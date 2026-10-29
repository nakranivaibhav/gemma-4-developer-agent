<!-- judge_rubric.md contract
WHAT: the binary task-quality rubric shared by the answer key (tasks/judge_answer_key.jsonl) and the Opus judge
(training/judge/). One home for the item definitions; the judge prompt quotes this file, the answer key labels against
it. Change an item here first, then relabel the key and rerun calibration. Date every change.
-->
# Task-quality rubric (v1.1, 2026-10-24)

v1.1: R6 narrowed to failures a correct fix can actually hit; plain version-floor markers are no (answer-key review, 2026-10-24).

A task is: an issue text (plus hints), a repo at a base commit, a reference fix, and test changes. The **required
tests** are the tests that must pass after a fix: the FAIL_TO_PASS list when the task has one, otherwise every test
function the test changes add or modify (how the Kaggle scorer reads it). "A competent developer" means someone who
reads the issue and the repo carefully but has never seen the reference fix or the test changes.

Every item is **yes / no**, with evidence: the exact test line, and what the issue says or fails to say.

| id | item | yes when |
|---|---|---|
| R1 | `unstated_name` | A required test uses a name the fix introduces (function, class, method, attribute, parameter, module or file path, route path, CLI command, setting or env var) that the issue does not give and that a competent developer could not settle on from the issue and the repo's own conventions. One defensible alternative name that would fail the test is enough. |
| R2 | `exact_output` | A required test checks exact text or structure (error message, log line, rendered output, JSON keys, schema field names, HTML) that the issue does not specify and that does not follow from existing code or a standard. |
| R3 | `tests_beyond_issue` | A required test checks behaviour the issue does not ask for (a second feature, a refactor, a changed default bundled into the fix), so a fix that does exactly what the issue asks still fails. |
| R4 | `unclear_issue` | From the issue alone a competent developer cannot tell what is wrong or what correct behaviour looks like (no symptom, no expected result, or several readings that the tests tell apart). |
| R5 | `outside_knowledge` | The fix needs facts that are in neither the issue nor the repo (an external spec version, a third-party service's behaviour, data or credentials). |
| R6 | `env_fragile` | A correct fix could still fail because a required test depends on something outside the code: the date or clock, the network, randomness, file-system order, the platform, or output that differs by Python or package version; or two required tests need opposite configurations (e.g. one only runs on pydantic v1, the other only on v2), so one is always skipped and a skipped required test fails the task. A marker that only limits a test to a recent, ordinary setup (e.g. `needs_py310`, `needs_pydanticv2`) is **no**. |
| R7 | `solvable_as_written` | Overall: a competent developer with only the issue and the repo could write a fix that passes the required tests. **no** when any of R1-R6 is a blocking yes. |

Notes
- R1/R2 are about what the tests need, not about the fix's style: a fix may add many names; only names the
  required tests touch count.
- "Could not settle on" (R1) and "does not follow" (R2) are judged against the repo too: if the repo already has the
  exact pattern (e.g. sibling functions named `get_x_for_y`), the name is inferable and R1 is no.
- R7 is the training-data verdict. R1-R6 say why.
