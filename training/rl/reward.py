"""Ledger row -> reward, or excluded.

Reward is 1 only when the verifier resolves the task inside the Kaggle clock (`resolved_kaggle`, set by
kits/swe-agent/run/kaggle_clock.py: resolved and not cut). A fail, a cut, an empty patch, or an ending the policy
caused (no submit, tool-call / turn budget, context overflow) is 0.

A try that ended on an infrastructure problem (server error, request timeout, Docker error, a crash in the runner,
the 240-min hang guard) is EXCLUDED: scoring it 0 would punish the policy for a broken machine. An error text not
in either list is also excluded, and counted, so a new failure kind is seen instead of silently scored.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# endings the policy caused: reward 0, the try is kept
POLICY_ERRORS = {
    "no_submit": r"^Agent completed execution without calling submit_patch",
    "tool_budget": r"^Agent exceeded tool call budget",
    "turn_budget": r"^Agent exceeded turns budget",
    "context_overflow": r"ContextWindowExceededError",
    "pytest_exit": r"^Missing JUnit XML report",   # the agent's patch makes the test run exit early
}
# endings caused by the machine: the try is excluded
INFRA_ERRORS = {
    "server_error": r"InternalServerError|ServiceUnavailable|APIConnectionError|Connection refused",
    "request_timeout": r"litellm\.Timeout|APITimeoutError|Request timed out",
    "bad_request": r"^(?!.*ContextWindowExceededError).*BadRequestError",   # context overflow is the policy's
    "docker": r"UnixHTTPConnectionPool|docker\.errors|Evaluation error",
    "hang_guard": r"^Agent exceeded session timeout",
}


@dataclass(frozen=True)
class Outcome:
    reward: float | None              # None = excluded
    kind: str                         # solved | unsolved | cut | policy:<x> | infra:<x> | runner_crash | unknown_error

    @property
    def excluded(self) -> bool:
        return self.reward is None


def classify(row: dict) -> Outcome:
    if "error" in row and "resolved" not in row:        # run_tasks.py caught an exception: no result at all
        return Outcome(None, "runner_crash")
    err = row.get("harness_error")
    if err:
        for name, pat in INFRA_ERRORS.items():
            if re.search(pat, err):
                return Outcome(None, f"infra:{name}")
    rk = row.get("resolved_kaggle")
    if rk is None:
        raise ValueError(f"{row.get('run_id')}/{row.get('id')}: no resolved_kaggle; RL rollouts need the Kaggle clock on")
    if row.get("kaggle_cut"):                         # the clock stops the agent via the tool budget: that is a cut
        if rk:
            raise ValueError(f"{row.get('run_id')}/{row.get('id')}: kaggle_cut and resolved_kaggle both true")
        return Outcome(0.0, "cut")
    if err:
        for name, pat in POLICY_ERRORS.items():
            if re.search(pat, err):
                return Outcome(1.0 if rk else 0.0, f"policy:{name}")
        return Outcome(None, "unknown_error")
    return Outcome(1.0, "solved") if rk else Outcome(0.0, "unsolved")
