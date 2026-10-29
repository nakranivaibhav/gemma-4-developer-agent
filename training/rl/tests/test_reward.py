import pytest

from training.rl import reward
from training.rl.tests.conftest import ledger_row


def oc(**kw):
    return reward.classify(ledger_row("r", "t", **kw))


def test_solved_inside_clock_is_one():
    assert oc(resolved_kaggle=True) == reward.Outcome(1.0, "solved")


def test_unsolved_is_zero():
    assert oc(resolved_kaggle=False) == reward.Outcome(0.0, "unsolved")


def test_cut_is_zero_even_if_tests_pass_on_the_cut_patch():
    # the clock stops the agent; the patch it had may still pass: resolved=True, resolved_kaggle=False
    o = reward.classify(ledger_row("r", "t", resolved_kaggle=False, cut=True, resolved=True,
                                   harness_error="Agent exceeded tool call budget (37 calls)"))
    assert o == reward.Outcome(0.0, "cut")


def test_cut_and_resolved_kaggle_together_is_a_bug():
    with pytest.raises(ValueError):
        oc(resolved_kaggle=True, cut=True)


def test_clock_off_raises_instead_of_scoring_zero():
    with pytest.raises(ValueError, match="Kaggle clock"):
        oc(resolved_kaggle=None)


@pytest.mark.parametrize("err,kind", [
    ("Agent completed execution without calling submit_patch.", "policy:no_submit"),
    ("Agent exceeded tool call budget (100 calls)", "policy:tool_budget"),
    ("Agent exceeded turns budget (500 turns)", "policy:turn_budget"),
    ("Sandbox execution error: litellm.ContextWindowExceededError: litellm.BadRequestError: x", "policy:context_overflow"),
    ("Missing JUnit XML report (possible premature os._exit(1))", "policy:pytest_exit"),
])
def test_policy_endings_score_zero(err, kind):
    assert oc(harness_error=err) == reward.Outcome(0.0, kind)


def test_policy_ending_keeps_a_real_solve():
    # no submit_patch, but the workspace diff still resolved inside the clock: the verifier decides
    assert oc(resolved_kaggle=True, harness_error="Agent completed execution without calling submit_patch.").reward == 1.0


@pytest.mark.parametrize("err,kind", [
    ("Sandbox execution error: litellm.InternalServerError: InternalServerError: OpenAIException", "infra:server_error"),
    ("Sandbox execution error: litellm.Timeout: APITimeoutError - Request timed out", "infra:request_timeout"),
    ("Sandbox execution error: litellm.BadRequestError: OpenAIException - No user query found", "infra:bad_request"),
    ("Evaluation error: UnixHTTPConnectionPool(host='localhost', port=None): Read timed out", "infra:docker"),
    ("Agent exceeded session timeout (240.0 min)", "infra:hang_guard"),
])
def test_infra_endings_are_excluded(err, kind):
    o = oc(harness_error=err)
    assert o.excluded and o.kind == kind


def test_infra_beats_resolved():
    # a server error mid-try makes the try unreliable even if it then resolved
    assert oc(resolved_kaggle=True, harness_error="litellm.InternalServerError: boom").excluded


def test_context_overflow_is_policy_not_bad_request():
    # ContextWindowExceededError text also contains BadRequestError: it must not be read as infra
    o = oc(harness_error="litellm.ContextWindowExceededError: litellm.BadRequestError: too long")
    assert o.kind == "policy:context_overflow" and o.reward == 0.0


def test_unknown_error_is_excluded_and_named():
    o = oc(harness_error="Something new went wrong")
    assert o.excluded and o.kind == "unknown_error"


def test_runner_crash_row_is_excluded():
    row = {"id": "t", "run_id": "r", "error": "AttributeError: 'NoneType' object has no attribute 'qsize'"}
    assert reward.classify(row).kind == "runner_crash"
