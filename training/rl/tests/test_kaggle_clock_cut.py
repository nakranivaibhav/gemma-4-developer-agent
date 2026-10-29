"""The Kaggle clock must END the agent loop at the cut, not only refuse tool calls (rl_smoke1, 2026-10-26: a cut try
ran on for 500 LLM calls). Drives kaggle_clock with a fake harness loop shaped like swegemma's agent_runner:
asyncio.timeout(hang guard) around an ADK-style inner loop that keeps calling the model."""
import asyncio
import dataclasses
import sys
import types

import pytest

from training.rl.tests.conftest import ROOT

sys.path.insert(0, str(ROOT / "kits/swe-agent/run"))
import kaggle_clock  # noqa: E402

PROFILE = {"per_call_s": 0.0, "decode": [[1000, 100.0], [30000, 100.0]], "tool_s_per_call": 0.0}   # 100 tok/s flat


@dataclasses.dataclass(frozen=True)
class Budget:
    tool_calls: int | None = 100


class Ctx:
    def __init__(self):
        self.budget, self.tool_calls_used = Budget(), 0


def fake_runner_module():
    """A module object with its own `asyncio` attribute, like swegemma.harness.agent_runner."""
    return types.SimpleNamespace(asyncio=asyncio)


async def harness_loop(mod, clock, ctx, out_tokens, max_calls=500, guard_s=3600):
    """Each model call: report usage (as handle_adk_event does), then one tool call, then yield to the loop.
    Returns (model calls made, how it ended)."""
    calls = 0
    try:
        async with mod.asyncio.timeout(guard_s):
            while calls < max_calls:
                calls += 1
                clock.record_usage(input_tokens=1000, output_tokens=out_tokens)
                await asyncio.sleep(0)                       # the next await: an expired timeout fires here
                if ctx.tool_calls_used < ctx.budget.tool_calls:
                    ctx.tool_calls_used += 1
        return calls, "turn_limit"
    except TimeoutError:
        return calls, "timeout"


def run(mod, out_tokens, cap_s=330.0):
    ctx = Ctx()
    clock = kaggle_clock.KaggleClock(ctx, PROFILE, cap_s=cap_s)
    calls, how = asyncio.run(harness_loop(mod, clock, ctx, out_tokens))
    return clock, ctx, calls, how


def test_cut_ends_loop_on_the_crossing_call():
    mod = fake_runner_module()
    kaggle_clock.install_cut_hook(mod)
    clock, ctx, calls, how = run(mod, out_tokens=3000)     # 30 s per call: call 11 crosses 330 s
    assert how == "timeout"
    assert calls == 11
    assert clock.cut and clock.cut_stopped is True
    assert clock.cut_at_tool_calls == 10                     # the crossing call's tool call never ran
    assert ctx.tool_calls_used == 10
    row = clock.row(resolved=True)
    assert row["kaggle_cut"] and row["kaggle_cut_stopped"] and row["resolved_kaggle"] is False


def test_exactly_at_cap_is_a_cut():
    mod = fake_runner_module()
    kaggle_clock.install_cut_hook(mod)
    clock, _, calls, how = run(mod, out_tokens=3300)     # 33 s per call: 10 calls = 330.0 s exactly
    assert clock.cut and how == "timeout" and calls == 10


def test_under_cap_runs_to_the_end_without_cut():
    mod = fake_runner_module()
    kaggle_clock.install_cut_hook(mod)
    clock, _, calls, how = run(mod, out_tokens=1, cap_s=1e9)
    assert not clock.cut and clock.cut_stopped is None
    assert how == "turn_limit" and calls == 500


def test_without_hook_the_old_bug_shows():
    """Documents the bug the hook fixes: only the tool-call cap, so the loop runs to the turn limit."""
    mod = fake_runner_module()                                # no install_cut_hook
    clock, _, calls, how = run(mod, out_tokens=3000)
    assert clock.cut and clock.cut_stopped is False
    assert how == "turn_limit" and calls == 500


def test_hook_is_idempotent_and_passes_through():
    mod = fake_runner_module()
    kaggle_clock.install_cut_hook(mod)
    shim = mod.asyncio
    kaggle_clock.install_cut_hook(mod)
    assert mod.asyncio is shim
    assert mod.asyncio.sleep is asyncio.sleep and mod.asyncio.Lock is asyncio.Lock


def test_concurrent_tasks_cut_independently():
    """Two tasks at once (CONCURRENCY > 1): only the one over its cap is stopped."""
    mod = fake_runner_module()
    kaggle_clock.install_cut_hook(mod)

    async def both():
        a, b = Ctx(), Ctx()
        ca = kaggle_clock.KaggleClock(a, PROFILE, cap_s=330.0)
        cb = kaggle_clock.KaggleClock(b, PROFILE, cap_s=1e9)
        return await asyncio.gather(harness_loop(mod, ca, a, 3000), harness_loop(mod, cb, b, 3000)), ca, cb

    (ra, rb), ca, cb = asyncio.run(both())
    assert ra == (11, "timeout") and ca.cut
    assert rb == (500, "turn_limit") and not cb.cut


def test_record_usage_outside_event_loop_does_not_raise():
    ctx = Ctx()
    clock = kaggle_clock.KaggleClock(ctx, PROFILE, cap_s=1.0)
    clock.record_usage(input_tokens=1000, output_tokens=1000)
    assert clock.cut and clock.cut_stopped is False and ctx.budget.tool_calls == 0


@pytest.mark.skipif(not (ROOT / "envs/harness/bin/python").exists(), reason="envs/harness missing")
def test_real_agent_runner_uses_module_level_asyncio_timeout():
    """The hook relies on agent_runner calling `asyncio.timeout(...)` through its module attribute and catching
    TimeoutError. Guard against a harness upgrade silently changing that."""
    src = next((ROOT / "envs/harness/lib").glob("python3*/site-packages/swegemma/harness/agent_runner.py")).read_text()
    assert "import asyncio\n" in src
    assert "asyncio.timeout(timeout_sec)" in src
    assert "except (TimeoutError, LlmCallsLimitExceededError)" in src
