"""Kaggle clock: a per-task budget in predicted Kaggle-seconds, so a local run is cut where the scorer's 4x L4 would cut it.

Local wall time means nothing (other GPU, several tasks at once, speculative decoding), so each model call is converted
into the seconds it would take on Kaggle, with the speed profile from kaggle_time_profiles.json and the same formula as
kaggle_time.py:

    call s  = per_call_s + output tokens / decode speed(context) + new prompt tokens / prefill speed
    task s  = sum of call s + tool_s_per_call * tool calls          (the ~30 s per-task setup is left out of the cap)

The running total only depends on token counts, so a run on any machine at any speed gives the same cut. When the total
passes the cap, the agent loop is ended at once: the clock expires the harness's own asyncio.timeout (the session-timeout
path a Kaggle wall kill takes), and caps the tool-call budget as a second stop. The patch so far is tested as usual, and the task is marked kaggle_cut: it counts as
unsolved in resolved_kaggle whatever the tests say, because on Kaggle that time would have been taken from other tasks.

STANDARD since 2026-10-23 (user: "we kill a run only if it exceeds the number given by our formula"): ON for every run.
The wall clock no longer ends a task; run_tasks.py sets the harness's max_time_minutes to a hang guard only
(WALL_GUARD_MINUTES, default 240). Env:
    KAGGLE_CLOCK         auto (default): profile from the agent's declared model, l4x4_<size>_nolora, or
                         l4x4_<size>_adapter for an agent with an adapter when that profile exists (31B, E2B); a profile name to count at another model's
                         speed (data runs of a teacher count at the STUDENT's speed, e.g. Qwen -> l4x4_e2b_nolora);
                         off = no clock (old behaviour: the eval_config wall limit ends tasks)
    KAGGLE_CLOCK_CAP_S   cap in Kaggle-seconds, default 330 (12 h / 120 tasks = 360 s, minus 30 s setup)
    KAGGLE_CLOCK_SPEED   speed factor; default 1.0, or 0.88 under auto for an E4B/12B agent with an adapter
                         (the 31B's LoRA penalty; a guess. E2B measured 2026-10-29 with the r64 SFT adapter on 4xL4:
                         0.65-0.73x of no-LoRA decode, so its 0.88 guess counted ~20% too little time)
Fixed 2026-10-26: before, only the tool-call budget was capped. The harness checks that budget only after ADK's
run_async returns, and ADK keeps looping inside one run_async while every tool call comes back BudgetExceeded, so a
model that kept calling tools ran on to the 500-turn limit (rl_smoke1: 4,878 s wall, 500 LLM calls, kaggle_s 4,655).
install_cut_hook(agent_runner_module) must be called once by the runner for the timeout stop to work.
Known approximation: the new-prompt-token count treats all agents of a task as one stream, so with sub-agents the small
prefill term can be a few seconds off. Output tokens and tool calls, which carry the time, are exact.
"""
import asyncio
import dataclasses
import os
import threading
import weakref

from kaggle_time import decode_rate, load_profiles


SIZES = ("31b", "12b", "e4b", "e2b")
_TIMEOUTS = weakref.WeakKeyDictionary()   # asyncio task -> the asyncio.Timeout the harness opened in it


class _AsyncioShim:
    """Stands in for the asyncio module inside the harness's agent runner: same module, but asyncio.timeout(...) also
    records the Timeout under the task that opened it, so a clock cut can expire it."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def timeout(self, delay):
        tm = self._real.timeout(delay)
        task = self._real.current_task()
        if task is not None:
            _TIMEOUTS[task] = tm
        return tm


def install_cut_hook(module):
    """Give a module (swegemma.harness.agent_runner) the asyncio shim. Idempotent."""
    if not isinstance(module.asyncio, _AsyncioShim):
        module.asyncio = _AsyncioShim(module.asyncio)


def expire_current_timeout() -> bool:
    """Expire the asyncio.Timeout of the running task now: its block raises TimeoutError at the next await.
    False when there is no running task or it has no recorded timeout (no hook, or no wall guard set)."""
    try:
        task = asyncio.current_task()
    except RuntimeError:          # not inside an event loop (e.g. called from a worker thread)
        return False
    tm = _TIMEOUTS.get(task) if task is not None else None
    if tm is None or tm.expired():
        return False
    tm.reschedule(asyncio.get_running_loop().time())
    return True


def auto_profile(model_name, has_adapter):
    """(profile name, default speed factor) for an agent's declared model; None for a model with no Kaggle profile."""
    size = next((s for s in SIZES if s in (model_name or "").lower()), None)
    if size is None:
        return None
    if has_adapter and f"l4x4_{size}_adapter" in load_profiles():   # measured LoRA speed (31B; E2B since 2026-10-29)
        return f"l4x4_{size}_adapter", 1.0
    return f"l4x4_{size}_nolora", (0.88 if has_adapter else 1.0)


def settings(model_name=None, has_adapter=False):
    """Env + agent -> (KaggleClock kwargs, the dict pinned into the run ledger). None when the clock is off."""
    name = os.environ.get("KAGGLE_CLOCK", "auto")
    if name == "off":
        return None
    speed_default = 1.0
    if name == "auto":
        auto = auto_profile(model_name, has_adapter)
        if auto is None:
            raise SystemExit(f"KAGGLE_CLOCK=auto: no Kaggle speed profile for model {model_name!r}. Set KAGGLE_CLOCK to the "
                             f"profile of the model whose budget applies (a teacher run counts at the student's speed, "
                             f"e.g. l4x4_e2b_nolora), or KAGGLE_CLOCK=off")
        name, speed_default = auto
    profile = load_profiles()[name]
    cap_s = float(os.environ.get("KAGGLE_CLOCK_CAP_S", "330"))
    speed = float(os.environ.get("KAGGLE_CLOCK_SPEED", speed_default))
    return dict(profile=profile, cap_s=cap_s, speed=speed), dict(profile=name, cap_s=cap_s, speed_factor=speed)


class KaggleClock:
    """Plugs into SwegemmaContext.token_budget: the harness calls record_usage after every model call."""

    max_budget_usd = None  # read by the harness; None = it keeps its own cost budget (unused here)

    def __init__(self, ctx, profile, cap_s, speed=1.0):
        self.ctx, self.p, self.cap_s, self.speed = ctx, profile, cap_s, speed
        self.llm_s = 0.0
        self.calls = self.out_tokens = 0
        self.cut = False
        self.cut_at_tool_calls = None
        self.cut_stopped = None   # True: the cut expired the harness timeout; False: only the tool-call cap was set
        self._prev = None  # (prompt tokens, output tokens) of the previous call, for the new-prompt-token count
        self._lock = threading.Lock()

    def record_usage(self, input_tokens=0, output_tokens=0, **_):
        p, o = int(input_tokens or 0), int(output_tokens or 0)
        if p <= 0:  # no usage on this event
            return
        with self._lock:
            # new tokens since the previous call; a shrink means the history was compacted, so it is all cold again
            new = p if self._prev is None or p < self._prev[0] else max(0, p - self._prev[0] - self._prev[1])
            self._prev = (p, o)
            s = self.p["per_call_s"] + o / (decode_rate(self.p, p) * self.speed)
            if self.p.get("prefill_tok_s"):
                s += new / (self.p["prefill_tok_s"] * self.speed)
            self.llm_s += s
            self.calls += 1
            self.out_tokens += o
            if not self.cut and self.seconds >= self.cap_s:
                self.cut = True
                self.cut_at_tool_calls = self.ctx.tool_calls_used
                # end the loop after this turn through the harness's tool-call-budget stop (EvaluationBudget is frozen)
                self.ctx.budget = dataclasses.replace(self.ctx.budget, tool_calls=self.ctx.tool_calls_used)
                self.cut_stopped = expire_current_timeout()

    @property
    def seconds(self):
        return self.llm_s + self.p.get("tool_s_per_call", 0.0) * self.ctx.tool_calls_used

    def row(self, resolved):
        """Ledger fields for this task (added, never renamed: results/task_runs.jsonl rule)."""
        return {"kaggle_s": round(self.seconds, 1), "kaggle_cut": self.cut, "kaggle_cut_at_tool_calls": self.cut_at_tool_calls,
                "kaggle_cut_stopped": self.cut_stopped, "kaggle_llm_calls": self.calls, "kaggle_out_tokens": self.out_tokens,
                "resolved_kaggle": bool(resolved) and not self.cut}
