"""Kaggle clock: a per-task budget in predicted Kaggle-seconds, so a local run is cut where the scorer's 4x L4 would cut it.

Local wall time means nothing (other GPU, several tasks at once, speculative decoding), so each model call is converted
into the seconds it would take on Kaggle, with the speed profile from kaggle_time_profiles.json and the same formula as
kaggle_time.py:

    call s  = per_call_s + output tokens / decode speed(context) + new prompt tokens / prefill speed
    task s  = sum of call s + tool_s_per_call * tool calls          (the ~30 s per-task setup is left out of the cap)

The running total only depends on token counts, so a run on any machine at any speed gives the same cut. When the total
passes the cap, the harness's own tool-call-budget stop ends the agent loop (the pending tool calls are refused, as a
wall-clock kill would have), the patch so far is tested as usual, and the task is marked kaggle_cut: it counts as
unsolved in resolved_kaggle whatever the tests say, because on Kaggle that time would have been taken from other tasks.

Switched on in run_tasks.py by env KAGGLE_CLOCK=<profile name, e.g. l4x4_12b_nolora>. Also:
    KAGGLE_CLOCK_CAP_S   cap in Kaggle-seconds, default 330 (12 h / 120 tasks = 360 s, minus 30 s setup)
    KAGGLE_CLOCK_SPEED   speed factor, default 1.0; 0.88 = the LoRA penalty measured on the 31B, for sizes with no
                         adapter profile of their own
Known approximation: the new-prompt-token count treats all agents of a task as one stream, so with sub-agents the small
prefill term can be a few seconds off. Output tokens and tool calls, which carry the time, are exact.
"""
import dataclasses
import os
import threading

from kaggle_time import decode_rate, load_profiles


def settings():
    """Env -> (KaggleClock kwargs, the dict pinned into the run ledger). None when the clock is off."""
    name = os.environ.get("KAGGLE_CLOCK")
    if not name:
        return None
    profile = load_profiles()[name]
    cap_s = float(os.environ.get("KAGGLE_CLOCK_CAP_S", "330"))
    speed = float(os.environ.get("KAGGLE_CLOCK_SPEED", "1.0"))
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

    @property
    def seconds(self):
        return self.llm_s + self.p.get("tool_s_per_call", 0.0) * self.ctx.tool_calls_used

    def row(self, resolved):
        """Ledger fields for this task (added, never renamed: results/task_runs.jsonl rule)."""
        return {"kaggle_s": round(self.seconds, 1), "kaggle_cut": self.cut, "kaggle_cut_at_tool_calls": self.cut_at_tool_calls,
                "kaggle_llm_calls": self.calls, "kaggle_out_tokens": self.out_tokens,
                "resolved_kaggle": bool(resolved) and not self.cut}
