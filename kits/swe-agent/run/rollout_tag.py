"""ROLLOUT_TAG=1 (RL rollouts): every model call carries the header X-Rollout-Id: <run_id>/<task_id>, so the
capture proxy (kits/swe-agent/experiments/log_proxy.py) can file each call under its try. Without it, calls of
concurrent tries could only be grouped by guessing from their content.

How: run_one sets ROLLOUT_ID in its own asyncio task; ADK's LiteLLMClient.acompletion is wrapped to add the header.
asyncio tasks and threads started inside the try copy the context, so the id follows the try and never leaks into
another one (tests: training/rl/tests/test_harness_tag.py). The header changes nothing the model sees: the proxy
drops it before forwarding, and the scorer never sets ROLLOUT_TAG.
"""
import contextvars

ROLLOUT_ID = contextvars.ContextVar("rollout_id", default=None)
HEADER = "X-Rollout-Id"
_installed = False


def install():
    global _installed
    if _installed:
        return
    import google.adk.models.lite_llm as adk_llm

    original = adk_llm.LiteLLMClient.acompletion

    async def acompletion_tagged(self, model, messages, tools, **kwargs):
        rid = ROLLOUT_ID.get()
        if rid:
            kwargs["extra_headers"] = {**(kwargs.get("extra_headers") or {}), HEADER: rid}
        return await original(self, model, messages, tools, **kwargs)

    adk_llm.LiteLLMClient.acompletion = acompletion_tagged
    _installed = True
