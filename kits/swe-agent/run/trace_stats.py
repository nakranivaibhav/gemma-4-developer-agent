"""Boil one agent trace down to counts: what the agent did, how it ended, where the tokens went.

usage: python -I kits/swe-agent/run/trace_stats.py <results_dir> [...]     (one line per task-run; run_tasks.py embeds the same dict)
The trace itself (every thought, tool call, tool output and per-step token count) stays in traces/trace_<id>.json.
"""
import collections
import json
import os
import sys

# tool -> (required args, allowed args), from the signatures in swegemma/tools/{workspace,execution,graph}.py
EXPECTED = {
    "read_file": ({"filepath"}, {"filepath", "start_line", "end_line"}),
    "edit_file": ({"filepath", "old_string", "new_string"}, {"filepath", "old_string", "new_string", "allow_multiple"}),
    "write_file": ({"filepath", "content"}, {"filepath", "content"}),
    "run_command": ({"command"}, {"command"}),
    "submit_patch": (set(), set()),
    "get_status": (set(), set()),
    "search_similar_code": ({"query"}, {"query", "k"}),
    "get_code_neighbors": ({"node"}, {"node", "edge_type", "max_neighbors"}),
    "get_code_subgraph": ({"nodes"}, {"nodes"}),
}
NO_ARG_TOOLS = {"submit_patch", "get_status"}  # the harness does not count these against the tool-call budget either


def _content(step):
    obs = step.get("observation")
    c = (obs or {}).get("content") if isinstance(obs, dict) else None
    return c if isinstance(c, str) else ""


REJECTED = "mandatory input parameters are not present"  # ADK refuses the call before the tool runs; the harness does not count it


def summarize(trace_path, harness_error=None, budgets=None, max_output_tokens=None):
    t = json.load(open(trace_path))
    steps = t["steps"]
    by_tool = collections.Counter()
    malformed = tool_errors = rejected = counted = 0
    seen = collections.Counter()
    files_read, files_edited, files_edit_failed = set(), set(), set()
    n_calls = 0
    last_call = None  # (tool name, failed) of the run's last tool call
    for s in steps:
        content = _content(s)
        failed = '"status": "error"' in content
        tool_errors += failed
        was_rejected = REJECTED in content
        rejected += was_rejected
        for c in s.get("tool_calls") or []:  # one call per step in practice; the step's observation is its result
            n_calls += 1
            name, args = c["function_name"], c.get("arguments") or {}
            by_tool[name] += 1
            last_call = (name, failed)
            counted += name not in NO_ARG_TOOLS and not was_rejected  # what the harness charges against the budget
            req, allowed = EXPECTED.get(name, (set(), None))
            if not req <= set(args) or (allowed is not None and not set(args) <= allowed):
                malformed += 1
            if name not in NO_ARG_TOOLS:
                seen[(name, json.dumps(args, sort_keys=True))] += 1
            if name == "read_file" and "filepath" in args:
                files_read.add(args["filepath"])
            if name in ("edit_file", "write_file") and "filepath" in args:
                (files_edit_failed if failed else files_edited).add(args["filepath"])
    metrics = [s["metrics"] for s in steps if s.get("metrics")]
    thinking = sum(len(s.get("message") or "") for s in steps if (s.get("extra") or {}).get("event_type") == "thinking")
    times = [(s.get("extra") or {}).get("timestamp") for s in steps if (s.get("extra") or {}).get("timestamp")]
    last = (steps[-1].get("extra") or {}).get("event_type") if steps else None
    err = (harness_error or "").lower()
    budgets = budgets or {}
    last_completion = next((s["metrics"].get("completion_tokens", 0) for s in reversed(steps) if s.get("metrics")), 0)
    if "tool call budget" in err:  # the harness's own verdict wins when we have it
        ended = "budget_tool_calls"
    elif "timeout" in err:
        ended = "budget_time"
    elif "contextwindowexceeded" in err or "context length" in err:
        ended = "context_overflow"  # prompt + max_output_tokens > the model's window: a hard 400, no retry
    elif budgets.get("max_tool_calls") and counted >= budgets["max_tool_calls"] and last != "final":
        ended = "budget_tool_calls"
    elif last == "final":
        ended = "final"
    elif last_call and last_call[0] == "submit_patch" and not last_call[1]:
        # the run stopped right after a successful submit_patch, with no closing text turn; a clean end
        # (until 2026-10-23 these were tagged "aborted", which dropped 304 solved runs from SFT selection)
        ended = "submitted"
    elif max_output_tokens and last_completion >= max_output_tokens:
        ended = "output_truncated"  # the closing turn hit max_output_tokens before it could call a tool
    else:
        ended = "aborted"  # an exception ended the run (server error, context overflow): harness_error has the text
    return {
        "llm_calls": len(metrics),
        "tool_calls": n_calls,
        "by_tool": dict(by_tool),
        "submit_patch_calls": by_tool.get("submit_patch", 0),
        "malformed_calls": malformed,  # arguments do not match the tool's signature (rejected or executed with a dropped key)
        "rejected_calls": rejected,  # refused before execution (missing mandatory argument)
        "budget_counted_calls": counted,  # the harness's own count: no submit_patch/get_status, no rejected calls
        "tool_errors": tool_errors,
        "repeated_calls": sum(n - 1 for n in seen.values() if n > 1),  # identical call with identical arguments, again
        "files_read": sorted(files_read),
        "files_edited": sorted(files_edited),  # edits that succeeded
        "files_edit_failed_only": sorted(files_edit_failed - files_edited),
        "prompt_tokens": sum(m.get("prompt_tokens", 0) for m in metrics),
        "completion_tokens": sum(m.get("completion_tokens", 0) for m in metrics),
        "max_prompt_tokens": max((m.get("prompt_tokens", 0) for m in metrics), default=0),
        "thinking_chars": thinking,
        "elapsed_s": round(times[-1] - times[0], 1) if len(times) > 1 else None,
        "ended_with": ended,
    }


if __name__ == "__main__":
    for d in sys.argv[1:]:
        errors, budgets, max_out = {}, {}, None
        if os.path.exists(os.path.join(d, "settings.json")):
            st = json.load(open(os.path.join(d, "settings.json")))
            budgets = {"max_tool_calls": st.get("budget_max_tool_calls")}
            max_out = st.get("max_output_tokens")
        try:  # the harness's error text per task, when the run kept probe_results.jsonl
            for line in open(os.path.join(d, "probe_results.jsonl")):
                r = json.loads(line)
                errors[r["id"]] = r.get("harness_error") or r.get("error")
        except FileNotFoundError:
            pass
        for f in sorted(os.listdir(os.path.join(d, "traces"))):
            tid = f[6:-5]
            s = summarize(os.path.join(d, "traces", f), errors.get(tid), budgets, max_out)
            s.pop("files_read"), s.pop("files_edited"), s.pop("files_edit_failed_only")
            print(tid, json.dumps(s))
