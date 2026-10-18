"""One line per trace step: who spoke, which tool, how big the prompt was.

usage: python -I analysis/trace_summary.py <trace.json> [--width N]
Everything printed is truncated; the trace is harness output and may be large.
"""
import json
import sys

path = sys.argv[1]
width = int(sys.argv[sys.argv.index("--width") + 1]) if "--width" in sys.argv else 110
trace = json.load(open(path))


def short(value, n=width):
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n] + f"…(+{len(text) - n})"


print("final_metrics", short(trace.get("final_metrics"), 300))
for step in trace["steps"]:
    metrics = step.get("metrics") or {}
    head = f"{step.get('step_id'):>3} {step.get('source', '?'):<6} {step.get('model_name', '-'):<10}"
    head += f" prompt={metrics.get('prompt_tokens', '-')} out={metrics.get('completion_tokens', '-')}"
    calls = step.get("tool_calls") or []
    if not calls:
        print(head, "| msg:", short(step.get("message", ""), 90))
        continue
    obs = step.get("observation")
    for call in calls:
        print(head, "|", call.get("function_name"), short(call.get("arguments"), 90))
    print(" " * 8, "-> observation chars:", len(json.dumps(obs, default=str)), "|", short(obs, 160))
