"""Summarise parallel real-task runs: throughput, cache reuse, faults and per-task outcomes.

usage: python -I kits/swe-agent/bench/real_summary.py <results_dir> [...]
Each results_dir holds probe_results.jsonl, metrics_before.txt, metrics_after.txt and traces/ (written by kits/swe-agent/bench/real_matrix.sh).
Wall time is taken from the metrics files' modification times.
"""
import json
import os
import sys


def metrics(path):
    out = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        key = line.split("{")[0].split(" ")[0]
        try:
            out[key] = out.get(key, 0) + float(line.rsplit(" ", 1)[1])
        except (ValueError, IndexError):
            pass
    return out


for o in sys.argv[1:]:
    res = os.path.join(o, "probe_results.jsonl")
    after = os.path.join(o, "metrics_after.txt")
    if not (os.path.exists(res) and os.path.exists(after)):
        print(o, "INCOMPLETE")
        continue
    rows = [json.loads(line) for line in open(res)]
    before = os.path.join(o, "metrics_before.txt")
    wall = os.path.getmtime(after) - os.path.getmtime(before)
    a, b = metrics(before), metrics(after)
    d = lambda k: round(b.get(k, 0) - a.get(k, 0))
    calls = sum(r["tool_calls"] for r in rows)
    secs = sum(r["agent_seconds"] for r in rows)
    pads = sum(open(os.path.join(o, "traces", f)).read().count("<pad>") for f in os.listdir(os.path.join(o, "traces")))
    hit = d("vllm:prefix_cache_hits_total") / max(d("vllm:prefix_cache_queries_total"), 1)
    print(f"{os.path.basename(o)}: tasks {len(rows)} wall_s {wall:.0f} resolved {sum(r['resolved'] for r in rows)} "
          f"calls {calls} calls_per_min {calls / wall * 60:.1f} tasks_per_hour {len(rows) / wall * 3600:.1f} "
          f"mean_task_s {secs / len(rows):.0f} cache_hit {hit:.2f} preemptions {d('vllm:num_preemptions_total')} "
          f"reply_tokens {d('vllm:generation_tokens_total')} pad_strings {pads}")
    print("   ", " ".join(f"{r['id']}:{'Y' if r['resolved'] else 'n'}/{r['tool_calls']}/{r['agent_seconds']:.0f}"
                          for r in sorted(rows, key=lambda r: r["id"])))
