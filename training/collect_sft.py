"""Turn solved local runs into chat-format training rows (supervised fine-tuning / rejection-sampling data).

One row per solved (run, task), rebuilt from the run's trace (ATIF-v1.7):
  system prompt, task prompt, then for each main-agent step an assistant turn (thinking text in
  `reasoning_content`, tool calls in OpenAI format) followed by one `tool` message per call.
  Harness continuation nudges become `user` turns (flagged in `nudges`). Sub-agent (code_analyzer)
  steps are dropped: the main agent sees only the sub-agent's returned result, which is its tool message.

Never used for training: tasks in --exclude (default holdout), unsolved rows, rows with a
harness error. A row is skipped (and counted) when a step's tool calls cannot be matched to results.

usage: collect_sft.py [--out runs/sft/solved_v0.jsonl] [--exclude holdout] [--runs l100_lean_p1 ...]
       [--model-substr 31b] [--best-per-task]
Prints a per-model summary: rows, distinct tasks, median output tokens.
"""
import argparse
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kits" / "swe-agent" / "run"))  # kit helpers
import manifest  # noqa: E402
from kaggle_time import load_profiles, task_minutes, trace_calls  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # the repo root


def truthy(v):
    return str(v).lower() == "true"


def rebuild(trace):
    """ATIF trace -> (messages, stats) or (None, reason)."""
    msgs, out_tok, n_calls, nudges, peak_ctx = [], 0, 0, 0, 0
    for s in trace["steps"]:
        ex = s.get("extra") or {}
        ev, src = ex.get("event_type"), s["source"]
        calls = s.get("tool_calls") or []
        if src == "system" and ev == "system_instruction":
            msgs.append({"role": "system", "content": s["message"]})
            continue
        if src == "user" and ev == "task_prompt":
            msgs.append({"role": "user", "content": s["message"]})
            continue
        if src == "system" and ev == "continuation_nudge":
            msgs.append({"role": "user", "content": s["message"]})
            nudges += 1
            continue
        if src != "agent" or ex.get("author", "swe_baseline_agent") != "swe_baseline_agent":
            continue  # sub-agent steps and their results: the main agent never saw them
        if any((c.get("extra") or {}).get("author") not in (None, "swe_baseline_agent") for c in calls):
            continue
        out_tok += (s.get("metrics") or {}).get("completion_tokens", 0)
        peak_ctx = max(peak_ctx, (s.get("metrics") or {}).get("total_tokens", 0))
        turn = {"role": "assistant", "content": s["message"] if ev == "final" else ""}
        if ev != "final" and s.get("message"):
            turn["reasoning_content"] = s["message"]
        if calls:
            turn["tool_calls"] = [{"id": c["tool_call_id"], "type": "function",
                                   "function": {"name": c["function_name"], "arguments": json.dumps(c.get("arguments") or {})}}
                                  for c in calls]
        msgs.append(turn)
        if calls:
            obs = s.get("observation")
            if len(calls) != 1 or not obs:
                return None, "unmatched_tool_result"
            msgs.append({"role": "tool", "tool_call_id": calls[0]["tool_call_id"], "name": calls[0]["function_name"],
                         "content": obs.get("content", "")})
            n_calls += 1
    if len(msgs) < 3 or msgs[0]["role"] != "system":
        return None, "no_conversation"
    return msgs, {"out_tokens": out_tok, "tool_calls": n_calls, "nudges": nudges, "peak_context_tokens": peak_ctx,
                  "prompt_tokens_total": trace.get("final_metrics", {}).get("total_prompt_tokens")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/sft/solved_v0.jsonl")
    ap.add_argument("--exclude", default="holdout")
    ap.add_argument("--runs", nargs="*", help="run ids (default: every run in the ledger)")
    ap.add_argument("--model-substr", default="", help="keep rows whose pinned model name contains this")
    ap.add_argument("--best-per-task", action="store_true", help="keep only the solve with the fewest output tokens per task")
    ap.add_argument("--cap-s", type=float, default=330.0, help="Kaggle per-task cap for the within_cap flag")
    a = ap.parse_args()
    profiles = load_profiles()
    prof_for = {"31b": "l4x4_31b_adapter", "12b": "l4x4_12b_nolora", "e4b": "l4x4_e4b_nolora", "e2b": "l4x4_e2b_nolora"}

    excluded = set(manifest.ids(a.exclude)) if a.exclude else set()
    latest = {}
    for line in open(ROOT / "results" / "task_runs.jsonl"):
        r = json.loads(line)
        latest[(r.get("run_id"), r["id"])] = r  # a rerun of the same task in the same run replaces the earlier row
    skipped, rows = Counter(), []
    for (rid, tid), r in latest.items():
        if a.runs and rid not in a.runs:
            continue
        if not truthy(r.get("resolved")) or r.get("harness_error") not in (None, "None", ""):
            continue
        if tid in excluded:
            skipped["holdout"] += 1
            continue
        settings = r.get("settings") or {}
        if isinstance(settings, str):
            settings = {}
        model = settings.get("model", "?")
        if a.model_substr and a.model_substr not in model:
            continue
        tp = ROOT / r["trace"] if r.get("trace") else None
        if not tp or not tp.exists():
            skipped["no_trace"] += 1
            continue
        msgs, info = rebuild(json.load(open(tp)))
        if msgs is None:
            skipped[info] += 1
            continue
        prof = next((profiles[v] for k, v in prof_for.items() if k in model), None)
        if prof:  # predicted Kaggle agent seconds on 4x L4 for this model (kaggle_time.py), setup excluded
            calls, tools = trace_calls(tp)
            info["kaggle_s"] = round(task_minutes(prof, calls, tools) * 60 - prof["setup_min"] * 60, 1)
            info["within_cap"] = info["kaggle_s"] <= a.cap_s
        rows.append({"task_id": tid, "run_id": rid, "model": model, "adapter": settings.get("adapter"),
                     "tests_passed": r.get("tests_passed"), "trace": r["trace"], **info, "messages": msgs})
    if a.best_per_task:
        best = {}
        for row in rows:
            k = (row["model"], row["task_id"])
            if k not in best or row["out_tokens"] < best[k]["out_tokens"]:
                best[k] = row
        rows = list(best.values())
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for row in sorted(rows, key=lambda x: (x["model"], x["task_id"], x["run_id"])):
            fh.write(json.dumps(row) + "\n")
    by = defaultdict(list)
    for row in rows:
        by[row["model"]].append(row)
    print(f"wrote {len(rows)} rows to {a.out} | skipped {dict(skipped)}")
    print(f"{'model':38s} rows  tasks  med out tok  med tool calls  within cap: rows tasks  max ctx")
    for m, rs in sorted(by.items()):
        print(f"{m:38s} {len(rs):4d}  {len({x['task_id'] for x in rs}):5d}  {st.median(x['out_tokens'] for x in rs):11.0f}"
              f"  {st.median(x['tool_calls'] for x in rs):14.0f}"
              f"  {sum(bool(x.get('within_cap')) for x in rs):16d} {len({x['task_id'] for x in rs if x.get('within_cap')}):5d}"
              f"  {max(x['peak_context_tokens'] for x in rs):7d}")
    print(f"distinct tasks, any model: {len({x['task_id'] for x in rows})}")


if __name__ == "__main__":
    sys.exit(main())
