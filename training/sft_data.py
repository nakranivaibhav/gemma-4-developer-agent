"""SFT data pipeline: ledger -> selected runs -> per-call examples -> packed sequences.

Three stages, each writes one file into --out (default runs/sft/):

  select   results/task_runs.jsonl  -> selected_runs.jsonl   which solved runs to imitate
  build    selected_runs.jsonl      -> examples.jsonl        one (context, target) per model call
  pack     examples.jsonl           -> packed.jsonl          one sequence per trace segment

usage:
  uv run training/sft_data.py select [--out runs/sft]
  uv run training/sft_data.py build  [--out runs/sft]
  uv run training/sft_data.py pack   [--out runs/sft]
  uv run training/sft_data.py all    [--out runs/sft]
  uv run training/sft_data.py all --student E4B --out runs/sft/e4b   # E4B: same tokenizer/template, slower budget
  uv run training/sft_data.py all --mask --out runs/sft/e2b_masked    # no loss on model-fault calls

Design notes (the why lives in docs/ and the notebook that produced v1):
- Every example is rendered with ONE fixed setup, the one RL will start from: the system prompt and
  tool declarations captured from a real vLLM request (REQUEST_SOURCE), with the tool list cut to
  RL_TOOLS. The original prompt of the run is dropped.
- A "model call" is one vLLM request. A trace step with `metrics` starts a call; a step without
  `metrics` continues the previous call (thought / visible text / call logged as separate steps).
- History sent back to the model leaves out calls that made no tool call (the harness drops them),
  and the Gemma template drops thoughts that precede the latest user turn. Both make the context of
  a later call NOT a prefix of the earlier text, which is where `pack` starts a new sequence.
- Target = what the model wrote in that call, ending with its stop token:
  `<|tool_response>` after a tool call, `<turn|>` after a text-only reply.
- Loss mask: tokens [n_ctx, n_total) of each example; packed sequences carry the same spans, plus
  n_tokens (the sequence length). With --mask, a model-fault call keeps its text but has no span.
"""
from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kits" / "swe-agent" / "run"))  # the Kaggle clock: one time model for runs and data
import kaggle_clock  # noqa: E402
import kaggle_time  # noqa: E402
import trace_stats  # noqa: E402


# ----------------------------------------------------------------------------- settings
@dataclass
class DataConfig:
    model_dir: Path = ROOT / "data/external/models/gemma-4-e2b-it-qat-w4a16-ct"
    ledger: Path = ROOT / "results/task_runs.jsonl"
    manifest: Path = ROOT / "tasks/manifest.csv"
    # captured request whose system prompt + tool declarations every example is rendered with
    request_source: Path = ROOT / "runs/v_e2b_lean_facts_v1_p1/requests.jsonl.gz"   # r08_lean_facts
    rl_tools: frozenset = frozenset({"run_command", "read_file", "edit_file", "write_file", "get_status",
                                     "submit_patch", "get_code_neighbors", "get_code_subgraph"})
    student: str = "E2B"                       # model being trained: sets the time budget
    teacher_models: tuple = ("E2B", "31B", "E4B", "Qwen")
    # teachers used only for a task with no Gemma pick. Off: the one candidate (Qwen, fastapi_15588) makes two tool
    # calls in one step, which trace_to_turns does not parse and the Gemma student never does (2026-10-23)
    fallback_models: tuple = ()
    max_tool_calls: int = 100                  # runs at/over this are dropped
    task_seconds: float = 330                  # Kaggle per-task slot (docs/measured_numbers.md)
    tok_per_s: dict = field(default_factory=lambda: {"E2B": 130, "E4B": 91, "12B": 66, "31B": 30.6})
    sec_per_tool_call: float = 1.9
    # runs made with this system.md sort first: v2 = the r08 prompt, the student's own (r09_* agents, dg_31b_r08,
    # dg_qwen_r08); v1 used "bd84a8fe23c12a5e" (r07)
    prefer_prompt_sha: str = "3e35a2ed30dd55b6"
    max_tokens: int = 32_768                   # E2B context window; longer examples are dropped
    # --mask: model-fault calls stay in the text (the next turn shows the recovery) but get no loss.
    # CommandError (command ran, e.g. a failing repro) and FileReadError (harness glitch) are not model faults.
    mask_errors: frozenset = frozenset()       # e.g. {"FileEditError", "ValueError"}
    mask_leaked_calls: bool = False            # no real call, but "<tool_call|>" written into the text


MODEL_KEYS = [("31b", "31B"), ("12b", "12B"), ("e4b", "E4B"), ("e2b", "E2B"), ("qwen", "Qwen")]


def model_of(r: dict) -> str:
    m = ((r.get("settings") or {}).get("model") or "").lower()
    return next((v for k, v in MODEL_KEYS if k in m), "unknown")


def write_jsonl(path: Path, rows) -> int:
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


# ----------------------------------------------------------------------------- stage 1: select
def load_ledger(cfg: DataConfig) -> list[dict]:
    """Deduped ledger rows; the last write for a (run_id, task) wins."""
    seen = {}
    for r in read_jsonl(cfg.ledger):
        seen[(r["run_id"], r["id"])] = r
    return list(seen.values())


def kaggle_seconds(r: dict, cfg: DataConfig) -> float:
    """Predicted Kaggle-seconds of this run's trace if the STUDENT (served with its adapter) had written it: the same
    profile and formula the run-time Kaggle clock cuts on (kits/swe-agent/run/kaggle_clock.py). Falls back to the old
    tokens / tok_per_s + sec_per_tool_call formula when the trace file is missing."""
    name, speed = kaggle_clock.auto_profile(cfg.student, has_adapter=True)
    trace = ROOT / r["trace"] if r.get("trace") else None
    if trace is not None and trace.exists():
        calls, tools = kaggle_time.trace_calls(trace)
        return kaggle_time.task_seconds(_PROFILES[name], calls, tools, speed)
    ts = r["trace_stats"]
    return ts["completion_tokens"] / cfg.tok_per_s[cfg.student] + cfg.sec_per_tool_call * ts["tool_calls"]


_PROFILES = kaggle_time.load_profiles()


def fits_budget(r: dict, cfg: DataConfig) -> bool:
    """Would this run fit the student's Kaggle slot (cfg.task_seconds Kaggle-seconds, 330 = the clock's cap)?"""
    return kaggle_seconds(r, cfg) <= cfg.task_seconds


def ends_clean(r: dict) -> bool:
    """The run ended normally: a closing text turn, or a successful submit_patch with nothing after it.
    Ledger rows written before 2026-10-23 tag the second kind "aborted", so those are re-read from the trace."""
    ended = (r.get("trace_stats") or {}).get("ended_with")
    if ended in ("final", "submitted", None):
        return True
    if ended == "aborted" and r.get("trace") and (ROOT / r["trace"]).exists():
        return trace_stats.summarize(ROOT / r["trace"])["ended_with"] in ("final", "submitted")
    return False


def edits_test(r: dict) -> bool:
    return any(f.startswith("tests/") for f in (r.get("trace_stats") or {}).get("files_edited", []))


def tools_ok(r: dict, cfg: DataConfig) -> bool:
    return set((r.get("trace_stats") or {}).get("by_tool", {})) <= cfg.rl_tools


def keep_reason(r: dict, split_of: dict, cfg: DataConfig) -> str | None:
    """None if the run is a usable demonstration, else the reason it is dropped."""
    ts = r.get("trace_stats") or {}
    if not r.get("resolved"):
        return "not_resolved"
    if split_of.get(r["id"]) != "tune":
        return "not_tune"
    if r.get("harness_error") or not ends_clean(r):
        return "aborted"
    if model_of(r) not in cfg.teacher_models:
        return "model_" + model_of(r)
    if model_of(r) in ("E2B", "E4B") and (r.get("settings") or {}).get("adapter"):
        return "trained_adapter"  # a run of one of our own SFT adapters, not a base model
    if ts.get("tool_calls", 0) >= cfg.max_tool_calls:
        return "too_many_tool_calls"
    if not fits_budget(r, cfg):
        return "over_time_budget"
    if edits_test(r):
        return "edits_tests"
    if not tools_ok(r, cfg):
        return "non_rl_tool"
    if not (ROOT / r["trace"]).exists():
        return "trace_missing"
    return None


def slots(p: float) -> tuple[int, int]:
    """(E2B slots, big-model slots) for a task from E2B's pass rate on it."""
    if p == 0:
        return 0, 4      # E2B never solves it: all teacher traces
    if p < 0.1:
        return 2, 2      # rare E2B wins
    if p < 0.9:
        return 2, 1      # already mixed
    return 1, 0          # E2B almost always solves it


def family(r: dict) -> str:
    """Strip repeat suffixes: hist_e2b_t07_r2_s3 -> hist_e2b_t07, v_e2b_lean_v1_p4 -> v_e2b_lean_v1."""
    return re.sub(r"(_r2)?(_[sp]\d+)?$", "", r["run_id"])


def pick(runs: list[dict], k: int, cfg: DataConfig) -> list[dict]:
    """Best k runs: preferred prompt first, then fewest tool calls, then fewest output tokens;
    at most one run per run family."""
    def key(r):
        ts = r["trace_stats"]
        sha = r["settings"].get("file_sha", {}).get("prompts/system.md")
        return (sha != cfg.prefer_prompt_sha, ts["tool_calls"], ts["completion_tokens"])

    chosen, fams = [], set()
    for r in sorted(runs, key=key):
        if len(chosen) >= k:
            break
        if family(r) not in fams:
            chosen.append(r)
            fams.add(family(r))
    return chosen


def select(cfg: DataConfig) -> list[dict]:
    split_of = {r["id"]: r["split"] for r in csv.DictReader(open(cfg.manifest))}
    rows = load_ledger(cfg)

    tries, wins = collections.Counter(), collections.Counter()
    for r in rows:
        if split_of.get(r["id"]) == "tune" and model_of(r) == "E2B" and not (r.get("settings") or {}).get("adapter"):
            tries[r["id"]] += 1
            wins[r["id"]] += bool(r.get("resolved"))
    rate = lambda t: wins[t] / tries[t] if tries[t] else 0.0

    drops = collections.Counter()
    by_task = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        why = keep_reason(r, split_of, cfg)
        if why:
            drops[why] += 1
        else:
            by_task[r["id"]][model_of(r)].append(r)

    selected = []
    for task, runs in sorted(by_task.items()):
        n_e2b, n_big = slots(rate(task))
        e2b = pick(runs["E2B"], n_e2b, cfg)
        n_big += n_e2b - len(e2b)                       # unfilled E2B slots go to the teachers
        big = pick(runs["31B"], n_big, cfg)
        big += pick(runs["E4B"], n_big - len(big), cfg)  # 31B short: top up with E4B
        if not e2b + big:  # no Gemma solve at all: one trace from a fallback teacher
            for m in cfg.fallback_models:
                big += pick(runs[m], 1 - len(big), cfg)
        for r in e2b + big:
            selected.append(dict(r, model=model_of(r), family=family(r), e2b_rate=round(rate(task), 3)))

    print("dropped:", dict(drops.most_common()))
    print(len(selected), "runs over", len({r["id"] for r in selected}), "tasks",
          dict(collections.Counter(r["model"] for r in selected)))
    return selected


# ----------------------------------------------------------------------------- stage 2: build
class Renderer:
    """Fixed system prompt + tools, and the E2B chat template."""

    def __init__(self, cfg: DataConfig):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(cfg.model_dir)
        with gzip.open(cfg.request_source, "rt") as f:
            req = json.loads(f.readline())["request"]
        self.system = req["messages"][0]["content"]
        self.tools = [t for t in req["tools"] if t["function"]["name"] in cfg.rl_tools]
        assert len(self.tools) == len(cfg.rl_tools), "request_source lacks some RL tools"

    def text(self, msgs: list[dict], generation_prompt: bool) -> str:
        return self.tok.apply_chat_template(msgs, tools=self.tools, tokenize=False,
                                            add_generation_prompt=generation_prompt, enable_thinking=True)


def trace_to_turns(path: Path) -> list[dict]:
    """ATIF trace -> user / model / tool turns, one model turn per model call."""
    turns = []
    for s in json.load(open(path))["steps"]:
        kind = (s.get("extra") or {}).get("event_type")
        if kind == "system_instruction":                 # replaced by the fixed system prompt
            continue
        if kind in ("task_prompt", "continuation_nudge"):
            turns.append({"role": "user", "content": s["message"]})
            continue
        if s.get("metrics"):                             # a new model call starts here
            turns.append({"role": "model", "thought": "", "text": "", "call": None})
        turn = turns[-1]
        if kind == "thinking":
            turn["thought"] += s.get("message") or ""
        else:
            turn["text"] += s.get("message") or ""
        if s.get("tool_calls"):
            c = s["tool_calls"][0]
            turn["call"] = {"name": c["function_name"], "args": c["arguments"]}
            turns.append({"role": "tool", "name": c["function_name"], "content": s["observation"]["content"]})
    return turns


def unescape(s):
    """Traces store \\uXXXX escapes in tool output; vLLM received the real characters."""
    try:
        return json.dumps(json.loads(s), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return s


def history_messages(turns: list[dict], system: str) -> list[dict]:
    """The chat history the harness would send: model calls without a tool call are left out."""
    msgs = [{"role": "system", "content": system}]
    for i, t in enumerate(turns):
        if t["role"] == "user":
            msgs.append({"role": "user", "content": t["content"]})
        elif t["role"] == "model":
            if not t["call"]:
                continue
            msgs.append({"role": "assistant", "content": t["text"] or None, "reasoning_content": t["thought"],
                         "tool_calls": [{"type": "function", "id": f"call_{i}",
                                         "function": {"name": t["call"]["name"], "arguments": t["call"]["args"]}}]})
        else:
            msgs.append({"role": "tool", "tool_call_id": f"call_{i-1}", "content": unescape(t["content"])})
    return msgs


def tool_error(turn: dict | None) -> str | None:
    """error_type of a tool result, or None when the call went through."""
    if not turn or turn["role"] != "tool":
        return None
    try:
        res = json.loads(turn["content"])
    except (json.JSONDecodeError, TypeError):
        return None
    return res.get("error_type") if isinstance(res, dict) and res.get("status") == "error" else None


def is_fault(turns: list[dict], j: int, cfg: DataConfig) -> str | None:
    """Why model call j should get no loss under --mask, else None."""
    t = turns[j]
    if t["call"]:
        err = tool_error(turns[j + 1] if j + 1 < len(turns) else None)
        return err if err in cfg.mask_errors else None
    if cfg.mask_leaked_calls and "<tool_call|>" in t["thought"] + t["text"]:
        return "leaked_call"
    return None


def make_examples(turns: list[dict], rd: Renderer) -> list[dict]:
    out = []
    for j, t in enumerate(turns):
        if t["role"] != "model":
            continue
        context = rd.text(history_messages(turns[:j], rd.system), generation_prompt=True)
        if t["call"]:
            full = rd.text(history_messages(turns[:j + 1], rd.system), generation_prompt=False)
            assert full.startswith(context)
            target = full[len(context):]                  # ... <tool_call|><|tool_response>
        else:                                             # template would drop the thought here
            target = f"<|channel>thought\n{t['thought']}\n<channel|>{t['text']}<turn|>"
        out.append({"context": context, "target": target, "has_call": bool(t["call"])})
    return out


def build(selected: list[dict], cfg: DataConfig):
    rd = Renderer(cfg)
    stats = collections.Counter()
    for r in selected:
        turns = trace_to_turns(ROOT / r["trace"])
        assert sum(t["role"] == "model" for t in turns) == r["trace_stats"]["llm_calls"], r["trace"]
        model_idx = [j for j, t in enumerate(turns) if t["role"] == "model"]
        for k, e in enumerate(make_examples(turns, rd)):
            fault = is_fault(turns, model_idx[k], cfg)
            ctx = rd.tok(e["context"], add_special_tokens=False)["input_ids"]
            full = rd.tok(e["context"] + e["target"], add_special_tokens=False)["input_ids"]
            assert full[:len(ctx)] == ctx, (r["run_id"], r["id"], k)   # clean token boundary
            if len(full) > cfg.max_tokens:
                stats["dropped_too_long"] += 1
                continue
            stats["kept"] += 1
            if fault:
                stats[f"masked_{fault}"] += 1
            yield {"run_id": r["run_id"], "task": r["id"], "model": r["model"], "call": k,
                   "has_call": e["has_call"], "context": e["context"], "target": e["target"],
                   "n_ctx": len(ctx), "n_total": len(full), "train": not fault, "fault": fault}
    print("examples:", dict(stats))


# ----------------------------------------------------------------------------- stage 3: pack
def pack(examples: list[dict]) -> list[dict]:
    """Merge a trace's calls into one sequence while each context extends the text so far."""
    by_run = collections.defaultdict(list)
    for e in examples:
        by_run[(e["run_id"], e["task"])].append(e)
    packed = []
    for (run_id, task), es in by_run.items():
        seq = None
        for e in sorted(es, key=lambda e: e["call"]):
            if seq is not None and e["context"].startswith(seq["text"]):
                seq["text"] = e["context"] + e["target"]
            else:
                seq = {"run_id": run_id, "task": task, "model": e["model"],
                       "text": e["context"] + e["target"], "spans": []}
                packed.append(seq)
            seq["n_tokens"] = e["n_total"]
            if e.get("train", True):                     # masked calls: text kept, no loss span
                seq["spans"].append([e["n_ctx"], e["n_total"]])
    dropped = [s for s in packed if not s["spans"]]
    packed = [s for s in packed if s["spans"]]
    if dropped:
        print(f"{len(dropped)} sequences have no loss span left after masking: dropped")
    n_tok = sum(s["n_tokens"] for s in packed)
    n_loss = sum(b - a for s in packed for a, b in s["spans"])
    print(f"{len(packed)} sequences from {len(by_run)} traces | {n_tok} tokens | {n_loss} loss tokens")
    return packed


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["select", "build", "pack", "all"])
    ap.add_argument("--out", type=Path, default=ROOT / "runs/sft")
    ap.add_argument("--student", default="E2B", help="model being trained (E2B | E4B | 12B | 31B): sets the time budget")
    ap.add_argument("--ledger", type=Path, default=None,
                    help="run ledger to select from (default results/task_runs.jsonl); pass a frozen copy so later runs, "
                         "e.g. of a trained adapter, do not leak into the selection")
    ap.add_argument("--mask", action="store_true",
                    help="no loss on model-fault calls: FileEditError, ValueError, tool call leaked as text")
    args = ap.parse_args()
    cfg = DataConfig(student=args.student)
    if args.ledger:
        cfg.ledger = args.ledger
    if args.mask:
        cfg.mask_errors, cfg.mask_leaked_calls = frozenset({"FileEditError", "ValueError"}), True
    args.out.mkdir(parents=True, exist_ok=True)
    sel, ex, pk = args.out / "selected_runs.jsonl", args.out / "examples.jsonl", args.out / "packed.jsonl"

    if args.stage in ("select", "all"):
        write_jsonl(sel, select(cfg))
    if args.stage in ("build", "all"):
        write_jsonl(ex, build(read_jsonl(sel), cfg))
    if args.stage in ("pack", "all"):
        write_jsonl(pk, pack(read_jsonl(ex)))


if __name__ == "__main__":
    main()
