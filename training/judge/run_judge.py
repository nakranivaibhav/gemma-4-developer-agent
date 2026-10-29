"""Run the Opus task-quality judge over task packets (training/judge/build_packets.py), one JSON verdict per task.

usage:
  envs/judge/bin/python training/judge/run_judge.py --run judge_v1_r1 --ids-file F [--packets DIR] [--concurrency 4]
         [--model claude-opus-5-5] [--effort medium] [--prompt training/judge/judge_prompt.md]

Output: runs/judge/<run>/<id>.json  {id, verdict (R1..R7 + summary), cost_usd, usage, turns, seconds, error}
        runs/judge/<run>/run.json    the settings of the run (model, effort, prompt sha, rubric sha)
A task with an existing verdict file is skipped, so a stopped run resumes where it left off.

Auth: the local Claude Code login (no API key). The judge's tools are Read / Grep / Glob inside the packet folder;
no shell, no writes, no web, and no user or project settings are loaded (so no hooks, no CLAUDE.md).

Usage limits: a limit hit arrives as a failure that cost nothing (same signature as in agent-video-v2's seat logic).
The runner then pauses ALL workers (not just the one), waits --pause-minutes, and retries the task; the task is
put back, never marked failed. Other failures are recorded in the verdict file's `error` and the run moves on.

Usage cap (2026-10-24, user: "cap 5h at 80%"): before each task the runner reads the live 5-hour and 7-day utilization
(training/judge/usage_guard.py, at most once per --probe-seconds). At or above --cap-5h (default 0.90; user 2026-10-24 raised it from 0.80) or --cap-7d no
new task starts until that window resets; tasks already running finish (8 in flight can add a point or two). A
heartbeat line every 10 minutes keeps the watchdog from reading the wait as a stall.
"""
import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query
from usage_guard import Cap

ROOT = Path(__file__).resolve().parents[2]
PACKETS = Path("/home/vaibhav/data/gemma4/judge/packets")
ITEMS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
# Flat keys (R1, R1_evidence, ...): a nested {answer, evidence} object per item broke the structured-output call in
# 48 of 152 calibration-v1 calls; flat strings do not. nest() turns it back into {R1: {answer, evidence}, ...}.
SCHEMA = {
    "type": "object",
    "properties": {
        **{k: {"type": "string", "enum": ["yes", "no"]} for k in ITEMS},
        **{f"{k}_evidence": {"type": "string"} for k in ITEMS},
        "summary": {"type": "string"},
    },
    "required": ITEMS + [f"{k}_evidence" for k in ITEMS] + ["summary"],
    "additionalProperties": False,
}


def nest(v):
    if not v:
        return v
    return {**{k: {"answer": v[k], "evidence": v.get(f"{k}_evidence", "")} for k in ITEMS},
            "summary": v.get("summary", "")}


LIMIT_WORDS = ("limit", "rate", "quota", "overloaded", "429", "usage")


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def system_prompt(prompt_path):
    rubric = (ROOT / "docs/judge_rubric.md").read_text()
    rubric = rubric.split("-->", 1)[-1].strip()  # drop the file's contract comment
    return (Path(prompt_path).read_text().replace("{RUBRIC}", rubric)
            .replace("{TODAY}", time.strftime("%Y-%m-%d", time.gmtime())))


async def judge_one(tid, a, sysp):
    opts = ClaudeAgentOptions(
        model=a.model,
        cwd=str(Path(a.packets) / tid),
        system_prompt=sysp,
        allowed_tools=["Read", "Grep", "Glob"],
        disallowed_tools=["Bash", "Write", "Edit", "MultiEdit", "NotebookEdit", "WebFetch", "WebSearch", "Task",
                          "Agent", "TodoWrite"],
        setting_sources=[],
        max_turns=a.max_turns,
        output_format={"type": "json_schema", "schema": SCHEMA},
        effort=a.effort,
    )
    t0, res, err = time.time(), None, None
    try:
        async for m in query(prompt=f"Review task {tid} by the rubric. Start with TASK.md.", options=opts):
            if isinstance(m, ResultMessage):
                res = m
    except Exception as e:  # transport errors, CLI exits
        err = f"{type(e).__name__}: {e}"
    out = {"id": tid, "seconds": round(time.time() - t0, 1), "error": err}
    if res is not None:
        out.update(cost_usd=res.total_cost_usd, usage=res.usage, turns=res.num_turns,
                   model_usage=res.model_usage, api_error_status=res.api_error_status,
                   verdict=nest(res.structured_output))
        if res.is_error and not out["error"]:
            out["error"] = f"result error: {str(res.result)[:500]}"
        if out.get("verdict") is None and not out["error"]:
            out["error"] = f"no structured output: {str(res.result)[:500]}"
    return out


def looks_like_limit(o):
    if o.get("api_error_status") == 429:
        return True
    return bool(o["error"]) and not (o.get("cost_usd") or 0) and any(w in o["error"].lower() for w in LIMIT_WORDS)


async def main(a):
    out_dir = ROOT / "runs/judge" / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = [l.strip() for l in open(a.ids_file) if l.strip()]
    sysp = system_prompt(a.prompt)
    (out_dir / "run.json").write_text(json.dumps({
        "model": a.model, "effort": a.effort, "prompt": a.prompt, "prompt_sha": sha(a.prompt),
        "rubric_sha": sha(ROOT / "docs/judge_rubric.md"), "packets": a.packets, "n": len(ids),
        "started": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime())}, indent=1))
    todo = asyncio.Queue()
    for t in ids:
        f = out_dir / f"{t}.json"
        if not (f.exists() and json.loads(f.read_text()).get("verdict")):  # redo tasks that ended without a verdict
            todo.put_nowait(t)
    print(f"{a.run}: {todo.qsize()} to judge ({len(ids) - todo.qsize()} already done)", flush=True)
    gate = asyncio.Event()
    gate.set()
    cap = Cap(a.cap_5h, a.cap_7d, every=a.probe_seconds, log=lambda m: print(m, flush=True))

    async def worker():
        while not todo.empty():
            await gate.wait()
            await cap.wait()
            if todo.empty():
                break
            t = await todo.get()
            o = await judge_one(t, a, sysp)
            if looks_like_limit(o):
                print(f"LIMIT {t}: {o['error'][:160]} -> pausing {a.pause_minutes} min", flush=True)
                todo.put_nowait(t)  # back in the queue, not failed
                if gate.is_set():
                    gate.clear()
                    await asyncio.sleep(a.pause_minutes * 60)
                    gate.set()
                continue
            (out_dir / f"{t}.json").write_text(json.dumps(o, indent=1))
            v = o.get("verdict") or {}
            flags = "".join(k[1] for k in ITEMS[:6] if (v.get(k) or {}).get("answer") == "yes")
            print(f"DONE {t} R7={(v.get('R7') or {}).get('answer')} yes=[{flags}] ${o.get('cost_usd') or 0:.3f} "
                  f"{o['seconds']}s err={bool(o['error'])}", flush=True)

    await asyncio.gather(*[worker() for _ in range(a.concurrency)])
    print("JUDGE_DONE", a.run, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--packets", default=str(PACKETS))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--prompt", default=str(ROOT / "training/judge/judge_prompt.md"))
    ap.add_argument("--pause-minutes", type=float, default=20)
    ap.add_argument("--cap-5h", type=float, default=0.90)
    ap.add_argument("--cap-7d", type=float, default=1.0)
    ap.add_argument("--probe-seconds", type=float, default=60)
    asyncio.run(main(ap.parse_args()))
