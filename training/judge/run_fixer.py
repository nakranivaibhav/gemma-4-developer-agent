"""Repair flagged tasks with a requirements addendum, then build fixed packets for a blind re-judge.

usage: envs/judge/bin/python training/judge/run_fixer.py --run fix_pilot --verdicts all_v3 --ids-file F
           [--packets DIR] [--concurrency 4] [--cap-5h 0.90]
then:  envs/judge/bin/python training/judge/run_judge.py --run fix_pilot_rejudge --packets <FIXED> --ids-file <fixable>

Per task (only tasks the judge run --verdicts called unsolvable and with no R6 yes):
  1. fixer (Opus, training/judge/fixer_prompt.md) reads the task, the judge's verdict (JUDGE.md) and the repo, and
     returns {fixable, reason, addendum, gaps_covered};
  2. leak check (Opus, a second blind call) reads only the original issue, the addendum and the reference fix, and
     answers whether the addendum gives away the implementation;
  3. a fixed packet: TASK.md with the addendum appended to the issue, the same diffs, the repo hard-linked (no copy).
Output runs/judge/<run>/<id>.json, fixed packets in /home/vaibhav/data/gemma4/judge/packets_fixed/<id>/.
The original task text is never changed; the addendum is stored on its own.
"""
import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

sys.path.insert(0, str(Path(__file__).parent))
from usage_guard import Cap  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/home/vaibhav/data/gemma4/judge")
ITEMS = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
FIX_SCHEMA = {"type": "object", "properties": {
    "fixable": {"type": "string", "enum": ["yes", "no"]}, "reason": {"type": "string"},
    "addendum": {"type": "string"}, "gaps_covered": {"type": "array", "items": {"type": "string"}}},
    "required": ["fixable", "reason", "addendum", "gaps_covered"], "additionalProperties": False}
LEAK_SCHEMA = {"type": "object", "properties": {
    "leaks_implementation": {"type": "string", "enum": ["yes", "no"]}, "evidence": {"type": "string"}},
    "required": ["leaks_implementation", "evidence"], "additionalProperties": False}
TRIM_SCHEMA = {"type": "object", "properties": {
    "lines": {"type": "array", "items": {"type": "object", "properties": {
        "text": {"type": "string"}, "keep": {"type": "string", "enum": ["yes", "no"]}, "reason": {"type": "string"}},
        "required": ["text", "keep", "reason"], "additionalProperties": False}},
    "addendum": {"type": "string"}},
    "required": ["lines", "addendum"], "additionalProperties": False}
TRIM_PROMPT = """You trim a requirements addendum that was added to a coding task's issue to make the task fair. Keep the
task as hard as intended: the addendum must hold ONLY what a solver truly cannot do without.
Files: TASK.md (the original issue), ADDENDUM.md (the added requirements), TEST_CHANGES.diff and REQUIRED_TESTS.txt
(when present; the tests a fix must pass). The repository at the base commit is at {REPO} (read-only).
For each requirement line (and each part of a line), keep it only if BOTH hold:
 (a) a developer who follows the issue and reads the repo could not work it out (search the repo before deciding), and
 (b) without it, a fix that follows the issue would fail a required test (find the test line).
Drop: behaviour that simply stays the same, details beyond what the tests check (full signatures when the name settles
it, defaults or types the tests do not depend on), internal names no required test imports or calls, restatements of
the issue. Shorten a line when only part of it is needed. Return every original line with keep yes/no and the reason
(quote the test line for a keep), then the trimmed addendum: the line `Requirements:` followed by the kept bullets, or
an empty string if nothing is needed. Answer with the JSON object only."""
LEAK_PROMPT = """You check one thing: does a requirements addendum, written to make a coding task fair, give away how
to implement the fix? Files: ISSUE.md (the original issue), ADDENDUM.md (the added requirements), REFERENCE_FIX.diff.
Allowed in the addendum: names, signatures, parameters, exact messages or output formats, inputs and expected results,
behaviour a user can observe. Not allowed: describing the implementation (which internal function to change, the
algorithm, data structures, control flow), naming internal helpers the user never touches, or pasting code from the
reference fix beyond a signature or exact string. Answer leaks_implementation yes only with a quote from the addendum
that crosses that line. Answer with the JSON object only."""


async def ask(prompt, sysp, cwd, schema, a, add_dirs=()):
    opts = ClaudeAgentOptions(model=a.model, cwd=str(cwd), system_prompt=sysp, allowed_tools=["Read", "Grep", "Glob"],
                              disallowed_tools=["Bash", "Write", "Edit", "MultiEdit", "NotebookEdit", "WebFetch",
                                                "WebSearch", "Task", "Agent", "TodoWrite"],
                              setting_sources=[], max_turns=a.max_turns, effort=a.effort, add_dirs=list(add_dirs),
                              output_format={"type": "json_schema", "schema": schema})
    res, err = None, None
    try:
        async for m in query(prompt=prompt, options=opts):
            if isinstance(m, ResultMessage):
                res = m
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    if res is not None and res.is_error and not err:
        err = f"result error: {str(res.result)[:300]}"
    return (res.structured_output if res else None), (res.total_cost_usd if res else 0), err


def judge_md(v):
    lines = ["# Reviewer's verdict", "", f"Summary: {v['summary']}", ""]
    for k in ITEMS:
        lines.append(f"- {k}: {v[k]['answer']} -- {v[k]['evidence']}")
    return "\n".join(lines) + "\n"


def with_addendum(task_md, addendum):
    head, sep, tail = task_md.partition("\n## Hints")
    return head.rstrip() + "\n\n" + addendum.strip() + "\n" + sep + tail


async def fix_one(tid, verdict, a, fixer_sysp):
    src = Path(a.packets) / tid
    work = DATA / "fixwork" / tid
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for f in ("TASK.md", "REFERENCE_FIX.diff", "TEST_CHANGES.diff", "REQUIRED_TESTS.txt"):
        if (src / f).exists():
            shutil.copy(src / f, work / f)
    (work / "JUDGE.md").write_text(judge_md(verdict))
    repo = (src / "repo").resolve()
    t0 = time.time()
    fix, c1, err = await ask(f"Repair task {tid}. Start with TASK.md and JUDGE.md.",
                             fixer_sysp.replace("{REPO}", str(repo)), work, FIX_SCHEMA, a, add_dirs=[repo])
    out = {"id": tid, "fix": fix, "error": err, "cost_usd": c1 or 0}
    final = None
    if fix and fix["fixable"] == "yes" and fix["addendum"].strip():
        tw = DATA / "trimwork" / tid
        shutil.rmtree(tw, ignore_errors=True)
        tw.mkdir(parents=True)
        for f in ("TASK.md", "TEST_CHANGES.diff", "REQUIRED_TESTS.txt"):
            if (src / f).exists():
                shutil.copy(src / f, tw / f)
        (tw / "ADDENDUM.md").write_text(fix["addendum"].strip() + "\n")
        trim, ct, errt = await ask("Trim the addendum. Read ADDENDUM.md, TASK.md and the tests.",
                                   TRIM_PROMPT.replace("{REPO}", str(repo)), tw, TRIM_SCHEMA, a, add_dirs=[repo])
        out.update(trim=trim, trim_error=errt, cost_usd=out["cost_usd"] + (ct or 0))
        final = (trim or {}).get("addendum", "").strip() if trim else fix["addendum"].strip()
        out["final_addendum"] = final
    if final:
        lw = DATA / "leakwork" / tid
        shutil.rmtree(lw, ignore_errors=True)
        lw.mkdir(parents=True)
        issue = (src / "TASK.md").read_text().split("## Issue", 1)[-1].split("\n## Hints", 1)[0]
        (lw / "ISSUE.md").write_text(issue.strip() + "\n")
        (lw / "ADDENDUM.md").write_text(final + "\n")
        shutil.copy(src / "REFERENCE_FIX.diff", lw / "REFERENCE_FIX.diff")
        leak, c2, err2 = await ask("Check the addendum. Read all three files.", LEAK_PROMPT, lw, LEAK_SCHEMA, a)
        out.update(leak=leak, leak_error=err2, cost_usd=out["cost_usd"] + (c2 or 0))
        # two packets: the trimmed addendum (what we want) and the full one (fallback if the trimmed one fails re-judge)
        for root, text in ((Path(a.fixed), final), (Path(a.fixed + "_full"), fix["addendum"])):
            dst = root / tid
            shutil.rmtree(dst, ignore_errors=True)
            dst.mkdir(parents=True)
            for f in ("REFERENCE_FIX.diff", "TEST_CHANGES.diff", "REQUIRED_TESTS.txt"):
                if (src / f).exists():
                    shutil.copy(src / f, dst / f)
            (dst / "TASK.md").write_text(with_addendum((src / "TASK.md").read_text(), text))
            subprocess.run(["cp", "-al", str(repo), str(dst / "repo")], check=True)  # hard links: no extra disk
    out["seconds"] = round(time.time() - t0, 1)
    return out


async def main(a):
    out_dir = ROOT / "runs/judge" / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    vdir = ROOT / "runs/judge" / a.verdicts
    ids = [l.strip() for l in open(a.ids_file) if l.strip()]
    todo = asyncio.Queue()
    for t in ids:
        vf = vdir / f"{t}.json"
        if (out_dir / f"{t}.json").exists() or not vf.exists():
            continue
        v = json.loads(vf.read_text()).get("verdict")
        if not v or v["R7"]["answer"] != "no" or v["R6"]["answer"] == "yes":
            continue
        todo.put_nowait((t, v))
    print(f"{a.run}: {todo.qsize()} flagged tasks to repair", flush=True)
    fixer_sysp = Path(a.prompt).read_text()
    cap = Cap(a.cap_5h, a.cap_7d, log=lambda s: print(s, flush=True))

    async def worker():
        while not todo.empty():
            await cap.wait()
            if todo.empty():
                break
            t, v = await todo.get()
            o = await fix_one(t, v, a, fixer_sysp)
            if o["error"] and "limit" in o["error"].lower() and not o["cost_usd"]:
                print(f"LIMIT {t}: {o['error'][:120]} -> pausing 20 min", flush=True)
                todo.put_nowait((t, v))
                await asyncio.sleep(1200)
                continue
            (out_dir / f"{t}.json").write_text(json.dumps(o, indent=1))
            f = o.get("fix") or {}
            nl = lambda x: sum(1 for l in (x or "").splitlines() if l.strip().startswith(("-", "*")))
            print(f"DONE {t} fixable={f.get('fixable')} bullets {nl(f.get('addendum'))}->{nl(o.get('final_addendum'))} "
                  f"leak={(o.get('leak') or {}).get('leaks_implementation')} "
                  f"${o['cost_usd']:.3f} {o['seconds']}s err={bool(o['error'])}", flush=True)

    await asyncio.gather(*[worker() for _ in range(a.concurrency)])
    print("FIX_DONE", a.run, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--verdicts", required=True, help="judge run whose verdicts pick and inform the repairs")
    ap.add_argument("--ids-file", required=True)
    ap.add_argument("--packets", default=str(DATA / "packets"))
    ap.add_argument("--fixed", default=str(DATA / "packets_fixed"))
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--max-turns", type=int, default=60)
    ap.add_argument("--prompt", default=str(ROOT / "training/judge/fixer_prompt.md"))
    ap.add_argument("--cap-5h", type=float, default=0.90)
    ap.add_argument("--cap-7d", type=float, default=1.0)
    asyncio.run(main(ap.parse_args()))
