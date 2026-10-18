"""Trace viewer: turn the run ledger + saved traces into a small static site (PART task-runs per JSON, loaded on click).

usage: envs/harness/bin/python -I analysis/trace_app.py build      -> runs/trace_app/site/{index.html,data/...}
       envs/harness/bin/python -I analysis/trace_app.py serve [port]  (build, then http://127.0.0.1:8765)

The page itself is runs/trace_app/page.html (hand-written, no build step). `build` reads results/task_runs.jsonl
(the last line per run_id + task wins), each line's trace / patch / test output, and data/tasks.jsonl for the problem
text and the real fix. Long tool results are cut to CUT characters with a note saying how much was cut, so a task
file stays small; the full trace remains on disk at the path shown in the page.
"""
import functools
import http.server
import json
import re
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the repo root
APP = ROOT / "runs" / "trace_app"
SITE = APP / "site"
CUT = 8000        # characters kept per string in a tool call or result
CUT_DOC = 60000   # characters kept for a patch / the test output / a prompt
PART = 10         # task-runs per data file: one published artifact version holds at most 511 files
ENDED = {
    "final": "Finished on its own",
    "context_overflow": "Stopped: the conversation outgrew the model's window",
    "budget_tool_calls": "Stopped: used up its tool calls",
    "output_truncated": "Stopped: a reply hit the output limit before any tool call",
    "aborted": "Stopped early by the harness",
    "budget_time": "Stopped: ran past its time limit",
}


def cut(s, limit=CUT):
    s = "" if s is None else str(s)
    return s if len(s) <= limit else s[:limit] + f"\n… [cut here: showing the first {limit:,} of {len(s):,} characters]"


def trim(x):
    if isinstance(x, str):
        return cut(x)
    if isinstance(x, dict):
        return {k: trim(v) for k, v in x.items()}
    if isinstance(x, list):
        return [trim(v) for v in x[:200]]
    return x


def files_in(diff):
    return sorted(set(re.findall(r"^\+\+\+ b/(\S+)", diff or "", re.M)))


def call_label(name, args):
    a = args if isinstance(args, dict) else {}
    if name == "run_command":
        return str(a.get("command", "")).strip().split("\n")[0][:200]
    path = a.get("filepath") or a.get("path") or a.get("file_path")
    if path:
        span = f"  (from line {a['start_line']})" if a.get("start_line") else ""
        return f"{path}{span}"
    if name == "submit_patch":
        return "hand in the patch"
    return json.dumps(a)[:200] if a else ""


def outcome(name, res):
    """(ok, one short phrase) for a parsed tool result."""
    if not isinstance(res, dict):
        return True, ""
    err = res.get("error_type") or res.get("error")
    if res.get("status") == "error" or err:
        msg = res.get("error_message") or (res.get("error") if isinstance(res.get("error"), str) else "") or ""
        if name == "run_command" and isinstance(res.get("details"), dict) and res["details"].get("exit_code") is not None:
            return False, f"exit {res['details']['exit_code']}"
        return False, (f"{res.get('error_type') or 'error'}: {msg}" if msg else str(res.get("error_type") or "error"))[:120]
    if "exit_code" in res:
        return res["exit_code"] == 0, f"exit {res['exit_code']}"
    if "total_lines" in res:
        return True, f"lines {res.get('start_line')}–{res.get('end_line')} of {res.get('total_lines')}"
    if "occurrences" in res:
        return True, f"{res['occurrences']} replacement(s)"
    if "patch_size" in res:
        return True, f"patch of {res['patch_size']:,} characters"
    if "size" in res:
        return True, f"{res['size']:,} characters written"
    return True, "ok"


def build_task(row, task):
    tpath = ROOT / row.get("trace", "")
    steps, prev_t = [], 0.0
    system_prompt = task_prompt = ""
    if tpath.exists():
        for s in json.load(open(tpath)).get("steps", []):
            ex = s.get("extra") or {}
            ev, t = ex.get("event_type"), float(ex.get("elapsed_s") or prev_t)
            if ev == "system_instruction":
                system_prompt = cut(s.get("message"), CUT_DOC)
                continue
            if ev == "task_prompt":
                task_prompt = cut(s.get("message"), CUT_DOC)
                continue
            m = s.get("metrics") or {}
            st = {"n": len(steps) + 1, "t": round(t, 1), "dt": round(max(0.0, t - prev_t), 1), "who": ex.get("author") or "",
                  "ptok": m.get("prompt_tokens"), "ctok": m.get("completion_tokens"), "note": cut(s.get("message")), "calls": []}
            prev_t = t
            obs = s.get("observation")
            raw = obs.get("content") if isinstance(obs, dict) else obs
            try:
                res = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                res = raw
            calls = s.get("tool_calls") or []
            for c in calls:
                name = c.get("function_name") or "?"
                ok, phrase = outcome(name, res) if len(calls) == 1 else (True, "")
                st["calls"].append({"tool": name, "label": call_label(name, c.get("arguments")), "args": trim(c.get("arguments")),
                                    "ok": ok, "outcome": phrase})
            if res is not None:
                st["result"] = trim(res)
            if calls:
                st["kind"] = "action"
            elif ev == "final":
                st["kind"] = "final"
            elif ev == "continuation_nudge":
                st["kind"] = "nudge"
            elif res is not None:
                st["kind"] = "subresult"
            else:
                st["kind"] = "talk"  # the model wrote text but called no tool
            steps.append(st)
    patch = (ROOT / row["patch"]).read_text(errors="replace") if row.get("patch") and (ROOT / row["patch"]).exists() else ""
    tout = (ROOT / row["test_output"]).read_text(errors="replace") if row.get("test_output") and (ROOT / row["test_output"]).exists() else ""
    stats, sett = row.get("trace_stats") or {}, row.get("settings") or {}
    ended = stats.get("ended_with") or ""
    ended_text = ENDED.get(ended, ended or "unknown")
    cap, used = sett.get("budget_max_tool_calls"), stats.get("budget_counted_calls")
    if ended == "final" and cap and used is not None and used >= cap:
        ended, ended_text = "final_at_budget", f"Used all {cap} of its tool calls" + (
            ", then handed in a patch" if stats.get("submit_patch_calls") else " and never handed in a patch")
    elif ended == "final" and not stats.get("submit_patch_calls"):
        ended_text = "Finished on its own, without calling submit_patch"
    return {
        "id": row["id"], "run": row.get("run_id"), "repo": (task or {}).get("repo", ""),
        "resolved": bool(row.get("resolved")), "test_exit": row.get("test_exit_code"),
        "tests": [row.get("tests_passed", 0), row.get("tests_failed", 0), row.get("tests_errors", 0)],
        "tool_calls": row.get("tool_calls"), "max_tool_calls": sett.get("budget_max_tool_calls"),
        "seconds": row.get("agent_seconds"), "ended": ended, "ended_text": ended_text,
        "harness_error": cut(row.get("harness_error") or row.get("error") or "", 600),
        "prompt_tokens": stats.get("prompt_tokens"), "completion_tokens": stats.get("completion_tokens"),
        "max_prompt_tokens": stats.get("max_prompt_tokens"), "max_output_tokens": sett.get("max_output_tokens"),
        "problem": cut((task or {}).get("problem_statement", ""), CUT_DOC),
        "system_prompt": system_prompt, "task_prompt": task_prompt, "steps": steps,
        "patch": cut(patch, CUT_DOC), "gold": cut((task or {}).get("patch", ""), CUT_DOC), "test_output": cut(tout, CUT_DOC),
        "files_model": files_in(patch), "files_gold": files_in((task or {}).get("patch", "")),
        "trace_path": row.get("trace", ""),
    }


def build():
    tasks = {}
    for line in open(ROOT / "data" / "tasks.jsonl"):
        t = json.loads(line)
        tasks[t["instance_id"]] = t
    latest = {}
    for line in open(ROOT / "results" / "task_runs.jsonl"):
        r = json.loads(line)
        if r.get("run_id") and r.get("id"):
            latest[(r["run_id"], r["id"])] = r
    runs, full = {}, {}
    data = SITE / "data"
    shutil.rmtree(data, ignore_errors=True)
    for (run, tid), r in latest.items():
        d = build_task(r, tasks.get(tid))
        full[(run, tid)] = d
        sett = r.get("settings") or {}
        info = runs.setdefault(run, {"run_id": run, "note": sett.get("note", ""), "tier": sett.get("tier", ""),
                                     "agent": sett.get("agent_name", ""), "max_output_tokens": sett.get("max_output_tokens"),
                                     "sandbox": (sett.get("sandbox") or {}).get("env", "shipped"), "tasks": []})
        info["tasks"].append({"id": tid, "resolved": d["resolved"], "calls": d["tool_calls"], "seconds": d["seconds"],
                              "ended": d["ended"], "tests": d["tests"], "steps": len(d["steps"])})
    order = sorted(runs.values(), key=lambda x: max((ROOT / "runs" / x["run_id"]).stat().st_mtime
                                                    if (ROOT / "runs" / x["run_id"]).exists() else 0, 0), reverse=True)
    for info in order:
        info["tasks"].sort(key=lambda t: t["id"])
        info["n"], info["resolved"] = len(info["tasks"]), sum(t["resolved"] for t in info["tasks"])
        for k in range(0, info["n"], PART):
            group = info["tasks"][k:k + PART]
            for t in group:
                t["part"] = k // PART
            out = data / info["run_id"] / f"part_{k // PART:02d}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps({t["id"]: full[(info["run_id"], t["id"])] for t in group}, separators=(",", ":")))
    data.mkdir(parents=True, exist_ok=True)
    (data / "index.json").write_text(json.dumps({"built_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "runs": order}))
    page = (APP / "page.html").read_text()
    (SITE / "index.html").write_text('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                                     '<meta name="viewport" content="width=device-width,initial-scale=1"></head><body>' + page + "</body></html>")
    files = sorted(str(p.relative_to(SITE)) for p in data.rglob("*.json"))
    (APP / "files.json").write_text(json.dumps({f: f for f in files}))
    size = sum((SITE / f).stat().st_size for f in files)
    print(f"built {len(full)} task-runs in {len(files) - 1} data files, {len(order)} runs, {size / 1e6:.1f} MB -> {SITE}")
    for info in order:
        print(f"  {info['run_id']:18s} {info['resolved']}/{info['n']} resolved")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    build()
    if cmd == "serve":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8765
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SITE))
        print(f"serving http://127.0.0.1:{port}  (ctrl-c to stop; re-run to pick up new runs)")
        http.server.ThreadingHTTPServer(("127.0.0.1", port), handler).serve_forever()
