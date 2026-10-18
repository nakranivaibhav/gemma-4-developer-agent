"""Rebuild a run's ledger lines from what is on disk (probe_results.jsonl, test_outputs/, traces/, settings.json).

usage: python -I kits/swe-agent/run/ledger_rebuild.py <results_dir>
Use after a fix to the parsing in run_tasks.py / trace_stats.py: the lines for that run_id are replaced in
results/task_runs.jsonl, every other line is kept byte for byte. Fields only the live run knew (harness_error) are kept
from the old line when present.
"""
import fcntl
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_stats import summarize  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]  # the repo root
LEDGER = ROOT / "results" / "task_runs.jsonl"


def rebuild(results_dir):
    results_dir = Path(results_dir).resolve()
    settings = json.load(open(results_dir / "settings.json"))
    run_id = settings["run_id"]
    pinned = {k: v for k, v in settings.items() if k not in ("files", "tasks", "sub_agents", "tools")}
    pinned["file_sha"] = {k: v["sha"] for k, v in settings["files"].items()}
    rel = results_dir.relative_to(ROOT) if results_dir.is_relative_to(ROOT) else results_dir
    old = {}
    kept = []
    LEDGER.touch()
    lock = open(LEDGER, "a")  # held for the whole read -> replace, so a concurrent run's append cannot be lost
    fcntl.flock(lock, fcntl.LOCK_EX)
    if LEDGER.exists():
        for line in open(LEDGER):
            r = json.loads(line)
            (old.__setitem__(r["id"], r) if r.get("run_id") == run_id else kept.append(line))
    new = []
    for line in open(results_dir / "probe_results.jsonl"):
        row = json.loads(line)
        tid = row["id"]
        row.pop("tests_total", None)
        test_log = results_dir / "test_outputs" / f"{tid}.log"
        if test_log.exists():
            counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|errors?|skipped)\b", test_log.read_text())}
            row.update(tests_passed=counts.get("passed", 0), tests_failed=counts.get("failed", 0),
                       tests_errors=counts.get("errors", counts.get("error", 0)))
        row.setdefault("harness_error", (old.get(tid) or {}).get("harness_error"))
        try:
            stats = summarize(results_dir / "traces" / f"trace_{tid}.json", harness_error=row.get("harness_error") or row.get("error"),
                              budgets={"max_tool_calls": settings.get("budget_max_tool_calls")}, max_output_tokens=settings.get("max_output_tokens"))
        except Exception as exc:
            stats = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        row.update(run_id=run_id, bin=(old.get(tid) or {}).get("bin"),
                   trace=f"{rel}/traces/trace_{tid}.json", patch=f"{rel}/patches/{tid}.diff",
                   test_output=f"{rel}/test_outputs/{tid}.log", trace_stats=stats, settings=pinned)
        new.append(json.dumps(row) + "\n")
    with open(LEDGER, "r+") as fh:  # rewritten in place: appenders lock the same inode, so nothing slips through
        fh.seek(0), fh.truncate(), fh.write("".join(kept + new))
    lock.close()
    print(f"{run_id}: replaced {len(old)} lines with {len(new)}; {len(kept)} other lines kept")


if __name__ == "__main__":
    rebuild(sys.argv[1])
