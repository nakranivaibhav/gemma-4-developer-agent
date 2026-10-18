"""One table comparing the Gemma 4 sizes on the same tasks, scored the Kaggle way.

For each model and pass, over the tasks every listed model ran:
  solved            resolved, no time limit
  within cap        resolved AND predicted Kaggle agent time <= cap (default 330 s = 6 min - 30 s setup); a cut task is unsolved
  out tok           median output tokens per task (thinking + tool-call text), from the traces
  ctx crash         tasks that died with "context window exceeded"
  120-task h        predicted Kaggle hours for 120 tasks like these with the cap applied (+30 s setup each), 12 h allowed
Kaggle time comes from kaggle_time.py with each model's 4x L4 profile; it uses token counts only, so the machine
the runs were made on does not matter. A row with a harness error counts as unsolved; its time comes from its trace.

usage: model_report.py [--cap-s 330] [--tasks usable]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kits" / "swe-agent" / "run"))  # kit helpers
import manifest  # noqa: E402
from kaggle_time import load_profiles, task_minutes, trace_calls  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # the repo root
MODELS = [  # name, Kaggle profile, run ids per pass (a pass may span several runs)
    ("31B + adapter", "l4x4_31b_adapter", [["l100_lean_p1"], ["l100_lean_p2"]]),
    ("12B", "l4x4_12b_nolora", [["v_12b_lean_p1", "v_12b_lean_rest_p1"], ["v_12b_lean_p2", "v_12b_lean_rest_p2"]]),
    ("E4B", "l4x4_e4b_nolora", [["v_e4b_lean_p1"], ["v_e4b_lean_p2"]]),
    ("E2B", "l4x4_e2b_nolora", [["v_e2b_lean_p1"], ["v_e2b_lean_p2"]]),
    # reference only: not submittable, no Kaggle profile, so no within-cap or hours
    ("Qwen 27B ref", None, [["v_qwen_lean_p1"], ["v_qwen_lean_p2"]]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap-s", type=float, default=330.0)
    ap.add_argument("--tasks", default="usable")
    a = ap.parse_args()
    rows = {}
    for r in map(json.loads, open(ROOT / "results" / "task_runs.jsonl")):
        rows[(r.get("run_id"), r["id"])] = r  # a rerun of the same task in the same run replaces the earlier row
    want = set(manifest.ids(a.tasks))
    profiles = load_profiles()
    table = []
    for name, prof, passes in MODELS:
        p = profiles[prof] if prof else None
        for i, runs in enumerate(passes, 1):
            got = {tid: r for (rid, tid), r in rows.items() if rid in runs and tid in want}
            if not got:
                continue
            solved = inside = crash = 0
            outs, mins = [], []
            for tid, r in got.items():
                err = str(r.get("harness_error") or r.get("error") or "")
                crash += "ContextWindow" in err
                tp = ROOT / r["trace"] if r.get("trace") else None
                s = None
                if tp and tp.exists():
                    calls, tools = trace_calls(tp)
                    outs.append(sum(o for _, _, o in calls))
                    if p:
                        s = task_minutes(p, calls, tools) * 60 - p["setup_min"] * 60
                        mins.append((min(s, a.cap_s) + p["setup_min"] * 60) / 60)
                ok = bool(r.get("resolved"))
                solved += ok
                inside += ok and s is not None and s <= a.cap_s
            hours = st.mean(mins) * 120 / 60 if mins else float("nan")
            table.append((name, i, len(got), solved, inside if p else None, st.median(outs) if outs else 0, crash, hours))
    print(f"cap {a.cap_s:g} Kaggle-s per task | tasks from {a.tasks} ({len(want)})")
    print(f"{'model':14s} pass  tasks   solved          within cap      out tok  ctx crash  120-task h")
    for name, i, n, s, w, o, c, h in table:
        wc = f"{w:3d} ({100*w/n:4.1f}%)" if w is not None else f"{'-':>11s}"
        print(f"{name:14s} {i:4d} {n:6d}   {s:3d} ({100*s/n:4.1f}%)   {wc}   {o/1000:6.1f}k  {c:9d}  {h:9.1f}")


if __name__ == "__main__":
    main()
