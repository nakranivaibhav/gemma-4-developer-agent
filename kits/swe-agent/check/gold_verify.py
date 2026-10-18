"""Gold check through the REAL harness verifier: can the real fix pass each public task in this environment?

usage: [PAR=4] [SHARD=i/n] [NO_PATCH=1] [WHEELS_DIR=..] [IMAGE=..] [NETWORK=..] [KAGGLE_SANDBOX_DIR=..] [TMPDIR=..] \
       envs/harness/bin/python -I kits/swe-agent/check/gold_verify.py <out_dir> [task_id ... | --prefix fastapi_ | --only-failed <prev.jsonl>]
NO_PATCH=1 runs the same check with an EMPTY patch: a task that still resolves needs no fix at all (a free pass), so a
sound environment gives 129/129 with the real fix and 0/129 without it. SHARD=i/n takes every n-th task (run n processes).
Just before pytest, a diagnostic runs in the same container and lands in <out_dir>/diag/<id>.txt: where the repo's own
package is imported from, key library versions, and the local repair log.

No model: swegemma's own verify_task() is called with agent_patch = the task's real fix, so everything the scorer does
(wheel injection, sandbox/setup.py, protected-file stripping, test patch, pytest command, JUnit validation) is the
harness's code, not a re-implementation. A task that fails here cannot be resolved by any agent in this environment.
Environment knobs are the harness's own: WHEELS_DIR (unpacked into site-packages; cached under $TMPDIR by a FIXED
name, so use one TMPDIR per wheel set), IMAGE (its /wheels is what pip can install from in the container),
KAGGLE_SANDBOX_DIR (directory holding the setup.py that runs inside every container).
Writes <out_dir>/gold.jsonl (one line per task) and <out_dir>/logs/<id>.log (full pytest output).
"""
import asyncio
import contextvars
import json
import os
import re
import sys
import time
from pathlib import Path

from adk_submission import VllmConfig, VllmServer
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.deduplication import resolve_task_snapshot_paths
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import verify_task
from swegemma.models import load_tasks

ROOT = Path(__file__).resolve().parents[3]  # the repo root
data = ROOT / "data"
out_dir = Path(sys.argv[1]).resolve()
args = sys.argv[2:]
(out_dir / "logs").mkdir(parents=True, exist_ok=True)
OUT = out_dir / "gold.jsonl"

tasks = {t.instance_id: t for t in load_tasks(data / "tasks.jsonl")}
if "--prefix" in args:
    ids = [t for t in tasks if t.startswith(args[args.index("--prefix") + 1])]
elif "--only-failed" in args:
    ids = [json.loads(l)["id"] for l in open(args[args.index("--only-failed") + 1]) if not json.loads(l)["resolved"]]
else:
    ids = args or list(tasks)
if os.environ.get("SHARD"):
    _i, _n = map(int, os.environ["SHARD"].split("/"))
    ids = [t for k, t in enumerate(sorted(ids)) if k % _n == _i]
NO_PATCH = os.environ.get("NO_PATCH") == "1"
(out_dir / "diag").mkdir(exist_ok=True)
CUR = contextvars.ContextVar("tid", default=None)  # asyncio.to_thread copies the context, so exec() knows its task

server = VllmServer(VllmConfig(model=str(data / "external" / "model"), port=8000, host="127.0.0.1"), adapter_manifest={})
limits, gen_constraints = build_submission_limits()
cfg = EvalConfig(
    tasks_path=data / "tasks.jsonl", snapshots_dir=data / "snapshots", results_dir=out_dir,
    submission_dir=ROOT / "agents" / "sample",
    models=server.create_model_registry(aliases=["gemma-4-31b-it-qat-w4a16-ct"], model_prefix="openai/", api_key="EMPTY"),
    sandbox="docker", image=os.environ.get("IMAGE", "swebench-sandbox:latest"),
    limits=limits, generation_constraints=gen_constraints,
    wheels_dir=Path(os.environ.get("WHEELS_DIR", data / "wheels")).resolve(), verbose=False, display_mode="quiet",
)
evaluator = Evaluator(cfg)
if os.environ.get("NETWORK"):  # the harness's ContainerConfig default is 'none'; a named docker network replaces it
    evaluator.docker.config.network_mode = os.environ["NETWORK"]
_exec = evaluator.docker.exec


def _exec_with_diag(container_id, cmd, *a, **k):
    tid = CUR.get()
    if tid and "-m pytest" in str(cmd):
        pkg = tasks[tid].repo.rsplit("/", 1)[-1].lower()
        diag = ("cd /workspace && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 python3 -s -c \"import " + pkg + ", sys; print('OWN_PACKAGE', " + pkg +
                ".__file__)\" 2>&1 | tail -1; pip list --format=freeze 2>/dev/null | grep -iE "
                "'^(starlette|pydantic|pygments|urllib3|httpx|httpx2|markdown.it.py|pytest|attrs|anyio|requests|rich|fastapi|typing.extensions|inline.snapshot|pytest.httpbin|httpbin|flask|werkzeug)=' | tr '\\n' ' '; "
                "echo; cat /tmp/local_repair.json 2>/dev/null; echo; cat /workspace/pytest.ini 2>/dev/null | head -12")
        r = _exec(container_id, diag)
        with open(out_dir / "diag" / f"{tid}.txt", "a") as fh:
            fh.write((r.stdout or "") + ((r.stderr or "") and "\n--- stderr\n" + r.stderr) + "\n=====\n")
    return _exec(container_id, cmd, *a, **k)


evaluator.docker.exec = _exec_with_diag
ENV = {"wheels_dir": str(cfg.wheels_dir), "image": cfg.image, "sandbox_dir": os.environ.get("KAGGLE_SANDBOX_DIR", ""),
       "tmpdir": os.environ.get("TMPDIR", ""), "network": evaluator.docker.config.network_mode}


async def one(tid, sem):
    async with sem:
        task = tasks[tid]
        CUR.set(tid)
        t0 = time.time()
        try:
            snap, base_snap, patch_path = resolve_task_snapshot_paths(cfg.snapshots_dir, tid, task.repo)
            r = await verify_task(evaluator.docker, cfg, task, snap, base_snapshot_path=base_snap, patch_path=patch_path,
                                  agent_patch="" if NO_PATCH else task.patch, start_time=time.perf_counter())
            out = r.test_output or ""
            (out_dir / "logs" / f"{tid}.log").write_text(out)
            summ = re.findall(r"^(?:=+ )?((?:\d+ \w+(?:, )?)+ in [\d.]+s)", out, re.M)
            first = re.search(r"^E +(\S.*)$", out, re.M)
            row = {"id": tid, "resolved": bool(r.resolved), "exit": r.test_exit_code, "summary": summ[-1] if summ else "",
                   "first_error": first.group(1)[:220] if first else None, "harness_error": (r.error or None) and str(r.error)[:300]}
        except Exception as exc:
            row = {"id": tid, "resolved": False, "exit": None, "summary": "", "first_error": None,
                   "harness_error": f"{type(exc).__name__}: {exc}"[:300]}
        row["seconds"] = round(time.time() - t0, 1)
        with open(OUT, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"{tid:15s} {'PASS' if row['resolved'] else 'FAIL'} {row['summary'][:44]:44s} {(row['first_error'] or row['harness_error'] or '')[:110]}", flush=True)
        return row


async def main():
    sem = asyncio.Semaphore(int(os.environ.get("PAR", "16")))
    rows = await asyncio.gather(*(one(t, sem) for t in ids))
    (out_dir / "env.json").write_text(json.dumps(ENV, indent=1))
    by = {}
    for r in rows:
        repo = r["id"].rsplit("_", 1)[0]
        by.setdefault(repo, [0, 0])
        by[repo][0] += r["resolved"]
        by[repo][1] += 1
    print(f"GOLD_DONE {sum(r['resolved'] for r in rows)}/{len(rows)} " + " ".join(f"{k}={a}/{b}" for k, (a, b) in sorted(by.items())), flush=True)


asyncio.run(main())
