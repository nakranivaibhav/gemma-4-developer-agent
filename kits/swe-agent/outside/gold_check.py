"""Gold / empty check for outside tasks: does the reference fix resolve each task in its own image, and does an
empty patch fail? (The outside counterpart of kits/swe-agent/check/gold_verify.py.)

usage: [GOLD_RUNS=2] [PAR=2] [NO_EMPTY=1] [NETWORK=gemma-tarpit] [OUTSIDE_VERIFY_TIMEOUT=1800] \
       envs/harness/bin/python -I kits/swe-agent/outside/gold_check.py <tasks.jsonl> <out_dir> [instance_id ...]
No model, no agent: verify_outside() (kits/swe-agent/outside/outside.py) is called with agent patch = the task's
reference fix (GOLD_RUNS times, to see flaky tests) and once with an empty patch. Same containers as a real run:
the task's image, the harness's ContainerManager limits (4 GB, 2 CPUs), the offline network.
A sound task gives gold resolved on every run and empty not resolved. `strict` is the Kaggle verifier's extra rule
(exit 0, nothing failed or errored in the whole run); tasks whose test files hold known-failing tests miss it.
Writes <out_dir>/gold.jsonl (one line per run) and <out_dir>/logs/<id>.<kind><n>.log (full test output).
Every image must be pulled first (`docker pull <docker_image>`); a missing image is reported, never pulled.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from swegemma.models import load_tasks
from swegemma.sandbox import ContainerConfig, ContainerManager

sys.path.insert(0, str(Path(__file__).resolve().parent))
import outside  # noqa: E402

tasks_path, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
specs = outside.load_specs(tasks_path)
tasks = {t.instance_id: t for t in load_tasks(tasks_path) if t.instance_id in specs}
ids = sys.argv[3:] or list(tasks)
RUNS = int(os.environ.get("GOLD_RUNS", "2"))
NO_EMPTY = os.environ.get("NO_EMPTY") == "1"
(out_dir / "logs").mkdir(parents=True, exist_ok=True)
OUT = out_dir / "gold.jsonl"

outside.install(specs)  # only the container image/env swap is used here; the agent hooks stay idle
# the image named here is never used (every container is an outside task's); it only has to exist for the resolver
docker = ContainerManager(ContainerConfig(image=os.environ.get("IMAGE", "swebench-sandbox-local:latest"),
                                          network_mode=os.environ.get("NETWORK", "gemma-tarpit")))


async def one_run(tid, kind, n):
    task, row = tasks[tid], specs[tid]
    t0 = time.time()
    v = await outside.verify_outside(docker, task, row, task.patch if kind == "gold" else "")
    (out_dir / "logs" / f"{tid}.{kind}{n}.log").write_text((v.get("test_cmd") or "") + "\n" + (v.get("test_output") or ""))
    rec = {"id": tid, "kind": kind, "run": n, "resolved": v["resolved"], "strict": v["strict"],
           "exit": v["test_exit_code"], "f2p": f"{v.get('f2p_pass')}/{v.get('f2p_total')}",
           "p2p": f"{v.get('p2p_pass')}/{v.get('p2p_total')}", "counts": v.get("counts"),
           "not_passed": v.get("not_passed"), "head_ok": v.get("head_ok"), "links": v.get("links"), "patch_stripped": v.get("patch_stripped"),
           "error": v.get("error"), "seconds": round(time.time() - t0, 1), "image": row["docker_image"]}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"{tid:40s} {kind}{n} {'PASS' if rec['resolved'] else 'FAIL'} strict={rec['strict']!s:5s} f2p={rec['f2p']} "
          f"p2p={rec['p2p']} {rec['seconds']}s {(rec['error'] or '')[:100]}", flush=True)
    return rec


async def main():
    missing = [t for t in ids if t not in tasks]
    if missing:
        sys.exit(f"not outside tasks in {tasks_path}: {missing}")
    sem = asyncio.Semaphore(int(os.environ.get("PAR", "2")))

    async def guarded(*a):
        async with sem:
            return await one_run(*a)

    jobs = [(t, "gold", n) for t in ids for n in range(1, RUNS + 1)] + ([] if NO_EMPTY else [(t, "empty", 1) for t in ids])
    rows = await asyncio.gather(*(guarded(*j) for j in jobs))
    ok = 0
    for t in ids:
        g = [r for r in rows if r["id"] == t and r["kind"] == "gold"]
        e = [r for r in rows if r["id"] == t and r["kind"] == "empty"]
        sound = all(r["resolved"] for r in g) and not any(r["resolved"] for r in e)
        ok += sound
        print(f"TASK {t} gold={sum(r['resolved'] for r in g)}/{len(g)} empty={'FAIL' if not any(r['resolved'] for r in e) else 'PASS'}"
              f"{'' if e else '(not run)'} strict_gold={sum(r['strict'] for r in g)}/{len(g)} -> {'SOUND' if sound else 'CHECK'}")
    print(f"GOLD_DONE sound={ok}/{len(ids)}", flush=True)
    docker.cleanup_all()


asyncio.run(main())
