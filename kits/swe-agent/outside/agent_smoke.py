"""Model-free smoke test of the AGENT path for outside tasks: the agent container, its tools, the prompt, the patch.

usage: [NETWORK=gemma-tarpit] envs/harness/bin/python -I kits/swe-agent/outside/agent_smoke.py <tasks.jsonl> <out_dir> \
           [instance_id ...]
For each task, in the order run_agent_sandbox uses (with the hooks run_tasks.py installs): start the container from
the task's image, the (hooked) setup steps, baseline commit, workspace tree, the agent prompt; then the agent's real
tool callables (context.create_tools()) are called the way the model would call them:
  run_command  `python -c "import <pkg>"` (must come from /workspace, in the task's Python), git refs left after the scrub
  read_file    a file the reference fix touches
  get_code_neighbors / get_code_subgraph   on a symbol of the outside graph
  run_command  `git apply` of the reference fix (stands in for the model's edits), then submit_patch
and the submitted patch goes through verify_outside, which must resolve. No model server is started or contacted.
Then the same tasks through the hooked Evaluator.evaluate_task with a stub agent (checks the TaskResult glue that
run_tasks.py reads), and with KAGGLE_CHECK=<our task id> one of our own tasks' agent-container setup with the hooks
installed (must still be swebench-sandbox-local + snapshot + sandbox hook).
Writes <out_dir>/smoke.jsonl and <out_dir>/<id>.prompt.md.
"""
import asyncio
import base64
import copy
import json
import os
import re
import sys
import time
from pathlib import Path

from adk_submission import VllmConfig, VllmServer
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.context import SwegemmaContext
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks
from swegemma.sandbox import sandbox_exec, sandbox_start, sandbox_stop

sys.path.insert(0, str(Path(__file__).resolve().parent))
import outside  # noqa: E402
import swegemma.harness.agent_runner as ar  # noqa: E402  (after outside.install: the hooked names live here)

ROOT = outside.ROOT
tasks_path, out_dir = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
out_dir.mkdir(parents=True, exist_ok=True)
specs = outside.load_specs(tasks_path)
tasks = {t.instance_id: t for t in load_tasks(tasks_path) if t.instance_id in specs}
ids = sys.argv[3:] or list(tasks)
AGENT = ROOT / "agents" / "r09_e2b_8tool"
DECLARED = ["run_command", "read_file", "edit_file", "write_file", "get_status", "submit_patch", "get_code_neighbors",
            "get_code_subgraph"]

outside.install(specs)
server = VllmServer(VllmConfig(model=str(ROOT / "data" / "external" / "model"), port=8000, host="127.0.0.1"), adapter_manifest={})
limits, gen = build_submission_limits()
cfg = EvalConfig(tasks_path=tasks_path, snapshots_dir=ROOT / "data" / "snapshots", results_dir=out_dir, submission_dir=AGENT,
                 models=server.create_model_registry(aliases=["gemma-4-31b-it-qat-w4a16-ct"], model_prefix="openai/", api_key="EMPTY"),
                 sandbox="docker", image="swebench-sandbox-local:latest", timeout_seconds=300, max_time_minutes=60,
                 max_tool_calls=100, limits=limits, generation_constraints=gen, graph_dir=str(ROOT / "data" / "graphs"),
                 embeddings_dir=str(ROOT / "data" / "embeddings"), verbose=False, display_mode="quiet")
evaluator = Evaluator(cfg)
evaluator.docker.config.network_mode = os.environ.get("NETWORK", "gemma-tarpit")
docker = evaluator.docker


def short(d, n=300):
    return json.dumps(d)[:n]


async def smoke(tid):
    task, row = tasks[tid], specs[tid]
    tcfg = outside.task_config(cfg)
    rec = {"id": tid, "image": row["docker_image"]}
    t0 = time.time()
    token = outside.CURRENT.set(row)
    sid = await sandbox_start(docker)
    try:
        # the setup sequence of run_agent_sandbox, through the names it calls (hooked for outside tasks)
        await asyncio.to_thread(ar.setup_container_wheels, docker, sid, tcfg)
        await asyncio.to_thread(ar.extract_snapshot, docker, sid, Path("/outside-task-has-no-snapshot"),
                                patch_path=None, base_snapshot_path=None)
        await asyncio.to_thread(ar.setup_git_exclude, docker, sid)
        await asyncio.to_thread(ar.install_editable_package, docker, sid)
        await asyncio.to_thread(ar.install_test_dependencies, docker, sid, task.repo, config=tcfg)
        await asyncio.to_thread(ar.setup_workspace_test_config, docker, sid, repo=task.repo)
        await asyncio.to_thread(ar.setup_baseline_commit, docker, sid, "baseline")
        rec["setup_seconds"] = round(time.time() - t0, 1)
        tree = await sandbox_exec(docker, sid, ar.build_workspace_tree_command())
        prompt = ar.build_agent_prompt(task, tcfg, tree.stdout.strip() if tree.exit_code == 0 else "", declared_tools=DECLARED)
        (out_dir / f"{tid}.prompt.md").write_text(prompt)
        rec["prompt_graph_block"] = "## Code Intelligence Tools" in prompt
        rec["prompt_tree_lines"] = len(tree.stdout.splitlines())

        ctx = SwegemmaContext(docker_manager=docker, container_id=sid, problem_statement=task.problem_statement,
                              hints_text=task.hints_text, task=task, repo=task.repo, graph_dir=tcfg.graph_dir,
                              embeddings_dir=tcfg.embeddings_dir, budget=tcfg.budget, harness=tcfg.harness)
        ctx.start_agent_session()
        tools = ctx.create_tools()
        files = re.findall(r"^diff --git a/(\S+) b/", task.patch, re.M)
        pyfile = next((f for f in files if f.endswith(".py")), files[0])
        top = pyfile.split("/")[0] if pyfile.split("/")[0] not in ("src", "lib") else pyfile.split("/")[1]
        pkg = top[:-3] if top.endswith(".py") else top
        r = tools["run_command"](f'python -c "import sys, {pkg}; print({pkg}.__file__); print(sys.executable, sys.version.split()[0])"')
        rec["import"] = r.get("stdout", r.get("error_message", "")).strip() if isinstance(r, dict) else str(r)[:200]
        rec["import_from_workspace"] = rec["import"].startswith(("/workspace/", "/testbed/"))  # the repo, not a copy
        r = tools["run_command"]("git log --all --oneline | wc -l; git branch -a | tr '\\n' ' '; echo; git remote | wc -l; "
                                 "git rev-parse HEAD; git status --short | wc -l; whoami; pwd")
        rec["git"] = (r.get("stdout") or "").split("\n")[:6]
        r = tools["read_file"](pyfile, 1, 5)
        rec["read_file"] = r.get("status", "?") if isinstance(r, dict) else str(r)[:80]
        g = json.loads((outside.GRAPH_DIR / f"{outside.graph_stem(row)}.json").read_text())
        mod = re.sub(r"\.__init__$", "", pyfile[:-3].replace("/", "."))
        node = next((n["id"] for n in g["nodes"] if n["id"].startswith(mod.split(".", 1)[-1] if mod.startswith("src.") else mod)),
                    g["nodes"][0]["id"])
        r = tools["get_code_neighbors"](node)
        rec["neighbors"] = {"node": node, "status": r.get("status"), "count": r.get("count"), "err": r.get("error_message")}
        nbrs = (r.get("neighbors") or [])[:2]
        r = tools["get_code_subgraph"]([node, *nbrs])
        rec["subgraph"] = {"status": r.get("status"), "keys": sorted(r)[:8], "err": r.get("error_message")}
        # stand-in for the model's edits: the reference fix, written and applied from inside the container
        b64 = base64.b64encode(task.patch.encode()).decode()
        r = tools["run_command"](f"echo {b64} | base64 -d > /tmp/fix.diff && git apply /tmp/fix.diff && git status --short")
        rec["apply_fix"] = r.get("status")
        r = tools["submit_patch"]()
        rec["submit"] = {k: r.get(k) for k in ("status", "patch_size", "files_changed")}
        patch = ctx.submitted_patch or ""
        (out_dir / f"{tid}.submitted.diff").write_text(patch)
    finally:
        await sandbox_stop(docker, sid)
        outside.CURRENT.reset(token)
    v = await outside.verify_outside(docker, task, row, patch)
    rec["verify"] = {k: v.get(k) for k in ("resolved", "strict", "f2p_pass", "f2p_total", "p2p_pass", "p2p_total", "error")}
    rec["seconds"] = round(time.time() - t0, 1)
    with open(out_dir / "smoke.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(short(rec, 2000), flush=True)
    return rec


async def glue(tid):
    """Evaluator.evaluate_task (hooked) -> evaluate_outside -> verify_outside, with run_agent_sandbox replaced by a stub
    that 'submits' the reference fix: checks the glue run_tasks.py relies on (TaskResult fields)."""
    from adk_eval_core.tracing import SessionTrace

    task = tasks[tid]
    real = ar.run_agent_sandbox

    async def stub(docker_, cfg_, task_, snapshot_path, **k):
        assert outside.CURRENT.get() is not None and cfg_.graph_dir == str(outside.GRAPH_DIR)
        return task_.patch, None, SessionTrace()

    ar.run_agent_sandbox = stub
    try:
        r = await evaluator.evaluate_task(task=task, task_index=1, total_tasks=1)
    finally:
        ar.run_agent_sandbox = real
    rec = {"id": tid, "glue_resolved": r.resolved, "exit": r.test_exit_code, "error": r.error,
           "outside": getattr(r, "outside", None), "trace": bool(r.trace_path)}
    print("GLUE " + json.dumps(rec), flush=True)
    return rec


async def kaggle_check(kid):
    """Regression for our own tasks with the outside hooks installed: the agent-container setup through the hooked
    names must still use swebench-sandbox-local, the snapshot and the sandbox hook (CURRENT is unset)."""
    os.environ.setdefault("KAGGLE_SANDBOX_DIR", str(ROOT / "kits" / "swe-agent" / "sandbox"))
    from swegemma.deduplication import resolve_task_snapshot_paths

    ktask = {t.instance_id: t for t in load_tasks(ROOT / "data" / "tasks.jsonl")}[kid]
    snap, base, patch_path = resolve_task_snapshot_paths(cfg.snapshots_dir, kid, ktask.repo)
    kcfg = copy.copy(cfg)
    object.__setattr__(kcfg, "wheels_dir", ROOT / "data" / "wheels")
    assert outside.CURRENT.get() is None
    sid = await sandbox_start(docker)
    try:
        await asyncio.to_thread(ar.setup_container_wheels, docker, sid, kcfg)
        await asyncio.to_thread(ar.extract_snapshot, docker, sid, snap, patch_path=patch_path, base_snapshot_path=base)
        await asyncio.to_thread(ar.setup_git_exclude, docker, sid)
        await asyncio.to_thread(ar.install_editable_package, docker, sid)
        await asyncio.to_thread(ar.install_test_dependencies, docker, sid, ktask.repo, config=kcfg)
        await asyncio.to_thread(ar.setup_workspace_test_config, docker, sid, repo=ktask.repo)
        await asyncio.to_thread(ar.setup_baseline_commit, docker, sid, "baseline")
        pkg = ktask.repo.split("/")[-1]
        r = await sandbox_exec(docker, sid, f'cd /workspace && python3 -c "import {pkg}; print({pkg}.__file__)"; ls /workspace | head -3; git log --oneline -1')
        img = docker.client.containers.get(sid).attrs["Config"]["Image"]
    finally:
        await sandbox_stop(docker, sid)
    rec = {"kaggle_task": kid, "image": img, "out": (r.stdout or r.stderr).strip().splitlines()[:5]}
    print("KAGGLE " + json.dumps(rec), flush=True)
    return rec


async def main():
    rows = [await smoke(t) for t in ids]
    ok = sum(bool(r["verify"]["resolved"] and r["import_from_workspace"] and r["prompt_graph_block"]
                  and r["neighbors"]["status"] == "ok" and r["subgraph"]["status"] == "ok") for r in rows)
    print(f"SMOKE_DONE ok={ok}/{len(rows)}")
    g = [await glue(t) for t in ids]
    print(f"GLUE_DONE resolved={sum(bool(x['glue_resolved']) for x in g)}/{len(g)}")
    if os.environ.get("KAGGLE_CHECK"):
        await kaggle_check(os.environ["KAGGLE_CHECK"])
    docker.cleanup_all()


asyncio.run(main())
