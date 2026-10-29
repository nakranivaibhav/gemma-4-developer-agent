"""Run a submission directory on chosen public tasks against the already-running local vLLM server.

Usage: [CONCURRENCY=N] [SANDBOX_ENV=repaired|shipped] envs/harness/bin/python -I kits/swe-agent/run/run_tasks.py <agent_dir> \
           <results_dir> <task> [...]
A <task> is a task id or a manifest selector (smoke6, holdout, tune, usable, ...: kits/swe-agent/run/manifest.py).
Re-running with the same RUN_ID skips the tasks that run already has a row for in results/task_runs.jsonl (resume).
Mirrors the starter notebook's evaluation cell (Phase 1 agent + Phase 2 test verification), docker sandbox.
Usually started through `make run` (kits/swe-agent/Makefile).

FLEET_DIR + FLEET_GPU (+ FLEET_SLOTS): fleet mode. The task ids on the command line are ignored; tasks come one at a time
from the shared queue in kits/swe-agent/run/fleet.py (several GPUs, one process per pass each), and every ledger row also gets
fleet_gpu, fleet_target (tasks at once on that GPU when the task was taken) and task_tier (tune / holdout).

SANDBOX_ENV picks the container environment. `repaired` (default) is the local one in which the real fix resolves 125/129
public tasks (kits/swe-agent/sandbox/: derived image, hook, offline tarpit network); `shipped` is the competition's own, where it
resolves 43/129. IMAGE / NETWORK / KAGGLE_SANDBOX_DIR override single parts. Whatever is used is pinned in the ledger.
"""
import hashlib
import re
import asyncio
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import litellm
import yaml
from adk_submission import VllmConfig, VllmServer, discover_adapters
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks
from swegemma.models.discovery import validate_single_declared_model

litellm.drop_params = True
sys.path.insert(0, str(Path(__file__).resolve().parent))  # the helpers next to this file (python -I leaves it off)

# ADK 1.x calls a sync tool function directly on the event loop (FunctionTool._invoke_callable), and the harness tools
# are sync (run_command = a blocking docker exec). So with CONCURRENCY > 1 every tool call paused every other task in
# this process until it returned, and one hung command paused them all for its full timeout (seen 2026-10-23: a
# repro server froze 16 tasks for 300 s). Sync tools now run in a worker thread; what a tool does and returns is
# unchanged, only the scheduling. The scorer runs one task at a time, where this makes no difference.
import concurrent.futures  # noqa: E402
import contextvars  # noqa: E402
import functools  # noqa: E402
import inspect  # noqa: E402

import google.adk.tools.function_tool as _adk_ft  # noqa: E402

_TOOL_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=max(8, int(os.environ.get("CONCURRENCY", "1")) * 2),
                                                   thread_name_prefix="tool")


async def _invoke_callable_threaded(self, target, args_to_call):
    if inspect.iscoroutinefunction(target) or (hasattr(target, "__call__") and inspect.iscoroutinefunction(target.__call__)):
        return await target(**args_to_call)
    call = functools.partial(contextvars.copy_context().run, target, **args_to_call)
    return await asyncio.get_running_loop().run_in_executor(_TOOL_POOL, call)


_adk_ft.FunctionTool._invoke_callable = _invoke_callable_threaded
import rollout_tag  # noqa: E402  ROLLOUT_TAG=1: each model call carries X-Rollout-Id <run_id>/<task_id> (RL capture)

if os.environ.get("ROLLOUT_TAG") == "1":
    rollout_tag.install()
import manifest  # noqa: E402

ROOT = manifest.ROOT
if len(sys.argv) < 3:
    sys.exit(__doc__)
agent_dir, results_dir = (Path(p).resolve() for p in sys.argv[1:3])
task_ids = manifest.ids(" ".join(sys.argv[3:]))
FLEET_DIR, FLEET_GPU = os.environ.get("FLEET_DIR"), os.environ.get("FLEET_GPU")
if FLEET_DIR:
    import fleet  # noqa: E402

    task_ids = fleet.manifest(FLEET_DIR)["all_tasks"]
data = ROOT / "data"
# The harness asks for the model by name. It accepts the agent's declared name and this one (the competition's
# base model id), whichever Gemma 4 size is actually loaded on the server.
TARGET = "gemma-4-31b-it-qat-w4a16-ct"

SANDBOX_ENV = os.environ.get("SANDBOX_ENV", "repaired")
assert SANDBOX_ENV in ("repaired", "shipped"), SANDBOX_ENV
if SANDBOX_ENV == "repaired":  # the harness reads KAGGLE_SANDBOX_DIR itself, at container setup
    os.environ.setdefault("KAGGLE_SANDBOX_DIR", str(ROOT / "kits" / "swe-agent" / "sandbox"))
IMAGE = os.environ.get("IMAGE", "swebench-sandbox-local:latest" if SANDBOX_ENV == "repaired" else "swebench-sandbox:latest")
NETWORK = os.environ.get("NETWORK", "gemma-tarpit" if SANDBOX_ENV == "repaired" else "none")
HOOK = Path(os.environ.get("KAGGLE_SANDBOX_DIR", data / "sandbox")) / "setup.py"


def _docker(*args):
    r = subprocess.run(["docker", *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


IMAGE_ID = _docker("image", "inspect", IMAGE, "--format", "{{.Id}}")


def _all_outside():
    """True when every task to run carries its own docker_image (outside tasks): the shared sandbox image is then
    unused, e.g. on a remote Docker host (REMOTE=revideo) that only holds task images."""
    tj = Path(os.environ.get("TASKS_JSONL") or data / "tasks.jsonl")
    own = set()
    with open(tj) as f:
        for line in f:
            r = json.loads(line)
            if r.get("docker_image"):
                own.add(r["instance_id"])
    return bool(task_ids) and set(task_ids) <= own


if IMAGE_ID is None:
    if not _all_outside():
        sys.exit(f"docker image {IMAGE} not found: run `make setup` (kits/swe-agent/Makefile)")
    IMAGE_ID = "absent (outside tasks only)"
if NETWORK != "none" and _docker("network", "inspect", NETWORK, "--format", "{{.Id}}") is None:
    sys.exit(f"docker network {NETWORK} not found: docker network create --internal --subnet 10.255.0.0/16 "
             f"--gateway 10.255.0.1 -o com.docker.network.bridge.enable_icc=false {NETWORK}")
if not HOOK.exists():
    sys.exit(f"sandbox hook {HOOK} not found: run `make setup` (kits/swe-agent/Makefile)")

def served_model():
    """The model path the harness asks the server for: MODEL_DIR if set, else the model the running server loaded
    (its note runs/logs/current_server.json, written by serve_model.py at start), else the competition default."""
    if os.environ.get("MODEL_DIR"):
        return Path(os.environ["MODEL_DIR"]).resolve()
    note = Path(os.environ.get("SERVER_JSON") or ROOT / "runs" / "logs" / "current_server.json")
    loaded = json.loads(note.read_text()).get("model_dir") if note.exists() else None  # remote notes have none
    if loaded:
        return (ROOT / loaded).resolve()  # stored relative to the repo root
    return (data / "external" / "model").resolve()


declared = validate_single_declared_model(agent_dir)
adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
# Kaggle clock (kits/swe-agent/run/kaggle_clock.py): the STANDARD task cut since 2026-10-23, on unless KAGGLE_CLOCK=off.
# A task ends when its predicted Kaggle-seconds pass the cap; the wall-clock limit is then only a hang guard.
import kaggle_clock  # noqa: E402

_has_adapter = bool(re.search(r"^adapter:\s*\S", (agent_dir / "agent.yaml").read_text(), re.M))
_clock, _clock_pin = kaggle_clock.settings(str(declared), _has_adapter) or (None, None)
server = VllmServer(
    VllmConfig(model=str(served_model()), port=int(os.environ.get("MODEL_PORT", "8000")), host="127.0.0.1"),
    adapter_manifest=adapters,
)
models = server.create_model_registry(aliases=[declared, TARGET], model_prefix="openai/", api_key="EMPTY")

# TASKS_JSONL: another task file with the same ids, e.g. repaired issue text (training/judge/run_fixer.py); default data/tasks.jsonl
TASKS_JSONL = Path(os.environ.get("TASKS_JSONL") or data / "tasks.jsonl")
ev = yaml.safe_load((agent_dir / "eval_config.yaml").read_text())
ev = ev.get("evaluation", ev)
limits, gen_constraints = build_submission_limits()
cfg = EvalConfig(
    tasks_path=TASKS_JSONL,
    snapshots_dir=data / "snapshots",
    results_dir=results_dir,
    submission_dir=agent_dir,
    models=models,
    sandbox="docker",
    image=IMAGE,
    timeout_seconds=int(ev.get("timeout_seconds", 300)),
    # with the clock on, the wall limit is a hang guard (local wall time depends on our load, not on Kaggle time)
    max_time_minutes=float(os.environ.get("WALL_GUARD_MINUTES", 240)) if _clock else float(ev.get("max_time_minutes", 60.0)),
    max_tool_calls=int(ev.get("max_tool_calls", 100)),
    max_turns=int(ev["max_turns"]) if ev.get("max_turns") is not None else None,
    limits=limits,
    generation_constraints=gen_constraints,
    adapter_manifest=adapters,
    context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
    events_compaction_config=EventsCompactionConfig(
        compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5
    ),
    graph_dir=str(data / "graphs"),
    # EMBEDDINGS_DIR: stand-in node embeddings (same file names as data/embeddings) for an official-vs-stand-in A/B
    embeddings_dir=os.path.abspath(os.environ.get("EMBEDDINGS_DIR", data / "embeddings")),
    wheels_dir=Path(os.path.abspath(os.environ.get("WHEELS_DIR", data / "wheels"))),  # WHEELS_DIR: a patched wheel set for env tests
    verbose=False,
    display_mode="quiet",
)
if True:  # every sandbox carries its run id (label swegemma_run), so a stopped run's leftovers can be removed by label;
    # optional: lower CPU weight for this run's sandboxes (SANDBOX_CPU_SHARES), fleet labels (FLEET_DIR)
    # swegemma 0.2.10 has no option for extra container settings, so its docker call is wrapped here.
    import docker.models.containers as _dc

    _orig_run = _dc.ContainerCollection.run

    def _run_with_shares(self, *a, **k):
        if os.environ.get("SANDBOX_CPU_SHARES"):
            k.setdefault("cpu_shares", int(os.environ["SANDBOX_CPU_SHARES"]))
        if FLEET_DIR:  # label the sandbox with its GPU, so a stopped GPU's leftovers can be removed by label
            k["labels"] = {**(k.get("labels") or {}), "fleet_gpu": FLEET_GPU, "fleet_run": os.environ.get("RUN_ID", "")}
        k["labels"] = {**(k.get("labels") or {}), "swegemma_run": os.environ.get("RUN_ID", "")}
        return _orig_run(self, *a, **k)

    _dc.ContainerCollection.run = _run_with_shares
tasks = {t.instance_id: t for t in load_tasks(TASKS_JSONL)}
evaluator = Evaluator(cfg)
evaluator.docker.config.network_mode = NETWORK  # the harness default is 'none'
# Outside tasks (SWE-rebench, kits/swe-agent/outside/): rows that carry docker_image run in their own image, with their
# own verification. Rows without it (all our Kaggle tasks) do not touch this path.
sys.path.append(str(ROOT / "kits" / "swe-agent" / "outside"))
import outside  # noqa: E402

OUTSIDE = {k: v for k, v in outside.load_specs(TASKS_JSONL).items() if k in task_ids}
if OUTSIDE:
    outside.install(OUTSIDE)
results_dir.mkdir(parents=True, exist_ok=True)
CONCURRENCY = max(1, int(os.environ.get("CONCURRENCY", "1")))  # tasks at once, same pattern as swegemma eval --concurrency

# the run ledger: every task-run is one line in results/task_runs.jsonl with the pinned settings embedded,
# and the agent folder is copied next to the results so a run can be reproduced exactly
from pin import pin_settings, rel, snapshot_agent  # noqa: E402
from trace_stats import summarize  # noqa: E402
from tag_tasks import split_line  # noqa: E402

RUN_ID = os.environ.get("RUN_ID") or f"{time.strftime('%Y%m%dT%H%MZ', time.gmtime())}_{results_dir.name}"
SETTINGS = pin_settings(agent_dir)
if os.environ.get("SANDBOX_CPU_SHARES"):
    SETTINGS["sandbox_cpu_shares"] = int(os.environ["SANDBOX_CPU_SHARES"])
if os.environ.get("EMBEDDINGS_DIR"):  # recorded only when it differs from the official vectors
    SETTINGS["embeddings_dir"] = rel(cfg.embeddings_dir)
SETTINGS.update(run_id=RUN_ID, tier=os.environ.get("EVAL_TIER", ""), note=os.environ.get("EVAL_NOTE", ""),
                concurrency=CONCURRENCY, tasks=task_ids,
                sandbox={"env": SANDBOX_ENV, "image": IMAGE, "image_id": IMAGE_ID[:19], "network": NETWORK, "hook": rel(HOOK),
                         "hook_sha": hashlib.sha256(HOOK.read_bytes()).hexdigest()[:16], "wheels_dir": rel(cfg.wheels_dir)})
if OUTSIDE:  # outside tasks: their own images and verifier (kits/swe-agent/outside/outside.py), pinned here
    SETTINGS["outside"] = {"tasks": len(OUTSIDE), "graph_dir": rel(outside.GRAPH_DIR), "embeddings_dir": rel(outside.EMB_DIR),
                           "verify_timeout": outside.VERIFY_TIMEOUT, "git_gc": outside.GIT_GC,
                           "adapter_sha": hashlib.sha256(Path(outside.__file__).read_bytes()).hexdigest()[:16]}
# Kaggle clock: cut each task where the scorer's 4x L4 clock would, counted in Kaggle-seconds from the token counts
# (kits/swe-agent/run/kaggle_clock.py). On by default (KAGGLE_CLOCK=off disables). One clock per task, attached to the
# harness context as it is built.
CLOCKS = {}
if _clock:
    from swegemma.context import SwegemmaContext  # noqa: E402

    SETTINGS["kaggle_clock"] = _clock_pin
    SETTINGS["budget_max_time_minutes"] = cfg.max_time_minutes  # the hang guard actually used, not eval_config's value
    SETTINGS["wall_limit_role"] = "hang_guard"
    # swegemma 0.2.10 builds the per-task context inside the evaluator with no hook for a custom token budget,
    # so the constructor is wrapped to attach the clock as the context's token_budget.
    _ctx_init = SwegemmaContext.__init__

    def _init_with_clock(self, *a, **k):
        _ctx_init(self, *a, **k)
        self.token_budget = CLOCKS[self.task_id] = kaggle_clock.KaggleClock(self, **_clock)

    SwegemmaContext.__init__ = _init_with_clock
    # a cut ends the agent loop at once through the harness's own session timeout (kaggle_clock.py docstring, 2026-10-26)
    import swegemma.harness.agent_runner as _agent_runner  # noqa: E402

    kaggle_clock.install_cut_hook(_agent_runner)
LEDGER = Path(os.environ.get("LEDGER") or ROOT / "results" / "task_runs.jsonl")
LEDGER.parent.mkdir(exist_ok=True)
# Resume: tasks this run id already has a ledger row for are skipped, so relaunching a stopped run with the same RUN_ID
# keeps its finished work. (A fresh run id, the default, has no rows: every task runs.)
if not FLEET_DIR and LEDGER.exists():
    mark = f'"run_id": {json.dumps(RUN_ID)}'
    done_before = {json.loads(line)["id"] for line in open(LEDGER) if mark in line}
    if done_before:
        todo = [t for t in task_ids if t not in done_before]
        print(f"RESUME {RUN_ID}: {len(task_ids) - len(todo)} of {len(task_ids)} tasks already in the ledger, {len(todo)} to run",
              flush=True)
        if not todo:
            print(f"ALL_DONE tasks=0 concurrency={CONCURRENCY} wall_seconds=0 (nothing left)", flush=True)
            sys.exit(0)
        if (results_dir / "settings.json").exists():  # keep the first launch's settings next to the new ones
            (results_dir / "settings_before_resume.json").write_text((results_dir / "settings.json").read_text())
        task_ids = todo
snapshot_agent(agent_dir, results_dir / "agent_snapshot")
(results_dir / "settings.json").write_text(json.dumps(SETTINGS, indent=1))
PINNED = {k: v for k, v in SETTINGS.items() if k not in ("files", "tasks", "sub_agents", "tools")}
PINNED["file_sha"] = {k: v["sha"] for k, v in SETTINGS["files"].items()}  # every prompt, config and adapter weight file
HOLDOUT = manifest.holdout()
TAGS = manifest.tags()  # labels from the task alone (kits/swe-agent/run/tag_tasks.py); tagged tasks still run, they only get a second figure
DONE = []  # (task id, resolved) for the closing line


async def run_one(i, tid, sem, extra=None):
    async with sem:
        t0 = time.time()
        rollout_tag.ROLLOUT_ID.set(f"{RUN_ID}/{tid}")  # this asyncio task only (see rollout_tag.py)
        print(f"START {tid}", flush=True)
        try:
            r = await evaluator.evaluate_task(task=tasks[tid], task_index=i, total_tasks=len(task_ids))
            # keep the patch and the pytest output: per-test pass counts and failure kinds come from them for free
            (results_dir / "patches").mkdir(exist_ok=True)
            (results_dir / "test_outputs").mkdir(exist_ok=True)
            (results_dir / "patches" / f"{tid}.diff").write_text(r.agent_patch or "")
            (results_dir / "test_outputs" / f"{tid}.log").write_text(r.test_output or "")
            # the harness runs pytest -q: counts come from the summary line "3 passed, 1 failed, 2 errors in 0.4s"
            counts = {k: int(n) for n, k in re.findall(r"(\d+) (passed|failed|errors?|skipped)\b", r.test_output or "")}
            row = {
                "id": tid, "resolved": r.resolved, "test_exit_code": r.test_exit_code,
                "patch_chars": len(r.agent_patch or ""), "tool_calls": r.tool_calls,
                "agent_seconds": round(r.duration_seconds, 1), "wall_seconds": round(time.time() - t0, 1),
                "tests_passed": counts.get("passed", 0), "tests_failed": counts.get("failed", 0),
                "tests_errors": counts.get("errors", counts.get("error", 0)),
                "harness_error": (r.error or None) and str(r.error)[:300],
            }
            if tid in OUTSIDE:  # the task's own image and the F2P / P2P counts of verify_outside
                row.update(image=OUTSIDE[tid]["docker_image"], outside=getattr(r, "outside", None))
        except Exception as exc:  # keep going so one bad task does not hide the others
            row = {"id": tid, "error": f"{type(exc).__name__}: {exc}"[:500], "wall_seconds": round(time.time() - t0, 1)}
        clock = CLOCKS.pop(tid, None)
        if clock is not None:
            row.update(clock.row(row.get("resolved")))
        print("RESULT " + json.dumps(row), flush=True)
        with open(results_dir / "probe_results.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        out = rel(results_dir)  # stored relative to the repo root
        trace_path = results_dir / "traces" / f"trace_{tid}.json"
        try:
            stats = summarize(trace_path, harness_error=row.get("harness_error") or row.get("error"),
                              budgets={"max_tool_calls": cfg.max_tool_calls}, max_output_tokens=SETTINGS.get("max_output_tokens"))
        except Exception as exc:  # a missing or half-written trace must not lose the ledger line
            stats = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        DONE.append((tid, bool(row.get("resolved"))))
        row.update(run_id=RUN_ID, bin=None, tags=TAGS.get(tid, []),  # bin: failure class, assigned later as the bins get defined
                   trace=f"{out}/traces/trace_{tid}.json", patch=f"{out}/patches/{tid}.diff",
                   test_output=f"{out}/test_outputs/{tid}.log", trace_stats=stats, settings=PINNED)
        if extra:
            row.update(extra)
        with open(LEDGER, "a") as fh:  # locked: ledger_rebuild.py rewrites the file under the same lock
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(json.dumps(row) + "\n")


def _big_pool():
    # The harness runs every docker call, tool call and verification step through asyncio.to_thread, i.e. the
    # default thread pool, which is min(32, cores + 4) = 16 threads here. At 16 tasks at once the 12B rest pass 2
    # froze all tasks together with the model server idle (2026-10-15 18:08Z, watchdog stall after 30 min).
    # Give the loop several threads per task so a slow call in one task cannot starve the others.
    from concurrent.futures import ThreadPoolExecutor
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=max(32, CONCURRENCY * 8)))


async def heartbeat():
    # The log only gets a line when a task ends, so a long tail (a few tasks near the 60 min session limit) looked
    # idle to the 1800 s watchdog and got killed while still working (E2B par pass, 2026-10-16 09:35Z). Print a line
    # every 5 min, but only while the request capture grows: a real freeze with no model traffic still goes quiet.
    cap, last = results_dir / "requests.jsonl.gz", -1
    while True:
        await asyncio.sleep(300)
        size = cap.stat().st_size if cap.exists() else -2
        if size != last or size == -2:
            print(f"HEARTBEAT {time.strftime('%H:%MZ', time.gmtime())} done={len(DONE)}/{len(task_ids)} capture={size}", flush=True)
        last = size


async def run_all():
    _big_pool()
    hb = asyncio.create_task(heartbeat())
    sem = asyncio.Semaphore(CONCURRENCY)
    await asyncio.gather(*(run_one(i, tid, sem) for i, tid in enumerate(task_ids, start=1)))
    hb.cancel()


async def fleet_worker(k, sem, counter):
    last_say = time.time()
    while True:
        try:
            got = await asyncio.to_thread(fleet.claim, FLEET_DIR, FLEET_GPU, RUN_ID, os.getpid())
        except Exception as exc:  # a queue hiccup must not end the worker
            print(f"FLEET_CLAIM_ERROR {type(exc).__name__}: {exc}", flush=True)
            got = None
        if got == "STOP":
            return
        if got is None:
            if await asyncio.to_thread(fleet.all_done, FLEET_DIR):
                return
            if k == 0 and time.time() - last_say > 600:  # a heartbeat, so the watchdog sees a live log while waiting
                print(f"FLEET_WAIT {time.strftime('%H:%M:%SZ', time.gmtime())}", flush=True)
                last_say = time.time()
            await asyncio.sleep(10 + k % 7)
            continue
        counter[0] += 1
        extra = {"fleet_gpu": FLEET_GPU, "fleet_target": fleet.target(FLEET_DIR, FLEET_GPU),
                 "task_tier": "holdout" if got in HOLDOUT else "tune"}
        try:
            await run_one(counter[0], got, sem, extra)
        finally:
            await asyncio.to_thread(fleet.finish, FLEET_DIR, FLEET_GPU, RUN_ID, os.getpid(), got)


async def fleet_all():
    _big_pool()
    sem, counter = asyncio.Semaphore(10**6), [0]  # the queue limits tasks at once, not this process
    await asyncio.gather(*(fleet_worker(k, sem, counter) for k in range(int(os.environ.get("FLEET_SLOTS", "32")))))


# on a future freeze: kill -USR1 <run_tasks pid> prints every thread's stack to the run log
import faulthandler  # noqa: E402
import signal  # noqa: E402
faulthandler.register(signal.SIGUSR1, all_threads=True)
t_all = time.time()
asyncio.run(fleet_all() if FLEET_DIR else run_all())
print(f"RESOLVED {split_line(DONE, TAGS)}", flush=True)
print(f"ALL_DONE tasks={len(task_ids)} concurrency={CONCURRENCY} wall_seconds={time.time() - t_all:.0f}", flush=True)
