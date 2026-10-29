"""Outside tasks (SWE-rebench) in our harness: every task runs in its OWN docker image.

A task row that carries `docker_image` (written by export_tasks.py) is an outside task. For those, and only those:

- Both containers (the agent's and the verifier's) start from the task's image instead of the shared sandbox image,
  on the same network and with the same limits (the harness's ContainerConfig: 4 GB, 2 CPUs). The swap happens in the
  docker-py `containers.run` call, keyed by a context variable that is set while the task runs, so tasks running
  side by side each get their own image.
- The repo is MOVED from /testbed to /workspace and /testbed becomes a symlink to /workspace. /workspace is then a real
  directory, which the harness's file tools require (copy_from refuses paths that resolve outside /workspace), and
  everything the image baked in under /testbed (the editable install's finder / .pth, conda paths) still resolves.
- Commands run in the task's conda env: its bin/ goes first on the container's PATH (plus CONDA_PREFIX and the task's
  own env_vars), so `python`, `pytest` and the installed dependencies are the task's.
- The snapshot extraction and the wheel/setup hook (sandbox/setup.py, pytest.ini/conftest writing) are skipped: the image
  already holds the repo at the base commit with its dependencies installed.
- In the AGENT container only, git refs that are not ancestors of HEAD are deleted (remotes, later branches and tags)
  and the objects pruned: the images ship origin/main past the base commit, i.e. possibly the fix itself.
- Verification is our own (verify_outside), SWE-bench style: apply the agent patch (protected test/config files
  stripped, as the Kaggle harness does), reset the test files, apply the test_patch, run the task's own test command on
  the test files, parse the `-rA` summary. resolved = every FAIL_TO_PASS and every PASS_TO_PASS test PASSED (or XFAIL).
  `strict` adds the Kaggle verifier's extra rule: exit code 0 and no failed/errored test anywhere in the run.

Our own Kaggle tasks never pass through any of this: every hook checks the context variable and falls through.
"""
from __future__ import annotations

import asyncio
import contextvars
import copy
import json
import os
import re
import shlex
import subprocess
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repo root
GRAPH_DIR = Path(os.environ.get("OUTSIDE_GRAPH_DIR") or ROOT / "data" / "outside" / "graphs")
EMB_DIR = Path(os.environ.get("OUTSIDE_EMB_DIR") or ROOT / "data" / "outside" / "embeddings")
VERIFY_TIMEOUT = int(os.environ.get("OUTSIDE_VERIFY_TIMEOUT", "1800"))  # SWE-bench's own per-instance test timeout
GIT_GC = os.environ.get("OUTSIDE_GIT_GC", "1") == "1"  # prune the scrubbed objects too (0: delete refs only)

CURRENT: contextvars.ContextVar[dict | None] = contextvars.ContextVar("outside_task", default=None)
SPECS: dict[str, dict] = {}

# SWE-bench's get_test_directives drops these from the files of the test_patch before running the test command
NON_TEST_EXTS = (".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".yml", ".yaml", ".toml")
STATUSES = ("PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL")
OK = ("PASSED", "XFAIL")


# ---------------------------------------------------------------------------------------------------------- task rows
def load_specs(tasks_jsonl) -> dict[str, dict]:
    """{instance_id: row} for the rows of a tasks file that carry docker_image (empty for our own tasks)."""
    out = {}
    with open(tasks_jsonl, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                if row.get("docker_image"):
                    out[row["instance_id"]] = row
    return out


def graph_stem(row) -> str:
    """File stem the harness looks for: <repo short name>_<base commit> (swegemma graph_repo_key)."""
    return f"{row['repo'].split('/')[-1]}_{row['base_commit']}"


def test_directives(test_patch: str) -> list[str]:
    """The files the test command runs on, exactly as SWE-bench builds them (b/ paths of the test_patch)."""
    files = re.findall(r"diff --git a/.* b/(.*)", test_patch)
    return [f for f in dict.fromkeys(files) if not f.endswith(NON_TEST_EXTS)]


def test_command(test_cmd: str, test_patch: str) -> str:
    """The task's test command on its test files. Some rows (a third of the leaderboard sets) already name the files in
    test_cmd; those run as written, the rest get the test_patch's files appended (SWE-bench's rule)."""
    try:
        words = set(shlex.split(test_cmd))
    except ValueError:
        words = set(test_cmd.split())
    directives = test_directives(test_patch)
    if any(d in words for d in directives):
        return test_cmd
    return f"{test_cmd} {' '.join(shlex.quote(d) for d in directives)}".strip()


def parse_pytest_log(log: str) -> dict[str, str]:
    """{test id: status} from pytest output; SWE-bench's parse_log_pytest_v2 (the `-rA` summary lines `STATUS id`,
    plus `id STATUS` lines of verbose runs). Test ids are cut at the first space, as in SWE-bench, which is also how
    the task's FAIL_TO_PASS / PASS_TO_PASS lists were written."""
    status = {}
    escapes = "".join(chr(c) for c in range(1, 32))
    table = str.maketrans("", "", escapes)
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line).translate(table)
        if line.startswith(STATUSES):
            if line.startswith("FAILED"):
                line = line.replace(" - ", " ")
            parts = line.split()
            if len(parts) >= 2:
                status[parts[1]] = parts[0]
        elif line.endswith(STATUSES):
            parts = line.split()
            if len(parts) >= 2:
                status[parts[0]] = parts[1]
    return status


# ------------------------------------------------------------------------------------------------------ the image
_IMAGE_INFO: dict[str, dict] = {}
_IMAGE_LOCK = threading.Lock()
_PROBE = r"""
for d in /opt/conda /opt/miniconda3 /opt/miniconda /root/miniconda3 /root/anaconda3 /usr/local/miniconda3; do
  [ -x "$d/envs/testbed/bin/python" ] && { echo "ENV=$d/envs/testbed"; break; }
done
for p in /opt/conda/bin/python3 /opt/miniconda3/bin/python3 /opt/miniconda/bin/python3 /root/miniconda3/bin/python3 \
         /usr/local/bin/python3 /usr/bin/python3; do
  [ -x "$p" ] && [ "$($p -c 'import sys; print(sys.version_info >= (3, 9))' 2>/dev/null)" = True ] && { echo "PY=$p"; break; }
done
[ -d /testbed/.git ] && echo TESTBED=1
command -v git >/dev/null && echo GIT=1
command -v patch >/dev/null && echo PATCH=1
"""


def image_info(image: str) -> dict:
    """Where the task's env and a Python >= 3.9 for the harness's own helper scripts live in this image (cached).
    The image must already be pulled: Docker Hub limits anonymous pulls, so pulling is always an explicit step."""
    with _IMAGE_LOCK:
        if image in _IMAGE_INFO:
            return _IMAGE_INFO[image]
        r = subprocess.run(["docker", "image", "inspect", image, "--format", "{{json .Config.Env}}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"outside: image {image} is not local; pull it first: docker pull {image}")
        env = dict(e.split("=", 1) for e in json.loads(r.stdout) or [] if "=" in e)
        p = subprocess.run(["docker", "run", "--rm", "--network", "none", "--entrypoint", "/bin/bash", image, "-c", _PROBE],
                           capture_output=True, text=True, timeout=120)
        got = dict(line.split("=", 1) for line in p.stdout.split() if "=" in line)
        missing = [k for k in ("ENV", "PY", "TESTBED", "GIT") if k not in got]
        if missing:
            raise RuntimeError(f"outside: image {image} lacks {missing} (probe stdout {p.stdout!r} stderr {p.stderr[-300:]!r})")
        info = {"env_prefix": got["ENV"], "harness_python": got["PY"], "has_patch": "PATCH" in got,
                "image_path": env.get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")}
        _IMAGE_INFO[image] = info
        return info


def container_env(row: dict) -> dict[str, str]:
    """Environment for the task's containers: the task's conda env first on PATH, plus the task's own env_vars."""
    info = image_info(row["docker_image"])
    env = {"PATH": f"{info['env_prefix']}/bin:{info['image_path']}", "CONDA_PREFIX": info["env_prefix"],
           "CONDA_DEFAULT_ENV": "testbed",
           # the harness commits a baseline before the agent starts and diffs against it; without an identity that
           # commit fails and the diff would fall back to HEAD, sweeping pre-existing untracked files into the patch
           "GIT_AUTHOR_NAME": "swegemma", "GIT_AUTHOR_EMAIL": "swegemma@localhost",
           "GIT_COMMITTER_NAME": "swegemma", "GIT_COMMITTER_EMAIL": "swegemma@localhost"}
    env.update({k: str(v) for k, v in (row.get("env_vars") or {}).items() if v is not None})
    return env


# ------------------------------------------------------------------------------------------------- container setup
PREP = r"""
set -e
if [ ! -L /testbed ]; then
  [ -d /testbed ] || { echo "outside: no /testbed in this image" >&2; exit 3; }
  rmdir /workspace 2>/dev/null || true          # docker made it (working_dir), empty
  [ -e /workspace ] && { echo "outside: /workspace exists and is not empty" >&2; exit 3; }
  mv /testbed /workspace
  ln -s /workspace /testbed
fi
# Many images were built with the repo at /<name> and only later moved to /testbed, so the env's editable install
# (egg-link, .pth, __editable__ finder) still points at /<name>, which no longer exists. Link each such top-level
# folder to /workspace when the rest of the path exists there.
for sp in "$CONDA_PREFIX"/lib/python*/site-packages; do
  for p in $( (grep -h -o '^/[^ ;]*' "$sp"/*.egg-link "$sp"/*.pth 2>/dev/null;
               grep -h -o "'/[^']*'" "$sp"/__editable__*finder.py 2>/dev/null | tr -d "'") | sort -u); do
    [ -e "$p" ] && continue
    top="/$(echo "$p" | cut -d/ -f2)"
    [ -e "$top" ] && continue
    [ -e "/workspace${p#"$top"}" ] || continue
    ln -s /workspace "$top" && echo "LINK=$top"
  done
done
cd /workspace
# the images leave the issue text as an untracked issue.md in the repo (and at /): not part of the repo
if [ -f issue.md ] && ! git ls-files --error-unmatch issue.md >/dev/null 2>&1; then rm -f issue.md; fi
rm -f /issue.md
echo "HEAD=$(git rev-parse HEAD)"
"""
# Agent container only: drop every ref the agent could reach a later commit (the fix) through.
SCRUB = r"""
set -e
cd /workspace
git remote | xargs -r -n1 git remote remove
cur=$(git symbolic-ref -q HEAD || true)
git for-each-ref --format='%(refname)' | while read -r r; do
  [ "$r" = "$cur" ] && continue
  git merge-base --is-ancestor "$r" HEAD 2>/dev/null || git update-ref -d "$r"
done
rm -f .git/FETCH_HEAD .git/ORIG_HEAD .git/MERGE_HEAD
git reflog expire --expire=now --all
"""


def prepare_workspace(docker, container_id: str, row: dict, *, agent: bool) -> str:
    """Move the image's /testbed to /workspace (symlink back); in the agent container also scrub later git refs.
    Returns the HEAD commit found in the image."""
    from swegemma.harness.container_setup import ContainerSetupError

    r = docker.exec(container_id, PREP, timeout=600)
    if r.exit_code != 0:
        raise ContainerSetupError(f"outside: workspace prep failed on {row['instance_id']}: {r.stderr or r.stdout}")
    head = (re.search(r"HEAD=(\S+)", r.stdout or "") or [None, ""])[1]
    row.setdefault("_links", re.findall(r"LINK=(\S+)", r.stdout or ""))  # informational: dangling-install links made
    if agent:
        s = docker.exec(container_id, SCRUB + ("git gc --prune=now -q\n" if GIT_GC else ""), timeout=900)
        if s.exit_code != 0:
            raise ContainerSetupError(f"outside: git scrub failed on {row['instance_id']}: {s.stderr or s.stdout}")
    return head


class _HarnessPython:
    """Docker manager view whose exec runs the harness's `python3 -S -c` helpers with a Python >= 3.9 (the patch
    applier uses Path.is_relative_to), while the task's own env (maybe 3.6) stays first on PATH everywhere else."""

    def __init__(self, docker, pydir: str):
        self._d, self._pydir = docker, pydir

    def exec(self, container_id, command, **k):
        return self._d.exec(container_id, f"PATH={shlex.quote(self._pydir)}:$PATH; " + command, **k)

    def __getattr__(self, name):
        return getattr(self._d, name)


def _apply(docker, container_id: str, row: dict, text: str) -> tuple[int, str]:
    """Apply a diff in /workspace with the harness's own multi-strategy applier. Returns (exit code, detail)."""
    from swegemma.harness.container_setup import apply_patch_in_container

    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as f:
        f.write(text if text.endswith("\n") else text + "\n")
        host = Path(f.name)
    try:
        docker.copy_to(container_id, host, "/tmp/")
        pydir = os.path.dirname(image_info(row["docker_image"])["harness_python"])
        code, out, err = apply_patch_in_container(_HarnessPython(docker, pydir), container_id, f"/tmp/{host.name}")
        return code, (err or out or "")
    finally:
        host.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------- verification
async def verify_outside(docker, task, row: dict, agent_patch: str, *, timeout: int | None = None) -> dict:
    """Score a patch for an outside task in a fresh container of the task's image. Sets CURRENT itself."""
    from swegemma.harness.verification import strip_protected_files_from_patch
    from swegemma.models import extract_test_files_from_patch
    from swegemma.sandbox import sandbox_exec, sandbox_start, sandbox_stop

    t0 = time.perf_counter()
    token = CURRENT.set(row)
    out = {"resolved": False, "strict": False, "test_exit_code": -1, "error": None, "test_output": ""}
    cid = None
    try:
        cid = await sandbox_start(docker)
        head = await asyncio.to_thread(prepare_workspace, docker, cid, row, agent=False)
        out["head_ok"] = head == row["base_commit"]
        out["links"] = row.get("_links")
        test_files = extract_test_files_from_patch(task.test_patch)
        # the agent may not touch tests or runner config: the same stripping the Kaggle harness applies
        patch = strip_protected_files_from_patch(agent_patch or "", extra_protected_paths=test_files)
        out["patch_stripped"] = (patch.count("diff --git") != (agent_patch or "").count("diff --git"))
        if patch.strip():
            code, detail = await asyncio.to_thread(_apply, docker, cid, row, patch)
            if code != 0:
                out.update(error=f"agent patch did not apply: {detail[-400:]}", test_output=detail)
                return out
        # reset the test files to the base commit (tracked ones) and drop untracked ones, then add the test_patch
        q = " ".join(shlex.quote(p) for p in test_files)
        if q:
            await sandbox_exec(docker, cid, f"cd /workspace && tracked=$(git ls-tree -r --name-only HEAD -- {q}) && "
                                            f"[ -n \"$tracked\" ] && git ls-tree -r -z --name-only HEAD -- {q} | "
                                            f"xargs -0 git checkout HEAD -- ; git clean -fq -- {q}; true")
        code, detail = await asyncio.to_thread(_apply, docker, cid, row, task.test_patch)
        if code != 0:
            out.update(error=f"test_patch did not apply: {detail[-400:]}", test_output=detail)
            return out
        cmd = f"cd /workspace && {test_command(row['test_cmd'], task.test_patch)}"
        res = await sandbox_exec(docker, cid, cmd, timeout=timeout or VERIFY_TIMEOUT)
        log = f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        status = parse_pytest_log(res.stdout or "")
        f2p, p2p = list(task.FAIL_TO_PASS), list(task.PASS_TO_PASS)
        bad = [t for t in f2p + p2p if status.get(t) not in OK]
        timed_out = bool(getattr(res, "timed_out", False)) or res.exit_code == 124
        out.update(
            resolved=not bad and not timed_out and bool(f2p),
            test_exit_code=res.exit_code, test_output=log, test_cmd=cmd,
            f2p_pass=sum(status.get(t) in OK for t in f2p), f2p_total=len(f2p),
            p2p_pass=sum(status.get(t) in OK for t in p2p), p2p_total=len(p2p),
            not_passed={t: status.get(t, "MISSING") for t in bad[:12]}, n_not_passed=len(bad),
            counts={s: sum(v == s for v in status.values()) for s in STATUSES},
            timed_out=timed_out,
        )
        # the Kaggle verifier's extra rule on top: the whole run is clean (exit 0, nothing failed or errored)
        out["strict"] = out["resolved"] and res.exit_code == 0 and not out["counts"]["FAILED"] and not out["counts"]["ERROR"]
        if bad:
            out["error"] = f"{len(bad)} required test(s) not passed, e.g. {bad[0]}: {status.get(bad[0], 'MISSING')}"
        elif timed_out:
            out["error"] = f"test command timed out after {timeout or VERIFY_TIMEOUT}s"
        return out
    except Exception as exc:  # noqa: BLE001 - one task's setup failure must not end a run
        out["error"] = f"{type(exc).__name__}: {exc}"[:600]
        return out
    finally:
        out["verify_seconds"] = round(time.perf_counter() - t0, 1)
        if cid is not None:
            await sandbox_stop(docker, cid)
        CURRENT.reset(token)


# ------------------------------------------------------------------------------------------------ harness hooks
_INSTALLED = False


def task_config(cfg):
    """Per-task copy of the EvalConfig whose graph/embedding dirs are the outside ones (the prompt's graph-tool block
    and the graph tools read these)."""
    c = copy.copy(cfg)
    object.__setattr__(c, "graph_dir", str(GRAPH_DIR))
    object.__setattr__(c, "embeddings_dir", str(EMB_DIR))
    return c


def install(specs: dict[str, dict]) -> None:
    """Hook the harness so the given outside tasks run in their own images. Idempotent; our own tasks untouched."""
    global _INSTALLED
    SPECS.update(specs)
    if _INSTALLED:
        return
    _INSTALLED = True
    import docker.models.containers as dc
    import swegemma.harness.agent_runner as ar
    from swegemma.evaluate import Evaluator

    orig_run = dc.ContainerCollection.run

    def run(self, image=None, *a, **k):
        row = CURRENT.get()
        if row is not None:
            image = row["docker_image"]
            k["environment"] = {**(k.get("environment") or {}), **container_env(row)}
            k["labels"] = {**(k.get("labels") or {}), "outside_task": row["instance_id"]}
        return orig_run(self, image, *a, **k)

    dc.ContainerCollection.run = run

    # agent container setup steps (names as imported into agent_runner): skipped for outside tasks
    def skip(name):
        orig = getattr(ar, name)

        def f(*a, **k):
            if CURRENT.get() is None:
                return orig(*a, **k)
            return None

        setattr(ar, name, f)

    for name in ("setup_container_wheels", "install_editable_package", "install_test_dependencies",
                 "setup_workspace_test_config"):
        skip(name)
    orig_extract = ar.extract_snapshot

    def extract(docker, container_id, snapshot_path, **k):
        row = CURRENT.get()
        if row is None:
            return orig_extract(docker, container_id, snapshot_path, **k)
        prepare_workspace(docker, container_id, row, agent=True)

    ar.extract_snapshot = extract

    orig_eval = Evaluator.evaluate_task

    async def evaluate_task(self, task, *a, **k):
        row = SPECS.get(task.instance_id)
        if row is None:
            return await orig_eval(self, task, *a, **k)
        return await evaluate_outside(self, task, row, **k)

    Evaluator.evaluate_task = evaluate_task


# ---- stream mode (OUTSIDE_STREAM=1, 2026-10-26): each task fetches its own image, checks itself, runs, cleans up ----
# Why: with one try per task every image is used once, and the remote host's disk (not the GPU) caps how many images
# can sit unpacked at once; pulling per task keeps every image on disk in use. Gates protect the Docker host:
#   disk: no pull while the host has < OUTSIDE_DISK_FLOOR_GB + OUTSIDE_PULL_GB free (30 + 8)
#   RAM : no new task container while the host has < OUTSIDE_RAM_GATE_GB available (8; the hard floor of 5 GB is held
#         by kits/swe-agent/outside/ram_guard.sh, which kills our newest container below 5.5 GB)
# Before the agent: reference fix must resolve and an empty patch must not (one run each); else the task is marked
# INVALID_TASK and the agent is skipped. Results go to OUTSIDE_STREAM_GOLD (a gold.jsonl the manifest reads).
STREAM = os.environ.get("OUTSIDE_STREAM") == "1"
REMOTE = os.environ.get("OUTSIDE_REMOTE", "")  # ssh host of the Docker daemon (for its disk / RAM), "" = this machine
DISK_FLOOR_GB = float(os.environ.get("OUTSIDE_DISK_FLOOR_GB", "30"))
PULL_GB = float(os.environ.get("OUTSIDE_PULL_GB", "15"))  # unpacked task image + headroom
RAM_GATE_GB = float(os.environ.get("OUTSIDE_RAM_GATE_GB", "8"))
_host = {"at": 0.0, "disk": None, "ram": None}
_host_lock = asyncio.Lock()
_pulls: dict[str, asyncio.Lock] = {}
_inflight = [0]
_csem: list = []
CHECK_SLOW_S = float(os.environ.get("OUTSIDE_CHECK_SLOW_S", "240"))
_sem: list = []


def _pull_sem():
    if not _sem:
        _sem.append(asyncio.Semaphore(int(os.environ.get("OUTSIDE_PULL_PAR", "12"))))
    return _sem[0]


async def host_free() -> tuple[float, float]:
    """(disk GB free on /, RAM GB available) on the Docker host; one reading shared for 10 s."""
    async with _host_lock:
        if time.time() - _host["at"] > 10:
            cmd = "df --output=avail -B1 / | tail -1; awk '/MemAvailable/ {print $2*1024}' /proc/meminfo"
            argv = ["ssh", "-o", "BatchMode=yes", REMOTE, cmd] if REMOTE else ["bash", "-c", cmd]
            r = await asyncio.to_thread(subprocess.run, argv, capture_output=True, text=True, timeout=60)
            try:
                d, m = (float(x) / 1e9 for x in r.stdout.split()[:2])
                _host.update(at=time.time(), disk=d, ram=m)
            except ValueError:
                _host.update(at=time.time())
        return _host["disk"] or 0.0, _host["ram"] or 0.0


async def _gate(need_disk: float, need_ram: float, what: str):
    waited = 0
    while True:
        d, m = await host_free()
        if d >= need_disk and m >= need_ram:
            return waited
        if waited % 300 == 0:
            print(f"GATE {what}: host disk {d:.0f} GB (need {need_disk:.0f}), ram {m:.1f} GB (need {need_ram:.1f})",
                  flush=True)
        await asyncio.sleep(15)
        waited += 15


def _docker_cli(*args, timeout=1800):
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


async def stream_fetch(row: dict):
    """Pull the task's image (disk-gated, 3 attempts). Returns an error string or None."""
    img = row["docker_image"]
    lock = _pulls.setdefault(img, asyncio.Lock())
    async with lock:
        if (await asyncio.to_thread(_docker_cli, "image", "inspect", img, timeout=120)).returncode == 0:
            return None
        # 2026-10-27: the gate alone let ~120 concurrent pulls all pass at once and filled revideo's disk; now at
        # most PULL_PAR pulls run, and each counts the space of every pull still in flight before it starts
        async with _pull_sem():
            while True:
                d, _ = await host_free()
                if d >= DISK_FLOOR_GB + PULL_GB * (_inflight[0] + 1):
                    break
                await asyncio.sleep(15)
            _inflight[0] += 1
            try:
                for wait in (0, 60, 180):
                    await asyncio.sleep(wait)
                    r = await asyncio.to_thread(_docker_cli, "pull", "-q", img)
                    if r.returncode == 0:
                        return None
                return f"PULL_FAILED: {(r.stderr or r.stdout).strip()[-200:]}"
            finally:
                _inflight[0] -= 1
                _host["at"] = 0.0  # disk changed


def stream_drop(row: dict):
    """Remove the task's image (the pinned base image stays: it is a second name for shared layers)."""
    _docker_cli("rmi", "-f", row["docker_image"], timeout=600)
    _host["at"] = 0.0


async def stream_check(docker, task, row: dict) -> str | None:
    """Reference fix must resolve, empty patch must not (run together). Logs to OUTSIDE_STREAM_GOLD. None = sound."""
    # 2026-10-27: 128 tasks checking at once on revideo stretched a check from ~40 s to ~750 s and timing-sensitive
    # tests failed, so sound tasks were marked INVALID. Now at most OUTSIDE_CHECK_PAR checks run at once, and a check
    # slower than OUTSIDE_CHECK_SLOW_S says the host was overloaded, not that the task is bad (retried later).
    if not _csem:
        _csem.append(asyncio.Semaphore(int(os.environ.get("OUTSIDE_CHECK_PAR", "12"))))
    async with _csem[0]:
        t0 = time.time()
        gold, empty = await asyncio.gather(verify_outside(docker, task, row, task.patch),
                                           verify_outside(docker, task, row, ""))
    out = os.environ.get("OUTSIDE_STREAM_GOLD")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "a") as fh:
            for kind, v in (("gold", gold), ("empty", empty)):
                fh.write(json.dumps({"id": row["instance_id"], "kind": kind, "run": 1, "resolved": v["resolved"],
                                     "strict": v.get("strict"), "exit": v.get("test_exit_code"),
                                     "f2p": f"{v.get('f2p_pass')}/{v.get('f2p_total')}",
                                     "p2p": f"{v.get('p2p_pass')}/{v.get('p2p_total')}", "error": v.get("error"),
                                     "seconds": round(time.time() - t0, 1), "image": row["docker_image"]}) + "\n")
    if any("Timeout" in str(v.get("error") or "") for v in (gold, empty)):  # host overloaded, not a bad task
        return f"PULL_FAILED: check timed out on the Docker host ({str(gold.get('error') or empty.get('error'))[:120]})"
    if (not gold["resolved"] or empty["resolved"]) and time.time() - t0 > CHECK_SLOW_S:
        return f"PULL_FAILED: check took {time.time() - t0:.0f} s (host overloaded); verdict not trusted"
    if not gold["resolved"]:
        return f"INVALID_TASK: reference fix does not resolve ({(gold.get('error') or '')[:150]})"
    if empty["resolved"]:
        return "INVALID_TASK: empty patch resolves"
    return None


def stream_graph(row: dict) -> str | None:
    """Make sure the task's code graph + .npz exist (built from its judge packet if missing). None = ready."""
    stem = graph_stem(row)
    if (GRAPH_DIR / f"{stem}.json").exists() and (EMB_DIR / f"{stem}.npz").exists():
        return None
    tj = os.environ.get("TASKS_JSONL")
    r = subprocess.run([str(ROOT / "envs/harness/bin/python"), "-I", str(Path(__file__).with_name("build_graphs.py")),
                        tj, row["instance_id"]], capture_output=True, text=True, timeout=1800)
    if (GRAPH_DIR / f"{stem}.json").exists() and (EMB_DIR / f"{stem}.npz").exists():
        return None
    return f"GRAPH_FAILED: {(r.stderr or r.stdout).strip()[-200:]}"


async def evaluate_outside(evaluator, task, row: dict, task_index: int = 1, total_tasks: int = 1, context=None, **_):
    """The harness's evaluate_task for an outside task: Phase 1 = the harness's own run_agent_sandbox (hooked above),
    Phase 2 = verify_outside. In stream mode: graph + fetch + check first, image removed at the end."""
    if not STREAM:
        return await _evaluate_outside(evaluator, task, row, task_index, total_tasks, context)
    from swegemma.models import TaskResult
    start = time.perf_counter()
    common = dict(instance_id=task.instance_id, repo=task.repo, agent_patch="", task_index=task_index)
    try:
        err = await asyncio.to_thread(stream_graph, row) or await stream_fetch(row)
        if err is None:
            await _gate(0, RAM_GATE_GB, f"check {row['instance_id']}")
            err = await stream_check(evaluator.docker, task, row)  # verify_outside sets CURRENT itself
        if err is not None:
            return TaskResult(resolved=False, test_output="", test_exit_code=-1, error=err,
                              duration_seconds=time.perf_counter() - start, **common)
        await _gate(0, RAM_GATE_GB, f"agent {row['instance_id']}")
        return await _evaluate_outside(evaluator, task, row, task_index, total_tasks, context)
    finally:
        await asyncio.to_thread(stream_drop, row)


async def _evaluate_outside(evaluator, task, row: dict, task_index: int = 1, total_tasks: int = 1, context=None):
    import swegemma.harness.agent_runner as ar
    from swegemma.context import SwegemmaContext
    from swegemma.harness.verification import save_trace_artifact
    from swegemma.models import TaskResult

    cfg = task_config(evaluator.config)
    start = time.perf_counter()
    token = CURRENT.set(row)
    try:
        if context is None:
            context = SwegemmaContext(
                docker_manager=evaluator.docker, container_id="", problem_statement=task.problem_statement,
                hints_text=getattr(task, "hints_text", "") or "", task=task, repo=task.repo,
                graph_dir=cfg.graph_dir, embeddings_dir=cfg.embeddings_dir, budget=cfg.budget, harness=cfg.harness)
        agent_patch, agent_error, trace = await ar.run_agent_sandbox(
            evaluator.docker, cfg, task, Path("/outside-task-has-no-snapshot"),
            task_index=task_index, total_tasks=total_tasks, context=context)
    finally:
        CURRENT.reset(token)
    json_path = save_trace_artifact(trace, cfg.results_dir, task.instance_id)
    common = dict(instance_id=task.instance_id, repo=task.repo, agent_patch=agent_patch or "", task_index=task_index,
                  trace_json_path=json_path, total_llm_calls=context.llm_calls_used, tool_calls=context.tool_calls_used)
    if agent_error and not agent_patch:
        return TaskResult(resolved=False, test_output="", test_exit_code=-1, error=agent_error,
                          duration_seconds=time.perf_counter() - start, **common)
    context.set_phase("Verifying")
    v = await verify_outside(evaluator.docker, task, row, agent_patch)
    extra = {k: v.get(k) for k in ("strict", "f2p_pass", "f2p_total", "p2p_pass", "p2p_total", "n_not_passed",
                                   "head_ok", "patch_stripped", "verify_seconds")}
    return TaskResult(resolved=v["resolved"], test_output=v["test_output"], test_exit_code=v["test_exit_code"],
                      error=None if v["resolved"] else (v["error"] or agent_error),
                      duration_seconds=time.perf_counter() - start, outside=extra, **common)
