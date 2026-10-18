"""A shared task queue for several model servers (GPUs) working on the same set of passes at once.

usage:
  python -I kits/swe-agent/run/fleet.py init <fleet_dir> <manifest.json>     write the manifest + an empty state
  python -I kits/swe-agent/run/fleet.py set <fleet_dir> <gpu> <target>       tasks-at-once for one GPU (0 = stop taking new tasks)
  python -I kits/swe-agent/run/fleet.py schedule <fleet_dir> <gpu> <m:t,...>  step plan from now: minute:target, e.g. 0:8,30:16,60:0
  python -I kits/swe-agent/run/fleet.py monitor <fleet_dir>                   every 60 s: apply schedules, save each server's /metrics,
                                                                  log host load; exits when every run of every task is done
  python -I kits/swe-agent/run/fleet.py status <fleet_dir>

The manifest names the passes (run ids) and, per GPU, an ordered task list; `spare` is a pool any GPU may take from.
A GPU takes a whole TASK: the 4 runs of a task (one per run id) all go to the GPU that claimed it, so the baseline and
the candidate meet the same server and the same load. The worker processes of one GPU (one per run id) stay in lockstep:
a process may claim a new task only when no sibling has claimed fewer. Nothing runs twice: a run is either done, in
flight, or free. When a GPU is marked gone, its tasks with runs left over go back to the free pool for the others.
Order of choice for a new task: own list, then spare, then the unstarted tail of another GPU's list (that GPU's list is
only open to others once it is gone, except `local`'s, which the rented GPUs may take from).
"""
import fcntl
import json
import os
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _locked(fleet_dir):
    fleet_dir = Path(fleet_dir)
    with open(fleet_dir / "lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        st = json.loads((fleet_dir / "state.json").read_text())
        yield st
        tmp = fleet_dir / "state.json.tmp"
        tmp.write_text(json.dumps(st, indent=1))
        tmp.replace(fleet_dir / "state.json")


def _alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def _event(fleet_dir, **kw):
    kw["t"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(Path(fleet_dir) / "events.jsonl", "a") as fh:
        fh.write(json.dumps(kw) + "\n")


def manifest(fleet_dir):
    return json.loads((Path(fleet_dir) / "manifest.json").read_text())


def _inflight(st, gpu):
    return sum(n for pid, d in st["inflight"].items() if d["gpu"] == gpu and _alive(pid) for n in [len(d["jobs"])])


def claim(fleet_dir, gpu, run_id, pid):
    """Next task for (gpu, run_id), or None (nothing for now), or 'STOP' (this GPU takes no more work)."""
    m = manifest(fleet_dir)
    with _locked(fleet_dir) as st:
        for p in [p for p in st["inflight"] if not _alive(p)]:  # a dead worker's runs are free again
            for tid, rid in st["inflight"][p]["jobs"]:
                _event(fleet_dir, ev="orphan_freed", gpu=st["inflight"][p]["gpu"], task=tid, run=rid)
            del st["inflight"][p]
        target = st["targets"].get(gpu, 0)
        if target <= 0:
            return "STOP"
        if _inflight(st, gpu) >= target:
            return None
        done = set(map(tuple, st["done"]))
        busy = {tuple(j) for d in st["inflight"].values() for j in d["jobs"]}
        mine = [t for t, g in st["claims"].items() if g == gpu]  # in claim order (dicts keep insertion order)
        # 1) a task this GPU already holds whose run for run_id is still free
        tid = next((t for t in mine if (t, run_id) not in done and (t, run_id) not in busy), None)
        if tid is None:
            counts = st["taken"].setdefault(gpu, {})
            sibs = [counts.get(r, 0) for r in m["runs"]]
            if counts.get(run_id, 0) > min(sibs):  # lockstep: wait for the slower siblings
                return None
            gone = set(st["gone"])
            free = lambda t: t not in st["claims"]  # noqa: E731
            pools = [m["gpus"].get(gpu, {}).get("tasks", []), m.get("spare", [])]
            for g, spec in m["gpus"].items():
                if g != gpu and (g in gone or g == "local"):
                    pools.append(list(reversed(spec.get("tasks", []))))
            tid = next((t for pool in pools for t in pool if free(t)), None)
            if tid is None:
                return None
            st["claims"][tid] = gpu
            _event(fleet_dir, ev="claim", gpu=gpu, task=tid, target=target)
        st["taken"].setdefault(gpu, {})[run_id] = st["taken"][gpu].get(run_id, 0) + 1
        st["inflight"].setdefault(str(pid), {"gpu": gpu, "jobs": []})["jobs"].append([tid, run_id])
        return tid


def finish(fleet_dir, gpu, run_id, pid, tid, ok=True):
    with _locked(fleet_dir) as st:
        d = st["inflight"].get(str(pid))
        if d and [tid, run_id] in d["jobs"]:
            d["jobs"].remove([tid, run_id])
        if ok:
            st["done"].append([tid, run_id])
        _event(fleet_dir, ev="done" if ok else "failed", gpu=gpu, task=tid, run=run_id)


def all_done(fleet_dir):
    m = manifest(fleet_dir)
    st = json.loads((Path(fleet_dir) / "state.json").read_text())
    return len({tuple(x) for x in st["done"]}) >= len(m["all_tasks"]) * len(m["runs"])


def target(fleet_dir, gpu):
    return json.loads((Path(fleet_dir) / "state.json").read_text())["targets"].get(gpu, 0)


def set_target(fleet_dir, gpu, n, why="manual"):
    with _locked(fleet_dir) as st:
        st["targets"][gpu] = int(n)
    _event(fleet_dir, ev="target", gpu=gpu, target=int(n), why=why)


def mark_gone(fleet_dir, gpu):
    """The GPU has left: its claims on tasks with runs not done go back to the free pool."""
    m = manifest(fleet_dir)
    with _locked(fleet_dir) as st:
        st["targets"][gpu] = 0
        if gpu not in st["gone"]:
            st["gone"].append(gpu)
        done = {tuple(x) for x in st["done"]}
        for t in [t for t, g in st["claims"].items() if g == gpu]:
            left = [r for r in m["runs"] if (t, r) not in done]
            if left:
                del st["claims"][t]
                _event(fleet_dir, ev="released", gpu=gpu, task=t, runs_left=left, split=len(left) < len(m["runs"]))
    _event(fleet_dir, ev="gone", gpu=gpu)


def _scrape(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15) as r:
            return "\n".join(l for l in r.read().decode().splitlines() if l.startswith("vllm:"))
    except Exception as exc:  # a missing server is logged, not fatal
        return f"# scrape failed: {exc}"


def monitor(fleet_dir):
    fleet_dir = Path(fleet_dir)
    (fleet_dir / "metrics").mkdir(exist_ok=True)
    host = fleet_dir / "host.csv"
    if not host.exists():
        host.write_text("t,load1,load5,mem_avail_gb,containers\n")
    while not all_done(fleet_dir):
        now = time.time()
        sch = json.loads((fleet_dir / "schedules.json").read_text()) if (fleet_dir / "schedules.json").exists() else {}
        st = json.loads((fleet_dir / "state.json").read_text())
        for gpu, plan in sch.items():  # plan: {"t0": epoch, "steps": [[minute, target], ...]}
            due = [tg for mi, tg in plan["steps"] if now >= plan["t0"] + 60 * mi]
            if due and st["targets"].get(gpu) != due[-1] and plan.get("applied", -1) < len(due) - 1:
                set_target(fleet_dir, gpu, due[-1], why=f"schedule step {len(due) - 1}")
                plan["applied"] = len(due) - 1
                (fleet_dir / "schedules.json").write_text(json.dumps(sch, indent=1))
        m = manifest(fleet_dir)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        for gpu, spec in m["gpus"].items():
            if gpu in st["gone"]:
                continue
            for port in spec.get("ports", []):
                d = fleet_dir / "metrics" / f"{gpu}_{port}"
                d.mkdir(exist_ok=True)
                (d / f"{stamp}.txt").write_text(_scrape(port))
        l1, l5 = open("/proc/loadavg").read().split()[:2]
        mem = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) / 2**20
        n = subprocess.run(["docker", "ps", "-q"], capture_output=True, text=True).stdout.count("\n")
        with open(host, "a") as fh:
            fh.write(f"{stamp},{l1},{l5},{mem:.1f},{n}\n")
        time.sleep(max(1, 60 - (time.time() - now)))
    _event(fleet_dir, ev="all_done")
    print("FLEET_ALL_DONE", flush=True)


def status(fleet_dir):
    m = manifest(fleet_dir)
    st = json.loads((Path(fleet_dir) / "state.json").read_text())
    done = {tuple(x) for x in st["done"]}
    print(f"runs done {len(done)}/{len(m['all_tasks']) * len(m['runs'])}  targets {st['targets']}  gone {st['gone']}")
    for g in m["gpus"]:
        held = [t for t, gg in st["claims"].items() if gg == g]
        fin = sum((t, r) in done for t in held for r in m["runs"])
        print(f"  {g:10s} in flight {_inflight(st, g):3d}  tasks held {len(held):3d}  runs done {fin}")


if __name__ == "__main__":
    cmd, fd = sys.argv[1], sys.argv[2]
    if cmd == "init":
        Path(fd).mkdir(parents=True, exist_ok=True)
        (Path(fd) / "manifest.json").write_text(Path(sys.argv[3]).read_text())
        if not (Path(fd) / "state.json").exists():
            (Path(fd) / "state.json").write_text(json.dumps(
                {"targets": {}, "claims": {}, "taken": {}, "inflight": {}, "done": [], "gone": []}))
    elif cmd == "set":
        set_target(fd, sys.argv[3], sys.argv[4])
    elif cmd == "gone":
        mark_gone(fd, sys.argv[3])
    elif cmd == "schedule":
        p = Path(fd) / "schedules.json"
        sch = json.loads(p.read_text()) if p.exists() else {}
        sch[sys.argv[3]] = {"t0": time.time(), "steps": [[int(a), int(b)] for a, b in
                                                         (x.split(":") for x in sys.argv[4].split(","))]}
        p.write_text(json.dumps(sch, indent=1))
        _event(fd, ev="schedule", gpu=sys.argv[3], steps=sys.argv[4])
    elif cmd == "monitor":
        monitor(fd)
    elif cmd == "status":
        status(fd)
