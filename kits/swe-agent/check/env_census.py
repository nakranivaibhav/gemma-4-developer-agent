"""Which public tasks can pass in the local sandbox at all? No model involved: apply the REAL fix + its tests, run pytest.

usage: python -I kits/swe-agent/check/env_census.py <out.jsonl> <base_tar> [--prefix fastapi_] [--par 4] [--honor-pins]
--honor-pins: after the editable install, re-resolve the snapshot's own Requires-Dist against /wheels with pip (what the
harness would see if it honoured version pins instead of injecting the newest wheel of everything). A what-if, not the harness.
<base_tar> is the harness's unpacked-wheel tar (swegemma_sp_cache_v8/sp_base.tar built from the wheel set under test).
Mirrors swegemma's sandbox setup: snapshot -> /workspace, base wheels -> site-packages, editable install --no-deps,
then git apply gold patch + test patch and `pytest <test files> -q -p no:anyio -o timeout=0` like the harness.
One line per task: gold_resolved (exit 0), the pytest summary, and the first error line when it failed.
"""
import json
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repo root
IMAGE = "swebench-sandbox:latest"
SP = "/usr/local/lib/python3.13/site-packages"
lock = threading.Lock()
HONOR_PINS = "--honor-pins" in sys.argv  # see census_one


def sh(*cmd, timeout=600, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)


def census_one(task, base_tar, out):
    tid = task["instance_id"]
    rec = {"id": tid, "repo": task["repo"]}
    cid = sh("docker", "run", "-d", IMAGE, "sleep", "infinity").stdout.strip()
    try:
        sh("docker", "cp", str(ROOT / "data" / "snapshots" / f"{tid}.tgz"), f"{cid}:/tmp/snap.tgz")
        sh("docker", "exec", cid, "sh", "-c", "mkdir -p /workspace && tar xzf /tmp/snap.tgz -C /workspace")
        sh("docker", "cp", str(base_tar), f"{cid}:/tmp/sp.tar")
        sh("docker", "exec", cid, "tar", "xf", "/tmp/sp.tar", "-C", SP)
        r = sh("docker", "exec", cid, "sh", "-c",
               "cd /workspace && pip install --no-index --find-links=/wheels --no-build-isolation --no-deps -e /workspace -q 2>&1 | tail -1")
        rec["install_tail"] = r.stdout.strip()[-200:]
        if HONOR_PINS:  # what-if: re-resolve the snapshot's own Requires-Dist against /wheels (the harness injects newest-wins)
            pkg = task["repo"].split("/")[-1]
            # the snapshot's own pins come from its pyproject (importlib.metadata would answer for the same-named wheel
            # that the base set already put in site-packages, e.g. fastapi 0.141.1). The injected packages carry no
            # RECORD, so pip cannot uninstall them: the violating ones are removed by directory, then reinstalled from /wheels.
            script = ("import tomllib, importlib.metadata as m, shutil, glob\n"
                      "from pip._vendor.packaging.requirements import Requirement\n"
                      "try: reqs = tomllib.load(open('/workspace/pyproject.toml','rb'))['project']['dependencies']\n"
                      f"except Exception: reqs = [r for r in (m.requires({pkg!r}) or []) if 'extra ==' not in r]\n"
                      f"sp = {SP!r}; todo = []\n"
                      "for r in reqs:\n"
                      "    req = Requirement(r)\n"
                      "    if req.marker and not req.marker.evaluate(): continue\n"
                      "    try: v = m.version(req.name)\n"
                      "    except m.PackageNotFoundError: v = None\n"
                      "    if v is None or (req.specifier and not req.specifier.contains(v, prereleases=True)):\n"
                      "        todo.append(str(req).split(';')[0])\n"
                      "        n = req.name.lower().replace('-', '_')\n"
                      "        for d in glob.glob(f'{sp}/{n}') + glob.glob(f'{sp}/{n}-*.dist-info') + glob.glob(f'{sp}/{req.name}-*.dist-info'): shutil.rmtree(d, ignore_errors=True)\n"
                      "open('/tmp/reqs.txt','w').write('\\n'.join(todo)+'\\n')\n"
                      "print('UNSATISFIED', todo)\n")
            subprocess.run(["docker", "exec", "-i", cid, "sh", "-c", "cat > /tmp/reqs.py"], input=script, text=True, capture_output=True)
            before = sh("docker", "exec", cid, "sh", "-c", "pip list --format=freeze 2>/dev/null").stdout
            p = sh("docker", "exec", cid, "sh", "-c",
                   "cd /workspace && python3 -s /tmp/reqs.py && { [ -s /tmp/reqs.txt ] && pip install --no-index --find-links=/wheels --no-deps --ignore-installed -r /tmp/reqs.txt -q 2>&1 | grep -v 'WARNING\\|notice' | tail -3; true; }")
            after = sh("docker", "exec", cid, "sh", "-c", "pip list --format=freeze 2>/dev/null").stdout
            rec["pins_unsatisfied"] = (re.search(r"UNSATISFIED (.*)", p.stdout) or [None, None])[1]
            rec["pins_applied"] = sorted(set(after.split()) - set(before.split()))  # packages whose version moved
            rec["pins_tail"] = p.stdout.replace("UNSATISFIED", "").strip()[-200:]
        sh("docker", "exec", cid, "sh", "-c", "cd /workspace && git init -q 2>/dev/null; git add -A >/dev/null 2>&1; git -c user.email=a@b -c user.name=a commit -qm base --allow-empty >/dev/null 2>&1")
        # the harness's hermetic test config: pytest.ini + a conftest.py at the workspace root (which also puts /workspace
        # first on sys.path, so the repo shadows any same-named wheel in site-packages)
        ini = "[pytest]\naddopts = -p no:anyio\nnorecursedirs = .* build dist venv\npython_classes = Test* *Test\npython_files = test_*.py *_test.py\nfilterwarnings =\n    ignore::DeprecationWarning\n    ignore::UserWarning\n"
        subprocess.run(["docker", "exec", "-i", cid, "sh", "-c", "cat > /workspace/pytest.ini"], input=ini, text=True, capture_output=True)
        sh("docker", "exec", cid, "sh", "-c", "cd /workspace && { printf '# Hermetic test discovery hook for SWE-gemma\n\n'; cat conftest.py 2>/dev/null; } > /tmp/ct && mv /tmp/ct conftest.py")
        for name, text in (("gold.diff", task["patch"]), ("tests.diff", task["test_patch"])):
            text = text if text.endswith("\n") else text + "\n"  # the task file stores patches without the final newline
            subprocess.run(["docker", "exec", "-i", cid, "sh", "-c", f"cat > /tmp/{name}"], input=text, text=True, capture_output=True)
            a = sh("docker", "exec", cid, "sh", "-c", f"cd /workspace && (git apply --whitespace=nowarn /tmp/{name} || git apply --recount --ignore-whitespace /tmp/{name} || git apply -3 /tmp/{name} || patch -p1 -N -s < /tmp/{name}) 2>&1")
            rec[f"apply_{name.split('.')[0]}"] = a.stdout.strip()[-200:]
        files = sorted(set(re.findall(r"^\+\+\+ b/(\S+)", task["test_patch"], re.M)))
        rec["test_files"] = files
        imp = sh("docker", "exec", cid, "sh", "-c", f"cd /workspace && PYTHONSAFEPATH=1 python3 -s -c 'import {task['repo'].split('/')[-1]}' 2>&1 | tail -1")
        rec["import_tail"] = imp.stdout.strip()[-200:]
        t = sh("docker", "exec", cid, "sh", "-c",
               "cd /workspace && PYTHONSAFEPATH=1 PYTHONNOUSERSITE=1 timeout 900 python3 -s -m pytest " + " ".join(files) +
               " -p no:anyio -o timeout=0 -o 'python_classes=Test* *Test' -q 2>&1 | tail -40", timeout=1000)
        tail = t.stdout
        summ = re.findall(r"^(?:=+ )?(\d+ (?:passed|failed|error)[^\n]*?)(?: =+)?$", tail, re.M)
        rec["gold_rc"] = t.returncode
        rec["summary"] = summ[-1] if summ else tail.strip().splitlines()[-1][:200] if tail.strip() else ""
        rec["gold_resolved"] = t.returncode == 0 and "passed" in rec["summary"] and "failed" not in rec["summary"] and "error" not in rec["summary"]
        err = re.search(r"^E +(\S.*)$", tail, re.M) or re.search(r"(ModuleNotFoundError|ImportError|TypeError|AttributeError)[^\n]*", tail)
        rec["first_error"] = (err.group(1) if err.lastindex else err.group(0))[:200] if err else None
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        sh("docker", "rm", "-f", cid)
    with lock:
        with open(out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{tid:16s} gold_resolved={rec.get('gold_resolved')} {rec.get('summary', rec.get('error'))!s:40.40s} {rec.get('first_error') or ''}"[:200], flush=True)


def main():
    out, base_tar = Path(sys.argv[1]), Path(sys.argv[2])
    prefix = sys.argv[sys.argv.index("--prefix") + 1] if "--prefix" in sys.argv else ""
    par = int(sys.argv[sys.argv.index("--par") + 1]) if "--par" in sys.argv else 4
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    tasks = [json.loads(l) for l in open(ROOT / "data" / "tasks.jsonl")]
    tasks = [t for t in tasks if t["instance_id"].startswith(prefix) and t["instance_id"] not in done]
    print(f"census: {len(tasks)} tasks, {par} at a time, base tar {base_tar} ({base_tar.stat().st_size >> 20} MB)", flush=True)
    with ThreadPoolExecutor(par) as ex:
        list(ex.map(lambda t: census_one(t, base_tar, out), tasks))
    print("CENSUS_DONE", flush=True)


if __name__ == "__main__":
    main()
