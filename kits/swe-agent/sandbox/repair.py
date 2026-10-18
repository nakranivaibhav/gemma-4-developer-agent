# ---------------------------------------------------------------------------------------------------------------------
# LOCAL REPAIR — appended to the organizer's sandbox/setup.py by kits/swe-agent/sandbox/build_hook.py. Not part of the competition.
#
# Why: the harness unpacks the NEWEST wheel of every package into site-packages, whatever the snapshot pins, and the
# competition wheel set lacks some libraries the public repos' tests import. The real fix then fails its own tests
# (measured: runs/gold_verify/). This step runs inside every container (agent's and verifier's) right after
# the organizer's setup, and only moves the environment toward what the snapshot itself declares:
#   1. PINS    — for each dependency the snapshot declares with a version range, if the injected version is outside the
#                range, swap in the newest wheel from /wheels that is inside it.
#   2. OWN CODE — the wheel set also holds RELEASED copies of the task repos themselves (requests, httpx, rich, fastapi).
#                Unpacked into site-packages they shadow /workspace, so tests run against the release, not the task's
#                code (measured: requests and httpx always). The released copy of the repo under test is removed and the
#                workspace is installed editable again.
#   3. EXTRAS  — libraries the repo's own tests import but the competition wheel set lacks. The local image carries them
#                as wheels on a shelf (/wheels_extra, kits/swe-agent/sandbox/Dockerfile); only the ones listed for the repo under
#                test are installed (EXTRAS below), because one repo's extra can break another's tests (measured:
#                brotlicffi, needed by httpbin for requests, changes the Accept-Encoding header fastapi's tests expect).
#   3b. LOCKS  — for the packages listed in LOCK_HONOR, the exact version in the snapshot's lock file is used (rich's
#                tests compare rendered output byte for byte against the Pygments version locked at that commit).
#   4. ASYNC   — the harness runs pytest with `-p no:anyio`, which blocks the anyio plugin by its entry-point NAME, so
#                tests marked @pytest.mark.anyio fail as "async def functions are not natively supported". When the
#                workspace's tests use that marker, the same plugin is registered under a second name (see below).
#   5. RULES   — explicit, documented per-repo overrides (LOCAL_RULES below) for what pins cannot express.
# Everything it does is written to /tmp/local_repair.json in the container.
# ---------------------------------------------------------------------------------------------------------------------
import json as _json
import shutil as _shutil

# repo short name -> list of {"when": <requirement string that must be true of the WORKSPACE package version, or "">,
#                             "require": <requirement string to enforce>, "why": <one line>}
LOCAL_RULES: dict = {
    "requests": [
        {"when": "requests<2.32", "require": "werkzeug<3",
         "why": "these snapshots' test_redirecting_to_bad_url expects httpbin to redirect to http://localhost:-1; on Werkzeug 3 "
                "httpbin answers 500 instead (ValueError in iri_to_uri), so no InvalidURL is raised. Upstream later changed "
                "the test; Werkzeug 2.3.8 is in the competition wheel set."},
        {"when": "requests<2.32", "require": "flask<3", "why": "Flask 3 requires Werkzeug 3; Flask 2.3.3 is in the competition wheel set."},
    ],
}
# repo short name -> distributions to install from the shelf when absent (--no-deps: the list is the whole closure)
EXTRAS: dict = {
    "fastapi": ["typing-inspection", "inline-snapshot", "executing", "asttokens", "dirty-equals", "python-multipart",
                "ujson", "orjson", "h11", "httpx2", "httpcore2", "truststore"],
    "httpx": ["h11"],
    "requests": ["pytest-httpbin", "httpbin", "blinker", "brotlicffi", "cffi", "pycparser", "decorator", "six", "greenlet",
                 "trustme", "cryptography"],
    "rich": ["attrs"],
}
# repo short name -> packages whose version is taken from the snapshot's lock file (poetry.lock / uv.lock)
LOCK_HONOR: dict = {"rich": ["pygments", "attrs"]}
SHELF = Path("/wheels_extra")
# top-level workspace directories the organizer's fast path must NOT put on sys.path (they shadow real libraries)
PTH_EXCLUDE: dict = {}


def _dist_name(dist_info_dir):
    """'starlette-1.6.0.dist-info' -> 'starlette' (the suffix itself contains a hyphen)."""
    return dist_info_dir.name[: -len(".dist-info")].rsplit("-", 1)[0]


def _wheel_index():
    from pip._vendor.packaging.version import InvalidVersion, Version
    index = {}
    for root in (WHEELS_DIR, SHELF):
        for f in os.listdir(root) if root.exists() else []:
            info = parse_wheel_filename(f)
            if info is None:
                continue
            try:
                index.setdefault(info["normalized"], []).append((Version(info["version"]), str(root / f)))
            except InvalidVersion:
                pass
    return index


def _locked_versions(ws):
    """{normalized name: version} from the snapshot's lock file, if it ships one."""
    out = {}
    for name in ("poetry.lock", "uv.lock"):
        lock = ws / name
        if lock.exists() and tomllib is not None:
            try:
                with open(lock, "rb") as fh:
                    for pkg in tomllib.load(fh).get("package", []):
                        out.setdefault(normalize_pkg_name(pkg["name"]), str(pkg["version"]))
            except Exception as exc:
                print(f"local repair: could not read {lock}: {exc}", file=sys.stderr)
    return out


def _declared_requirements(ws):
    """Version-ranged core dependencies the snapshot declares for itself: pyproject [project].dependencies, else setup.cfg."""
    reqs = []
    pp = ws / "pyproject.toml"
    if pp.exists() and tomllib is not None:
        try:
            with open(pp, "rb") as fh:
                data = tomllib.load(fh)
            reqs += list(data.get("project", {}).get("dependencies", []) or [])
            poetry = data.get("tool", {}).get("poetry", {}).get("dependencies", {}) or {}
            for name, spec in poetry.items():  # poetry: {"pygments": "^2.13.0"} or {"pygments": {version = "..."}}
                if name.lower() == "python":
                    continue
                spec = spec.get("version", "") if isinstance(spec, dict) else spec
                reqs.append(name + _poetry_to_pep440(str(spec)))
        except Exception as exc:
            print(f"local repair: could not read {pp}: {exc}", file=sys.stderr)
    return reqs


def _poetry_to_pep440(spec):
    spec = spec.strip()
    if not spec or spec == "*":
        return ""
    out = []
    for part in [p.strip() for p in spec.split(",") if p.strip()]:
        if part[0] in "^~" and part[1:2].isdigit():
            nums = [int(n) for n in re.findall(r"\d+", part[1:])][:3]
            if part[0] == "^":
                i = next((k for k, n in enumerate(nums) if n != 0), len(nums) - 1)
                upper = nums[:i] + [nums[i] + 1]
            else:
                upper = [nums[0], nums[1] + 1] if len(nums) > 1 else [nums[0] + 1]
            out.append(f">={part[1:]},<{'.'.join(map(str, upper))}")
        elif part[0].isdigit():
            out.append("==" + part)
        else:
            out.append(part)
    return ",".join(out)


def _installed_version(name):
    import importlib.metadata as md
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _swap(sp, norm, filename, log):
    """Replace an injected package with a specific wheel file (injected packages have no RECORD: remove by directory)."""
    import importlib.metadata as md
    tops = {norm.replace("-", "_")}
    try:
        dist = md.distribution(norm)
        tops |= {t.strip() for t in (dist.read_text("top_level.txt") or "").split() if t.strip()}
    except Exception:
        pass
    # a pip-installed copy (image or shelf) has a RECORD and is removed cleanly; injected leftovers go by directory
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "--disable-pip-version-check", norm],
                   capture_output=True, text=True, check=False)
    for top in tops:
        for cand in (sp / top, sp / (top + ".py")):
            if cand.is_dir():
                _shutil.rmtree(cand, ignore_errors=True)
            elif cand.exists():
                cand.unlink()
    for di in list(sp.glob("*.dist-info")):
        if normalize_pkg_name(_dist_name(di)) == norm:
            _shutil.rmtree(di, ignore_errors=True)
    res = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-build-isolation", "-q",
                          "--disable-pip-version-check", filename], capture_output=True, text=True, check=False)
    log.append({"package": norm, "installed": os.path.basename(filename), "rc": res.returncode, "err": (res.stderr or "")[-200:] if res.returncode else ""})


def _own_code_from_workspace(sp, ws, repo_short):
    """Make `import <repo>` resolve to /workspace: drop the released copy that the wheel injection put in site-packages."""
    top = repo_short.replace("-", "_")
    src = next((d for d in (ws / top, ws / "src" / top) if (d / "__init__.py").exists()), None)
    if src is None:
        return {"own_package": top, "action": "not found in workspace"}
    removed = []
    injected = sp / top
    if injected.is_dir() and not injected.is_symlink():
        _shutil.rmtree(injected, ignore_errors=True)
        removed.append(str(injected))
    for di in list(sp.glob("*.dist-info")):
        if normalize_pkg_name(_dist_name(di)) == normalize_pkg_name(repo_short) and not (di / "direct_url.json").exists():
            _shutil.rmtree(di, ignore_errors=True)  # the release's metadata; an editable install carries direct_url.json
            removed.append(str(di))
    res = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={WHEELS_DIR}", "--no-build-isolation",
                          "--no-deps", "-q", "--disable-pip-version-check", "-e", str(ws)], capture_output=True, text=True, check=False)
    probe = subprocess.run([sys.executable, "-s", "-c", f"import {top}; print({top}.__file__)"], capture_output=True, text=True,
                           check=False, cwd="/", env={**os.environ, "PYTHONSAFEPATH": "1", "PYTHONNOUSERSITE": "1"})
    return {"own_package": top, "source": str(src), "removed": removed, "editable_rc": res.returncode,
            "editable_err": (res.stderr or "").strip()[-300:] if res.returncode else "",
            "imports_from": (probe.stdout or probe.stderr).strip().splitlines()[-1][:200] if (probe.stdout or probe.stderr).strip() else ""}


def _enable_anyio_tests(ws, sp):
    """Register anyio's pytest plugin under a second entry-point name when the workspace's tests use it.

    The harness passes `-p no:anyio` on the pytest command line and rewrites pytest.ini just before the run, so the
    plugin cannot be re-enabled from the workspace. The block is by NAME ('anyio'); a metadata-only distribution that
    exposes the same module as 'anyio_local' is not covered by it and touches no workspace file."""
    import importlib.util
    if importlib.util.find_spec("anyio") is None:
        return "anyio not installed"
    di = sp / "anyio_local_plugin-0.dist-info"
    if di.exists():
        return "already enabled"
    used = False
    for path in ws.rglob("*.py"):
        if ".git" in path.parts or not (path.name.startswith("test_") or path.name.endswith("_test.py") or path.name == "conftest.py"):
            continue
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        if "pytest.mark.anyio" in text or "anyio_backend" in text:
            used = True
            break
    if not used:
        return "no anyio-marked tests"
    di.mkdir()
    (di / "METADATA").write_text("Metadata-Version: 2.1\nName: anyio-local-plugin\nVersion: 0\n")
    (di / "entry_points.txt").write_text("[pytest11]\nanyio_local = anyio.pytest_plugin\n")
    return "enabled"


def local_repair(ws, repo=""):
    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
    from pip._vendor.packaging.version import Version
    log, skipped = [], []
    sp = find_site_packages_dir()
    if sp is None:
        return
    index = _wheel_index()
    repo_short = repo.rsplit("/", 1)[-1].lower() if repo else ""
    own = _own_code_from_workspace(sp, ws, repo_short) if repo_short else {}
    extras = []
    locked = _locked_versions(ws) if LOCK_HONOR.get(repo_short) else {}
    honored = {normalize_pkg_name(n) for n in LOCK_HONOR.get(repo_short, [])}
    for name in EXTRAS.get(repo_short, []):
        norm = normalize_pkg_name(name)
        if _installed_version(name) is not None:
            continue
        cands = sorted(index.get(norm, []), reverse=True)
        if norm in honored and norm in locked:  # install the locked version straight away when the shelf has it
            cands = [c for c in cands if str(c[0]) == locked[norm]] or cands
        if not cands:
            skipped.append({"requirement": name, "installed": None, "why": "extra not on the shelf"})
            continue
        res = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-build-isolation", "-q",
                              "--disable-pip-version-check", cands[0][1]], capture_output=True, text=True, check=False)
        extras.append({"package": norm, "installed": os.path.basename(cands[0][1]), "rc": res.returncode,
                       "err": (res.stderr or "")[-200:] if res.returncode else ""})
    reqs = _declared_requirements(ws)
    for name in LOCK_HONOR.get(repo_short, []):
        if normalize_pkg_name(name) in locked:
            reqs.append(f"{name}=={locked[normalize_pkg_name(name)]}")  # applied after the ranges: the lock is the tighter statement
    ws_version = None
    for rule in LOCAL_RULES.get(repo_short, []):
        if rule.get("when"):
            if ws_version is None:
                ws_version = _installed_version(repo_short) or "0"
            try:
                if not Requirement(rule["when"]).specifier.contains(ws_version, prereleases=True):
                    continue
            except InvalidRequirement:
                continue
        reqs.append(rule["require"])
    for raw in reqs:
        try:
            req = Requirement(raw)
        except InvalidRequirement:
            continue
        if req.marker is not None and not req.marker.evaluate():
            continue
        norm = normalize_pkg_name(req.name)
        if norm == normalize_pkg_name(repo_short) or not req.specifier:
            continue
        have = _installed_version(req.name)
        if have is not None and req.specifier.contains(have, prereleases=True):
            continue
        ok = sorted(((v, f) for v, f in index.get(norm, []) if req.specifier.contains(v, prereleases=True)), reverse=True)
        if not ok:
            skipped.append({"requirement": raw, "installed": have, "why": "no wheel in /wheels satisfies it"})
            continue
        if have is not None and Version(have) == ok[0][0]:
            continue
        _swap(sp, norm, ok[0][1], log)
        log[-1].update(requirement=raw, was=have)
    # the organizer's fast path puts every top-level workspace directory on sys.path; drop the ones that shadow libraries
    pth = sp / "workspace_paths.pth"
    dropped = []
    if pth.exists() and PTH_EXCLUDE.get(repo_short):
        keep = []
        for line in pth.read_text().splitlines():
            (dropped if Path(line).name in PTH_EXCLUDE[repo_short] else keep).append(line)
        if dropped:
            pth.write_text("\n".join(keep) + "\n")
    try:
        anyio_state = _enable_anyio_tests(ws, sp)
    except Exception as exc:
        anyio_state = f"failed: {exc}"
    with contextlib.suppress(Exception):
        Path("/tmp/local_repair.json").write_text(_json.dumps({"repo": repo, "own": own, "extras": extras, "anyio": anyio_state, "swapped": log, "skipped": skipped, "pth_dropped": dropped}, indent=1))
    if log or dropped:
        print(f"local repair: swapped {[(e['package'], e['installed']) for e in log]} pth_dropped={dropped}")


def _local_main():
    argv = [a for a in sys.argv[1:]]
    repo, ws, i = "", None, 0
    while i < len(argv):
        if argv[i] in ("--workspace", "-w") and i + 1 < len(argv):
            ws = Path(argv[i + 1])
            i += 1
        elif not argv[i].startswith("-") and not repo:
            repo = argv[i]
        i += 1
    ws = ws or (Path("/workspace") if Path("/workspace").exists() else Path.cwd())
    try:
        local_repair(ws.resolve(), repo)
    except Exception as exc:  # the repair must never break the organizer's setup
        print(f"local repair skipped: {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
    _local_main()
