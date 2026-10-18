"""Build kits/swe-agent/sandbox/setup.py = the organizer's data/sandbox/setup.py (verbatim) + kits/swe-agent/sandbox/repair.py.

usage: python -I kits/swe-agent/sandbox/build_hook.py
The harness copies ONE file (setup.py) into each container and runs it, so the local repair has to live in that file.
Point the harness at it with KAGGLE_SANDBOX_DIR=<repo>/kits/swe-agent/sandbox (make setup does this). data/sandbox/setup.py is never modified.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # the repo root
orig = (ROOT / "data" / "sandbox" / "setup.py").read_text()
tail = 'if __name__ == "__main__":\n    main()\n'
assert orig.rstrip("\n").endswith(tail.rstrip("\n")), "organizer setup.py no longer ends with the expected main guard"
body = orig[: orig.rindex(tail)]
repair = (ROOT / "kits/swe-agent/sandbox" / "repair.py").read_text()
sha = hashlib.sha256(orig.encode()).hexdigest()[:16]
out = ROOT / "kits/swe-agent/sandbox" / "setup.py"
out.write_text(body + f"# ^^^ organizer's data/sandbox/setup.py, verbatim up to its main guard (sha256[:16] {sha}) ^^^\n\n" + repair)
print(f"wrote {out} ({len(out.read_text().splitlines())} lines; organizer part sha {sha})")
