"""Download the extra test wheels the local sandbox image puts on its shelf (/wheels_extra), by exact file name.

usage: python3 kits/swe-agent/sandbox/fetch_wheels.py [dest_dir]     (default: kits/swe-agent/sandbox/wheels_extra)

The list is wheels_extra.txt next to this file. Each file is looked up on PyPI (JSON API), downloaded, and checked
against PyPI's sha256. Files already present with the right hash are skipped. Standard library only.
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pypi_file(fname):
    name, version = fname.split("-")[:2]
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60) as r:
        meta = json.load(r)
    for u in meta["urls"]:
        if u["filename"] == fname:
            return u["url"], u["digests"]["sha256"]
    raise SystemExit(f"{fname}: not on PyPI under {name} {version}")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "wheels_extra"
    dest.mkdir(parents=True, exist_ok=True)
    names = (HERE / "wheels_extra.txt").read_text().split()
    got = 0
    for fname in names:
        url, want = pypi_file(fname)
        out = dest / fname
        if out.exists() and sha256(out) == want:
            continue
        tmp = out.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if sha256(tmp) != want:
            tmp.unlink()
            raise SystemExit(f"{fname}: sha256 mismatch")
        tmp.rename(out)
        got += 1
    print(f"wheels_extra: {len(names)} files in {dest} ({got} downloaded)")


if __name__ == "__main__":
    main()
