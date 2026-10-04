"""Forward liveness from a job's REAL log into the log the watchdog is watching.

THE FAILURE THIS FIXES, which cost three GPU runs before it was diagnosed.
run_with_watchdog.py redirects the child's stdout into --log and kills the process group
when that FILE's mtime goes stale.  Our run.sh scripts pipe the trainer through a grep
filter so the watchdog log stays readable:

    ... | tee -a "$LOG" | tr '\r' '\n' | grep -aE "INIT_FULL|Model parameters|^  Epoch|..."

The unfiltered output lands in "$LOG" (arm_*.log) and only the matched lines reach stdout.
If the trainer's per-epoch line does not match the pattern -- a two-space anchor that does
not fit, a tqdm bar that carries the only progress -- the watchdog log receives NOTHING for
the whole run while training is perfectly healthy, and the watchdog kills it on schedule.
node_0050 died this way three times (twice at 900 s, once at 1800 s) at batch 505/685 with
its arm log growing the entire time.  Raising --idle-seconds cannot fix it; the heartbeat
is not slow, it is absent.

WHAT THIS DOES.  Polls the file that IS growing and appends one line to the watchdog's log
each time it grows.  That refreshes the mtime the watchdog reads.

WHY IT DOES NOT DEFEAT THE WATCHDOG.  The tick is CONDITIONAL on real growth.  A genuinely
hung job stops writing its arm log, no tick is emitted, the watchdog log goes stale on
schedule and the kill still happens.  A heartbeat that ticked unconditionally would turn the
watchdog off, which is the opposite of the fix.

USAGE:
  uv run tools/log_heartbeat.py --tick <watchdog --log path> --watch <arm log> [--watch ...]
        [--interval 120] [--stop-marker /tmp/x.done] [--max-hours 12]
Globs are allowed in --watch and are re-expanded every poll, so a log that does not exist
yet (a second arm) is picked up when it appears.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path


def sizes(patterns):
    out = {}
    for pat in patterns:
        for p in glob.glob(pat):
            try:
                out[p] = os.path.getsize(p)
            except OSError:
                pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tick", required=True, type=Path, help="the watchdog's --log file")
    ap.add_argument("--watch", required=True, action="append",
                    help="file or glob whose GROWTH means the job is alive (repeatable)")
    ap.add_argument("--interval", type=float, default=120.0)
    ap.add_argument("--stop-marker", type=Path, default=None)
    ap.add_argument("--max-hours", type=float, default=12.0)
    a = ap.parse_args()

    prev = sizes(a.watch)
    deadline = time.time() + a.max_hours * 3600
    while time.time() < deadline:
        time.sleep(a.interval)
        if a.stop_marker is not None and a.stop_marker.exists():
            break
        cur = sizes(a.watch)
        grew = [(p, cur[p] - prev.get(p, 0)) for p in cur if cur[p] > prev.get(p, 0)]
        prev = cur
        if not grew:
            continue            # no growth -> no tick -> the watchdog still owns the stall
        ts = time.strftime("%H:%M:%SZ", time.gmtime())
        note = " ".join(f"{Path(p).name}+{d}B" for p, d in sorted(grew))
        try:
            with a.tick.open("ab", buffering=0) as f:
                f.write(f"HEARTBEAT {ts} {note}\n".encode())
        except OSError:
            pass                # the watchdog log may be gone if the job already ended
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
