"""Run a long job and stop it if its log heartbeat goes stale.

The child runs in its own process group. Any stdout/stderr activity refreshes the
heartbeat; therefore long training loops must print at least once per unit/epoch.

--marker PATH   touched when the job ends (success, failure, or stall-kill alike)
                — the "run ended" signal the autopilot gate and resume logic read.
--on-done CMD   shell command run after the marker (best-effort) — used to nudge
                an idle Codex session awake; Claude Code wakes natively via the
                harness's background-task notification instead.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--idle-seconds", type=float, default=900.0)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--marker", type=Path, default=None)
    parser.add_argument("--on-done", default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    if args.idle_seconds <= 0 or args.poll_seconds <= 0:
        parser.error("timeouts must be positive")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    rc = 0
    with args.log.open("ab", buffering=0) as log:
        started = time.monotonic()
        proc = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        while proc.poll() is None:
            time.sleep(args.poll_seconds)
            try:
                last_output = args.log.stat().st_mtime
            except FileNotFoundError:
                last_output = time.time() - (time.monotonic() - started)
            idle = time.time() - last_output
            if idle < args.idle_seconds:
                continue
            message = (
                f"WATCHDOG_STALL pid={proc.pid} idle_seconds={idle:.1f} "
                f"limit={args.idle_seconds:.1f}; terminating process group\n"
            )
            log.write(message.encode())
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            rc = 124
            break
        else:
            rc = int(proc.returncode or 0)
    if args.marker is not None:  # touched on EVERY exit path — "the run ended"
        args.marker.parent.mkdir(parents=True, exist_ok=True)
        args.marker.touch()
    if args.on_done:  # best-effort nudge; never lets a failure mask the run's rc
        subprocess.run(args.on_done, shell=True, timeout=60, check=False)
    return rc


if __name__ == "__main__":
    sys.exit(main())
