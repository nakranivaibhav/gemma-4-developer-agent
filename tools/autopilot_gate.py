#!/usr/bin/env python3
"""Stop-hook gate: keep an autonomous session moving; let a blocked one stop.

Registered as a `Stop` lifecycle hook in BOTH harnesses (.claude/settings.json
and .codex/hooks.json). When the agent is about to end its turn and go idle,
the harness runs this script. It reads ONLY the comp's own disk state (never
the transcript) and answers one question: is there work the session should
continue with right now?

  allow stop  -> exit 0, no output           (nothing enters the context)
  keep going  -> print {"decision":"block","reason":"<next step>"}, exit 0
                 (the reason becomes the continuation prompt — one short line)

Stops are ALLOWED whenever any of these hold:
  - no comp dir / no journal.md yet
  - control.md says `halt: true` or `autonomy: interactive`
  - a gated Decision Card is waiting on the human (.waiting-on-human sentinel)
  - every in-flight training is still running (its LAUNCH marker file is
    absent) and nothing else is actionable — the wake comes from the harness
    background-task notification (Claude Code) or the watchdog --on-done nudge
    (Codex), not from refusing to stop
  - circuit breaker: 3 consecutive blocked-stops with zero journal growth
    (something is wedged — a human should look)

Otherwise it blocks with the first actionable item, in priority order:
finished-but-ungated run > registered node needing a build/launch > closed
round needing the next ROUND-OPEN > pre-experiment stage still unticked
(full_auto only). A proposed node with an executable run.sh is already built:
it queues quietly behind an in-flight run, then becomes actionable to launch.

Stdlib only, no uv, fast: a hook runs on every turn end. Self-test:
    python3 tools/autopilot_gate.py --selftest
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_state import STAGES, parse_journal, replay  # noqa: E402

BREAKER_N = 3


def _read_control(comp: Path) -> dict:
    out = {"autonomy": "interactive", "halt": False}
    ctl = comp / "control.md"
    if ctl.exists():
        text = ctl.read_text()
        m = re.search(r"^autonomy:\s*(\S+)", text, re.M)
        if m:
            out["autonomy"] = m.group(1)
        if re.search(r"^halt:\s*true\b", text, re.M):
            out["halt"] = True
    return out


def _find_comp(root: Path) -> Path | None:
    comps = sorted(
        (p for p in (root / "comps").glob("*/") if (p / "journal.md").exists()),
        key=lambda p: (p / "journal.md").stat().st_mtime,
        reverse=True,
    )
    return comps[0] if comps else None


def decide(root: Path) -> dict | None:
    """Return the block payload, or None to allow the stop."""
    comp = _find_comp(root)
    if comp is None:
        return None
    ctl = _read_control(comp)
    if ctl["halt"] or ctl["autonomy"] == "interactive":
        return None
    if (comp / ".waiting-on-human").exists():
        return None

    s = replay(parse_journal(comp / "journal.md"))
    reason = None

    # 1) a launched run whose marker file exists but has no SCORE yet -> gate it
    for nid, n in s["nodes"].items():
        if n["status"] == "running":
            marker = n.get("marker")
            if marker and Path(marker).exists():
                reason = f"training for {nid} has finished (marker {marker}) — gate its outputs, append the SCORE line, and continue the round"
                break

    # 2) a registered node needing a build or a launch. Journal status remains
    # "proposed" while a long BUILD job is ready_to_run, so honor the artifact
    # lifecycle: run.sh = built/preflighted and awaiting orchestrator launch.
    # GPU runs are serialized; a ready node is not actionable while another run
    # is in flight.
    if reason is None:
        in_flight = any(n["status"] == "running" for n in s["nodes"].values())
        for nid, n in s["nodes"].items():
            if n["status"] == "proposed":
                node_dir = comp / "nodes" / nid
                run_sh = node_dir / "run.sh"
                src = node_dir / "src"
                # A developer subagent currently building this node is invisible
                # to the journal (no LAUNCH line until it returns a run.sh), so
                # without this the gate demands a build that is already under
                # way and a second agent races the first over the same files.
                # The orchestrator touches .building-<nid> on dispatch and
                # removes it when the developer reports.
                if (comp / f".building-{nid}").exists():
                    continue
                if run_sh.is_file():
                    if not in_flight:
                        reason = (
                            f"{nid} is built and ready_to_run — launch its run.sh "
                            "setsid-detached through the watchdog, append LAUNCH, and continue the round"
                        )
                        break
                    continue
                # A build is not actionable while another run holds the GPU: the
                # developer's mandatory timing probe would contend with the live
                # run and mis-measure itself, and distort the live run's timings.
                if in_flight:
                    continue
                # REGISTER creates an empty src/ — that is "not built yet", not a
                # partial build. Only real files mean a build was started.
                built_files = [
                    p for p in src.rglob("*")
                    if p.is_file() and "__pycache__" not in p.parts
                ] if src.is_dir() else []
                if built_files:
                    reason = (
                        f"{nid} has a partial build but no run.sh or SCORE — hand it "
                        "back to kaggle-developer to finish the build and continue the round"
                    )
                else:
                    reason = (
                        f"{nid} is registered but not built — hand it to "
                        "kaggle-developer and continue the round"
                    )
                break

    # 3) nothing running/pending and the last round is closed -> open the next
    if reason is None:
        if not in_flight and s["nodes"]:
            last = s["rounds"][-1] if s["rounds"] else None
            if last is None or last["close"] is not None:
                reason = "no round is open and nothing is running — open the next experiment round (/kaggle-experiment)"

    # 4) pre-experiment stages (full_auto only — earlier modes gate via the card sentinel)
    if reason is None and not s["nodes"] and ctl["autonomy"] == "full_auto":
        for st in STAGES[:-1]:
            if not s["stages"].get(st):
                reason = f"stage '{st}' is not done — run its skill and continue the pipeline"
                break

    if reason is None:
        return None

    # circuit breaker: N consecutive blocks with zero journal growth
    size = (comp / "journal.md").stat().st_size
    blog = comp / ".autopilot_blocks"
    hist = []
    if blog.exists():
        hist = [int(x) for x in blog.read_text().split() if x.isdigit()]
    if hist and hist[-1] != size:
        hist = []  # progress since the last block — fresh streak
    if len(hist) >= BREAKER_N - 1:
        blog.unlink(missing_ok=True)
        return {  # allow the stop, but tell the human why on the way out
            "systemMessage": (
                f"autopilot breaker: {BREAKER_N} continuations with no journal progress — "
                f"stopping so a human can look ({comp.name})."
            )
        }
    blog.write_text(" ".join(str(x) for x in hist + [size]) + "\n")
    return {"decision": "block", "reason": reason}


def _selftest() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        comp = root / "comps" / "t"
        comp.mkdir(parents=True)
        j = comp / "journal.md"
        ctl = comp / "control.md"

        # no journal -> allow
        assert decide(root) is None
        j.write_text("2026-01-01T10:00Z  STAGE understand done\n")
        # interactive -> allow
        ctl.write_text("autonomy: interactive\nhalt: false\n")
        assert decide(root) is None
        # full_auto, stages unticked -> block on next stage
        ctl.write_text("autonomy: full_auto\nhalt: false\n")
        r = decide(root)
        assert r and "toolkit" in r["reason"], r
        # halt kills it
        ctl.write_text("autonomy: full_auto\nhalt: true\n")
        assert decide(root) is None
        ctl.write_text("autonomy: full_auto\nhalt: false\n")
        # waiting-on-human sentinel -> allow
        (comp / ".waiting-on-human").touch()
        assert decide(root) is None
        (comp / ".waiting-on-human").unlink()
        # registered node -> block to build it
        j.open("a").write(
            "2026-01-01T11:00Z  REGISTER node_0001 op=draft parents=[root] family=gbdt well=exploit uses_data=[] round=round_0001 — lgbm\n")
        (comp / ".autopilot_blocks").unlink(missing_ok=True)
        r = decide(root)
        assert r and r.get("decision") == "block" and "node_0001" in r["reason"], r
        # a developer subagent already building it -> not actionable
        (comp / ".building-node_0001").touch()
        (comp / ".autopilot_blocks").unlink(missing_ok=True)
        _r = decide(root)
        # the build demand is suppressed; falling through to another reason is
        # fine, demanding node_0001 again is not
        assert _r is None or "node_0001" not in _r["reason"], _r
        (comp / ".building-node_0001").unlink()
        # REGISTER creates an empty src/; that is "not built", not a partial build
        (comp / "nodes" / "node_0001" / "src").mkdir(parents=True)
        (comp / ".autopilot_blocks").unlink(missing_ok=True)
        r = decide(root)
        assert r and "registered but not built" in r["reason"], r
        # a stray __pycache__ must not read as a started build either
        (comp / "nodes" / "node_0001" / "src" / "__pycache__").mkdir()
        (comp / "nodes" / "node_0001" / "src" / "__pycache__" / "x.pyc").write_text("")
        (comp / ".autopilot_blocks").unlink(missing_ok=True)
        r = decide(root)
        assert r and "registered but not built" in r["reason"], r
        # real source in src/ -> a genuine partial build
        (comp / "nodes" / "node_0001" / "src" / "solution.py").write_text("x = 1\n")
        (comp / ".autopilot_blocks").unlink(missing_ok=True)
        r = decide(root)
        assert r and "partial build" in r["reason"], r
        # a long BUILD job with run.sh is ready to launch
        node1 = comp / "nodes" / "node_0001"
        node1.mkdir(parents=True, exist_ok=True)
        (node1 / "run.sh").write_text("#!/bin/sh\n")
        (comp / ".autopilot_blocks").unlink(missing_ok=True)
        r = decide(root)
        assert r and "built and ready_to_run" in r["reason"], r
        # launched, marker absent -> in flight -> allow (wake comes from elsewhere)
        marker = root / "run.done"
        j.open("a").write(f"2026-01-01T11:05Z  LAUNCH node_0001 marker={marker} — 1h\n")
        # A second ready node queues behind the live run instead of causing a
        # false "not built" continuation.
        j.open("a").write(
            "2026-01-01T11:06Z  REGISTER node_0002 op=draft parents=[root] family=nn well=exploit uses_data=[] round=round_0001 — queued\n")
        node2 = comp / "nodes" / "node_0002"
        node2.mkdir(parents=True)
        (node2 / "run.sh").write_text("#!/bin/sh\n")
        (comp / ".autopilot_blocks").unlink(missing_ok=True)
        assert decide(root) is None
        # nor may an UNBUILT node be demanded while a run holds the GPU: the
        # developer's timing probe would contend with the live run.
        j.open("a").write(
            "2026-01-01T11:07Z  REGISTER node_0003 op=draft parents=[root] family=nn well=exploit uses_data=[] round=round_0001 — unbuilt\n")
        (comp / ".autopilot_blocks").unlink(missing_ok=True)
        assert decide(root) is None
        # marker appears -> block to gate it
        marker.touch()
        r = decide(root)
        assert r and "gate its outputs" in r["reason"], r
        # scored + round closed -> block to open next round
        j.open("a").write(
            "2026-01-01T12:00Z  SCORE node_0001 status=valid cv=1.0 sem=0.1 — ok\n"
            "2026-01-01T12:00Z  SCORE node_0002 status=dead cv=null sem=null — test cleanup\n"
            "2026-01-01T12:00Z  SCORE node_0003 status=dead cv=null sem=null — test cleanup\n"
            "2026-01-01T12:01Z  ROUND-OPEN round_0001 nodes=[node_0001] — r1\n"
            "2026-01-01T12:02Z  ROUND-CLOSE round_0001 — done\n")
        (comp / ".autopilot_blocks").unlink(missing_ok=True)
        r = decide(root)  # blocked stop 1 of the streak
        assert r and "next experiment round" in r["reason"], r
        # breaker: the 3rd consecutive no-progress block becomes an allow + message
        r2 = decide(root)
        assert r2.get("decision") == "block", r2
        r3 = decide(root)
        assert "decision" not in r3 and "breaker" in r3.get("systemMessage", ""), r3
        # journal grew -> streak resets, blocks again
        j.open("a").write("2026-01-01T12:10Z  NOTE — progress\n")
        r = decide(root)
        assert r and r.get("decision") == "block", r
    print("autopilot_gate selftest OK")
    return 0


def main() -> int:
    if "--selftest" in sys.argv[1:]:
        return _selftest()
    try:
        json.load(sys.stdin)  # hook input — read + discard (decisions come from disk)
    except Exception:
        pass
    try:
        result = decide(Path.cwd())
    except Exception:
        return 0  # a broken gate must never wedge the session — fail open (allow stop)
    if result:
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
