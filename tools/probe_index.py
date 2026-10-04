"""probe_index -- derive a retrievable probe status table from an append-only journal.

    uv run tools/probe_index.py comps/<slug>/journal.md            # dangling probes + summary
    uv run tools/probe_index.py comps/<slug>/journal.md --all      # every probe, newest event first
    uv run tools/probe_index.py comps/<slug>/journal.md --name foo # one probe's full event history
    uv run tools/probe_index.py comps/<slug>/journal.md --topic 'ensemble|secondary'
                                       # EVERY event type (NOTE/LB/SCORE/...), closures first.
                                       # Run this BEFORE proposing in an area the board has touched.
    uv run tools/probe_index.py comps/<slug>/journal.md --unlogged comps/<slug>/probes/out
                                       # DISK -> LOG: probe families with artifacts but no journal
                                       # mention = a FINISHED run nobody gated.  Run at session start.

WHY THIS EXISTS.  An append-only log is the right SOURCE OF TRUTH and the wrong INDEX: its guarantee
(nothing is ever edited) is exactly what stops it answering "what is the current status of X?", because
that answer is smeared across a dozen entries, some of them retracted.  Measured on this journal at
3,834 lines: 644 PROBE events under 510 distinct names, of which only 89 carried a structured
`PROBE <name> RESULT` line.  The other 421 were prose mentions with no parseable verdict -- so a family
closed a week earlier was genuinely unfindable, and re-proposing it looked like a judgement failure
when it was a retrieval failure.

WHAT IT IS NOT.  This is DERIVED and is not evidence.  It never edits the journal, it cites line
numbers so every row can be read at source, and when it disagrees with the journal the journal wins.
Its job is to tell you WHERE to look and WHICH probes never got a verdict -- not to summarise findings,
because a one-line summary is exactly where the constraining number gets dropped.

THE DANGLING LIST IS THE POINT.  A probe with a LAUNCHED line and no terminal event is either still
running, or finished and written up in prose that no grep will find later.  Those are the rows that
cause re-treading, so they print first and by default.

BLINDSPOT, FOUND BY TESTING IT ON A KNOWN TRAP -- READ THIS BEFORE TRUSTING A `verdict` COLUMN.  This
tool reports what the JOURNAL's own grammar says, so a result overturned only in a DERIVED artifact is
invisible to it.  Worked example: `freeadd` (L2568/L2571) shows verdict PASS, and its own line reads
"THE ADD PASS IS REAL AND IT IS THE BIGGE...", but it is 0.00322 BELOW the champion on the honest
instrument -- a correction that lives in LEVER_LEDGER.md's section-B note and in node_0037's `folds:`
line, with NO journal CORRECT line pointing at L2571.  So the index flags it as a win with no
retraction.  Two consequences: (1) when a result is overturned, append a journal CORRECT line naming
the superseded line number, not only a ledger edit -- the ledger is not greppable by line reference;
(2) treat the `verdict` column as "where to look", never as status, and read the cited line plus the
ledger row before acting.  A one-word verdict is precisely where a constraining number gets dropped.
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

# A journal line: "<ts>  <EVENT> <rest>".  Two spaces after the timestamp is the grammar's separator.
LINE = re.compile(r"^(?P<ts>\S+)\s\s(?P<event>[A-Z]+)\s+(?P<rest>.*)$")

# Terminal / opening markers, COUNTED off the journal rather than assumed.  The first pass of this
# tool guessed a four-word terminal vocabulary and reported 47 dangling probes; the real grammar (from
# `grep -oP '  PROBE \S+ \K[A-Z][A-Z_-]{2,}' | sort | uniq -c`) is LAUNCHED 84, GATED 68, RESULT 60,
# DONE 37, READ 20, WRITTEN 17, SCORED 11, STARTED 7, CLOSED 5 ...  Most of those "dangling" probes had
# concluded under a word the parser did not know -- an index that invents its own grammar manufactures
# exactly the retrieval failure it exists to measure.
TERMINAL = ("RESULT", "GATED", "DONE", "READ", "WRITTEN", "SCORED", "CLOSED", "MEASURED",
            "REDUNDANT", "COMPLETE", "KILLED", "REFUTED", "ABORTED", "VOID", "PASSES")
OPENING = ("LAUNCHED", "LAUNCH", "STARTED", "PARTIAL", "RUNNING", "PRE-REGISTERED", "CHAINED")

# Verdict words worth surfacing when they appear early in the line -- these are the ones that flip a
# decision.  Deliberately NOT a summary: just a flag pointing you at the source line.
VERDICT = re.compile(
    r"\b(PASS|FAIL|REFUTED|DO NOT SCORE|DO NOT SHIP|NO-OP|EXACT|NULL|INVERTS|"
    r"CONTROL EXACT|ASSERT EXACT|SCREEN PASS|VOID)\b")


def parse(path: Path):
    """name -> [(lineno, ts, kind, verdict, snippet)], plus the CORRECT retractions by target line."""
    probes: dict[str, list] = defaultdict(list)
    retracted: set[int] = set()
    for n, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        m = LINE.match(raw)
        if not m:
            continue
        event, rest = m.group("event"), m.group("rest")
        if event == "CORRECT":
            # "... retracts lines 3874." / "retracts line 3874" -- collect every integer cited
            for tgt in re.findall(r"retracts?\s+lines?\s+([\d,\s]+)", rest):
                retracted.update(int(x) for x in re.findall(r"\d+", tgt))
            continue
        if event != "PROBE":
            continue
        parts = rest.split(None, 2)
        if not parts:
            continue
        name = parts[0]
        tail = parts[1] if len(parts) > 1 else ""
        kind = next((k for k in TERMINAL + OPENING if tail.startswith(k)), None)
        v = VERDICT.search(rest[:400])
        probes[name].append((n, m.group("ts"), kind, v.group(1) if v else None, rest[:150]))
    return probes, retracted


def status(events) -> str:
    kinds = [k for _, _, k, _, _ in events]
    if any(k in TERMINAL for k in kinds if k):
        return "concluded"
    if any(k in OPENING for k in kinds if k):
        return "DANGLING"
    return "prose-only"


CLOSURE = re.compile(
    r"\bCLOSED\b|\bRETIRED\b|\bREFUTED\b|\bNEGATIVE\b|cannot be built|\bDEAD END\b|"
    r"\bREOPEN-IF\b|already priced|is not\b.{0,30}\bmissed alternative", re.I)


def topic(path: Path, pat: str, retracted: set[int], limit: int = 25) -> int:
    """Every journal event of ANY type whose text matches `pat`, newest first.

    WHY THIS EXISTS, and why it is a DIFFERENT question from the probe table above.  `parse()` walks
    PROBE events only (`if event != "PROBE": continue`), so a family closed in a NOTE and priced by an
    LB line is invisible to the index built to prevent re-treading.  That blindspot cost a real
    session: the dual-seed detector ensemble was traced in NOTE L2025, measured in LB L2284 (0.927 vs
    0.945) and closed in NOTE L2465 ("a third member cannot be built at all") -- not one of them a
    PROBE event -- and nine days later it was re-derived from source across eight tool calls as if new.
    The closure vocabulary is surfaced FIRST because the line that ends a family is the line that stops
    the rediscovery, and it is usually a NOTE.

    This stays DERIVED and is not evidence: it cites line numbers so every row is read at source.
    """
    rx = re.compile(pat, re.I)
    hits = []
    for n, raw in enumerate(path.read_text(errors="replace").splitlines(), 1):
        m = LINE.match(raw)
        if not m or not rx.search(raw):
            continue
        hits.append((n, m.group("ts"), m.group("event"), m.group("rest")))
    print(f"{path}: {len(hits)} event(s) matching /{pat}/i, newest first"
          + (f" (showing {limit})" if len(hits) > limit else ""))
    closers = [h for h in hits if CLOSURE.search(h[3][:600])]
    if closers:
        print(f"\n  !! {len(closers)} carry CLOSURE vocabulary -- read these at source BEFORE proposing:")
        for n, ts, ev, rest in sorted(closers, reverse=True)[:8]:
            print(f"    L{n:<5} {ev:<9} {rest[:115]}")
    print()
    for n, ts, ev, rest in sorted(hits, reverse=True)[:limit]:
        mark = " [RETRACTED]" if n in retracted else ""
        print(f"  L{n:<5} {ts}  {ev:<9}{mark} {rest[:125]}")
    print(f"\n(read any row in full: sed -n '<N>p' {path} | cut -c1-1500)")
    return 0


def unlogged(path: Path, outdir: Path, limit: int = 40) -> int:
    """Artifacts on disk whose stem appears NOWHERE in the journal -- finished work nobody read.

    WHY THIS EXISTS.  The journal answers "what did I decide?"; it cannot answer "what finished while
    I was looking elsewhere?", because a run that completes writes to DISK and the log only learns of
    it if a session happens to come back.  Measured cost of that gap on this board: a probe wrote
    698 KB of per-edge output plus a metadata table and got no journal line of any kind, and its
    pre-registered verdict sat unread for EIGHT DAYS; separately, two finished runs (`dbox`,
    `divapp_wide`) were found unread 2.5 h after their markers, in a single session, both already
    carrying their answers.  In every case the experiment was DONE and the only thing missing was
    someone looking.  This is the reconciliation in the other direction: disk -> log, not log -> disk.

    A hit is not necessarily a failure -- intermediates, caches and inputs legitimately go unnamed.
    The rows that matter are LOGS, VERDICT FILES and SCORE ARRAYS whose stem the journal never cites.

    A NOTE ON MATCHING, learned by getting it wrong first.  Matching the WHOLE stem reported 891
    files, because `dbox_scores_arm.npz` is not cited verbatim even though the `dbox` probe is
    journaled at length.  The unit that gets journaled is the PROBE FAMILY, not the file, so the key
    is the stem's leading token and files are grouped under it.  A family is "unlogged" only when
    NEITHER its key nor any full stem under it appears anywhere in the journal.
    """
    import datetime as _dt
    txt = path.read_text(errors="replace")
    if not outdir.is_dir():
        print(f"no such output dir: {outdir}")
        return 2
    fam: dict[str, list] = defaultdict(list)
    for f in outdir.rglob("*"):
        if not f.is_file():
            continue
        stem = f.stem.split(".")[0]
        key = re.split(r"[_\-.]", stem)[0]
        if len(key) < 4:
            key = stem
        fam[key].append(f)
    rows = []
    for key, files in fam.items():
        if key in txt or any(f.stem.split(".")[0] in txt for f in files):
            continue
        newest = max(files, key=lambda q: q.stat().st_mtime)
        rows.append((newest.stat().st_mtime, key, len(files),
                     sum(f.stat().st_size for f in files), newest.relative_to(outdir)))
    rows.sort(reverse=True)
    print(f"{outdir}: {len(rows)} probe-family key(s) absent from {path.name}, of {len(fam)} on disk")
    if not rows:
        print("  (clean -- every artifact family is cited somewhere in the journal)")
    for mt, key, n, sz, rel in rows[:limit]:
        when = _dt.datetime.fromtimestamp(mt, _dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
        print(f"  {when}  {key:<22} {n:>3} file(s) {sz/1e6:8.2f} MB   newest: {rel}")
    print("\n(a hit is not automatically a failure -- scratch and intermediates go uncited; the rows\n"
          " that matter carry a .log or a score/verdict file, which mean a FINISHED run nobody gated)")
    return 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    path = Path(argv[0])
    if not path.exists():
        print(f"no such journal: {path}")
        return 2
    show_all = "--all" in argv
    one = argv[argv.index("--name") + 1] if "--name" in argv else None
    probes, retracted = parse(path)

    if "--topic" in argv:
        return topic(path, argv[argv.index("--topic") + 1], retracted)

    if "--unlogged" in argv:
        return unlogged(path, Path(argv[argv.index("--unlogged") + 1]))

    if one:
        ev = probes.get(one)
        if not ev:
            near = [n for n in probes if one.lower() in n.lower()]
            print(f"no probe named {one!r}." + (f" did you mean: {', '.join(sorted(near)[:8])}" if near else ""))
            return 1
        print(f"{one} -- {len(ev)} event(s), status {status(ev)}")
        for n, ts, kind, v, snip in ev:
            mark = "  [RETRACTED]" if n in retracted else ""
            print(f"  L{n:<5} {ts}  {kind or '-':<9} {v or '':<13}{mark}")
            print(f"         {snip}")
        return 0

    groups = defaultdict(list)
    for name, ev in probes.items():
        groups[status(ev)].append((max(e[0] for e in ev), name, ev))

    tot = sum(len(v) for v in groups.values())
    nev = sum(len(e) for e in probes.values())
    print(f"{path}: {nev} PROBE events under {tot} distinct names")
    for k in ("concluded", "DANGLING", "prose-only"):
        print(f"  {k:<11} {len(groups[k]):>4}")
    print(f"  retracted lines cited by CORRECT events: {len(retracted)}")

    want = ("concluded", "DANGLING", "prose-only") if show_all else ("DANGLING", "prose-only")
    for k in want:
        rows = sorted(groups[k], reverse=True)
        if not rows:
            continue
        print(f"\n=== {k} ({len(rows)}) "
              + ("-- launched with no terminal event: still running, or concluded only in prose"
                 if k == "DANGLING" else
                 "-- mentioned with no structured launch or verdict" if k == "prose-only" else "")
              )
        for last, name, ev in rows:
            v = next((x[3] for x in reversed(ev) if x[3]), None)
            flag = " RETRACTED" if any(n in retracted for n, *_ in ev) else ""
            print(f"  L{last:<5} {name:<24} {len(ev):>2} ev  {v or '':<13}{flag}")
    if not show_all:
        print("\n(--all also lists concluded probes; --name <probe> prints one probe's history)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
