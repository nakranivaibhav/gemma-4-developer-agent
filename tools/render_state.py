#!/usr/bin/env python3
"""Render comps/<slug>/state.md from the append-only journal — the ONE derived view.

`journal.md` is the single written source of truth (event-sourced, append-only).
This tool replays it and prints the entire current picture — header, setup/stage
checklists, Mermaid DAG, node table, data lineage, submissions ledger — into
`state.md`, which is a GENERATED file and must never be edited by hand. If
state.md looks wrong, re-run this; if it is still wrong, fix ONE journal line
(via an appended CORRECT line, never an edit).

Journal line grammar (this docstring + the journal.md contract are its home):

    <ts>  <EVENT> [<positional> ...] [key=value ...] [— <free prose>]

  ts       = date -u +%Y-%m-%dT%H:%MZ   (UTC, minute precision)
  EVENT    = one of the UPPERCASE events below; positionals never contain '='
  values   = bare tokens (no spaces); null; numbers; [a,b] lists (no spaces)
  prose    = anything after the first " — " (em dash); never parsed, always kept

Events (writer in parentheses — appends happen only in the main session or the
proposer's sequential REGISTER job, never in parallel workers):

  SETUP <item> done                                  (start)   item: auth|rules|verified|data|spec
  STAGE <name> done                                  (stage skills)  name: understand|toolkit|eda|validation|baseline
  FEATURESET fs_<n> class=stateless|fit_in_fold from=<base|fs_*> producer=node_NNNN — <what>   (proposer REGISTER)
  REGISTER node_NNNN op=<op> parents=[..] family=<f> well=<w> uses_data=[..] round=round_NNNN — <desc>  (proposer REGISTER)
  LAUNCH node_NNNN marker=/tmp/<slug>_node_NNNN.done — <runtime projection>     (orchestrator)
  SCORE node_NNNN status=valid|buggy|dead cv=<f|null> sem=<f|null> holdout=<f|null> folds=[..] — <note>  (orchestrator, from the RESULT line)
  PROMOTE node_NNNN over=<node|none> pboot=<f|null> — <why>                     (orchestrator)
  SUBMIT node_NNNN cv=<f> lb=<f|pending> — <note; PROBE for human-directed probes>  (submit skill)
  LB node_NNNN lb=<f> — <async public-score backfill>                           (submit skill)
  ROUND-OPEN round_NNNN nodes=[..] — <plan>                                     (orchestrator)
  ROUND-CLOSE round_NNNN — <summary>                                            (orchestrator)
  OUTSIDE source=<url-or-ref> — <the concrete lever + numbers claimed>          (any main-session actor)
  PROBE — <one-off diagnostic + numbers>                                        (orchestrator)
  NOTE — <narrative, scoped closures: tried X, measured Y, reopen-if Z>         (any main-session actor)
  CORRECT node_NNNN key=value ... — <what was wrong>   (fixes a node field, last-wins)

Replay semantics: REGISTER=proposed, LAUNCH=running, SCORE=its status,
PROMOTE=champion (the previous champion demotes to valid) — later lines win.
Budget = SUBMIT lines whose UTC date == today, vs spec.md's daily_submission_limit.

Usage:
    uv run tools/render_state.py comps/<slug>          # writes comps/<slug>/state.md
    uv run tools/render_state.py --selftest
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timezone
from pathlib import Path

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z)\s+([A-Z][A-Z-]*)\s*(.*)$")
NODE_STATUSES = ("proposed", "running", "buggy", "dead", "valid", "champion")
STAGES = ("understand", "toolkit", "eda", "validation", "baseline", "experiment")
SETUP_ITEMS = ("auth", "rules", "verified", "data", "spec")


def _parse_value(v: str):
    if v == "null":
        return None
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [x for x in (t.strip() for t in inner.split(",")) if x] if inner else []
    try:
        return float(v) if ("." in v or "e" in v.lower()) else int(v)
    except ValueError:
        return v


KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_line(line: str) -> dict | None:
    """One journal line -> {ts, event, args(list), kv(dict), prose} or None."""
    m = TS_RE.match(line.strip())
    if not m:
        return None
    ts, event, rest = m.groups()
    prose = ""
    for dash in (" — ", " -- "):
        if dash in rest:
            rest, prose = rest.split(dash, 1)
            break
    args, kv = [], {}
    for tok in rest.split():
        k, eq, v = tok.partition("=")
        # Only an identifier-shaped key is a field. Prose comparisons ("<= 12 hours",
        # "cv>=0.8") otherwise mint junk keys, and a junk key on a CORRECT line flips it
        # from a harmless prose correction into a "sets unknown state" render error.
        if eq and KEY_RE.fullmatch(k):
            kv[k] = _parse_value(v)
        else:
            args.append(tok)
    return {"ts": ts, "event": event, "args": args, "kv": kv, "prose": prose.strip()}


def parse_journal(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        ev = parse_line(line)
        if ev:
            ev["lineno"] = i
            out.append(ev)
    return out


def replay(events: list[dict]) -> dict:
    """Replay events into the current state dict."""
    s = {
        "setup": {}, "stages": {}, "nodes": {}, "featuresets": {},
        "champion": None, "rounds": [], "submits": [], "outside": [], "errors": [],
        "retracted": set(),
    }
    for e in events:
        ev, args, kv = e["event"], e["args"], e["kv"]
        node = args[0] if args else None
        try:
            if ev == "SETUP" and len(args) >= 2:
                s["setup"][args[0]] = args[1] == "done"
            elif ev == "STAGE" and len(args) >= 2:
                s["stages"][args[0]] = args[1] == "done"
            elif ev == "FEATURESET":
                s["featuresets"][node] = {**kv, "what": e["prose"], "consumers": []}
            elif ev == "REGISTER":
                s["nodes"][node] = {
                    "id": node, "op": kv.get("op"), "parents": kv.get("parents", []),
                    "family": kv.get("family"), "well": kv.get("well"),
                    "uses_data": kv.get("uses_data", []), "round": kv.get("round"),
                    "desc": e["prose"], "status": "proposed", "cv": None, "sem": None,
                    "holdout": None, "lb": None, "ts": e["ts"],
                }
                for fs in kv.get("uses_data", []) or []:
                    if fs in s["featuresets"]:
                        s["featuresets"][fs]["consumers"].append(node)
            elif ev == "LAUNCH" and node in s["nodes"]:
                s["nodes"][node]["status"] = "running"
                s["nodes"][node]["marker"] = kv.get("marker")
            elif ev == "SCORE" and node in s["nodes"]:
                n = s["nodes"][node]
                n["status"] = kv.get("status", "valid")
                for k in ("cv", "sem", "holdout"):
                    if k in kv:
                        n[k] = kv[k]
            elif ev == "PROMOTE" and node in s["nodes"]:
                if s["champion"] and s["champion"] != node and s["champion"] in s["nodes"]:
                    s["nodes"][s["champion"]]["status"] = "valid"
                s["nodes"][node]["status"] = "champion"
                s["champion"] = node
            elif ev == "SUBMIT":
                s["submits"].append({"ts": e["ts"], "node": node, "cv": kv.get("cv"),
                                     "lb": kv.get("lb"), "note": e["prose"]})
                if node in s["nodes"] and kv.get("lb") not in (None, "pending"):
                    s["nodes"][node]["lb"] = kv.get("lb")
            elif ev == "LB":
                for row in reversed(s["submits"]):
                    if row["node"] == node:
                        row["lb"] = kv.get("lb")
                        break
                if node in s["nodes"]:
                    s["nodes"][node]["lb"] = kv.get("lb")
            elif ev == "ROUND-OPEN":
                s["rounds"].append({"id": node, "nodes": kv.get("nodes", []), "open": e["ts"], "close": None})
            elif ev == "ROUND-CLOSE":
                for r in reversed(s["rounds"]):
                    if r["id"] == node:
                        r["close"] = e["ts"]
                        break
            elif ev == "OUTSIDE":
                s["outside"].append({"ts": e["ts"], "source": kv.get("source"), "prose": e["prose"]})
            elif ev == "CORRECT" and node in s["nodes"]:
                s["nodes"][node].update(kv)
                if kv.get("status") == "champion":
                    s["champion"] = node
            elif ev in ("PROBE", "NOTE"):
                pass
            elif ev == "CORRECT" and not kv:
                # A CORRECT with NO key=value pairs is a prose correction to a NOTE-level
                # topic ("journal-timestamps", "seqnet/calib"), not an edit to a node field
                # — the journal is append-only, so correcting a NOTE has nowhere else to
                # live. It carries no state, so there is nothing to replay and nothing to
                # validate a target against. A CORRECT that DOES carry key=value pairs is
                # still an error when its node is unknown: that one is trying to set state.
                #
                # RETRACTION. Such a line may also carry "retracts line N" (or "retracts
                # lines N, M") in its prose, which drops the render error that line N
                # raised. This exists because the journal is append-only: a verb typo —
                # LAUNCH or SCORE aimed at a probe name instead of a node_NNNN directory —
                # can never be edited out, so without a retraction its error is permanent.
                # Permanent errors are the real damage: once the error list carries known
                # noise, a NEW error (two champions, a ghost node) hides inside it and the
                # channel stops being read. A retraction is deliberately narrow — it only
                # silences the report, it replays no state, and a line that actually SET
                # state (a CORRECT with kv) cannot be retracted this way.
                for grp in re.findall(r"retracts? lines?\s+([\d,\s]+)", e["prose"] or ""):
                    for num in re.findall(r"\d+", grp):
                        s["retracted"].add(int(num))
            elif ev in ("SETUP", "STAGE", "FEATURESET", "REGISTER", "LAUNCH", "SCORE",
                        "PROMOTE", "LB", "CORRECT"):
                s["errors"].append(f"line {e['lineno']}: {ev} references unknown target {node!r}")
        except Exception as ex:  # a malformed line must fail loudly, not silently
            s["errors"].append(f"line {e['lineno']}: {ev}: {ex}")
    # invariant: exactly one champion
    if s["retracted"]:
        s["errors"] = [x for x in s["errors"]
                       if not (m := re.match(r"line (\d+):", x)) or int(m.group(1)) not in s["retracted"]]
    champs = [n for n, d in s["nodes"].items() if d["status"] == "champion"]
    if len(champs) > 1:
        s["errors"].append(f"INVARIANT: {len(champs)} champions: {champs}")
    return s


def _spec_fields(comp: Path) -> dict:
    """Pull the few spec.md machine-block fields the header needs."""
    out = {}
    spec = comp / "spec.md"
    if spec.exists():
        text = spec.read_text()
        for key in ("metric", "metric_direction", "daily_submission_limit", "deadline", "slug"):
            m = re.search(rf"^{key}:\s*(.+)$", text, re.M)
            if m:
                out[key] = m.group(1).strip()
    return out


def _fmt(v, nd=6):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return str(v)


def render(comp: Path, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    today = now.strftime("%Y-%m-%d")
    events = parse_journal(comp / "journal.md")
    s = replay(events)
    spec = _spec_fields(comp)
    slug = spec.get("slug", comp.name)

    used = sum(1 for r in s["submits"] if r["ts"].startswith(today))
    lim = spec.get("daily_submission_limit", "?")
    deadline = spec.get("deadline")
    days_left = "?"
    if deadline:
        try:
            days_left = (datetime.strptime(deadline, "%Y-%m-%d").replace(tzinfo=timezone.utc) - now).days
        except ValueError:
            pass
    champ = s["champion"]
    cn = s["nodes"].get(champ, {}) if champ else {}
    champ_txt = (f"{champ} (cv {_fmt(cn.get('cv'))} · holdout {_fmt(cn.get('holdout'))} · "
                 f"lb {_fmt(cn.get('lb'))})") if champ else "none"

    L = []
    L.append("<!-- GENERATED by tools/render_state.py from journal.md — NEVER edit by hand; re-run the tool instead. -->")
    L.append(f"# state — {slug}")
    L.append(f"metric: {spec.get('metric', '?')} ({spec.get('metric_direction', '?')}) · champion: {champ_txt}")
    L.append(f"today (UTC): {today}   submissions: {used}/{lim} (resets 00:00 UTC)   deadline: {deadline or 'n/a'} ({days_left} left)")
    if s["errors"]:
        L.append("")
        L.append("## RENDER ERRORS — fix the journal (append a CORRECT line)")
        L.extend(f"- {e}" for e in s["errors"])

    L.append("")
    L.append("## setup")
    for item in SETUP_ITEMS:
        L.append(f"- [{'x' if s['setup'].get(item) else ' '}] {item}")
    L.append("")
    L.append("## stages")
    for st in STAGES:
        if st == "experiment":
            mark = "~" if s["nodes"] else " "
            L.append(f"- [{mark}] experiment (terminal loop — {len(s['nodes'])} nodes, {len(s['rounds'])} rounds)")
        else:
            L.append(f"- [{'x' if s['stages'].get(st) else ' '}] {st}")

    L.append("")
    L.append("## graph")
    L.append("```mermaid")
    L.append("graph LR")
    for nid, n in s["nodes"].items():
        label = f"{nid} · {n['desc'] or n['op']} · {_fmt(n['cv'], 4) if n['cv'] is not None else n['status']}"
        L.append(f'    {nid}["{label}"]' + (":::champ" if n["status"] == "champion" else ""))
        for p in n["parents"] or ["root"]:
            L.append(f"    {p} --> {nid}")
    L.append("    classDef champ fill:#2b6,stroke:#333,stroke-width:2px")
    L.append("```")

    L.append("")
    L.append("## nodes")
    L.append("| node | op | family | parents | cv ± sem | holdout | lb | status | what | detail |")
    L.append("|------|----|--------|---------|----------|---------|----|--------|------|--------|")
    for nid, n in s["nodes"].items():
        cv = f"{_fmt(n['cv'])} ± {_fmt(n['sem'])}" if n["cv"] is not None else "—"
        L.append(f"| {nid} | {n['op']} | {n['family']} | {','.join(n['parents'] or ['root'])} "
                 f"| {cv} | {_fmt(n['holdout'])} | {_fmt(n['lb'])} | {n['status']} | {n['desc']} "
                 f"| nodes/{nid}/node.md |")

    if s["featuresets"]:
        L.append("")
        L.append("## data lineage")
        L.append("| id | class | from | producer | consumers | what (recipe: producer's node.md) |")
        L.append("|----|-------|------|----------|-----------|-----------------------------------|")
        for fid, f in s["featuresets"].items():
            L.append(f"| {fid} | {f.get('class')} | {f.get('from')} | {f.get('producer')} "
                     f"| {','.join(f['consumers']) or '—'} | {f.get('what', '')} |")

    L.append("")
    L.append(f"## submissions ({len(s['submits'])} total)")
    L.append("| ts (UTC) | node | cv | lb | note |")
    L.append("|----------|------|----|----|------|")
    for r in s["submits"][-10:]:
        L.append(f"| {r['ts']} | {r['node']} | {_fmt(r['cv'])} | {_fmt(r['lb'])} | {r['note']} |")

    if s["outside"]:
        L.append("")
        L.append(f"## outside intel ({len(s['outside'])} finds)")
        for o in s["outside"][-8:]:
            L.append(f"- {o['ts']} · {o['source']} — {o['prose']}")
    L.append("")
    return "\n".join(L)


def _selftest() -> int:
    import tempfile
    j = """
2026-01-01T10:00Z  SETUP auth done
2026-01-01T10:01Z  STAGE understand done
2026-01-01T10:02Z  STAGE baseline done
2026-01-01T10:03Z  FEATURESET fs_lag class=fit_in_fold from=base producer=node_0001 — lag features
2026-01-01T10:04Z  REGISTER node_0000 op=draft parents=[root] family=baseline well=exploit uses_data=[] round=round_0001 — mean baseline
2026-01-01T10:05Z  REGISTER node_0001 op=draft parents=[root] family=gbdt well=exploit uses_data=[fs_lag] round=round_0001 — lgbm on lags
2026-01-01T10:06Z  ROUND-OPEN round_0001 nodes=[node_0000,node_0001] — first round
2026-01-01T10:07Z  LAUNCH node_0001 marker=/tmp/x_node_0001.done — ~40m projected
2026-01-01T11:00Z  SCORE node_0000 status=valid cv=16.28 sem=0.4 holdout=15.9 — dumb baseline
2026-01-01T11:01Z  PROMOTE node_0000 over=none pboot=null — first champion
2026-01-01T12:00Z  SCORE node_0001 status=valid cv=15.10 sem=0.35 holdout=14.8 — beats baseline
2026-01-01T12:01Z  PROMOTE node_0001 over=node_0000 pboot=0.97 — clean win
2026-01-01T12:05Z  SUBMIT node_0001 cv=15.10 lb=pending — first real submit
2026-01-01T12:30Z  LB node_0001 lb=14.9 — scored
2026-01-01T12:31Z  ROUND-CLOSE round_0001 — good round
2026-01-01T12:32Z  NOTE — tried deeper trees, measured no gain, reopen-if new features land
"""
    with tempfile.TemporaryDirectory() as d:
        comp = Path(d)
        (comp / "journal.md").write_text(j)
        (comp / "spec.md").write_text(
            "```yaml\nslug: testcomp\nmetric: RMSE\nmetric_direction: minimize\n"
            "daily_submission_limit: 5\ndeadline: 2026-02-01\n```\n")
        now = datetime(2026, 1, 1, 13, 0, tzinfo=timezone.utc)
        out = render(comp, now=now)
        s = replay(parse_journal(comp / "journal.md"))
        assert s["champion"] == "node_0001", s["champion"]
        assert s["nodes"]["node_0000"]["status"] == "valid"  # demoted
        assert s["nodes"]["node_0001"]["status"] == "champion"
        assert s["nodes"]["node_0001"]["lb"] == 14.9
        assert s["featuresets"]["fs_lag"]["consumers"] == ["node_0001"]
        assert s["submits"][0]["lb"] == 14.9  # LB backfilled the row
        assert not s["errors"], s["errors"]
        assert out.count(":::champ") == 1
        assert "root --> node_0001" in out
        assert "submissions: 1/5" in out
        assert "- [x] understand" in out and "- [ ] eda" in out
        # a CORRECT line re-points a field
        (comp / "journal.md").open("a").write("2026-01-01T13:05Z  CORRECT node_0001 cv=15.09 — typo\n")
        s2 = replay(parse_journal(comp / "journal.md"))
        assert s2["nodes"]["node_0001"]["cv"] == 15.09
        # unknown target -> loud error
        (comp / "journal.md").open("a").write("2026-01-01T13:06Z  SCORE node_9999 status=valid cv=1 — ghost\n")
        s3 = replay(parse_journal(comp / "journal.md"))
        assert s3["errors"]
        ghost_line = int(s3["errors"][0].split()[1].rstrip(":"))
        # a no-kv CORRECT naming that line retracts its error, and ONLY that one
        (comp / "journal.md").open("a").write(
            "2026-01-01T13:07Z  SCORE node_8888 status=valid cv=1 — second ghost\n")
        (comp / "journal.md").open("a").write(
            f"2026-01-01T13:08Z  CORRECT verbs — retracts line {ghost_line}, a verb typo\n")
        s4 = replay(parse_journal(comp / "journal.md"))
        assert len(s4["errors"]) == 1, s4["errors"]
        assert "node_8888" in s4["errors"][0], s4["errors"]
    print("render_state selftest OK")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if "--selftest" in argv:
        return _selftest()
    if not argv:
        sys.exit("usage: render_state.py comps/<slug> | --selftest")
    comp = Path(argv[0])
    if not (comp / "journal.md").exists():
        sys.exit(f"no journal.md under {comp}")
    out = render(comp)
    (comp / "state.md").write_text(out)
    s = replay(parse_journal(comp / "journal.md"))
    err = f" · {len(s['errors'])} RENDER ERRORS" if s["errors"] else ""
    print(f"state.md rendered: {len(s['nodes'])} nodes · champion {s['champion'] or 'none'}{err}")
    return 2 if s["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
