import gzip
import json
from types import SimpleNamespace

import pytest

from training.rl import batch as B, rollouts as R
from training.rl.tests.conftest import CH, NEWLINES, NL1, NL2, call, ledger_row

SYS = [1, 2, 3]


def algo(**kw):
    d = dict(adv_estimator="rloo", norm_adv_by_std=False, min_group_size=2, count_zero_adv_tries=True)
    d.update(kw)
    return SimpleNamespace(**d)


def write_capture(path, rows):
    with gzip.open(path, "wt") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def cap_row(rid, c, utc, status=200, tito=True):
    row = {"utc": utc, "status": status, "request": {}, "response": {}, "rollout_id": rid}
    if tito:
        row["tito"] = {k: c[k] for k in ("prompt_ids", "completion_ids", "logprobs", "finish_reason")}
    return row


def test_read_capture_groups_by_rollout_and_orders_by_time(tmp_path):
    p = tmp_path / "c.jsonl.gz"
    a0, a1 = call(SYS, [10, 50]), call(SYS + [10, 50, 60], [20, 106])
    write_capture(p, [cap_row("r/a", a1, 2.0), cap_row("r/b", call(SYS, [11, 50]), 1.5), cap_row("r/a", a0, 1.0),
                      {"utc": 3.0, "status": 200, "request": {}, "response": {}}])
    cap, stats = R.read_capture(p)
    assert [c["prompt_ids"] for c in cap["r/a"]] == [a0["prompt_ids"], a1["prompt_ids"]]
    assert stats == {"calls": 4, "untagged": 1}


def test_failed_call_has_error_record(tmp_path):
    p = tmp_path / "c.jsonl.gz"
    write_capture(p, [cap_row("r/a", call(SYS, [10]), 1.0, status=500, tito=False)])
    cap, _ = R.read_capture(p)
    assert "error" in cap["r/a"][0]


def test_build_classifies_and_excludes():
    cap = {"p1/a": [call(SYS, [10, 50])], "p2/a": [call(SYS, [11, 50])], "p4/a": [{"utc": 0, "error": "x"}]}
    rows = [ledger_row("p1", "a", True), ledger_row("p2", "a", False),
            ledger_row("p3", "a", harness_error="litellm.InternalServerError: x"),   # infra
            ledger_row("p4", "a", False), ledger_row("p5", "a", False)]               # tito error, no calls
    ro = {r.rollout_id: r for r in R.build(rows, cap, "aaaa")}
    assert ro["p1/a"].reward == 1.0 and ro["p2/a"].reward == 0.0
    assert ro["p3/a"].exclude == "infra:server_error" and ro["p3/a"].reward is None
    assert ro["p4/a"].exclude == "tito_missing" and ro["p5/a"].exclude == "no_calls"


def test_restarted_try_is_excluded():
    cap = {"p1/a": [call(SYS, [10, 50]), call(SYS + [10, 50, 60], [11, 50]), call(SYS, [12, 50])]}
    (r,) = R.build([ledger_row("p1", "a", True)], cap, "aaaa")
    assert r.exclude == "restarted"


def test_on_policy_guard():
    with pytest.raises(ValueError, match="on-policy guard"):
        R.build([ledger_row("p1", "a", True, sha="bbbb")], {"p1/a": [call(SYS, [10])]}, "aaaa")


def test_duplicate_ledger_rows_and_orphan_calls_raise():
    c = {"p1/a": [call(SYS, [10])]}
    with pytest.raises(ValueError, match="two ledger rows"):
        R.build([ledger_row("p1", "a"), ledger_row("p1", "a")], c, "aaaa")
    with pytest.raises(ValueError, match="no ledger row"):
        R.build([], c, "aaaa")
    with pytest.raises(ValueError, match="rollout id"):
        R.build([], {}, "aaaa", capture_stats={"untagged": 3})


def group(task, rewards, n_calls=1):
    """Ledger rows + capture for one task; each try makes n_calls append-only calls with one newline splice."""
    rows, cap = [], {}
    for i, r in enumerate(rewards):
        run = f"p{i}"
        hist, calls = list(SYS), []
        for k in range(n_calls):
            g = [10 + i, NL1, CH, 30, 50]
            calls.append(call(hist, g))
            hist = hist + [10 + i, NL2, CH, 30, 50] + [60 + k]     # template re-render: NL2
        rows.append(ledger_row(run, task, bool(r)))
        cap[f"{run}/{task}"] = calls
    return rows, cap


def make(groups, **kw):
    rows, cap = [], {}
    for task, rw, n in groups:
        r, c = group(task, rw, n)
        rows += r
        cap.update(c)
    ro = R.build(rows, cap, "aaaa")
    return B.make(ro, algo(**kw), kw.pop("max_train_tokens", 10_000) if "max_train_tokens" in kw else 10_000,
                  "aaaa", NEWLINES, CH)


def test_batch_counts_B_tries_including_zero_advantage():
    b = make([("a", [1, 0, 0], 2), ("b", [1, 1, 1], 2)])
    assert b.n_tries == 6
    assert b.n_tokens == 6 * 2 * 5
    assert {s["rollout_id"].split("/")[1] for s in b.segments} == {"a"}   # uniform group b is not trained
    assert b.stats["zero_adv"] == 3 and b.stats["mixed_groups"] == 1


def test_batch_counts_only_nonzero_when_configured():
    b = make([("a", [1, 0, 0], 2), ("b", [1, 1, 1], 2)], count_zero_adv_tries=False)
    assert b.n_tries == 3 and b.n_tokens == 3 * 2 * 5


def test_splice_makes_one_segment_per_try_and_targets_match():
    b = make([("a", [1, 0], 3)])
    assert len(b.segments) == 2 and b.stats["swaps"] == 2 * 2
    for s in b.segments:
        assert sum(s["mask"]) == 3 * 5
        assert s["adv"] in (1.0, -1.0)


def test_excluded_try_shrinks_group_and_is_not_counted():
    rows, cap = group("a", [1, 0, 0], 1)
    rows[2]["harness_error"] = "litellm.InternalServerError: x"
    b = B.make(R.build(rows, cap, "aaaa"), algo(), 10_000, "aaaa", NEWLINES, CH)
    assert b.n_tries == 2
    assert sorted(s["adv"] for s in b.segments) == [-1.0, 1.0]
    assert b.stats["excluded:infra:server_error"] == 1


def test_too_long_try_is_untrained_but_counted():
    rows, cap = group("a", [1, 0], 1)
    ro = R.build(rows, cap, "aaaa")
    b = B.make(ro, algo(), 4, "aaaa", NEWLINES, CH)
    assert b.segments == [] and b.n_tries == 2 and b.stats["too_long"] == 2


def test_round_trip():
    b = make([("a", [1, 0], 1)])
    assert B.Batch.from_dict(json.loads(json.dumps(b.to_dict()))).to_dict() == b.to_dict()


def test_tokenizer_ids_on_real_tokenizer():
    from transformers import AutoTokenizer
    from training.rl.config import ROOT
    tok = AutoTokenizer.from_pretrained(ROOT / "data/external/models/gemma-4-e2b-it-qat-w4a16-ct")
    nl, ch = B.tokenizer_ids(tok)
    assert tok.convert_ids_to_tokens(ch) == "<channel|>"
    assert {107, 108} <= nl and all(set(tok.decode([i])) == {"\n"} for i in nl)
