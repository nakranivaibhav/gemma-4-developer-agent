"""Segments: the place off-by-one bugs live. Synthetic ids: 1-5 system+task, 10.. thoughts/calls, 50 = stop
<|tool_response>, 106 = <turn|>, NL1/NL2 newline tokens, CH = <channel|>."""
import pytest

from training.rl import segments as S
from training.rl.tests.conftest import CH, NEWLINES, NL1, NL2, NL3, call

SYS = [1, 2, 3, 4, 5]


def split(calls):
    segs = S.split(calls, NEWLINES, CH)
    S.check(segs, calls)
    return segs


def test_single_call():
    c = [call(SYS, [10, 11, 50])]
    (s,) = split(c)
    assert s.ids == SYS + [10, 11, 50]
    assert s.mask == [0] * 5 + [1, 1, 1]
    assert s.old_logp[:5] == [None] * 5 and s.old_logp[5:] == pytest.approx([-0.1, -0.2, -0.3])
    assert s.call_of == [None] * 5 + [0, 0, 0]


def test_append_only_history_is_one_segment_and_tool_output_is_unmasked():
    g1, r1, g2 = [10, 11, 50], [60, 61], [20, 106]
    c = [call(SYS, g1), call(SYS + g1 + r1, g2)]
    (s,) = split(c)
    assert s.ids == SYS + g1 + r1 + g2
    assert s.mask == [0] * 5 + [1] * 3 + [0, 0] + [1, 1]
    assert s.call_of[-2:] == [1, 1] and s.calls == [0, 1]


def test_stop_tokens_are_targets():
    c = [call(SYS, [10, 50]), call(SYS + [10, 50, 60], [20, 106])]
    (s,) = split(c)
    assert [t for t, m in zip(s.ids, s.mask) if m] == [10, 50, 20, 106]


def test_newline_swap_before_channel_splices_and_keeps_sampled_id():
    g1 = [10, NL1, CH, 30, 50]                     # thought, "\n", <channel|>, call, stop
    rendered = [10, NL2, CH, 30, 50]               # template re-renders "\n\n"
    c = [call(SYS, g1), call(SYS + rendered + [60], [20, 106])]
    (s,) = split(c)
    assert s.n_swaps == 1
    assert s.ids == SYS + g1 + [60, 20, 106]       # sampled NL1 kept, not the rendered NL2


def test_longer_newline_runs_also_splice():
    g1 = [10, NL2, CH, 30, 50]
    c = [call(SYS, g1), call(SYS + [10, NL3, CH, 30, 50, 60], [20, 106])]
    assert len(split(c)) == 1


def test_newline_swap_not_before_channel_starts_new_segment():
    g1 = [10, NL1, 12, 30, 50]
    c = [call(SYS, g1), call(SYS + [10, NL2, 12, 30, 50, 60], [20, 106])]
    segs = split(c)
    assert len(segs) == 2 and segs[1].ids[:len(SYS) + 5] == SYS + [10, NL2, 12, 30, 50]


def test_non_newline_difference_before_channel_starts_new_segment():
    g1 = [10, 11, CH, 30, 50]
    c = [call(SYS, g1), call(SYS + [10, 12, CH, 30, 50, 60], [20, 106])]
    assert len(split(c)) == 2


def test_swap_at_last_position_is_not_spliced():
    # the differing token must be FOLLOWED by <channel|> inside the segment; at the end there is nothing to check
    g1 = [10, NL1]
    c = [call(SYS, g1, finish="length"), call(SYS + [10, NL2, 60], [20, 106])]
    assert len(split(c)) == 2


def test_nudge_with_dropped_thought_starts_new_segment_from_full_prompt():
    T1, C1, R1, N = [10, 11, NL1, CH], [30, 50], [60], [70]
    c = [call(SYS, T1 + C1),
         call(SYS + T1 + C1 + R1, [20, 106]),             # no tool call -> dropped, nudge follows
         call(SYS + C1 + R1 + N, [40, NL1, CH, 41, 50])]  # thought T1 gone
    segs = split(c)
    assert [x.calls for x in segs] == [[0, 1], [2]]
    assert segs[1].ids[:len(SYS + C1 + R1 + N)] == SYS + C1 + R1 + N
    assert segs[1].mask[:len(SYS + C1 + R1 + N)] == [0] * len(SYS + C1 + R1 + N)


def test_prompt_shorter_than_segment_starts_new_segment():
    c = [call(SYS, [10, 11, 50]), call(SYS[:3], [20, 106])]
    assert len(split(c)) == 2


def test_new_segment_can_be_continued_again():
    c = [call(SYS, [10, 50]), call([1, 9], [20, 50]), call([1, 9, 20, 50, 60], [21, 106])]
    segs = split(c)
    assert [x.calls for x in segs] == [[0], [1, 2]]


def test_length_cut_without_stop_token_is_trained_as_sampled():
    c = [call(SYS, [10, 11, 12], finish="length")]
    (s,) = split(c)
    assert s.mask[-3:] == [1, 1, 1]


def test_empty_completion_adds_no_targets():
    c = [call(SYS, []), call(SYS + [60], [10, 106])]
    segs = split(c)
    assert sum(x.n_targets for x in segs) == 2


def test_logprob_count_mismatch_raises():
    with pytest.raises(ValueError, match="logprobs"):
        S.split([call(SYS, [10, 11], logprobs=[-0.1])], NEWLINES, CH)
    with pytest.raises(ValueError):
        S.split([{"prompt_ids": SYS, "completion_ids": [10]}], NEWLINES, CH)


def test_empty_prompt_raises():
    with pytest.raises(ValueError, match="empty prompt"):
        S.split([call([], [10])], NEWLINES, CH)


def test_first_token_never_a_target_and_invariants_on_random_tries():
    import random
    rng = random.Random(0)
    for _ in range(200):
        calls, hist = [], list(SYS)
        for k in range(rng.randint(1, 8)):
            g = [rng.randint(10, 40) for _ in range(rng.randint(0, 6))] + [50]
            if rng.random() < 0.2:                     # history rewrite
                hist = hist[:rng.randint(1, len(hist))] + [70]
            calls.append(call(hist, g))
            hist = hist + g + [rng.randint(60, 69) for _ in range(rng.randint(1, 3))]
        segs = split(calls)                            # check() asserts every invariant
        assert all(s.mask[0] == 0 for s in segs)
