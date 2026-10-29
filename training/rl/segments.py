"""One try's model calls (exact vLLM token ids, from the TITO capture) -> training segments.

A segment is one token sequence that one forward pass can score: every sampled token in it sees exactly (up to the
splice below) the prompt vLLM gave it. Call k joins the open segment when its prompt starts with the segment's ids
so far (the previous prompt + the previous sampled ids); otherwise the history really changed (a nudge made the
template drop old thoughts, the harness dropped a no-tool-call reply) and a new segment starts from call k's full
prompt.

The splice: the model ends a thought with a newline token then <channel|>; vLLM's reasoning parser keeps that
newline in the reasoning text and the chat template re-renders it as thinking_text + "\\n<channel|>", so the next
prompt holds a longer newline token (e.g. 108 "\\n\\n") where the model sampled 107 "\\n". The scorer does the same.
`joins` accepts exactly that difference (a newline token replaced by another newline token, immediately before
<channel|>) and keeps the SAMPLED id, so every trained target is exactly what the model produced; the cost is that
later calls see the sampled newline where vLLM showed the longer one (measured by the drift check).

Indexing convention used everywhere downstream: ids[i] is a target where mask[i] == 1, its vLLM log-prob is
old_logp[i], and the model predicts it from position i - 1. mask[0] is always 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    ids: list[int] = field(default_factory=list)
    mask: list[int] = field(default_factory=list)          # 1 = sampled by the policy (gets loss)
    old_logp: list[float | None] = field(default_factory=list)  # vLLM raw log-prob where mask = 1, else None
    call_of: list[int | None] = field(default_factory=list)     # which call sampled the token, else None
    calls: list[int] = field(default_factory=list)
    swapped: set = field(default_factory=set)                # positions where a newline token was spliced

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def n_swaps(self) -> int:
        return len(self.swapped)

    @property
    def n_targets(self) -> int:
        return sum(self.mask)


def joins(seg_ids: list[int], prompt: list[int], newline_ids: frozenset, channel_id: int) -> set | None:
    """Positions of spliced newline tokens if `prompt` continues `seg_ids` (empty set = exact), else None."""
    n = len(seg_ids)
    if len(prompt) < n:
        return None
    if prompt[:n] == seg_ids:
        return set()
    swaps = set()
    for i in range(n):
        a, b = seg_ids[i], prompt[i]
        if a == b:
            continue
        if a in newline_ids and b in newline_ids and i + 1 < n and seg_ids[i + 1] == channel_id:
            swaps.add(i)
            continue
        return None
    return swaps


def split(calls: list[dict], newline_ids: frozenset, channel_id: int) -> list[Segment]:
    """calls: [{"prompt_ids", "completion_ids", "logprobs"}] in the order the try made them."""
    segs: list[Segment] = []
    for k, c in enumerate(calls):
        p, g = list(c["prompt_ids"]), list(c["completion_ids"])
        lp = c.get("logprobs")
        if not p:
            raise ValueError(f"call {k}: empty prompt")
        if lp is None or len(lp) != len(g):
            raise ValueError(f"call {k}: {0 if lp is None else len(lp)} logprobs for {len(g)} sampled ids")
        s = segs[-1] if segs else None
        swaps = joins(s.ids, p, newline_ids, channel_id) if s is not None else None
        if swaps is None:
            s = Segment()
            segs.append(s)
            new = p
        else:
            s.swapped |= swaps
            new = p[len(s.ids):]                 # keep the sampled ids; only the new tail of the prompt is added
        s.ids += new + g
        s.mask += [0] * len(new) + [1] * len(g)
        s.old_logp += [None] * len(new) + [float(x) for x in lp]
        s.call_of += [None] * len(new) + [k] * len(g)
        s.calls.append(k)
    return segs


def check(segs: list[Segment], calls: list[dict]) -> None:
    """Invariants every consumer relies on; raises AssertionError with the broken one."""
    trained = [t for s in segs for t, m in zip(s.ids, s.mask) if m]
    sampled = [t for c in calls for t in c["completion_ids"]]
    assert trained == sampled, "every sampled id trained exactly once, in order"
    for s in segs:
        assert len(s.ids) == len(s.mask) == len(s.old_logp) == len(s.call_of), "parallel lists differ in length"
        assert not s.mask or s.mask[0] == 0, "first token has no predecessor and cannot be a target"
        assert all((m == 1) == (lp is not None) == (c is not None)
                   for m, lp, c in zip(s.mask, s.old_logp, s.call_of)), "old_logp / call_of exactly on targets"
