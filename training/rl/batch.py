"""Rollouts -> the training batch of one step: advantages, segments of the tries that teach something, and the
global counts the loss needs.

Counting rules (all tested):
- n_tries (B): valid (not excluded) tries; with count_zero_adv_tries=False only tries with advantage != 0.
- n_tokens: sampled tokens of the same tries (used by the token-mean / token-sum modes).
- Tries with advantage 0 are not trained (their gradient is exactly 0) but still count when count_zero_adv_tries.
- A try with any segment longer than max_train_tokens is not trained at all (never half a try) but still counts.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

from . import advantage, segments as seg_mod
from .rollouts import Rollout


@dataclass
class Batch:
    policy_sha: str
    n_tries: int
    n_tokens: int
    segments: list[dict] = field(default_factory=list)   # {"rollout_id", "adv", "ids", "mask", "old_logp"}
    tries: list[dict] = field(default_factory=list)      # per try: id, task, reward, adv, kind, exclude, segments
    stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Batch":
        return cls(**d)


def make(rollouts: list[Rollout], algo, max_train_tokens: int, policy_sha: str,
         newline_ids: frozenset, channel_id: int) -> Batch:
    valid = [r for r in rollouts if not r.exclude]
    adv = advantage.compute([r.reward for r in valid], [r.task for r in valid], algo.adv_estimator,
                            algo.norm_adv_by_std, algo.min_group_size)
    adv_of = {r.rollout_id: a for r, a in zip(valid, adv)}

    counted = [r for r in valid if algo.count_zero_adv_tries or adv_of[r.rollout_id] != 0]
    n_tokens = sum(len(c["completion_ids"]) for r in counted for c in r.calls)
    b = Batch(policy_sha=policy_sha, n_tries=len(counted), n_tokens=n_tokens)
    stats = Counter()
    for r in rollouts:
        info = {"rollout_id": r.rollout_id, "task": r.task, "kind": r.outcome.kind, "exclude": r.exclude,
                "reward": r.reward, "adv": adv_of.get(r.rollout_id), "n_calls": len(r.calls), "segments": 0}
        b.tries.append(info)
        stats[f"kind:{r.outcome.kind}"] += 1
        if r.exclude:
            stats[f"excluded:{r.exclude}"] += 1
            continue
        a = adv_of[r.rollout_id]
        if a == 0:
            stats["zero_adv"] += 1
            continue
        segs = seg_mod.split(r.calls, newline_ids, channel_id)
        seg_mod.check(segs, r.calls)
        if max(len(s) for s in segs) > max_train_tokens:
            stats["too_long"] += 1
            info["exclude"] = "too_long_for_training"
            continue
        info["segments"] = len(segs)
        stats["trained_tries"] += 1
        stats["swaps"] += sum(s.n_swaps for s in segs)
        for s in segs:
            b.segments.append({"rollout_id": r.rollout_id, "adv": a, "ids": s.ids, "mask": s.mask,
                               "old_logp": s.old_logp})
    stats["segments"] = len(b.segments)
    stats["segment_tokens"] = sum(len(s["ids"]) for s in b.segments)
    stats["target_tokens"] = sum(sum(s["mask"]) for s in b.segments)
    stats["valid_tries"] = len(valid)
    stats["mixed_groups"] = len({r.task for r in valid if adv_of[r.rollout_id] != 0})
    stats["groups"] = len({r.task for r in valid})
    stats["mean_reward"] = round(sum(r.reward for r in valid) / len(valid), 4) if valid else None
    b.stats = dict(stats)
    return b


def tokenizer_ids(tok) -> tuple[frozenset, int]:
    """(ids of tokens that are only newlines, id of <channel|>) for the splice rule."""
    channel = tok.convert_tokens_to_ids("<channel|>")
    if channel is None or channel == tok.unk_token_id:
        raise ValueError("tokenizer has no <channel|> token")
    newline = frozenset(i for t, i in tok.get_vocab().items()
                        if tok.convert_tokens_to_string([t]) and set(tok.convert_tokens_to_string([t])) == {"\n"})
    if not newline:
        raise ValueError("tokenizer has no newline-only tokens")
    return newline, channel
