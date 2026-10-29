"""Per-try advantages from verl's estimator registry (verl/trainer/ppo/core_algos.py), chosen by name.

Only outcome estimators that give ONE number per try are allowed: they are called with a one-slot reward row per
try, and the number is then spread over the try's sampled tokens by the loss. Token-level estimators (GAE,
REINFORCE++ whitening, OPO's length weights) would compute something else with a one-slot row, so they are refused.

Our one change to verl: a group with fewer than `min_group_size` valid tries gets advantage 0. verl's RLOO returns
the raw reward for a group of one, which would push a lone success up with no baseline (excluded tries can leave a
group that small).
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import torch

SUPPORTED = {"rloo", "rloo_vectorized", "grpo", "grpo_vectorized"}


def compute(rewards: list[float], groups: list[str], estimator: str = "rloo", norm_adv_by_std: bool = False,
            min_group_size: int = 2) -> list[float]:
    if estimator not in SUPPORTED:
        raise ValueError(f"advantage estimator {estimator!r} not supported; use one of {sorted(SUPPORTED)}")
    if len(rewards) != len(groups):
        raise ValueError(f"{len(rewards)} rewards for {len(groups)} group ids")
    if not rewards:
        return []
    if any(r is None or not np.isfinite(r) for r in rewards):
        raise ValueError("rewards must be finite numbers (excluded tries are removed before this)")
    from verl.trainer.ppo.core_algos import get_adv_estimator_fn

    fn = get_adv_estimator_fn(estimator)
    scores = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)    # (n, 1): outcome reward in one slot
    mask = torch.ones_like(scores)
    kw = {"norm_adv_by_std_in_grpo": norm_adv_by_std} if estimator.startswith("grpo") else {}
    adv, _ = fn(token_level_rewards=scores, response_mask=mask, index=np.array(groups, dtype=object), **kw)
    out = adv[:, 0].tolist()
    size = Counter(groups)
    return [0.0 if size[g] < min_group_size else a for a, g in zip(out, groups)]
