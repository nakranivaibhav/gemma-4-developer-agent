"""Policy loss through verl's registry (verl/trainer/ppo/core_algos.py), applied one segment at a time.

We train with gradient accumulation: one backward per segment, one optimizer step per epoch over the batch. verl's
loss functions aggregate with `agg_loss(..., **global_batch_info)`; passing the GLOBAL counts of the whole step
(B tries, total target tokens) makes each segment's loss its exact share of the full-batch loss, so the summed
gradient equals one backward over the whole padded batch (tested in tests/test_losses.py). This is the same
mechanism verl's own workers use (verl/workers/utils/losses.py).

B counts tries, not segments: a try split into 3 segments still has weight 1 in the seq-mean modes. That is why
seq-mean-token-mean is not allowed: it would average inside each segment, not inside each try.
"""
from __future__ import annotations

import torch

ALLOWED_AGG = {"token-mean", "token-sum", "seq-mean-token-sum", "seq-mean-token-sum-norm"}


def make_config(algo, n_tries: int, n_tokens: int):
    """A real verl ActorConfig (some losses assert the type) carrying our clip settings and the step's global counts."""
    from verl.workers.config import ActorConfig

    if algo.loss_agg_mode not in ALLOWED_AGG:
        raise ValueError(f"loss_agg_mode {algo.loss_agg_mode!r} not in {sorted(ALLOWED_AGG)}")
    if n_tries < 1 or n_tokens < 1:
        raise ValueError(f"empty step: {n_tries} tries, {n_tokens} tokens")
    cfg = ActorConfig(strategy="fsdp", ppo_micro_batch_size_per_gpu=1, ppo_mini_batch_size=1, rollout_n=1,
                      clip_ratio=algo.clip_ratio_low, clip_ratio_low=algo.clip_ratio_low,
                      clip_ratio_high=algo.clip_ratio_high, clip_ratio_c=algo.clip_ratio_c,
                      loss_agg_mode=algo.loss_agg_mode, loss_scale_factor=algo.loss_scale_factor)
    cfg.global_batch_info.update(dp_size=1, batch_num_tokens=n_tokens, global_batch_size=n_tries,
                                 loss_scale_factor=algo.loss_scale_factor)
    return cfg


def segment_loss(name: str, log_prob: torch.Tensor, old_log_prob: torch.Tensor, adv: float,
                 resp_mask: torch.Tensor, cfg) -> tuple[torch.Tensor, dict]:
    """log_prob, old_log_prob, resp_mask: (T,) over predicted positions; returns this segment's share of the loss."""
    from verl.trainer.ppo.core_algos import get_policy_loss_fn

    if not (log_prob.shape == old_log_prob.shape == resp_mask.shape) or log_prob.dim() != 1:
        raise ValueError(f"shapes differ: {log_prob.shape} {old_log_prob.shape} {resp_mask.shape}")
    m = resp_mask.to(log_prob.dtype)[None]
    a = torch.full_like(m, float(adv)) * m
    loss, metrics = get_policy_loss_fn(name)(
        old_log_prob=old_log_prob[None].detach(), log_prob=log_prob[None], advantages=a, response_mask=m,
        loss_agg_mode=cfg.loss_agg_mode, config=cfg)
    return loss, metrics
