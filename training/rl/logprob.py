"""Per-token log-probabilities of a segment under the trainer's policy, without the full logits.

logp[j] = log pi(ids[j + 1] | ids[:j + 1]) for j in 0..T-2, computed from the final hidden state with liger's fused
linear cross-entropy (chunked over tokens, full vocabulary per token). Gemma's logit softcap and the sampling
temperature are both folded in exactly: (cap/T) * tanh((z/T) / (cap/T)) = cap * tanh(z/cap) / T, so we pass
hidden / T and softcap cap / T (checked against the model's own logits in tests/test_gpu.py).
Positions whose target is not a sampled token (mask[j + 1] == 0) come back as 0.
"""
from __future__ import annotations

import torch

from training import gemma_lora


class Scorer:
    def __init__(self, model, softcap: float):
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        self.model, self.softcap = model, softcap
        self._ce = {}
        self._make = lambda t: LigerFusedLinearCrossEntropyLoss(softcap=softcap / t, reduction="none")

    def _loss(self, temperature):
        if temperature not in self._ce:
            self._ce[temperature] = self._make(temperature)
        return self._ce[temperature]

    def __call__(self, ids: list[int], mask: list[int], temperature: float, raw: bool = False):
        """-> (logp (T-1,) float32 with grad if grad is enabled, raw_logp (T-1,) at temperature 1 or None)."""
        if len(ids) != len(mask) or len(ids) < 2:
            raise ValueError(f"segment of {len(ids)} ids / {len(mask)} mask values")
        dev = "cuda"
        x = torch.tensor([ids], device=dev)
        y = torch.tensor(ids[1:], device=dev).masked_fill(torch.tensor(mask[1:], device=dev) == 0, -100)
        W = gemma_lora.lm_head_weight(self.model)
        with gemma_lora.final_hidden(self.model) as grab, gemma_lora.sdpa_ctx():
            self.model(input_ids=x, logits_to_keep=1)          # logits_to_keep=1: skip the 262k-wide logits
        h = grab["h"][0, :-1].to(W.dtype)                      # position j predicts ids[j + 1]
        logp = -self._loss(temperature)(W, h / temperature, y).float()
        raw_lp = None
        if raw:
            with torch.no_grad():
                raw_lp = -self._loss(1.0)(W, h.detach(), y).float()
        return logp, raw_lp
