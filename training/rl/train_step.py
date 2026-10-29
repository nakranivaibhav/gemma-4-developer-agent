"""One RL update: adapter vN + the step's batch -> adapter vN+1 (+ optimizer state, metrics).

usage: uv run python -m training.rl.train_step --config C.yaml --step-dir runs/rl/<name>/step_0003 \
           --adapter-in DIR [--optim-in FILE] --step 3

Per epoch: zero_grad; for every segment: forward, verl policy loss with the step's global counts, backward
(gradient accumulation); then one clip + optimizer step. Epoch 1 runs at the rollout weights, so its detached
log-probs are the "old" log-probs (ratio exactly 1, verl-style recompute); later epochs reuse them (PPO).
vLLM's own log-probs are raw (temperature 1, before top-k/top-p): they are only compared, at temperature 1, as the
rollout-vs-trainer mismatch metric. (vLLM's processed_logprobs mode would switch its top-k/top-p sampler from
FlashInfer to the PyTorch path: a sampling change, so it stays off.)

The on-policy guard: the batch was rolled out with adapter sha X; the adapter loaded here must be X.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # before torch import

import torch

from . import config as C, losses
from .batch import Batch
from .publish import sha16


def lr_at(optim, step: int) -> float:
    return optim.lr * min(1.0, (step + 1) / optim.warmup_steps) if optim.warmup_steps else optim.lr


def train_batch(score, params, opt, batch: Batch, algo, temperature: float, max_grad_norm: float,
                log_mismatch: bool = True, device="cuda") -> dict:
    """score(ids, mask, temperature, raw) -> (logp (T-1,), raw_logp (T-1,) | None). Returns metrics."""
    cfg = losses.make_config(algo, batch.n_tries, batch.n_tokens)
    old: dict[int, torch.Tensor] = {}
    diffs: list[torch.Tensor] = []
    epochs = []
    for epoch in range(algo.ppo_epochs):
        opt.zero_grad(set_to_none=True)
        t0, loss_sum, n_seg, agg = time.time(), 0.0, 0, {}
        for i, s in enumerate(batch.segments):
            resp = torch.tensor(s["mask"][1:], device=device)
            if not resp.any() or s["adv"] == 0:
                continue
            lp, raw = score(s["ids"], s["mask"], temperature, raw=log_mismatch and epoch == 0)
            if epoch == 0:
                old[i] = lp.detach()
            loss, m = losses.segment_loss(algo.policy_loss, lp, old[i], s["adv"], resp, cfg)
            loss.backward()
            loss_sum += loss.item()
            n_seg += 1
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v * float(resp.sum())
            if raw is not None:
                v_old = torch.tensor([x if x is not None else 0.0 for x in s["old_logp"][1:]], device=device)
                diffs.append((raw - v_old)[resp.bool()].float().cpu())
            del lp, raw, loss
        grads = [p.grad for p in params if p.grad is not None]
        gn = torch.nn.utils.clip_grad_norm_(params, max_grad_norm).item() if grads else 0.0
        if not math.isfinite(gn):
            raise FloatingPointError(f"epoch {epoch}: grad norm {gn}")
        opt.step()
        tok = sum(sum(s["mask"]) for s in batch.segments if s["adv"] != 0)
        epochs.append({"loss": loss_sum, "grad_norm": gn, "segments": n_seg, "seconds": round(time.time() - t0, 1),
                       **{k: v / max(tok, 1) for k, v in agg.items()}})
    out = {"epochs": epochs, "n_tries": batch.n_tries, "n_tokens": batch.n_tokens,
           "trained_target_tokens": sum(sum(s["mask"]) for s in batch.segments if s["adv"] != 0)}
    if diffs:
        d = torch.cat(diffs)
        r = d.exp()
        out["mismatch"] = {"n": d.numel(), "mean_diff": d.mean().item(), "mean_abs_diff": d.abs().mean().item(),
                           "p99_abs_diff": d.abs().quantile(0.99).item() if d.numel() < 2 ** 24 else None,
                           "k3_kl": (r - 1 - d).mean().item(),   # KL(vLLM || trainer) estimate at temperature 1
                           "ratio_in_0.8_1.2": ((r > 0.8) & (r < 1.2)).float().mean().item()}
    return out


def load_policy(cfg: C.RLConfig, adapter_dir: Path):
    from peft import PeftModel

    from training import gemma_lora

    base = gemma_lora.load_base(C.ROOT / cfg.model.model_dir, offload_ple=cfg.model.offload_ple)
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    for m in model.modules():                 # pi here = the policy vLLM sampled from: vLLM has no dropout
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


def read_batch(path: Path) -> Batch:
    with gzip.open(path, "rt") as fh:
        return Batch.from_dict(json.load(fh))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--step-dir", type=Path, required=True)
    ap.add_argument("--adapter-in", type=Path, required=True)
    ap.add_argument("--optim-in", type=Path)
    ap.add_argument("--step", type=int, required=True)
    a = ap.parse_args(argv)
    cfg = C.load(a.config, a.set)
    sd = a.step_dir
    batch = read_batch(sd / "batch.json.gz")
    sha_in = sha16(a.adapter_in / "adapter_model.safetensors")
    if batch.policy_sha != sha_in:
        raise SystemExit(f"on-policy guard: batch rolled out with {batch.policy_sha}, adapter-in is {sha_in}")

    from training import gemma_lora
    from .logprob import Scorer

    torch.manual_seed(a.step)
    model = load_policy(cfg, a.adapter_in)
    params = [p for p in model.parameters() if p.requires_grad]
    o = cfg.optim
    opt = torch.optim.AdamW(params, lr=lr_at(o, a.step), betas=tuple(o.betas), eps=o.eps, weight_decay=o.weight_decay)
    if a.optim_in:
        opt.load_state_dict(torch.load(a.optim_in, map_location="cuda"))
        for g in opt.param_groups:
            g["lr"] = lr_at(o, a.step)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with gemma_lora.sdpa_ctx():               # forward AND the checkpoint recompute in backward
        metrics = train_batch(Scorer(model, cfg.model.logit_softcap), params, opt, batch, cfg.algo,
                              cfg.temperature, o.max_grad_norm, cfg.model.log_mismatch)
    out = sd / "adapter_out"
    model.save_pretrained(out)
    torch.save(opt.state_dict(), sd / "optim.pt")
    metrics.update(step=a.step, lr=lr_at(o, a.step), seconds=round(time.time() - t0, 1),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), adapter_in=sha_in,
                   adapter_out=sha16(out / "adapter_model.safetensors"))
    (sd / "train_metrics.json").write_text(json.dumps(metrics, indent=1))
    print("TRAIN_DONE " + json.dumps({k: v for k, v in metrics.items() if k != "epochs"}), flush=True)


if __name__ == "__main__":
    main()
