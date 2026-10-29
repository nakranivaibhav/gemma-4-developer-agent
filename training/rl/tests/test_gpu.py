"""GPU tests (RL_GPU_TESTS=1). The real E2B policy (int4 unpacked + e2b_sft_v1 adapter) is loaded once.
Data: runs/tito_smoke1 (4 real TITO tries of one tune task under that adapter; local, not in the repo)."""
import gzip
import json
from collections import defaultdict

import pytest
import torch

from training.rl.tests.conftest import ROOT

pytestmark = pytest.mark.gpu
CAPTURE = ROOT / "runs/tito_smoke1/requests.jsonl.gz"
ADAPTER = ROOT / "runs/sft/e2b_v1/adapter"


def test_liger_per_token_logprob_and_gradient_match_hand():
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    torch.manual_seed(0)
    T, H, V, cap, temp = 300, 64, 5000, 30.0, 0.7
    h = (torch.randn(T, H, device="cuda") * 3).requires_grad_()
    W = torch.randn(V, H, device="cuda")
    y = torch.randint(0, V, (T,), device="cuda")
    y[::7] = -100
    w = torch.randn(T, device="cuda")                       # per-token weights, like A*ratio terms
    keep = y != -100
    lp_hand = torch.log_softmax(cap * torch.tanh(h @ W.T / cap) / temp, -1).gather(-1, y.clamp(min=0)[:, None])[:, 0]
    lp_hand = lp_hand * keep
    g_hand, = torch.autograd.grad((w * lp_hand).sum(), h)
    lp_lig = -LigerFusedLinearCrossEntropyLoss(softcap=cap / temp, reduction="none")(W, h / temp, y)
    g_lig, = torch.autograd.grad((w * lp_lig).sum(), h)
    assert torch.allclose(lp_lig, lp_hand, atol=1e-4)
    assert torch.allclose(g_lig, g_hand, atol=1e-4)


@pytest.fixture(scope="module")
def policy():
    from types import SimpleNamespace

    from training.rl import train_step
    from training.rl.logprob import Scorer
    cfg = SimpleNamespace(model=SimpleNamespace(model_dir="data/external/models/gemma-4-e2b-it-qat-w4a16-ct",
                                                offload_ple=False))
    model = train_step.load_policy(cfg, ADAPTER)
    yield model, Scorer(model, 30.0)
    del model
    torch.cuda.empty_cache()


def tries():
    rows = sorted((json.loads(line) for line in gzip.open(CAPTURE, "rt")), key=lambda r: r["utc"])

    def key(r):                      # tito_smoke1 predates the rollout tag: group by the first tool call id
        a = [m for m in r["request"]["messages"] if m["role"] == "assistant" and m.get("tool_calls")]
        if a:
            return a[0]["tool_calls"][0]["id"]
        tc = r["response"]["choices"][0]["message"].get("tool_calls")
        return tc[0]["id"] if tc else f"single_{r['utc']}"
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r["tito"])
    return list(out.values())


def spliced_segments():
    from transformers import AutoTokenizer

    from training.rl import batch, segments
    nl, ch = batch.tokenizer_ids(AutoTokenizer.from_pretrained(ROOT / "data/external/models/gemma-4-e2b-it-qat-w4a16-ct"))
    return [s for calls in tries() for s in segments.split(calls, nl, ch)]


def test_scorer_matches_fp32_reference_and_model_alignment(policy):
    """Shift, softcap and temperature on a real segment prefix. Reference: the same final hidden state through the
    LM head in fp32 (cap * tanh(z / cap) / T). The model's own logits are bf16 (a logit near 20 is rounded to 1/8),
    so they only check the alignment, loosely."""
    from training import gemma_lora
    model, score = policy
    s = min(spliced_segments(), key=len)
    n = 3000
    ids, mask = s.ids[:n], s.mask[:n]
    if sum(mask[1:]) == 0:
        mask = [0] + [1] * (n - 1)
    with torch.no_grad():
        lp07, raw = score(ids, mask, 0.7, raw=True)
        with gemma_lora.final_hidden(model) as grab, gemma_lora.sdpa_ctx():
            logits_bf16 = model(input_ids=torch.tensor([ids], device="cuda")).logits[0, :-1].float()
        h = grab["h"][0, :-1].float()
        z = h @ gemma_lora.lm_head_weight(model).float().T
        capped = 30.0 * torch.tanh(z / 30.0)
    y = torch.tensor(ids[1:], device="cuda")
    m = torch.tensor(mask[1:], device="cuda").bool()

    def pick(lg, t):
        return torch.log_softmax(lg / t, -1).gather(-1, y[:, None])[:, 0]
    d07, d1 = (lp07 - pick(capped, 0.7))[m].abs(), (raw - pick(capped, 1.0))[m].abs()
    dm = (lp07 - pick(logits_bf16, 0.7))[m].abs()
    print(f"\nscorer vs fp32 ref: T=0.7 mean|d| {d07.mean():.2e} max {d07.max():.2e}; T=1 mean|d| {d1.mean():.2e}"
          f" | vs bf16 model logits T=0.7 mean|d| {dm.mean():.2e}")
    # measured 2026-10-26: 1.2e-2 / 8.4e-3 (liger rounds each logit chunk to bf16, as vLLM's LM head does);
    # HF's own bf16 logits + bf16 softcap: 3.3e-2. The error is rounding, so it must also be unbiased.
    assert d07.mean() < 2e-2 and d1.mean() < 2e-2
    assert d07.mean() < dm.mean()
    assert abs((lp07 - pick(capped, 0.7))[m].mean()) < 3e-3
    assert torch.all(lp07[~m] == 0)
    shifted = (lp07[1:] - pick(capped, 0.7)[:-1])[m[1:]].abs().mean()
    assert shifted > 10 * d07.mean()                    # an off-by-one comparison is far off


def test_drift_of_the_newline_splice(policy):
    """Trainer (T=1) vs vLLM raw logprobs on the spliced segment with the most calls: first call (no spliced token
    in its history) vs later calls (1+ spliced newlines). Records the numbers; fails only on a gross gap."""
    _, score = policy
    seg = max((s for s in spliced_segments() if len(s.calls) > 1), key=lambda s: len(s.calls))
    with torch.no_grad():
        _, raw = score(seg.ids, seg.mask, 1.0, raw=True)
    first, later = [], []
    for i in range(1, len(seg.ids)):
        if seg.mask[i]:
            d = raw[i - 1].item() - seg.old_logp[i]
            (first if seg.call_of[i] == seg.calls[0] else later).append(d)
    f, lt = torch.tensor(first), torch.tensor(later)
    rep = {"tokens": len(seg.ids), "calls": len(seg.calls), "swaps": seg.n_swaps,
           "first": {"n": len(first), "mean": f.mean().item(), "mean_abs": f.abs().mean().item(),
                     "p99_abs": f.abs().quantile(0.99).item()},
           "later": {"n": len(later), "mean": lt.mean().item(), "mean_abs": lt.abs().mean().item(),
                     "p99_abs": lt.abs().quantile(0.99).item()}}
    (ROOT / "runs/tito_smoke1/drift.json").write_text(json.dumps(rep, indent=1))
    print("\ndrift:", json.dumps(rep))
    assert rep["first"]["mean_abs"] < 0.1 and rep["later"]["mean_abs"] < 0.1


def test_one_update_on_a_32k_segment_fits(policy):
    from types import SimpleNamespace

    from training import gemma_lora
    from training.rl.batch import Batch
    from training.rl.train_step import train_batch
    model, score = policy
    base = max(spliced_segments(), key=len)
    ids = (base.ids * (32_700 // len(base.ids) + 1))[:32_700]
    mask = [0] * 24_000 + [1] * 8_700
    b = Batch("x", 1, 8700, [{"rollout_id": "a", "adv": 1.0, "ids": ids, "mask": mask, "old_logp": [None] * 32_700}])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=0.0)                 # lr 0: the module-scoped policy stays unchanged
    algo = SimpleNamespace(policy_loss="vanilla", loss_agg_mode="seq-mean-token-sum-norm", loss_scale_factor=4096.0,
                           clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0, ppo_epochs=1)
    torch.cuda.reset_peak_memory_stats()
    with gemma_lora.sdpa_ctx():
        out = train_batch(score, params, opt, b, algo, 0.7, 1.0, log_mismatch=False)
    peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n32.7k-token segment: peak {peak:.1f} GB, grad norm {out['epochs'][0]['grad_norm']:.3f}, "
          f"{out['epochs'][0]['seconds']} s")
    assert out["epochs"][0]["grad_norm"] > 0 and peak < 31.0
    opt.zero_grad(set_to_none=True)
