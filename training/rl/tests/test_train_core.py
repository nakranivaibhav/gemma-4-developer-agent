"""train_step.train_batch on a tiny CPU model whose score function has the same contract as logprob.Scorer:
logp[j] = log softmax(logits_j / T)[ids[j + 1]], 0 where mask[j + 1] == 0."""
from types import SimpleNamespace

import pytest
import torch

from training.rl.batch import Batch
from training.rl.train_step import lr_at, train_batch

V, D = 40, 8


class Tiny(torch.nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.emb = torch.nn.Embedding(V, D)
        self.out = torch.nn.Linear(D, V)

    def logits(self, ids):
        return self.out(self.emb(torch.tensor(ids[:-1])))          # position j sees token j only (causal)

    def score(self, ids, mask, temperature, raw=False):
        lg = self.logits(ids)
        y = torch.tensor(ids[1:])
        m = torch.tensor(mask[1:], dtype=torch.float32)
        lp = torch.log_softmax(lg / temperature, -1).gather(-1, y[:, None])[:, 0] * m
        r = None
        if raw:
            with torch.no_grad():
                r = torch.log_softmax(lg, -1).gather(-1, y[:, None])[:, 0] * m
        return lp, r


def algo(**kw):
    d = dict(policy_loss="vanilla", loss_agg_mode="seq-mean-token-sum-norm", loss_scale_factor=16.0,
             clip_ratio_low=0.2, clip_ratio_high=0.2, clip_ratio_c=3.0, ppo_epochs=1)
    d.update(kw)
    return SimpleNamespace(**d)


def seg(rid, adv, ids, mask, old=None):
    return {"rollout_id": rid, "adv": adv, "ids": ids, "mask": mask,
            "old_logp": old or [(-1.0 if m else None) for m in mask]}


def toy_batch(extra=()):
    s = [seg("a", 1.0, [1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 0, 1]),
         seg("a", 1.0, [1, 2, 9, 10], [0, 0, 1, 1]),               # second segment of the same try
         seg("b", -0.5, [1, 2, 7, 8, 9], [0, 0, 1, 1, 1]),
         seg("c", -0.5, [1, 2, 11, 12], [0, 0, 0, 1])]
    return Batch(policy_sha="x", n_tries=3, n_tokens=99, segments=s + list(extra))


class NoStep:
    def zero_grad(self, set_to_none=True):
        for p in self.params:
            p.grad = None

    def __init__(self, params):
        self.params = params

    def step(self):
        pass


def grads_of(model, batch, T=0.7, **kw):
    params = list(model.parameters())
    train_batch(model.score, params, NoStep(params), batch, algo(**kw), T, 1e9, log_mismatch=False, device="cpu")
    return [p.grad.clone() for p in params]


def hand_grads(model, batch, T=0.7, C=16.0):
    model.zero_grad()
    loss = 0.0
    for s in batch.segments:
        lp, _ = model.score(s["ids"], s["mask"], T)
        loss = loss - s["adv"] * lp.sum() / (C * batch.n_tries)
    loss.backward()
    return [p.grad.clone() for p in model.parameters()]


def test_gradient_equals_hand_reinforce():
    m = Tiny()
    b = toy_batch()
    for g1, g2 in zip(grads_of(m, b), hand_grads(m, b)):
        assert torch.allclose(g1, g2, atol=1e-6)


def test_zero_advantage_segment_changes_nothing():
    m = Tiny()
    g1 = grads_of(m, toy_batch())
    g2 = grads_of(m, toy_batch([seg("d", 0.0, [1, 2, 3, 4], [0, 0, 1, 1])]))
    for a, b in zip(g1, g2):
        assert torch.equal(a, b)


def test_temperature_enters_the_loss():
    m = Tiny()
    g07, g10 = grads_of(m, toy_batch(), T=0.7), grads_of(m, toy_batch(), T=1.0)
    assert not torch.allclose(g07[1], g10[1])


def test_positive_advantage_raises_its_logprob():
    m = Tiny()
    b = Batch("x", 1, 4, [seg("a", 1.0, [1, 2, 3, 4], [0, 0, 1, 1])])
    before = m.score(b.segments[0]["ids"], b.segments[0]["mask"], 0.7)[0].sum().item()
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    train_batch(m.score, list(m.parameters()), opt, b, algo(), 0.7, 1e9, log_mismatch=False, device="cpu")
    assert m.score(b.segments[0]["ids"], b.segments[0]["mask"], 0.7)[0].sum().item() > before


def test_ppo_epochs_reuse_first_pass_logprobs():
    m = Tiny()
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    out = train_batch(m.score, list(m.parameters()), opt, toy_batch(), algo(ppo_epochs=3), 0.7, 1e9,
                      log_mismatch=False, device="cpu")
    kl = [e["actor/ppo_kl"] for e in out["epochs"]]
    assert kl[0] == 0.0 and kl[1] != 0.0 and len(kl) == 3


def test_mismatch_aligns_vllm_logprob_with_the_token_it_belongs_to():
    m = Tiny()
    ids, mask = [1, 2, 3, 4, 5, 6], [0, 0, 1, 1, 0, 1]
    _, raw = m.score(ids, mask, 1.0, raw=True)
    # old_logp[i] belongs to ids[i], which the model predicts from position i - 1  -> raw[i - 1]
    old = [None] + [raw[i - 1].item() - 0.1 if mask[i] else None for i in range(1, len(ids))]
    b = Batch("x", 1, 3, [seg("a", 1.0, ids, mask, old)])
    out = train_batch(m.score, list(m.parameters()), NoStep(list(m.parameters())), b, algo(), 0.7, 1e9,
                      log_mismatch=True, device="cpu")
    assert out["mismatch"]["n"] == 3
    assert out["mismatch"]["mean_diff"] == pytest.approx(0.1, abs=1e-5)
    assert out["mismatch"]["mean_abs_diff"] == pytest.approx(0.1, abs=1e-5)


def test_grad_norm_is_clipped_and_reported():
    m = Tiny()
    params = list(m.parameters())
    out = train_batch(m.score, params, NoStep(params), toy_batch(), algo(), 0.7, 1e-6, log_mismatch=False,
                      device="cpu")
    total = torch.sqrt(sum((p.grad ** 2).sum() for p in params)).item()
    assert out["epochs"][0]["grad_norm"] > 1e-6 and total == pytest.approx(1e-6, rel=1e-3)


def test_lr_warmup():
    o = SimpleNamespace(lr=1.0, warmup_steps=4)
    assert [lr_at(o, k) for k in range(5)] == [0.25, 0.5, 0.75, 1.0, 1.0]
    assert lr_at(SimpleNamespace(lr=2.0, warmup_steps=0), 0) == 2.0
