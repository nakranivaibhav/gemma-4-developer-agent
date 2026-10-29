"""Per-segment accumulation with global counts == verl on the whole padded batch, for every allowed mode."""
from types import SimpleNamespace

import pytest
import torch

from training.rl import losses as L


def algo(mode, **kw):
    d = dict(policy_loss="vanilla", loss_agg_mode=mode, loss_scale_factor=64.0, clip_ratio_low=0.2,
             clip_ratio_high=0.2, clip_ratio_c=3.0)
    d.update(kw)
    return SimpleNamespace(**d)


def toy_batch(seed=0):
    """3 tries; try 0 has two segments. Returns segments [(try, adv, logp leaf (T,), mask (T,))]."""
    g = torch.Generator().manual_seed(seed)
    spec = [(0, 1.0, 7), (0, 1.0, 5), (1, -0.5, 9), (2, -0.5, 4)]
    segs = []
    for t, a, n in spec:
        lp = (-torch.rand(n, generator=g, dtype=torch.float64) * 2).requires_grad_()
        m = (torch.rand(n, generator=g) > 0.3).double()
        m[0] = 1.0
        segs.append((t, a, lp, m))
    return segs


def accumulate(segs, cfg, name="vanilla", old=None):
    total = 0.0
    for i, (t, a, lp, m) in enumerate(segs):
        o = lp.detach() if old is None else old[i]
        loss, _ = L.segment_loss(name, lp, o, a, m, cfg)
        loss.backward()
        total += loss.item()
    return total, [lp.grad.clone() for _, _, lp, _ in segs]


def full_batch(segs, mode, n_tries, n_tokens, C, name="vanilla", old=None):
    """verl on the padded batch, one row per segment, with the same global info."""
    from verl.trainer.ppo.core_algos import get_policy_loss_fn
    T = max(len(lp) for _, _, lp, _ in segs)
    leaves = [lp.detach().clone().requires_grad_() for _, _, lp, _ in segs]
    LP = torch.stack([torch.nn.functional.pad(x, (0, T - len(x))) for x in leaves])
    M = torch.stack([torch.nn.functional.pad(m, (0, T - len(m))) for _, _, _, m in segs])
    A = torch.stack([torch.full((T,), a, dtype=torch.float64) for _, a, _, _ in segs]) * M
    OLD = LP.detach() if old is None else torch.stack([torch.nn.functional.pad(o, (0, T - len(o))) for o in old])
    cfg = L.make_config(algo(mode, policy_loss=name, loss_scale_factor=C), n_tries, n_tokens)
    loss, _ = get_policy_loss_fn(name)(old_log_prob=OLD, log_prob=LP, advantages=A, response_mask=M,
                                       loss_agg_mode=mode, config=cfg)
    loss.backward()
    return loss.item(), [x.grad for x in leaves]


@pytest.mark.parametrize("mode", sorted(L.ALLOWED_AGG))
@pytest.mark.parametrize("name", ["vanilla", "gspo"])
def test_accumulated_equals_full_batch(mode, name):
    if name == "gspo" and mode != "seq-mean-token-sum-norm":
        pytest.skip("gspo: one mode is enough to show the plumbing")
    segs = toy_batch()
    n_tries, n_tokens = 3, int(sum(m.sum() for *_, m in segs))
    cfg = L.make_config(algo(mode, policy_loss=name), n_tries, n_tokens)
    acc, g_acc = accumulate(segs, cfg, name)
    full, g_full = full_batch(segs, mode, n_tries, n_tokens, 64.0, name)
    assert acc == pytest.approx(full, abs=1e-9)
    for a, b in zip(g_acc, g_full):
        assert torch.allclose(a, b, atol=1e-12)


def test_ratio_one_vanilla_is_reinforce_sum_over_C_times_B():
    segs = toy_batch(1)
    C, n_tries = 64.0, 3
    cfg = L.make_config(algo("seq-mean-token-sum-norm", loss_scale_factor=C), n_tries, 999)
    _, g = accumulate(segs, cfg)
    for (t, a, lp, m), gi in zip(segs, g):
        assert torch.allclose(gi, -a * m / (C * n_tries))     # d/dlogp of -A*sum(logp)/(C*B)


def test_B_counts_tries_not_segments():
    # splitting try 0 into two segments must not change its weight
    segs = toy_batch(2)
    merged = [(0, 1.0, torch.cat([segs[0][2].detach(), segs[1][2].detach()]).requires_grad_(),
               torch.cat([segs[0][3], segs[1][3]]))] + segs[2:]
    cfg = L.make_config(algo("seq-mean-token-sum-norm"), 3, 999)
    a, _ = accumulate(segs, cfg)
    b, _ = accumulate(merged, cfg)
    assert a == pytest.approx(b, abs=1e-12)


def test_zero_advantage_segment_adds_nothing():
    segs = toy_batch(3)
    cfg = L.make_config(algo("seq-mean-token-sum-norm"), 3, 999)
    t, _, lp, m = segs[0]
    loss, _ = L.segment_loss("vanilla", lp, lp.detach(), 0.0, m, cfg)
    loss.backward()
    assert loss.item() == 0.0 and torch.all(lp.grad == 0)


def test_clipping_stops_gradient_when_ratio_left_the_band():
    lp = torch.tensor([0.0, 0.0], dtype=torch.float64, requires_grad=True)
    old = torch.tensor([-1.0, 0.0], dtype=torch.float64)       # token 0: ratio e = 2.7 > 1.2
    cfg = L.make_config(algo("token-sum"), 1, 2)
    loss, m = L.segment_loss("vanilla", lp, old, 1.0, torch.ones(2, dtype=torch.float64), cfg)
    loss.backward()
    assert lp.grad[0] == 0 and lp.grad[1] == -1.0
    assert m["actor/pg_clipfrac"] == pytest.approx(0.5)


def test_bad_inputs():
    with pytest.raises(ValueError):
        L.make_config(algo("seq-mean-token-mean"), 1, 1)
    with pytest.raises(ValueError):
        L.make_config(algo("token-mean"), 0, 1)
    cfg = L.make_config(algo("token-mean"), 1, 1)
    with pytest.raises(ValueError):
        L.segment_loss("vanilla", torch.zeros(3), torch.zeros(2), 1.0, torch.ones(3), cfg)
