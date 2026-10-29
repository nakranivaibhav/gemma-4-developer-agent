import random

import pytest

from training.rl import advantage


def rloo_hand(rewards, groups):
    out = []
    for i, (r, g) in enumerate(zip(rewards, groups)):
        others = [x for j, (x, h) in enumerate(zip(rewards, groups)) if h == g and j != i]
        out.append(r - sum(others) / len(others) if others else 0.0)
    return out


@pytest.mark.parametrize("est", ["rloo", "rloo_vectorized"])
@pytest.mark.parametrize("n", [2, 3, 8])
def test_rloo_matches_hand_formula(est, n):
    rng = random.Random(n)
    groups = [f"t{k}" for k in range(5) for _ in range(n)]
    rewards = [float(rng.random() < 0.3) for _ in groups]
    got = advantage.compute(rewards, groups, est)
    assert got == pytest.approx(rloo_hand(rewards, groups), abs=1e-6)


def test_known_values():
    assert advantage.compute([1, 0, 0], ["a"] * 3) == pytest.approx([1.0, -0.5, -0.5])
    a8 = advantage.compute([1] + [0] * 7, ["a"] * 8)
    assert a8[0] == pytest.approx(1.0) and a8[1] == pytest.approx(-1 / 7)


@pytest.mark.parametrize("est", sorted(advantage.SUPPORTED))
def test_group_sums_to_zero_and_uniform_groups_are_exactly_zero(est):
    groups = ["a"] * 8 + ["b"] * 8 + ["c"] * 8
    rewards = [1, 0, 0, 1, 0, 0, 0, 1] + [1] * 8 + [0] * 8
    adv = advantage.compute(rewards, groups, est)
    assert abs(sum(adv[:8])) < 1e-5
    assert adv[8:] == [0.0] * 16          # exact zeros: these tries are skipped by != 0


def test_interleaved_order_gives_same_advantage_per_try():
    groups = ["a", "b", "a", "b", "a", "b"]
    rewards = [1, 0, 0, 1, 0, 0]
    adv = advantage.compute(rewards, groups)
    sorted_idx = sorted(range(6), key=lambda i: groups[i])
    adv2 = advantage.compute([rewards[i] for i in sorted_idx], [groups[i] for i in sorted_idx])
    assert [adv[i] for i in sorted_idx] == pytest.approx(adv2)


def test_group_of_one_gets_zero_not_raw_reward():
    # verl's RLOO returns the raw reward for a lone try; we refuse to push it without a baseline
    adv = advantage.compute([1, 1, 0], ["lone", "a", "a"])
    assert adv == pytest.approx([0.0, 1.0, -1.0])


def test_min_group_size_threshold():
    adv = advantage.compute([1, 0, 0], ["a"] * 3, min_group_size=4)
    assert adv == [0.0, 0.0, 0.0]


def test_grpo_std_normalisation_switch():
    r, g = [1, 0, 0, 0], ["a"] * 4
    plain = advantage.compute(r, g, "grpo", norm_adv_by_std=False)
    normed = advantage.compute(r, g, "grpo", norm_adv_by_std=True)
    assert plain == pytest.approx([0.75, -0.25, -0.25, -0.25])
    assert normed[0] == pytest.approx(0.75 / 0.5, rel=1e-4)          # unbiased std of [1,0,0,0] = 0.5


def test_inputs_are_checked():
    with pytest.raises(ValueError):
        advantage.compute([1.0], ["a", "b"])
    with pytest.raises(ValueError):
        advantage.compute([None, 1.0], ["a", "a"])
    with pytest.raises(ValueError, match="not supported"):
        advantage.compute([1.0, 0.0], ["a", "a"], "gae")
    assert advantage.compute([], []) == []
