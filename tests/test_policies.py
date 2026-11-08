"""Mode-choice policies: interface, priors, and that learners actually learn in the sim."""
import numpy as np
import pytest

from sim.env import AudienceEnv, Context
from sim.evaluate_policies import career
from sim.policies import (N_FEATURES, EpsilonGreedy, EpsilonSoft, Fixed, LinTS, LinUCB, Random, features, n_features,
                          prior_means)

BAR = Context("bar", "normal", "mixed")


def obs(last_mode=None, last_reward=None, ctx=BAR):
    return {"t": 0, "n_bits": 8, "context": ctx, "last_mode": last_mode, "last_reward": last_reward}


def test_features_encode_context_and_history_per_arm():
    x = features(obs(last_mode=2, last_reward=0.5), arm=2)
    assert len(x) == N_FEATURES and x[0] == 1
    assert x[-2] == 1 and x[-1] == 0.5
    assert features(obs(last_mode=2), arm=3)[-2] == 0


def test_prior_kinds():
    env = AudienceEnv()
    hg = prior_means(env, "humorgen")
    assert env.modes[int(hg.argmax())] == "neurotic"  # HumorGen's top persona
    uni = prior_means(env, "uniform")
    assert np.allclose(uni, hg.mean())
    wrong = prior_means(env, "wrong")
    assert sorted(wrong) == pytest.approx(sorted(hg))
    assert int(wrong.argmax()) != int(hg.argmax())
    with pytest.raises(ValueError):
        prior_means(env, "vibes")


@pytest.mark.parametrize("cls", [LinUCB, LinTS])
def test_linear_policy_starts_at_prior_best(cls):
    prior = np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
    picks = [cls(prior, prior_strength=100.0, seed=s).act(obs()) for s in range(5)]
    assert all(p == 2 for p in picks)


def test_epsilon_greedy_follows_the_prior_until_data_says_otherwise():
    p = EpsilonGreedy(np.array([0.9, 0.1, 0.1, 0.1, 0.1, 0.1]), eps=0.0)
    assert p.act(obs()) == 0
    for _ in range(5):
        p.update(obs(), 0, 0.0)
        p.update(obs(), 1, 1.0)
    assert p.act(obs()) == 1


def career_laughs(policy, shows=60, seed=0):
    env = AudienceEnv()
    total = 0.0
    for s in range(shows):
        o = env.reset(BAR, seed=seed * 1000 + s)
        done = False
        while not done:
            a = policy.act(o)
            nxt, r, done, info = env.step(a)
            policy.update(o, a, r)
            o = nxt
            if s >= shows - 20:
                total += sum(info["levels"])
    return total / 20


@pytest.mark.parametrize("cls", [LinUCB, LinTS])
def test_learners_beat_random_after_experience(cls):
    env = AudienceEnv()
    uniform = prior_means(env, "uniform")
    assert career_laughs(cls(uniform, seed=0)) > career_laughs(Random(env.n_modes, seed=0)) + 2


def test_reaction_feature_is_this_modes_last_reward():
    x = features(obs(last_mode=2, last_reward=0.4), arm=2, react=True)
    assert len(x) == N_FEATURES + 1 and x[-1] == 0.4
    assert features(obs(last_mode=2, last_reward=0.4), arm=3, react=True)[-1] == 0
    assert len(features(obs(), arm=0)) == N_FEATURES  # default unchanged: #4's results stand


@pytest.mark.parametrize("cls", [LinUCB, LinTS])
def test_react_policies_act_and_learn(cls):
    p = cls(np.full(6, 0.3), seed=0, react=True)
    a = p.act(obs(last_mode=1, last_reward=0.0))
    p.update(obs(last_mode=1, last_reward=0.0), a, 0.5)
    assert p.A.shape[-1] == N_FEATURES + 1


def test_show_state_features_are_tonights_record_for_this_mode():
    o = obs(last_mode=1, last_reward=0.2)
    o["tonight"] = ((1, 0.6), (3, 0.1), (1, 0.2))
    x = features(o, arm=1, show_state=True)
    assert len(x) == n_features(show_state=True)
    assert x[-2] == 1.0 and x[-1] == pytest.approx(0.4)     # tried tonight, mean of 0.6 and 0.2
    assert list(features(o, arm=5, show_state=True)[-2:]) == [0.0, 0.0]  # untried tonight
    assert len(features(o, arm=1)) == N_FEATURES             # default unchanged


@pytest.mark.parametrize("cls", [LinUCB, LinTS])
def test_show_state_policies_act_and_learn(cls):
    p = cls(np.full(6, 0.3), seed=0, show_state=True)
    o = obs()
    o["tonight"] = ((0, 0.5),)
    a = p.act(o)
    p.update(o, a, 0.4)
    assert p.A.shape[-1] == n_features(show_state=True)


# --- show state step B: two timescales ---
from sim.policies import TwoTimescale


def test_linucb_scores_split_keeps_act_unchanged():
    p = LinUCB(np.array([0.1, 0.5, 0.2, 0.3, 0.2, 0.1]), seed=0)
    o = obs(last_mode=1, last_reward=0.3)
    s = p.scores(o)
    assert s.shape == (6, 2) and p.act(o) == int(np.argmax(s[:, 0] + s[:, 1]))


def test_tonight_correction_lowers_a_mode_that_bombed_and_resets_next_show():
    two = TwoTimescale(LinUCB(np.full(6, 0.3), seed=0), kappa=1.0)
    first = {**obs(), "t": 0}
    two.act(first)
    two.update(first, 2, 0.0)                    # mode 2 bombed vs a ~0.3 prediction
    corr = two.correction(6)
    assert corr[2] < -0.1 and all(corr[a] == 0 for a in (0, 1, 3, 4, 5))
    two.act({**obs(), "t": 0})                   # next show starts
    assert not two.correction(6).any()


def test_large_kappa_behaves_like_the_career_model():
    prior = np.array([0.1, 0.5, 0.2, 0.3, 0.2, 0.1])
    slow, plain = TwoTimescale(LinUCB(prior, seed=0), kappa=1e9), LinUCB(prior, seed=0)
    env = AudienceEnv()
    for show in range(3):
        o = env.reset(Context("bar", "normal", "mixed"), seed=show)
        done = False
        while not done:
            a = slow.act(o)
            assert a == plain.act(o)
            nxt, r, done, _ = env.step(a)
            slow.update(o, a, r)
            plain.update(o, a, r)
            o = nxt


K = 6


def test_epsilon_soft_probs_follow_the_formula_and_sum_to_one():
    p = EpsilonSoft(Fixed(2), 0.2, K).probs(obs())
    assert p.sum() == pytest.approx(1)
    assert p[2] == pytest.approx(1 - 0.2 + 0.2 / K)
    assert np.delete(p, 2) == pytest.approx(np.full(K - 1, 0.2 / K))


def test_epsilon_soft_empirical_pick_rates_match_probs():
    pol = EpsilonSoft(Fixed(2), 0.2, K, seed=1)
    n = 60_000
    picks = np.bincount([pol.act(obs()) for _ in range(n)], minlength=K) / n
    tol = 5 * np.sqrt((1 - 0.2 + 0.2 / K) * (0.2 - 0.2 / K) / n)  # 5 sigma at the largest probability
    assert picks == pytest.approx(pol.probs(obs()), abs=tol)
    assert pol.explored == pytest.approx(n * 0.2 * (K - 1) / K, rel=0.03)


def test_epsilon_soft_probs_put_the_mass_on_the_inner_greedy_mode():
    env = AudienceEnv()
    inner = LinUCB(prior_means(env, "uniform"))
    pol = EpsilonSoft(inner, 0.2, K)
    o = env.reset(BAR, seed=0)
    for _ in range(6):
        a = inner.act(o)
        assert pol.probs(o).argmax() == a
        nxt, r, *_ = env.step(a)
        inner.update(o, a, r)
        o = nxt


def test_epsilon_soft_decide_reports_one_consistent_decision_with_a_stochastic_inner_policy():
    env = AudienceEnv()
    pol = EpsilonSoft(LinTS(prior_means(env, "uniform"), seed=3), 0.2, K, seed=4)
    o = env.reset(BAR, seed=0)
    greedy_seen = set()
    for _ in range(8):
        action, greedy, probs = pol.decide(o)
        greedy_seen.add(greedy)
        assert probs.argmax() == greedy
        assert probs[greedy] == pytest.approx(1 - 0.2 + 0.2 / K) and probs.sum() == pytest.approx(1)
        nxt, r, *_ = env.step(action)
        pol.update(o, action, r)
        o = nxt
    assert len(greedy_seen) > 1  # the inner policy really is stochastic, so a second query would have disagreed


def test_epsilon_zero_is_exactly_plain_linucb():
    def plain(env, s):
        return LinUCB(prior_means(env, "uniform", seed=s), seed=s)

    def wrapped(env, s):
        return EpsilonSoft(plain(env, s), 0.0, env.n_modes, seed=s + 10_000)

    for seed in range(3):
        (la, lo), (wa, wo) = career(plain, 4, seed), career(wrapped, 4, seed)
        assert np.array_equal(la, wa) and np.array_equal(lo, wo)


def test_epsilon_zero_never_explores_and_epsilon_one_is_uniform():
    pol = EpsilonSoft(Fixed(0), 0.0, K)
    assert [pol.act(obs()) for _ in range(200)] == [0] * 200 and pol.explored == 0
    assert EpsilonSoft(Fixed(0), 1.0, K).probs(obs()) == pytest.approx(np.full(K, 1 / K))


def test_epsilon_soft_update_passes_through_unchanged():
    env = AudienceEnv()
    a, b = (LinUCB(prior_means(env, "uniform")) for _ in range(2))
    o = env.reset(BAR, seed=0)
    EpsilonSoft(a, 0.2, K).update(o, 3, 0.4)
    b.update(o, 3, 0.4)
    assert np.array_equal(a.A, b.A) and np.array_equal(a.b, b.b)


def test_epsilon_soft_rejects_eps_outside_the_unit_interval():
    with pytest.raises(ValueError):
        EpsilonSoft(Fixed(0), 1.1, K)
