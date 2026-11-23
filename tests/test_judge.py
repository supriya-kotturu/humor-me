"""judge/: Bradley-Terry fit, pair selection, position randomization, and the rank() loop.
Synthetic jokes only; no network."""
import json

import numpy as np
import pytest

from judge import LLMJudge, Ranking, SimulatedJudge, fit_bt, rank, write_judgments  # NEW
from judge.bt import borda_scores, win_prob, win_prob_interval  # NEW
from judge.judges import parse_ab  # NEW
from judge.select import adaptive_next

N_JOKES = 24
JOKES = [f"joke {i}" for i in range(N_JOKES)]


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]


def pooled_all_pairs(judge, repeats):
    """Every pair `repeats` times (one all_pairs pass per seed), pooled into one fit."""
    comps = [c for s in range(repeats) for c in rank("p", JOKES, judge, method="all_pairs", budget=None, seed=s).comparisons]
    return comps, fit_bt(N_JOKES, [c.winner_loser() for c in comps])[0]


def test_fit_bt_orders_by_wins_and_centers():
    theta, raw = fit_bt(3, [(0, 1), (0, 2), (1, 2), (0, 1)])
    assert theta[0] > theta[1] > theta[2]
    assert abs(theta.mean()) < 1e-12
    assert np.allclose(theta, raw - raw.mean())


def test_winless_item_gets_finite_theta():
    theta, _ = fit_bt(3, [(0, 2), (1, 2), (0, 2)])
    assert np.all(np.isfinite(theta))
    assert theta[2] == theta.min()


def test_borda_scores():
    b = borda_scores(4, [(0, 1), (0, 1), (1, 0), (0, 2)])
    assert np.isclose(b[0], (2 / 3 + 1) / 2) and b[2] == 0.0 and np.isnan(b[3])


def test_all_pairs_recovers_true_order():
    theta_true = np.random.default_rng(1).normal(size=N_JOKES)
    _, theta = pooled_all_pairs(SimulatedJudge(theta_true, JOKES, seed=2), repeats=10)
    assert spearman(theta, theta_true) > 0.9


def test_all_pairs_single_pass_and_budget():
    judge = SimulatedJudge(np.zeros(N_JOKES), JOKES)
    r = rank("p", JOKES, judge, method="all_pairs", budget=None)
    assert r.n_calls == N_JOKES * (N_JOKES - 1) // 2 and r.stop_reason == "exhausted"
    assert len({(min(c.a, c.b), max(c.a, c.b)) for c in r.comparisons}) == r.n_calls
    r = rank("p", JOKES, judge, method="all_pairs", budget=30)
    assert r.n_calls == 30 and r.stop_reason == "budget"


def test_position_randomization_cancels_first_bias():
    judge = SimulatedJudge(np.zeros(N_JOKES), JOKES, seed=3, first_bias=3.0)
    comps, theta = pooled_all_pairs(judge, repeats=10)
    share_a_first = np.mean([c.shown_first == "a" for c in comps])
    assert 0.45 < share_a_first < 0.55
    # The bias shows up in the raw calls: the first-shown joke wins ~95% of the time...
    assert np.mean([c.winner == c.shown_first for c in comps]) > 0.9
    # ...but the coin spreads it evenly, so no joke's strength moves.
    assert np.all(np.abs(theta) < 0.5)


def test_same_seed_same_comparisons():
    theta_true = np.random.default_rng(1).normal(size=N_JOKES)

    def run(seed):
        return rank("p", JOKES, SimulatedJudge(theta_true, JOKES, seed=0), method="all_pairs", budget=60, seed=seed).comparisons

    assert run(0) == run(0)
    assert run(0) != run(1)


def test_adaptive_next_stops_when_separated():
    theta = np.array([3.0, 2.0, 1.0, -1.0, -2.0, -3.0])
    assert adaptive_next(theta, np.full(6, 0.01), [2, 4], beta=1.96) is None


def test_adaptive_next_picks_the_most_overlapping_boundary():
    theta = np.array([3.0, 2.0, 1.9, 0.0, -2.0, -3.0])
    se = np.full(6, 0.1)
    # Cut-off 2 overlaps (2.0 vs 1.9), cut-off 4 is clear: compare the two jokes at the boundary.
    assert adaptive_next(theta, se, [2, 4], beta=1.96) == (1, 2)
    # A wider interval on joke 0 makes it the weakest insider instead.
    se[0] = 1.0
    assert adaptive_next(theta, se, [2, 4], beta=1.96) == (0, 2)


def test_adaptive_rank_respects_budget():
    theta_true = np.random.default_rng(1).normal(size=N_JOKES)
    for seed in range(3):
        r = rank("p", JOKES, SimulatedJudge(theta_true, JOKES, seed=seed), budget=80, seed=seed)
        assert r.n_calls == len(r.comparisons) <= 80
        assert r.stop_reason in ("separated", "budget")
        top, bottom = r.top(5), r.bottom(5)
        assert len(top) == len(bottom) == 5 and not set(top) & set(bottom)


def test_adaptive_rank_separates_an_easy_set():
    theta_true = np.repeat([4.0, 0.0, -4.0], 4)  # three clear tiers of 4
    jokes = JOKES[:12]
    r = rank("p", jokes, SimulatedJudge(theta_true, jokes), budget=500, top_cutoffs=(4,), bottom_cutoffs=(4,))
    assert r.stop_reason == "separated" and r.n_calls < 500
    assert set(r.top(4)) == {0, 1, 2, 3} and set(r.bottom(4)) == {8, 9, 10, 11}


def test_input_validation():
    judge = SimulatedJudge(np.zeros(N_JOKES), JOKES)
    with pytest.raises(ValueError):
        rank("p", JOKES[:1], judge)
    with pytest.raises(ValueError):
        rank("p", JOKES, judge, method="vibes")
    with pytest.raises(ValueError):
        rank("p", JOKES, judge, method="all_pairs", budget=-1)
    with pytest.raises(ValueError):
        SimulatedJudge([0.0, 1.0], ["same", "same"])
    with pytest.raises(ValueError):
        rank("p", JOKES[:8], judge, top_cutoffs=(10,), bottom_cutoffs=())  # m = 10 >= n
    with pytest.raises(ValueError):
        rank("p", JOKES[:8], judge, top_cutoffs=(), bottom_cutoffs=(8,))  # m = 0
    with pytest.raises(ValueError):
        rank("p", JOKES, judge, budget=None)  # adaptive needs a budget


def test_parse_ab():  # NEW
    assert parse_ab("A") == 0 and parse_ab(" b.") == 1  # NEW
    assert parse_ab("<think>A is better... or B</think>Answer: B") == 1  # NEW
    assert parse_ab("A is clever, but B is funnier. B") == 1  # NEW
    assert parse_ab("Neither.") is None and parse_ab("<think>A") is None  # NEW


def test_llm_judge_reads_the_jokes_in_order(tmp_path):  # NEW
    """Fake transport that answers by content (the joke containing 'funny' wins): the judge must
    map it to the right position, rank() must undo the coin, and every answer is logged."""
    def chat(model, system, user):  # NEW
        a = user.split("Joke A:")[1].split("Joke B:")[0]  # NEW
        return "A" if "funny" in a else "B"  # NEW
    jokes = ["plain 0", "plain 1", "funny 2", "plain 3"]  # NEW
    judge = LLMJudge(chat=chat, log_path=str(tmp_path / "log.jsonl"))  # NEW
    assert judge.name == "llm:llama3.1:8b"  # NEW
    assert judge.compare("p", "funny x", "plain y") == 0 and judge.compare("p", "plain y", "funny x") == 1  # NEW
    r = rank("p", jokes, judge, method="all_pairs", budget=None)  # NEW
    assert r.order()[0] == 2 and r.borda[2] == 1.0  # NEW
    assert len((tmp_path / "log.jsonl").read_text().splitlines()) == 2 + 6  # NEW


def test_llm_judge_retries_then_raises(monkeypatch):  # NEW
    monkeypatch.setattr("judge.judges.time.sleep", lambda s: None)  # NEW
    replies = iter(["no idea", "B"])  # NEW
    assert LLMJudge(chat=lambda *a: next(replies)).compare("p", "x", "y") == 1  # NEW
    with pytest.raises(RuntimeError):  # NEW
        LLMJudge(chat=lambda *a: "neither").compare("p", "x", "y")  # NEW


def test_win_prob_is_bt_formula_and_symmetric():  # NEW
    theta = np.array([1.0, 0.0, -2.0])  # NEW
    assert win_prob(theta, 0, 1) == pytest.approx(1 / (1 + np.exp(-1)))  # NEW
    assert win_prob(theta, 0, 2) + win_prob(theta, 2, 0) == pytest.approx(1)  # NEW
    assert win_prob(theta, 1, 1) == 0.5  # NEW
    lo, hi = win_prob_interval(theta, np.array([0.5, 0.5, 0.5]), 0, 1)  # NEW
    assert lo < win_prob(theta, 0, 1) < hi  # NEW


def test_ranking_predicts_a_pair_it_never_asked():  # NEW
    """A beat B 3/4, B beat C 3/4, A and C never met: the fit still says A beats C, more
    confidently than either observed pair."""
    comps = [(0, 1)] * 3 + [(1, 0)] + [(1, 2)] * 3 + [(2, 1)]  # NEW
    theta, _ = fit_bt(3, comps, alpha=0.01)  # NEW
    r = Ranking(theta=theta, se=np.ones(3), borda=np.zeros(3), comparisons=[], n_calls=0, stop_reason="budget")  # NEW
    assert r.win_prob(0, 2) > r.win_prob(0, 1) > 0.5  # NEW
    assert r.compare(0, 2) == 0 and r.compare(2, 0) == 1  # NEW


def test_simulated_ranking_predictions_track_true_probabilities():  # NEW
    theta_true = np.linspace(2, -2, 8)  # NEW
    judge = SimulatedJudge(theta_true, JOKES[:8], seed=3)  # NEW
    comps = [c for s in range(20) for c in rank("p", JOKES[:8], judge, method="all_pairs", budget=None, seed=s).comparisons]  # NEW
    theta, _ = fit_bt(8, [c.winner_loser() for c in comps])  # NEW
    true_p = 1 / (1 + np.exp(-(theta_true[0] - theta_true[7])))  # NEW
    assert abs(win_prob(theta, 0, 7) - true_p) < 0.05  # NEW


def test_write_judgments(tmp_path):
    r = rank("p", JOKES[:4], SimulatedJudge(np.zeros(4), JOKES[:4]), method="all_pairs", budget=None, prompt_id="p1")
    lines = write_judgments(tmp_path / "j.jsonl", r.comparisons).read_text().splitlines()
    assert len(lines) == 6
    assert json.loads(lines[0]).keys() == {"prompt_id", "a", "b", "shown_first", "winner", "judge"}
    assert json.loads(lines[0])["prompt_id"] == "p1"
