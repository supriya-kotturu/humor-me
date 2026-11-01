"""Experiment functions behind notebooks/decisions.ipynb: cache semantics and tiny runs."""
import pytest

from sim import cache, evaluate_env, evaluate_policies, evaluate_shift


@pytest.fixture
def tmp_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "METRICS", tmp_path)
    return tmp_path


def test_cache_reuses_same_params_and_reruns_on_change(tmp_metrics):
    calls = []

    def fn(x):
        calls.append(x)
        return {"value": x * 2}

    assert cache.run_or_load("t", fn, x=1) == {"value": 2}
    assert cache.run_or_load("t", fn, x=1) == {"value": 2}
    assert calls == [1]                      # second call came from disk
    cache.run_or_load("t", fn, x=2)
    assert calls == [1, 2]                   # new params → recompute
    cache.run_or_load("t", fn, rerun=True, x=2)
    assert calls == [1, 2, 2]                # explicit rerun


def test_env_run_and_table():
    res = evaluate_env.run(episodes=3)
    assert [r["venue"] for r in res["venues"]] == ["bar", "company_event", "showcase"]
    assert "| bar |" in evaluate_env.table(res)


def test_per_joke_config_is_reproducible_via_params():
    env = evaluate_env.make_env(n_bits=40, jokes_per_bit=1, fatigue_window=5)
    assert env.n_jokes == 40 and env.p["dynamics"]["fatigue_window"] == 5
    assert evaluate_env.make_env(8, 5, 2).p["dynamics"]["fatigue_window"] == 2  # no leak between envs


def test_policies_run_table_plot():
    res = evaluate_policies.run(shows=3, seeds=2)
    assert len(res["oracle"]["lost_curve"]) == 3
    assert "LinUCB" in evaluate_policies.table(res)
    import matplotlib
    matplotlib.use("Agg")
    assert len(evaluate_policies.plot(res).axes) == 3


def test_shift_run_and_table():
    res = evaluate_shift.run(shows=4, seeds=2, shift_at=2)
    assert set(res["oracle"]) == {"shifted", "steady"}
    assert "LinUCB + react" in evaluate_shift.table(res)


def test_set_length_experiment_holds_total_bits_and_records_curves():
    res = evaluate_shift.run_lengths(lengths=(8, 16), total_bits=64, seeds=2)
    assert res["8"]["shows"] == 8 and res["16"]["shows"] == 4      # same experience, fewer shows
    assert res["16"]["shift_at"] == 8                               # shift at the midpoint
    assert len(res["16"]["curve"]["LinUCB"]) == 16
    assert all(abs(q - 1) < 1e-9 for q in res["8"]["steady_curve"]["oracle"])  # oracle always picks best
    assert "| 16 |" in evaluate_shift.table_lengths(res)


def test_exp5_policy_set_is_pinned():
    res = evaluate_shift.run(shows=4, seeds=1, shift_at=2)
    assert tuple(res) == evaluate_shift.SHIFT_POLICIES  # new policies can't leak into Exp 5


def test_show_state_comparison_rows():
    res = evaluate_shift.run_lengths(lengths=(8,), total_bits=64, seeds=2, names=evaluate_shift.SHOW_STATE_POLICIES)
    rows = evaluate_shift.show_state_deltas(res)
    assert {r["algo"] for r in rows} == {"LinUCB", "LinTS"}
    assert "Gain" in evaluate_shift.table_show_state(res)


def test_pick_kappa_follows_the_preset_rule():
    from sim.evaluate_shift import pick_kappa, two_ts_name
    def length(gains, steadies):
        curve = {"LinUCB": [0.5] * 8}
        steady = {"LinUCB": [0.9] * 8}
        for k, g, st in zip((0.5, 1, 2, 4, 8), gains, steadies):
            curve[two_ts_name(k)] = [0.5 + g] * 8
            steady[two_ts_name(k)] = [0.9 + st] * 8
        return {"shift_at": 4, "curve": curve, "steady_curve": steady}
    res = {"8": length([0.10, 0.08, 0.06, 0.02, 0.01], [-0.05, -0.005, 0.0, 0.0, 0.0]),
           "16": length([0.09, 0.07, 0.05, 0.03, 0.01], [-0.04, 0.0, 0.0, 0.0, 0.0])}
    assert pick_kappa(res) == 1        # 0.5 fails steady; 1 passes both lengths with the larger gain
    none = {"8": length([0.01] * 5, [0.0] * 5)}
    assert pick_kappa(none) is None


def test_behaviour_diagnostic_rates():
    res = evaluate_shift.run_behaviour(names=("LinUCB", "random"), shows=8, seeds=1)
    for r in res.values():
        assert 0 <= r["repeat_rate"] <= 1 and 0 <= r["hurt_pick_rate"] <= 1
    assert "Repeats" in evaluate_shift.table_behaviour(res)
