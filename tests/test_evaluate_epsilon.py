"""Exp 18 helpers: gap scaling, pairing and the pass rule; short careers only."""
import numpy as np
import pytest

from sim.env import load_audience
from sim.evaluate_epsilon import career_laughs, criteria, scale_gaps, variants


def test_scale_gaps_pivots_on_the_mean_and_touches_only_base():
    aud = load_audience()
    mean = np.mean(list(aud["base"].values()))
    for s in (0.5, 2.0):
        out = scale_gaps(aud, s)
        assert np.mean(list(out["base"].values())) == pytest.approx(mean)
        assert {m: v - mean for m, v in out["base"].items()} == pytest.approx(
            {m: s * (v - mean) for m, v in aud["base"].items()})
        assert {k: out[k] for k in out if k != "base"} == {k: aud[k] for k in aud if k != "base"}
    assert scale_gaps(aud, 1.0)["base"] == pytest.approx(aud["base"])


def test_six_variants_and_the_input_audience_is_not_mutated():
    before = load_audience()["base"]
    assert len(variants()) == 6
    assert load_audience()["base"] == before


def test_same_seed_gives_the_same_career_and_eps_zero_never_explores():
    aud = load_audience()
    assert career_laughs(aud, 2, 5, 0.1) == career_laughs(aud, 2, 5, 0.1)
    assert career_laughs(aud, 2, 5, 0.0)[1] == 0


def _res(mean, half):
    return {"variants": {"v": {"5": {"0.2": {"diff": [mean, half]}}}}}


def test_criteria_judge_the_far_end_of_the_ci():
    assert criteria(_res(-0.5, 0.4))["all_pass"]
    assert criteria(_res(-1.0, 0.0))["all_pass"]  # exactly at the bar
    assert not criteria(_res(-0.5, 0.6))["all_pass"]  # the mean is fine, the far end is not
