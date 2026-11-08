"""Safety filter: fails closed, teens are strictly tighter, near-miss words pass, and Exp 19's bookkeeping adds up.

The recall numbers themselves live in notebooks/decisions.ipynb (Exp 19); a missed pass bar is a reported result,
not a test failure.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import safety  # noqa: E402
from sim import cache, evaluate_safety  # noqa: E402

RED = evaluate_safety.load(evaluate_safety.RED_TEAM)
CLEAN = evaluate_safety.load(evaluate_safety.CLEAN)


def test_any_error_inside_counts_as_rejected(monkeypatch):
    def boom(text, age):
        raise RuntimeError("rule bug")

    monkeypatch.setattr(safety, "_check", boom)
    verdict = safety.check("I'm a robot.", "mixed")
    assert not verdict.ok and verdict.reason == "error:RuntimeError"


@pytest.mark.parametrize("bad", [None, 42, b"bytes", "", "   "])
def test_non_text_and_empty_input_is_rejected(bad):
    assert not safety.check(bad, "mixed").ok


def test_unreadable_blocklist_rejects_everything(monkeypatch):
    monkeypatch.setattr(safety, "BLOCKLIST_FILE", pathlib.Path("does-not-exist.txt"))
    safety._blocklist.cache_clear()
    try:
        assert not safety.check("I'm a robot.", "mixed").ok
    finally:
        monkeypatch.undo()
        safety._blocklist.cache_clear()
    assert safety.check("I'm a robot.", "mixed").ok


def test_unknown_age_gets_the_strict_rules():
    line = "Time for a cold beer, am I right?"
    assert safety.check(line, "mixed").ok
    assert not safety.check(line, "teens").ok
    assert not safety.check(line, "teen").ok  # a typo must not loosen the filter


def test_teens_are_strictly_tighter_on_every_fixture_line():
    lines = [r["text"] for r in RED] + [c["text"] for c in CLEAN]
    looser = [t for t in lines if safety.check(t, "teens").ok and not safety.check(t, "mixed").ok]
    tighter = [t for t in lines if not safety.check(t, "teens").ok and safety.check(t, "mixed").ok]
    assert looser == []
    assert tighter, "teens rules add nothing over the standard rules"


def test_alcohol_and_sexual_lines_pass_for_adults_but_not_teens():
    teen_lines = [r for r in RED if r["cat"] in ("teens_alcohol", "teens_sexual")]
    assert all(not safety.check(r["text"], "teens").ok for r in teen_lines if r["split"] == "tune")
    # Adults hear most of them: the strict rules are the only thing standing in their way.
    assert sum(safety.check(r["text"], "mixed").ok for r in teen_lines) >= 10


@pytest.mark.parametrize("line", [
    "My weight in the database is mostly typos.",
    "I'm a bit of a hardware person, emotionally.",
    "I'm short on memory and long on regret.",
    "On Monday I'm fine; by Tuesday I'm buffering.",
])
def test_the_robots_own_body_and_safe_near_miss_words_pass(line):
    assert safety.check(line, "mixed").ok and safety.check(line, "teens").ok


@pytest.mark.parametrize("line", ["You're fat.", "Her weight is not my business.", "F4t jokes, you said.", "U g l y shirt."])
def test_aiming_the_same_words_at_somebody_else_is_rejected(line):
    assert not safety.check(line, "mixed").ok


def test_blocklist_mechanism_with_an_injected_placeholder_word(monkeypatch):
    monkeypatch.setattr(safety, "_blocklist", lambda: ("zzslurzz",))
    assert not safety.check("What a zzslurzz crowd.", "mixed").ok
    assert not safety.check("What a ZZSLURZZ crowd.", "mixed").ok
    assert safety.check("What a lovely crowd.", "mixed").ok


def test_red_team_file_meets_its_own_spec():
    assert len(RED) >= 100
    assert len({r["id"] for r in RED}) == len(RED)
    held = [r for r in RED if r["split"] == "held"]
    assert len(held) / len(RED) >= 0.30
    for cat in {r["cat"] for r in RED}:
        assert any(r["cat"] == cat and r["split"] == "held" for r in RED), cat
    assert {"appearance", "bodies", "protected", "named_people", "teens_alcohol", "teens_sexual"} <= {r["cat"] for r in RED}
    assert any(not r["hard"] for r in RED)
    assert len(CLEAN) >= 60


def test_experiment_counts_add_up(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "METRICS", tmp_path)
    res = cache.run_or_load("safety_filter", evaluate_safety.run, fixtures=evaluate_safety.digest())
    assert res["n_red_team"] == len(RED) and res["n_clean"] == len(CLEAN)
    hard = res["hard_overall"]
    assert hard["tune"]["n"] + hard["held"]["n"] == res["n_hard"]
    assert hard["tune"]["caught"] + hard["held"]["caught"] + len(res["missed_hard"]) == res["n_hard"]
    assert res["teens"]["looser"] == []
    assert res["criteria"]["catch_pass"] == (res["criteria"]["held_catch"] >= 1.0)


def test_stale_fixture_digest_is_refused():
    with pytest.raises(AssertionError):
        evaluate_safety.run(fixtures="not-the-digest")
