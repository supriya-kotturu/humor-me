"""Exp 24's calibrated judge with a fake judge model: the few-shot prompt, shuffled orders, the leave-one-out check, the
gate and its bias check, caching, and how a passing judge joins the votes. No network."""
import json
import pathlib
import re
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import llm  # noqa: E402
from sim import compare_generators as cg  # noqa: E402
from sim import judge_generators as jg  # noqa: E402
from tests.test_compare_generators import (BY_ID, MODELS, TAG, FakeFilter, FakeModel, NoWait, data,  # noqa: E402,F401
                                           fresh_safety, run)  # noqa: F401

PREF = {"charlie": 0, "bravo": 1, "alpha": 2}  # this user's taste: claude, flash, flash-lite
BY_TAG = {t: k for k, t in TAG.items()}


class FakeJudge:
    """Stands in for the judge's client: ranks the options of the NEW setting by `score(tag)` (lower is better), or by
    letter order when `by_letter`. Keeps every message it was sent."""
    model = "claude-haiku-5-5"

    def __init__(self, score=lambda tag: PREF[tag], by_letter=False, fail=None):
        self.score, self.by_letter, self.fail = score, by_letter, fail
        self.sent, self.attempts, self.input_tokens, self.output_tokens = [], 0, 0, 0

    def chat(self, system, user, *, json_reply=False, temperature=0.0, validate=None):
        self.sent.append((system, user))
        self.attempts += 1
        if self.fail:
            raise self.fail
        self.input_tokens += 1000
        self.output_tokens += 20
        target = user.split("## New setting to rank")[1]
        options = re.findall(r"Option ([A-C]):\n1\. ([a-z]+?)(?:v\d+)? thread", target)
        letters = [L for L, _ in options] if self.by_letter else [L for L, t in sorted(options, key=lambda o: self.score(o[1]))]
        return json.dumps({"ranking": "".join(letters)})


def prep(n_ballot=30, n=60, models=MODELS, vary=True):
    clients = {m: FakeModel(cg.MODELS[m].model, vary=vary) for m in models}
    run(n=n, models=models, clients=clients)
    cg.export_ballot(models, n_ballot)
    d = cg.run_dir()
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    votes = {p: "".join(sorted(k["letters"], key=lambda c: PREF[TAG[k["letters"][c]]]))
             for p, k in key.items()}
    cg.write_votes(d / "votes.csv", votes)
    return d, key, votes


def judge_run(judge, d, models=MODELS, **kw):
    return jg.run_judge(models, 60, 30, lambda spec: judge, "claude-haiku-5-5", directory=d, pacer=NoWait(), log=lambda s: None, **kw)


def test_the_few_shot_prompt_is_built_from_the_votes_with_blind_letters_only(data):
    d, key, votes = prep()
    by_id = {p["id"]: p for p in cg.prompts(60)}
    ex = jg.user_examples(d, MODELS, key, votes, by_id)
    assert sorted(ex) == sorted(key)  # every ballot prompt had at least two bits shown
    system, user = jg.build_prompt([ex[k] for k in sorted(ex)][:3], jg.block("bar, low energy, mixed audience", {"A": ["x."], "B": ["y."]}))
    for p in sorted(ex)[:3]:
        assert f"Their ranking, best first: {votes[p]}" in user
        for lines in ex[p]["options"].values():
            assert all(line in user for line in lines)
    assert user.count("## Example") == 3 and user.rstrip().endswith("Rank these options as that person would.")
    assert not any(w in (system + user).lower() for w in ("gemini", "claude", "haiku", "flash", "lite"))
    assert "Their ranking" not in user.split("## New setting to rank")[1]


def test_every_prompt_is_judged_in_two_shuffled_orders_and_combined_by_mean_rank(data):
    d, key, votes = prep()
    shown = list(MODELS)
    orders = {tuple(jg.target_letters(f"p{i:02d}", o, shown).values()) for i in range(30) for o in range(jg.ORDERS)}
    assert len(orders) > 2  # not one fixed order
    assert any(jg.target_letters(f"p{i:02d}", 0, shown) != jg.target_letters(f"p{i:02d}", 1, shown) for i in range(30))
    judge = FakeJudge()
    result = judge_run(judge, d)
    assert result["n_calls"] == len(judge.sent) == 2 * (30 + 30)  # leave-one-out and new prompts, two orders each
    # a perfect judge ranks claude 1, flash 2, flash-lite 3 in either order, so the mean rank is the same
    assert all(r == {"claude-haiku-5-5": 1.0, "gemini-flash": 2.0, "gemini-flash-lite": 3.0} for r in result["judge_ranks"].values())


def test_two_orders_cancel_a_judge_that_always_picks_the_first_bit_and_the_bias_is_reported(data):
    d, key, votes = prep()
    result = judge_run(FakeJudge(by_letter=True), d)
    assert result["position_bias"]["all"]["first_top"] == 1.0 and result["position_bias"]["all"]["chance"] == pytest.approx(1 / 3)
    assert result["agreement"]["rate"] < 0.6 and not result["joined"]  # no knowledge of the votes, so no agreement
    ranks = list(result["judge_ranks"].values())
    mean = np.mean([r["gemini-flash"] for r in ranks])
    assert 1.5 <= mean <= 2.5  # position is spread across models rather than favoring one


def test_a_model_with_no_safe_thread_is_ranked_last_by_the_judge_too(data):
    E = llm.LLMError
    clients = {m: FakeModel(cg.MODELS[m].model, vary=True) for m in MODELS}
    clients["gemini-flash"].fail = [None] * 30 + [E("refused by the model")] * 2  # prompts 31 and 32 get no bit from flash
    run(n=60, clients=clients)
    cg.export_ballot(MODELS, 30)
    d = cg.run_dir()
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    cg.write_votes(d / "votes.csv", {p: "".join(sorted(k["letters"], key=lambda c: PREF[TAG[k["letters"][c]]])) for p, k in key.items()})
    result = judge_run(FakeJudge(), d)
    r = result["judge_ranks"]["p30"]
    assert r["gemini-flash"] == 3.0 and r["claude-haiku-5-5"] == 1.0 and r["gemini-flash-lite"] == 2.0
    sent = jg.shown_threads(d, MODELS, "p30", {})
    assert "gemini-flash" not in sent[0] and sent[1] == ["gemini-flash"]


def test_leave_one_out_examples_exclude_the_held_out_prompt(data):
    d, key, votes = prep()
    judge = FakeJudge()
    judge_run(judge, d)
    by_id = {p["id"]: p for p in cg.prompts(60)}
    ex = jg.user_examples(d, MODELS, key, votes, by_id)
    loo = [(s, u) for s, u in judge.sent[:60]]  # the first 60 calls are the leave-one-out ones (30 prompts x 2 orders)
    assert all(u.count("## Example") == 29 for _, u in loo)
    for i, pid in enumerate(sorted(ex)):
        held_text = ex[pid]["options"][next(iter(ex[pid]["options"]))][0]  # a unique line of the held-out prompt
        for _, u in loo[2 * i:2 * i + 2]:
            examples, target = u.split("## New setting to rank")
            assert held_text not in examples and held_text in target  # it is only the thing being ranked
    new = judge.sent[60:]
    assert all(u.count("## Example") == 30 for _, u in new)  # the later prompts see every vote


def test_top_pick_credit_and_agreement_arithmetic():
    shown = ["a", "b", "c"]
    assert jg.top_credit({"a": 1.0, "b": 2.0, "c": 3.0}, shown, "a") == 1.0
    assert jg.top_credit({"a": 1.5, "b": 1.5, "c": 3.0}, shown, "b") == 0.5  # a tie for the top shares the credit
    assert jg.top_credit({"a": 1.5, "b": 1.5, "c": 3.0}, shown, "c") == 0.0
    ag = jg.agreement([1.0] * 20 + [0.0] * 10, [1 / 3] * 30)
    assert ag["rate"] == pytest.approx(2 / 3) and ag["pass"] and ag["ci95"] == cg.wilson(20, 30) and ag["ci95"][0] > ag["chance"]
    assert jg.agreement([1.0] * 18 + [0.0] * 12, [1 / 3] * 30)["pass"]  # exactly 60%, lower end 42% above chance 33%
    assert not jg.agreement([1.0] * 17 + [0.0] * 13, [1 / 3] * 30)["pass"]  # 57%: under the bar
    assert not jg.agreement([1.0] * 18 + [0.0] * 12, [0.5] * 30)["pass"]  # 60% but not clear of a chance of 50%
    two = jg.agreement([1.0] * 20 + [0.0] * 10, [1 / 3] * 15 + [1 / 2] * 15)  # fewer bits shown raise the chance
    assert two["chance"] == pytest.approx(5 / 12)


def test_the_bias_check_blocks_a_model_the_judge_systematically_moves(data):
    models = list(MODELS)
    user = {f"p{i:02d}": {models[0]: 1.0, models[1]: 2.0, models[2]: 3.0} for i in range(30)}

    def shifted(delta):
        return {p: {models[0]: r[models[0]] + delta, models[1]: r[models[1]], models[2]: r[models[2]]} for p, r in user.items()}

    ok = jg.model_bias(shifted(0.2), user, models)
    assert ok["pass"] and ok["models"][models[0]]["mean"] == pytest.approx(0.2)
    bad = jg.model_bias(shifted(0.5), user, models)
    assert not bad["pass"] and bad["models"][models[0]]["blocks"] and not bad["models"][models[1]]["blocks"]
    assert not jg.model_bias(shifted(-0.5), user, models)["pass"]  # favoring a model is as bad as disfavoring it
    agree = jg.agreement([1.0] * 25 + [0.0] * 5, [1 / 3] * 30)
    joined, reason = jg.gate(agree, bad)
    assert not joined and "biased on gemini-flash-lite" in reason and "decide alone" in reason
    assert jg.gate(agree, ok)[0]
    assert "does not clear" in jg.gate(jg.agreement([1.0] * 5 + [0.0] * 25, [1 / 3] * 30), ok)[1]


def test_a_judge_that_agrees_joins_its_rankings_to_the_users_votes(data, monkeypatch):
    monkeypatch.setattr(cg.cache, "METRICS", data / "metrics")
    d, key, votes = prep()
    result = judge_run(FakeJudge(), d)
    assert result["joined"] and result["agreement"]["rate"] == 1.0 and result["bias"]["pass"]
    assert set(result["judge_ranks"]) == {p["id"] for p in cg.prompts(60)[30:]}
    text = cg.judge_path(d).read_text(encoding="utf-8")
    assert "thread " not in text and "line 0" not in text  # numbers and prompt ids only
    res = cg.run(**cg.current_params(60, tuple(MODELS), 30))
    assert (res["n_user"], res["n_judge"], res["n_scored"]) == (30, 30, 60) and res["judge"]["state"] == "joined"
    assert {r["rater"] for r in res["per_prompt"]} == {"user", "judge"} and res["votes"]["n"] == 60 and res["votes_user_only"]["n"] == 30
    assert res["votes"]["mean_rank"]["claude-haiku-5-5"] == 1.0
    assert "judge_ranks" not in res["judge"] and "loo" not in res["judge"]


def test_a_judge_that_disagrees_is_reported_and_the_users_votes_decide_alone(data, monkeypatch):
    monkeypatch.setattr(cg.cache, "METRICS", data / "metrics")
    d, key, votes = prep()
    result = judge_run(FakeJudge(score=lambda tag: -PREF[tag]), d)  # the opposite taste
    assert not result["joined"] and result["agreement"]["rate"] == 0.0 and "decide alone" in result["reason"]
    res = cg.run(**cg.current_params(60, tuple(MODELS), 30))
    assert (res["n_judge"], res["n_scored"], res["judge"]["state"]) == (0, 30, "not joined")


def test_a_vote_changed_after_the_judge_ran_makes_its_result_stale(data):
    d, key, votes = prep()
    judge_run(FakeJudge(), d)
    assert jg.cg.judge_ranks(d, MODELS, 30)[0]
    cg.write_votes(d / "votes.csv", {**votes, "p00": votes["p00"][::-1]})
    ranks, summary = cg.judge_ranks(d, MODELS, 30)
    assert ranks == {} and summary["state"] == "stale"


def test_judge_replies_are_cached_and_a_changed_vote_asks_again(data):
    d, key, votes = prep()
    first = FakeJudge()
    judge_run(first, d)
    again = FakeJudge()
    judge_run(again, d)
    assert len(first.sent) == 120 and len(again.sent) == 0
    cg.write_votes(d / "votes.csv", {**votes, "p00": votes["p00"][::-1]})
    third = FakeJudge()
    judge_run(third, d)
    assert 0 < len(third.sent) < 120 and len(third.sent) >= 58  # p00 is an example for everyone else
    assert not (jg.cache_file(d, "claude-haiku-5-5", "loo", "p05", 0, [], "x") == jg.cache_file(d, "gemini-flash", "loo", "p05", 0, [], "x"))


def test_the_judge_waits_for_every_vote(data):
    d, key, votes = prep()
    cg.write_votes(d / "votes.csv", {**votes, "p03": ""})
    with pytest.raises(SystemExit, match="vote on every ballot prompt"):
        judge_run(FakeJudge(), d)


def test_a_config_failure_on_the_judges_first_call_stops_the_run_and_caches_nothing(data):
    d, key, votes = prep()
    with pytest.raises(cg.ConfigError):
        judge_run(FakeJudge(fail=llm.LLMError("http 401", 401)), d)
    assert not (d / "judge_cache").exists() and not cg.judge_path(d).exists()


def test_an_outage_is_not_cached_as_the_judges_answer(data):
    d, key, votes = prep()
    judge = FakeJudge()
    judge_run(judge, d)  # a good run first, then wipe one reply and break the judge
    cache = sorted((d / "judge_cache").rglob("new-p31-o0-*.json"))
    assert len(cache) == 1
    cache[0].unlink()
    broken = FakeJudge(fail=llm.LLMError("http 503", 503, retryable=True))
    result = judge_run(broken, d)
    assert not list((d / "judge_cache").rglob("new-p31-o0-*.json"))  # infra failures are asked again, not remembered
    assert result["n_failed_calls"] == 1
