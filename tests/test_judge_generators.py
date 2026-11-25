"""Exp 24's calibrated judge with a fake judge model: the few-shot prompt, rotated orders, the leave-one-out check, the gate
(top pick, pairwise, per-model bias, incomplete runs), caching, the Explorer bundle, and how a passing judge joins the
votes. No network.

FakeJudge ranks by the tag the fake generator put in the bit's text (alpha, bravo, charlie), not by anything funny: these
tests check the plumbing and the arithmetic, they do not measure whether a real judge has taste."""
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
from tests.test_compare_generators import (MODELS, TAG, FakeModel, NoWait, data,  # noqa: E402,F401
                                           fresh_safety, run)  # noqa: F401

PREF = {"charlie": 0, "bravo": 1, "alpha": 2}  # this user's taste: claude, flash, flash-lite
JUDGE = "claude-haiku-5-5"  # an API judge, so the fake client stands in for it


class FakeJudge:
    """Ranks the options of the NEW setting by `score(tag)` (lower is better), or by letter order when `by_letter`. Keeps every
    message it was sent. `fail` is an exception to raise, or a callable(user) -> exception or None."""
    model = JUDGE

    def __init__(self, score=lambda tag: PREF[tag], by_letter=False, fail=None):
        self.score, self.by_letter, self.fail = score, by_letter, fail
        self.sent, self.attempts, self.input_tokens, self.output_tokens = [], 0, 0, 0

    def chat(self, system, user, *, json_reply=False, temperature=0.0, validate=None):
        self.sent.append((system, user))
        self.attempts += 1
        err = self.fail(user) if callable(self.fail) else self.fail
        if err:
            raise err
        self.input_tokens += 1000
        self.output_tokens += 20
        return json.dumps({"ranking": rank_by(user, self.score, self.by_letter)})


def rank_by(user, score, by_letter=False):
    target = user.split("## New setting to rank")[1]
    options = re.findall(r"Option ([A-C]):\n1\. ([a-z]+?)(?:v\d+)? thread", target)
    return "".join(L for L, _ in options) if by_letter else "".join(L for L, t in sorted(options, key=lambda o: score(o[1])))


def prep(n_ballot=30, n=60, models=MODELS, vary=True, clients=None):
    clients = clients if clients is not None else {m: FakeModel(cg.MODELS[m].model, vary=vary) for m in models}
    run(n=n, models=models, clients=clients)
    cg.export_ballot(models, n_ballot)
    d = cg.run_dir()
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    votes = {p: "".join(sorted(k["letters"], key=lambda c: PREF[TAG[k["letters"][c]]])) for p, k in key.items()}
    cg.write_votes(d / "votes.csv", votes)
    return d, key, votes


def judge_run(judge, d, models=MODELS, **kw):
    return jg.run_judge(models, 60, 30, lambda spec: judge, JUDGE, directory=d, pacer=NoWait(), log=lambda s: None, **kw)


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
    assert "new letters" not in system and "reused" in system  # the letters repeat from one setting to the next


def test_user_examples_with_a_model_subset_keep_the_users_order_among_the_rest(data):
    d, key, votes = prep()
    subset = ["gemini-flash-lite", "claude-haiku-5-5"]
    ex = jg.user_examples(d, subset, key, votes, {p["id"]: p for p in cg.prompts(60)})
    assert sorted(ex) == sorted(key)
    for pid, e in ex.items():
        letters = key[pid]["letters"]
        assert set(e["options"]) == {c for c, m in letters.items() if m in subset} and sorted(e["ranking"]) == sorted(e["options"])
        assert e["ranking"] == "".join(c for c in votes[pid] if letters[c] in subset)


def test_every_prompt_is_judged_once_per_position_and_each_bit_is_first_exactly_once():
    for i in range(60):
        pid = f"p{i:02d}"
        for shown in (["a", "b", "c"], ["a", "b"]):
            orders = jg.orders_for(pid, shown)
            assert len(orders) == len(shown)
            firsts = [o["A"] for o in orders]
            assert sorted(firsts) == sorted(shown)  # the first-shown bit differs in every order
            for m in shown:  # and every bit takes every position once: position cancels in the mean
                assert sorted(L for o in orders for L, mm in o.items() if mm == m) == sorted(cg.LETTERS[:len(shown)])
    assert any(jg.orders_for(f"p{i:02d}", ["a", "b", "c"])[0]["A"] != jg.orders_for("p00", ["a", "b", "c"])[0]["A"] for i in range(1, 30))


def test_a_perfect_judge_ranks_the_same_in_every_order_and_the_mean_is_exact(data):
    d, key, votes = prep()
    judge = FakeJudge()
    result = judge_run(judge, d)
    assert result["n_calls"] == len(judge.sent) == 3 * (30 + 30)  # three orders for each of 30 leave-one-out and 30 later prompts
    assert all(r == {"claude-haiku-5-5": 1.0, "gemini-flash": 2.0, "gemini-flash-lite": 3.0} for r in result["judge_ranks"].values())
    assert result["joined"] and not result["incomplete"] and result["pairwise"]["rate"] == 1.0


def test_a_judge_that_always_picks_the_first_bit_is_exposed_and_its_position_cancels(data):
    d, key, votes = prep()
    result = judge_run(FakeJudge(by_letter=True), d)
    assert result["position_bias"]["all"]["first_top"] == 1.0 and result["position_bias"]["all"]["chance"] == pytest.approx(1 / 3)
    assert all(r == {m: 2.0 for m in MODELS} for r in result["judge_ranks"].values())  # every bit was first once: no preference left
    assert result["agreement"]["rate"] == pytest.approx(1 / 3) and not result["joined"]


def test_a_model_with_no_safe_thread_is_ranked_last_by_the_judge_too(data):
    E = llm.LLMError
    clients = {m: FakeModel(cg.MODELS[m].model, vary=True) for m in MODELS}
    clients["gemini-flash"].fail = [None] * 30 + [E("refused by the model")] * 2  # prompts 31 and 32 get no bit from flash
    d, key, votes = prep(clients=clients)
    result = judge_run(FakeJudge(), d)
    r = result["judge_ranks"]["p30"]
    assert r["gemini-flash"] == 3.0 and r["claude-haiku-5-5"] == 1.0 and r["gemini-flash-lite"] == 2.0
    shown, last = jg.shown_threads(d, MODELS, "p30", {})
    assert "gemini-flash" not in shown and last == ["gemini-flash"]


def test_leave_one_out_examples_exclude_the_held_out_prompt(data):
    d, key, votes = prep()
    judge = FakeJudge()
    judge_run(judge, d)
    ex = jg.user_examples(d, MODELS, key, votes, {p["id"]: p for p in cg.prompts(60)})
    loo = judge.sent[:90]  # 30 prompts x 3 orders, in sorted prompt order
    assert all(u.count("## Example") == 29 for _, u in loo)
    for i, pid in enumerate(sorted(ex)):
        held_text = ex[pid]["options"][next(iter(ex[pid]["options"]))][0]  # a unique line of the held-out prompt
        for _, u in loo[3 * i:3 * i + 3]:
            examples, target = u.split("## New setting to rank")
            assert held_text not in examples and held_text in target
    assert all(u.count("## Example") == 30 for _, u in judge.sent[90:])  # the later prompts see every vote


def test_the_cache_key_hashes_everything_the_judge_is_shown(data):
    d, key, votes = prep()
    judge_run(FakeJudge(), d)
    cached = {p.name for p in (d / "judge_cache" / JUDGE).iterdir()}
    pl = jg.plan(d, MODELS, 60, 30, JUDGE)
    assert {c["id"] + ".json" for c in pl["calls"]} == cached  # one file per call, named by the hash of system + user
    a = pl["calls"][0]
    assert jg.call_id(JUDGE, a["kind"], a["pid"], a["order"], a["system"], a["user"]) == a["id"]
    assert jg.call_id(JUDGE, a["kind"], a["pid"], a["order"], a["system"], a["user"] + " ") != a["id"]
    assert jg.call_id("other-judge", a["kind"], a["pid"], a["order"], a["system"], a["user"]) != a["id"]
    # a changed bit in a later prompt re-asks exactly that prompt, in every order, and nothing else
    path = d / "gemini-flash-lite" / "p45.json"
    rec = json.loads(path.read_text(encoding="utf-8"))
    att = cg.final(rec)
    t = cg.playable_thread(rec)
    att["threads"][t][0] += " (edited)"
    path.write_text(json.dumps(rec), encoding="utf-8")
    again = FakeJudge()
    judge_run(again, d)
    assert len(again.sent) == 3 and all("(edited)" in u.split("## New setting to rank")[1] for _, u in again.sent)


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
    assert jg.agreement([1.0] * 20 + [0.0] * 10, [1 / 3] * 15 + [1 / 2] * 15)["chance"] == pytest.approx(5 / 12)


def test_pairwise_agreement_arithmetic():
    models = ["a", "b", "c"]
    user = {f"p{i:02d}": {"a": 1.0, "b": 2.0, "c": 3.0} for i in range(30)}
    shown = {p: models for p in user}
    same = jg.pairwise_agreement(user, user, shown)
    assert (same["n"], same["rate"], same["pass"]) == (90, 1.0, True)
    reverse = {p: {"a": 3.0, "b": 2.0, "c": 1.0} for p in user}
    assert jg.pairwise_agreement(reverse, user, shown)["rate"] == 0.0  # every pair ordered the other way
    tied = {p: {"a": 2.0, "b": 2.0, "c": 2.0} for p in user}
    t = jg.pairwise_agreement(tied, user, shown)
    assert t["rate"] == 0.5 and not t["pass"]  # a judge that ties everything earns exactly chance
    assert t["ci95"] == cg.wilson(45, 90)


def test_the_bias_check_blocks_a_model_the_judge_systematically_moves_with_noisy_ranks(data):
    models = list(MODELS)
    rng = np.random.default_rng(3)
    user = {}
    for i in range(30):  # a user whose rankings differ prompt to prompt
        perm = rng.permutation(3) + 1.0
        user[f"p{i:02d}"] = dict(zip(models, perm))
    noise = {p: {m: rng.normal(0, 0.5) for m in models} for p in user}

    def judge(shift):
        return {p: {m: r[m] + noise[p][m] + (shift if m == models[0] else 0.0) for m in models} for p, r in user.items()}

    none, small, big = (jg.model_bias(judge(s), user, models) for s in (0.0, 0.2, 0.7))
    for b in (none, small, big):  # the bootstrap interval has real width here
        assert all(v["ci95"][1] - v["ci95"][0] > 0.1 for v in b["models"].values())
    assert none["pass"] and small["pass"]
    assert not big["pass"] and big["models"][models[0]]["blocks"] and big["models"][models[0]]["ci95"][0] > 0.3
    assert not jg.model_bias(judge(-0.7), user, models)["pass"]  # favoring a model is as bad as disfavoring it
    agree, pair = jg.agreement([1.0] * 25 + [0.0] * 5, [1 / 3] * 30), {"pass": True, "rate": 1.0, "ci95": [0.9, 1.0]}
    joined, reason = jg.gate(agree, pair, big)
    assert not joined and f"biased on {models[0]}" in reason and "decide alone" in reason
    assert jg.gate(agree, pair, small)[0]
    assert "does not clear" in jg.gate(jg.agreement([1.0] * 5 + [0.0] * 25, [1 / 3] * 30), pair, small)[1]
    assert "pairwise" in jg.gate(agree, {"pass": False, "rate": 0.5, "ci95": [0.4, 0.6]}, small)[1]


def test_what_the_gate_decides_under_an_outage_and_a_bad_key(data, monkeypatch):
    d, key, votes = prep()
    E = llm.LLMError
    # a judge that works for leave-one-out and then goes down for the later prompts: complete agreement, yet nothing joins
    down = FakeJudge(fail=lambda user: E("http 503", 503, retryable=True) if user.count("## Example") == 30 else None)
    result = judge_run(down, d)
    assert result["agreement"]["rate"] == 1.0 and result["incomplete"] and result["n_unanswered"] == 90
    assert not result["joined"] and "incomplete" in result["reason"]
    ranks, summary = cg.judge_ranks(d, MODELS, 30, 60)
    assert ranks == {} and summary["state"] == "incomplete"
    assert jg.summary(result).startswith("judge") and "INCOMPLETE" in jg.summary(result)
    # the outage ends: the answered calls come from the cache, only the missing ones are asked, and the result is current
    healthy = FakeJudge()
    result = judge_run(healthy, d)
    assert len(healthy.sent) == 90 and not result["incomplete"] and result["joined"]
    assert cg.judge_ranks(d, MODELS, 30, 60)[0]
    # nothing cached and every call fails: the gate is not decided on silence
    import shutil
    shutil.rmtree(d / "judge_cache")
    (d / "judge_result.json").unlink()
    res = jg.run_judge(MODELS, 60, 30, lambda s: jg.ReplayClient(), JUDGE, directory=d, pacer=NoWait(), log=lambda s: None)
    assert res["incomplete"] and res["n_unanswered"] == res["n_calls"] and not res["joined"]
    assert res["agreement"]["n"] == 30 and res["agreement"]["rate"] == 0.0  # unjudged prompts are misses, but the result is flagged anyway


def test_the_first_call_guard_survives_cache_hits_and_is_used_up_by_a_live_call(data):
    d, key, votes = prep()
    judge_run(FakeJudge(), d)
    next(p for p in (d / "judge_cache" / JUDGE).glob("new-p45-o0-*.json")).unlink()  # one call will have to go live
    with pytest.raises(cg.ConfigError):  # 179 cache hits did not use the guard up
        judge_run(FakeJudge(fail=llm.LLMError("http 401", 401)), d)
    assert not list((d / "judge_cache" / JUDGE).glob("new-p45-o0-*.json"))
    # once a live call has been answered, a later 401 is data about that call, not a reason to stop the run
    first = {JUDGE}
    pl = jg.plan(d, MODELS, 60, 30, JUDGE)
    c0, c1 = [c for c in pl["calls"] if c["kind"] == "new"][:2]
    cg_dir = d
    for c in (c0, c1):
        jg.cache_path(cg_dir, JUDGE, c["id"]).unlink(missing_ok=True)
    jg.ask(FakeJudge(), jg.JUDGES[JUDGE], c0, d, NoWait(), first)
    assert JUDGE not in first
    rec = jg.ask(FakeJudge(fail=llm.LLMError("http 401", 401)), jg.JUDGES[JUDGE], c1, d, NoWait(), first)
    assert not rec["ok"] and rec["kind"] == "config"


def test_a_judge_that_agrees_joins_its_rankings_to_the_users_votes(data, monkeypatch):
    monkeypatch.setattr(cg.cache, "METRICS", data / "metrics")
    d, key, votes = prep()
    result = judge_run(FakeJudge(), d)
    assert result["joined"] and result["agreement"]["rate"] == 1.0 and result["bias"]["pass"] and result["pairwise"]["pass"]
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


def test_a_result_is_stale_when_votes_prompt_scheme_or_prompt_count_change(data, monkeypatch):
    d, key, votes = prep()
    judge_run(FakeJudge(), d)
    assert cg.judge_ranks(d, MODELS, 30, 60)[0]
    assert cg.judge_ranks(d, MODELS, 30, 72)[1]["state"] == "stale"  # the run now generates more prompts
    monkeypatch.setattr(jg, "SYSTEM", jg.SYSTEM + " changed")
    assert cg.judge_ranks(d, MODELS, 30, 60)[1]["state"] == "stale"
    monkeypatch.undo()
    monkeypatch.setattr(jg, "SCHEME", "something-else")
    assert cg.judge_ranks(d, MODELS, 30, 60)[1]["state"] == "stale"
    monkeypatch.undo()
    cg.write_votes(d / "votes.csv", {**votes, "p00": votes["p00"][::-1]})
    ranks, summary = cg.judge_ranks(d, MODELS, 30, 60)
    assert ranks == {} and summary["state"] == "stale"


def test_judge_replies_are_cached_and_a_changed_vote_asks_again(data):
    d, key, votes = prep()
    first, again = FakeJudge(), FakeJudge()
    judge_run(first, d)
    judge_run(again, d)
    assert len(first.sent) == 180 and len(again.sent) == 0
    cg.write_votes(d / "votes.csv", {**votes, "p00": votes["p00"][::-1]})
    third = FakeJudge()
    judge_run(third, d)
    assert 170 < len(third.sent) < 180  # p00 is an example everywhere but in its own leave-one-out calls


def test_the_judge_waits_for_every_vote_and_every_record(data):
    d, key, votes = prep()
    cg.write_votes(d / "votes.csv", {**votes, "p03": ""})
    with pytest.raises(SystemExit, match="vote on every ballot prompt"):
        judge_run(FakeJudge(), d)
    cg.write_votes(d / "votes.csv", votes)
    (d / "gemini-flash" / "p50.json").unlink()
    with pytest.raises(SystemExit, match="no record for some model"):
        judge_run(FakeJudge(), d)


def test_an_outage_is_not_cached_as_the_judges_answer(data):
    d, key, votes = prep()
    judge_run(FakeJudge(), d)
    victim = next((d / "judge_cache" / JUDGE).glob("new-p31-o0-*.json"))
    victim.unlink()
    result = judge_run(FakeJudge(fail=llm.LLMError("http 503", 503, retryable=True)), d)
    assert not list((d / "judge_cache" / JUDGE).glob("new-p31-o0-*.json")) and result["n_unanswered"] == 1 and result["incomplete"]


def test_a_reply_that_is_not_a_ranking_is_the_judges_miss_not_an_outage(data):
    d, key, votes = prep()

    class Rambler(FakeJudge):
        def chat(self, system, user, **kw):
            self.sent.append((system, user))
            return json.dumps({"ranking": "I like all of them"})

    result = judge_run(Rambler(), d)
    assert not result["incomplete"] and result["n_failed_calls"] == result["n_calls"] and result["agreement"]["rate"] == 0.0
    assert not result["joined"] and len(result["unjudged_new"]) == 30


def test_the_explorer_bundle_names_no_model_and_the_replies_come_back_through_the_manifest(data):
    d, key, votes = prep()
    out = d.parent / "judge_bundle"
    info = jg.export_bundle(MODELS, 60, 30, JUDGE, out, directory=d)
    assert info["calls"] == 180 and (info["loo"], info["new"]) == (90, 90)
    text = (out / "calls.jsonl").read_text(encoding="utf-8")
    assert not any(w in text.lower() for w in ("gemini", "claude", "haiku", "flash", "lite", "key.json"))
    assert [p.name for p in out.iterdir()] == ["calls.jsonl"]  # the manifest stays in the run folder
    assert jg.manifest_path(d).exists()
    rows = [json.loads(line) for line in text.splitlines()]
    assert all(set(r) == {"id", "system", "user", "letters"} and 2 <= len(r["letters"]) <= 3 for r in rows)
    # the cluster, simulated: rank each call the way FakeJudge does and write replies.jsonl
    replies = d.parent / "replies.jsonl"
    with open(replies, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "ranking": rank_by(r["user"], lambda t: PREF[t])}) + "\n")
        f.write(json.dumps({"id": "loo-p99-o0-deadbeef", "ranking": "ABC"}) + "\n")  # an id from some other bundle
        f.write(json.dumps({"id": rows[0]["id"], "ranking": "ZZZ"}) + "\n")  # not a ranking of its letters
    assert jg.import_replies(replies, JUDGE, d) == {"imported": 180, "unknown_ids": 1, "not_a_ranking": 1}
    result = jg.run_judge(MODELS, 60, 30, lambda s: jg.ReplayClient(), JUDGE, directory=d, pacer=NoWait(), log=lambda s: None)
    assert not result["incomplete"] and result["joined"] and result["agreement"]["rate"] == 1.0
    # a reply file with some calls missing leaves the run incomplete: nothing is made up
    rows_missing = rows[:-5]
    short = d.parent / "short.jsonl"
    short.write_text("".join(json.dumps({"id": r["id"], "ranking": rank_by(r["user"], lambda t: PREF[t])}) + "\n" for r in rows_missing), encoding="utf-8")
    import shutil
    shutil.rmtree(d / "judge_cache")
    jg.import_replies(short, JUDGE, d)
    res = jg.run_judge(MODELS, 60, 30, lambda s: jg.ReplayClient(), JUDGE, directory=d, pacer=NoWait(), log=lambda s: None)
    assert res["incomplete"] and res["n_unanswered"] == 5 and not res["joined"]


def test_the_smoke_bundle_holds_three_prompts_of_leave_one_out_calls(data):
    d, key, votes = prep()
    info = jg.export_bundle(MODELS, 60, 30, JUDGE, d.parent / "smoke", directory=d, smoke=3)
    assert info == {"calls": 9, "loo": 9, "new": 0, "bundle": str(d.parent / "smoke" / "calls.jsonl")}
    rows = [json.loads(line) for line in (d.parent / "smoke" / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len({r["id"].split("-")[1] for r in rows}) == 3


def test_the_default_judge_runs_on_explorer_and_needs_no_live_client():
    assert jg.JUDGE_MODEL == "qwen2.5-7b-instruct" and jg.JUDGES[jg.JUDGE_MODEL].provider == "explorer"
    assert jg.JUDGES[jg.JUDGE_MODEL].model == "Qwen/Qwen2.5-7B-Instruct"
    assert {"claude-haiku-5-5", "gemini-flash-lite"} <= set(jg.JUDGES)
    with pytest.raises(SystemExit, match="runs on Explorer"):
        jg.main(MODELS, 60, 30, None)
