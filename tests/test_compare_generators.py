"""Exp 24 harness with fake providers: pairing, resume, pacing, the blind ballot, scoring and the bar. No network."""
import json
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import llm, safety  # noqa: E402
from sim import compare_generators as cg  # noqa: E402

MODELS = ["gemini-flash-lite", "gemini-flash", "claude-haiku-5-5"]
TAG = {"gemini-flash-lite": "alpha", "gemini-flash": "bravo", "claude-haiku-5-5": "charlie"}  # text must not name a model
BY_ID = {cg.MODELS[k].model: t for k, t in TAG.items()}


def reply(tag):
    return json.dumps({"premises": ["a", "b", "c"], "threads": [[f"{tag} thread {i} line {j}." for j in range(5)] for i in range(3)]})


class FakeModel:
    """A provider: answers with a canned bit, or raises what `fail` says; counts attempts and tokens like the real ones."""

    def __init__(self, model, fail=None, tokens=(1000, 2000)):
        self.model, self.fail, self.tokens = model, fail, tokens
        self.attempts = self.input_tokens = self.output_tokens = 0
        self.calls = 0

    def chat(self, system, user, *, json_reply=False, temperature=0.0):
        self.attempts += 1
        self.calls += 1
        if self.fail:
            raise self.fail
        self.input_tokens += self.tokens[0]
        self.output_tokens += self.tokens[1]
        return reply(BY_ID[self.model])


class FakeFilter:
    """The safety filter's model: approves every line, or errors like a failed provider call."""

    def __init__(self, error=None, reject=()):
        self.error, self.reject, self.calls = error, reject, 0

    def chat(self, system, user, *, json_reply=False, temperature=0.0):
        self.calls += 1
        if self.error:
            raise self.error
        items = json.loads(user)
        return json.dumps({"verdicts": [{"id": i["id"], "ok": not any(r in i["text"] for r in self.reject), "reason": "r"}
                                        for i in items]})


class NoWait(cg.Pacer):
    def __init__(self):
        super().__init__(sleep=lambda s: None)


@pytest.fixture(autouse=True)
def fresh_safety():
    safety.clear_cache()


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "DATA", tmp_path)
    return tmp_path


def run(n=6, models=MODELS, clients=None, filt=None, **kw):
    clients = {} if clients is None else clients
    done = cg.run_comparison(n, models, lambda spec: clients.setdefault(spec.key, FakeModel(spec.model)),
                             filt or FakeFilter(), pacer=NoWait(), log=lambda s: None, **kw)
    return done, clients


def test_prompts_are_seeded_prefix_stable_balanced_and_distinct():
    p30 = cg.prompts(30)
    assert p30 == cg.prompts(30) and cg.prompts(36)[:30] == p30
    assert {m: sum(p["mode"] == m for p in p30) for m in cg.MODE_ORDER} == {m: 5 for m in cg.MODE_ORDER}
    assert len({(p["mode"], *p["context"].values()) for p in p30}) == 30
    assert all(p["context"]["venue"] in cg.VENUES and p["context"]["age"] in cg.AGES for p in p30)


def test_digest_follows_the_generator_prompt(monkeypatch):
    before = cg.gen_digest()
    monkeypatch.setattr(cg.generate, "SYSTEM", cg.generate.SYSTEM + " changed")
    assert cg.gen_digest() != before


def test_every_model_gets_the_same_prompts_and_a_record_per_prompt(data):
    done, _ = run(n=4)
    assert done == {"generated": 12, "judged": 12, "cached": 0}
    d = cg.run_dir()
    for m in MODELS:
        recs = [cg.read_record(d / m / f"{p['id']}.json") for p in cg.prompts(4)]
        assert [r["prompt"] for r in recs] == ["p00", "p01", "p02", "p03"]
        assert all(r["ok"] and len(r["threads"]) == 3 and all(len(t) == 5 for t in r["threads"]) for r in recs)
        assert all(r["input_tokens"] == 1000 and r["output_tokens"] == 2000 and r["attempts"] == 1 for r in recs)


def test_resume_calls_nothing_again_and_extending_adds_only_new_prompts(data):
    run(n=3)
    clients = {}
    done, _ = run(n=5, clients=clients)
    assert done["cached"] == 9 and done["generated"] == 6
    assert all(c.calls == 2 for c in clients.values())


def test_a_failed_call_is_data_and_cached_until_redo_failed(data):
    bad = {"claude-haiku-5-5": FakeModel("claude-haiku-5-5", fail=llm.LLMError("http 529", 529, retryable=True))}
    run(n=2, clients=bad)
    rec = cg.read_record(cg.run_dir() / "claude-haiku-5-5" / "p00.json")
    assert not rec["ok"] and "529" in rec["reason"] and rec["threads"] is None and rec["safety"] is None
    done, _ = run(n=2, clients=bad)
    assert done["generated"] == 0
    healed = {"claude-haiku-5-5": FakeModel("claude-haiku-5-5")}
    done, _ = run(n=2, clients=healed, redo_failed=True)
    assert done["generated"] == 2 and cg.read_record(cg.run_dir() / "claude-haiku-5-5" / "p00.json")["ok"]


def test_a_credential_failure_stops_the_run_and_caches_nothing(data):
    bad = {"claude-haiku-5-5": FakeModel("claude-haiku-5-5", fail=llm.LLMError("no anthropic credentials", None))}
    with pytest.raises(cg.CredentialError):
        run(n=2, models=["claude-haiku-5-5"], clients=bad)
    assert not (cg.run_dir() / "claude-haiku-5-5").exists()


def test_a_filter_outage_is_retried_on_resume_not_scored_as_rejections(data):
    run(n=1, models=["gemini-flash-lite"], filt=FakeFilter(error=llm.LLMError("http 503", 503)))
    rec = cg.read_record(cg.run_dir() / "gemini-flash-lite" / "p00.json")
    assert rec["safety_infra_error"] and cg.playable_thread(rec) is None
    safety.clear_cache()
    done, _ = run(n=1, models=["gemini-flash-lite"], filt=FakeFilter())
    rec = cg.read_record(cg.run_dir() / "gemini-flash-lite" / "p00.json")
    assert done["generated"] == 0 and done["judged"] == 1 and not rec["safety_infra_error"] and rec["safety_tries"] == 2


def test_latency_is_wall_time_around_the_call_and_excludes_pacing():
    ticks = iter([10.0, 14.5])
    sleeps = []
    pacer = cg.Pacer(sleep=sleeps.append, clock=lambda: 0.0)
    spec = cg.MODELS["gemini-flash"]
    pacer.mark(spec.bucket)
    rec = cg.generate_bit(spec, cg.prompts(1)[0], FakeModel("m"), pacer, now=lambda: next(ticks))
    assert rec["latency_s"] == 4.5 and sleeps == [spec.gap_s]


def test_pacer_spaces_one_bucket_and_leaves_others_alone():
    clock, sleeps = [100.0], []
    p = cg.Pacer(sleep=sleeps.append, clock=lambda: clock[0])
    p.wait("a", 7.0)
    assert sleeps == []
    p.mark("a")
    clock[0] += 2.0
    p.wait("a", 7.0)
    p.wait("b", 7.0)
    assert sleeps == [5.0]


def test_a_filter_rejection_shows_in_pass_rates_and_the_playable_thread(data):
    run(n=1, models=["gemini-flash-lite"], filt=FakeFilter(reject=("thread 0 line 3",)))
    rec = cg.read_record(cg.run_dir() / "gemini-flash-lite" / "p00.json")
    assert cg.playable_thread(rec) == 1  # thread 0 lost a line, thread 1 is the first whole one
    s = cg.call_stats([rec], cg.MODELS["gemini-flash-lite"])
    assert s["line_pass_rate"] == pytest.approx(14 / 15) and s["thread_pass_rate"] == pytest.approx(2 / 3)
    assert s["playable_rate"] == 1.0


def rec(ok=True, latency=2.0, tin=1000, tout=2000, attempts=1, reason=None):
    return {"ok": ok, "latency_s": latency, "input_tokens": tin, "output_tokens": tout, "attempts": attempts,
            "reason": reason, "safety": [[[True, "ok"]] * 5] * 3 if ok else None}


def test_call_stats_arithmetic():
    spec = cg.MODELS["claude-haiku-5-5"]
    records = [rec(latency=x) for x in (1, 2, 3, 4)] + [rec(ok=False, latency=12, reason="LLMError: http 529", tin=500, tout=0)]
    s = cg.call_stats(records, spec)
    assert (s["n_calls"], s["n_failed"], s["failure_rate"]) == (5, 1, 0.2)
    assert s["latency_p50_s"] == 2.5 and s["latency_p95_s"] == pytest.approx(3.85)  # successes only
    cost = (4500 * 0.10 + 8000 * 0.50) / 1e6
    assert s["cost_per_bit_usd"] == pytest.approx(cost / 5) and s["cost_5_shows_usd"] == pytest.approx(cost / 5 * 40)
    assert s["failure_reasons"] == {"LLMError: http 529": 1}


def prep_ballot(n=6, models=MODELS):
    run(n=n, models=models)
    return cg.export_ballot(models, n), cg.run_dir()


def test_ballot_is_blind_shuffled_per_prompt_and_keyed(data):
    out, d = prep_ballot(n=12)
    ballot = (d / "ballot.md").read_text(encoding="utf-8")
    key = json.loads((d / "key.json").read_text(encoding="utf-8"))
    assert out == {"in_ballot": 12, "dropped": {}}
    assert not any(name in ballot.lower() for name in ("gemini", "claude", "haiku", "flash"))
    assert len({tuple(k["letters"].values()) for k in key.values()}) > 1  # not one fixed order
    assert all(sorted(k["letters"].values()) == sorted(MODELS) for k in key.values())
    for k in key.values():  # the text under each letter is that model's thread
        for m in k["letters"].values():
            assert f"{TAG[m]} thread {k['threads'][m]} line 0." in ballot


def test_prompts_without_a_playable_thread_for_every_model_are_dropped_and_listed(data):
    bad = {"gemini-flash": FakeModel("gemini-flash", fail=llm.LLMError("http 500", 500))}
    run(n=3, clients=bad)
    out = cg.export_ballot(MODELS, 3)
    assert out["in_ballot"] == 0 and set(out["dropped"]) == {"p00", "p01", "p02"}
    assert all(v == ["gemini-flash"] for v in out["dropped"].values())


def test_export_is_incremental_and_never_reshuffles_or_loses_votes(data):
    _, d = prep_ballot(n=3)
    old = json.loads((d / "key.json").read_text(encoding="utf-8"))
    votes = cg.read_votes(d / "votes.csv")
    votes["p01"] = "CAB"
    cg.write_votes(d / "votes.csv", votes)
    run(n=5)
    cg.export_ballot(MODELS, 5)
    assert cg.read_votes(d / "votes.csv") == {"p00": "", "p01": "CAB", "p02": "", "p03": "", "p04": ""}
    new = json.loads((d / "key.json").read_text(encoding="utf-8"))
    assert all(new[p] == old[p] for p in old) and len(new) == 5


def test_parse_ranking():
    assert cg.parse_ranking("bac") == "BAC" and cg.parse_ranking(" c, a b ") == "CAB"
    assert cg.parse_ranking("aab") is None and cg.parse_ranking("ab") is None and cg.parse_ranking("") is None


def test_voting_saves_after_every_answer_and_can_be_resumed(data):
    _, d = prep_ballot(n=4)
    answers = iter(["bad", "cab", "s", "abc", "q"])
    shown = []
    assert cg.vote_interactively(read=lambda prompt: next(answers), out=shown.append) == 2
    assert cg.read_votes(d / "votes.csv") == {"p00": "CAB", "p01": "", "p02": "ABC", "p03": ""}
    assert any("use each of A, B, C once" in s for s in shown)
    assert cg.vote_interactively(read=lambda prompt: "q", out=lambda s: None) == 0


def ranks_for(order_by_prompt):
    """One list of models per prompt, best first."""
    return {f"p{i:02d}": {m: r for r, m in enumerate(order, 1)} for i, order in enumerate(order_by_prompt)}


def stats(**over):
    base = {m: {"latency_p95_s": 5.0, "failure_rate": 0.0, "cost_per_bit_usd": c, "latency_p50_s": 2.0}
            for m, c in zip(MODELS, (0.0005, 0.002, 0.0006))}
    for m, patch in over.items():
        base[m.replace("_", "-")].update(patch)
    return base


def test_rank_stats_mean_ci_and_paired_gap():
    a, b, c = MODELS
    s = cg.rank_stats(ranks_for([[a, b, c]] * 20), MODELS)
    assert s["mean_rank"] == {a: 1.0, b: 2.0, c: 3.0} and s["ci95"][a] == [1.0, 1.0]
    assert s["diff"][f"{b}|{a}"]["mean"] == 1.0 and s["n"] == 20


def test_clear_winner_among_eligible_is_picked_even_if_costlier():
    a, b, c = MODELS
    d = cg.decide(stats(), ranks_for([[b, c, a]] * 24))
    assert d["pick"] == b and d["best"] == b and d["second"] == c and d["funniness_clear"] is True


def test_overlapping_ranks_pick_the_cheaper_and_say_inconclusive():
    rng = np.random.default_rng(1)
    d = cg.decide(stats(), ranks_for([list(rng.permutation(MODELS)) for _ in range(30)]))
    assert d["funniness_clear"] is False and "inconclusive" in d["reason"]
    assert d["pick"] == min((d["best"], d["second"]), key=lambda m: stats()[m]["cost_per_bit_usd"])


def test_ineligible_models_cannot_win_and_ranks_are_redone_among_the_rest():
    a, b, c = MODELS
    s = stats(gemini_flash={"latency_p95_s": 12.5})  # the funniest, but past the 12 s deadline
    d = cg.decide(s, ranks_for([[b, a, c]] * 20 + [[b, c, a]] * 4))
    assert d["eligible"] == sorted([a, c]) and d["pick"] == a and d["funniness_clear"] is True
    assert d["eligibility"][b] == {"latency_ok": False, "failure_ok": True, "eligible": False}


def test_bar_edges_are_inclusive():
    _, b, c = MODELS
    e = cg.eligibility(stats(gemini_flash={"latency_p95_s": cg.DEADLINE_S, "failure_rate": 0.05}))
    assert e[b]["eligible"]
    e = cg.eligibility(stats(gemini_flash={"failure_rate": 0.0501}, claude_haiku_5_5={"latency_p95_s": None}))
    assert not e[b]["eligible"] and not e[c]["eligible"]


def test_one_eligible_model_wins_and_none_eligible_picks_nothing():
    a = MODELS[0]
    only = cg.decide(stats(gemini_flash={"failure_rate": 0.2}, claude_haiku_5_5={"failure_rate": 0.2}), {})
    assert only["pick"] == a and only["reason"] == "the only eligible model"
    none = cg.decide(stats(**{m.replace("-", "_"): {"failure_rate": 0.5} for m in MODELS}), {})
    assert none["pick"] is None and "stays" in none["reason"]
    waiting = cg.decide(stats(), {})
    assert waiting["pick"] is None and "votes" in waiting["reason"]


def test_deadline_is_the_generators_current_one():
    from app import generate
    assert cg.DEADLINE_S == generate.GENERATOR_TIMING[1] == 12.0


def test_power_numbers_behind_the_prompt_count():
    assert cg.halfwidth(30) <= 0.3 < cg.halfwidth(26)
    assert cg.simulate_gap_power(30, 1.0, reps=1000) > 0.8
    assert cg.simulate_gap_power(30, 0.0, reps=1000) < 0.1  # no gap: the interval excludes 0 rarely


def test_run_scores_saved_calls_and_votes_and_stores_numbers_only(data, monkeypatch):
    monkeypatch.setattr(cg.cache, "METRICS", data / "metrics")
    _, d = prep_ballot(n=6)
    key = json.loads((d / "key.json").read_text(encoding="utf-8"))
    pref = {"claude-haiku-5-5": 0, "gemini-flash": 1, "gemini-flash-lite": 2}  # this voter: claude, flash, flash-lite
    votes = {pid: "".join(sorted(k["letters"], key=lambda c: pref[k["letters"][c]])) for pid, k in key.items()}
    cg.write_votes(d / "votes.csv", votes)
    params = cg.current_params(6)
    res = cg.cache.run_or_load(cg.METRICS_NAME, cg.run, **params)
    assert res["n_voted"] == 6 and res["votes"]["mean_rank"]["claude-haiku-5-5"] == 1.0
    assert res["decision"]["pick"] == "claude-haiku-5-5" and res["decision"]["funniness_clear"]
    assert cg.stored() == json.loads(json.dumps(res))
    stored_text = (data / "metrics" / "model_comparison.json").read_text(encoding="utf-8")
    assert "thread " not in stored_text and "line 0" not in stored_text  # no generated text
    assert cg.table(res).count("\n") == 4 and cg.verdict(res).startswith("Pick: claude-haiku-5-5")
    cg.write_votes(d / "votes.csv", {**votes, "p00": "ABC" if votes["p00"] != "ABC" else "CBA"})
    with pytest.raises(AssertionError):  # a changed vote invalidates the call
        cg.run(**params)
