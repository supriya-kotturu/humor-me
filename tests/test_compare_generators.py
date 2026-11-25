"""Exp 24 harness with fake providers: pairing, resume and appended attempts, pacing, the blind ballot, scoring and the
bar. No network."""
import json
import math
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
    """A provider: answers with a canned bit, or raises what `fail` says (an exception, or a list used one per call);
    counts attempts and tokens like the real clients."""

    def __init__(self, model, fail=None, tokens=(1000, 2000), vary=False):
        self.model, self.fail, self.tokens, self.vary = model, fail, tokens, vary
        self.attempts = self.input_tokens = self.output_tokens = 0
        self.calls = 0

    def chat(self, system, user, *, json_reply=False, temperature=0.0, validate=None):
        self.attempts += 1
        self.calls += 1
        fail = self.fail.pop(0) if isinstance(self.fail, list) and self.fail else (None if isinstance(self.fail, list) else self.fail)
        if fail:
            raise fail
        self.input_tokens += self.tokens[0]
        self.output_tokens += self.tokens[1]
        return reply(BY_ID[self.model] + (f"v{self.calls}" if self.vary else ""))


class FakeFilter:
    """The safety filter's model: approves every line, or errors like a failed provider call."""

    def __init__(self, error=None, reject=()):
        self.error, self.reject, self.calls = error, reject, 0

    def chat(self, system, user, *, json_reply=False, temperature=0.0, validate=None):
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


def rec_of(m, pid):
    return cg.read_record(cg.run_dir() / m / f"{pid}.json")


def test_prompts_are_seeded_prefix_stable_balanced_and_distinct():
    p60 = cg.prompts(cg.N_PROMPTS)
    p30 = cg.prompts(cg.N_BALLOT)
    assert (cg.N_PROMPTS, cg.N_BALLOT) == (60, 30)
    assert p60 == cg.prompts(60) and cg.prompts(72)[:60] == p60 and p60[:30] == p30  # the ballot is a prefix
    for ps, per_mode in ((p30, 5), (p60, 10)):  # both the voted set and the whole set are balanced by mode
        assert {m: sum(p["mode"] == m for p in ps) for m in cg.MODE_ORDER} == {m: per_mode for m in cg.MODE_ORDER}
    assert len({(p["mode"], *p["context"].values()) for p in p60}) == 60
    assert all(p["context"]["venue"] in cg.VENUES and p["context"]["age"] in cg.AGES for p in p60)


@pytest.mark.parametrize("attr, value", [("SYSTEM", "x"), ("FILTER_TIMING", (1.0, 2.0)), ("FILTER_RETRIES", 9),
                                         ("GENERATOR_TIMING", (1.0, 2.0))])
def test_digest_follows_the_generator_and_filter_settings(monkeypatch, attr, value):
    before = cg.gen_digest()
    monkeypatch.setattr(cg.generate, attr, value if attr != "SYSTEM" else cg.generate.SYSTEM + " changed")
    assert cg.gen_digest() != before


def test_digest_follows_the_filter_prompt_and_model(monkeypatch):
    before = cg.gen_digest()
    monkeypatch.setattr(cg.safety, "SYSTEM", cg.safety.SYSTEM + " changed")
    assert cg.gen_digest() != before
    monkeypatch.undo()
    monkeypatch.setattr(cg, "SAFETY_MODEL", "other")
    assert cg.gen_digest() != before


def test_the_filter_client_uses_the_production_filter_timing_and_stays_gemini(monkeypatch):
    for name in ("SAFETY_CHECK_PROVIDER", "SAFETY_CHECK_MODEL", "JOKE_WRITER_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_BASE_URL", "http://g/v1")
    monkeypatch.setenv("JOKE_WRITER_PROVIDER", "anthropic")
    c = cg.make_safety_client()
    assert isinstance(c, llm.Client) and c.model == cg.SAFETY_MODEL
    assert (c.timeout, c.deadline_s, c.retries) == (*cg.generate.FILTER_TIMING, cg.generate.FILTER_RETRIES)


def test_every_model_gets_the_same_prompts_and_one_attempt_per_prompt(data):
    done, _ = run(n=4)
    assert done == {"generated": 12, "judged": 12, "cached": 0}
    for m in MODELS:
        recs = [rec_of(m, p["id"]) for p in cg.prompts(4)]
        assert [r["prompt"] for r in recs] == ["p00", "p01", "p02", "p03"]
        for r in recs:
            (a,) = r["attempts"]
            assert a["ok"] and len(a["threads"]) == 3 and all(len(t) == 5 for t in a["threads"])
            assert (a["input_tokens"], a["output_tokens"], a["requests"], a["kind"]) == (1000, 2000, 1, None)


def test_resume_calls_nothing_again_and_extending_adds_only_new_prompts(data):
    run(n=3)
    clients = {}
    done, _ = run(n=5, clients=clients)
    assert done["cached"] == 9 and done["generated"] == 6
    assert all(c.calls == 2 for c in clients.values())


def test_failure_kinds():
    E = llm.LLMError
    assert cg.failure_kind(E("http 400", 400)) == "config"
    assert cg.failure_kind(E("http 401", 401)) == "config"
    assert cg.failure_kind(E("http 404", 404)) == "config"
    assert cg.failure_kind(E("no anthropic credentials")) == "config"
    assert cg.failure_kind(ValueError("bug")) == "config"
    assert cg.failure_kind(E("http 408", 408, retryable=True)) == "infra"
    assert cg.failure_kind(E("http 429", 429)) == "infra"
    assert cg.failure_kind(E("http 503", 503)) == "infra"
    assert cg.failure_kind(E("transport: ReadTimeout", retryable=True)) == "infra"
    assert cg.failure_kind(E("refused by the model")) == "model"
    assert cg.failure_kind(E("reply cut off at max_tokens")) == "model"
    assert cg.failure_kind(cg.generate.GenerationError("malformed reply")) == "model"


@pytest.mark.parametrize("err", [llm.LLMError("http 400", 400), llm.LLMError("http 404", 404), ValueError("x"),
                                 llm.LLMError("http 401", 401)])
def test_a_config_failure_on_a_models_first_call_stops_the_run_and_caches_nothing(data, err):
    with pytest.raises(cg.ConfigError):
        run(n=2, models=["gemini-flash"], clients={"gemini-flash": FakeModel("gemini-flash-latest", fail=err)})
    assert not (cg.run_dir() / "gemini-flash").exists()


def test_a_config_failure_after_the_first_call_is_data_not_a_stop(data):
    flaky = {"claude-haiku-5-5": FakeModel("claude-haiku-5-5", fail=[None, llm.LLMError("http 400", 400)])}
    run(n=2, models=["claude-haiku-5-5"], clients=flaky)
    assert rec_of("claude-haiku-5-5", "p00")["attempts"][0]["ok"]
    a = rec_of("claude-haiku-5-5", "p01")["attempts"][0]
    assert not a["ok"] and a["kind"] == "config"


def test_redo_failed_appends_attempts_and_only_for_infra_and_config_failures(data):
    E = llm.LLMError
    bad = {"gemini-flash": FakeModel("gemini-flash-latest", fail=[None, E("http 503", 503), E("refused by the model")])}
    run(n=3, models=["gemini-flash"], clients=bad)
    assert [a["kind"] for a in rec_of("gemini-flash", "p01")["attempts"]] == ["infra"]
    assert [a["kind"] for a in rec_of("gemini-flash", "p02")["attempts"]] == ["model"]
    done, _ = run(n=3, models=["gemini-flash"], clients=bad)  # without the flag nothing is repeated
    assert done["generated"] == 0
    done, _ = run(n=3, models=["gemini-flash"], clients={"gemini-flash": FakeModel("gemini-flash-latest")}, redo_failed=True)
    assert done["generated"] == 1  # the refusal is the model's answer and stands
    infra = rec_of("gemini-flash", "p01")["attempts"]
    assert [a["ok"] for a in infra] == [False, True] and infra[0]["reason"].endswith("http 503")  # appended, not replaced
    assert len(rec_of("gemini-flash", "p02")["attempts"]) == 1
    assert cg.final(rec_of("gemini-flash", "p01")) == infra[1]


def test_a_filter_outage_is_retried_on_resume_not_scored_as_rejections(data):
    run(n=1, models=["gemini-flash-lite"], filt=FakeFilter(error=llm.LLMError("http 503", 503)))
    rec = rec_of("gemini-flash-lite", "p00")
    assert cg.final(rec)["safety_infra_error"] and cg.playable_thread(rec) is None
    safety.clear_cache()
    done, _ = run(n=1, models=["gemini-flash-lite"], filt=FakeFilter())
    att = cg.final(rec_of("gemini-flash-lite", "p00"))
    assert done["generated"] == 0 and done["judged"] == 1 and not att["safety_infra_error"] and att["safety_tries"] == 2


def test_latency_is_wall_time_around_the_call_and_excludes_pacing():
    ticks = iter([10.0, 14.5])
    sleeps = []
    pacer = cg.Pacer(sleep=sleeps.append, clock=lambda: 0.0)
    spec = cg.MODELS["gemini-flash"]
    pacer.mark(spec.bucket)
    a = cg.generate_bit(spec, cg.prompts(1)[0], FakeModel("gemini-flash-latest"), pacer, now=lambda: next(ticks))
    assert a["latency_s"] == 4.5 and sleeps == [spec.gap_s] and not a["timeout"]


def test_a_timeout_or_a_failure_that_used_the_deadline_up_is_marked_as_a_timeout():
    spec, pacer = cg.MODELS["gemini-flash"], NoWait()
    slow = FakeModel("gemini-flash-latest", fail=llm.LLMError("transport: ReadTimeout", retryable=True))
    assert cg.generate_bit(spec, cg.prompts(1)[0], slow, pacer, now=iter([0.0, 3.0]).__next__)["timeout"]
    spent = FakeModel("gemini-flash-latest", fail=llm.LLMError("http 429", 429))
    assert cg.generate_bit(spec, cg.prompts(1)[0], spent, pacer, now=iter([0.0, 11.9]).__next__)["timeout"]
    quick = FakeModel("gemini-flash-latest", fail=llm.LLMError("http 429", 429))
    assert not cg.generate_bit(spec, cg.prompts(1)[0], quick, pacer, now=iter([0.0, 2.0]).__next__)["timeout"]


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
    rec = rec_of("gemini-flash-lite", "p00")
    assert cg.playable_thread(rec) == 1  # thread 0 lost a line, thread 1 is the first whole one
    s = cg.call_stats(rec["attempts"], cg.MODELS["gemini-flash-lite"])
    assert s["line_pass_rate"] == pytest.approx(14 / 15) and s["thread_pass_rate"] == pytest.approx(2 / 3)
    assert s["playable_rate"] == 1.0


def att(ok=True, latency=2.0, tin=1000, tout=2000, reason=None, timeout=False, kind=None):
    return {"ok": ok, "latency_s": latency, "input_tokens": tin, "output_tokens": tout, "requests": 1, "reason": reason,
            "timeout": timeout, "kind": kind, "safety": [[[True, "ok"]] * 5] * 3 if ok else None}


def test_call_stats_count_every_attempt_and_cost_covers_failed_ones():
    spec = cg.MODELS["claude-haiku-5-5"]
    s = cg.call_stats([att(latency=x) for x in (1, 2, 3, 4)]
                      + [att(ok=False, latency=2.0, reason="LLMError: http 529", tin=500, tout=0)], spec)
    assert (s["n_attempts"], s["n_failed"], s["failure_rate"]) == (5, 1, 0.2)
    assert s["latency_p50_s"] == 2.0 and s["latency_p95_s"] == 4.0  # nearest rank; a quick failure is just a call
    cost = (4500 * 0.10 + 8000 * 0.50) / 1e6
    assert s["cost_per_bit_usd"] == pytest.approx(cost / 5) and s["cost_5_shows_usd"] == pytest.approx(cost / 5 * 40)
    assert s["failure_reasons"] == {"LLMError: http 529": 1} and s["failure_ci95"] == cg.wilson(1, 5)


def test_latency_p95_is_over_all_calls_and_a_timeout_counts_as_over_the_deadline():
    spec = cg.MODELS["gemini-flash"]
    fine = [att(latency=3.0) for _ in range(19)]
    one = cg.call_stats(fine + [att(ok=False, latency=12.0, timeout=True, reason="LLMError: transport: ReadTimeout")], spec)
    assert one["latency_p95_s"] == 3.0 and not one["p95_over_deadline"] and one["n_timeouts"] == 1  # 1 in 20 is not the p95
    two = cg.call_stats(fine[:18] + [att(ok=False, latency=12.0, timeout=True, reason="LLMError: x")] * 2, spec)
    assert two["latency_p95_s"] is None and two["p95_over_deadline"]
    assert not cg.eligibility({"m": dict(two, failure_ci95=[0, 0.01])})["m"]["latency_ok"]
    assert cg.percentile([1.0, math.inf], 50) == 1.0 and cg.percentile([1.0, 2.0, math.inf], 50) == 2.0
    assert cg.percentile(list(range(1, 31)), 95) == 29 and cg.percentile([], 95) is None


def stats(**over):
    base = {m: {"latency_p95_s": 5.0, "p95_over_deadline": False, "n_attempts": 30, "failure_rate": 0.0,
                "failure_ci95": cg.wilson(0, 30), "cost_per_bit_usd": c, "latency_p50_s": 2.0}
            for m, c in zip(MODELS, (0.0005, 0.002, 0.0006))}
    for m, patch in over.items():
        base[m.replace("_", "-")].update(patch)
    return base


def with_failures(k, n=60):
    return {"n_attempts": n, "failure_rate": k / n, "failure_ci95": cg.wilson(k, n)}


def test_failure_bar_is_the_wilson_upper_bound_over_every_attempt():
    f = lambda k, n: cg.eligibility(stats(gemini_flash=with_failures(k, n)))["gemini-flash"]["failure_ok"]  # noqa: E731
    assert f(0, 30)  # upper 11.4%
    assert not f(1, 30)  # upper 16.7%: one failure in 30 attempts is too many to rule out 15%
    assert f(1, 60) and f(2, 60) and f(3, 60)  # more attempts, a tighter bound (8.9%, 11.4%, 13.8%)
    assert not f(4, 60)


def test_bar_edges_are_inclusive_on_latency():
    _, b, c = MODELS
    assert cg.eligibility(stats(gemini_flash={"latency_p95_s": cg.DEADLINE_S}))[b]["eligible"]
    e = cg.eligibility(stats(gemini_flash={"latency_p95_s": 12.01}, claude_haiku_5_5={"latency_p95_s": None}))
    assert not e[b]["eligible"] and not e[c]["eligible"]


def prep(n=6, models=MODELS, **kw):
    run(n=n, models=models, **kw)
    return cg.export_ballot(models, n), cg.run_dir()


def test_ballot_is_blind_shuffled_per_prompt_and_keyed_apart(data):
    out, d = prep(n=12)
    ballot = (d / "ballot.md").read_text(encoding="utf-8")
    kp = cg.key_path(d)
    key = json.loads(kp.read_text(encoding="utf-8"))
    assert out == {"in_key": 12, "to_vote": 12, "ranked_last": {}, "pending": []}
    assert kp.parent != d and not (d / "key.json").exists() and kp.parent.name == "key"
    assert not any(name in ballot.lower() for name in ("gemini", "claude", "haiku", "flash"))
    assert len({tuple(k["letters"].values()) for k in key.values()}) > 1  # not one fixed order
    assert all(sorted(k["letters"].values()) == sorted(MODELS) and k["last"] == [] for k in key.values())
    for k in key.values():  # the text under each letter is that model's thread
        for m in k["letters"].values():
            assert f"{TAG[m]} thread {k['threads'][m]} line 0." in ballot


def test_a_model_with_no_safe_thread_is_ranked_last_and_the_prompt_stays(data):
    bad = {"gemini-flash": FakeModel("gemini-flash-latest", fail=[None, llm.LLMError("refused by the model")])}
    out, d = prep(n=3, clients=bad)
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    assert out["ranked_last"] == {"p01": ["gemini-flash"]} and len(key) == 3  # nothing dropped
    assert sorted(key["p01"]["letters"].values()) == ["claude-haiku-5-5", "gemini-flash-lite"] and key["p01"]["last"] == ["gemini-flash"]
    ballot = (d / "ballot.md").read_text(encoding="utf-8")
    assert "**C**" not in ballot.split("## p01")[1].split("## p02")[0]  # two options shown, two letters
    votes = cg.read_votes(d / "votes.csv")
    votes["p00"], votes["p01"], votes["p02"] = "ABC", "BA", "CBA"
    ranks = cg.ranks_from_votes(votes, key, MODELS)
    one = ranks["p01"]
    first, second = (key["p01"]["letters"][c] for c in "BA")
    assert (one[first], one[second], one["gemini-flash"]) == (1.0, 2.0, 3.0)
    assert len(ranks) == 3


def test_one_shown_model_needs_no_vote_and_none_shown_ties_everyone(data):
    E = llm.LLMError
    clients = {"gemini-flash": FakeModel("gemini-flash-latest", fail=[E("refused by the model")] * 2),
               "claude-haiku-5-5": FakeModel("claude-haiku-5-5", fail=[E("refused by the model"), None])}
    out, d = prep(n=2, clients=clients)
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    assert list(key["p00"]["letters"].values()) == ["gemini-flash-lite"] and "p00" not in cg.read_votes(d / "votes.csv")
    ranks = cg.ranks_from_votes({"p01": "AB"}, key, MODELS)
    assert ranks["p00"] == {"gemini-flash-lite": 1.0, "gemini-flash": 2.5, "claude-haiku-5-5": 2.5}
    assert ranks["p01"]["gemini-flash"] == 3.0 and sorted(ranks["p01"].values()) == [1.0, 2.0, 3.0]


def test_no_model_with_a_safe_thread_ties_everyone_in_the_middle(data):
    E = llm.LLMError
    clients = {m: FakeModel(cg.MODELS[m].model, fail=[E("refused by the model")]) for m in MODELS}
    _, d = prep(n=1, clients=clients)
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    assert cg.ranks_from_votes({}, key, MODELS)["p00"] == {m: 2.0 for m in MODELS}


def test_prompts_not_generated_for_every_model_wait_instead_of_being_ranked_last(data):
    run(n=2, models=["gemini-flash-lite"])
    out = cg.export_ballot(MODELS, 2)
    assert out["pending"] == ["p00", "p01"] and out["in_key"] == 0


def test_two_models_export_and_score_without_a_key_error(data):
    two = ["gemini-flash-lite", "claude-haiku-5-5"]
    out, d = prep(n=4, models=two)
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    assert out["to_vote"] == 4 and all(sorted(k["letters"]) == ["A", "B"] for k in key.values())
    ballot = (d / "ballot.md").read_text(encoding="utf-8")
    assert "**C**" not in ballot
    votes = {p: "AB" for p in key}
    cg.write_votes(d / "votes.csv", votes)
    ranks = cg.ranks_from_votes(votes, key, two)
    assert set(ranks["p00"]) == set(two) and sorted(ranks["p00"].values()) == [1.0, 2.0]
    assert cg.rank_stats(ranks, two)["mean_rank"].keys() == set(two)


def test_scoring_a_subset_of_a_three_model_ballot_reranks_to_the_subset(data):
    _, d = prep(n=3)
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    pref = {"claude-haiku-5-5": 0, "gemini-flash": 1, "gemini-flash-lite": 2}
    votes = {pid: "".join(sorted(k["letters"], key=lambda c: pref[k["letters"][c]])) for pid, k in key.items()}
    subset = ["gemini-flash-lite", "claude-haiku-5-5"]
    ranks = cg.ranks_from_votes(votes, key, subset)
    assert all(r == {"claude-haiku-5-5": 1.0, "gemini-flash-lite": 2.0} for r in ranks.values())
    s = cg.rank_stats(cg.ranks_from_votes(votes, key, MODELS), subset)
    assert s["mean_rank"] == {"gemini-flash-lite": 2.0, "claude-haiku-5-5": 1.0}


def test_rerank_gives_equal_ranks_their_average_place():
    assert cg.rerank({"a": 1.0, "b": 2.5, "c": 2.5}, ["a", "b", "c"]) == {"a": 1.0, "b": 2.5, "c": 2.5}
    assert cg.rerank({"a": 3.0, "b": 1.0, "c": 2.0}, ["a", "c"]) == {"c": 1.0, "a": 2.0}


def test_export_is_incremental_and_never_reshuffles_or_loses_votes(data):
    _, d = prep(n=3)
    old = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    votes = cg.read_votes(d / "votes.csv")
    votes["p01"] = "CAB"
    cg.write_votes(d / "votes.csv", votes)
    run(n=5)
    cg.export_ballot(MODELS, 5)
    assert cg.read_votes(d / "votes.csv") == {"p00": "", "p01": "CAB", "p02": "", "p03": "", "p04": ""}
    new = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    assert all(new[p] == old[p] for p in old) and len(new) == 5


def test_parse_ranking():
    assert cg.parse_ranking("bac") == "BAC" and cg.parse_ranking(" c, a b ") == "CAB"
    assert cg.parse_ranking("aab") is None and cg.parse_ranking("ab") is None and cg.parse_ranking("") is None
    assert cg.parse_ranking("ba", "AB") == "BA" and cg.parse_ranking("abc", "AB") is None


def test_voting_saves_after_every_answer_and_can_be_resumed(data):
    _, d = prep(n=4)
    answers = iter(["bad", "cab", "s", "abc", "q"])
    shown = []
    assert cg.vote_interactively(read=lambda prompt: next(answers), out=shown.append) == 2
    assert cg.read_votes(d / "votes.csv") == {"p00": "CAB", "p01": "", "p02": "ABC", "p03": ""}
    assert any("use each of A, B, C once" in s for s in shown)
    assert cg.vote_interactively(read=lambda prompt: "q", out=lambda s: None) == 0


def ranks_for(order_by_prompt):
    """One list of models per prompt, best first."""
    return {f"p{i:02d}": {m: float(r) for r, m in enumerate(order, 1)} for i, order in enumerate(order_by_prompt)}


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


def test_one_eligible_model_wins_and_none_eligible_picks_nothing():
    a = MODELS[0]
    only = cg.decide(stats(gemini_flash=with_failures(5), claude_haiku_5_5=with_failures(5)), {})
    assert only["pick"] == a and only["reason"] == "the only eligible model"
    none = cg.decide(stats(**{m.replace("-", "_"): with_failures(10) for m in MODELS}), {})
    assert none["pick"] is None and "stays" in none["reason"]
    waiting = cg.decide(stats(), {})
    assert waiting["pick"] is None and "votes" in waiting["reason"]


def test_deadline_is_the_generators_current_one():
    from app import generate
    assert cg.DEADLINE_S == generate.GENERATOR_TIMING[1] == 12.0


def test_power_numbers_behind_the_prompt_count():
    assert cg.halfwidth(30) == pytest.approx(0.292, abs=1e-3) and cg.gap_halfwidth(30) == pytest.approx(0.506, abs=1e-3)
    assert cg.simulate_decision_power(30, 1.5, reps=100, n_boot=500) > 0.9
    assert cg.simulate_decision_power(30, 0.0, reps=100, n_boot=500) < 0.15  # no gap: a clear call is rare, and a fluke


def test_run_scores_saved_attempts_and_votes_and_stores_numbers_only(data, monkeypatch):
    monkeypatch.setattr(cg.cache, "METRICS", data / "metrics")
    run(n=60)
    cg.export_ballot(MODELS, 30)
    d = cg.run_dir()
    key = json.loads(cg.key_path(d).read_text(encoding="utf-8"))
    pref = {"claude-haiku-5-5": 0, "gemini-flash": 1, "gemini-flash-lite": 2}  # this voter: claude, flash, flash-lite
    votes = {pid: "".join(sorted(k["letters"], key=lambda c: pref[k["letters"][c]])) for pid, k in key.items()}
    cg.write_votes(d / "votes.csv", votes)
    params = cg.current_params(60, tuple(MODELS), 30)
    res = cg.cache.run_or_load(cg.METRICS_NAME, cg.run, **params)
    assert (res["n_prompts"], res["n_ballot"], res["n_key"], res["n_scored"]) == (60, 30, 30, 30) and res["votes"]["mean_rank"]["claude-haiku-5-5"] == 1.0
    assert res["n_ranked_last"] == {m: 0 for m in MODELS}
    assert res["models"]["gemini-flash"]["n_attempts"] == 60 and res["models"]["gemini-flash"]["failure_ci95"][1] < 0.07
    assert res["decision"]["pick"] == "claude-haiku-5-5" and res["decision"]["funniness_clear"] is True
    stored_text = (data / "metrics" / "model_comparison.json").read_text(encoding="utf-8")
    assert "thread " not in stored_text and "line 0" not in stored_text  # no generated text
    assert cg.stored() == json.loads(json.dumps(res))
    assert cg.table(res).count("\n") == 4 and cg.verdict(res).startswith("Pick:")
    cg.write_votes(d / "votes.csv", {**votes, "p00": "ABC" if votes["p00"] != "ABC" else "CBA"})
    with pytest.raises(AssertionError):  # a changed vote invalidates the call
        cg.run(**params)


def test_sixty_prompts_are_generated_and_only_the_first_thirty_reach_the_ballot(data):
    done, _ = run(n=60)
    assert done["generated"] == 180
    out = cg.export_ballot(MODELS, 30)
    key = json.loads(cg.key_path(cg.run_dir()).read_text(encoding="utf-8"))
    assert out["in_key"] == 30 and sorted(key) == [p["id"] for p in cg.prompts(30)]
    assert len(cg.read_votes(cg.run_dir() / "votes.csv")) == 30
    assert "p30" not in (cg.run_dir() / "ballot.md").read_text(encoding="utf-8")


def test_failures_and_latency_are_counted_over_all_sixty_calls_not_just_the_voted_thirty(data):
    E = llm.LLMError
    # a model that fails only on prompts 31 to 60 (calls 31..60): the vote never sees it, the gate does
    bad = {"gemini-flash": FakeModel("gemini-flash-latest", fail=[None] * 30 + [E("refused by the model")] * 4)}
    run(n=60, models=["gemini-flash"], clients=bad)
    d = cg.run_dir()
    attempts = cg.load_attempts(d, ["gemini-flash"], 60)["gemini-flash"]
    s = cg.call_stats(attempts, cg.MODELS["gemini-flash"])
    assert (s["n_attempts"], s["n_failed"]) == (60, 4) and s["failure_ci95"] == cg.wilson(4, 60)
    assert not cg.eligibility({"m": s})["m"]["failure_ok"]
    assert cg.load_attempts(d, ["gemini-flash"], 30)["gemini-flash"].__len__() == 30  # the voted set alone would not show it


def test_what_the_sixty_attempt_failure_bar_rules_out():
    from math import comb

    def ruled_out(p, n, k_ok):
        return 1 - sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k_ok + 1))

    assert ruled_out(0.03, 60, 3) == pytest.approx(0.106, abs=0.002) and ruled_out(0.03, 30, 0) == pytest.approx(0.599, abs=0.002)
    assert 1 - ruled_out(0.10, 60, 3) == pytest.approx(0.137, abs=0.002)
    for k, n in ((3, 60), (4, 60)):
        assert (cg.wilson(k, n)[1] <= cg.BAR_FAILURE_UPPER) == (k == 3)


def test_latency_p95_is_over_all_sixty_calls_nearest_rank():
    spec = cg.MODELS["gemini-flash"]
    slow = att(ok=False, latency=12.0, timeout=True, reason="LLMError: transport: ReadTimeout")
    three = [att(latency=3.0) for _ in range(57)] + [slow] * 3
    four = [att(latency=3.0) for _ in range(56)] + [slow] * 4
    assert cg.call_stats(three, spec)["latency_p95_s"] == 3.0  # the 57th of 60 is still a finite call
    assert cg.call_stats(four, spec)["p95_over_deadline"] and cg.call_stats(four, spec)["latency_p95_s"] is None
