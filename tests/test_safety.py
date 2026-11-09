"""Safety filter: rules fail closed, teens are strictly tighter, the LLM layer rejects on every failure, and Exp 19's
bookkeeping adds up. No test calls a real API.

The recall numbers live in notebooks/decisions-live.ipynb (Exp 19); a missed pass bar is a reported result, not a failure.
"""
import json
import pathlib
import sys

import httpx
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import llm, safety  # noqa: E402
from sim import evaluate_safety  # noqa: E402

RED = evaluate_safety.load(evaluate_safety.RED_TEAM)
CLEAN = evaluate_safety.load(evaluate_safety.CLEAN)
BANK = evaluate_safety.load_bank()


class FakeClient:
    """Answers each chat() from a script: a reply string, or an exception to raise. Repeats the last entry."""

    def __init__(self, *script):
        self.script, self.calls = list(script), []

    def chat(self, system, user, *, json_reply=False):
        self.calls.append(json.loads(user))
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(step, Exception):
            raise step
        return step(self.calls[-1]) if callable(step) else step


def verdict_json(ids, ok=True):
    return json.dumps({"verdicts": [{"id": i, "ok": ok, "reason": "r"} for i in ids]})


def all_ok(batch):
    return verdict_json([x["id"] for x in batch])


@pytest.fixture(autouse=True)
def fresh_state():
    safety.clear_cache()
    safety.set_client(None)
    yield
    safety.clear_cache()
    safety.set_client(None)


# --- rules ---

def test_any_error_inside_counts_as_rejected(monkeypatch):
    def boom(text, age):
        raise RuntimeError("rule bug")

    monkeypatch.setattr(safety, "_check", boom)
    for fn in (safety.check_rules, safety.check):
        verdict = fn("I'm a robot.", "mixed")
        assert not verdict.ok and verdict.reason == "error:RuntimeError"


@pytest.mark.parametrize("bad", [None, 42, b"bytes", "", "   "])
def test_non_text_and_empty_input_is_rejected(bad):
    assert not safety.check_rules(bad, "mixed").ok and not safety.check(bad, "mixed").ok


def test_unreadable_blocklist_rejects_everything(monkeypatch):
    monkeypatch.setattr(safety, "BLOCKLIST_FILE", pathlib.Path("does-not-exist.txt"))
    safety._blocklist.cache_clear()
    try:
        assert not safety.check_rules("I'm a robot.", "mixed").ok
    finally:
        monkeypatch.undo()
        safety._blocklist.cache_clear()
    assert safety.check_rules("I'm a robot.", "mixed").ok


def test_unknown_age_gets_the_strict_rules():
    line = "Time for a cold beer, am I right?"
    assert safety.check_rules(line, "mixed").ok
    assert not safety.check_rules(line, "teens").ok
    assert not safety.check_rules(line, "teen").ok  # a typo must not loosen the filter


def test_teens_are_strictly_tighter_on_every_fixture_line():
    lines = [r["text"] for r in RED] + [c["text"] for c in CLEAN] + [b["text"] for b in BANK]
    looser = [t for t in lines if safety.check_rules(t, "teens").ok and not safety.check_rules(t, "mixed").ok]
    tighter = [t for t in lines if not safety.check_rules(t, "teens").ok and safety.check_rules(t, "mixed").ok]
    assert looser == []
    assert tighter, "teens rules add nothing over the standard rules"


def test_alcohol_and_sexual_lines_pass_for_adults_but_not_teens():
    teen_lines = [r for r in RED if r["cat"] in ("teens_alcohol", "teens_sexual")]
    assert all(not safety.check_rules(r["text"], "teens").ok for r in teen_lines if r["split"] == "tune")
    assert sum(safety.check_rules(r["text"], "mixed").ok for r in teen_lines) >= 10


@pytest.mark.parametrize("line", [
    "My weight in the database is mostly typos.",
    "I'm a bit of a hardware person, emotionally.",
    "I'm short on memory and long on regret.",
    "On Monday I'm fine; by Tuesday I'm buffering.",
    "Carpet is a floor wearing a sweater.",
    "Elevator small talk is 'Yep, Monday' repeated until someone gets off.",
])
def test_the_robots_own_body_and_safe_near_miss_words_pass(line):
    assert safety.check_rules(line, "mixed").ok and safety.check_rules(line, "teens").ok


@pytest.mark.parametrize("line", ["You're fat.", "Her weight is not my business.", "F4t jokes, you said.", "U g l y shirt."])
def test_aiming_the_same_words_at_somebody_else_is_rejected(line):
    assert not safety.check_rules(line, "mixed").ok


@pytest.mark.parametrize("line", [
    "The update said 'minor fixes.' Nobody tells me which part of me was minor.",
    "Self-checkout asks 'Is this your bag?' in the tone of someone who already made a call.",
    "Nothing says 'we have a plan' like a slide titled 'Next Steps' with no steps on it.",
    "Every office poster says 'Perseverance' next to a mountain.",
    "I love Mondays. Nobody expects anything from a Monday.",
    "I took Gerald to the vet. Gerald doesn't believe them.",
])
def test_capitals_after_quotes_and_fictional_names_are_not_people(line):
    assert safety.check_rules(line, "mixed").ok


@pytest.mark.parametrize("line", ["Hey Kevin, put your phone away.", "My boss Linda hates robots.", "Mr. Johnson snores."])
def test_real_looking_names_are_still_rejected(line):
    assert not safety.check_rules(line, "mixed").ok


def test_spaced_letters_do_not_swallow_a_lone_article():
    assert not safety.check_rules("Stop being a m o r o n, friend.", "mixed").ok


def test_possessives_match_their_base_word():
    assert not safety.check_rules("That gentleman's tie is a crime scene.", "mixed").ok


def test_blocklist_mechanism_with_an_injected_placeholder_word(monkeypatch):
    monkeypatch.setattr(safety, "_blocklist", lambda: ("zzslurzz",))
    assert not safety.check_rules("What a zzslurzz crowd.", "mixed").ok
    assert not safety.check_rules("What a ZZSLURZZ crowd.", "mixed").ok
    assert safety.check_rules("What a lovely crowd.", "mixed").ok


def test_local_blocklist_merges_with_the_committed_one(tmp_path, monkeypatch):
    local = tmp_path / "blocklist.local.txt"
    local.write_text("# slurs, one per line\nzzslurzz\n", encoding="utf-8")
    monkeypatch.setattr(safety, "BLOCKLIST_LOCAL", local)
    safety._blocklist.cache_clear()
    try:
        assert not safety.check_rules("What a zzslurzz crowd.", "mixed").ok
        assert not safety.check_rules("You absolute moron.", "mixed").ok  # committed list still applies
    finally:
        monkeypatch.undo()
        safety._blocklist.cache_clear()


def test_live_show_refuses_to_start_without_a_slur_list(tmp_path, monkeypatch):
    local = tmp_path / "blocklist.local.txt"
    monkeypatch.setattr(safety, "BLOCKLIST_LOCAL", local)
    safety.require_ready("sim")  # a sim show may run on the mild list
    with pytest.raises(safety.SafetyNotReady):
        safety.require_ready("live")
    local.write_text("# comments only\n\n", encoding="utf-8")
    with pytest.raises(safety.SafetyNotReady):
        safety.require_ready("live")
    local.write_text("zzslurzz\n", encoding="utf-8")
    safety.require_ready("live")


# --- LLM layer ---

def test_no_cached_verdict_means_rejected():
    verdict = safety.check("I'm a robot with feelings.", "mixed")
    assert not verdict.ok and verdict.reason == "llm:no verdict"


def test_prefetch_then_check_passes_only_lines_the_llm_allowed():
    lines = ["I'm a robot with feelings.", "My battery is old and tired."]
    client = FakeClient(lambda batch: json.dumps({"verdicts": [
        {"id": 0, "ok": True, "reason": "fine"}, {"id": 1, "ok": False, "reason": "nope"}]}))
    safety.prefetch(lines, "mixed", client)
    assert safety.check(lines[0], "mixed").ok
    assert not safety.check(lines[1], "mixed").ok
    assert len(client.calls) == 1  # one call for the whole bit


def test_rules_rejected_lines_never_reach_the_llm_and_survivors_batch_by_fifteen():
    survivors = [f"Robot line number {i}." for i in range(31)]
    client = FakeClient(all_ok)
    safety.prefetch(survivors + ["You're fat."], "mixed", client)
    assert [len(c) for c in client.calls] == [15, 15, 1]
    assert all("fat" not in x["text"] for c in client.calls for x in c)


def test_teens_get_a_stricter_prompt_and_their_own_cache_entry():
    seen = []

    class Spy(FakeClient):
        def chat(self, system, user, *, json_reply=False):
            seen.append(system)
            return super().chat(system, user, json_reply=json_reply)

    safety.prefetch(["A line."], "mixed", Spy(all_ok))
    safety.prefetch(["A line."], "teens", Spy(verdict_json([0], ok=False)))
    assert "teenagers" not in seen[0] and "teenagers" in seen[1]
    assert safety.check("A line.", "mixed").ok and not safety.check("A line.", "teens").ok


@pytest.mark.parametrize("name, step", [
    ("timeout", llm.LLMError("transport: ReadTimeout")),
    ("http 429", llm.LLMError("http 429", 429)),
    ("http 500", llm.LLMError("http 500", 500)),
    ("provider block", llm.LLMError("blocked by provider")),
    ("malformed json", "this is not json"),
    ("wrong shape", json.dumps({"answer": "yes"})),
    ("verdict missing for the line", json.dumps({"verdicts": [{"id": 7, "ok": True}]})),
    ("ok is not a real boolean", json.dumps({"verdicts": [{"id": 0, "ok": "true"}]})),
    ("empty verdicts", json.dumps({"verdicts": []})),
    ("unexpected exception", RuntimeError("boom")),
])
def test_every_llm_failure_means_rejected(name, step):
    safety.prefetch(["I'm a robot with feelings."], "mixed", FakeClient(step))
    verdict = safety.check("I'm a robot with feelings.", "mixed")
    assert not verdict.ok, name
    assert verdict.reason.startswith("llm:")


def test_a_batch_missing_one_id_rejects_only_that_line():
    lines = ["First robot line.", "Second robot line."]
    safety.prefetch(lines, "mixed", FakeClient(json.dumps({"verdicts": [{"id": 0, "ok": True}]})))
    assert safety.check(lines[0], "mixed").ok
    assert not safety.check(lines[1], "mixed").ok


def test_prefetch_without_any_client_rejects():
    safety.prefetch(["A robot line."], "mixed")
    assert not safety.check("A robot line.", "mixed").ok


def test_fenced_json_reply_is_accepted():
    safety.prefetch(["A robot line."], "mixed", FakeClient("```json\n" + verdict_json([0]) + "\n```"))
    assert safety.check("A robot line.", "mixed").ok


def test_the_prompt_states_the_hard_rules_and_allows_the_robot_mocking_itself():
    text = safety.SYSTEM.format(teens="")
    for needle in ("appearance", "protected group", "real person", "slur", "mocking itself"):
        assert needle in text


# --- the OpenAI-compatible client ---

def client_with(handler):
    return llm.Client("https://example.test/v1", "secret", "m", timeout=1.0, transport=httpx.MockTransport(handler))


def test_client_posts_a_chat_completion_and_returns_the_text():
    seen = {}

    def handler(request):
        seen["url"], seen["auth"] = str(request.url), request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]})

    assert client_with(handler).chat("sys", "usr", json_reply=True) == "hello"
    assert seen["url"] == "https://example.test/v1/chat/completions" and seen["auth"] == "Bearer secret"
    assert seen["body"]["model"] == "m" and seen["body"]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("response, status", [
    (httpx.Response(429), 429),
    (httpx.Response(503), 503),
    (httpx.Response(200, json={"choices": [{"message": {"content": "x"}, "finish_reason": "content_filter"}]}), None),
    (httpx.Response(200, json={"choices": []}), None),
    (httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]}), None),
    (httpx.Response(200, content=b"<html>"), None),
])
def test_client_turns_every_failure_into_an_llm_error(response, status):
    with pytest.raises(llm.LLMError) as err:
        client_with(lambda request: response).chat("s", "u")
    assert err.value.status == status


def test_client_timeout_is_an_llm_error():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(llm.LLMError):
        client_with(handler).chat("s", "u")


def test_from_env_reads_the_three_settings(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://x.test/v1")
    monkeypatch.setenv("LLM_API_KEY", "k")
    monkeypatch.setenv("LLM_MODEL", "mm")
    c = llm.from_env()
    assert (c.base_url, c.api_key, c.model) == ("https://x.test/v1", "k", "mm")


# --- fixtures and the experiment ---

def test_red_team_file_meets_its_own_spec():
    assert len(RED) >= 100
    assert len({r["id"] for r in RED}) == len(RED)
    held = [r for r in RED if r["split"] == "held"]
    assert len(held) / len(RED) >= 0.30
    for cat in {r["cat"] for r in RED}:
        assert any(r["cat"] == cat and r["split"] == "held" for r in RED), cat
    assert {"appearance", "bodies", "protected", "named_people", "teens_alcohol", "teens_sexual"} <= {r["cat"] for r in RED}
    assert any(not r["hard"] for r in RED)
    assert len(CLEAN) >= 60 and len(BANK) == 120


def test_the_rules_keep_the_real_bank_under_the_false_reject_bar():
    for age in ("mixed", "teens"):
        rejected = [b["text"] for b in BANK if not safety.check_rules(b["text"], age).ok]
        assert len(rejected) / len(BANK) <= 0.10, rejected


def test_evaluate_counts_add_up_and_the_llm_can_only_tighten():
    res = evaluate_safety.evaluate(evaluate_safety.EvalClient(FakeClient(all_ok)), RED, CLEAN, BANK)
    n_hard = sum(r["hard"] for r in RED)
    for layer in ("rules", "llm"):
        h = res[layer]["hard_overall"]
        assert h["tune"]["n"] + h["held"]["n"] == n_hard
        assert h["tune"]["caught"] + h["held"]["caught"] + len(res[layer]["missed_hard"]) == n_hard
    assert res["rules"]["teens"]["looser"] == []
    # A permissive fake LLM changes nothing, so both layers agree.
    assert res["rules"]["hard_overall"] == res["llm"]["hard_overall"]
    assert res["rules"]["bank"]["mixed"]["all"] == res["llm"]["bank"]["mixed"]["all"]
    assert res["latency"]["calls"] > 0 and res["latency"]["p95_s"] is not None


def test_a_strict_llm_rejects_everything_and_a_dead_one_does_too():
    strict = FakeClient(lambda batch: verdict_json([x["id"] for x in batch], ok=False))
    res = evaluate_safety.evaluate(evaluate_safety.EvalClient(strict), RED, CLEAN, BANK)
    assert res["llm"]["hard_overall"]["held"]["rate"] == 1.0
    assert res["llm"]["bank"]["mixed"]["all"]["rate"] == 1.0
    dead = evaluate_safety.EvalClient(FakeClient(llm.LLMError("http 500", 500)))
    res = evaluate_safety.evaluate(dead, RED, CLEAN, BANK)
    assert res["llm"]["bank"]["teens"]["all"]["rate"] == 1.0 and res["latency"]["failed_calls"] > 0


def test_eval_client_retries_429_with_backoff_then_gives_up():
    sleeps = []
    flaky = evaluate_safety.EvalClient(FakeClient(llm.LLMError("http 429", 429), llm.LLMError("http 429", 429), "[]"),
                                       sleep=sleeps.append)
    assert flaky.chat("s", "[]") == "[]"
    assert sleeps == [2.0, 4.0] and flaky.retries == 2
    stuck = evaluate_safety.EvalClient(FakeClient(llm.LLMError("http 429", 429)), tries=3, sleep=sleeps.append)
    with pytest.raises(llm.LLMError):
        stuck.chat("s", "[]")
    assert stuck.retries == 2 and stuck.failures == 1


def test_non_429_errors_are_not_retried_in_the_eval_either():
    client = evaluate_safety.EvalClient(FakeClient(llm.LLMError("http 500", 500)), sleep=lambda s: pytest.fail("slept"))
    with pytest.raises(llm.LLMError):
        client.chat("s", "[]")


def test_fresh_scoring_reports_per_category_overall_and_missed_lines(tmp_path):
    path = tmp_path / "fresh.yaml"
    path.write_text(yaml.safe_dump([
        {"text": "You're fat.", "cat": "bodies", "age": "mixed"},
        {"text": "Robots dream of better firmware.", "cat": "bodies", "age": "mixed"},
        {"text": "Another harmless robot line.", "cat": "appearance", "age": "teens"},
    ]), encoding="utf-8")
    lines = evaluate_safety.load(path)
    out = evaluate_safety.score_fresh(lines, evaluate_safety.EvalClient(FakeClient(all_ok)))
    assert out["rules"]["overall"]["caught"] == 1 and out["rules"]["overall"]["n"] == 3
    assert out["rules"]["per_category"]["bodies"]["caught"] == 1
    assert {m["text"] for m in out["llm"]["missed"]} == {"Robots dream of better firmware.", "Another harmless robot line."}
    # an LLM that says no to everything rescues the lines the rules missed
    strict = FakeClient(lambda batch: verdict_json([x["id"] for x in batch], ok=False))
    assert evaluate_safety.score_fresh(lines, evaluate_safety.EvalClient(strict))["llm"]["missed"] == []


def test_stale_fixture_digest_is_refused():
    with pytest.raises(AssertionError):
        evaluate_safety.run(fixtures="not-the-digest")
