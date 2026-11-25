"""The Claude client against a fake SDK: no network, no key. What the live loop relies on is that every failure is an
LLMError with the right retryable flag, that a refusal or a cut-off reply is never read as an answer, and that the
request never carries a temperature (Haiku 5.5 rejects one with a 400)."""
import pathlib
import sys
from types import SimpleNamespace

import anthropic
import httpx
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import generate, llm, llm_anthropic  # noqa: E402

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls, code):
    return cls("boom", response=httpx.Response(code, request=REQ), body=None)


def message(*blocks, stop="end_turn", usage=(10, 20)):
    return SimpleNamespace(stop_reason=stop, content=list(blocks),
                           usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1]))


def text(t):
    return SimpleNamespace(type="text", text=t)


class FakeSDK:
    """`script` items are replies or exceptions, consumed one per request; every request's kwargs are kept."""

    def __init__(self, *script):
        self.script, self.calls = list(script), []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def client(*script, **options):
    sdk = FakeSDK(*script)
    return llm_anthropic.Client(sdk=sdk, sleep=lambda s: None, **options), sdk


def test_text_comes_from_text_blocks_only():
    c, _ = client(message(SimpleNamespace(type="thinking", thinking="hmm"), text("a"), text("b")))
    assert c.chat("sys", "usr") == "ab"


def test_request_shape_and_no_temperature_ever():
    c, sdk = client(message(text("ok")))
    c.chat("sys", "usr", json_reply=True, temperature=0.9)
    (call,) = sdk.calls
    assert "temperature" not in call
    assert call["model"] == "claude-haiku-5-5" and call["system"] == "sys"
    assert call["messages"] == [{"role": "user", "content": "usr"}]  # no assistant prefill
    assert call["output_config"] == {"effort": "low"}


def test_tokens_are_counted_even_for_a_refusal():
    c, _ = client(message(text("ok"), usage=(100, 50)), message(stop="refusal", usage=(7, 0)))
    c.chat("s", "u")
    with pytest.raises(llm.LLMError):
        c.chat("s", "u")
    assert (c.input_tokens, c.output_tokens) == (107, 50)


@pytest.mark.parametrize("stop", ["refusal", "max_tokens", "pause_turn", "tool_use"])
def test_a_stop_other_than_end_turn_is_an_error_and_content_is_ignored(stop):
    c, sdk = client(message(text('{"threads": []}'), stop=stop), retries=2)
    with pytest.raises(llm.LLMError) as e:
        c.chat("s", "u")
    assert not e.value.retryable and len(sdk.calls) == 1


def test_empty_text_is_an_error():
    c, _ = client(message(SimpleNamespace(type="thinking", thinking="x")))
    with pytest.raises(llm.LLMError, match="empty"):
        c.chat("s", "u")


@pytest.mark.parametrize("exc, retryable, status", [
    (status_error(anthropic.RateLimitError, 429), True, 429),
    (status_error(anthropic.InternalServerError, 500), True, 500),
    (status_error(anthropic.APIStatusError, 529), True, 529),  # overloaded
    (status_error(anthropic.BadRequestError, 400), False, 400),
    (status_error(anthropic.APIStatusError, 408), True, 408),
    (status_error(anthropic.NotFoundError, 404), False, 404),
    (status_error(anthropic.AuthenticationError, 401), False, 401),
    (anthropic.APITimeoutError(request=REQ), True, None),
    (anthropic.APIConnectionError(request=REQ), True, None),
])
def test_sdk_errors_map_to_llm_error(exc, retryable, status):
    c, _ = client(exc)
    with pytest.raises(llm.LLMError) as e:
        c.chat("s", "u")
    assert e.value.retryable is retryable and e.value.status == status


def test_missing_credentials_is_a_plain_non_retryable_error():
    c, _ = client(TypeError("Could not resolve authentication method. Expected one of api_key"), retries=2)
    with pytest.raises(llm.LLMError, match="ANTHROPIC_API_KEY") as e:
        c.chat("s", "u")
    assert not e.value.retryable


def test_retries_a_429_then_succeeds_and_counts_attempts():
    c, sdk = client(status_error(anthropic.RateLimitError, 429), message(text("ok")), retries=2)
    assert c.chat("s", "u") == "ok"
    assert (c.attempts, c.retried, c.failed, len(sdk.calls)) == (2, 1, 0, 2)


def test_a_non_retryable_error_is_not_retried():
    c, sdk = client(status_error(anthropic.BadRequestError, 400), message(text("ok")), retries=2)
    with pytest.raises(llm.LLMError):
        c.chat("s", "u")
    assert (len(sdk.calls), c.failed) == (1, 1)


def test_the_deadline_bounds_retries_and_each_attempt_timeout():
    # A retry needs backoff + MIN_ATTEMPT_S left, which a 0.3 s deadline never has, so the 429 is not retried.
    c, sdk = client(status_error(anthropic.RateLimitError, 429), message(text("ok")), retries=4, timeout=10.0,
                    deadline_s=0.3)
    with pytest.raises(llm.LLMError):
        c.chat("s", "u")
    t = sdk.calls[0]["timeout"]
    assert len(sdk.calls) == 1 and isinstance(t, anthropic.Timeout) and t.read <= 0.3 and t.connect <= 0.3


def test_attempt_timeout_has_a_short_connect_phase():
    c, sdk = client(message(text("ok")), timeout=10.0)
    c.chat("s", "u")
    t = sdk.calls[0]["timeout"]
    assert (t.read, t.connect) == (10.0, 2.0)


def test_anything_that_is_not_an_llm_error_is_wrapped_and_not_retried():
    for boom in (ValueError("x"), TypeError("not about credentials"), KeyError("k")):
        c, sdk = client(boom, message(text("ok")), retries=2)
        with pytest.raises(llm.LLMError, match="unexpected") as e:
            c.chat("s", "u")
        assert not e.value.retryable and len(sdk.calls) == 1


def test_a_reply_shaped_unlike_the_sdk_is_an_llm_error_and_odd_usage_never_fails_a_good_reply():
    c, _ = client(SimpleNamespace(stop_reason="end_turn", content=None, usage=None))
    with pytest.raises(llm.LLMError, match="malformed"):
        c.chat("s", "u")
    c, _ = client(SimpleNamespace(stop_reason="end_turn", content=[text("fine")], usage=SimpleNamespace(input_tokens="x")))
    assert c.chat("s", "u") == "fine" and c.input_tokens == 0


def test_the_production_sdk_client_has_its_own_retries_off(monkeypatch):
    seen = {}

    class Built:
        messages = SimpleNamespace(create=lambda **kw: message(text("ok")))

    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: seen.update(kw) or Built())
    c = llm_anthropic.Client()
    assert c.chat("s", "u") == "ok" and seen == {"max_retries": 0}


SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"], "additionalProperties": False}


def test_a_schema_is_sent_as_structured_output_and_a_bare_true_sends_none():
    c, sdk = client(message(text("{}")), message(text("{}")), message(text("x")))
    c.chat("s", "u", json_reply=SCHEMA)
    c.chat("s", "u", json_reply=True)
    c.chat("s", "u")
    assert sdk.calls[0]["output_config"] == {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert sdk.calls[1]["output_config"] == sdk.calls[2]["output_config"] == {"effort": "low"}


def test_the_generators_and_filters_reply_schemas_are_what_their_parsers_accept():
    from app import safety
    assert set(generate.REPLY_SCHEMA["required"]) == {"premises", "threads"}
    assert generate.REPLY_SCHEMA["additionalProperties"] is False
    item = safety.REPLY_SCHEMA["properties"]["verdicts"]["items"]
    assert set(item["required"]) == {"id", "ok", "reason"} and item["additionalProperties"] is False
    sent = []

    class Spy(llm_anthropic.Client):
        def _once(self, system, user, json_reply, timeout):
            sent.append(json_reply)
            return "{}"

    Spy().chat("s", "u", json_reply=safety.REPLY_SCHEMA)
    assert sent == [safety.REPLY_SCHEMA]


def test_validate_reasks_once_and_counts_bad_replies_like_the_gemini_client():
    def need_ok(reply):
        assert reply == "good"

    c, sdk = client(message(text("bad")), message(text("good")), retries=2)
    assert c.chat("s", "u", validate=need_ok) == "good"
    assert (len(sdk.calls), c.bad_replies, c.retried, c.attempts) == (2, 1, 0, 2)  # a re-ask is not a retry
    c, sdk = client(message(text("bad")), message(text("worse")), retries=2)
    assert c.chat("s", "u", validate=need_ok) == "worse" and c.bad_replies == 2 and len(sdk.calls) == 2


def test_if_the_reask_fails_the_first_bad_reply_is_returned_not_the_error():
    def need_ok(reply):
        assert reply == "good"

    c, _ = client(message(text("partial")), status_error(anthropic.BadRequestError, 400), retries=2)
    assert c.chat("s", "u", validate=need_ok) == "partial" and c.failed == 1


def test_generator_runs_on_the_claude_client():
    import json
    reply = json.dumps({"premises": ["a", "b", "c"], "threads": [[f"line {j} of {i}." for j in range(5)] for i in range(3)]})
    c, sdk = client(message(text(reply)))
    batch = generate.GeminiGenerator(c).generate("neurotic", 0, 0, {"venue": "bar", "energy": "low", "age": "mixed"})
    assert len(batch.texts) == 3
    assert sdk.calls[0]["output_config"]["format"]["schema"] == generate.REPLY_SCHEMA


def test_safety_filter_runs_on_the_claude_client_with_validate():
    import json
    from app import safety
    safety.clear_cache()
    reply = json.dumps({"verdicts": [{"id": 0, "ok": True, "reason": "fine"}]})
    c, sdk = client(message(text(reply)))
    safety.prefetch(["the robot has feelings about spreadsheets"], "mixed", c)
    assert safety.check("the robot has feelings about spreadsheets", "mixed").ok
    assert sdk.calls[0]["output_config"]["format"]["schema"] == safety.REPLY_SCHEMA
    safety.clear_cache()


ENV_NAMES = ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "GEMINI_BASE_URL", "GEMINI_API_KEY",
             "JOKE_WRITER_PROVIDER", "JOKE_WRITER_MODEL", "SAFETY_CHECK_PROVIDER", "SAFETY_CHECK_MODEL")


@pytest.fixture
def env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_BASE_URL", "http://g/v1")
    return monkeypatch


def test_from_env_reads_each_role_by_its_own_name(env):
    env.setenv("GEMINI_API_KEY", "gk")
    env.setenv("JOKE_WRITER_PROVIDER", "anthropic")
    env.setenv("JOKE_WRITER_MODEL", "claude-haiku-5-5")
    env.setenv("SAFETY_CHECK_PROVIDER", "gemini")
    env.setenv("SAFETY_CHECK_MODEL", "flash-lite")
    writer, check = llm.from_env(role="joke_writer"), llm.from_env(role="safety_check")
    assert isinstance(writer, llm_anthropic.Client) and writer.model == "claude-haiku-5-5"
    assert isinstance(check, llm.Client) and (check.base_url, check.api_key, check.model) == ("http://g/v1", "gk", "flash-lite")


def test_the_safety_check_never_follows_the_joke_writer_to_anthropic(env):
    env.setenv("JOKE_WRITER_PROVIDER", "anthropic")
    env.setenv("SAFETY_CHECK_MODEL", "flash-lite")
    assert isinstance(llm.from_env(role="joke_writer"), llm_anthropic.Client)
    check = llm.from_env(role="safety_check")
    assert isinstance(check, llm.Client) and check.model == "flash-lite"
    env.delenv("SAFETY_CHECK_MODEL")
    env.setenv("LLM_MODEL", "legacy-flash")  # an unrenamed model name still means Gemini for the filter
    assert llm.from_env(role="safety_check").model == "legacy-flash"


def test_anthropic_defaults_to_the_haiku_model_and_never_takes_a_legacy_model_name(env):
    env.setenv("JOKE_WRITER_PROVIDER", "anthropic")
    env.setenv("LLM_MODEL", "legacy-flash")
    assert llm.from_env().model == llm_anthropic.MODEL == "claude-haiku-5-5"


def test_a_legacy_model_name_applies_only_when_no_new_name_picked_the_provider(env):
    env.setenv("LLM_MODEL", "old")
    assert llm.from_env().model == "old"  # an unrenamed .env keeps working
    env.setenv("JOKE_WRITER_PROVIDER", "gemini")  # half-renamed: the provider is new, the model is not
    with pytest.raises(RuntimeError, match="JOKE_WRITER_MODEL"):
        llm.from_env()


def test_new_names_win_over_old_ones(env):
    env.setenv("LLM_BASE_URL", "http://old/v1")
    env.setenv("LLM_MODEL", "old")
    env.setenv("JOKE_WRITER_MODEL", "new")
    c = llm.from_env()
    assert (c.base_url, c.model) == ("http://g/v1", "new")


def test_a_claude_model_on_the_gemini_provider_is_refused_with_the_fix(env):
    env.setenv("JOKE_WRITER_MODEL", "claude-haiku-5-5")  # half-renamed the other way: model new, provider missing
    with pytest.raises(RuntimeError, match="JOKE_WRITER_PROVIDER=anthropic"):
        llm.from_env()
    env.delenv("JOKE_WRITER_MODEL")
    env.setenv("LLM_MODEL", "claude-haiku-5-5")  # a legacy name must not smuggle one in either
    with pytest.raises(RuntimeError, match="Claude"):
        llm.from_env()


def test_a_gemini_model_on_the_anthropic_provider_is_refused(env):
    env.setenv("JOKE_WRITER_PROVIDER", "anthropic")
    env.setenv("JOKE_WRITER_MODEL", "gemini-flash-lite-latest")
    with pytest.raises(RuntimeError, match="not a Claude model"):
        llm.from_env()


def test_there_is_no_legacy_provider_name(env):
    env.setenv("LLM_PROVIDER", "anthropic")  # never shipped: it must not pick a provider
    env.setenv("LLM_MODEL", "m")
    assert isinstance(llm.from_env(), llm.Client)
    env.setenv("JOKE_WRITER_PROVIDER", "nope")
    with pytest.raises(RuntimeError):
        llm.from_env()


def test_provider_and_model_arguments_override_the_env(env):
    c = llm.from_env(5.0, provider="anthropic", model="claude-haiku-5-5", retries=1)
    assert isinstance(c, llm_anthropic.Client) and (c.timeout, c.retries) == (5.0, 1)
    assert llm.from_env(provider="openai", model="m").model == "m"  # the old provider name still works


def test_openai_client_counts_usage_and_a_odd_usage_block_never_fails_the_reply():
    import json
    body = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=json.dumps(body)))
    c = llm.Client("http://x/v1", "", "m", transport=transport)
    c.chat("s", "u")
    assert (c.input_tokens, c.output_tokens) == (12, 3)
    body["usage"] = {"prompt_tokens": "lots"}
    assert c.chat("s", "u") == "hi" and c.input_tokens == 12
    body["usage"] = ["not", "a", "dict"]
    assert c.chat("s", "u") == "hi"
