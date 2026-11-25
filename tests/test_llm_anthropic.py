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
    assert len(sdk.calls) == 1 and sdk.calls[0]["timeout"] <= 0.3


def test_generator_runs_on_the_claude_client():
    import json
    reply = json.dumps({"premises": ["a", "b", "c"], "threads": [[f"line {j} of {i}." for j in range(5)] for i in range(3)]})
    c, _ = client(message(text(reply)))
    batch = generate.GeminiGenerator(c).generate("neurotic", 0, 0, {"venue": "bar", "energy": "low", "age": "mixed"})
    assert len(batch.texts) == 3


def test_from_env_reads_each_role_by_its_own_name(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GEMINI_BASE_URL", "http://g/v1")
    monkeypatch.setenv("GEMINI_API_KEY", "gk")
    monkeypatch.setenv("JOKE_WRITER_PROVIDER", "anthropic")
    monkeypatch.setenv("JOKE_WRITER_MODEL", "claude-haiku-5-5")
    monkeypatch.setenv("SAFETY_CHECK_PROVIDER", "gemini")
    monkeypatch.setenv("SAFETY_CHECK_MODEL", "flash-lite")
    writer, check = llm.from_env(role="joke_writer"), llm.from_env(role="safety_check")
    assert isinstance(writer, llm_anthropic.Client) and writer.model == "claude-haiku-5-5"
    assert isinstance(check, llm.Client) and (check.base_url, check.api_key, check.model) == ("http://g/v1", "gk", "flash-lite")


def test_new_names_win_over_old_ones(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://old/v1")
    monkeypatch.setenv("LLM_MODEL", "old")
    monkeypatch.setenv("GEMINI_BASE_URL", "http://new/v1")
    monkeypatch.setenv("JOKE_WRITER_MODEL", "new")
    monkeypatch.delenv("JOKE_WRITER_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    c = llm.from_env()
    assert (c.base_url, c.model) == ("http://new/v1", "new")


def test_from_env_dispatches_on_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-haiku-5-5")
    c = llm.from_env(5.0, retries=1)
    assert isinstance(c, llm_anthropic.Client) and (c.model, c.timeout, c.retries) == ("claude-haiku-5-5", 5.0, 1)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_BASE_URL", "http://x/v1")
    monkeypatch.setenv("LLM_MODEL", "m")
    assert isinstance(llm.from_env(), llm.Client)
    monkeypatch.setenv("LLM_PROVIDER", "nope")
    with pytest.raises(RuntimeError):
        llm.from_env()


def test_openai_client_counts_usage():
    import json
    body = {"choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 12, "completion_tokens": 3}}
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=json.dumps(body)))
    c = llm.Client("http://x/v1", "", "m", transport=transport)
    c.chat("s", "u")
    assert (c.input_tokens, c.output_tokens) == (12, 3)
