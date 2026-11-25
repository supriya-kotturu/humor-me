"""A small OpenAI-compatible chat call. `GEMINI_BASE_URL` picks the endpoint, `JOKE_WRITER_MODEL` and
`SAFETY_CHECK_MODEL` the models, `GEMINI_API_KEY` the account (backend/sample.env); `from_env` also builds the Claude
client (app.llm_anthropic).

Used by the safety filter's LLM check and the generator. Every failure is an `LLMError`, so callers can tell "the
model said no" from "the model could not answer". Retries are opt-in and bounded by a deadline: a live show can wait
a second or two on a 429, not longer (the bridge line covers that, docs/plans/live-loop.md T9).
"""
import json
import os
import time
from dataclasses import dataclass
from typing import Callable

import httpx


# Overloaded or rate-limited: the same request can succeed a moment later. 29% of Exp 19's attempts were 429 or 503.
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
MIN_ATTEMPT_S = 0.5  # an attempt with less time than this left cannot finish, so it is not started


def strip_fence(reply: str) -> str:
    """The text of a reply that the model wrapped in a markdown code fence."""
    reply = reply.strip()
    if reply.startswith("```"):
        reply = reply.strip("`").removeprefix("json").strip()
    return reply


class LLMError(Exception):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable or status in RETRY_STATUS


def run_with_retries(client, once: Callable[[float], str], validate: Callable[[str], None] | None = None) -> str:
    """The bounded retry loop shared by every provider's client. `client` carries `timeout`, `retries`, `backoff_s`,
    `deadline_s`, `sleep` and the `attempts`/`retried`/`failed`/`bad_replies` counters; `once(timeout)` makes one attempt.

    `validate` raises on a reply the caller cannot use. That reply is asked for once more, inside the same retries and
    deadline and with no backoff (nothing is overloaded); a second bad reply is returned as it is, so the caller
    decides what a bad reply means (the safety filter fails closed on it). If the re-ask itself fails, the first bad
    reply is returned instead of the error: a partial reply still clears the lines it does judge."""
    start = time.monotonic()
    bad = False
    last_bad: str | None = None
    for attempt in range(client.retries + 1):
        client.attempts += 1
        left = None if client.deadline_s is None else client.deadline_s - (time.monotonic() - start)
        try:
            text = once(client.timeout if left is None else max(min(client.timeout, left), 0.1))
            if validate is None:
                return text
            try:
                validate(text)
                return text
            except Exception:
                client.bad_replies += 1
                spent = time.monotonic() - start
                if bad or attempt == client.retries or (
                        client.deadline_s is not None and spent + MIN_ATTEMPT_S > client.deadline_s):
                    return text
                bad, last_bad = True, text
                continue  # a re-ask, not a retry: `retried` counts provider failures only
        except LLMError as e:
            wait = client.backoff_s * 2 ** attempt
            late = (client.deadline_s is not None
                    and time.monotonic() - start + wait + MIN_ATTEMPT_S > client.deadline_s)
            if not e.retryable or attempt == client.retries or late:
                client.failed += 1
                if last_bad is not None:
                    return last_bad
                raise
            client.retried += 1
            client.sleep(wait)


@dataclass
class Client:
    base_url: str
    api_key: str
    model: str
    timeout: float = 8.0
    transport: httpx.BaseTransport | None = None  # tests pass a fake transport; production leaves it None
    retries: int = 0
    backoff_s: float = 0.5
    deadline_s: float | None = None  # wall time across all attempts, each attempt's timeout included
    sleep: Callable[[float], None] = time.sleep
    # Quota bookkeeping: a 429 still counts against the provider's limit, so every request is an attempt.
    attempts: int = 0
    retried: int = 0
    failed: int = 0
    bad_replies: int = 0
    # Token bookkeeping for cost estimates (Exp 24): summed over every reply that carried a usage block.
    input_tokens: int = 0
    output_tokens: int = 0

    def chat(self, system: str, user: str, *, json_reply: bool = False, temperature: float = 0.0,
             validate: Callable[[str], None] | None = None) -> str:
        return run_with_retries(self, lambda timeout: self._once(system, user, json_reply, temperature, timeout),
                                validate)

    def _once(self, system: str, user: str, json_reply: bool, temperature: float, timeout: float) -> str:
        body = {"model": self.model, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_reply:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        # httpx's timeouts are per phase and the read one is per chunk, so they alone do not bound a slow-dripping
        # reply: the body is read in chunks and cut off once `timeout` has passed in total. What remains unbounded is
        # one stalled chunk, which is still capped by the read timeout.
        stop_at = time.monotonic() + timeout
        try:
            with httpx.Client(timeout=httpx.Timeout(timeout, connect=min(2.0, timeout)), transport=self.transport) as http:
                with http.stream("POST", f"{self.base_url.rstrip('/')}/chat/completions", json=body,
                                 headers=headers) as resp:
                    resp.raise_for_status()
                    chunks = []
                    for chunk in resp.iter_bytes():
                        if time.monotonic() > stop_at:
                            raise LLMError("deadline passed while reading", retryable=True)
                        chunks.append(chunk)
            body = json.loads(b"".join(chunks))
            choice = body["choices"][0]
        except httpx.HTTPStatusError as e:
            raise LLMError(f"http {e.response.status_code}", e.response.status_code) from e
        except httpx.HTTPError as e:
            raise LLMError(f"transport: {type(e).__name__}", retryable=True) from e
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMError("malformed response") from e
        try:  # bookkeeping must never fail a good reply
            usage = body.get("usage") or {}
            self.input_tokens += int(usage.get("prompt_tokens") or 0)
            self.output_tokens += int(usage.get("completion_tokens") or 0)
        except (AttributeError, TypeError, ValueError):
            pass
        # A provider safety block arrives as a normal 200 with no usable text.
        if choice.get("finish_reason") == "content_filter":
            raise LLMError("blocked by provider")
        text = (choice.get("message") or {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("empty reply")
        return text


# Settings are named by the job a model does (backend/sample.env). Older .env files used one model for everything;
# their names are still read when the new one is unset, so a teammate's .env keeps working until they rename it.
# There is no legacy provider: `LLM_PROVIDER` never shipped, and a legacy model name is only ever a Gemini one.
LEGACY = {"GEMINI_BASE_URL": "LLM_BASE_URL", "GEMINI_API_KEY": "LLM_API_KEY", "JOKE_SOURCE": "GENERATOR",
          "TRANSCRIPT_PIPELINE_MODEL": "PIPELINE_MODEL",
          "JOKE_WRITER_MODEL": "LLM_MODEL", "SAFETY_CHECK_MODEL": "LLM_MODEL"}
ROLES = {"joke_writer": "JOKE_WRITER", "safety_check": "SAFETY_CHECK"}
PROVIDERS = ("gemini", "anthropic")  # gemini = any OpenAI-compatible endpoint at GEMINI_BASE_URL (Groq, Ollama too)


def setting(name: str, default: str = "") -> str:
    return os.environ.get(name) or os.environ.get(LEGACY.get(name, ""), "") or default


def from_env(timeout: float = 8.0, *, role: str = "joke_writer", provider: str | None = None, model: str | None = None,
             **options):
    """The client for one job: `role` is `joke_writer` (writes the live jokes) or `safety_check` (judges each line
    before the voice). Each role reads only its own `<ROLE>_PROVIDER` (`gemini`, the default, or `anthropic`, the
    official SDK in app.llm_anthropic) and `<ROLE>_MODEL`, so the safety check never follows the joke writer to
    another provider. Both clients expose `chat(system, user, *, json_reply, temperature, validate)` and raise
    `LLMError`. `provider` and `model` override the env (the Exp 24 harness compares several in one process).
    Raises RuntimeError for a missing setting, or a model name that belongs to the other provider."""
    prefix = ROLES[role]
    named = provider or os.environ.get(f"{prefix}_PROVIDER")  # a provider from any new name, not the default
    provider = named or "gemini"
    provider = "gemini" if provider == "openai" else provider  # the old name for the same OpenAI-compatible client
    if provider not in PROVIDERS:
        raise RuntimeError(f"{prefix}_PROVIDER={provider!r}: expected one of {PROVIDERS}")
    model = model or os.environ.get(f"{prefix}_MODEL", "")
    if provider == "anthropic":
        from app import llm_anthropic  # lazy: llm_anthropic imports this module for LLMError
        model = model or llm_anthropic.MODEL
        if not model.startswith("claude"):
            raise RuntimeError(f"{prefix}_MODEL={model!r} is not a Claude model but the provider is anthropic")
        return llm_anthropic.from_env(timeout, model=model, **options)
    if not model and not named:
        model = os.environ.get("LLM_MODEL", "")  # an unrenamed .env: the old name, a Gemini model by construction
    if model.startswith("claude"):
        raise RuntimeError(f"{prefix}_MODEL={model!r} is a Claude model but the provider is gemini: "
                           f"set {prefix}_PROVIDER=anthropic")
    base = setting("GEMINI_BASE_URL")
    missing = [name for name, value in (("GEMINI_BASE_URL", base), (f"{prefix}_MODEL", model)) if not value]
    if missing:
        raise RuntimeError(f"{' and '.join(missing)} not set: JOKE_SOURCE=llm needs them in backend/.env "
                           "(see backend/sample.env), and uvicorn only reads that file with --env-file .env")
    return Client(base, setting("GEMINI_API_KEY"), model, timeout, **options)
