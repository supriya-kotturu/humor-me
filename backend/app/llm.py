"""A small OpenAI-compatible chat call. `LLM_BASE_URL` and `LLM_MODEL` choose the provider, `LLM_API_KEY` the account.

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

    def chat(self, system: str, user: str, *, json_reply: bool = False, temperature: float = 0.0,
             validate: Callable[[str], None] | None = None) -> str:
        """`validate` raises on a reply the caller cannot use. That reply is asked for once more, inside the same
        retries and deadline and with no backoff (nothing is overloaded); a second bad reply is returned as it is,
        so the caller decides what a bad reply means (the safety filter fails closed on it)."""
        start = time.monotonic()
        bad = False
        for attempt in range(self.retries + 1):
            self.attempts += 1
            left = None if self.deadline_s is None else self.deadline_s - (time.monotonic() - start)
            try:
                text = self._once(system, user, json_reply, temperature,
                                  self.timeout if left is None else max(min(self.timeout, left), 0.1))
                if validate is None:
                    return text
                try:
                    validate(text)
                    return text
                except Exception:
                    self.bad_replies += 1
                    spent = time.monotonic() - start
                    if bad or attempt == self.retries or (
                            self.deadline_s is not None and spent + MIN_ATTEMPT_S > self.deadline_s):
                        return text
                    bad = True
                    self.retried += 1
                    continue
            except LLMError as e:
                wait = self.backoff_s * 2 ** attempt
                late = (self.deadline_s is not None
                        and time.monotonic() - start + wait + MIN_ATTEMPT_S > self.deadline_s)
                if not e.retryable or attempt == self.retries or late:
                    self.failed += 1
                    raise
                self.retried += 1
                self.sleep(wait)

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
            choice = json.loads(b"".join(chunks))["choices"][0]
        except httpx.HTTPStatusError as e:
            raise LLMError(f"http {e.response.status_code}", e.response.status_code) from e
        except httpx.HTTPError as e:
            raise LLMError(f"transport: {type(e).__name__}", retryable=True) from e
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMError("malformed response") from e
        # A provider safety block arrives as a normal 200 with no usable text.
        if choice.get("finish_reason") == "content_filter":
            raise LLMError("blocked by provider")
        text = (choice.get("message") or {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("empty reply")
        return text


def from_env(timeout: float = 8.0, **options) -> Client:
    """Raises RuntimeError when LLM_BASE_URL or LLM_MODEL is unset; the key may be empty (local Ollama)."""
    missing = [name for name in ("LLM_BASE_URL", "LLM_MODEL") if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"{' and '.join(missing)} not set: GENERATOR=llm needs both LLM_BASE_URL and LLM_MODEL in "
                           "backend/.env, and uvicorn only reads that file with --env-file .env")
    return Client(os.environ["LLM_BASE_URL"], os.environ.get("LLM_API_KEY", ""), os.environ["LLM_MODEL"], timeout,
                  **options)
