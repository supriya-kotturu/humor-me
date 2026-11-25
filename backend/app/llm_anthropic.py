"""Claude through the official `anthropic` SDK, with the same `chat` surface, `LLMError` and deadline/retry behaviour
as `app.llm.Client`, so the generator and the safety filter do not care which provider they hold. Selected with
`JOKE_WRITER_PROVIDER=anthropic` and `JOKE_WRITER_MODEL=claude-haiku-5-5`; the key comes from the SDK's own resolution
(`ANTHROPIC_API_KEY`), never from this repo's settings.
"""
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import anthropic

from app.llm import LLMError, run_with_retries

MODEL = "claude-haiku-5-5"
# Per million tokens, USD (Exp 24 cost estimates).
PRICE_PER_MTOK = {"input": 0.10, "output": 0.50}
# Adaptive thinking tokens count against max_tokens, so leave room for them; this is a cap, not a spend.
MAX_TOKENS = 4096


@dataclass
class Client:
    model: str = MODEL
    timeout: float = 8.0
    retries: int = 0
    backoff_s: float = 0.5
    deadline_s: float | None = None
    sleep: Callable[[float], None] = time.sleep
    sdk: Any = None  # tests pass a fake; production builds the SDK client on first use
    attempts: int = 0
    retried: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _built: Any = field(default=None, repr=False)

    def chat(self, system: str, user: str, *, json_reply: bool = False, temperature: float = 0.0) -> str:
        # `temperature` is accepted for the shared surface and never sent: Haiku 5.5 answers a non-default one with
        # a 400. `json_reply` needs nothing either: there is no prefill, and the prompts already demand JSON only.
        return run_with_retries(self, lambda timeout: self._once(system, user, timeout))

    def _client(self):
        if self.sdk is not None:
            return self.sdk
        if self._built is None:
            # The SDK's own retries (default 2) would run outside our deadline, so they are off.
            self._built = anthropic.Anthropic(max_retries=0)
        return self._built

    def _once(self, system: str, user: str, timeout: float) -> str:
        try:
            resp = self._client().messages.create(
                model=self.model, max_tokens=MAX_TOKENS, system=system,
                messages=[{"role": "user", "content": user}],
                # Effort defaults to medium with adaptive thinking on; low keeps a whole bit inside the deadline.
                output_config={"effort": "low"}, timeout=timeout)
        # Most specific first: RateLimitError is an APIStatusError, APITimeoutError an APIConnectionError.
        except anthropic.RateLimitError as e:
            raise LLMError("http 429", 429, retryable=True) from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"http {e.status_code}", e.status_code, retryable=e.status_code >= 500) from e
        except anthropic.APITimeoutError as e:
            raise LLMError("transport: timeout", retryable=True) from e
        except anthropic.APIConnectionError as e:
            raise LLMError("transport: connection", retryable=True) from e
        except TypeError as e:  # the SDK raises a bare TypeError when no credentials resolve
            if "authentication" in str(e):
                raise LLMError("no anthropic credentials: set ANTHROPIC_API_KEY") from e
            raise
        usage = getattr(resp, "usage", None)
        self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
        self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        # Content is only meaningful for a normal stop: a refusal or a cut-off reply must not be read as an answer.
        if resp.stop_reason == "refusal":
            raise LLMError("refused by the model")
        if resp.stop_reason == "max_tokens":
            raise LLMError("reply cut off at max_tokens")
        if resp.stop_reason != "end_turn":
            raise LLMError(f"unexpected stop_reason {resp.stop_reason!r}")
        text = "".join(b.text for b in resp.content if b.type == "text")
        if not text.strip():
            raise LLMError("empty reply")
        return text


def from_env(timeout: float = 8.0, *, model: str | None = None, **options) -> Client:
    return Client(model or MODEL, timeout, **options)
