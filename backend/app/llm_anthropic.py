"""Claude through the official `anthropic` SDK, with the same `chat` surface, `LLMError` and deadline/retry behaviour
as `app.llm.Client` (one shared loop, `llm.run_with_retries`), so the generator and the safety filter do not care which
provider they hold. Selected with `JOKE_WRITER_PROVIDER=anthropic` (and `SAFETY_CHECK_PROVIDER` for the filter); the
model defaults to `claude-haiku-5-5`; the key comes from the SDK's own resolution (`ANTHROPIC_API_KEY`), never from this
repo's settings.
"""
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
RETRY_STATUS_EXTRA = frozenset({408})  # a request timeout is the server saying "send it again"


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
    bad_replies: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    _built: Any = field(default=None, repr=False)

    def chat(self, system: str, user: str, *, json_reply: bool | dict = False, temperature: float = 0.0,
             validate: Callable[[str], None] | None = None) -> str:
        # `temperature` is accepted for the shared surface and never sent: Haiku 5.5 answers a non-default one with
        # a 400. `json_reply` may be the reply's JSON schema (a dict): it is sent as structured output, since there is
        # no prefill to force JSON. A bare True sends nothing, and the prompt's "JSON only" has to do.
        return run_with_retries(self, lambda timeout: self._once(system, user, json_reply, timeout), validate)

    def _client(self):
        if self.sdk is not None:
            return self.sdk
        if self._built is None:
            # The SDK's own retries (default 2) would run outside our deadline, so they are off.
            self._built = anthropic.Anthropic(max_retries=0)
        return self._built

    def _once(self, system: str, user: str, json_reply: bool | dict, timeout: float) -> str:
        # Effort defaults to medium with adaptive thinking on; low keeps a whole bit inside the deadline.
        output_config: dict = {"effort": "low"}
        if isinstance(json_reply, dict):
            output_config["format"] = {"type": "json_schema", "schema": json_reply}
        try:
            resp = self._client().messages.create(
                model=self.model, max_tokens=MAX_TOKENS, system=system,
                messages=[{"role": "user", "content": user}], output_config=output_config,
                timeout=anthropic.Timeout(timeout, connect=min(2.0, timeout)))
        # Most specific first: RateLimitError is an APIStatusError, APITimeoutError an APIConnectionError.
        except anthropic.RateLimitError as e:
            raise LLMError("http 429", 429, retryable=True) from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"http {e.status_code}", e.status_code,
                           retryable=e.status_code >= 500 or e.status_code in RETRY_STATUS_EXTRA) from e
        except anthropic.APITimeoutError as e:
            raise LLMError("transport: timeout", retryable=True) from e
        except anthropic.APIConnectionError as e:
            raise LLMError("transport: connection", retryable=True) from e
        except TypeError as e:  # the SDK raises a bare TypeError when no credentials resolve
            if "authentication" in str(e):
                raise LLMError("no anthropic credentials: set ANTHROPIC_API_KEY") from e
            raise LLMError(f"unexpected: {type(e).__name__}") from e
        except Exception as e:  # callers handle LLMError only: nothing else may escape a chat call
            raise LLMError(f"unexpected: {type(e).__name__}") from e
        try:
            return self._text(resp)
        except LLMError:
            raise
        except Exception as e:  # a reply shaped unlike the SDK's own types
            raise LLMError(f"malformed response: {type(e).__name__}") from e

    def _text(self, resp) -> str:
        try:  # bookkeeping must never fail a good reply
            self.input_tokens += int(getattr(resp.usage, "input_tokens", 0) or 0)
            self.output_tokens += int(getattr(resp.usage, "output_tokens", 0) or 0)
        except Exception:
            pass
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
