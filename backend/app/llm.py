"""A small OpenAI-compatible chat call. `LLM_BASE_URL` and `LLM_MODEL` choose the provider, `LLM_API_KEY` the account.

Used by the safety filter's LLM check and, later, the generator. Every failure is an `LLMError`, so callers can
tell "the model said no" from "the model could not answer". No retries here: a live show cannot wait on a 429.
"""
import os
from dataclasses import dataclass

import httpx


class LLMError(Exception):
    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass
class Client:
    base_url: str
    api_key: str
    model: str
    timeout: float = 8.0
    transport: httpx.BaseTransport | None = None  # tests pass a fake transport; production leaves it None

    def chat(self, system: str, user: str, *, json_reply: bool = False) -> str:
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if json_reply:
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            with httpx.Client(timeout=self.timeout, transport=self.transport) as http:
                resp = http.post(f"{self.base_url.rstrip('/')}/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            choice = resp.json()["choices"][0]
        except httpx.HTTPStatusError as e:
            raise LLMError(f"http {e.response.status_code}", e.response.status_code) from e
        except httpx.HTTPError as e:
            raise LLMError(f"transport: {type(e).__name__}") from e
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LLMError("malformed response") from e
        # A provider safety block arrives as a normal 200 with no usable text.
        if choice.get("finish_reason") == "content_filter":
            raise LLMError("blocked by provider")
        text = (choice.get("message") or {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise LLMError("empty reply")
        return text


def from_env(timeout: float = 8.0) -> Client:
    """Raises KeyError when LLM_BASE_URL or LLM_MODEL is unset; the key may be empty (local Ollama)."""
    return Client(os.environ["LLM_BASE_URL"], os.environ.get("LLM_API_KEY", ""), os.environ["LLM_MODEL"], timeout)
