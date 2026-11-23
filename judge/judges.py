"""Judges: given a prompt and two jokes, say which is funnier.

A judge is any object with a `name` and `compare(prompt, first, second) -> int` that returns 0
if `first` is funnier and 1 if `second` is. Two exist: SimulatedJudge (known strengths, for
tests) and LLMJudge (llama3.1:8b on Ollama). A judge never randomizes the order itself; rank()
flips the coin for which joke is shown first and records it, so position bias can be measured
and averaged out in one place.
"""
import json  
import re 
import time  
import urllib.error  
import urllib.request  
from typing import Callable, Sequence  

import numpy as np


class SimulatedJudge:
    """A Bradley-Terry judge with known true strengths, for tests and simulations.

    Jokes are identified by looking up `first` / `second` in `jokes`. The first-shown joke wins
    with probability sigma(theta_first - theta_second + first_bias); first_bias > 0 is a judge
    that favors whatever it reads first.
    """

    def __init__(self, theta_true: Sequence[float], jokes: Sequence[str], seed: int = 0,
                 first_bias: float = 0.0, name: str = "sim"):
        if len(theta_true) != len(jokes):
            raise ValueError(f"{len(theta_true)} strengths for {len(jokes)} jokes")
        if len(set(jokes)) != len(jokes):
            raise ValueError("jokes must be distinct strings: the judge identifies them by text")
        self.theta_true = np.asarray(theta_true, dtype=float)
        self.index = {joke: i for i, joke in enumerate(jokes)}
        self.first_bias = first_bias
        self.name = name
        self.rng = np.random.default_rng(seed)

    def compare(self, prompt: str, first: str, second: str) -> int:
        d = self.theta_true[self.index[first]] - self.theta_true[self.index[second]] + self.first_bias
        return 0 if self.rng.random() < 1.0 / (1.0 + np.exp(-d)) else 1


SYSTEM = "You are a careful judge of comedy. You compare two jokes written for the same prompt."  # NEW
TEMPLATE = """Prompt: {prompt}

Joke A: {first}

Joke B: {second}

Which joke is funnier as a response to this prompt? Answer with exactly one letter: A or B."""


def ollama_chat(model: str, system: str, user: str, url: str = "http://localhost:11434/api/chat",  # NEW
                timeout: float = 300) -> str:  # NEW
    """One non-streaming, temperature-0 chat call to a local Ollama server; returns the reply text."""  # NEW
    body = json.dumps({"model": model, "stream": False, "options": {"temperature": 0},  # NEW
                       "messages": [{"role": "system", "content": system},  # NEW
                                    {"role": "user", "content": user}]}).encode()  # NEW
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})  # NEW
    with urllib.request.urlopen(req, timeout=timeout) as r:  # NEW
        return json.load(r)["message"]["content"] or ""  # NEW


def parse_ab(reply: str) -> int | None:  # NEW
    """0 for A, 1 for B, None if the reply has no standalone A/B. Reasoning inside <think> is
    ignored and the last A/B wins, so "A is clever, but B" parses as B."""
    final = re.sub(r"<think>.*?(</think>|$)", "", reply, flags=re.S)  # NEW
    found = re.findall(r"\b([AB])\b", final.upper())  # NEW
    return None if not found else (0 if found[-1] == "A" else 1)  # NEW


class LLMJudge:
    """An LLM asked "which is funnier?", by default llama3.1:8b on a local Ollama server
    (`ollama serve`, `ollama pull llama3.1:8b`).

    `first` is labelled A and `second` B, in the order given: rank() flips the coin for which
    joke is shown first, so this class never shuffles. `chat(model, system, user) -> str` is
    the transport; pass a fake one in tests. With `log_path`, every answer is appended as JSON.
    """

    def __init__(self, model: str = "llama3.1:8b", name: str | None = None, *,  # NEW
                 chat: Callable[[str, str, str], str] = ollama_chat, system: str = SYSTEM,  # NEW
                 template: str = TEMPLATE, retries: int = 3, log_path: str | None = None):  # NEW
        self.model = model
        self.name = name or f"llm:{model}"
        self.chat, self.system, self.template, self.retries = chat, system, template, retries  # NEW
        self.log_path = log_path  # NEW

    def compare(self, prompt: str, first: str, second: str) -> int:
        """0 if `first` (A) is funnier, 1 if `second` (B) is. Retries a few times on an
        unparseable reply or a transient error, then raises."""
        user = self.template.format(prompt=prompt, first=first, second=second)  # NEW
        reply = ""  # NEW
        for attempt in range(self.retries):  # NEW
            try:  # NEW
                reply = self.chat(self.model, self.system, user)  # NEW
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:  # NEW
                reply = f"<error: {e!r}>"  # NEW
            answer = parse_ab(reply)  # NEW
            if answer is not None:  # NEW
                if self.log_path:  # NEW
                    with open(self.log_path, "a", encoding="utf-8") as f:  # NEW
                        f.write(json.dumps({"model": self.model, "prompt": prompt, "first": first,  # NEW
                                            "second": second, "answer": "AB"[answer], "raw": reply}) + "\n")  # NEW
                return answer  # NEW
            if attempt < self.retries - 1:  # NEW
                time.sleep(2 ** attempt)  # NEW
        raise RuntimeError(f"{self.name}: no A/B answer after {self.retries} tries; last reply: {reply!r}")  # NEW
