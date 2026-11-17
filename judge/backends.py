"""Judges: anything that, given a prompt and two jokes, says which is funnier.

A judge never randomizes the order itself; rank() flips the coin for which joke is shown first
and records it, so position bias can be measured and averaged out in one place.
"""
from typing import Protocol, Sequence

import numpy as np


class Judge(Protocol):
    name: str

    def compare(self, prompt: str, first: str, second: str) -> int:
        """0 if `first` is funnier, 1 if `second` is."""
        ...


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


class LLMJudge:
    """An LLM asked "which is funnier?". Not implemented yet: no network calls in this package."""

    def __init__(self, model: str, name: str | None = None):
        self.model = model
        self.name = name or f"llm:{model}"

    def compare(self, prompt: str, first: str, second: str) -> int:
        """To implement: send the model the prompt and both jokes, labelled "1" (`first`) and
        "2" (`second`), and ask which is funnier, answering with "1" or "2" only. Parse the reply
        (strip whitespace; accept a reply that is exactly or starts with "1"/"2"), retry a few
        times on a parse failure or transient API error, then raise. Return 0 for "1", 1 for "2".

        Keep the order as given: do NOT shuffle `first` / `second` here. rank() already flips a
        seeded coin per call and records which joke was shown first.
        """
        raise NotImplementedError("LLMJudge.compare: see the docstring for what to implement")
