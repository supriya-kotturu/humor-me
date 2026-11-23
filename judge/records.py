"""Records the judge produces: one Comparison per judge call, one Ranking per prompt."""
from dataclasses import asdict, dataclass

import numpy as np

from .bt import win_prob, win_prob_interval  # NEW

SIDES = ("a", "b")
STOP_REASONS = ("separated", "budget", "exhausted")


@dataclass(frozen=True)
class Comparison:
    """One judge call: jokes `a` and `b` (indices into the prompt's joke list), which one the
    judge saw first, and which one it said was funnier."""
    prompt_id: str
    a: int
    b: int
    shown_first: str
    winner: str
    judge: str

    def __post_init__(self):
        if self.shown_first not in SIDES or self.winner not in SIDES:
            raise ValueError(f"shown_first and winner must be 'a' or 'b', got {self.shown_first!r}, {self.winner!r}")

    def winner_loser(self) -> tuple[int, int]:
        return (self.a, self.b) if self.winner == "a" else (self.b, self.a)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Ranking:
    """Fitted Bradley-Terry strengths (theta, centered to mean 0), their standard errors, empirical
    Borda scores, and every comparison that went into them."""
    theta: np.ndarray
    se: np.ndarray
    borda: np.ndarray
    comparisons: list[Comparison]
    n_calls: int
    stop_reason: str

    def order(self) -> np.ndarray:
        """Joke indices, funniest first (ties keep index order)."""
        return np.argsort(-self.theta, kind="stable")

    def top(self, k: int) -> np.ndarray:
        """The k funniest, funniest first."""
        return self.order()[:k]

    def bottom(self, k: int) -> np.ndarray:
        """The k least funny, least funny first."""
        return self.order()[::-1][:k]

    def win_prob(self, i: int, j: int) -> float:  # NEW
        """P(joke i beats joke j) from the fitted theta, including pairs the judge never compared."""  # NEW
        return win_prob(self.theta, i, j)  # NEW

    def win_prob_interval(self, i: int, j: int, z: float = 1.96) -> tuple[float, float]:  # NEW
        """Rough band around win_prob(i, j) from the standard errors (see bt.win_prob_interval)."""  # NEW
        return win_prob_interval(self.theta, self.se, i, j, z)  # NEW

    def compare(self, i: int, j: int) -> int:  # NEW
        """Same convention as a judge: 0 if joke i is the funnier by theta, 1 if joke j is."""  # NEW
        return 0 if self.theta[i] >= self.theta[j] else 1  # NEW
