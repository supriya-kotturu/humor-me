"""rank(): rank one prompt's candidate jokes from pairwise judge calls."""
import hashlib
import json
import pathlib
from typing import Sequence

import numpy as np

from .backends import Judge
from .functions import borda_scores, bt_se, fit_bt, games_matrix
from .select import adaptive_next, all_pairs, warm_start_pairs
from .orders import Comparison, Ranking

METHODS = ("adaptive", "all_pairs")


def _ask(prompt: str, prompt_id: str, jokes: Sequence[str], judge: Judge, a: int, b: int,
         rng: np.random.Generator) -> Comparison:
    """The only place a judge is called: a fair coin decides which joke is shown first."""
    shown_first = "a" if rng.random() < 0.5 else "b"
    first, second = (a, b) if shown_first == "a" else (b, a)
    answer = judge.compare(prompt, jokes[first], jokes[second])
    if answer not in (0, 1):
        raise ValueError(f"judge {judge.name!r} returned {answer!r}, expected 0 or 1")
    winner_index = first if answer == 0 else second
    return Comparison(prompt_id=prompt_id, a=a, b=b, shown_first=shown_first,
                      winner="a" if winner_index == a else "b", judge=judge.name)


def _fit(n: int, comparisons: list[Comparison], alpha: float) -> tuple[np.ndarray, np.ndarray]:
    pairs = [c.winner_loser() for c in comparisons]
    theta, raw = fit_bt(n, pairs, alpha=alpha)
    return theta, bt_se(raw, games_matrix(n, pairs), alpha=alpha)


def rank(prompt: str, jokes: Sequence[str], judge: Judge, *, method: str = "adaptive",
         budget: int | None = 80, top_cutoffs: Sequence[int] = (5, 10),
         bottom_cutoffs: Sequence[int] = (5,), warm_rounds: int = 2, beta: float = 1.96,
         alpha: float = 1.0, seed: int = 0, prompt_id: str | None = None) -> Ranking:
    """Rank `jokes` for `prompt` with Bradley-Terry fitted to judge comparisons.

    method="all_pairs": every pair once, in random order, until done or `budget` calls
    (None = no limit). method="adaptive": `warm_rounds` random matchings, then LUCB picks the
    pair at the most uncertain top/bottom cut-off until every cut-off is separated or the budget
    is spent. Same seed, same judge state -> same comparisons.
    """
    n = len(jokes)
    if n < 2:
        raise ValueError(f"need at least 2 jokes, got {n}")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if budget is not None and budget < 0:
        raise ValueError(f"budget must be >= 0 or None, got {budget}")
    if method == "adaptive":
        if budget is None:
            raise ValueError("method='adaptive' needs a budget: tied jokes may never separate")
        cutoffs_m = sorted({*top_cutoffs, *(n - k for k in bottom_cutoffs)})
        if not all(0 < m < n for m in cutoffs_m):
            raise ValueError(f"cut-offs must give 0 < m < {n}: top {tuple(top_cutoffs)}, bottom {tuple(bottom_cutoffs)}")
    if prompt_id is None:
        prompt_id = hashlib.sha1(prompt.encode()).hexdigest()[:12]
    rng = np.random.default_rng(seed)
    comparisons: list[Comparison] = []

    if method == "all_pairs":
        pairs = all_pairs(n, rng)
        if budget is not None:
            pairs = pairs[:budget]
        comparisons = [_ask(prompt, prompt_id, jokes, judge, a, b, rng) for a, b in pairs]
        stop_reason = "exhausted" if len(comparisons) == n * (n - 1) // 2 else "budget"
    else:
        for a, b in warm_start_pairs(n, warm_rounds, rng)[:budget]:
            comparisons.append(_ask(prompt, prompt_id, jokes, judge, a, b, rng))
        while True:
            theta, se = _fit(n, comparisons, alpha)
            pair = adaptive_next(theta, se, cutoffs_m, beta)
            if pair is None:
                stop_reason = "separated"
                break
            if len(comparisons) >= budget:
                stop_reason = "budget"
                break
            comparisons.append(_ask(prompt, prompt_id, jokes, judge, *pair, rng))

    theta, se = _fit(n, comparisons, alpha)
    pairs = [c.winner_loser() for c in comparisons]
    return Ranking(theta=theta, se=se, borda=borda_scores(n, pairs), comparisons=comparisons,
                   n_calls=len(comparisons), stop_reason=stop_reason)


def write_judgments(path, comparisons: Sequence[Comparison]) -> pathlib.Path:
    """One JSON object per comparison per line."""
    path = pathlib.Path(path)
    path.write_text("".join(json.dumps(c.to_dict()) + "\n" for c in comparisons))
    return path
