"""Which pairs to compare: a random warm start, every pair, or the adaptive (LUCB) next pair.

Pairs come back as (a, b) with no position meaning; rank() decides which is shown first.
"""
import numpy as np


def warm_start_pairs(n: int, rounds: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """`rounds` random perfect matchings; with odd n, one random joke sits out each round."""
    pairs = []
    for _ in range(rounds):
        perm = rng.permutation(n)[: n - n % 2]
        pairs += [(int(perm[i]), int(perm[i + 1])) for i in range(0, len(perm), 2)]
    return pairs


def all_pairs(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """Every unordered pair once, as (i, j) with i < j, in shuffled order."""
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return [pairs[k] for k in rng.permutation(len(pairs))]


def adaptive_next(theta: np.ndarray, se: np.ndarray, cutoffs_m, beta: float) -> tuple[int, int] | None:
    """LUCB for top-k identification at several cut-offs.

    Cut-off m splits the current top m by theta (T) from the rest. h = the member of T with the
    lowest lower bound, l = the outsider with the highest upper bound. The cut-off is separated
    when LCB_h > UCB_l. Returns None if every cut-off is separated, else (h, l) of the cut-off
    whose bounds overlap most.
    """
    lcb, ucb = theta - beta * se, theta + beta * se
    order = np.argsort(-theta, kind="stable")
    best, best_overlap = None, -np.inf
    for m in cutoffs_m:
        top, rest = order[:m], order[m:]
        h, l = top[np.argmin(lcb[top])], rest[np.argmax(ucb[rest])]
        overlap = ucb[l] - lcb[h]
        if lcb[h] <= ucb[l] and overlap > best_overlap:
            best, best_overlap = (int(h), int(l)), overlap
    return best
