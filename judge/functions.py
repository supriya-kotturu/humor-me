"""Bradley-Terry fit with a prior (Hunter 2004 MM), diagonal-Fisher standard errors, and
empirical Borda scores. Math: judge/README.md."""
import numpy as np


def _wins_matrix(n_items: int, comparisons) -> np.ndarray:
    """W[i, j] = times i beat j."""
    W = np.zeros((n_items, n_items))
    if len(comparisons):
        w, l = np.asarray(comparisons, dtype=int).T
        np.add.at(W, (w, l), 1)
    return W


def fit_bt(n_items: int, comparisons, alpha: float = 1.0, iters: int = 500,
           tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """Fit theta from (winner, loser) pairs. Every item also gets `alpha` virtual wins and losses
    against a dummy with pi_0 = 1, which keeps theta finite and fixes the scale.

    Returns (theta centered to mean 0, theta uncentered); the uncentered one is what bt_se needs.
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0 (it keeps winless items finite)")
    W = _wins_matrix(n_items, comparisons)
    N = W + W.T
    wins = W.sum(axis=1) + alpha
    pi = np.ones(n_items)
    for _ in range(iters):
        denom = (N / (pi[:, None] + pi[None, :])).sum(axis=1) + 2 * alpha / (pi + 1)
        new = wins / denom
        done = np.max(np.abs(np.log(new) - np.log(pi))) < tol
        pi = new
        if done:
            break
    theta = np.log(pi)
    return theta - theta.mean(), theta


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def bt_se(theta_raw: np.ndarray, N: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Standard errors from the diagonal of the Fisher information. N[i, j] = games between i
    and j (symmetric, zero diagonal); theta_raw is the uncentered fit."""
    p = _sigmoid(theta_raw[:, None] - theta_raw[None, :])
    p0 = _sigmoid(theta_raw)
    info = (N * p * (1 - p)).sum(axis=1) + 2 * alpha * p0 * (1 - p0)
    return 1.0 / np.sqrt(info)


def games_matrix(n_items: int, comparisons) -> np.ndarray:
    """N[i, j] = games between i and j, from (winner, loser) pairs."""
    W = _wins_matrix(n_items, comparisons)
    return W + W.T


def borda_scores(n_items: int, comparisons) -> np.ndarray:
    """Mean over the opponents each item has played of its win rate against them; NaN if it
    has played no one."""
    W = _wins_matrix(n_items, comparisons)
    N = W + W.T
    played = N > 0
    rates = np.divide(W, N, out=np.zeros_like(W), where=played)
    n_opp = played.sum(axis=1)
    with np.errstate(invalid="ignore"):
        return np.where(n_opp > 0, rates.sum(axis=1) / np.maximum(n_opp, 1), np.nan)
