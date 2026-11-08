"""What does exploring at ε cost the director? LinUCB wrapped in ε-soft picks, against plain LinUCB.

Usage:
    python -m sim.evaluate_epsilon

Exp 18 (docs/plans/live-loop.md, T1). Each seed is one career of `shows` shows, with the same contexts and
environment seeds for every ε, so the cost is a paired difference per seed. Rooms: judge and human taste,
each with the gaps between modes as they are, halved and doubled. Writes reports/metrics/epsilon_cost.json
(cached by params). Importable: run(), table(), criteria().
"""
import copy
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from sim.cache import run_or_load
from sim.evaluate_policies import career, ci
from sim.evaluate_tastes import tastes
from sim.policies import EpsilonSoft, LinUCB, prior_means

EPSILONS = (0.0, 0.1, 0.2)
GAP_SCALES = (0.5, 1.0, 2.0)
HORIZONS = ((5, 1000), (30, 200))  # (shows, seeds)
BAR = 1.0  # laughs per show that ε = 0.2 may lose at 5 shows, at the far end of the CI
EXPLORE_SEED = 10_000  # keeps the wrapper's draws apart from the policy's and the environment's


def scale_gaps(audience: dict, scale: float) -> dict:
    """Scale each mode's `base` around the mean of `base`; venue, age, energy and dynamics stay as they are."""
    out = copy.deepcopy(audience)
    mean = float(np.mean(list(out["base"].values())))
    out["base"] = {m: mean + scale * (b - mean) for m, b in out["base"].items()}
    return out


def variants() -> dict:
    return {f"{taste}, gaps x{scale:g}": scale_gaps(audience, scale)
            for taste, audience in tastes().items() for scale in GAP_SCALES}


def career_laughs(audience: dict, shows: int, seed: int, eps: float) -> tuple[float, int]:
    """Mean laughs per show over one career, and how many picks differed from the greedy mode."""
    made = []

    def make(env, s):
        made.append(EpsilonSoft(LinUCB(prior_means(env, "uniform", seed=s), seed=s), eps, env.n_modes,
                                seed=s + EXPLORE_SEED))
        return made[0]

    return float(career(make, shows, seed, audience=audience)[0].mean()), made[0].explored


def _job(args):
    return career_laughs(*args)


def paired(audience: dict, shows: int, seeds: int, epsilons: tuple, pool) -> dict:
    """Laughs per show for each ε on the same seeds, and each ε's paired difference from ε = 0."""
    laughs, explored = {}, {}
    for eps in epsilons:
        out = list(pool.map(_job, [(audience, shows, seed, eps) for seed in range(seeds)], chunksize=20))
        laughs[eps], explored[eps] = np.array([o[0] for o in out]), np.array([o[1] for o in out])
    base = laughs[0.0]
    return {str(eps): {"laughs": ci(laughs[eps]), "diff": ci(laughs[eps] - base),
                       "diff_pct": float(100 * (laughs[eps] - base).mean() / base.mean()),
                       "non_greedy": float(explored[eps].mean())} for eps in epsilons}


def run(horizons: list = [list(h) for h in HORIZONS], epsilons: list = list(EPSILONS)) -> dict:
    out = {"base": {}, "variants": {}}
    with ProcessPoolExecutor() as pool:
        for name, audience in variants().items():
            out["base"][name] = audience["base"]
            out["variants"][name] = {str(shows): paired(audience, shows, seeds, tuple(epsilons), pool)
                                     for shows, seeds in horizons}
            print(name, flush=True)
    out["criteria"] = criteria(out)
    return out


def criteria(res: dict) -> dict:
    """ε = 0.2 passes a variant if its loss at 5 shows, at the far end of the 95% CI, is within the bar."""
    rows = {}
    for name, by_horizon in res["variants"].items():
        mean, half = by_horizon["5"]["0.2"]["diff"]
        rows[name] = {"far_end": mean - half, "pass": bool(mean - half >= -BAR)}
    return {"variants": rows, "all_pass": all(r["pass"] for r in rows.values())}


def table(res: dict) -> str:
    out = ["| Room | Shows | ε | Laughs per show | Change vs ε = 0 (paired, 95% CI) | Change, % | Non-greedy picks per career | Passes the bar |",
           "|---|---|---|---|---|---|---|---|"]
    for name, by_horizon in res["variants"].items():
        for shows, by_eps in by_horizon.items():
            for eps, r in by_eps.items():
                verdict = ""
                if shows == "5" and eps == "0.2":
                    c = res["criteria"]["variants"][name]
                    verdict = f"{'yes' if c['pass'] else 'NO'} (far end {c['far_end']:+.2f})"
                diff = "–" if eps == "0.0" else f"{r['diff'][0]:+.2f} ± {r['diff'][1]:.2f}"
                pct = "–" if eps == "0.0" else f"{r['diff_pct']:+.1f}%"
                out.append(f"| {name} | {shows} | {eps} | {r['laughs'][0]:.2f} ± {r['laughs'][1]:.2f} | {diff} | {pct} | "
                           f"{r['non_greedy']:.1f} | {verdict} |")
    return "\n".join(out)


if __name__ == "__main__":
    res = run_or_load("epsilon_cost", run, horizons=[list(h) for h in HORIZONS], epsilons=list(EPSILONS))
    print(table(res))
    print(res["criteria"])
