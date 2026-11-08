"""Sanity metrics for the simulated audience, before any learning policy exists.

Usage:
    python -m sim.evaluate_env [--episodes 1000] [--n-bits 8 --jokes-per-bit 5 --fatigue-window 2]
    # per-joke decisions (the v1 setup the per-bit decision replaced):
    python -m sim.evaluate_env --n-bits 40 --jokes-per-bit 1 --fatigue-window 5 --name sim_env_per_joke

Answers: is there anything to learn (oracle vs best fixed mode vs random), does the best
mode differ by venue, and how many bits (and shows) a learner needs to tell the top two
modes apart. Importable: run(), table() (used by notebooks/decisions-director.ipynb).
"""
import argparse
import copy
import math

import numpy as np

from sim.cache import run_or_load
from sim.env import AGES, ENERGIES, VENUES, AudienceEnv, Context, load_audience

Z95 = 1.96


def make_env(n_bits: int, jokes_per_bit: int, fatigue_window: int) -> AudienceEnv:
    audience = copy.deepcopy(load_audience())
    audience["dynamics"]["fatigue_window"] = fatigue_window
    return AudienceEnv(audience, n_bits=n_bits, jokes_per_bit=jokes_per_bit)


def _episodes(env: AudienceEnv, ctx: Context, policy, episodes: int, seed: int) -> float:
    levels = []
    for ep in range(episodes):
        env.reset(ctx, seed=seed + ep)
        lv, done = 0.0, False
        while not done:
            _, _, done, info = env.step(policy(env))
            lv += sum(info["levels"])
        levels.append(lv)
    return float(np.mean(levels))


def bits_to_separate(gap: float, sd: float) -> int:
    # Two-arm z-test at 95% with equal pulls: each arm needs 2(z·sd/gap)^2, so total doubles that.
    return math.ceil(4 * (Z95 * sd / gap) ** 2) if gap > 0 else -1


def static_gap(env: AudienceEnv, ctx: Context) -> tuple[str, str, float]:
    """Best vs runner-up on the context part only (no novelty, no fatigue)."""
    env.reset(ctx, seed=0)
    env.t = 10**6  # novelty fully decayed
    r = env.expected_rewards()
    o = r.argsort()[::-1]
    return env.modes[o[0]], env.modes[o[1]], float(r[o[0]] - r[o[1]])


def env_reward_sd(env: AudienceEnv, ctx: Context, n: int = 20000) -> float:
    """Per-decision reward sd (mean of jokes_per_bit levels / 3) for the best mode, sampled directly."""
    env.reset(ctx, seed=0)
    env.t = 10**6
    mu = env.mu().max()
    lat = mu + np.random.default_rng(0).normal(0, env.noise, (n, env.jokes_per_bit))
    return float(((lat[..., None] > env.thresholds).sum(-1).mean(1) / 3).std())


def run(episodes: int = 1000, n_bits: int = 8, jokes_per_bit: int = 5, fatigue_window: int = 2) -> dict:
    env = make_env(n_bits, jokes_per_bit, fatigue_window)
    rng = np.random.default_rng(0)
    oracle = lambda e: int(e.expected_rewards().argmax())  # noqa: E731
    rand = lambda e: int(rng.integers(e.n_modes))  # noqa: E731
    rows = []
    for venue in VENUES:
        ctx = Context(venue, "normal", "mixed")
        fixed = {m: _episodes(env, ctx, lambda e, i=i: i, episodes, 0) for i, m in enumerate(env.modes)}
        best_fixed = max(fixed, key=fixed.get)
        best, second, gap = static_gap(env, ctx)
        n_sep = bits_to_separate(gap, env_reward_sd(env, ctx))
        rows.append({"venue": venue, "oracle": _episodes(env, ctx, oracle, episodes, 0),
                     "random": _episodes(env, ctx, rand, episodes, 0),
                     "best_fixed_mode": best_fixed, "best_fixed": fixed[best_fixed],
                     "fixed_neurotic": fixed["neurotic"], "best_mode": best, "runner_up": second,
                     "gap": gap, "bits_to_separate": n_sep, "shows_to_separate": round(n_sep / n_bits, 1)})
    best_by_ctx = {f"{v}/{e}/{a}": static_gap(env, Context(v, e, a))[0]
                   for v in VENUES for e in ENERGIES for a in AGES}
    return {"venues": rows, "best_mode_by_context": best_by_ctx,
            "distinct_best_modes": sorted(set(best_by_ctx.values()))}


def table(res: dict) -> str:
    lines = ["| Venue | Oracle | Best fixed (mode) | Fixed neurotic | Random | Best − runner-up gap | Decisions (shows) to separate |",
             "|---|---|---|---|---|---|---|"]
    for r in res["venues"]:
        lines.append(f"| {r['venue']} | {r['oracle']:.1f} | {r['best_fixed']:.1f} ({r['best_fixed_mode']}) | "
                     f"{r['fixed_neurotic']:.1f} | {r['random']:.1f} | {r['best_mode']} − {r['runner_up']} = "
                     f"{r['gap']:.3f} | {r['bits_to_separate']} ({r['shows_to_separate']}) |")
    lines.append(f"\nDistinct best modes over 27 contexts: {', '.join(res['distinct_best_modes'])}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--n-bits", type=int, default=8)
    ap.add_argument("--jokes-per-bit", type=int, default=5)
    ap.add_argument("--fatigue-window", type=int, default=2)
    ap.add_argument("--name", default="sim_env")
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    res = run_or_load(a.name, run, rerun=a.rerun, episodes=a.episodes, n_bits=a.n_bits,
                      jokes_per_bit=a.jokes_per_bit, fatigue_window=a.fatigue_window)
    print(table(res))


if __name__ == "__main__":
    main()
