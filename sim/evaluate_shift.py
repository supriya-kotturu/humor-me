"""Mid-set preference shift: can the director notice the room turning and switch?

Usage:
    python -m sim.evaluate_shift [--shows 400 --seeds 10]

Each career: `--shows` shows in random rooms; in a random half of them the room's taste shifts
after bit 4 of 8 (audience.yaml `shift`: the modes that were working drop, others rise). The
director is never told. It learns throughout; metrics come from the last 100 shows (experienced).

Compares the chosen director (LinUCB, uniform prior) with and without the reaction feature
(same mode × last reward), plus LinTS and baselines. Writes reports/metrics/shift.json
(cached by params). Importable: run(), table().
"""
import argparse
import time

import numpy as np

from sim.cache import run_or_load
from sim.env import AudienceEnv
from sim.evaluate_policies import CONTEXTS, Oracle, ci
from sim.policies import Fixed, LinQ, LinTS, LinUCB, Random, TwoTimescale, prior_means

EVAL = 100
RECOVERED = 0.02  # per-bit regret (reward units, ≈ 0.3 laughs) counted as "back on the best mode"


def policies(modes) -> dict:
    uni = lambda env, s: prior_means(env, "uniform")  # noqa: E731
    return {
        "oracle": lambda env, s: Oracle(env),
        "fixed neurotic": lambda env, s: Fixed(modes.index("neurotic")),
        "random": lambda env, s: Random(len(modes), seed=s),
        "LinUCB": lambda env, s: LinUCB(uni(env, s), seed=s),
        "LinUCB + react": lambda env, s: LinUCB(uni(env, s), seed=s, react=True),
        "LinTS": lambda env, s: LinTS(uni(env, s), seed=s),
        "LinTS + react": lambda env, s: LinTS(uni(env, s), seed=s, react=True),
        "LinUCB + show state": lambda env, s: LinUCB(uni(env, s), seed=s, show_state=True),
        "LinTS + show state": lambda env, s: LinTS(uni(env, s), seed=s, show_state=True),
        "LinQ": lambda env, s: LinQ(uni(env, s), seed=s),
        **{two_ts_name(k): (lambda k: lambda env, s: TwoTimescale(LinUCB(uni(env, s), seed=s), k))(k)
           for k in TWO_TS_KAPPAS},
    }


# Exp 5's policy set, pinned so registering new policies can't change its results on re-run.
SHIFT_POLICIES = ("oracle", "fixed neurotic", "random", "LinUCB", "LinUCB + react", "LinTS", "LinTS + react")
SHOW_STATE_POLICIES = ("oracle", "random", "LinUCB", "LinUCB + show state", "LinTS", "LinTS + show state")
# Step B sweep: kappa = how many bits of evidence tonight needs before it outweighs the career.
TWO_TS_KAPPAS = (0.5, 1, 2, 4, 8)


def two_ts_name(kappa: float) -> str:
    return f"LinUCB + tonight κ={kappa:g}"


TWO_TS_POLICIES = ("oracle", "random", "LinUCB", *map(two_ts_name, TWO_TS_KAPPAS))
C1_POLICIES = ("oracle", "random", "LinUCB", "LinQ")


def career(make, n_shows: int, seed: int, shift_at: int, n_bits: int = 8) -> list[dict]:
    plain, shifting = AudienceEnv(n_bits=n_bits), AudienceEnv(n_bits=n_bits, shift_at=shift_at)
    # One policy object across both envs: the director can't tell which room it's in.
    policy = make(plain, seed)
    rng = np.random.default_rng(seed)
    shows = []
    for s in range(n_shows):
        shifted = bool(rng.random() < 0.5)
        env = shifting if shifted else plain
        if isinstance(policy, Oracle):
            policy.env = env
        obs = env.reset(CONTEXTS[rng.integers(len(CONTEXTS))], seed=seed * 100_000 + s)
        laughs, regret, quality = [], [], []
        done = False
        while not done:
            a = policy.act(obs)
            nxt, r, done, info = env.step(a)
            policy.update(obs, a, r)
            obs = nxt
            laughs.append(sum(info["levels"]))
            regret.append(info["regret"])
            # 1.0 = picked the mode that was best for this bit; noise-free, unlike laughs
            quality.append(float(info["expected"][a] / info["expected"].max()))
        shows.append({"shifted": shifted, "laughs": laughs, "regret": regret, "quality": quality})
    return shows


def bits_to_recover(regret: list[float], shift_at: int) -> int:
    """Bits after the shift until the director first picks a (near-)best mode; censored at the bits left."""
    for k, r in enumerate(regret[shift_at:]):
        if r < RECOVERED:
            return k
    return len(regret) - shift_at


def summarize(careers: list[list[dict]], shift_at: int, eval_last: int = EVAL) -> dict:
    out = {}
    for label, want in (("shifted", True), ("steady", False)):
        pre, post, rec = [], [], []
        for shows in careers:
            sel = [sh for sh in shows[-eval_last:] if sh["shifted"] == want]
            if not sel:  # tiny runs can draw no shifted (or no steady) shows
                continue
            pre.append(np.mean([sum(sh["laughs"][:shift_at]) for sh in sel]))
            post.append(np.mean([sum(sh["laughs"][shift_at:]) for sh in sel]))
            rec.append(np.mean([bits_to_recover(sh["regret"], shift_at) for sh in sel]))
        out[label] = {"pre": ci(np.array(pre)), "post": ci(np.array(post)), "recover": ci(np.array(rec))}
    return out


def run(shows: int = 400, seeds: int = 10, shift_at: int = 4) -> dict:
    return {name: summarize([career(make, shows, s, shift_at) for s in range(seeds)], shift_at)
            for name, make in policies(AudienceEnv().modes).items() if name in SHIFT_POLICIES}


def table(results: dict) -> str:
    o = results["oracle"]
    lines = ["| Policy | Shifted shows: after shift | % of oracle | Bits to recover | Steady shows: same bits | % of oracle |",
             "|---|---|---|---|---|---|"]
    for name, r in results.items():
        sp, st = r["shifted"]["post"], r["steady"]["post"]
        lines.append(f"| {name} | {sp[0]:.1f} ± {sp[1]:.1f} | {100 * sp[0] / o['shifted']['post'][0]:.0f}% | "
                     f"{r['shifted']['recover'][0]:.2f} ± {r['shifted']['recover'][1]:.2f} | "
                     f"{st[0]:.1f} ± {st[1]:.1f} | {100 * st[0] / o['steady']['post'][0]:.0f}% |")
    return "\n".join(lines)


LENGTH_POLICIES = ("oracle", "random", "LinUCB", "LinTS")


def run_lengths(lengths: tuple = (8, 16, 32, 52), total_bits: int = 3200, seeds: int = 10,
                names: tuple = LENGTH_POLICIES) -> dict:
    """Does a longer set let the director recover from a mid-set shift?

    Shift at the midpoint of each set length. Every career has the same total experience
    (total_bits), so longer sets mean fewer shows, not more data. Evaluated on the last quarter
    of each career. Returns, per length: summary per policy, and the mean choice quality at each
    bit position in shifted vs steady shows (the recovery curve).
    """
    makers = {k: v for k, v in policies(AudienceEnv().modes).items() if k in names}
    out = {}
    for n_bits in lengths:
        shift_at, shows = n_bits // 2, max(4, total_bits // n_bits)
        eval_last = max(1, shows // 4)
        res = {"shows": shows, "shift_at": shift_at, "summary": {}, "curve": {}, "steady_curve": {}}
        for name, make in makers.items():
            careers = [career(make, shows, s, shift_at, n_bits) for s in range(seeds)]
            res["summary"][name] = summarize(careers, shift_at, eval_last)
            for key, want in (("curve", True), ("steady_curve", False)):
                q = [sh["quality"] for c in careers for sh in c[-eval_last:] if sh["shifted"] == want]
                res[key][name] = [round(float(v), 4) for v in np.mean(q, axis=0)]
        out[str(n_bits)] = res
    return out


def table_lengths(res: dict) -> str:
    """Per set length: what the shift costs, and whether choices recover late in the set."""
    lines = ["| Set length (bits) | Shows per career | Policy | After shift: % of oracle laughs | "
             "Choice quality: first 2 bits after shift | Last quarter of set | Same bits, steady shows | "
             "Bits to recover (max) |", "|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        n_int, s = int(n), r["shift_at"]
        last_q = slice(n_int - max(1, n_int // 4), n_int)
        for name in r["summary"]:
            if name == "oracle":
                continue
            sm, osm = r["summary"][name]["shifted"], r["summary"]["oracle"]["shifted"]
            c, sc = r["curve"][name], r["steady_curve"][name]
            lines.append(f"| {n} | {r['shows']} | {name} | {100 * sm['post'][0] / osm['post'][0]:.0f}% | "
                         f"{np.mean(c[s:s + 2]):.3f} | {np.mean(c[last_q]):.3f} | {np.mean(sc[last_q]):.3f} | "
                         f"{sm['recover'][0]:.1f} ({n_int - s}) |")
    return "\n".join(lines)


def plot_lengths(res: dict):
    """Recovery curves: choice quality by bit position relative to the shift, one panel per policy."""
    import matplotlib.pyplot as plt

    ramp = ("#6da7ec", "#3987e5", "#256abf", "#104281")  # dataviz blue ramp: longer set = darker
    ink, muted, grid = "#2b2a27", "#8a8880", "#e6e4dd"
    names = [p for p in next(iter(res.values()))["summary"] if p != "oracle"]
    fig, axes = plt.subplots(1, len(names), figsize=(12, 3.8), sharey=True)
    for ax, name in zip(axes, names):
        for color, (n, r) in zip(ramp, res.items()):
            x = np.arange(int(n)) - r["shift_at"]
            ax.plot(x, r["steady_curve"][name], color=color, lw=1, ls=":", alpha=0.8)
            ax.plot(x, r["curve"][name], color=color, lw=2, label=f"{n} bits")
        ax.axvline(0, color=muted, lw=1)
        ax.annotate("shift", (0, 1.0), xytext=(3, -2), textcoords="offset points", color=muted, fontsize=8, va="top")
        ax.set_title(name, color=ink, fontsize=10, loc="left")
        ax.set_xlabel("bits since the shift", color=ink)
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=muted)
    axes[0].set_ylabel("choice quality (1 = best mode)", color=ink)
    # LinUCB's lower-left is empty (pre-shift quality is high), so the legend can't cover data there.
    axes[1].legend(frameon=False, fontsize=8, loc="lower left", title="solid: shifted · dotted: steady",
                   title_fontsize=7)
    fig.suptitle("Does a longer set let the director recover after the room turns?", color=ink, x=0.01, ha="left")
    fig.set_facecolor("#fcfcfb")
    fig.tight_layout()
    return fig


def show_state_deltas(res: dict) -> list[dict]:
    """Per set length and algorithm: + show state vs without, on both pass criteria (step A).

    shifted_gain: mean choice quality after the shift, with minus without (want ≥ 0.05).
    steady_change: mean choice quality over the same bits in steady shows (want ≥ −0.01).
    """
    rows = []
    for n, r in res.items():
        for base in ("LinUCB", "LinTS"):
            new = f"{base} + show state"
            if new in r["curve"]:
                rows.append({"bits": int(n), "algo": base, **variant_delta(r, base, new)})
    return rows


def variant_delta(r: dict, base: str, new: str) -> dict:
    """Both pass criteria for one variant vs its baseline, from one set length's results."""
    s = r["shift_at"]
    q = lambda name, key: float(np.mean(r[key][name][s:]))  # noqa: E731
    return {"shifted_base": q(base, "curve"), "shifted_new": q(new, "curve"),
            "steady_base": q(base, "steady_curve"), "steady_new": q(new, "steady_curve"),
            "shifted_gain": q(new, "curve") - q(base, "curve"),
            "steady_change": q(new, "steady_curve") - q(base, "steady_curve")}


GAIN_MIN, STEADY_MIN = 0.05, -0.01  # pass criteria, fixed before step A ran (decisions.md #29)


def passes(d: dict) -> bool:
    return d["shifted_gain"] >= GAIN_MIN and d["steady_change"] >= STEADY_MIN


def two_ts_deltas(res: dict) -> list[dict]:
    """Step B: each kappa vs plain LinUCB, per set length."""
    return [{"bits": int(n), "kappa": k, **variant_delta(r, "LinUCB", two_ts_name(k))}
            for n, r in res.items() for k in TWO_TS_KAPPAS if two_ts_name(k) in r["curve"]]


def pick_kappa(res: dict) -> float | None:
    """Pre-set rule: among kappas passing both criteria at every set length, the one with the
    largest mean shifted-show gain. None = no kappa passes → step B not adopted."""
    rows = two_ts_deltas(res)
    ok = [k for k in TWO_TS_KAPPAS if all(passes(d) for d in rows if d["kappa"] == k)]
    gain = lambda k: np.mean([d["shifted_gain"] for d in rows if d["kappa"] == k])  # noqa: E731
    return max(ok, key=gain) if ok else None


def table_two_ts(res: dict) -> str:
    lines = ["| Set (bits) | κ (bits to trust tonight) | Shifted: quality after shift (LinUCB → B) | Gain (want ≥ 0.05) | "
             "Steady: same bits (LinUCB → B) | Change (want ≥ −0.01) | Passes both? |", "|---|---|---|---|---|---|---|"]
    for d in two_ts_deltas(res):
        lines.append(f"| {d['bits']} | {d['kappa']:g} | {d['shifted_base']:.3f} → {d['shifted_new']:.3f} | "
                     f"{d['shifted_gain']:+.3f} | {d['steady_base']:.3f} → {d['steady_new']:.3f} | "
                     f"{d['steady_change']:+.3f} | {'yes' if passes(d) else 'no'} |")
    return "\n".join(lines)


def plot_two_ts(res: dict):
    """Recovery curves after a shift for each kappa (darker = trusts tonight more slowly), with
    plain LinUCB dashed and its steady-show level dotted; one panel per set length."""
    import matplotlib.pyplot as plt

    ramp = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")  # dataviz blue ramp, ordinal
    base_c, ink, muted, grid = "#eb6834", "#2b2a27", "#8a8880", "#e6e4dd"
    fig, axes = plt.subplots(1, len(res), figsize=(5.5 * len(res), 3.8), sharey=True, squeeze=False)
    for ax, (n, r) in zip(axes[0], res.items()):
        x = np.arange(int(n)) - r["shift_at"]
        steady = float(np.mean(r["steady_curve"]["LinUCB"]))
        ax.axhline(steady, color=muted, lw=1, ls=":")
        ax.annotate("steady shows (LinUCB)", (x[0], steady), xytext=(0, 3), textcoords="offset points",
                    color=muted, fontsize=7)
        ax.plot(x, r["curve"]["LinUCB"], color=base_c, lw=1.8, ls="--", label="LinUCB (no tonight)")
        for c, k in zip(ramp, TWO_TS_KAPPAS):
            if two_ts_name(k) in r["curve"]:
                ax.plot(x, r["curve"][two_ts_name(k)], color=c, lw=2, label=f"κ = {k:g}")
        ax.axvline(0, color=muted, lw=1)
        ax.set_title(f"{n}-bit sets: shifted shows", color=ink, fontsize=10, loc="left")
        ax.set_xlabel("bits since the shift", color=ink)
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=muted)
    axes[0][0].set_ylabel("choice quality (1 = best mode)", color=ink)
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Step B: career estimate + tonight's correction, by how fast it trusts tonight",
                 color=ink, x=0.01, ha="left")
    fig.set_facecolor("#fcfcfb")
    fig.tight_layout()
    return fig


def table_show_state(res: dict) -> str:
    lines = ["| Set (bits) | Algorithm | Shifted: quality after shift (without → with show state) | Gain (want ≥ 0.05) | "
             "Steady: same bits (without → with) | Change (want ≥ −0.01) |", "|---|---|---|---|---|---|"]
    for d in show_state_deltas(res):
        lines.append(f"| {d['bits']} | {d['algo']} | {d['shifted_base']:.3f} → {d['shifted_new']:.3f} | "
                     f"{d['shifted_gain']:+.3f} | {d['steady_base']:.3f} → {d['steady_new']:.3f} | {d['steady_change']:+.3f} |")
    return "\n".join(lines)


def plot_show_state(res: dict):
    """Recovery curves in shifted shows, with vs without show state; one panel per set length."""
    import matplotlib.pyplot as plt

    colors = {"LinUCB": "#eb6834", "LinTS": "#1baf7a"}  # same hues as the Exp 4 figure
    ink, muted, grid = "#2b2a27", "#8a8880", "#e6e4dd"
    fig, axes = plt.subplots(1, len(res), figsize=(5.5 * len(res), 3.8), sharey=True, squeeze=False)
    for ax, (n, r) in zip(axes[0], res.items()):
        x = np.arange(int(n)) - r["shift_at"]
        ax.axhline(float(np.mean(r["steady_curve"]["LinUCB"])), color=muted, lw=1, ls=":")
        ax.annotate("steady shows (LinUCB)", (x[0], float(np.mean(r["steady_curve"]["LinUCB"]))),
                    xytext=(0, 3), textcoords="offset points", color=muted, fontsize=7)
        for base, c in colors.items():
            ax.plot(x, r["curve"][base], color=c, lw=1.5, ls="--", label=f"{base}")
            ax.plot(x, r["curve"][f"{base} + show state"], color=c, lw=2.2, label=f"{base} + show state")
        ax.axvline(0, color=muted, lw=1)
        ax.set_title(f"{n}-bit sets: shifted shows", color=ink, fontsize=10, loc="left")
        ax.set_xlabel("bits since the shift", color=ink)
        ax.grid(color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=muted)
    axes[0][0].set_ylabel("choice quality (1 = best mode)", color=ink)
    axes[0][0].legend(frameon=False, fontsize=8, loc="lower left")  # pre-shift bits sit high; lower-left is empty
    fig.suptitle("Step A: does remembering tonight help the director recover?", color=ink, x=0.01, ha="left")
    fig.set_facecolor("#fcfcfb")
    fig.tight_layout()
    return fig


HURT_BY_SHIFT = ("cynic", "observer", "absurdist")  # modes audience.yaml's default shift lowers


def run_behaviour(names: tuple = ("LinUCB", two_ts_name(1), two_ts_name(8)), shows: int = 400,
                  seeds: int = 3, shift_at: int = 4) -> dict:
    """Why a director variant wins or loses, not just whether: per policy, how often it repeats
    the previous bit's mode (the room tires of repeats) and, after a shift, how often it still
    picks a mode the shift hurt. Last quarter of each career, 8-bit sets."""
    modes = AudienceEnv().modes
    hurt = {modes.index(m) for m in HURT_BY_SHIFT}
    makers = policies(modes)
    out = {}
    for name in names:
        repeats, hurt_picks = [], []
        for seed in range(seeds):
            plain, shifting = AudienceEnv(), AudienceEnv(shift_at=shift_at)
            policy = makers[name](plain, seed)
            rng = np.random.default_rng(seed)
            for s in range(shows):
                shifted = bool(rng.random() < 0.5)
                env = shifting if shifted else plain
                if isinstance(policy, Oracle):
                    policy.env = env
                obs = env.reset(CONTEXTS[rng.integers(len(CONTEXTS))], seed=seed * 100_000 + s)
                acts, done = [], False
                while not done:
                    a = policy.act(obs)
                    nxt, r, done, _ = env.step(a)
                    policy.update(obs, a, r)
                    obs, acts = nxt, acts + [a]
                if s >= shows - shows // 4:
                    repeats += [acts[i] == acts[i - 1] for i in range(1, len(acts))]
                    if shifted:
                        hurt_picks += [a in hurt for a in acts[shift_at:]]
        out[name] = {"repeat_rate": float(np.mean(repeats)), "hurt_pick_rate": float(np.mean(hurt_picks))}
    return out


def table_behaviour(res: dict) -> str:
    lines = ["| Policy | Repeats the last bit's mode | After a shift, picks a mode the shift hurt |", "|---|---|---|"]
    for name, r in res.items():
        lines.append(f"| {name} | {r['repeat_rate']:.0%} | {r['hurt_pick_rate']:.0%} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shows", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--shift-at", type=int, default=4)
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    results = run_or_load("shift", run, rerun=a.rerun, shows=a.shows, seeds=a.seeds, shift_at=a.shift_at)
    print(f"{a.seeds} careers x {a.shows} shows ({time.time() - t0:.0f}s); last {EVAL} shows.\n")
    print(table(results))


if __name__ == "__main__":
    main()
