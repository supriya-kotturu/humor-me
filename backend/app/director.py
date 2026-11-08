"""Rebuild the director from session logs: replay every finished bit of live shows, in order, through the same
`update` calls the show made. Reward is recomputed from the reaction rows, never read from `bit_end.reward`.
A reaction row counts toward the reward whenever it is in the log, so the live loop must not write reaction rows
for a window that has closed."""
from app.reward import reward_v1
from sim.env import AudienceEnv, Context
from sim.policies import EpsilonSoft, LinUCB, prior_means

EPSILON = 0.2


def joke_levels(rows: list[dict], bit: int) -> list[int | None]:
    """One level per joke of the bit. Among button rows the last in log order wins, a null timeout included
    (a later press overrides an earlier one, a later timeout does not resurrect it). With no button row, the
    last row of any other source."""
    by_joke: dict[int, dict[str, dict]] = {}
    for r in rows:
        if r["event"] == "reaction" and r["bit"] == bit:
            by_joke.setdefault(r["joke"], {})["button" if r["source"] == "button" else "other"] = r
    out = []
    for joke in sorted(by_joke):
        picked = by_joke[joke].get("button") or by_joke[joke]["other"]
        out.append(picked["level"])
    return out


def finished_bits(sessions: list[list[dict]], room: str = "live", reward_fn=reward_v1) -> list[dict]:
    """Every bit with a `bit_end` row, from shows of `room`, shows ordered by `show_start.at`.
    Each item: obs (raw, context as a dict), mode (name), last_reward (recomputed), reward (None = no update)."""
    shows = []
    for rows in sessions:
        start = next((r for r in rows if r["event"] == "show_start"), None)
        if start is None:
            raise ValueError("session has no show_start row")
        if start["room"] == room:
            shows.append((start["at"], rows))
    out = []
    for _, rows in sorted(shows, key=lambda s: s[0]):
        decisions = {r["bit"]: r for r in rows if r["event"] == "bit_decision"}
        for end in (r for r in rows if r["event"] == "bit_end"):
            bit = end["bit"]
            d = decisions[bit]
            previous = reward_fn(joke_levels(rows, bit - 1)) if bit - 1 in decisions else None
            out.append({"session": end["session"], "bit": bit, "obs": d["obs"], "mode": d["action"],
                        "last_reward": previous, "reward": reward_fn(joke_levels(rows, bit))})
    return out


def load_director(sessions: list[list[dict]], room: str = "live", reward_fn=reward_v1,
                  seed: int | None = None) -> EpsilonSoft:
    """A fresh uniform-prior LinUCB in an ε = 0.2 wrapper, updated once per finished bit that has a reward.
    `seed` is for the wrapper's exploration draws only (replay uses logged actions); None draws fresh entropy, so
    each show explores differently and the logged pick probabilities stay true across shows."""
    env = AudienceEnv()  # mode order and the prior come from the room config, as in the sim
    modes = {name: i for i, name in enumerate(env.modes)}
    director = EpsilonSoft(LinUCB(prior_means(env, "uniform")), EPSILON, env.n_modes, seed)

    def index(name):
        if name is not None and name not in modes:
            raise ValueError(f"unknown mode {name!r}; sim/audience.yaml has {tuple(modes)}")
        return None if name is None else modes[name]

    for bit in finished_bits(sessions, room, reward_fn):
        if bit["reward"] is None:
            continue
        o = bit["obs"]
        obs = {"t": o["t"], "n_bits": o["n_bits"], "context": Context(**o["context"]),
               "last_mode": index(o["last_mode"]), "last_reward": bit["last_reward"]}
        director.update(obs, index(bit["mode"]), bit["reward"])
    return director
