"""Mode-choice policies for the per-bit director.

All share one interface: `act(obs) -> mode index`, `update(obs, action, reward)`, where `obs`
is the observation the action was chosen on. Policies persist across shows, so learning about
a venue carries from one show to the next (a comedian's experience).

Linear policies use one ridge model per mode (LinUCB "disjoint", Li et al. 2010) on context
features. The prior enters as the starting value of each mode's bias weight, with weight
`prior_strength`; that's how HumorGen / uniform / wrong priors are compared.
"""
import math

import numpy as np

from sim.env import AGES, ENERGIES, MAX_LEVEL, VENUES, AudienceEnv, Context, _phi

BIT_REWARD_SD = 0.15  # per-bit reward noise in [0, 1] units; measured 0.14 (evaluate_env.env_reward_sd)


def features(obs: dict, arm: int, react: bool = False, show_state: bool = False) -> np.ndarray:
    """bias, venue/energy/age one-hots, 'same mode as last bit' (fatigue), last bit's reward.

    react=True adds 'same mode × last reward': how *this* mode just landed. Without it the model
    can't express "this mode just bombed, don't repeat it" (mid-set shift experiment).

    show_state=True (step A) adds tonight's record for this mode: tried tonight (0/1) and its mean
    reward tonight (0 if untried). Career averages can't tell "this room, tonight" apart from
    hundreds of past shows; these two inputs can.
    """
    c: Context = obs["context"]
    x = [1.0]
    x += [float(c.venue == v) for v in VENUES]
    x += [float(c.energy == e) for e in ENERGIES]
    x += [float(c.age == a) for a in AGES]
    x.append(float(obs["last_mode"] == arm))
    x.append(obs["last_reward"] or 0.0)
    if react:
        x.append(x[-2] * x[-1])
    if show_state:
        mine = [r for m, r in obs.get("tonight", ()) if m == arm]
        x.append(float(bool(mine)))
        x.append(float(np.mean(mine)) if mine else 0.0)
    return np.array(x)


N_FEATURES = 1 + len(VENUES) + len(ENERGIES) + len(AGES) + 2


def n_features(react: bool = False, show_state: bool = False) -> int:
    return N_FEATURES + react + 2 * show_state


def prior_means(env: AudienceEnv, kind: str, seed: int = 0) -> np.ndarray:
    """Expected bit reward per mode from `base` alone (what a source like HumorGen tells you,
    with no knowledge of the venue).

    kind: 'humorgen' = base as in audience.yaml; 'uniform' = all modes equal (same overall mean);
    'wrong' = humorgen values shuffled across modes (a confidently wrong prior).
    """
    base = np.array([env.p["base"][m] for m in env.modes])
    means = np.array([sum(_phi((b - th) / env.noise) for th in env.thresholds) for b in base]) / MAX_LEVEL
    if kind == "humorgen":
        return means
    if kind == "uniform":
        return np.full_like(means, means.mean())
    if kind == "wrong":
        rng = np.random.default_rng(seed)
        while True:
            perm = rng.permutation(len(means))
            if perm[0] != 0 and (perm != np.arange(len(means))).all():
                return means[perm]
    raise ValueError(kind)


class Fixed:
    def __init__(self, arm: int):
        self.arm = arm

    def act(self, obs):
        return self.arm

    def update(self, obs, action, reward):
        pass


class Random:
    def __init__(self, n_arms: int, seed: int = 0):
        self.n, self.rng = n_arms, np.random.default_rng(seed)

    def act(self, obs):
        return int(self.rng.integers(self.n))

    def update(self, obs, action, reward):
        pass


class EpsilonGreedy:
    """Tabular mean per (venue, energy, age, mode); no pooling across contexts, no history."""

    def __init__(self, prior: np.ndarray, eps: float = 0.1, seed: int = 0):
        self.prior, self.eps, self.rng = prior, eps, np.random.default_rng(seed)
        self.sums: dict = {}
        self.counts: dict = {}

    def _key(self, obs):
        return obs["context"]

    def act(self, obs):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(len(self.prior)))
        k = self._key(obs)
        # The prior counts as one pseudo-observation per mode.
        s = self.sums.get(k, self.prior.copy())
        n = self.counts.get(k, np.ones(len(self.prior)))
        means = s / n
        return int(self.rng.choice(np.flatnonzero(means == means.max())))

    def update(self, obs, action, reward):
        k = self._key(obs)
        self.sums.setdefault(k, self.prior.copy())[action] += reward
        self.counts.setdefault(k, np.ones(len(self.prior)))[action] += 1


class _Linear:
    def __init__(self, prior: np.ndarray, prior_strength: float = 1.0, seed: int = 0,
                 react: bool = False, show_state: bool = False, extra_features: int = 0):
        self.react, self.show_state = react, show_state
        d, k = n_features(react, show_state) + extra_features, len(prior)
        self.A = np.stack([prior_strength * np.eye(d) for _ in range(k)])
        theta0 = np.zeros((k, d))
        theta0[:, 0] = prior
        self.b = np.einsum("kij,kj->ki", self.A, theta0)
        self.rng = np.random.default_rng(seed)

    def _x(self, obs, arm):
        return features(obs, arm, self.react, self.show_state)

    def update(self, obs, action, reward):
        x = self._x(obs, action)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x


class LinUCB(_Linear):
    def __init__(self, prior, alpha: float = BIT_REWARD_SD, **kw):
        super().__init__(prior, **kw)
        self.alpha = alpha

    def scores(self, obs) -> np.ndarray:
        """Per mode: (predicted reward, exploration bonus). Split so a wrapper can correct the
        prediction without touching the bonus."""
        out = []
        for a in range(len(self.A)):
            x = self._x(obs, a)
            A_inv = np.linalg.inv(self.A[a])
            out.append((A_inv @ self.b[a] @ x, self.alpha * math.sqrt(x @ A_inv @ x)))
        return np.array(out)

    def act(self, obs):
        s = self.scores(obs)
        return int(np.argmax(s[:, 0] + s[:, 1]))


class TwoTimescale:
    """Show state step B: career model + a per-show correction that resets every show.

    After each bit, the surprise (reward − the career model's prediction for that mode) is kept
    for tonight. A mode's correction = its total surprise tonight / (bits it had tonight + kappa).
    kappa is "how slowly to believe tonight", in bits: small = trust tonight after one bit,
    large = need many bits before tonight outweighs the career. The career model keeps learning
    from every bit as before.
    """

    def __init__(self, base: LinUCB, kappa: float):
        self.base, self.kappa = base, kappa
        self._surprise: dict[int, list[float]] = {}

    def correction(self, n_modes: int) -> np.ndarray:
        return np.array([sum(self._surprise.get(a, ())) / (len(self._surprise.get(a, ())) + self.kappa)
                         for a in range(n_modes)])

    def act(self, obs):
        if obs["t"] == 0:  # new show: tonight starts with no evidence
            self._surprise = {}
        s = self.base.scores(obs)
        return int(np.argmax(s[:, 0] + s[:, 1] + self.correction(len(s))))

    def update(self, obs, action, reward):
        predicted = self.base.scores(obs)[action, 0]
        self._surprise.setdefault(action, []).append(reward - predicted)
        self.base.update(obs, action, reward)


PLANNING_FEATURES = 5


def planning_features(obs: dict, arm: int) -> list[float]:
    """Step C1's extra inputs: what a choice does to later bits depends on these."""
    tonight = obs.get("tonight", ())
    modes = [m for m, _ in tonight]
    earlier = [r for _, r in tonight[:-1]]
    drop = max(0.0, float(np.mean(earlier)) - tonight[-1][1]) if earlier else 0.0
    untried = float(arm not in modes)
    left = 1 - obs["t"] / obs["n_bits"]
    return [float(len(modes) >= 2 and modes[-2] == arm), left, drop, drop * untried, untried * left]


class LinQ(LinUCB):
    """Show state step C1: linear Q-learning over the bits of a show, undiscounted.

    Same ridge model and bonus as LinUCB, but the target is reward + the next bit's best value, so
    a mode is valued for what it does to the rest of the show. update() only sees the bit just
    played, so each bit's target is completed when the next bit's observation arrives.
    """

    def __init__(self, prior, **kw):
        super().__init__(prior, react=True, show_state=True, extra_features=PLANNING_FEATURES, **kw)
        self._pending = None

    def _x(self, obs, arm):
        return np.concatenate([features(obs, arm, react=True, show_state=True), planning_features(obs, arm)])

    def _learn(self, obs, action, target):
        x = self._x(obs, action)
        self.A[action] += np.outer(x, x)
        self.b[action] += target * x

    def update(self, obs, action, reward):
        if self._pending is not None:
            prev_obs, prev_action, prev_reward = self._pending
            self._learn(prev_obs, prev_action, prev_reward + self.scores(obs)[:, 0].max())
        if obs["t"] == obs["n_bits"] - 1:  # last bit: nothing follows it
            self._learn(obs, action, reward)
            self._pending = None
        else:
            self._pending = (obs, action, reward)


class LinTS(_Linear):
    """Thompson sampling on the same Bayesian ridge model (Agrawal & Goyal 2013)."""

    def __init__(self, prior, v: float = BIT_REWARD_SD, **kw):
        super().__init__(prior, **kw)
        self.v = v

    def act(self, obs):
        scores = []
        for a in range(len(self.A)):
            A_inv = np.linalg.inv(self.A[a])
            theta = self.rng.multivariate_normal(A_inv @ self.b[a], self.v**2 * A_inv)
            scores.append(theta @ self._x(obs, a))
        return int(np.argmax(scores))
