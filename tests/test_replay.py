"""Replay rebuilds the director from logs: A and b must equal what the show learned online, bit for bit."""
import copy
import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import log as log_module  # noqa: E402
from app.director import EPSILON, finished_bits, joke_levels, load_director  # noqa: E402
from app.log import SessionLog, read_session  # noqa: E402
from app.reward import reward_v1  # noqa: E402
from sim.env import AudienceEnv, Context  # noqa: E402
from sim.policies import EpsilonSoft, LinUCB, prior_means  # noqa: E402

ENV = AudienceEnv()
MODES = ENV.modes
CTX = {"venue": "bar", "energy": "high", "age": "mixed"}
START = dict(context=CTX, policy={"name": "linucb", "epsilon": 0.2, "alpha": 0.15, "prior": "uniform"},
             generator_id="bank", selector_id="uniform", git_sha="abc", operator="op",
             audio={"consent": False, "path": None})
KEYS = {3: "W", 2: "K", 1: "Space", 0: "D"}


@pytest.fixture
def clock(monkeypatch):
    """Deterministic `at`; set clock.now to start a show at a chosen time."""
    class Clock:
        now = 1000.0

        def __call__(self):
            self.now += 1.0
            return self.now

    c = Clock()
    monkeypatch.setattr(log_module.time, "time", c)
    return c


def react(log, bit, joke, level, source="button"):
    key = KEYS[level] if level is not None and source == "button" else None
    log.write("reaction", bit=bit, joke=joke, source=source, level=level, key=key, latency_s=0.5)


def write_bit(log, bit, mode, last_mode, levels, ctx=CTX, n_bits=8, end=True, probs=None):
    log.write("bit_decision", bit=bit, obs={"t": bit, "n_bits": n_bits, "context": ctx, "last_mode": last_mode},
              scores={m: 0.0 for m in MODES}, greedy=mode, action=mode, probs=probs or {m: 1 / 6 for m in MODES})
    for joke, level in enumerate(levels):
        react(log, bit, joke, level)
    if end:
        given = [lv for lv in levels if lv is not None]
        log.write("bit_end", bit=bit, n_labels=len(given), reward_version=1, reward=reward_v1(levels))


def show(tmp_path, bits, room="live", status="finished", at=None, clock=None):
    """bits: list of (mode, levels). Writes a consistent show; returns its rows."""
    if clock is not None and at is not None:
        clock.now = at
    log = SessionLog(tmp_path)
    log.write("show_start", room=room, **START)
    last = None
    for i, (mode, levels) in enumerate(bits):
        write_bit(log, i, mode, last, levels)
        last = mode
    if status:
        log.write("show_end", status=status)
    return read_session(log.path)


def fresh() -> EpsilonSoft:
    return EpsilonSoft(LinUCB(prior_means(ENV, "uniform")), EPSILON, len(MODES))


def state(director):
    return director.inner.A, director.inner.b


def same(d1, d2):
    return all(np.array_equal(a, b) for a, b in zip(state(d1), state(d2)))


def untouched(director):
    return same(director, fresh())


# reward_v1 ---------------------------------------------------------------------------------------------------

def test_reward_is_mean_level_over_three():
    assert reward_v1([3, 3, 3, 3, 3]) == 1.0
    assert reward_v1([0, 0, 0, 0, 0]) == 0.0
    assert reward_v1([3, 2, 1, 0, 2]) == pytest.approx(8 / 5 / 3)


def test_reward_averages_only_labeled_jokes():
    assert reward_v1([3, None, 3, None, 0]) == pytest.approx(2 / 3)


def test_reward_needs_three_labels_of_five():
    assert reward_v1([3, 3, None, None, None]) is None
    assert reward_v1([3, 3, 3, None, None]) == 1.0
    assert reward_v1([None] * 5) is None


# the core guarantee ------------------------------------------------------------------------------------------

def test_replay_reproduces_an_online_run_exactly(tmp_path, clock):
    rng = np.random.default_rng(7)
    online = fresh()
    files, none_bits, updates = [], 0, 0
    for s in range(5):
        room = "sim" if s == 2 else "live"  # a sim show in the middle that the online director never saw
        ctx = {"venue": str(rng.choice(["bar", "company_event", "showcase"])), "energy": str(rng.choice(["low", "normal", "high"])),
               "age": str(rng.choice(["teens", "young_adults", "mixed"]))}
        clock.now = 5000.0 + 100 * s  # float sums depend on order, so replay must follow the shows' start times
        log = SessionLog(tmp_path)
        files.append(log.path)
        log.write("show_start", room=room, **{**START, "context": ctx})
        last_mode, last_reward = None, None
        for bit in range(8):
            obs = {"t": bit, "n_bits": 8, "context": Context(**ctx),
                   "last_mode": last_mode, "last_reward": last_reward}
            action, greedy, probs = online.decide(obs)
            levels = [None if rng.random() < 0.25 else int(rng.integers(4)) for _ in range(5)]
            log.write("bit_decision", bit=bit,
                      obs={"t": bit, "n_bits": 8, "context": ctx, "last_mode": None if last_mode is None else MODES[last_mode]},
                      scores={m: 0.0 for m in MODES}, greedy=MODES[greedy], action=MODES[action],
                      probs={m: float(p) for m, p in zip(MODES, probs)})
            for joke, level in enumerate(levels):
                react(log, bit, joke, level)
            reward = reward_v1(levels)
            log.write("bit_end", bit=bit, n_labels=sum(lv is not None for lv in levels), reward_version=1, reward=reward)
            if room == "live":
                if reward is None:
                    none_bits += 1
                else:
                    online.update(obs, action, reward)
                    updates += 1
            last_mode, last_reward = action, reward
        log.write("show_end", status="finished")
    assert none_bits >= 1 and updates >= 20  # the run exercised skipped bits and real updates
    assert not untouched(online)
    sessions = [read_session(p) for p in reversed(files)]  # handed over in the wrong order on purpose
    replayed = load_director(sessions)
    assert np.array_equal(replayed.inner.A, online.inner.A)
    assert np.array_equal(replayed.inner.b, online.inner.b)


# what counts -------------------------------------------------------------------------------------------------

def test_sim_shows_never_reach_the_director(tmp_path):
    live = show(tmp_path, [("cynic", [3, 3, 3, 3, 3])])
    sim = show(tmp_path, [("absurdist", [2, 2, 2, 2, 2])], room="sim")
    assert same(load_director([live, sim]), load_director([live]))
    assert not same(load_director([live, sim]), load_director([live], room="sim"))  # the filter is what separates them
    assert untouched(load_director([sim]))


def test_a_bit_with_under_three_labels_makes_no_update(tmp_path):
    thin = show(tmp_path, [("cynic", [3, 3, None, None, None])])
    assert untouched(load_director([thin]))
    enough = show(tmp_path, [("cynic", [3, 3, 3, None, None])])
    assert not untouched(load_director([enough]))


def test_a_skipped_bit_is_still_the_next_bits_last_mode(tmp_path):
    rows = show(tmp_path, [("cynic", [1, None, None, None, None]), ("absurdist", [3, 3, 3, 3, 3])])
    expected = fresh()
    obs = {"t": 1, "n_bits": 8, "context": Context(**CTX), "last_mode": MODES.index("cynic"), "last_reward": None}
    expected.update(obs, MODES.index("absurdist"), 1.0)
    assert same(load_director([rows]), expected)


def test_an_aborted_shows_unfinished_bit_is_ignored(tmp_path, clock):
    log = SessionLog(tmp_path)
    log.write("show_start", room="live", **START)
    write_bit(log, 0, "cynic", None, [2, 2, 2, 2, 2])
    write_bit(log, 1, "absurdist", "cynic", [3, 3, 3, 3, 3], end=False)
    log.write("show_end", status="aborted")
    aborted = read_session(log.path)
    one_bit = show(tmp_path, [("cynic", [2, 2, 2, 2, 2])])
    assert not untouched(load_director([one_bit]))
    assert same(load_director([aborted]), load_director([one_bit]))
    assert [b["bit"] for b in finished_bits([aborted])] == [0]


def test_bits_finished_before_an_abort_still_count(tmp_path):
    aborted = show(tmp_path, [("cynic", [2, 2, 2, 2, 2])], status="aborted")
    assert not untouched(load_director([aborted]))


def test_shows_replay_in_start_order_not_file_order(tmp_path, clock):
    late = show(tmp_path, [("cynic", [3, 3, 3, 3, 3])], at=9000.0, clock=clock)  # written first
    early = show(tmp_path, [("absurdist", [1, 1, 1, 1, 1])], at=100.0, clock=clock)
    order = [b["mode"] for b in finished_bits([late, early])]
    assert order == ["absurdist", "cynic"]


# labels ------------------------------------------------------------------------------------------------------

def rows_for(*specs):
    return [{"event": "reaction", "bit": 0, "joke": j, "source": s, "level": lv, "key": None, "latency_s": 0.1}
            for j, s, lv in specs]


def test_last_button_press_wins():
    assert joke_levels(rows_for((0, "button", 0), (0, "button", 3)), 0) == [3]
    assert joke_levels(rows_for((0, "button", 3), (0, "button", 0)), 0) == [0]


def test_a_later_timeout_row_overrides_an_earlier_press():
    assert joke_levels(rows_for((0, "button", 2), (0, "button", None)), 0) == [None]


def test_button_beats_a_later_detector_row_and_a_detector_row_fills_a_gap():
    assert joke_levels(rows_for((0, "button", 1), (0, "detector", 3)), 0) == [1]
    assert joke_levels(rows_for((0, "detector", 3), (0, "button", 1)), 0) == [1]
    assert joke_levels(rows_for((0, "detector", 1), (0, "sim", 2)), 0) == [2]
    assert joke_levels(rows_for((0, "button", None), (0, "detector", 3)), 0) == [None]  # null timeout is a button row


def test_last_press_wins_all_the_way_to_the_director(tmp_path):
    def run(levels_per_joke):
        log = SessionLog(tmp_path)
        log.write("show_start", room="live", **START)
        log.write("bit_decision", bit=0, obs={"t": 0, "n_bits": 8, "context": CTX, "last_mode": None},
                  scores={m: 0.0 for m in MODES}, greedy="cynic", action="cynic", probs={m: 1 / 6 for m in MODES})
        for joke, presses in enumerate(levels_per_joke):
            for level in presses:
                react(log, 0, joke, level)
        log.write("bit_end", bit=0, n_labels=5, reward_version=1, reward=None)
        return load_director([read_session(log.path)])
    corrected = run([[0, 3]] * 5)
    assert same(corrected, run([[3]] * 5))
    assert not same(corrected, run([[3, 0]] * 5))


# derived, not read -------------------------------------------------------------------------------------------

def test_replay_ignores_the_logged_bit_end_reward(tmp_path):
    rows = show(tmp_path, [("cynic", [3, 3, 3, 3, 3]), ("observer", [0, 1, 0, 1, 0])])
    tampered = copy.deepcopy(rows)
    for r in tampered:
        if r["event"] == "bit_end":
            r["reward"] = 0.123
    assert same(load_director([rows]), load_director([tampered]))
    assert [b["reward"] for b in finished_bits([tampered])] == [1.0, pytest.approx(2 / 5 / 3)]


def test_last_reward_is_recomputed_from_the_previous_bits_labels(tmp_path):
    rows = show(tmp_path, [("cynic", [3, 3, 3, 3, 3]), ("observer", [0, 0, 0, 0, 0]), ("wordsmith", [3, None, None, None, None]),
                           ("cynic", [2, 2, 2, 2, 2])])
    planted = copy.deepcopy(rows)
    for r in planted:
        if r["event"] == "bit_decision":
            r["obs"]["last_reward"] = 0.777  # a stale value in the log must not be believed
        if r["event"] == "bit_end":
            r["reward"] = 0.777
    for source in (rows, planted):
        bits = finished_bits([source])
        assert [b["last_reward"] for b in bits] == [None, 1.0, 0.0, None]  # None after an unrewarded bit too
        assert [b["obs"]["last_mode"] for b in bits] == [None, "cynic", "observer", "wordsmith"]  # mode stays set
    assert same(load_director([rows]), load_director([planted]))


def test_last_reward_follows_the_reward_function(tmp_path):
    rows = show(tmp_path, [("cynic", [3, 3, 3, 3, 3]), ("observer", [2, 2, 2, 2, 2])])
    halved = finished_bits([rows], reward_fn=lambda lv: reward_v1(lv) / 2)
    assert halved[1]["last_reward"] == 0.5


def test_a_different_reward_function_changes_b_and_leaves_the_logs_alone(tmp_path):
    rows = show(tmp_path, [("cynic", [3, 3, 3, 3, 3]), ("observer", [2, 2, 2, 2, 2])])
    path = next(tmp_path.glob("*.jsonl"))
    before = path.read_bytes()
    base = load_director([rows])
    other = load_director([rows], reward_fn=lambda lv: 0.0 if reward_v1(lv) is None else 0.25)
    assert not np.array_equal(base.inner.b, other.inner.b)
    assert path.read_bytes() == before
    assert untouched(load_director([rows], reward_fn=lambda lv: None))


def test_a_ridge_models_a_and_b_change_together(tmp_path):
    rows = show(tmp_path, [("cynic", [3, 3, 3, 3, 3]), ("absurdist", [2, 2, 2, 2, 2])])
    director, start = load_director([rows]), fresh()
    changed_a = {a for a in range(len(MODES)) if not np.array_equal(director.inner.A[a], start.inner.A[a])}
    changed_b = {a for a in range(len(MODES)) if not np.array_equal(director.inner.b[a], start.inner.b[a])}
    assert changed_a == changed_b == {MODES.index("cynic"), MODES.index("absurdist")}


# setup -------------------------------------------------------------------------------------------------------

def test_director_is_epsilon_soft_linucb_with_the_uniform_prior():
    d = load_director([])
    assert isinstance(d, EpsilonSoft) and isinstance(d.inner, LinUCB) and d.eps == 0.2 and d.n == len(MODES)
    assert untouched(d)


def test_an_unknown_mode_name_is_rejected(tmp_path):
    rows = show(tmp_path, [("juggler", [3, 3, 3, 3, 3])])
    with pytest.raises(ValueError, match="juggler"):
        load_director([rows])


# exploration seed --------------------------------------------------------------------------------------------

def draws(director, n=50):
    return [director.rng.random() for _ in range(n)]


def test_each_load_explores_differently_by_default():
    assert draws(load_director([])) != draws(load_director([]))


def test_an_explicit_seed_reproduces_exploration():
    assert draws(load_director([], seed=3)) == draws(load_director([], seed=3))
    assert draws(load_director([], seed=3)) != draws(load_director([], seed=4))


# importing ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("module", ["app.reward", "app.director"])
def test_each_app_module_imports_alone_from_backend(module):
    backend = pathlib.Path(__file__).resolve().parents[1] / "backend"
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    result = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=backend, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
