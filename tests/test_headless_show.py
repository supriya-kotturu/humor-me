"""The headless show loop: a whole show with a simulated room and a fake voice, no browser or network."""
import asyncio
import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app.director import EPSILON, finished_bits, load_director  # noqa: E402
from app.log import SessionLog, read_session  # noqa: E402
from app.rooms import REACTION, SimRoom  # noqa: E402
from app.show import Batch, UniformSelector, run_show  # noqa: E402
from app.tts import to_wav, wav_seconds  # noqa: E402
from sim.env import AudienceEnv, Context  # noqa: E402
from sim.policies import EpsilonSoft, LinUCB, prior_means  # noqa: E402

ENV = AudienceEnv()
MODES = ENV.modes
CTX = {"venue": "bar", "energy": "high", "age": "mixed"}


class FakeClock:
    def __init__(self):
        self.t, self.sleeps = 1000.0, []

    def now(self):
        return self.t

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


class FakePlayer:
    def __init__(self, clock):
        self.clock = clock

    async def play(self, wav):
        self.clock.t += wav_seconds(wav)


class FakeTTS:
    def synthesize(self, text):
        return to_wav(np.zeros(2400))  # 0.1 s


class FakeGenerator:
    generator_id = "fake"

    def __init__(self, n=3):
        self.n = n

    def generate(self, mode, bit, joke, context):
        return Batch(f"p{bit}-{joke}", f"mode={mode} bit={bit} joke={joke}",
                     [f"{mode} line {bit}.{joke}.{i}" for i in range(self.n)])


class PassAll:
    def check(self, text, age):
        return SimpleNamespace(ok=True, reason=None)


class RejectAll:
    def check(self, text, age):
        return SimpleNamespace(ok=False, reason="nope")


class Raises:
    def check(self, text, age):
        raise RuntimeError("model down")


class Scripted:
    """A RewardSource that answers from a list of levels, cycling."""
    source = "button"

    def __init__(self, levels):
        self.levels, self.n = levels, 0

    def begin_bit(self, action):
        pass

    async def reaction(self, bit, joke, timeout):
        level = self.levels[self.n % len(self.levels)]
        self.n += 1
        key = {3: "W", 2: "K", 1: "Space", 0: "D"}.get(level)
        return level, key, 0.4 if level is not None else timeout


def director(seed=1):
    return EpsilonSoft(LinUCB(prior_means(ENV, "uniform")), EPSILON, len(MODES), seed=seed)


def go(tmp_path, d=None, room="sim", ctx=CTX, source=None, safety=None, clock=None, tts=None, seed=0, **kw):
    clock = clock or FakeClock()
    log = SessionLog(tmp_path)
    asyncio.run(run_show(d or director(), room, ctx, generator=FakeGenerator(), safety=safety or PassAll(),
                         selector=UniformSelector(seed), tts=tts or FakeTTS(), player=FakePlayer(clock),
                         source=source or SimRoom(ctx, seed), clock=clock, log=log, **kw))
    return read_session(log.path), clock


def kinds(rows):
    return [r["event"] for r in rows]


def same(d1, d2):
    return np.array_equal(d1.inner.A, d2.inner.A) and np.array_equal(d1.inner.b, d2.inner.b)


def test_a_full_show_runs_fast_and_every_row_validates(tmp_path):
    start = time.perf_counter()
    rows, _ = go(tmp_path, explore_seed=1)
    assert time.perf_counter() - start < 5
    assert kinds(rows)[0] == "show_start" and kinds(rows)[-1] == "show_end" and rows[-1]["status"] == "finished"
    for event, n in [("bit_decision", 8), ("bit_end", 8), ("candidates", 40), ("joke_played", 40), ("reaction", 40),
                     ("gap", 40)]:
        assert kinds(rows).count(event) == n
    assert {r["source"] for r in rows if r["event"] == "reaction"} == {"sim"}
    assert rows[0]["policy"]["seed"] == 1 and rows[0]["generator_id"] == "fake" and rows[0]["selector_id"] == "uniform"
    assert all(r["probs"] and abs(sum(r["probs"].values()) - 1) < 1e-9 for r in rows if r["event"] == "bit_decision")


def test_logged_prompt_and_pick_are_the_ones_used(tmp_path):
    rows, _ = go(tmp_path)
    cand = {(r["bit"], r["joke"]): r for r in rows if r["event"] == "candidates"}
    for r in (r for r in rows if r["event"] == "joke_played"):
        c = cand[(r["bit"], r["joke"])]
        assert c["prompt"].startswith("mode=") and c["items"][r["chosen"]]["text"] == r["text"]
        assert r["pick_prob"] == pytest.approx(1 / 3)


def test_replay_reproduces_every_greedy_mode_and_the_online_matrices(tmp_path):
    online = director(seed=7)
    shows = []
    for s in range(3):
        rows, _ = go(tmp_path, online, source=SimRoom(CTX, 100 + s), seed=s, explore_seed=7)
        shows.append(rows)
    rewards = {b["reward"] for b in finished_bits(shows, room="sim")}
    assert len(rewards) > 3  # rewards come from the room's reactions, not a constant
    assert online.explored >= 1  # the replay covers picks that differ from the greedy mode
    decisions = {(r["session"], r["bit"]): r for rows in shows for r in rows if r["event"] == "bit_decision"}
    rebuilt = director()
    for b in finished_bits(shows, room="sim"):
        o = b["obs"]
        obs = {"t": o["t"], "n_bits": o["n_bits"], "context": Context(**o["context"]),
               "last_mode": None if o["last_mode"] is None else MODES.index(o["last_mode"]),
               "last_reward": b["last_reward"]}
        assert MODES[rebuilt.inner.act(obs)] == decisions[(b["session"], b["bit"])]["greedy"]
        rebuilt.update(obs, MODES.index(b["mode"]), b["reward"])
    assert same(rebuilt, online) and same(load_director(shows, room="sim"), online)
    assert not same(online, director())


def test_loop_laughs_equal_a_direct_env_run(tmp_path):
    for seed in range(4):
        ctx = {"venue": ("bar", "company_event", "showcase")[seed % 3], "energy": "normal", "age": "teens"}
        env, d = AudienceEnv(), director(seed=seed + 50)
        obs, done, direct = env.reset(Context(**ctx), seed=seed), False, 0
        while not done:
            a = d.act(obs)
            nxt, r, done, info = env.step(a)
            d.update(obs, a, r)
            obs, direct = nxt, direct + sum(info["levels"])
        rows, _ = go(tmp_path, director(seed=seed + 50), ctx=ctx, source=SimRoom(ctx, seed), seed=seed)
        assert sum(r["level"] for r in rows if r["event"] == "reaction") == direct


def test_all_candidates_rejected_skips_the_joke_and_the_show_goes_on(tmp_path):
    d = director()
    rows, _ = go(tmp_path, d, safety=RejectAll())
    assert kinds(rows).count("candidates") == 40 and rows[-1]["status"] == "finished"
    assert not {"joke_played", "reaction", "gap"} & set(kinds(rows))
    assert all(r["reward"] is None and r["n_labels"] == 0 for r in rows if r["event"] == "bit_end")
    assert same(d, director())  # nothing labeled, nothing learned
    assert {i["verdict"] for r in rows if r["event"] == "candidates" for i in r["items"]} == {"reject"}


def test_a_filter_that_raises_rejects_the_candidate(tmp_path):
    rows, _ = go(tmp_path, safety=Raises())
    items = [i for r in rows if r["event"] == "candidates" for i in r["items"]]
    assert all(i["verdict"] == "reject" and "model down" in i["reason"] for i in items)
    assert "joke_played" not in kinds(rows)


def test_a_filter_that_raises_on_one_candidate_leaves_the_others(tmp_path):
    class Picky:
        def check(self, text, age):
            if text.endswith(".0"):
                raise RuntimeError("boom")
            return SimpleNamespace(ok=True, reason=None)
    rows, _ = go(tmp_path, safety=Picky())
    assert all(r["chosen"] != 0 for r in rows if r["event"] == "joke_played")
    assert kinds(rows).count("joke_played") == 40


def test_a_crash_mid_show_leaves_a_readable_log_ending_in_aborted(tmp_path):
    class Crashes:
        def synthesize(self, text):
            Crashes.n = getattr(Crashes, "n", 0) + 1
            if Crashes.n == 12:
                raise RuntimeError("voice died")
            return to_wav(np.zeros(2400))
    log = SessionLog(tmp_path)
    clock = FakeClock()
    with pytest.raises(RuntimeError, match="voice died"):
        asyncio.run(run_show(director(), "sim", CTX, generator=FakeGenerator(), safety=PassAll(),
                             selector=UniformSelector(0), tts=Crashes(), player=FakePlayer(clock),
                             source=SimRoom(CTX, 0), clock=clock, log=log))
    rows = read_session(log.path)
    assert rows[-1]["event"] == "show_end" and rows[-1]["status"] == "aborted"
    assert kinds(rows).count("bit_end") == 2  # bit 2 was cut off


def test_a_bit_with_under_three_labels_makes_no_update(tmp_path):
    class Spy(EpsilonSoft):
        updates = 0

        def update(self, obs, action, reward):
            Spy.updates += 1
            super().update(obs, action, reward)
    thin = Spy(LinUCB(prior_means(ENV, "uniform")), EPSILON, len(MODES), seed=1)
    rows, _ = go(tmp_path, thin, room="live", source=Scripted([3, 3, None, None, None]))
    assert Spy.updates == 0 and same(thin, director())
    assert [r["reward"] for r in rows if r["event"] == "bit_end"] == [None] * 8
    enough = Spy(LinUCB(prior_means(ENV, "uniform")), EPSILON, len(MODES), seed=1)
    go(tmp_path, enough, room="live", source=Scripted([3, 3, 3, None, None]))
    assert Spy.updates == 8 and not same(enough, director())


def test_holds_come_from_the_config_table(tmp_path):
    rows, clock = go(tmp_path, source=Scripted([3, 2, 1, 0, None]))
    assert clock.sleeps[:5] == [REACTION.holds[x] for x in (3, 2, 1, 0, None)] == [2.5, 1.5, 1.0, 0.3, 0.0]
    gaps = [(r["planned_hold_s"], r["actual_gap_s"]) for r in rows if r["event"] == "gap"][:5]
    assert gaps == [(h, pytest.approx(h)) for h in clock.sleeps[:5]]
    assert [r["level"] for r in rows if r["event"] == "reaction"][:5] == [3, 2, 1, 0, None]
