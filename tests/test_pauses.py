"""D-92: a joke whose playback overlapped a pause loses its label from the reward; live and replay agree."""
import asyncio
import importlib
import json
import math
import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app.director import finished_bits, joke_levels, load_director  # noqa: E402
from app.live import ButtonSource  # noqa: E402
from app.log import SessionLog, read_session  # noqa: E402
from app.pauses import Pauses, intervals, overlaps, paused_jokes  # noqa: E402
from app.protocol import ProtocolError, check_pause, parse  # noqa: E402
from app.reward import reward_v1  # noqa: E402
from app.rooms import REACTION  # noqa: E402
from app.show import UniformSelector, run_show  # noqa: E402
from app.tts import wav_seconds  # noqa: E402
from tests.test_headless_show import (CTX, FakeClock, FakeGenerator, FakePlayer, FakeTTS, PassAll,  # noqa: E402
                                      Scripted, director, same)
from tests.test_replay import MODES, START, react  # noqa: E402


def rows_of(*pairs):
    """pause/resume rows from (kind, time) pairs."""
    return [{"event": k, "paused_at" if k == "pause" else "resumed_at": t} for k, t in pairs]


# --- the shared rule -----------------------------------------------------------------------------------------------

def test_overlap_edges():
    p = [(10.0, 20.0)]
    assert overlaps(5, 15, p) and overlaps(15, 25, p) and overlaps(12, 14, p) and overlaps(5, 25, p)
    assert not overlaps(0, 10, p)  # the line ended the instant the pause began
    assert not overlaps(20, 30, p)  # the line began the instant the pause ended
    assert not overlaps(0, 5, p) and not overlaps(25, 30, p)  # pause before start, after end
    assert overlaps(9.99, 10.01, p)


def test_an_unmatched_pause_runs_to_the_end_of_the_show():
    ivs = intervals(rows_of(("pause", 10.0)))
    assert ivs == [(10.0, math.inf)] and overlaps(500, 600, ivs) and not overlaps(0, 10, ivs)


def test_intervals_pair_pauses_in_order_and_ignore_a_repeat_or_stray_resume():
    rows = rows_of(("resume", 1.0), ("pause", 5.0), ("pause", 6.0), ("resume", 8.0), ("pause", 9.0), ("resume", 10.0))
    assert intervals(rows) == [(5.0, 8.0), (9.0, 10.0)]


def test_two_pauses_in_one_joke_drop_it_once_and_a_pause_between_lines_drops_nothing():
    played = [{"event": "joke_played", "bit": 0, "joke": j, "audio_start_at": 10.0 * j, "audio_end_at": 10.0 * j + 4}
              for j in range(4)]
    rows = rows_of(("pause", 11.0), ("resume", 12.0), ("pause", 13.0), ("resume", 14.0),  # both inside joke 1 [10, 14]
                   ("pause", 15.0), ("resume", 19.0),  # reaction window of joke 1 and the gap before joke 2
                   ("pause", 24.0), ("resume", 25.0)) + played  # exactly at the end of joke 2 [20, 24]
    assert paused_jokes(rows, 0) == {1}
    assert paused_jokes(rows, 1) == set()  # another bit


def test_the_live_tracker_matches_the_rows():
    t = Pauses()
    assert not t.paused and t.intervals() == []
    t.pause(5.0)
    assert t.paused and t.intervals() == [(5.0, math.inf)]
    t.resume(8.0)
    assert not t.paused and t.intervals() == [(5.0, 8.0)]


# --- protocol -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["pause", "resume"])
def test_the_browser_may_send_pause_and_resume(kind):
    assert parse({"text": json.dumps({"type": kind})}) == {"type": kind}


@pytest.mark.parametrize("kind, running, paused, ok", [
    ("pause", True, False, True), ("resume", True, True, True),
    ("resume", True, False, False), ("pause", True, True, False),
    ("pause", False, False, False), ("resume", False, True, False)])
def test_out_of_order_pauses_are_refused(kind, running, paused, ok):
    if ok:
        check_pause(kind, running=running, paused=paused)
    else:
        with pytest.raises(ProtocolError):
            check_pause(kind, running=running, paused=paused)


# --- replay -------------------------------------------------------------------------------------------------------

def played(log, bit, joke, start, end):
    log.write("joke_played", bit=bit, joke=joke, chosen=0, pick_prob=1.0, text="t", tts_seconds=0.1, audio_seconds=1.0,
              audio_start_at=start, audio_end_at=end)


def write_show(tmp_path, bits, pauses=(), mode="cynic"):
    """bits: levels per bit; joke j of bit b plays [100 b + 10 j, +2]. `pauses`: (start, end) on that clock."""
    log = SessionLog(tmp_path)
    log.write("show_start", room="live", **START)
    for paused_at, resumed_at in pauses:
        log.write("pause", paused_at=paused_at)
        if resumed_at is not None:
            log.write("resume", resumed_at=resumed_at)
    last = None
    for b, levels in enumerate(bits):
        log.write("bit_decision", bit=b, obs={"t": b, "n_bits": 8, "context": CTX, "last_mode": last},
                  scores={m: 0.0 for m in MODES}, greedy=mode, action=mode, probs={m: 1 / 6 for m in MODES})
        for j, level in enumerate(levels):
            played(log, b, j, 100 * b + 10 * j, 100 * b + 10 * j + 2)
            react(log, b, j, level)
        log.write("bit_end", bit=b, n_labels=sum(lv is not None for lv in levels), reward_version=1,
                  reward=reward_v1(levels))
        last = mode
    log.write("show_end", status="finished")
    return read_session(log.path)


def test_a_bit_with_one_paused_joke_still_updates_with_the_mean_of_the_other_four(tmp_path):
    rows = write_show(tmp_path, [[3, 3, 0, 3, 3]], pauses=[(21.0, 30.0)])  # joke 2 [20, 22] is the 0
    assert joke_levels(rows, 0) == [3, 3, None, 3, 3]
    (bit,) = finished_bits([rows])
    assert bit["reward"] == pytest.approx(1.0)  # (3+3+3+3)/4/3; with the 0 kept it would be 0.8
    plain = write_show(tmp_path, [[3, 3, 3, 3, 3]])
    assert same(load_director([rows], seed=0), load_director([plain], seed=0))


def test_a_bit_with_three_paused_jokes_does_not_update_the_director(tmp_path):
    rows = write_show(tmp_path, [[3, 3, 3, 3, 3]], pauses=[(1.0, 25.0)])  # jokes 0, 1, 2 overlap; 3 and 4 do not
    assert joke_levels(rows, 0) == [None, None, None, 3, 3]
    assert finished_bits([rows])[0]["reward"] is None
    assert same(load_director([rows], seed=0), load_director([], seed=0))


def test_an_unmatched_pause_drops_every_later_joke_of_the_show(tmp_path):
    rows = write_show(tmp_path, [[2] * 5, [2] * 5], pauses=[(101.0, None)])  # bit 1 joke 0 plays [100, 102]
    assert joke_levels(rows, 1) == [None] * 5
    assert finished_bits([rows])[1]["reward"] is None


def test_last_reward_in_the_next_bits_observation_follows_the_dropped_label(tmp_path):
    rows = write_show(tmp_path, [[3, 3, 0, 3, 3], [2] * 5], pauses=[(21.0, 30.0)])
    assert finished_bits([rows])[1]["last_reward"] == pytest.approx(1.0)


# --- the real loop ----------------------------------------------------------------------------------------------

class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_json(self, msg):
        self.sent.append(msg)


def handler(log, clock, pauses):
    """The server's own pause handler (main._pause) on a fake socket, so ordering and checks are the real ones."""
    main = pytest.importorskip("fastapi") and importlib.import_module("app.main")
    live = types.SimpleNamespace(running=True, log=log, clock=clock, pauses=pauses)
    ws = FakeSocket()

    async def call(kind):
        await main._pause(ws, live, kind)
    return call, ws


class PausingPlayer(FakePlayer):
    """Pauses the show during chosen plays (0-based, in play order): stalls the clock for `stall` seconds mid-line."""

    def __init__(self, clock, call, during, stall=30.0):
        super().__init__(clock)
        self.call, self.during, self.stall, self.n = call, set(during), stall, 0

    async def play(self, wav, text):
        n, self.n = self.n, self.n + 1
        if n not in self.during:
            return await super().play(wav, text)
        self.clock.t += wav_seconds(wav) / 2
        await self.call("pause")
        self.clock.t += self.stall
        await self.call("resume")
        self.clock.t += wav_seconds(wav) / 2


def run(tmp_path, player_for, source, n_bits=3, pauses=None, **kw):
    clock, log, pauses = FakeClock(), SessionLog(tmp_path), pauses or Pauses()
    call, ws = handler(log, clock, pauses)
    online = director(seed=3)
    asyncio.run(run_show(online, "live", CTX, generator=FakeGenerator(), safety=PassAll(), selector=UniformSelector(0),
                         tts=FakeTTS(), player=player_for(clock, call), source=source(clock, call, pauses),
                         clock=clock, log=log, pauses=pauses, n_bits=n_bits, explore_seed=3, config=REACTION, **kw))
    return read_session(log.path), online


def test_the_live_reward_equals_the_replayed_reward_for_the_same_log(tmp_path):
    # play n is bit n // 5, joke n % 5: bit 0 loses joke 2 (4 labels left), bit 1 loses jokes 0-2 (no reward),
    # bit 2 is untouched
    rows, online = run(tmp_path, lambda clock, call: PausingPlayer(clock, call, {2, 5, 6, 7}),
                       lambda clock, call, pauses: Scripted([3, 2, 1, 0, 3]))
    ends = [r for r in rows if r["event"] == "bit_end"]
    replayed = finished_bits([rows])
    assert [e["n_labels"] for e in ends] == [4, 2, 5]
    assert [e["reward"] for e in ends] == [b["reward"] for b in replayed]
    assert ends[1]["reward"] is None and ends[0]["reward"] is not None
    assert len([r for r in rows if r["event"] == "reaction"]) == 15  # every label stays in the log
    assert same(load_director([rows], seed=3), online)


class PausingHold(Scripted):
    """Opens a pause when the hold of `at` begins and resumes it after a few loop turns, 30 s later on the show clock.
    The next line may not start until then."""

    def __init__(self, levels, clock, call, at):
        super().__init__(levels)
        self.clock, self.call, self.at, self.tasks = clock, call, at, []

    async def hold(self, bit, joke, seconds, clock):
        if (bit, joke) == self.at:
            await self.call("pause")

            async def later():
                await asyncio.sleep(0.05)
                self.clock.t += 30
                await self.call("resume")
            self.tasks.append(asyncio.ensure_future(later()))
        return await super().hold(bit, joke, seconds, clock)


@pytest.mark.parametrize("at", [(0, 0), (0, 4)], ids=["between jokes", "between bits"])
def test_a_pause_after_a_line_holds_the_show_and_drops_nothing(tmp_path, at):
    rows, _ = run(tmp_path, lambda clock, call: FakePlayer(clock),
                  lambda clock, call, pauses: PausingHold([3, 2, 1, 0, 3], clock, call, at), n_bits=2)
    (pause,) = [r for r in rows if r["event"] == "pause"]
    (resume,) = [r for r in rows if r["event"] == "resume"]
    after = [r for r in rows if r["event"] == "joke_played"
             and (r["bit"], r["joke"]) == ((0, 1) if at == (0, 0) else (1, 0))]
    assert after[0]["audio_start_at"] >= resume["resumed_at"] > pause["paused_at"]  # it waited for the resume
    assert joke_levels(rows, 0) == [3, 2, 1, 0, 3] and joke_levels(rows, 1) == [3, 2, 1, 0, 3]
    assert [r["n_labels"] for r in rows if r["event"] == "bit_end"] == [5, 5]


def test_a_cached_line_after_a_pause_waits_for_the_resume(tmp_path):
    pytest.importorskip("fastapi")
    from app.cache import CachedBit, CachedLine
    from app.show import _Cached, _play_cached

    class Player:
        def __init__(self, clock):
            self.clock = clock

        async def play(self, wav, text):
            self.clock.t += 1

    async def go():
        clock, pauses = FakeClock(), Pauses()
        log = SessionLog(tmp_path)
        entry = _Cached("line", "cynic", {"session": "s", "bit": 0, "joke": 0, "mode": "cynic"}, wav=b"", seconds=1.0)
        pauses.pause(clock.t)
        task = asyncio.ensure_future(_play_cached(entry, Player(clock), bit=0, clock=clock, log=log, pauses=pauses))
        await asyncio.sleep(0.02)
        assert not task.done()  # held
        clock.t += 30
        pauses.resume(clock.t)
        await task
        return clock.t

    asyncio.run(go())
    (row,) = [r for r in read_session(next(tmp_path.glob("*.jsonl"))) if r["event"] == "bridge"]
    assert row["audio_start_at"] == 1030.0


# --- the reaction window and the hold freeze while paused ---------------------------------------------------------

def _source(pauses):
    class Clock:
        def now(self):
            return 0.0

    async def send(msg):
        pass
    return ButtonSource(send, Clock(), pauses)


def test_a_pause_longer_than_the_window_does_not_time_it_out():
    async def go():
        pauses = Pauses()
        source = _source(pauses)
        task = asyncio.ensure_future(source.reaction(0, 0, 0.3))
        await asyncio.sleep(0.1)
        pauses.pause(1.0)
        await asyncio.sleep(0.5)  # longer than the whole window
        assert not task.done()
        source.press(0, 0, "W")  # a key pressed while paused still counts
        pauses.resume(2.0)
        return await asyncio.wait_for(task, 1)

    level, key, _ = asyncio.run(go())
    assert (level, key) == (3, "W")


def test_the_time_left_after_a_pause_is_what_was_left_before_it():
    async def go():
        pauses, loop = Pauses(), asyncio.get_running_loop()
        source = _source(pauses)
        task = asyncio.ensure_future(source.reaction(0, 0, 0.3))
        await asyncio.sleep(0.2)
        pauses.pause(1.0)
        await asyncio.sleep(0.4)
        pauses.resume(2.0)
        resumed = loop.time()
        result = await asyncio.wait_for(task, 1)
        return result, loop.time() - resumed

    (level, key, _), waited = asyncio.run(go())
    assert level is None and key is None
    assert 0.03 < waited < 0.2  # about the 0.1 s that was left, not a fresh 0.3 s and not nothing


def test_a_pause_freezes_the_hold_and_a_press_after_it_still_changes_the_label():
    async def go():
        pauses = Pauses()
        source = _source(pauses)
        first = asyncio.ensure_future(source.reaction(0, 0, 1))
        await asyncio.sleep(0.01)
        source.press(0, 0, "K")
        await first
        task = asyncio.ensure_future(source.hold(0, 0, 0.2, None))
        await asyncio.sleep(0.05)
        pauses.pause(1.0)
        await asyncio.sleep(0.4)
        assert not task.done()
        pauses.resume(2.0)
        source.press(0, 0, "W")
        return await asyncio.wait_for(task, 1)

    level, key, _ = asyncio.run(go())
    assert (level, key) == (3, "W")
