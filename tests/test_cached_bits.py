"""Cached bits (D-90): which bits are popular, the safety re-check for the room at hand, how the show plays them, that
they leave no label behind, and Exp 23's scoring on synthetic logs. Fake voice and clock; no API, no Kokoro."""
import asyncio
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import cache, safety  # noqa: E402
from app import log as log_module  # noqa: E402
from app.cache import CachedBit, CachedLine, popular_bits, screen  # noqa: E402
from app.director import load_director  # noqa: E402
from app.log import SessionLog, read_session, validate  # noqa: E402
from app.rooms import RealClock, SimRoom  # noqa: E402
from app.show import OPEN_CAP, UniformSelector, _Cached, _next_cached, run_show  # noqa: E402
from sim import evaluate_cached as ev  # noqa: E402
from tests.test_generate import CTX, FAST, FakePlayer, FakeVoice, Filter, Gen, played  # noqa: E402
from tests.test_replay import MODES, START, react, same  # noqa: E402


@pytest.fixture
def frozen_time(monkeypatch):
    """Every row's `at` is `now[0]`, so a test places a show in time by setting it."""
    now = [1000.0]
    monkeypatch.setattr(log_module.time, "time", lambda: now[0])
    return now


def live_show(directory, bits, room="live", at=1000.0, now=None, age="mixed", bank=()):
    """bits: (mode, levels, line texts[, bit_end reward to write]). One finished show of `room`; returns its rows.
    A text in `bank` was played from the hand-written bank (its candidates row says so)."""
    if now is not None:
        now[0] = at
    log = SessionLog(directory)
    log.write("show_start", room=room, **{**START, "context": {**CTX, "age": age}})
    for i, (mode, levels, texts, *fake) in enumerate(bits):
        log.write("bit_decision", bit=i, obs={"t": i, "n_bits": 8, "context": CTX, "last_mode": None},
                  scores={m: 0.0 for m in MODES}, greedy=mode, action=mode, probs={m: 1 / 6 for m in MODES})
        for joke, text in enumerate(texts):
            from_bank = text in bank
            log.write("candidates", bit=i, joke=joke, prompt_id="p", prompt="", generator_id="bank-v1" if from_bank else "gen",
                      items=[{"text": text, "verdict": "pass",
                              "reason": "hand-checked bank: voice failed (bank-v1); rules: ok" if from_bank else None}])
            log.write("joke_played", bit=i, joke=joke, chosen=0, pick_prob=1.0, text=text, tts_seconds=0.1,
                      audio_seconds=1.0, audio_start_at=1.0, audio_end_at=2.0)
            react(log, i, joke, levels[joke])
        given = [lv for lv in levels if lv is not None]
        reward = fake[0] if fake else sum(given) / len(given) / 3 if given else None
        log.write("bit_end", bit=i, n_labels=len(given), reward_version=1, reward=reward)
    log.write("show_end", status="finished")
    return read_session(log.path)


def lines(tag, n=5):
    return [f"{tag} line {j}" for j in range(n)]


# --- which bits are popular ---


def test_bits_are_ranked_by_reward_and_a_tie_goes_to_the_later_show(tmp_path, frozen_time):
    early = live_show(tmp_path, [("neurotic", [2] * 5, lines("a")), ("neurotic", [3] * 5, lines("b"))], at=1000.0, now=frozen_time)
    late = live_show(tmp_path, [("neurotic", [2] * 5, lines("c"))], at=2000.0, now=frozen_time)
    got = popular_bits([early, late], per_mode=3)["neurotic"]
    assert [b.lines[0].text for b in got] == ["b line 0", "c line 0", "a line 0"]
    assert [b.reward for b in got] == [1.0, pytest.approx(2 / 3), pytest.approx(2 / 3)]
    assert got[0].session == early[0]["session"] and (got[0].bit, got[0].mode) == (1, "neurotic")
    assert [(ln.joke, ln.text) for ln in got[0].lines] == list(enumerate(lines("b")))


def test_simulated_shows_never_make_a_bit_popular(tmp_path, frozen_time):
    sim = live_show(tmp_path, [("cynic", [3] * 5, lines("s"))], room="sim", now=frozen_time)
    assert popular_bits([sim]) == {}


def test_reward_is_recomputed_from_the_labels_not_read_from_bit_end(tmp_path, frozen_time):
    rows = live_show(tmp_path, [("optimist", [1] * 5, lines("liar"), 1.0), ("optimist", [3] * 5, lines("honest"), 0.0)],
                     now=frozen_time)
    got = popular_bits([rows])["optimist"]
    assert [b.lines[0].text for b in got] == ["honest line 0", "liar line 0"] and [b.reward for b in got] == [1.0, pytest.approx(1 / 3)]


def test_a_bit_with_too_few_labels_is_skipped(tmp_path, frozen_time):
    rows = live_show(tmp_path, [("cynic", [3, 3, None, None, None], lines("two")), ("cynic", [3, 3, 3, None, None], lines("three")),
                                ("cynic", [1] * 5, lines("five"))], now=frozen_time)
    # two labels is no reward at all (reward_v1), three passes the default, a stricter min_labels drops it
    assert [b.lines[0].text for b in popular_bits([rows], per_mode=3)["cynic"]] == ["three line 0", "five line 0"]
    assert [b.lines[0].text for b in popular_bits([rows], per_mode=3, min_labels=4)["cynic"]] == ["five line 0"]


def test_a_bit_that_repeats_a_better_ones_lines_is_dropped(tmp_path, frozen_time):
    a = live_show(tmp_path, [("cynic", [3] * 5, lines("same"))], at=1000.0, now=frozen_time)
    b = live_show(tmp_path, [("cynic", [1] * 5, lines("same")), ("cynic", [1] * 5, lines("other"))], at=2000.0, now=frozen_time)
    got = popular_bits([a, b], per_mode=3)["cynic"]
    assert [x.lines[0].text for x in got] == ["same line 0", "other line 0"] and got[0].session == a[0]["session"]


def test_each_mode_keeps_only_its_best_per_mode(tmp_path, frozen_time):
    rows = live_show(tmp_path, [(m, [lv] * 5, lines(f"{m}{lv}")) for m in ("cynic", "optimist") for lv in (1, 2, 3)],
                     now=frozen_time)
    got = popular_bits([rows], per_mode=2)
    assert set(got) == {"cynic", "optimist"}
    assert [b.reward for b in got["cynic"]] == [1.0, pytest.approx(2 / 3)]
    assert popular_bits([rows], per_mode=1)["optimist"][0].reward == 1.0


def test_a_line_the_bank_played_is_never_cached_and_a_level_zero_line_is_dropped(tmp_path, frozen_time):
    rows = live_show(tmp_path, [("cynic", [3, 3, 3, 0, 3], lines("g")), ("optimist", [3] * 5, lines("b"))],
                     now=frozen_time, bank={"g line 1", "b line 0", "b line 1", "b line 2", "b line 3", "b line 4"})
    got = popular_bits([rows])
    assert [ln.text for ln in got["cynic"][0].lines] == ["g line 0", "g line 2", "g line 4"]  # 1: bank, 3: level 0
    assert "optimist" not in got  # every line came from the bank, so the bit is gone


def test_a_bit_carries_its_source_shows_age_and_venue(tmp_path, frozen_time):
    rows = live_show(tmp_path, [("cynic", [3] * 5, lines("t"))], age="teens", now=frozen_time)
    b = popular_bits([rows])["cynic"][0]
    assert (b.age, b.venue) == ("teens", CTX["venue"])


# --- the safety re-check ---


def bit_of(*texts, mode="neurotic", reward=0.9, age="mixed", venue="bar", session="s"):
    return CachedBit(session, 0, mode, reward, tuple(CachedLine(i, t) for i, t in enumerate(texts)), age, venue)


def test_a_line_that_passed_a_mixed_room_is_dropped_for_a_teen_room():
    beer = "I drank a beer before the show"
    cached = {"neurotic": [bit_of("My code is anxious.", beer, "My fan is loud.", age="teens")]}
    mixed, _ = screen(cached, safety.check_rules, "mixed")
    teens, report = screen(cached, safety.check_rules, "teens")
    assert [ln.text for ln in mixed[0].lines] == ["My code is anxious.", beer, "My fan is loud."]
    assert [ln.text for ln in teens[0].lines] == ["My code is anxious.", "My fan is loud."]
    assert [(x["verdict"], x["reason"]) for x in report[0]["lines"]][1] == ("reject", "rules: teens_alcohol:beer")
    assert screen({"neurotic": [bit_of(beer, age="teens")]}, safety.check_rules, "teens")[0] == []  # the bit goes


def test_a_teen_room_takes_only_bits_from_teen_shows():
    clean = "My fan is loud."  # passes the rules for teens: only the source audience tells the rooms apart
    cached = {"neurotic": [bit_of(clean, age="mixed", session="adult")], "cynic": [bit_of("My code sighs.", mode="cynic", age="teens", session="teen")]}
    kept, report = screen(cached, safety.check_rules, "teens")
    assert [b.session for b in kept] == ["teen"]
    assert next(b for b in report if b["session"] == "adult")["lines"][0]["reason"] == "teen room, source show was mixed"
    assert next(b for b in report if b["session"] == "adult")["source_age"] == "mixed"
    assert len(screen(cached, safety.check_rules, "mixed")[0]) == 2  # an adult room takes both
    odd, _ = screen({"neurotic": [bit_of(clean, age="kids")]}, safety.check_rules, "children")  # any non-standard age is teens
    assert len(odd) == 1
    assert screen({"neurotic": [bit_of(clean, age="mixed")]}, safety.check_rules, "kids")[0] == []


def test_a_line_in_two_pooled_bits_is_kept_once_and_a_same_venue_bit_goes_first():
    cached = {"neurotic": [bit_of("same", "a1", reward=0.9, venue="bar", session="x")],
              "cynic": [bit_of("same", "b1", mode="cynic", reward=0.5, venue="club", session="y")]}
    kept, report = screen(cached, safety.check_rules, "mixed", venue="club")
    assert [b.session for b in kept] == ["y", "x"]  # the club bit leads despite the lower reward
    assert [ln.text for ln in kept[0].lines] == ["same", "b1"] and [ln.text for ln in kept[1].lines] == ["a1"]
    assert report[1]["lines"][0]["reason"] == "duplicate of a pooled line"


def test_a_filter_that_cannot_decide_drops_the_line():
    def broken(text, age):
        raise RuntimeError("down")
    kept, report = screen({"neurotic": [bit_of("anything")]}, broken, "mixed")
    assert kept == [] and report[0]["lines"][0]["reason"] == "filter error: RuntimeError"


def test_the_show_re_checks_cached_lines_for_its_own_room_and_never_says_a_failing_one(tmp_path):
    beer = "I drank a beer before the show"
    cached = {"neurotic": [bit_of("My code is anxious.", beer, age="teens")]}
    teen_ctx = {**CTX, "age": "teens"}
    voice, log = FakeVoice(), SessionLog(tmp_path)
    asyncio.run(run_show(load_director([], seed=1), "sim", teen_ctx, generator=Gen(delay=0.5), safety=_RulesFilter(),
                         selector=UniformSelector(0), tts=voice, player=FakePlayer(voice, seconds=0.05),
                         source=SimRoom(teen_ctx, 0), clock=RealClock(), log=log, n_bits=1, config=FAST, cached=cached))
    spoken = [t for k, t in voice.events if k == "start"]
    assert "My code is anxious." in spoken and beer not in spoken
    assert beer not in {t for k, t in voice.events if k == "synth"}  # not even rendered


class _RulesFilter(Filter):
    """The real rules for check_rules, the fake rule for the per-bit check."""

    def check_rules(self, text, age):
        return safety.check_rules(text, age)


# --- how the show plays them ---


def run(tmp_path, cached, gen=None, voice=None, bridges=None, n_bits=3, seconds=0.05, ctx=CTX):
    voice = voice or FakeVoice()
    log = SessionLog(tmp_path)
    asyncio.run(run_show(load_director([], seed=1), "sim", ctx, generator=gen or Gen(delay=0.1), safety=_RulesFilter(),
                         selector=UniformSelector(0), tts=voice, player=FakePlayer(voice, seconds=seconds),
                         source=SimRoom(ctx, 0), clock=RealClock(), log=log, n_bits=n_bits, config=FAST,
                         bridge_lines=bridges, cached=cached))
    return read_session(log.path), voice


def pool(n_bits=2, n_lines=5, modes=("neurotic", "cynic")):
    return {m: [CachedBit(f"s{m}{i}", i, m, 0.9 - i / 10, tuple(CachedLine(j, f"cached {m} {i}.{j}") for j in range(n_lines)))
                for i in range(n_bits)] for m in modes}


def test_the_show_opens_with_cached_lines_only_while_bit_zero_is_still_preparing(tmp_path):
    # prep takes 5 x 0.1 s (the fake generator sleeps per joke); a line plays for 0.35 s: line 0 (0 to 0.35) and
    # line 1 (0.35 to 0.7) start before prep is done, line 2 does not.
    rows, voice = run(tmp_path, pool(), gen=Gen(delay=0.1), seconds=0.35, n_bits=1)
    plays = [t for k, t in voice.events if k == "start"]
    opener = [t for t in plays if t.startswith("cached")]
    assert 1 <= len(opener) < OPEN_CAP  # prep ends mid-open: never the whole cap
    assert plays[:2] == opener and plays[2].startswith("ok 0.0")  # the bit follows the cached lines at once
    assert len({t.split()[1] for t in opener}) == 1 and opener == sorted(opener)  # one bit's lines, in order


def test_the_open_never_exceeds_its_cap_even_when_prep_is_slow(tmp_path):
    _, voice = run(tmp_path, pool(), gen=Gen(delay=0.4), seconds=0.05, n_bits=1)
    assert sum(t.startswith("cached") for k, t in voice.events if k == "start") == OPEN_CAP


def test_the_open_prefers_a_cached_bit_of_bit_zeros_mode_else_any(tmp_path):
    rows, voice = run(tmp_path, pool(), gen=Gen(delay=0.1), seconds=0.05, n_bits=1)
    mode = next(r["action"] for r in rows if r["event"] == "bit_decision")
    first = next(t for k, t in voice.events if k == "start")
    assert first.startswith(f"cached {mode} 0.")  # the mode's best bit
    rows, voice = run(tmp_path, {"cynic": pool()["cynic"]}, gen=Gen(delay=0.1), seconds=0.05, n_bits=1)
    # a director that does not pick cynic still gets a cached opener, from whichever mode has one
    assert next(t for k, t in voice.events if k == "start").startswith("cached cynic")


def test_no_cache_means_no_wait_and_no_change(tmp_path):
    rows, voice = run(tmp_path, None, gen=Gen(delay=0.02), n_bits=1)
    assert not [r for r in rows if r["event"] == "bridge"] and rows[-1]["status"] == "finished"
    rows, voice = run(tmp_path, {}, gen=Gen(delay=0.02), n_bits=1)
    assert not [r for r in rows if r["event"] == "bridge"]


def test_a_cached_line_is_a_bridge_row_marked_cached_with_its_source_and_nothing_else(tmp_path):
    rows, voice = run(tmp_path, pool(), gen=Gen(delay=0.1), seconds=0.1, n_bits=2)
    cached_rows = [r for r in rows if r["event"] == "bridge" and r["kind"] == "cached"]
    assert cached_rows and cached_rows[0]["bit"] == 0
    for r in cached_rows:
        validate(r)
        assert r["text"].startswith("cached") and set(r["source"]) == {"session", "bit", "joke", "mode"}
        assert r["text"] == f"cached {r['source']['mode']} {r['source']['bit']}.{r['source']['joke']}"
    # every reaction, candidate and joke row belongs to a generated joke: a cached line opened no window
    cached_texts = {r["text"] for r in cached_rows}
    assert not cached_texts & {r["text"] for r in rows if r["event"] == "joke_played"}
    keys = {(r["bit"], r["joke"]) for r in rows if r["event"] == "joke_played"}
    assert {(r["bit"], r["joke"]) for r in rows if r["event"] in ("reaction", "gap", "candidates")} == keys
    assert len([r for r in rows if r["event"] == "bit_end"]) == 2 and len(keys) == 10


def test_cached_lines_leave_the_director_as_it_would_have_been_without_them(tmp_path):
    with_cache, _ = run(tmp_path, pool(), gen=Gen(delay=0.05), n_bits=3)
    without, _ = run(tmp_path, None, gen=Gen(delay=0.05), n_bits=3)
    d1, d2 = load_director([with_cache], room="sim", seed=1), load_director([without], room="sim", seed=1)
    assert same(d1, d2)


def test_between_bits_cached_lines_come_before_generic_ones_and_none_repeats(tmp_path):
    cached = pool(n_bits=1, n_lines=2)  # 4 cached lines in all
    rows, voice = run(tmp_path, cached, gen=Gen(delay=0.15), bridges=["b1", "b2", "b3", "b4", "b5"], n_bits=4)
    bridges = [r for r in rows if r["event"] == "bridge"]
    texts, kinds = [r["text"] for r in bridges], [r["kind"] for r in bridges]
    said = [t for t, k in zip(texts, kinds) if k == "cached"]
    assert len(set(said)) == len(said) == 4  # a cached line is never said twice (fillers repeat once all are used)
    assert "filler" in kinds
    assert kinds[:kinds.index("filler")].count("cached") == 4  # every cached line goes before the first filler


def test_the_next_bits_mode_is_preferred_between_bits():
    def entry(text, mode, ready=True):
        return _Cached(text, mode, {}, wav=b"w" if ready else None, ready=asyncio.Event())
    groups = [[entry("n0", "neurotic"), entry("n1", "neurotic")], [entry("c0", "cynic")], [entry("o0", "optimist", False)]]
    assert _next_cached(groups, "cynic").text == "c0"
    assert _next_cached(groups, "optimist").text == "n0"  # its only line is not rendered: never wait, take the best
    groups[0][0].used = True
    assert _next_cached(groups, "neurotic").text == "n1"
    for g in groups[:2]:
        for e in g:
            e.used = True
    assert _next_cached(groups, "optimist") is None


def test_a_cached_line_that_will_not_render_never_stops_the_show(tmp_path):
    class Down(FakeVoice):
        def synthesize(self, text):
            if text.startswith("cached"):
                raise RuntimeError("tts down")
            return super().synthesize(text)
    rows, voice = run(tmp_path, pool(), gen=Gen(delay=0.3), voice=Down(), n_bits=2)
    assert rows[-1]["status"] == "finished" and len(played(rows)) == 10
    assert not [r for r in rows if r["event"] == "bridge"]


def test_the_bank_stack_plays_cached_lines_that_are_in_the_bank_only():
    from app.live import BankOnlySafety
    from app.bank import load_bank
    bank_text = next(iter(BankOnlySafety().allowed))
    cached = {"neurotic": [bit_of(bank_text, "A line from an llm show.")]}
    kept, _ = screen(cached, BankOnlySafety().check, "mixed")
    assert [ln.text for ln in kept[0].lines] == [bank_text] and load_bank()


# --- replay and the log format ---


def bridge_fields(**over):
    base = {"schema": 2, "session": "s", "at": 1.0, "event": "bridge", "bit": 0, "text": "t", "tts_seconds": 0.1,
            "audio_start_at": 1.0, "audio_end_at": 2.0}
    return {**base, **over}


def test_bridge_rows_validate_with_and_without_the_new_fields():
    validate(bridge_fields())  # a log written before `kind` existed
    validate(bridge_fields(kind="filler"))
    validate(bridge_fields(kind="cached", source={"session": "s", "bit": 1, "joke": 2}))
    validate(bridge_fields(kind="cached", source={"session": "s", "bit": 1, "joke": 2, "mode": "cynic"}))
    for bad in (bridge_fields(kind="popular"), bridge_fields(kind=None), bridge_fields(kind="cached"),
                bridge_fields(kind="cached", source={"session": "s", "bit": 1})):
        with pytest.raises(ValueError):
            validate(bad)


def test_a_director_rebuilt_from_a_log_with_cached_bridge_rows_equals_one_without(tmp_path, frozen_time):
    def show(with_bridges, at):
        frozen_time[0] = at
        log = SessionLog(tmp_path)
        log.write("show_start", room="live", **START)
        last = None
        for i, (mode, levels) in enumerate([("neurotic", [3, 3, 2, 3, 3]), ("cynic", [0, 1, 0, 0, 1]), ("neurotic", [2, 2, 3, 2, 2])]):
            if with_bridges:
                log.write("bridge", bit=i, text="x", tts_seconds=0.1, audio_start_at=1.0, audio_end_at=2.0, kind="cached",
                          source={"session": "old", "bit": 3, "joke": i})
                log.write("bridge", bit=i, text="y", tts_seconds=0.1, audio_start_at=2.0, audio_end_at=3.0, kind="filler")
            log.write("bit_decision", bit=i, obs={"t": i, "n_bits": 8, "context": CTX, "last_mode": last},
                      scores={m: 0.0 for m in MODES}, greedy=mode, action=mode, probs={m: 1 / 6 for m in MODES})
            for j, lv in enumerate(levels):
                react(log, i, j, lv)
            log.write("bit_end", bit=i, n_labels=5, reward_version=1, reward=0.5)
            last = mode
        log.write("show_end", status="finished")
        return read_session(log.path)
    plain, bridged = show(False, 1000.0), show(True, 1000.0)
    assert any(r["event"] == "bridge" for r in bridged) and not any(r["event"] == "bridge" for r in plain)
    d1, d2 = load_director([plain]), load_director([bridged])
    assert same(d1, d2)
    assert popular_bits([plain]).keys() == popular_bits([bridged]).keys()  # and the cache itself ignores bridge rows


# --- Exp 23's scoring ---


def timeline(frozen_time, tmp_path, first, kind="cached", mode="neurotic", room="sim", generator="gemini-x", status="finished",
             gap=None, pool="neurotic", verdict="pass"):
    """A show starting at t=1000 whose first audio starts `first` seconds in: an opener (kind) then bit 0's joke.
    `pool` is the mode of the one bit in the show's `cache` row (None: no row)."""
    frozen_time[0] = 1000.0
    log = SessionLog(tmp_path)
    log.write("show_start", room=room, **{**START, "generator_id": generator})
    if pool:
        log.write("cache", bits=[{"session": "o", "bit": 1, "mode": pool, "reward": 0.7, "source_age": "mixed",
                                  "lines": [{"joke": 0, "text": "c", "verdict": verdict, "reason": "rules: ok"}]}])
    log.write("bit_decision", bit=0, obs={"t": 0, "n_bits": 8, "context": CTX, "last_mode": None},
              scores={m: 0.0 for m in MODES}, greedy=mode, action=mode, probs={m: 1 / 6 for m in MODES})
    t = 1000.0 + first
    extra = {"kind": "cached", "source": {"session": "o", "bit": 1, "joke": 0, "mode": mode}} if kind == "cached" else {"kind": "filler"}
    if kind:
        log.write("bridge", bit=0, text="c", tts_seconds=0.1, audio_start_at=t, audio_end_at=t + 3.0, **extra)
    log.write("joke_played", bit=0, joke=0, chosen=0, pick_prob=1.0, text="j", tts_seconds=0.1, audio_seconds=1.0,
              audio_start_at=t + 3.0 + (gap if gap is not None else 0.5), audio_end_at=t + 6.0)
    log.write("gap", bit=0, joke=0, planned_hold_s=1.0, actual_gap_s=1.0)
    if status:
        log.write("show_end", status=status)
    return log.session_id


def test_open_silence_runs_from_show_start_to_the_first_audio_of_any_kind():
    rows = [{"event": "show_start", "at": 100.0}, {"event": "joke_played", "audio_start_at": 109.0},
            {"event": "bridge", "audio_start_at": 101.5}]
    assert ev.open_silence(rows) == pytest.approx(1.5)
    assert ev.open_silence(rows[:2]) == pytest.approx(9.0)  # a show with no opener waits for its first joke
    assert ev.open_silence([rows[0]]) is None and ev.open_silence([]) is None
    assert ev.open_silence([rows[0], {"event": "bridge", "audio_start_at": 99.0}]) == 0.0


def test_between_bit_silence_counts_cached_and_filler_lines_as_audio():
    def joke(j, s, e, bit=0):
        return {"event": "joke_played", "bit": bit, "joke": j, "audio_start_at": s, "audio_end_at": e}
    rows = [{"event": "show_start", "at": 0.0},
            {"event": "bridge", "bit": 0, "audio_start_at": 1.0, "audio_end_at": 4.0, "kind": "cached"},
            joke(0, 4.5, 6.0), {"event": "gap", "at": 8.0}, joke(1, 8.25, 9.0),
            {"event": "gap", "at": 12.0},
            {"event": "bridge", "bit": 1, "audio_start_at": 12.5, "audio_end_at": 15.0, "kind": "cached"},
            joke(0, 18.0, 19.0, bit=1)]
    assert ev.between_bit_silence(rows) == [pytest.approx(0.5), pytest.approx(0.5), pytest.approx(3.0)]


def rows_of(tmp_path, sid):
    return read_session(tmp_path / f"{sid}.jsonl")


def test_a_show_qualifies_by_the_cache_it_had_at_the_start_not_by_what_it_played(frozen_time, tmp_path):
    had = timeline(frozen_time, tmp_path, 1.2)
    assert ev.had_cache_for_bit_zero(rows_of(tmp_path, had))
    assert not ev.had_cache_for_bit_zero(rows_of(tmp_path, timeline(frozen_time, tmp_path, 8.0, kind=None, pool=None)))
    assert not ev.had_cache_for_bit_zero(rows_of(tmp_path, timeline(frozen_time, tmp_path, 8.0, kind=None, pool="cynic")))
    assert not ev.had_cache_for_bit_zero(rows_of(tmp_path, timeline(frozen_time, tmp_path, 8.0, kind=None, verdict="reject")))
    # a cached bit of bit 0's mode that never played (slow or failed render) still qualifies, and is scored as it ran
    failed = timeline(frozen_time, tmp_path, 9.0, kind=None)
    s = ev.show_report(failed, tmp_path)
    assert s["counts"] and s["open_silence_s"] == pytest.approx(12.5)  # 9 s + the fake joke's 3.5 s offset
    assert ev.show_report(had, tmp_path)["open_silence_s"] == pytest.approx(1.2)


def test_the_bar_needs_three_qualifying_shows_each_at_most_two_seconds(frozen_time, tmp_path):
    good = [timeline(frozen_time, tmp_path, x) for x in (0.8, 1.5, 2.0)]
    res = ev.run(good, directory=tmp_path)
    assert res["n_counted"] == 3 and res["pass_open"] and res["open_max_s"] == pytest.approx(2.0)
    late = ev.run([*good[:2], timeline(frozen_time, tmp_path, 2.1)], directory=tmp_path)
    assert late["n_counted"] == 3 and not late["pass_open"] and "MISS" in ev.verdict(late)
    slow_render = ev.run([*good[:2], timeline(frozen_time, tmp_path, 9.0, kind=None)], directory=tmp_path)
    assert slow_render["n_counted"] == 3 and not slow_render["pass_open"]  # a MISS, not an exclusion
    two = ev.run(good[:2], directory=tmp_path)
    assert not two["pass_open"] and "NOT DECIDED" in ev.verdict(two)
    # a show that does not qualify is reported and not counted: bank, live, aborted, no cache for bit 0's mode
    extras = [timeline(frozen_time, tmp_path, 0.5, generator="bank-v1"), timeline(frozen_time, tmp_path, 0.5, room="live"),
              timeline(frozen_time, tmp_path, 0.5, status="aborted"), timeline(frozen_time, tmp_path, 9.0, kind=None, pool=None)]
    mixed = ev.run([*good[:2], *extras], directory=tmp_path)
    assert mixed["n_counted"] == 2 and not mixed["pass_open"]
    assert [s["counts"] for s in mixed["shows"]] == [True, True, False, False, False, False]
    assert "NO:" in ev.table(ev.run(["nope"], directory=tmp_path))


def test_the_report_stores_numbers_and_ids_only(frozen_time, tmp_path):
    sid = timeline(frozen_time, tmp_path, 1.0, gap=4.0)
    res = ev.run([sid], directory=tmp_path)
    s = res["shows"][0]
    assert s["between_n"] == 1 and s["between_max_s"] == pytest.approx(4.0) and s["n_cached"] == 1
    blob = str(res)
    assert "\"c\"" not in blob and "'text'" not in blob  # no line text anywhere
    assert ev.digest([sid], tmp_path) != ev.digest([sid, "missing"], tmp_path)


# --- review round 1: the log, the stack, aborts and the voice ---


def test_the_show_logs_its_cache_pool_right_after_show_start_with_every_verdict(tmp_path):
    beer = "I drank a beer before the show"
    cached = {"neurotic": [bit_of("My code is anxious.", beer, age="teens", session="src")],
              "cynic": [bit_of("My code is anxious.", "A fresh one.", mode="cynic", age="mixed", session="adult")]}
    teen_ctx = {**CTX, "age": "teens"}
    rows, _ = run(tmp_path, cached, gen=Gen(delay=0.05), ctx=teen_ctx, n_bits=1)
    assert [r["event"] for r in rows[:2]] == ["show_start", "cache"]
    by = {b["session"]: b for b in rows[1]["bits"]}
    assert [(x["text"], x["verdict"]) for x in by["src"]["lines"]] == [("My code is anxious.", "pass"), (beer, "reject")]
    assert by["src"]["source_age"] == "teens" and by["src"]["mode"] == "neurotic" and by["src"]["reward"] == 0.9
    assert all(x["verdict"] == "reject" and "teen room" in x["reason"] for x in by["adult"]["lines"])
    rows, _ = run(tmp_path, None, gen=Gen(delay=0.02), n_bits=1)
    assert not [r for r in rows if r["event"] == "cache"]  # no cache, no row


def test_a_log_with_a_cache_row_validates_without_it_and_replays_the_same(tmp_path, frozen_time):
    validate({"schema": 2, "session": "s", "at": 1.0, "event": "cache", "bits": []})
    with pytest.raises(ValueError):
        validate({"schema": 1, "session": "s", "at": 1.0, "event": "cache", "bits": []})
    with pytest.raises(ValueError):
        validate({"schema": 2, "session": "s", "at": 1.0, "event": "cache", "bits": [{"session": "x"}]})
    with_cache, _ = run(tmp_path, pool(), gen=Gen(delay=0.05), n_bits=2)
    assert any(r["event"] == "cache" for r in with_cache)
    stripped = [r for r in with_cache if r["event"] not in ("cache", "bridge")]
    assert same(load_director([with_cache], room="sim", seed=1), load_director([stripped], room="sim", seed=1))


def test_only_the_llm_stack_gets_cached_lines():
    sessions = []
    assert cache.cached_for(None, sessions) is None  # the bank stack has no filler, so no cached line either
    assert cache.cached_for(["a bridge"], sessions) == {}
    called = []
    orig = cache.popular_bits
    cache.popular_bits = lambda s: called.append(s) or {"x": []}
    try:
        assert cache.cached_for(None, sessions) is None and not called
        assert cache.cached_for(["b"], sessions) == {"x": []} and called
    finally:
        cache.popular_bits = orig


def test_the_bank_stack_run_end_to_end_plays_no_cached_row(tmp_path):
    from app.bank import BankGenerator
    from app.live import BankOnlySafety
    gen, flt, fallback, bridges = BankGenerator(), BankOnlySafety(), None, None
    voice, log = FakeVoice(), SessionLog(tmp_path)
    asyncio.run(run_show(load_director([], seed=1), "sim", CTX, generator=gen, safety=flt, selector=UniformSelector(0),
                         tts=voice, player=FakePlayer(voice, seconds=0.01), source=SimRoom(CTX, 0), clock=RealClock(),
                         log=log, n_bits=2, config=FAST, fallback=fallback, bridge_lines=bridges,
                         cached=cache.cached_for(bridges, [])))
    assert not [r for r in read_session(log.path) if r["event"] in ("bridge", "cache")]


def test_an_abort_during_the_open_leaves_no_pending_task_and_writes_show_end_aborted(tmp_path):
    voice, log = FakeVoice(), SessionLog(tmp_path)

    async def go():
        task = asyncio.ensure_future(run_show(load_director([], seed=1), "sim", CTX, generator=Gen(delay=0.5), safety=_RulesFilter(),
                                              selector=UniformSelector(0), tts=voice, player=FakePlayer(voice, seconds=0.05),
                                              source=SimRoom(CTX, 0), clock=RealClock(), log=log, n_bits=1, config=FAST,
                                              cached=pool(n_lines=3)))
        await asyncio.sleep(0.3)  # inside the open: lines played, a wait for the next render or for prep
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)
        return [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
    assert asyncio.run(go()) == []
    assert read_session(log.path)[-1] == {**read_session(log.path)[-1], "event": "show_end", "status": "aborted"}


def test_extra_cached_renders_go_round_robin_by_mode():
    from app.show import _round_robin

    def group(mode, i, n=2):
        return [_Cached(f"{mode}{i}.{j}", mode, {}, ready=asyncio.Event()) for j in range(n)]
    groups = [group("a", 0), group("a", 1), group("b", 0), group("c", 0)]
    got = [e.text for e in _round_robin(groups, skip=groups[0])]
    assert got[:3] == ["a1.0", "b0.0", "c0.0"] and sorted(got) == sorted(e.text for g in groups[1:] for e in g)


def test_cached_renders_are_chained_so_bit_zeros_first_line_waits_behind_at_most_one(tmp_path):
    import threading

    class FairSlowVoice(FakeVoice):
        """A serial voice that serves requests in the order they came, like the real engine's lock, and records how
        many cached lines were outstanding (asked for, not yet done) whenever any line asked."""

        def __init__(self):
            super().__init__(delay=0.1)
            self.cond, self.next_ticket, self.serving = threading.Condition(), 0, 0
            self.outstanding, self.at_request = [], []

        def synthesize(self, text):
            with self.cond:
                ticket = self.next_ticket
                self.next_ticket += 1
                self.at_request.append((text, sum(t.startswith("cached") for t in self.outstanding)))
                self.outstanding.append(text)
                while ticket != self.serving:
                    self.cond.wait()
            try:
                return super().synthesize(text)
            finally:
                with self.cond:
                    self.serving += 1
                    self.outstanding.remove(text)
                    self.cond.notify_all()
    voice = FairSlowVoice()
    rows, _ = run(tmp_path, pool(n_bits=2, n_lines=5), gen=Gen(delay=0.05), voice=voice, n_bits=2, seconds=0.02)
    assert rows[-1]["status"] == "finished" and any(t.startswith("cached") for t, _ in voice.at_request)
    ahead = dict(voice.at_request)  # cached renders in front of each line when it asked
    assert ahead[next(t for t, _ in voice.at_request if t.startswith("ok 0.0"))] <= 1
    # chained: a cached line never asks while another cached line is still outstanding, in the open or after it
    assert all(n == 0 for t, n in voice.at_request if t.startswith("cached"))
