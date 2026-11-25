"""The acceptance scorer (Exp 22) on synthetic logs: it must call a clean show clean, an aborted one not, and recover a
known noise level, so a verdict on the user's real shows means something."""
import json
import pathlib
import sys
import wave

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app import log as log_module  # noqa: E402
from app.log import SessionLog  # noqa: E402
from sim import acceptance  # noqa: E402
from sim.env import AudienceEnv  # noqa: E402

MODES = AudienceEnv().modes
CTX = {"venue": "bar", "energy": "high", "age": "mixed"}
KEYS = {3: "W", 2: "K", 1: "Space", 0: "D"}
POLICY = {"name": "linucb", "epsilon": 0.2, "alpha": 0.15, "prior": "uniform"}


@pytest.fixture(autouse=True)
def real_clock():
    saved = log_module.time.time
    yield
    log_module.time.time = saved


def write_show(directory, levels_per_bit, room="live", status="finished", rating=None, at=1000.0, bank_bits=(),
               bridges=0, audio=None, generator="llm", pauses=()):
    """A consistent show; `at` is the first row's time. `levels_per_bit`: one list of levels per bit. Returns its id."""
    clock = iter(range(int(at), int(at) + 100_000))
    log_module.time.time = lambda: next(clock)
    log = SessionLog(directory)
    log.write("show_start", room=room, context=CTX, policy=POLICY, generator_id=generator, selector_id="uniform",
              git_sha="x", operator="t", audio=audio or {"consent": False, "path": None})
    for paused_at, resumed_at in pauses:  # (start, end) on the show clock; end None = the show ended paused
        log.write("pause", paused_at=paused_at)
        if resumed_at is not None:
            log.write("resume", resumed_at=resumed_at)
    for bit, levels in enumerate(levels_per_bit):
        log.write("bit_decision", bit=bit, obs={"t": bit, "n_bits": 8, "context": CTX, "last_mode": None},
                  scores={m: 0.0 for m in MODES}, greedy=MODES[0], action=MODES[0], probs={m: 1 / 6 for m in MODES})
        for _ in range(bridges if bit else 0):
            log.write("bridge", bit=bit, text="um", tts_seconds=0.1, audio_start_at=1.0, audio_end_at=2.0)
        for joke, level in enumerate(levels):
            reason = "hand-checked bank: x (bank-v1); rules: ok" if bit in bank_bits else None
            log.write("candidates", bit=bit, joke=joke, prompt_id="p", prompt="p", generator_id=generator,
                      items=[{"text": "t", "verdict": "pass", "reason": reason}])
            log.write("joke_played", bit=bit, joke=joke, chosen=0, pick_prob=1.0, text="t", tts_seconds=0.1,
                      audio_seconds=1.0, audio_start_at=100 * bit + 10 * joke, audio_end_at=100 * bit + 10 * joke + 2)
            log.write("reaction", bit=bit, joke=joke, source="button" if room == "live" else "sim", level=level,
                      key=None if level is None else KEYS[level], latency_s=0.5)
        given = [lv for lv in levels if lv is not None]
        log.write("bit_end", bit=bit, n_labels=len(given), reward_version=1,
                  reward=sum(given) / len(given) / 3 if len(given) >= 3 else None)
    log.write("show_end", status=status)
    if rating:
        log.write("rating", value=rating)
    return log.session_id


def full(level=2):
    return [[level] * 5 for _ in range(8)]


def test_a_clean_simulated_show_is_reported_in_full(tmp_path):
    sid = write_show(tmp_path, full(2), room="sim", rating=4, bank_bits=(0,), bridges=2)
    s = acceptance.show_report(sid, tmp_path)
    assert (s["valid"], s["status"], s["room"], s["generator_id"]) == (True, "finished", "sim", "llm")
    assert (s["n_jokes"], s["n_bits"], s["n_bridges"], s["rating"]) == (40, 8, 14, 4)
    assert s["bank_jokes"] == 5 and s["bank_share"] == pytest.approx(5 / 40)
    assert s["labels"]["counts"] == {"W": 0, "K": 40, "Space": 0, "D": 0, "none": 0}
    assert s["audio"] is None


def test_three_clean_simulated_shows_in_a_row_pass_and_an_abort_breaks_the_row(tmp_path):
    ids = [write_show(tmp_path, full(), room="sim", at=1000 * (i + 1)) for i in range(3)]
    assert acceptance.run(ids, directory=tmp_path)["pass_a_sims_in_a_row"]
    aborted = write_show(tmp_path, full()[:3], room="sim", status="aborted", at=2500)
    res = acceptance.run([ids[0], ids[1], aborted, ids[2]], directory=tmp_path)  # in start order: ok, ok, abort, ok
    assert not res["pass_a_sims_in_a_row"]
    assert acceptance.show_report(aborted, tmp_path)["status"] == "aborted"


def test_two_clean_simulated_shows_are_not_three(tmp_path):
    ids = [write_show(tmp_path, full(), room="sim", at=1000 * (i + 1)) for i in range(2)]
    assert not acceptance.run(ids, directory=tmp_path)["pass_a_sims_in_a_row"]


def test_a_corrupt_log_is_invalid_not_an_exception_and_cannot_pass(tmp_path):
    ids = [write_show(tmp_path, full(), room="sim", at=1000 * (i + 1)) for i in range(3)]
    path = tmp_path / f"{ids[1]}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], "{torn", *lines[1:]]) + "\n", encoding="utf-8")
    res = acceptance.run(ids, directory=tmp_path)
    assert res["shows"][1]["valid"] is False and "corrupt" in res["shows"][1]["error"]
    assert not res["pass_a_sims_in_a_row"]
    assert acceptance.show_report("nope", tmp_path)["valid"] is False


def test_an_unreadable_log_among_the_ids_fails_the_row_even_between_clean_shows(tmp_path):
    ids = [write_show(tmp_path, full(), room="sim", at=1000 * (i + 1)) for i in range(4)]
    assert acceptance.run(ids, directory=tmp_path)["pass_a_sims_in_a_row"]
    path = tmp_path / f"{ids[1]}.jsonl"
    path.write_text("{torn" + chr(10) + path.read_text(encoding="utf-8"), encoding="utf-8")
    assert not acceptance.run(ids, directory=tmp_path)["pass_a_sims_in_a_row"]  # three clean ones remain, still fails


def test_bank_share_counts_a_bank_only_show_and_a_fallback_inside_an_llm_show(tmp_path):
    bank_only = write_show(tmp_path, full(), generator="bank-v1", at=1000)
    mixed = write_show(tmp_path, full(), bank_bits=(0, 1), at=5000)
    plain = write_show(tmp_path, full(), at=9000)
    shares = [acceptance.show_report(i, tmp_path)["bank_share"] for i in (bank_only, mixed, plain)]
    assert shares == [1.0, 10 / 40, 0.0]


def test_a_live_show_that_finished_passes_b_and_an_aborted_one_does_not(tmp_path):
    ok = write_show(tmp_path, full())
    bad = write_show(tmp_path, full()[:2], status="aborted")
    assert acceptance.run([ok], directory=tmp_path)["pass_b_live_show"]
    assert not acceptance.run([bad], directory=tmp_path)["pass_b_live_show"]


def test_label_spread_flags_a_key_over_60_percent_of_presses_and_only_in_live_shows(tmp_path):
    over = [[3] * 5 for _ in range(5)] + [[2] * 5 for _ in range(3)]  # 25 W of 40 = 62.5%
    even = [[3] * 5 for _ in range(4)] + [[2, 2, 1, 1, 0]] * 4  # W is 20 of 40 = 50%
    skewed = [[3] * 5 for _ in range(6)] + [[2] * 5, [1] * 5]  # 75% W, but in a simulated room
    live_over, live_even = write_show(tmp_path, over), write_show(tmp_path, even, at=5000)
    sim_skewed = write_show(tmp_path, skewed, room="sim", at=9000)
    res = acceptance.run([live_over, live_even, sim_skewed], directory=tmp_path)
    assert res["flag_c_label_spread"] == [live_over]  # a sim's labels are not an operator's
    spread = res["shows"][0]["labels"]
    assert spread["top_key"] == "W" and spread["top_share_of_presses"] == pytest.approx(0.625)
    assert acceptance.label_spread([3] * 6 + [2] * 4)["flagged"] is False  # exactly 60% is not over it


def test_unanswered_windows_count_as_none_and_not_as_presses():
    spread = acceptance.label_spread([3, 3, None, None, 2])
    assert spread["counts"]["none"] == 2 and spread["presses"] == 3
    assert spread["share"]["none"] == pytest.approx(0.4)


def test_noise_estimate_recovers_a_known_sd():
    rng = np.random.default_rng(7)
    bits = [[int(x) for x in rng.integers(0, 4, 5)] for _ in range(400)]  # uniform on 0..3: sd of level/3 = sqrt(1.25)/3
    est = acceptance.noise(bits)
    truth = (1.25 ** 0.5 / 3) / 5 ** 0.5
    assert est["bit_sd"] == pytest.approx(truth, rel=0.03)
    assert est["ci95"][0] < truth < est["ci95"][1]
    assert est["n_bits"] == 400 and est["df"] == 1600


def test_noise_skips_bits_with_too_few_labels_and_handles_a_flat_room():
    assert acceptance.noise([[1, None, None, None, 2]])["bit_sd"] is None
    flat = acceptance.noise([[2] * 5] * 8)
    assert flat["bit_sd"] == 0.0 and flat["df"] == 32


def test_rerun_exp18_when_the_estimate_is_more_than_half_away_from_the_sims_015():
    wide = acceptance.noise_verdict({"bit_sd": 0.30, "ci95": [0.27, 0.33]})
    narrow = acceptance.noise_verdict({"bit_sd": 0.05, "ci95": [0.04, 0.06]})
    near = acceptance.noise_verdict({"bit_sd": 0.16, "ci95": [0.15, 0.17]})
    assert wide["rerun_exp18"] and wide["decided"] and narrow["rerun_exp18"]
    assert not near["rerun_exp18"] and near["decided"]
    straddle = acceptance.noise_verdict({"bit_sd": 0.20, "ci95": [0.10, 0.31]})
    assert straddle["rerun_exp18"] is False and straddle["decided"] is False  # a call on few bits is marked a guess


def test_pooling_two_live_shows_equals_one_estimate_over_all_their_bits(tmp_path):
    rng = np.random.default_rng(3)
    a = [[int(x) for x in rng.integers(0, 4, 5)] for _ in range(8)]
    b = [[int(x) for x in rng.integers(0, 4, 5)] for _ in range(8)]
    ids = [write_show(tmp_path, a, at=1000), write_show(tmp_path, b, at=5000)]
    pooled = acceptance.run(ids, directory=tmp_path)["noise"]
    direct = acceptance.noise(a + b)
    assert pooled["bit_sd"] == pytest.approx(direct["bit_sd"]) and pooled["df"] == direct["df"]
    assert {"joke_sd", "scale", "ci95"} <= acceptance.noise(a).keys() == acceptance.noise([]).keys()  # one shape
    assert pooled["n_live_shows"] == 2


def test_audio_coverage_compares_the_wav_with_the_logged_span(tmp_path):
    wav = tmp_path / "rec.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000 * 50)  # 50 s
    sid = write_show(tmp_path, full(), audio={"consent": True, "path": "rec.wav", "start_at": 1000.0})
    s = acceptance.show_report(sid, tmp_path)
    assert s["audio"]["wav_s"] == 50.0 and s["audio"]["coverage"] == pytest.approx(50 / s["audio"]["log_span_s"])
    wav.unlink()
    assert acceptance.show_report(sid, tmp_path)["audio"]["coverage"] is None  # a missing file is reported, not a crash


def test_a_rated_show_scores_the_same_apart_from_its_rating(tmp_path):
    plain = acceptance.show_report(write_show(tmp_path, full()), tmp_path)
    rated = acceptance.show_report(write_show(tmp_path, full(), rating=5), tmp_path)
    drop = {"session", "rating"}
    assert rated["rating"] == 5 and plain["rating"] is None
    assert {k: v for k, v in plain.items() if k not in drop} == {k: v for k, v in rated.items() if k not in drop}


def test_noise_follows_the_reward_rule_for_which_bits_count(tmp_path):
    from app.reward import reward_v1
    bits = [[1, 2, None, None, None], [1, 2, 3, None, None]]
    assert acceptance.noise(bits)["n_bits"] == sum(reward_v1(b) is not None for b in bits) == 1


def test_a_plain_run_saves_nothing_and_record_is_the_only_writer(tmp_path, monkeypatch):
    from sim import cache
    monkeypatch.setattr(cache, "METRICS", tmp_path / "metrics")
    sid = write_show(tmp_path, full())
    assert acceptance.run([sid], directory=tmp_path)["sessions"] == [sid]
    assert acceptance.stored() is None
    first = acceptance.record([sid], tmp_path)
    assert acceptance.stored() == first
    stored = (tmp_path / "metrics" / "acceptance.json").read_text()
    assert str(tmp_path) not in stored and "hand-checked" not in stored
    assert f'"sessions": ["{sid}"]' in stored  # the notebook can say which shows the evidence is
    other = write_show(tmp_path, full(), at=9000)
    acceptance.run([other], directory=tmp_path)  # ad-hoc scoring afterwards leaves the record alone
    assert acceptance.stored() == first


def test_the_cache_key_changes_when_the_recording_does(tmp_path):
    audio = {"consent": True, "path": "rec.wav", "start_at": 1000.0}
    sid = write_show(tmp_path, full(), audio=audio)
    wav = tmp_path / "rec.wav"
    wav.write_bytes(b"RIFF" + bytes(40))
    before = acceptance.digest([sid], tmp_path)
    assert acceptance.digest([sid], tmp_path) == before
    wav.write_bytes(b"RIFF" + bytes(80))
    assert acceptance.digest([sid], tmp_path) != before
    wav.unlink()
    assert acceptance.digest([sid], tmp_path) not in (before,)  # a missing recording is not the same key either


def test_a_pause_is_reported_with_its_seconds_and_the_jokes_it_cost(tmp_path):
    # joke (0, 1) plays [10, 12]; (0, 2) starts at 20, exactly when the pause ends, so it is untouched
    sid = write_show(tmp_path, full(), pauses=[(11.0, 20.0)])
    s = acceptance.show_report(sid, tmp_path)
    assert s["pauses"] == {"count": 1, "paused_s": 9.0, "jokes_dropped": 1}
    assert s["labels"]["counts"]["K"] == 40 and s["labels"]["counts"]["none"] == 0  # key bias is the operator's presses
    assert s["noise_stats"]["n_bits"] == 8  # the reward side still drops the label: bit 0 has 4 of 5
    assert acceptance.show_report(write_show(tmp_path, full()), tmp_path)["pauses"] ==         {"count": 0, "paused_s": 0.0, "jokes_dropped": 0}


def test_a_show_that_ended_paused_runs_the_pause_to_the_end_of_the_show(tmp_path):
    s = acceptance.show_report(write_show(tmp_path, full(), pauses=[(200.0, None)], status="aborted"), tmp_path)
    assert s["pauses"]["count"] == 1 and s["pauses"]["jokes_dropped"] == 30  # bits 2 to 7
    assert s["pauses"]["paused_s"] > 700


def test_paused_and_aborted_ratings_are_listed_but_not_in_the_reported_rating(tmp_path):
    ids = [write_show(tmp_path, full(), rating=4, at=1000),
           write_show(tmp_path, full(), rating=5, at=2000),
           write_show(tmp_path, full(), rating=1, pauses=[(11.0, 12.0)], at=3000),
           write_show(tmp_path, full()[:3], rating=2, status="aborted", at=4000)]
    res = acceptance.run(ids, directory=tmp_path)
    r = res["ratings"]
    assert (r["n"], r["mean"], r["values"]) == (2, 4.5, [4, 5])
    assert [(x["rating"], x["reason"]) for x in r["excluded"]] == [(1, "paused"), (2, "aborted")]
    assert [s["rating"] for s in res["shows"]] == [4, 5, 1, 2]  # still listed
    assert "(not counted)" in acceptance.table(res) and "not counted (paused)" in acceptance.verdict(res)


def test_a_paused_show_is_not_marked_as_missing_audio(tmp_path):
    # The recording keeps running through a pause, so the WAV spans it and the expected span is not reduced.
    sid = write_show(tmp_path, full(), pauses=[(11.0, 20.0)],
                     audio={"consent": True, "path": "rec.wav", "start_at": 1000.0})
    span = acceptance.show_report(sid, tmp_path)["audio"]["log_span_s"]
    with wave.open(str(tmp_path / "rec.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\0\0" * 16000 * int(span))
    assert acceptance.show_report(sid, tmp_path)["audio"]["coverage"] == pytest.approx(1.0, abs=0.05)


def test_a_pause_row_after_show_end_is_not_part_of_the_show(tmp_path):
    sid = write_show(tmp_path, full())
    path = tmp_path / f"{sid}.jsonl"
    rows = path.read_text(encoding="utf-8").splitlines()
    last = json.loads(rows[-1])
    assert last["event"] == "show_end"
    stray = {**last, "event": "pause", "paused_at": 11.0, "at": last["at"] + 5}
    stray.pop("status")
    path.write_text("\n".join(rows + [json.dumps(stray)]) + "\n", encoding="utf-8")
    assert acceptance.show_report(sid, tmp_path)["pauses"] == {"count": 0, "paused_s": 0.0, "jokes_dropped": 0}
