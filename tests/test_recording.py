"""Consent-gated raw audio: the file exists only after a consented first frame and stays readable if the connection
drops; the logged start time is real. No microphone or browser needed."""
import importlib
import json
import pathlib
import subprocess
import types
import sys
import wave

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "tests"))
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app import llm  # noqa: E402
from app.log import SessionLog, read_session  # noqa: E402
from app.protocol import ProtocolError, parse  # noqa: E402
from app.recording import RATE, Recorder  # noqa: E402
from sim import evaluate_clock as ev  # noqa: E402
from test_show_loop import CTX, FAST, FakeTTS, pump  # noqa: E402

main = importlib.import_module("app.main")
CHUNK = np.arange(4096, dtype="<i2").tobytes()
POLICY = {"name": "x", "epsilon": 0, "alpha": 1, "prior": "u"}


def start(consent, room="sim"):
    return {"type": "start", "context": CTX, "room": room, "operator": "t", "consent": consent}


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "get_tts", FakeTTS)
    monkeypatch.setattr(main, "sessions_dir", tmp_path)
    monkeypatch.setattr(main, "reaction_config", FAST)
    return tmp_path


def until(ws, kind):
    while True:
        frame = ws.receive()
        if frame.get("bytes") is not None:
            ws.send_json({"type": "played"})
            continue
        msg = json.loads(frame["text"])
        if msg["type"] == kind:
            return msg


def frames(path):
    with wave.open(str(path), "rb") as w:
        return w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()


def test_no_file_exists_before_a_frame_and_the_wav_is_readable_without_close(tmp_path):
    rec = Recorder(tmp_path / "a.wav", clock=lambda: 100.0)
    assert not (tmp_path / "a.wav").exists()
    rec.write(CHUNK)
    rec.write(CHUNK)  # never closed: a crash here must leave this file
    assert frames(tmp_path / "a.wav") == (8192, RATE, 1, 2)
    assert (tmp_path / "a.wav").read_bytes()[44:48] == CHUNK[:4]
    assert rec.start_at == pytest.approx(100.0 - 4096 / RATE)


@pytest.mark.parametrize("bad", [b"", b"\x00", b"\x00" * (64 * 1024 + 2)])
def test_malformed_frames_are_refused_and_write_nothing(tmp_path, bad):
    rec = Recorder(tmp_path / "a.wav")
    with pytest.raises(ValueError):
        rec.write(bad)
    assert not (tmp_path / "a.wav").exists()


def test_a_closed_recorder_drops_late_audio_without_complaint(tmp_path):
    rec = Recorder(tmp_path / "a.wav")
    rec.write(CHUNK)
    rec.close()
    rec.write(CHUNK)
    assert frames(tmp_path / "a.wav")[0] == 4096


def test_coverage_reports_audio_missing_against_the_wall_clock(tmp_path, caplog):
    now = [100.0]
    rec = Recorder(tmp_path / "c.wav", clock=lambda: now[0])
    frame = b"\x00\x00" * RATE  # one second of samples per frame
    rec.write(frame)  # first sample at 99.0
    for _ in range(7):
        now[0] += 1.0
        rec.write(frame)
    now[0] += 2.0  # two seconds pass with no samples: the gap a stalled Bluetooth clock leaves
    assert rec.coverage() == pytest.approx(8 / 10)  # 8 s of samples between 99.0 and 109.0
    with caplog.at_level("WARNING", logger="app.recording"):
        rec.close()
    assert "20% missing" in caplog.text


def test_full_coverage_logs_nothing(tmp_path, caplog):
    now = [100.0]
    rec = Recorder(tmp_path / "d.wav", clock=lambda: now[0])
    for _ in range(5):
        rec.write(b"\x00\x00" * RATE)
        now[0] += 1.0
    now[0] -= 1.0
    with caplog.at_level("WARNING", logger="app.recording"):
        rec.close()
    assert rec.coverage() == pytest.approx(1.0) and "missing" not in caplog.text


def test_a_binary_frame_is_never_a_control_message():
    with pytest.raises(ProtocolError):
        parse({"bytes": json.dumps({"type": "played"}).encode(), "text": None})


def test_audio_without_consent_is_refused_and_no_file_appears(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_bytes(CHUNK)  # before any show
        assert "refused" in until(ws, "error")["message"]
        ws.send_json(start(False))
        ws.send_bytes(CHUNK)  # during a show that was never agreed
        assert "refused" in until(ws, "error")["message"]
        pump(ws)
    assert list(server.glob("*.wav")) == []
    assert read_session(next(server.glob("*.jsonl")))[0]["audio"] == {"consent": False, "path": None}


def test_consented_show_records_and_logs_when_the_audio_starts(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start(True))
        until(ws, "record")
        ws.send_bytes(CHUNK)
        ws.send_bytes(CHUNK)
        pump(ws)
    (wav,) = server.glob("*.wav")
    rows = read_session(next(server.glob("*.jsonl")))
    audio = rows[0]["audio"]
    assert audio["consent"] is True and audio["path"] == wav.name and wav.stem == rows[0]["session"]
    # sample 0 predates the first frame's arrival, which predates show_start
    assert audio["start_at"] <= rows[0]["at"] and rows[0]["at"] - audio["start_at"] < 10
    assert frames(wav) == (8192, RATE, 1, 2)


def test_a_dropped_connection_leaves_a_readable_wav_with_every_sample(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start(True, room="live"))  # a live room waits for keys, so the show is still running
        until(ws, "record")
        for _ in range(3):
            ws.send_bytes(CHUNK)
        until(ws, "line")
    (wav,) = server.glob("*.wav")
    assert frames(wav)[0] == 3 * 4096
    assert read_session(next(server.glob("*.jsonl")))[-1]["status"] == "aborted"


def next_error(ws):
    """The first error the server sends after a bad message of ours. An error for earlier frames would come first, so
    getting ours means those were dropped silently; the show's own lines are skipped over."""
    ws.send_json({"type": "nonsense"})
    while True:
        frame = ws.receive()
        if frame.get("bytes") is not None:
            ws.send_json({"type": "played"})
            continue
        msg = json.loads(frame["text"])
        if msg["type"] == "error":
            return msg


def test_a_microphone_that_failed_runs_the_show_unrecorded_and_carries_the_reason(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start(True))
        until(ws, "record")
        ws.send_json({"type": "audio_failed", "reason": "NotAllowedError: denied"})
        seen = pump(ws)
    assert [m["reason"] for m in seen if m["type"] == "record_stop"] == ["NotAllowedError: denied"]
    assert list(server.glob("*.wav")) == []
    assert read_session(next(server.glob("*.jsonl")))[0]["audio"] == {"consent": True, "path": None, "start_at": None}


def test_a_mic_that_opens_after_the_server_gave_up_is_told_once_and_its_frames_dropped_silently(server, monkeypatch):
    monkeypatch.setattr(main, "AUDIO_WAIT_S", 0.05)
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start(True))
        until(ws, "record")
        assert until(ws, "record_stop")["reason"]
        for _ in range(3):
            ws.send_bytes(CHUNK)  # the late microphone
        assert next_error(ws)["message"].startswith("unknown message type")
        seen = pump(ws)
    assert [m["type"] for m in seen if m["type"] in ("error", "record_stop")] == []
    assert list(server.glob("*.wav")) == []
    assert read_session(next(server.glob("*.jsonl")))[0]["audio"]["path"] is None


def test_frames_in_flight_when_the_show_ends_are_dropped_silently(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start(True))
        until(ws, "record")
        ws.send_bytes(CHUNK)
        seen = pump(ws)
        ws.send_bytes(CHUNK)
        assert next_error(ws)["message"].startswith("unknown message type")
    assert [m["type"] for m in seen].count("record_stop") == 1
    (wav,) = server.glob("*.wav")
    assert frames(wav)[0] == 4096


def test_a_disk_error_stops_the_recording_but_not_the_show(server, monkeypatch):
    class FailingDisk(Recorder):
        def _write(self, pcm):
            if self.samples:
                raise OSError("disk full")
            super()._write(pcm)

    monkeypatch.setattr(main, "Recorder", FailingDisk)
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start(True))
        until(ws, "record")
        for _ in range(3):
            ws.send_bytes(CHUNK)
        seen = pump(ws)
    stops = [m for m in seen if m["type"] == "record_stop"]
    assert len(stops) == 1 and "disk" in stops[0]["reason"] and not any(m["type"] == "error" for m in seen)
    assert seen[-1] == {"type": "show_end", "status": "finished", "ratable": True}
    assert frames(next(server.glob("*.wav")))[0] == 4096


def test_a_recording_must_have_a_start_time_in_the_log(tmp_path):
    log = SessionLog(tmp_path)
    fields = dict(room="sim", context=CTX, policy=POLICY, generator_id="g", selector_id="s", git_sha="x", operator="t")
    with pytest.raises(ValueError, match="start_at"):
        log.write("show_start", audio={"consent": True, "path": "a.wav"}, **fields)
    log.write("show_start", audio={"consent": True, "path": "a.wav", "start_at": 1.5}, **fields)


@pytest.mark.parametrize("name", ["abc123.wav", "abc123.jsonl"])
def test_sessions_are_gitignored(name):
    path = ROOT / "data" / "sessions" / name
    assert subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=ROOT).returncode == 0


def test_llm_without_its_settings_says_what_to_set(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.delenv("GEMINI_BASE_URL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "m")
    with pytest.raises(RuntimeError, match=r"GEMINI_BASE_URL.*--env-file"):
        llm.from_env()


def clicked_show(tmp_path, true_times, start_at, rate=16000, seconds=60, speech_at=(), amp=0.5, floor=0.002, burst=0.2):
    """Beeps as `play` makes them over a quiet room; `speech_at` adds half-second broadband bursts starting just
    before those times, standing in for the voice and laughter."""
    rng = np.random.default_rng(0)
    audio = rng.normal(0, floor, rate * seconds)
    tone = amp * np.sin(2 * np.pi * ev.BEEP_HZ * np.arange(int(rate * ev.BEEP_MS / 1000)) / rate)
    for t in true_times:
        i = int((t - start_at) * rate)
        audio[i : i + tone.size] += tone
    for t in speech_at:
        i = int((t - start_at - 0.1) * rate)
        audio[i : i + rate // 2] += rng.normal(0, burst, rate // 2)
    rec = Recorder(tmp_path / "s.wav", clock=lambda: start_at + 4096 / RATE)
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    for i in range(0, len(pcm), 8192):
        rec.write(pcm[i : i + 8192])
    log = SessionLog(tmp_path)
    log.write("show_start", room="live", context=CTX, policy=POLICY, generator_id="g", selector_id="s",
              git_sha="x", operator="t", audio={"consent": True, "path": "s.wav", "start_at": start_at})
    return log.path, tmp_path / "s.wav"


START = 1_000_000.0
TRUE = [START + 3 + 5 * k for k in range(10)]


def test_click_offsets_are_recovered_at_known_times(tmp_path):
    log_path, wav = clicked_show(tmp_path, TRUE, START)
    # the logged times are 30 ms earlier than the beeps really sound
    (tmp_path / "clicks.json").write_text(json.dumps([t - 0.03 for t in TRUE]))
    out = ev.run(str(log_path), str(tmp_path / "clicks.json"), str(wav))
    assert out["n_matched"] == 10 and out["n_missed"] == 0
    assert out["median_offset_s"] == pytest.approx(0.03, abs=0.002)
    assert out["p95_abs_offset_s"] == pytest.approx(0.03, abs=0.002)
    assert out["pass_spread"] and out["pass_drift"] and out["pass_matched"] and out["pass_old_p95"]
    assert abs(out["drift_over_show_s"]) < 0.002
    assert "| 10/10 | 0 |" in ev.table(out)


def test_speech_over_some_beeps_does_not_move_the_offsets(tmp_path):
    log_path, wav = clicked_show(tmp_path, TRUE, START, speech_at=[TRUE[2], TRUE[3], TRUE[7]])
    (tmp_path / "clicks.json").write_text(json.dumps([t - 0.03 for t in TRUE]))
    out = ev.run(str(log_path), str(tmp_path / "clicks.json"), str(wav))
    assert out["n_matched"] == 10
    assert out["median_offset_s"] == pytest.approx(0.03, abs=0.004)
    assert out["p95_abs_offset_s"] == pytest.approx(0.03, abs=0.004)


def test_speech_alone_is_not_a_beep():
    speech = np.random.default_rng(1).normal(0, 0.2, 16000 * 5)
    assert ev.detect_clicks(speech, 16000) == []


def test_quiet_beeps_over_louder_broadband_noise_are_all_found(tmp_path):
    # a laptop mic records the beeps at ~0.003; bursts of speech-like noise three times that loud sit on some of them
    log_path, wav = clicked_show(tmp_path, TRUE, START, speech_at=[TRUE[1], TRUE[4], TRUE[5]], amp=0.003, floor=0.0003,
                                 burst=0.01)
    (tmp_path / "clicks.json").write_text(json.dumps([t - 0.03 for t in TRUE]))
    out = ev.run(str(log_path), str(tmp_path / "clicks.json"), str(wav))
    assert out["n_matched"] == 10 and out["n_missed"] == 0
    assert out["median_offset_s"] == pytest.approx(0.03, abs=0.004) and out["max_spread_s"] < 0.005
    assert out["peak_over_noise_min"] > ev.NOISE_FACTOR


def test_beeps_outside_the_recording_are_counted_apart_and_are_not_misses(tmp_path):
    log_path, wav = clicked_show(tmp_path, TRUE, START)
    before, after = [START - 20, START - 1], [START + 60 - 0.2, START + 70]  # the third could not be heard in full
    (tmp_path / "clicks.json").write_text(json.dumps(before + [t - 0.03 for t in TRUE] + after))
    out = ev.run(str(log_path), str(tmp_path / "clicks.json"), str(wav))
    assert out["n_logged"] == 10 and out["n_outside"] == 4 and out["n_matched"] == 10 and out["n_missed"] == 0
    assert out["pass_matched"]


def test_a_constant_offset_with_jitter_passes_the_spread_bar_and_misses_the_old_one(tmp_path):
    jitter = np.random.default_rng(3).uniform(-0.02, 0.02, len(TRUE))  # 20 ms either way
    log_path, wav = clicked_show(tmp_path, TRUE, START)
    (tmp_path / "c.json").write_text(json.dumps([t - 0.24 + j for t, j in zip(TRUE, jitter)]))
    out = ev.run(str(log_path), str(tmp_path / "c.json"), str(wav))
    assert out["median_offset_s"] == pytest.approx(0.24, abs=0.01)
    assert out["pass_spread"] and out["pass_drift"] and out["pass_matched"]
    assert not out["pass_old_p95"] and out["p95_abs_offset_s"] > 0.2


def test_sixty_milliseconds_of_drift_fails_the_drift_bar(tmp_path):
    log_path, wav = clicked_show(tmp_path, TRUE, START)
    drifting = [t - 0.24 - 0.06 * (t - TRUE[0]) / (TRUE[-1] - TRUE[0]) for t in TRUE]
    (tmp_path / "c.json").write_text(json.dumps(drifting))
    out = ev.run(str(log_path), str(tmp_path / "c.json"), str(wav))
    assert out["drift_over_show_s"] == pytest.approx(0.06, abs=0.003)
    assert not out["pass_drift"] and out["pass_spread"]  # 60 ms of drift is spread inside 100 ms, so only the drift bar sees it


def test_a_drifting_clock_fails_the_drift_bar_and_a_late_one_the_old_p95_bar(tmp_path):
    log_path, wav = clicked_show(tmp_path, TRUE, START)
    drifting = [t - 0.2 * (t - TRUE[0]) / (TRUE[-1] - TRUE[0]) for t in TRUE]  # 200 ms across the show
    (tmp_path / "c.json").write_text(json.dumps(drifting))
    out = ev.run(str(log_path), str(tmp_path / "c.json"), str(wav))
    assert out["drift_over_show_s"] == pytest.approx(0.2, abs=0.005)
    assert not out["pass_drift"] and not out["pass_old_p95"]


def test_check_on_an_unrecorded_show_says_so_instead_of_crashing(tmp_path, monkeypatch):
    log = SessionLog(tmp_path)
    log.write("show_start", room="sim", context=CTX, policy=POLICY, generator_id="g", selector_id="s", git_sha="x",
              operator="t", audio={"consent": False, "path": None})
    (tmp_path / "c.json").write_text("[1.0]")
    monkeypatch.setattr(sys, "argv", ["evaluate_clock", "check", str(log.path), str(tmp_path / "c.json")])
    with pytest.raises(SystemExit, match="not recorded"):
        ev.main()


def health(produced, elapsed=5.0, wall=5.0):
    return {"type": "audio_health", "produced_s": produced, "elapsed_s": elapsed, "gap_ms": 0, "wall_s": wall,
            "rate": 48000}


def test_audio_health_parses_and_rejects_bad_numbers():
    assert parse({"text": json.dumps(health(4.9))})["produced_s"] == 4.9
    for bad in ({**health(1), "produced_s": -1}, {**health(1), "rate": "48k"}, {**health(1), "gap_ms": True}):
        with pytest.raises(ProtocolError):
            parse({"text": json.dumps(bad)})
    missing = health(1)
    del missing["wall_s"]
    with pytest.raises(ProtocolError):
        parse({"text": json.dumps(missing)})


def test_the_server_warns_when_the_mic_skipped_time_or_frames_are_held_up(server, caplog):
    with caplog.at_level("WARNING"), TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(health(4.98))  # fine
        ws.send_json(health(3.25))  # the mic delivered 65% of capture time
        ws.send_json(health(5.0, elapsed=5.0, wall=8.0))  # capture time ran 62% of wall time
        assert next_error(ws)["message"].startswith("unknown message type")
    text = [r.getMessage() for r in caplog.records]
    assert sum("microphone delivered" in t for t in text) == 1 and any("65%" in t for t in text)
    assert sum("held up in the browser" in t for t in text) == 1


def test_note_health_stores_the_last_one():
    live = types.SimpleNamespace(health=None)
    main._note_health(live, health(5.0) | {"gap_ms": 7})
    assert live.health["gap_ms"] == 7


def test_audio_start_parses_and_rejects_nonsense():
    assert parse({"text": json.dumps({"type": "audio_start", "wall_ms": 1.7e12, "source": "capture"})})["wall_ms"] == 1.7e12
    for bad in (0, -5, "now", True, None):
        with pytest.raises(ProtocolError):
            parse({"text": json.dumps({"type": "audio_start", "wall_ms": bad})})


def test_the_capture_clock_start_replaces_the_arrival_estimate_when_it_is_plausible(tmp_path):
    rec = Recorder(tmp_path / "a.wav", clock=lambda: 100.0)
    rec.hint_start(99.6)  # captured 0.6 s ago, so 0.34 s before the arrival estimate
    rec.write(CHUNK)
    assert rec.start_at == 99.6
    far = Recorder(tmp_path / "b.wav", clock=lambda: 100.0)
    far.hint_start(5.0)  # a clock this far off is not the same machine's: keep the arrival estimate
    far.write(CHUNK)
    assert far.start_at == pytest.approx(100.0 - 4096 / RATE)


def test_the_hint_cannot_move_a_recording_that_already_started(tmp_path):
    rec = Recorder(tmp_path / "a.wav", clock=lambda: 100.0)
    rec.write(CHUNK)
    first = rec.start_at
    rec.hint_start(50.0)
    assert rec.start_at == first


def test_the_server_uses_the_browsers_capture_start_in_the_log(server):
    import time

    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start(True))
        until(ws, "record")
        captured = time.time() - 0.5
        ws.send_json({"type": "audio_start", "wall_ms": captured * 1000, "source": "capture"})
        ws.send_bytes(CHUNK)
        pump(ws)
    assert read_session(next(server.glob("*.jsonl")))[0]["audio"]["start_at"] == pytest.approx(captured, abs=1e-3)


def test_a_mic_that_fails_after_audio_began_keeps_the_file_and_the_show(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start(True, room="live"))
        until(ws, "record")
        ws.send_bytes(CHUNK)
        until(ws, "line")
        ws.send_json({"type": "audio_failed", "reason": "track ended"})
        assert until(ws, "record_stop")["reason"] == "track ended"
        ws.send_bytes(CHUNK)
        assert next_error(ws)["message"].startswith("unknown message type")
    assert frames(next(server.glob("*.wav")))[0] == 4096
