"""The show over the websocket, with a fake voice; no Kokoro, audio device or browser needed."""
import importlib
import json
import pathlib
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.log import read_session  # noqa: E402
from app.rooms import ReactionConfig  # noqa: E402
from app.tts import to_wav, wav_seconds  # noqa: E402
from sim.env import AudienceEnv  # noqa: E402

main = importlib.import_module("app.main")
CTX = {"venue": "bar", "energy": "normal", "age": "mixed"}
FAST = ReactionConfig(holds={3: 0.003, 2: 0.002, 1: 0.001, 0: 0.001, None: 0.0}, timeout_s=0.5)
# Written out here, not imported, so a wrong table in app code fails these tests.
LEVEL_OF = {"W": 3, "K": 2, "Space": 1, "D": 0}


class FakeTTS:
    def synthesize(self, text):
        return to_wav(np.zeros(2400))


@pytest.fixture
def server(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "get_tts", FakeTTS)
    monkeypatch.setattr(main, "sessions_dir", tmp_path)
    monkeypatch.setattr(main, "reaction_config", FAST)
    return tmp_path


def start(room, operator="t"):
    return {"type": "start", "context": CTX, "room": room, "operator": operator}


def pump(ws, on_open=None, on_closed=None, stop=lambda m: m["type"] == "show_end"):
    """Play the browser: acknowledge every audio blob, hand each window event to a callback, return what arrived."""
    seen = []
    while True:
        frame = ws.receive()
        if frame.get("bytes") is not None:
            assert frame["bytes"].startswith(b"RIFF")
            ws.send_json({"type": "played"})
            continue
        msg = json.loads(frame["text"])
        seen.append(msg)
        if msg["type"] == "reaction_window":
            callback = on_open if msg["state"] == "open" else on_closed
            if callback:
                callback(msg)
        if stop(msg):
            return seen


def only_log(tmp_path):
    (path,) = tmp_path.glob("*.jsonl")
    return read_session(path)


def reactions(rows):
    return [r for r in rows if r["event"] == "reaction"]


def press(ws, msg, key):
    ws.send_json({"type": "reaction", "bit": msg["bit"], "joke": msg["joke"], "key": key})


def test_wav_round_trip_reports_its_length():
    assert abs(wav_seconds(to_wav(np.zeros(12_000))) - 0.5) < 1e-9


def test_loud_audio_is_clipped_rather_than_wrapped_around():
    wav = to_wav(np.array([2.0, -2.0]))
    assert int.from_bytes(wav[-4:-2], "little", signed=True) == 32767
    assert int.from_bytes(wav[-2:], "little", signed=True) == -32767


def test_health_still_works():
    assert TestClient(main.app).get("/health").json() == {"status": "ok"}


def test_a_whole_simulated_show_logs_40_reactions(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        seen = pump(ws)
    rows = only_log(server)
    assert seen[-1] == {"type": "show_end", "status": "finished", "ratable": True}
    assert len(reactions(rows)) == 40 and {r["source"] for r in reactions(rows)} == {"sim"}
    assert rows[0]["room"] == "sim" and rows[0]["generator_id"] == "bank-v1" and rows[0]["operator"] == "t"
    assert isinstance(rows[0]["policy"]["seed"], int) and rows[0]["git_sha"]
    assert rows[-1]["status"] == "finished"
    keys = [m["key"] for m in seen if m["type"] == "sim_reaction"]
    assert keys == [r["key"] for r in reactions(rows)] and len(keys) == 40
    first = next(r for r in rows if r["event"] == "candidates")
    assert first["prompt_id"].startswith("bank-v1:") and first["prompt"] == "bank-v1.yaml"
    assert first["items"][0]["verdict"] == "pass"


def test_a_live_show_answered_with_keys_logs_the_frozen_levels(server):
    order = ["W", "K", "Space", "D"]
    answered = []

    def answer(msg):
        key = order[len(answered) % 4]
        answered.append((msg["bit"], msg["joke"]))
        press(ws, msg, key)

    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("live"))
        seen = pump(ws, on_open=answer)
    rows = only_log(server)
    got = reactions(rows)
    assert seen[-1]["status"] == "finished" and len(got) == 40
    assert [(r["key"], r["level"]) for r in got] == [(order[i % 4], LEVEL_OF[order[i % 4]]) for i in range(40)]
    assert {r["source"] for r in got} == {"button"} and all(r["latency_s"] >= 0 for r in got)
    assert answered == [(b, j) for b in range(8) for j in range(5)]
    assert [r["planned_hold_s"] for r in rows if r["event"] == "gap"][:4] == [0.003, 0.002, 0.001, 0.001]


def test_the_last_press_wins_during_the_hold_and_the_hold_restarts(server, monkeypatch):
    monkeypatch.setattr(main, "reaction_config", ReactionConfig(holds={3: 0.02, 1: 0.5, 2: 0.02, 0: 0.02}, timeout_s=2))
    fired = []

    def two_presses(msg):
        if not fired:
            fired.append(msg)
            press(ws, msg, "W")
            press(ws, msg, "Space")

    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("live"))
        # Stop at the first line after the window: joke 0's gap row is written by then.
        pump(ws, on_open=two_presses, stop=lambda m: m["type"] == "line" and bool(fired))
    rows = only_log(server)
    first = [r for r in reactions(rows) if r["joke"] == 0 and r["bit"] == 0]
    assert [(r["key"], r["level"]) for r in first] == [("W", 3), ("Space", 1)]
    (gap,) = [r for r in rows if r["event"] == "gap" and r["bit"] == 0 and r["joke"] == 0]
    assert gap["planned_hold_s"] == 0.5 and gap["actual_gap_s"] >= 0.45  # restarted at Space's hold, not W's


def test_a_late_or_misdirected_press_is_not_a_reaction_and_changes_no_reward(server):
    sent = []

    def wrong_then_right(msg):
        ws.send_json({"type": "reaction", "bit": msg["bit"], "joke": (msg["joke"] + 1) % 5, "key": "D"})
        press(ws, msg, "K")

    def late(msg):
        press(ws, msg, "D")  # the window has just closed
        sent.append(msg)

    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("live"))
        pump(ws, on_open=wrong_then_right, on_closed=late)
    rows = only_log(server)
    assert len(sent) == 40
    assert [r["level"] for r in reactions(rows)] == [2] * 40
    assert [r["reward"] for r in rows if r["event"] == "bit_end"] == pytest.approx([2 / 3] * 8)


def test_a_timeout_writes_a_null_reaction(server):
    seen_open = []

    def ignore(msg):
        seen_open.append(msg)

    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("live"))
        pump(ws, on_open=ignore, stop=lambda m: m["type"] == "reaction_window" and m["state"] == "closed")
        pump(ws, on_open=ignore, stop=lambda m: m["type"] == "line")
    rows = only_log(server)
    first = reactions(rows)[0]
    assert (first["source"], first["level"], first["key"]) == ("button", None, None)
    assert [r["planned_hold_s"] for r in rows if r["event"] == "gap"][0] == 0.0


def test_a_malformed_message_gets_an_error_and_the_show_goes_on(server):
    junk = [lambda: ws.send_text("not json"), lambda: ws.send_json({"type": "nope"}),
            lambda: ws.send_json({"type": "reaction", "bit": "0", "joke": 0, "key": "W"}),
            lambda: ws.send_json({"type": "reaction", "bit": 0, "joke": 0, "key": "X"}),
            lambda: ws.send_json({"type": "reaction", "bit": 0, "joke": 0, "key": ["W"]}),
            lambda: ws.send_json({"type": "reaction", "bit": 0, "joke": 0, "key": {"W": 1}}),
            lambda: ws.send_json([1, 2]), lambda: ws.send_bytes(b"\x00")]
    errors = []
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "start", "room": "live"})
        errors.append(ws.receive_json()["type"])
        ws.send_json(start("live"))
        opened = pump(ws, stop=lambda m: m["type"] == "reaction_window")[-1]
        for send in junk:
            send()
            errors.append(ws.receive_json()["type"])
        ws.send_json(start("live"))
        errors.append(ws.receive_json()["type"])  # a second show may not start beside the first
        press(ws, opened, "K")
        pump(ws, stop=lambda m: m["type"] == "line")
    assert errors == ["error"] * 10
    assert len(reactions(only_log(server))) == 1


def test_a_disconnect_writes_show_end_aborted(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("live"))
        pump(ws, stop=lambda m: m["type"] == "reaction_window")
    rows = only_log(server)
    assert rows[-1]["event"] == "show_end" and rows[-1]["status"] == "aborted"


def test_a_corrupt_session_log_fails_the_start_visibly(server):
    good = {"schema": 1, "session": "s", "at": 1.0, "event": "rating", "value": 5}
    (server / "bad.jsonl").write_text("\n".join([json.dumps(good), "{torn", json.dumps(good)]) + "\n")
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        first, second = ws.receive_json(), ws.receive_json()
    assert first["type"] == "error" and "corrupt" in first["message"]
    assert second == {"type": "show_end", "status": "aborted", "ratable": False}
    assert len(list(server.glob("*.jsonl"))) == 1  # no new show started


def test_the_mode_never_reaches_the_client(server):
    fields = {"line": {"type", "text"}, "reaction_window": {"type", "state", "bit", "joke"},
              "sim_reaction": {"type", "bit", "joke", "key"}, "show_end": {"type", "status", "ratable"}}
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        seen = pump(ws)
    assert {m["type"] for m in seen} == set(fields)
    for m in seen:
        assert set(m) == fields[m["type"]]
        rest = json.dumps({k: v for k, v in m.items() if k != "text"})
        assert not [mode for mode in AudienceEnv().modes if mode in rest]


def test_the_selector_and_the_director_get_different_seeds(server, monkeypatch):
    counter = iter(range(1000, 2000))
    monkeypatch.setattr(main.secrets, "randbits", lambda n: next(counter))
    seeds = {}
    real_director, real_selector = main.load_director, main.UniformSelector

    def spy_director(sessions, seed=None):
        seeds["director"] = seed
        return real_director(sessions, seed=seed)

    def spy_selector(seed=None):
        seeds["selector"] = seed
        return real_selector(seed)

    monkeypatch.setattr(main, "load_director", spy_director)
    monkeypatch.setattr(main, "UniformSelector", spy_selector)
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        pump(ws, stop=lambda m: m["type"] == "line")
    rows = only_log(server)
    assert seeds["director"] != seeds["selector"] and rows[0]["policy"]["seed"] == seeds["director"]


def rate(ws, value):
    ws.send_json({"type": "rating", "value": value})
    return ws.receive_json()


def test_a_finished_show_takes_one_rating_into_its_own_log_and_refuses_a_second(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        pump(ws)
        assert rate(ws, 4) == {"type": "rating_saved", "value": 4}
        second = rate(ws, 2)
    assert second["type"] == "error" and "already rated" in second["message"]
    rows = only_log(server)  # still one file, and still readable
    assert [r["event"] for r in rows[-2:]] == ["show_end", "rating"] and rows[-1]["value"] == 4
    assert [r["event"] for r in rows].count("rating") == 1


def test_a_rating_with_no_show_or_a_show_in_progress_is_refused(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        assert "no finished show" in rate(ws, 3)["message"]
        ws.send_json(start("live"))
        pump(ws, stop=lambda m: m["type"] == "reaction_window")
        assert "no finished show" in rate(ws, 3)["message"]
    assert "rating" not in {r["event"] for r in only_log(server)}


def test_an_aborted_show_can_be_rated_but_a_show_that_never_started_cannot(server, monkeypatch):
    class Boom:
        selector_id = "boom"

        def pick(self, texts):
            raise RuntimeError("selector down")

    monkeypatch.setattr(main, "UniformSelector", lambda seed: Boom())
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("live"))
        seen = pump(ws, stop=lambda m: m["type"] == "show_end")
        assert seen[-1]["status"] == "aborted" and seen[-1]["ratable"] is True
        assert rate(ws, 1) == {"type": "rating_saved", "value": 1}
    rows = only_log(server)
    assert [r["event"] for r in rows[-2:]] == ["show_end", "rating"] and rows[-2]["status"] == "aborted"


def test_a_show_that_died_before_show_start_cannot_be_rated(server, monkeypatch):
    monkeypatch.setattr(main, "make_stack", lambda room: 1 / 0)
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        end = pump(ws)[-1]
        assert end["status"] == "aborted" and end["ratable"] is False  # the UI keeps the failure notice, no prompt
        assert "no finished show" in rate(ws, 5)["message"]
    assert list(server.glob("*.jsonl")) == []  # no file: nothing a rating could corrupt


def test_a_rating_that_cannot_be_written_is_an_error_and_can_be_retried(server, monkeypatch):
    from app.log import SessionLog
    real = SessionLog.write
    failing = {"on": True}

    def flaky(self, event, **fields):
        if event == "rating" and failing["on"]:
            raise OSError("disk full")
        return real(self, event, **fields)

    monkeypatch.setattr(SessionLog, "write", flaky)
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        pump(ws)
        refused = rate(ws, 4)
        assert refused["type"] == "error" and refused["message"].startswith("couldn't save the rating") and "disk full" in refused["message"]
        failing["on"] = False
        assert rate(ws, 4) == {"type": "rating_saved", "value": 4}  # same socket, same show
    assert [r["event"] for r in only_log(server)][-2:] == ["show_end", "rating"]


def test_a_rating_on_an_unreadable_log_is_an_error_not_a_dropped_socket(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        pump(ws)
        (path,) = server.glob("*.jsonl")
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text(chr(10).join([lines[0], "{torn", *lines[1:]]) + chr(10), encoding="utf-8")
        assert rate(ws, 2)["message"].startswith("couldn't save the rating")
        ws.send_json({"type": "rating", "value": 2})
        assert ws.receive_json()["type"] == "error"  # still answering



def test_pause_and_resume_land_in_the_log_and_out_of_order_ones_are_refused(server):
    errors = []
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "pause"})  # no show yet
        errors.append(json.loads(ws.receive()["text"]))
        ws.send_json(start("sim"))
        sent = []

        def on_open(msg):
            if sent:
                return
            sent.append(1)
            for kind in ("resume", "pause", "pause", "resume"):  # refused, ok, refused, ok
                ws.send_json({"type": kind})

        seen = pump(ws, on_open=on_open)
    assert errors[0]["type"] == "error" and "no show" in errors[0]["message"]
    replies = [(m["type"], m.get("paused")) for m in seen if m["type"] in ("error", "pause_state")]
    assert replies == [("error", None), ("pause_state", True), ("error", None), ("pause_state", False)]
    assert seen[-1]["status"] == "finished"
    rows = only_log(server)
    marks = [r for r in rows if r["event"] in ("pause", "resume")]
    assert [r["event"] for r in marks] == ["pause", "resume"]
    assert all(r["about"] == "pause" for r in seen if r["type"] == "error")
    i = rows.index(marks[0])  # stamped on the show clock just before its row's own `at`, and after the row before it
    assert rows[i - 1]["at"] <= marks[0]["paused_at"] <= marks[0]["at"]
    assert marks[0]["at"] <= marks[1]["resumed_at"] <= marks[1]["at"]


def test_a_show_closed_while_paused_leaves_an_unmatched_pause_before_show_end(server):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("live"))
        pump(ws, stop=lambda m: m["type"] == "reaction_window")
        ws.send_json({"type": "pause"})
        assert json.loads(ws.receive()["text"])["type"] == "pause_state"
    rows = only_log(server)
    assert [r["event"] for r in rows[-2:]] == ["pause", "show_end"] and rows[-1]["status"] == "aborted"


def test_the_show_holds_while_paused_and_the_next_joke_keeps_its_label(server, monkeypatch):
    monkeypatch.setattr(main, "reaction_config", ReactionConfig(holds={3: 0.05, 2: 0.05, 1: 0.05, 0: 0.05, None: 0.0},
                                                                 timeout_s=0.3))
    first = []

    def on_open(msg):
        if not first:
            first.append(msg)
            ws.send_json({"type": "pause"})
            assert json.loads(ws.receive()["text"])["type"] == "pause_state"
            time.sleep(0.8)  # past the window and the hold: the show must wait, not run on
            press(ws, msg, "W")  # a key pressed while paused still counts
            ws.send_json({"type": "resume"})
            assert json.loads(ws.receive()["text"])["type"] == "pause_state"
        else:
            press(ws, msg, "K")

    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("live"))
        seen = pump(ws, on_open=on_open, stop=lambda m: m["type"] == "reaction_window" and m["joke"] == 2)
    rows = only_log(server)
    (resume,) = [r for r in rows if r["event"] == "resume"]
    played = [r for r in rows if r["event"] == "joke_played"]
    assert [r["key"] for r in reactions(rows)][:2] == ["W", "K"]  # the first window did not time out
    assert played[1]["audio_start_at"] >= resume["resumed_at"]  # the next line waited for the resume
    from app.director import joke_levels
    assert joke_levels(rows, 0)[:2] == [3, 2]


def test_a_pause_after_the_show_ended_is_refused_and_not_logged(server, monkeypatch):
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        pump(ws)
        ws.send_json({"type": "pause"})
        refusal = json.loads(ws.receive()["text"])
    assert refusal["type"] == "error" and refusal["about"] == "pause"
    assert [r["event"] for r in only_log(server)][-1] == "show_end"


def test_a_pause_is_refused_once_run_show_has_raised(server, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(main, "run_show", boom)
    with TestClient(main.app).websocket_connect("/ws") as ws:
        ws.send_json(start("sim"))
        pump(ws)
        ws.send_json({"type": "pause"})
        refusal = json.loads(ws.receive()["text"])
    assert refusal["type"] == "error" and refusal["about"] == "pause"
