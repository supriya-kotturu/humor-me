"""Session log schema v1: the writer rejects bad rows, the reader survives a crash but not corruption."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app.log import EVENTS, NESTED, SessionLog, read_session  # noqa: E402

VALID = {
    "show_start": dict(room="sim", context={"venue": "bar", "energy": "high", "age": "30s"},
                       policy={"name": "linucb", "epsilon": 0.2, "alpha": 1.0, "prior": "uniform"},
                       generator_id="bank", selector_id="uniform", git_sha="abc123", operator="op",
                       audio={"consent": False, "path": None}),
    "bit_decision": dict(bit=0, obs={"t": 0, "n_bits": 8, "context": {"venue": "bar", "energy": "high", "age": "mixed"}, "last_mode": None},
                         scores={"a": 0.1}, greedy="a", action="b", probs={"a": 0.8, "b": 0.2}),
    "candidates": dict(bit=0, joke=0, prompt_id="p1", prompt="text", generator_id="bank",
                       items=[{"text": "x", "verdict": "pass", "reason": None}]),
    "joke_played": dict(bit=0, joke=0, chosen=0, pick_prob=0.5, text="x", tts_seconds=0.1, audio_seconds=1.0,
                        audio_start_at=1.0, audio_end_at=2.0),
    "reaction": dict(bit=0, joke=0, source="button", level=2, key="K", latency_s=0.4),
    "gap": dict(bit=0, joke=0, planned_hold_s=1.5, actual_gap_s=1.6),
    "bit_end": dict(bit=0, n_labels=5, reward_version=1, reward=0.5),
    "rating": dict(value=4),
    "show_end": dict(status="finished"),
}
NON_ASCII = "café ☃"


def test_every_event_in_the_schema_has_a_valid_example():
    assert set(VALID) == set(EVENTS)


def test_every_event_round_trips_with_envelope_and_unicode(tmp_path):
    log = SessionLog(tmp_path)
    for event, fields in VALID.items():
        log.write(event, **fields)
    log.write("joke_played", **{**VALID["joke_played"], "text": NON_ASCII})
    rows = read_session(log.path)
    assert [r["event"] for r in rows[:-1]] == list(VALID)
    for row in rows:
        assert row["schema"] == 1 and row["session"] == log.session_id and isinstance(row["at"], float)
    for row in rows[:-1]:
        assert {k: v for k, v in row.items() if k in VALID[row["event"]]} == VALID[row["event"]]
    assert rows[-1]["text"] == NON_ASCII


def test_unknown_event_is_rejected_and_nothing_is_written(tmp_path):
    log = SessionLog(tmp_path)
    with pytest.raises(ValueError, match="unknown event"):
        log.write("laugh", level=3)
    assert not log.path.exists()


@pytest.mark.parametrize("event", list(VALID))
def test_each_required_field_is_enforced(tmp_path, event):
    for field in VALID[event]:
        log = SessionLog(tmp_path)
        fields = {k: v for k, v in VALID[event].items() if k != field}
        with pytest.raises(ValueError, match=field):
            log.write(event, **fields)
        assert not log.path.exists()


@pytest.mark.parametrize("event,field", list(NESTED))
def test_each_required_nested_key_is_enforced(tmp_path, event, field):
    for key in VALID[event][field]:
        nested = {k: v for k, v in VALID[event][field].items() if k != key}
        log = SessionLog(tmp_path)
        with pytest.raises(ValueError, match=key):
            log.write(event, **{**VALID[event], field: nested})
        assert not log.path.exists()


def test_each_candidate_item_key_is_enforced(tmp_path):
    for key in ("text", "verdict", "reason"):
        item = {k: v for k, v in VALID["candidates"]["items"][0].items() if k != key}
        with pytest.raises(ValueError, match=key):
            SessionLog(tmp_path).write("candidates", **{**VALID["candidates"], "items": [item]})


def test_null_is_allowed_where_the_plan_says_null(tmp_path):
    log = SessionLog(tmp_path)
    log.write("reaction", **{**VALID["reaction"], "level": None, "key": None, "latency_s": None})
    assert read_session(log.path)[0]["level"] is None


@pytest.mark.parametrize("event,change", [
    ("show_start", {"room": "studio"}),
    ("reaction", {"source": "mic"}),
    ("reaction", {"level": 4}),
    ("rating", {"value": 0}),
    ("rating", {"value": 6}),
    ("rating", {"value": True}),
    ("rating", {"value": 4.0}),
    ("reaction", {"level": True, "key": None}),
    ("reaction", {"level": 3, "key": "D"}),
    ("reaction", {"level": 2, "key": "Q"}),
    ("reaction", {"level": 2, "key": None}),  # a button press with no key is a timeout
    ("show_end", {"status": "crashed"}),
])
def test_values_outside_the_schema_are_rejected(tmp_path, event, change):
    with pytest.raises(ValueError):
        SessionLog(tmp_path).write(event, **{**VALID[event], **change})


def _two_rows(tmp_path):
    log = SessionLog(tmp_path)
    log.write("rating", value=3)
    log.write("show_end", status="finished")
    return log.path


def test_a_real_partial_write_of_the_last_row_is_ignored(tmp_path):
    path = _two_rows(tmp_path)
    full = json.dumps({"schema": 1, "session": "s", "at": 1.0, "event": "rating", "value": 5}) + "\n"
    with path.open("ab") as f:  # a crash mid-write: the first bytes of a row reach the disk
        f.write(full.encode()[: len(full) // 2])
    assert [r["event"] for r in read_session(path)] == ["rating", "show_end"]


def test_a_torn_row_in_the_middle_is_corruption(tmp_path):
    path = _two_rows(tmp_path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text(lines[0][:20] + "\n" + lines[1] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_session(path)


def test_a_complete_row_without_final_newline_is_kept(tmp_path):
    path = _two_rows(tmp_path)
    path.write_text(path.read_text(encoding="utf-8").rstrip("\n"), encoding="utf-8")
    assert len(read_session(path)) == 2


def test_reader_rejects_other_schemas_and_invalid_rows(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text(json.dumps({"schema": 2, "session": "s", "at": 1.0, "event": "rating", "value": 3}) + "\n")
    with pytest.raises(ValueError, match="schema"):
        read_session(path)
    path.write_text(json.dumps({"schema": 1, "session": "s", "at": 1.0, "event": "rating"}) + "\n")
    with pytest.raises(ValueError, match="value"):
        read_session(path)


def test_reaction_key_table_and_nulls(tmp_path):
    log = SessionLog(tmp_path)
    for key, level in {"W": 3, "K": 2, "Space": 1, "D": 0}.items():
        log.write("reaction", **{**VALID["reaction"], "key": key, "level": level})
    log.write("reaction", **{**VALID["reaction"], "key": None, "level": None})  # timeout
    log.write("reaction", **{**VALID["reaction"], "source": "sim", "key": None, "level": 2})
    assert len(read_session(log.path)) == 6


def test_unicode_line_separators_in_text_do_not_look_like_corruption(tmp_path):
    log = SessionLog(tmp_path)
    text = "a\u2028b\u2029c\u0085d\x0ce"
    log.write("joke_played", **{**VALID["joke_played"], "text": text})
    log.write("show_end", status="finished")
    rows = read_session(log.path)
    assert [r["event"] for r in rows] == ["joke_played", "show_end"] and rows[0]["text"] == text


def test_a_tear_inside_a_multibyte_character_on_the_last_row_is_ignored(tmp_path):
    path = _two_rows(tmp_path)
    row = {"schema": 1, "session": "s", "at": 1.0, "event": "rating", "value": 5, "x": "\u2603"}
    full = (json.dumps(row, ensure_ascii=False) + "\n").encode()
    cut = full.index("\u2603".encode()) + 1  # one byte into the 3-byte snowman
    with path.open("ab") as f:
        f.write(full[:cut])
    assert [r["event"] for r in read_session(path)] == ["rating", "show_end"]


def test_a_tear_inside_a_multibyte_character_in_the_middle_raises(tmp_path):
    path = _two_rows(tmp_path)
    lines = path.read_bytes().split(b"\n")
    path.write_bytes(lines[0][:10] + "\u2603".encode()[:1] + b"\n" + lines[1] + b"\n")
    with pytest.raises(ValueError, match="line 1"):
        read_session(path)


@pytest.mark.parametrize("text", ["[1, 2]", "3", "null", '{"schema": 1, "session": "s", "at": 1.0, "event": 5}'])
def test_rows_that_are_not_schema_objects_raise_value_error(tmp_path, text):
    path = tmp_path / "x.jsonl"
    path.write_text(text + "\n" + text + "\n")
    with pytest.raises(ValueError):
        read_session(path)


def test_demo_rows_are_not_schema_v1(tmp_path):
    log = SessionLog(tmp_path)
    log.write_demo("line", index=0)
    with pytest.raises(ValueError, match="schema"):
        read_session(log.path)


def test_log_module_does_not_import_fastapi():
    source = (pathlib.Path(__file__).resolve().parents[1] / "backend" / "app" / "log.py").read_text(encoding="utf-8")
    assert "fastapi" not in source
