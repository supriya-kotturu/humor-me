"""The Explorer side of the Exp 24 judge: the tolerant ranking parser and the resumable, checkpointed answering loop, with a
fake generator. The model itself only runs on the cluster."""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from explorer import judge_generators as ej  # noqa: E402


@pytest.mark.parametrize("text, want", [
    ('{"ranking": "BAC"}', "BAC"),
    ('Sure! Here it is: {"ranking": "b, a, c"} hope that helps', "BAC"),
    ('```json\n{"ranking": "CAB"}\n```', "CAB"),
    ('ranking: ACB', "ACB"),
    ('"ranking": "B > A > C"', "BAC"),
    ("B > A > C", "BAC"),
    ("CBA", "CBA"),
])
def test_the_parser_accepts_the_shapes_a_model_gives_without_a_json_mode(text, want):
    assert ej.parse_ranking(text, ["A", "B", "C"]) == want


@pytest.mark.parametrize("text", ["", "I like option B best.", '{"ranking": "AAB"}', '{"ranking": "AB"}',
                                  '{"ranking": "ABCD"}', '{"rank": "ABC", "x": 1', '{"ranking": ["A", "B", "C"]}'])
def test_the_parser_rejects_anything_that_is_not_every_letter_once(text):
    assert ej.parse_ranking(text, ["A", "B", "C"]) is None


def test_the_parser_ranks_exactly_the_letters_of_the_call():
    assert ej.parse_ranking('{"ranking": "BA"}', ["A", "B"]) == "BA"
    assert ej.parse_ranking('{"ranking": "BAC"}', ["A", "B"]) is None


def calls(n=5):
    return [{"id": f"c{i}", "system": "s", "user": f"u{i}", "letters": ["A", "B", "C"]} for i in range(n)]


def test_answers_are_checkpointed_and_a_rerun_skips_them(tmp_path):
    out = tmp_path / "replies.jsonl"
    seen = []

    def generate(convs):
        seen.extend(c[1]["content"] for c in convs)
        return ['{"ranking": "CBA"}'] * len(convs)

    msgs = []
    first = ej.answer_calls(calls(5), generate, out, chunk=2, log=msgs.append)
    assert first == {"answered": 5, "failed": 0, "already": 0}
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == [f"c{i}" for i in range(5)] and all(r["ranking"] == "CBA" for r in rows)
    seen.clear()
    assert ej.answer_calls(calls(7), generate, out, chunk=2, log=msgs.append) == {"answered": 2, "failed": 0, "already": 5}
    assert seen == ["u5", "u6"]  # only the new calls were asked


def test_a_torn_last_line_from_a_killed_job_is_dropped_on_resume(tmp_path):
    out = tmp_path / "replies.jsonl"
    out.write_text(json.dumps({"id": "c0", "ranking": "ABC"}) + "\n" + '{"id": "c1", "rank', encoding="utf-8")
    assert ej.load_done(out) == {"c0"}
    assert out.read_text(encoding="utf-8") == json.dumps({"id": "c0", "ranking": "ABC"}) + "\n"


def test_a_bad_reply_is_asked_once_more_with_a_nudge_and_never_written_as_an_answer_if_it_fails_again(tmp_path):
    out, convs_seen = tmp_path / "replies.jsonl", []

    def generate(convs):
        convs_seen.append(convs)
        nudged = [len(c) == 3 and c[-1]["content"] == ej.NUDGE for c in convs]
        return [('{"ranking": "ABC"}' if n and c[1]["content"] != "u1" else "no idea") for c, n in zip(convs, nudged)]

    counts = ej.answer_calls(calls(3), generate, out, chunk=8, log=lambda m: None)
    assert counts == {"answered": 2, "failed": 1, "already": 0}  # c0 and c2 parsed after the nudge, c1 never did
    assert {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()} == {"c0", "c2"}
    failed = [json.loads(line) for line in (tmp_path / "replies.jsonl.failed.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [f["id"] for f in failed] == ["c1"] and failed[0]["reason"] == "not a ranking"
    # the next run asks c1 again: a parse failure is infrastructure, not the judge's answer
    asked = []
    ej.answer_calls(calls(3), lambda convs: asked.extend(c[1]["content"] for c in convs) or ['{"ranking": "CBA"}'] * len(convs), out, log=lambda m: None)
    assert asked == ["u1"]


def test_the_limit_for_a_smoke_test_and_a_greedy_chat_layout(tmp_path):
    c = calls(1)[0]
    assert ej.conv(c) == [{"role": "system", "content": "s"}, {"role": "user", "content": "u0"}]
    assert ej.conv(c, ej.NUDGE)[-1] == {"role": "user", "content": ej.NUDGE}
    assert ej.MAX_NEW_TOKENS <= 60  # a ranking is a few tokens
