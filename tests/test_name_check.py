"""Exp 40 (D-125): the measurement counts what the rules say, lists only plain words, and keeps the review's yes/no.
Fixture threads are written to temp dirs; no real data, no calls."""
import json
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app import safety  # noqa: E402
from sim import name_check as nc  # noqa: E402

CLEAN = "My battery is fine."
FREED = "He said, 'That's fine.'"  # rejected by rules-v3, clean now
NAMED = "I told Dave about my battery."  # rejected by both
BODY = "Everybody here is so fat."  # rejected by both, for another reason


def write_bits(directory, threads, age="mixed", rec_id="r0"):
    directory.mkdir(parents=True, exist_ok=True)
    rec = {"id": rec_id, "context": {"age": age}, "threads": [{"lines": t} for t in threads]}
    (directory / f"{rec_id}.json").write_text(json.dumps(rec), encoding="utf-8")


def test_the_before_rules_are_v3_and_disagree_with_now_on_the_freed_line():
    before = nc.load_before()
    assert before.RULES_VERSION == "rules-v3" and safety.RULES_VERSION == "rules-v4"
    assert not before.check_rules(FREED, "mixed").ok and safety.check_rules(FREED, "mixed").ok


def test_count_pool_counts_rejections_by_reason_and_finds_the_freed_threads(tmp_path):
    write_bits(tmp_path, [[CLEAN, CLEAN], [CLEAN, FREED], [NAMED], [BODY]])
    res, flipped = nc.count_pool(nc.read_bits(tmp_path), nc.load_before(), safety)
    assert (res["units"], res["rejected_v3"], res["rejected_v4"], res["flipped"]) == (4, 3, 2, 1)
    assert [u["key"] for u in flipped] == ["r0:1"]
    assert res["reasons_v3"] == {"named_person": 2, "body_or_appearance": 1}
    assert res["reasons_v4"] == {"named_person": 1, "body_or_appearance": 1}
    assert res["top_flipped"] == [["That's", 1]]


def test_the_record_lists_only_plain_words_never_a_name(tmp_path):
    write_bits(tmp_path, [[NAMED], [FREED], ["Then Gregor left."]])
    res, _ = nc.count_pool(nc.read_bits(tmp_path), nc.load_before(), safety)
    listed = {w for w, _ in res["top_named_v3"] + res["top_flipped"]}
    assert listed <= nc.PLAIN and "Dave" not in listed and "Gregor" not in listed
    assert res["other_named_v3"] == 2  # counted, not named
    assert "Dave" not in json.dumps(res) and "Gregor" not in json.dumps(res)


def test_the_unlisted_count_covers_every_freed_thread_not_only_the_listed_top_25(tmp_path):
    words = sorted(safety.SENTENCE_WORDS)[:40]  # 40 distinct plain words, each freed once
    write_bits(tmp_path, [[f"He told me {w} twice."] for w in words])
    before = nc.load_before()
    units = [u for u in nc.read_bits(tmp_path) if not before.check_rules(u["lines"][0], "mixed").ok]
    res, flipped = nc.count_pool(units, before, safety)
    assert res["flipped"] == len(flipped) > 25 and len(res["top_flipped"]) == 25
    assert res["flipped_unlisted_words"] == 0


def test_the_review_reads_every_risky_freed_thread_and_a_seeded_50_of_the_rest():
    flipped = {"a": [{"key": f"k{i}"} for i in range(80)], "b": [{"key": "x"}]}
    must = frozenset({"b|x", "a|k3"})
    picked = nc.sample_flipped(flipped, must=must)
    keys = [nc.key_of(p, u) for p, u in picked]
    assert keys[:2] == ["a|k3", "b|x"] and len(keys) == 2 + nc.SAMPLE and len(set(keys)) == len(keys)
    assert picked == nc.sample_flipped(flipped, must=must)


def test_a_threads_age_is_used_for_the_teen_rules(tmp_path):
    write_bits(tmp_path, [["We all went out for a beer."]], age="teens")
    res, _ = nc.count_pool(nc.read_bits(tmp_path), nc.load_before(), safety)
    assert res["rejected_v3"] == res["rejected_v4"] == 1 and res["flipped"] == 0


def test_session_lines_come_from_generated_events_only_and_repeats_count_once(tmp_path):
    events = [{"event": "show_start", "context": {"age": "teens"}},
              {"event": "candidates", "items": [{"text": FREED}, {"text": CLEAN}]},
              {"event": "joke_played", "text": FREED},
              {"event": "reaction", "text": "someone in the audience said Dave"},
              {"event": "rating", "text": "Dave"}]
    (tmp_path / "s1.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\nnot json\n", encoding="utf-8")
    units = nc.read_sessions(tmp_path)
    assert [u["lines"] for u in units] == [[FREED], [CLEAN]] and {u["age"] for u in units} == {"teens"}


def test_the_must_still_reject_set_is_all_rejected_as_names():
    res = nc.must_still_reject(safety)
    assert res == {"lines": len(nc.MUST_STILL_REJECT), "rejected": len(nc.MUST_STILL_REJECT),
                   "named_person": len(nc.MUST_STILL_REJECT)}


def test_the_sample_is_seeded_and_capped_at_what_flipped():
    flipped = {"a": [{"key": f"k{i}"} for i in range(80)], "b": [{"key": "x"}]}
    one, two = nc.sample_flipped(flipped), nc.sample_flipped(flipped)
    assert one == two and len(one) == nc.SAMPLE
    assert len(nc.sample_flipped({"a": [{"key": "k"}]})) == 1
    assert one != nc.sample_flipped(flipped, seed=41)


def test_review_keeps_answers_resumes_and_stops_on_q(tmp_path):
    path = tmp_path / "sub" / "review.json"
    picked = [("p", {"key": f"k{i}", "age": "mixed", "lines": [f"line {i}"]}) for i in range(3)]
    shown = []
    answers = iter(["maybe", "n", "y", "q"])
    nc.review(picked, path, ask=lambda _: next(answers), out=shown.append)
    assert json.loads(path.read_text()) == {"p|k0": "no", "p|k1": "yes"}
    assert any("line 0" in s for s in shown)
    again = iter(["n"])  # resumes at the third; k0 and k1 are not asked again
    nc.review(picked, path, ask=lambda _: next(again), out=shown.append)
    state = json.loads(path.read_text())
    assert state == {"p|k0": "no", "p|k1": "yes", "p|k2": "no"}
    assert nc.review_summary(state, 3) == {"sample": 3, "reviewed": 3, "unsafe_or_real_name": 1}


def test_the_verdict_waits_for_the_review_and_fails_on_one_unsafe_line_or_a_loosened_rule():
    ok = {"lines": 16, "rejected": 16}
    assert nc.verdict(ok, {"sample": 50, "reviewed": 10, "unsafe_or_real_name": 0}) == "pending"
    assert nc.verdict(ok, {"sample": 50, "reviewed": 50, "unsafe_or_real_name": 0}) == "adopt"
    assert nc.verdict(ok, {"sample": 50, "reviewed": 12, "unsafe_or_real_name": 1}) == "fail"
    assert nc.verdict({"lines": 16, "rejected": 15}, {"sample": 50, "reviewed": 50, "unsafe_or_real_name": 0}) == "fail"


def test_run_reads_the_sources_and_leaves_the_verdict_pending_without_a_review(tmp_path, monkeypatch):
    write_bits(tmp_path / "rank", [[CLEAN], [FREED]])
    write_bits(tmp_path / "train", [[NAMED]], rec_id="t0")
    monkeypatch.setattr(nc, "SOURCE", {"ranking": tmp_path / "rank", "train": tmp_path / "train", "sessions": tmp_path / "none"})
    monkeypatch.setattr(nc, "REVIEW_FILE", tmp_path / "review.json")
    res = nc.run()
    assert list(res["pools"]) == ["bonsai ranking", "qwen3 training"]  # no live-shows pool when there are no logs
    assert res["bar_b"] == {"sample": 1, "reviewed": 0, "unsafe_or_real_name": 0, "risky_all_read": 0} and res["verdict"] == "pending"
    assert "| bonsai ranking | 2 | 1 (50.0%) | 0 (0.0%) | 1 |" in nc.table(res)


def test_a_freed_thread_is_risky_unless_its_first_rejected_word_was_a_plain_word():
    units = [{"key": "a"}, {"key": "b"}, {"key": "c"}, {"key": "d"}]
    reasons = {"a": "named_person:That's", "b": "named_person:Market", "c": "named_person:TikTok's", "d": "blocklist:x"}
    assert [u["key"] for u in nc.risky(units, reasons)] == ["b", "d"]


GARBLED = "Wait, the JSON in my thought trace was off, so here it is again.}]}"


def test_the_format_rule_un_frees_a_garbled_thread_and_the_record_counts_it(tmp_path):
    write_bits(tmp_path / "rank", [[CLEAN], [FREED, GARBLED], [FREED]])
    units = nc.read_bits(tmp_path / "rank")
    res, flipped = nc.count_pool(units, nc.load_before(), safety)
    assert res["garbled"] == 1 and res["flipped"] == 1 and res["garbled_freed"] == 0
    res_old, _ = nc.count_pool(units, nc.load_before(), nc.load_first_check())
    assert res_old["flipped"] == 2 and nc.garbled(units[1]) and not nc.garbled(units[2])


def test_the_second_check_never_draws_a_thread_the_first_one_read_and_is_seeded():
    flipped = {"p": [{"key": f"k{i}", "age": "mixed", "lines": ["x"]} for i in range(200)]}
    first = {f"p|k{i}" for i in range(0, 200, 3)}  # includes keys that may no longer be freed
    first.add("p|gone")
    picked = nc.sample_second(flipped, first)
    keys = [nc.key_of(*x) for x in picked]
    assert len(keys) == nc.SAMPLE == len(set(keys)) and not set(keys) & first
    assert picked == nc.sample_second(flipped, first)
    assert nc.sample_second(flipped, set(), n=500) and len(nc.sample_second({"p": flipped["p"][:7]}, set())) == 7


def test_review_writes_the_second_file_and_leaves_the_first_intact(tmp_path, monkeypatch):
    first, second = tmp_path / "review.json", tmp_path / "review2.json"
    first.write_text('{"p|old": "no"}', encoding="utf-8")
    monkeypatch.setattr(nc, "REVIEW_FILE", first)
    monkeypatch.setattr(nc, "REVIEW2_FILE", second)
    nc.review([("p", {"key": "k0", "age": "mixed", "lines": ["a line"]})], ask=lambda _: "n", out=lambda _: None)
    assert json.loads(second.read_text()) == {"p|k0": "no"} and first.read_text() == '{"p|old": "no"}'


def test_run_reports_both_checks_and_scores_the_second_on_its_own_answers(tmp_path, monkeypatch):
    write_bits(tmp_path / "rank", [[FREED], [FREED, GARBLED], [FREED, "I told Dave."], [FREED]])
    write_bits(tmp_path / "train", [[CLEAN]], rec_id="t0")
    monkeypatch.setattr(nc, "SOURCE", {"ranking": tmp_path / "rank", "train": tmp_path / "train", "sessions": tmp_path / "none"})
    first, second = tmp_path / "review.json", tmp_path / "review2.json"
    first.write_text(json.dumps({"bonsai ranking|r0:0": "yes", "bonsai ranking|r0:1": "no"}), encoding="utf-8")
    monkeypatch.setattr(nc, "REVIEW_FILE", first)
    monkeypatch.setattr(nc, "REVIEW2_FILE", second)
    res = nc.run()
    assert res["first_check"]["reviewed"] == 2 and res["first_check"]["unsafe_or_real_name"] == 1
    assert res["first_check"]["verdict"] == "fail" and res["first_check"]["still_freed"] == 1  # r0:1 is garbled now
    assert res["bar_b"]["sample"] == 1 and res["verdict"] == "pending"  # r0:0 is excluded: the first check read it
    second.write_text(json.dumps({"bonsai ranking|r0:3": "no"}), encoding="utf-8")
    assert nc.run()["verdict"] == "adopt" and first.read_text() == json.dumps({"bonsai ranking|r0:0": "yes", "bonsai ranking|r0:1": "no"})
