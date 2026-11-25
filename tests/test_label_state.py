import json

from explorer.label_state import AGE, ENERGY, PROMPT, VENUE, export, parse, sample_ids
from sim.env import AGES, ENERGIES, VENUES


def test_vocab_matches_sim():
    # Labels must be values the live loop sends, or the DPO prompts won't match show prompts.
    assert (tuple(VENUE), tuple(ENERGY), tuple(AGE)) == (VENUES, ENERGIES, AGES)


def test_prompt_has_joke_slot_and_every_value():
    assert PROMPT.count("{joke}") == 1
    for v in (*VENUES, *ENERGIES, *AGES):
        assert f'"{v}"' in PROMPT


def test_parse_accepts_valid_and_normalizes_case():
    assert parse('sure: {"venue": "Bar", "energy": "low", "age": "teens"}') == \
        {"venue": "bar", "energy": "low", "age": "teens"}


def test_parse_rejects_unknown_or_missing_values():
    assert parse('{"venue": "pub", "energy": "low", "age": "teens"}') is None
    assert parse('{"venue": "bar", "energy": "low"}') is None
    assert parse("no json") is None
    assert parse("{not json}") is None


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_sample_ids_keeps_file_order(tmp_path):
    _write(tmp_path / "m.jsonl", [{"id": "c"}, {"id": "a"}, {"id": "b"}])
    assert sample_ids(tmp_path / "m.jsonl") == ["c", "a", "b"]


def test_export_joins_only_rows_with_both_labels(tmp_path):
    _write(tmp_path / "train.jsonl", [{"id": i, "text": f"joke {i}"} for i in "abcd"])
    _write(tmp_path / "modes.jsonl", [
        {"id": "a", "score": 5, "pct": 0.5, "mode": "cynic", "mode2": None, "safety": {"slur": False}},
        {"id": "b", "score": 1, "pct": 0.1, "raw": "parse failure"},
        {"id": "c", "score": 2, "pct": 0.2, "mode": "observer"},
    ])
    _write(tmp_path / "state.jsonl", [
        {"id": "a", "venue": "bar", "energy": "high", "age": "mixed"},
        {"id": "b", "venue": "bar", "energy": "low", "age": "teens"},
        {"id": "c", "raw": "parse failure"},
    ])
    written, skipped = export(tmp_path / "modes.jsonl", tmp_path / "state.jsonl",
                              tmp_path / "train.jsonl", tmp_path / "out.jsonl")
    rows = [json.loads(line) for line in open(tmp_path / "out.jsonl", encoding="utf-8")]
    assert (written, skipped) == (1, 2)
    assert rows == [{"id": "a", "text": "joke a", "score": 5, "pct": 0.5, "mode": "cynic", "mode2": None,
                     "safety": {"slur": False}, "venue": "bar", "energy": "high", "age": "mixed"}]


def test_spread_counts_every_value_and_flags_constant_fields(tmp_path):
    from explorer.label_state import run, table
    rows = [{"venue": "showcase", "energy": "high", "age": "young_adults"}] * 9 + \
           [{"venue": "bar", "energy": "low", "age": "teens"}]
    _write(tmp_path / "x.jsonl", rows)
    _write(tmp_path / "s.jsonl", rows + [{"id": "z", "raw": "bad"}])
    _write(tmp_path / "m.jsonl", [{"id": i} for i in range(12)])
    r = run(tmp_path / "x.jsonl", tmp_path / "s.jsonl", tmp_path / "m.jsonl")
    assert (r["n"], r["left_out"], r["parse_failures"]) == (10, 2, 1)
    assert dict(r["fields"]["venue"]) == {"showcase": 9, "bar": 1, "company_event": 0}  # an unused value still shows
    assert r["top_share"] == {"venue": 0.9, "energy": 0.9, "age": 0.9}
    assert "| venue | showcase | 9 | 90.0% |" in table(r)
