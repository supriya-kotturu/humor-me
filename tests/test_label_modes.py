"""Parser for local-LLM label output; no model needed."""
from explorer.label_modes import PROMPT, SAFETY, parse


def test_parses_json_inside_chatter():
    d = parse('Sure! {"mode": "Cynic", "mode2": "observer", "sexual": true}')
    assert d["mode"] == "cynic" and d["mode2"] == "observer"
    assert d["safety"]["sexual"] is True and d["safety"]["slur"] is False


def test_unknown_mode_is_a_failure_not_a_guess():
    assert parse('{"mode": "dad joke"}') is None
    assert parse("no json here") is None


def test_mode2_equal_to_mode_is_dropped():
    assert parse('{"mode": "absurdist", "mode2": "absurdist"}')["mode2"] is None


def test_prompt_lists_every_safety_key():
    assert all(f'"{k}"' in PROMPT for k in SAFETY)
    assert "{joke}" in PROMPT


# --- resumable labeling (no GPU needed) ---
import json

from explorer.label_modes import load_done, pending
from explorer.mode_precision import per_mode

ROWS = [{"id": f"j{i}", "text": "x", "score": 1, "pct": 0.5} for i in range(10)]


def test_pending_is_a_fixed_order_that_resumes_and_extends():
    first = pending(ROWS, set(), n=4, seed=0)
    assert pending(ROWS, set(), n=4, seed=0) == first                      # same seed, same sample
    done = {r["id"] for r in first[:2]}
    assert pending(ROWS, done, n=4, seed=0) == first[2:]                    # resume skips done
    grown = pending(ROWS, {r["id"] for r in first}, n=7, seed=0)
    assert len(grown) == 3 and not {r["id"] for r in grown} & {r["id"] for r in first}  # extend


def test_load_done_drops_a_torn_last_line(tmp_path):
    out = tmp_path / "modes.jsonl"
    out.write_text(json.dumps({"id": "j1", "model": "m"}) + "\n" + '{"id": "j2", "mod', encoding="utf-8")
    done, models = load_done(out)
    assert done == {"j1"} and models == {"m"}
    assert out.read_text(encoding="utf-8").endswith("\n")                   # safe to append again
    assert load_done(tmp_path / "missing.jsonl") == (set(), set())


def test_precision_halfwidth_shrinks_with_more_jokes():
    small = per_mode([{"mode": "cynic", "pct": p} for p in (0.1, 0.9) * 50])
    big = per_mode([{"mode": "cynic", "pct": p} for p in (0.1, 0.9) * 5000])
    assert big["cynic"]["halfwidth"] < small["cynic"]["halfwidth"] / 9      # ~1/sqrt(100)
    assert "raw" not in per_mode([{"raw": "garbage", "pct": 0.5}])          # failures ignored
