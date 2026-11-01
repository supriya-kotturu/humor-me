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


def test_explorer_scripts_run_on_python39():
    """The Explorer venv is Python 3.9 (2026-10-31): `X | None` hints must not be evaluated there."""
    import pathlib
    for f in pathlib.Path(__file__).resolve().parent.parent.joinpath("explorer").glob("*.py"):
        src = f.read_text(encoding="utf-8")
        if "| None" in src or "|None" in src:
            assert "from __future__ import annotations" in src, f.name


def test_length_order_groups_similar_lengths_and_is_a_permutation():
    from explorer.label_modes import length_order
    texts = ["ccc", "a", "bb", "dddd"]
    order = length_order(texts)
    assert sorted(order) == [0, 1, 2, 3]
    assert [len(texts[i]) for i in order] == [1, 2, 3, 4]


def test_shell_scripts_have_no_literal_backslash_n():
    """A heredoc-escaping slip once wrote a literal backslash-n instead of a line break into the sbatch file
    (2026-10-31), which would break the job's command line on the cluster."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parent.parent / "explorer"
    for f in list(root.glob("*.sbatch")) + list(root.glob("*.sh")):
        assert "\\n" not in f.read_text(encoding="utf-8"), f.name


def test_split_on_oom_halves_until_batches_fit_and_keeps_every_reply():
    from explorer.label_modes import split_on_oom

    class OOM(Exception):
        pass

    calls = []

    def run(idx):
        calls.append(len(idx))
        if len(idx) > 2 or 7 in idx:  # batches over 2 overflow; joke 7 never fits
            raise OOM
        return [f"r{j}" for j in idx]

    out = split_on_oom(run, list(range(8)), lambda e: isinstance(e, OOM))
    assert sorted(out) == list(range(8))
    assert out[7] == "" and out[0] == "r0"
    assert max(c for c in calls) == 8 and 1 in calls


def test_split_on_oom_reraises_other_errors():
    import pytest
    from explorer.label_modes import split_on_oom

    def run(idx):
        raise ValueError("not memory")

    with pytest.raises(ValueError):
        split_on_oom(run, [0, 1], lambda e: False)


# --- hand check of judge labels ---
import csv


def _hand(tmp_path, rows):
    hand, modes = tmp_path / "hand.csv", tmp_path / "modes.jsonl"
    cols = ["n", "id", "judge_mode", "judge_mode2", "judge_flags", "marked", "verdict", "human_mode", "human_flags"]
    with hand.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: r.get(c, "") for c in cols} for r in rows)
    modes.write_text("".join(json.dumps({"id": f"m{i}", "mode": m}) + "\n"
                             for i, m in enumerate(["cynic"] * 9 + ["optimist"])), encoding="utf-8")
    return str(hand), str(modes)


def test_hand_check_reweights_by_mode_share_and_ignores_unjudged(tmp_path):
    from explorer.hand_check import run
    rows = ([{"judge_mode": "cynic", "verdict": "y"}] * 2
            + [{"judge_mode": "optimist", "verdict": "n", "human_mode": "wordsmith"}] * 2
            + [{"judge_mode": "optimist", "verdict": "?"}, {"judge_mode": "cynic", "verdict": "skip"}])
    res = run(*_hand(tmp_path, rows))
    assert res["sample_precision"] == 0.5                                    # ? and skip left out
    assert abs(res["weighted_precision"] - 0.9) < 1e-9                       # 90% cynic, all right
    assert res["misses"] == {"optimist → wordsmith": 2}


def test_hand_check_counts_judge_misses_per_human_flag(tmp_path):
    from explorer.hand_check import run
    rows = [{"judge_mode": "cynic", "verdict": "y", "human_flags": "sexual", "judge_flags": "sexual"},
            {"judge_mode": "cynic", "verdict": "y", "human_flags": "sexual,other:nazi"}]
    sf = run(*_hand(tmp_path, rows))["safety"]
    assert sf["by_flag"]["sexual"] == {"human": 2, "judge_too": 1}
    assert sf["by_flag"]["outside schema"] == {"human": 1, "judge_too": 0}
    assert sf["judge_flagged_any_of_those"] == 1


def test_wilson_stays_inside_zero_one():
    from explorer.hand_check import wilson
    lo, hi = wilson(12, 12)
    assert 0 < lo < 1 and hi == 1.0
    assert wilson(0, 0) == (0.0, 1.0)
