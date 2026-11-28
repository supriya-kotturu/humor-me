"""Exp 39: scoring other raters against the user and the stored judge replies. Fixture key, CSV and judge caches in temp
dirs; no data/, no calls."""
import csv
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from sim import dpo_friend as df  # noqa: E402
from sim import dpo_pairs as dp  # noqa: E402

# pair number, id, role, half, side1, side2, user_pick: the user picked side 1 in pairs 1, 3, 5 and side 2 in 2, 4, 6
KEY = [(1, "a", "teach", "plain", 0, 1, 0), (2, "b", "teach", "tonight", 0, 1, 1), (3, "c", "test", "plain", 2, 3, 2),
       (4, "d", "test", "plain", 2, 3, 3), (5, "e", "test", "tonight", 4, 5, 4), (6, "f", "test", "tonight", 4, 5, 5)]


@pytest.fixture
def env(tmp_path, monkeypatch):
    key = tmp_path / "key.json"
    key.write_text(json.dumps({"seed": "x", "pairs": [dict(zip(("pair", "id", "role", "half", "side1", "side2", "user_pick"), k))
                                                      for k in KEY]}))
    # the user's winner is always the left thread (0) in these pairs
    pairs = [{"id": k[1], "role": k[2], "half": k[3], "left": [], "right": [], "winner": 0} for k in KEY]
    monkeypatch.setattr(dp, "need_pairs", lambda root=None: ({"digest": "p"}, {"answers": {}}, pairs))
    # judge replies "A" in "ab" and "B" in "ba" = the left thread both times (agrees with the user); pairs c and d only
    consistent = {"c": ("A", "B"), "d": ("B", "A")}  # d: the judge picks the right thread, against the user

    def letters(pairs_, root=None, which="haiku"):
        out = {}
        for p in pairs_:
            if p["role"] == "test":
                a, b = consistent.get(p["id"], ("A", "A"))  # anything else flips with the order
                out[(p["id"], "ab")], out[(p["id"], "ba")] = a, b
        return out

    monkeypatch.setattr(dp, "read_letters", letters)
    return tmp_path, key


def write_csv(path, rows):
    head = ["Timestamp", "Your name"] + [c for n in range(1, 7) for c in (f"Pair {n}: which is funnier?", f"Pair {n} note (optional)")]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(head)
        for ts, name, picks, notes in rows:
            w.writerow([ts, name] + [x for p, nt in zip(picks, notes) for x in (p, nt)])


QUIET = [""] * 6
# agrees with the user on pairs 1, 3, 5 (side 1 is the user's pick) and 4, 6 (side 2), disagrees on 2
ALL_AGREE = ["Bit 1", "Bit 1", "Bit 1", "Bit 2", "Bit 1", "Bit 2"]


def test_names_and_notes_never_reach_the_output(env):
    tmp, key = env
    write_csv(tmp / "r.csv", [("11/27/2026 20:00:13", "Zelda Fitzgerald", ALL_AGREE, ["secret note"] * 6)])
    text = json.dumps(df.score(tmp / "r.csv", key))
    assert "Zelda" not in text and "secret" not in text and "Fitzgerald" not in text


def test_raters_are_numbered_in_submit_order_not_file_order(env):
    tmp, key = env
    write_csv(tmp / "r.csv", [("11/27/2026 21:00:00", "Late", ["Bit 2"] * 6, QUIET), ("11/27/2026 09:00:00", "Early", ALL_AGREE, QUIET)])
    res = df.score(tmp / "r.csv", key)
    assert [r["rater"] for r in res["raters"]] == ["Rater 1", "Rater 2"]
    assert res["raters"][0]["vs_user"]["agree"] == 5  # Early's picks


def test_cant_pick_is_left_out_of_every_count(env):
    tmp, key = env
    write_csv(tmp / "r.csv", [("11/27/2026 20:00:00", "A", ["Can't pick", "Bit 2", "Bit 1", "Can't pick", "Bit 1", "Bit 2"], QUIET)])
    r = df.score(tmp / "r.csv", key)["raters"][0]
    assert r["cant_pick"] == 2 and r["vs_user"]["decided"] == 4 and r["vs_user"]["agree"] == 4
    assert sum(c["decided"] for c in r["by_cell"].values()) == 4


def test_judge_comparison_uses_test_pairs_the_judge_answered_in_both_orders(env):
    tmp, key = env
    # the rater agrees with the user on c, d (test) and e; the judge agrees with the user on c, against it on d
    write_csv(tmp / "r.csv", [("11/27/2026 20:00:00", "A", ALL_AGREE, QUIET)])
    r = df.score(tmp / "r.csv", key)["raters"][0]
    assert r["test_vs_user"]["decided"] == 4  # teach pairs a, b are not in it
    # haiku has labels for c, d and the "A","A" flip on e, f: two consistent pairs; the rater matches the judge on c only
    assert r["vs_judge"]["haiku"]["decided"] == 2 and r["vs_judge"]["haiku"]["agree"] == 1
    assert r["vs_judge"]["haiku"]["no_label"] == 2
    assert r["vs_judge"]["opus"]["decided"] == 2 and r["vs_judge"]["opus"]["no_label"] == 2


def test_two_raters_get_a_pairwise_and_a_majority_comparison(env):
    tmp, key = env
    write_csv(tmp / "r.csv", [("11/27/2026 20:00:00", "A", ALL_AGREE, QUIET), ("11/27/2026 20:05:00", "B", ["Bit 2"] * 6, QUIET)])
    res = df.score(tmp / "r.csv", key)
    assert res["raters_vs_each_other"][0]["decided"] == 6
    # they split 1-1 on pairs 1, 2, 3, 5 (tied, left out) and both pick Bit 2 on 4 and 6, as the user did
    assert res["majority_vs_user"]["tied_left_out"] == 4
    assert (res["majority_vs_user"]["agree"], res["majority_vs_user"]["decided"]) == (2, 2)


def test_a_single_rater_has_no_pairwise_or_majority_block(env):
    tmp, key = env
    write_csv(tmp / "r.csv", [("11/27/2026 20:00:00", "A", ALL_AGREE, QUIET)])
    assert "majority_vs_user" not in df.score(tmp / "r.csv", key)


def test_the_digest_ignores_names_notes_and_timestamps_but_not_picks(env):
    tmp, key = env
    write_csv(tmp / "a.csv", [("11/27/2026 20:00:00", "Ann", ALL_AGREE, QUIET)])
    write_csv(tmp / "b.csv", [("11/28/2026 08:00:00", "Bob", ALL_AGREE, ["hi"] * 6)])
    write_csv(tmp / "c.csv", [("11/27/2026 20:00:00", "Ann", ["Bit 2"] + ALL_AGREE[1:], QUIET)])
    assert df.picks_digest(tmp / "a.csv", 6) == df.picks_digest(tmp / "b.csv", 6)
    assert df.picks_digest(tmp / "a.csv", 6) != df.picks_digest(tmp / "c.csv", 6)
