"""Exp 39 (reported only, no bar; evidence under D-122): do other people pick like you, or like the judges?

Raters saw a subset of the user's ranked pairs in a form, blind (key.json: which pair, which thread on which side).
This scores their picks against the user's and, on test pairs, against the stored judge replies. Reads local files only:
no calls. Numbers only in the record: no names, no notes, and raters are "Rater 1", "Rater 2" in submit order.
"""
import argparse
import csv
import datetime
import hashlib
import itertools
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from sim import cache  # noqa: E402
from sim import dpo_pairs as dp  # noqa: E402
from sim import judge_calibration as jc  # noqa: E402

NAME = "dpo_friend"
FRIEND = dp.DATA / "friend"
COLUMN = "Pair {n}: which is funnier?"
STAMP = "%m/%d/%Y %H:%M:%S"
SOURCE = {"csv": FRIEND / "responses.csv"}  # set by the CLI: a path is not a param, it changes with the machine


def rng(k: int, n: int) -> list[float] | None:
    return [round(x, 4) for x in jc.wilson(k, n, 1.96)] if n else None


def stat(k: int, n: int) -> dict:
    return {"agree": k, "decided": n, "range95": rng(k, n)}


def read_key(path: pathlib.Path | None = None) -> dict[int, dict]:
    return {k["pair"]: k for k in json.loads((path or FRIEND / "key.json").read_text(encoding="utf-8"))["pairs"]}


def read_raters(csv_path: pathlib.Path, n_pairs: int) -> list[dict[int, int]]:
    """One dict per response row, in submit order: pair number -> 1 or 2 (the form's Bit 1 / Bit 2). "Can't pick" is
    absent. Names, notes and timestamps never leave this function."""
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    def when(row):
        try:
            return datetime.datetime.strptime(row.get("Timestamp", "").strip(), STAMP)
        except ValueError:
            return datetime.datetime.max

    out = []
    for row in sorted(rows, key=when):  # stable: unparsable stamps keep file order
        picks = {}
        for n in range(1, n_pairs + 1):
            v = row.get(COLUMN.format(n=n), "").strip()
            if v in ("Bit 1", "Bit 2"):
                picks[n] = 1 if v == "Bit 1" else 2
        out.append(picks)
    return out


def picks_digest(csv_path: pathlib.Path, n_pairs: int) -> str:
    """Over the pick columns only, so a name or a note changing does not change the record's key."""
    return hashlib.sha256(json.dumps(read_raters(csv_path, n_pairs), sort_keys=True).encode()).hexdigest()[:8]


def judge_picks(pairs: list[dict], root: pathlib.Path | None = None) -> dict[str, dict[str, bool]]:
    """judge -> {pair id: picks the user's thread}, only test pairs answered the same way in both orders."""
    out = {}
    for w in ("haiku", "opus"):
        letters = dp.read_letters(pairs, root, w)
        out[w] = {}
        for p in pairs:
            if p["role"] != "test":
                continue
            two = [dp.picked_thread(letters.get((p["id"], o)), o) for o in dp.ORDERS]
            if None not in two and two[0] == two[1]:
                out[w][p["id"]] = two[0] == p["winner"]
    return out


def score(csv_path: pathlib.Path, key_path: pathlib.Path | None = None, root: pathlib.Path | None = None) -> dict:
    key = read_key(key_path)
    user = {n: 1 if k["user_pick"] == k["side1"] else 2 for n, k in key.items()}
    _, _, pairs = dp.need_pairs(root)
    assert all(k["id"] in {p["id"] for p in pairs} for k in key.values()), "key.json names a pair that is not in the ranking"
    judges = judge_picks(pairs, root)
    raters = read_raters(csv_path, len(key))
    cells = sorted({(k["role"], k["half"]) for k in key.values()})
    out = []
    for i, picks in enumerate(raters, 1):
        agree = {n: picks[n] == user[n] for n in picks}
        res = {"rater": f"Rater {i}", "pairs": len(key), "cant_pick": len(key) - len(picks),
               "vs_user": stat(sum(agree.values()), len(agree)), "by_cell": {}}
        for role, half in cells:
            ns = [n for n in agree if (key[n]["role"], key[n]["half"]) == (role, half)]
            res["by_cell"][f"{role}/{half}"] = stat(sum(agree[n] for n in ns), len(ns))
        test = [n for n in agree if key[n]["role"] == "test"]
        res["test_vs_user"] = stat(sum(agree[n] for n in test), len(test))
        res["vs_judge"] = {}
        for w, said in judges.items():
            both = [n for n in test if key[n]["id"] in said]
            # the rater and the judge are consistent when both agree with the user, or both disagree
            res["vs_judge"][w] = {**stat(sum(agree[n] == said[key[n]["id"]] for n in both), len(both)),
                                  "no_label": len(test) - len(both)}
        out.append(res)
    result = {"raters": out, "n_raters": len(out), "pairs": len(key),
              "cells": {f"{r}/{h}": sum((k["role"], k["half"]) == (r, h) for k in key.values()) for r, h in cells}}
    if len(raters) > 1:
        result["raters_vs_each_other"] = []
        for (a, pa), (b, pb) in itertools.combinations(enumerate(raters, 1), 2):
            both = set(pa) & set(pb)
            result["raters_vs_each_other"].append({"pair_of_raters": f"Rater {a} / Rater {b}",
                                                   **stat(sum(pa[n] == pb[n] for n in both), len(both))})
        maj = {}
        for n in key:
            votes = [p[n] for p in raters if n in p]
            if votes.count(1) != votes.count(2):
                maj[n] = 1 if votes.count(1) > votes.count(2) else 2
        result["majority_vs_user"] = {**stat(sum(maj[n] == user[n] for n in maj), len(maj)), "tied_left_out": len(key) - len(maj)}
    return result


def params(csv_path: pathlib.Path, key_path: pathlib.Path | None = None, root: pathlib.Path | None = None) -> dict:
    key_path = key_path or FRIEND / "key.json"
    pool, state, pairs = dp.need_pairs(root)
    teach = dp.teach_prompt(pairs)
    return {"picks_digest": picks_digest(csv_path, len(read_key(key_path))),
            "key_digest": hashlib.sha256(key_path.read_bytes()).hexdigest()[:8],
            "answers_digest": dp.answers_digest(state), "pool_digest": pool["digest"],
            "judge_digest_": dp.judge_digest(teach, "haiku"), "judge_digest_opus": dp.judge_digest(teach, "opus")}


def run(picks_digest: str = "", key_digest: str = "", answers_digest: str = "", pool_digest: str = "",
        judge_digest_: str = "", judge_digest_opus: str = "") -> dict:
    """Exp 39's record: the params pin which picks, key, pool and judge caches the numbers are for."""
    return score(SOURCE["csv"])


def stored() -> dict | None:
    """The committed record (numbers only), for the notebook; None until `score --record` has run."""
    path = cache.METRICS / f"{NAME}.json"
    return json.loads(path.read_text())["results"] if path.exists() else None


def table(res: dict) -> str:
    def fmt(s: dict) -> str:
        return "n/a" if not s["decided"] else f"{s['agree']} of {s['decided']}"

    rows = ["| Rater | Vs you | Test pairs vs you | Vs Haiku | Vs Opus | Can't pick |", "|---|---|---|---|---|---|"]
    for r in res["raters"]:
        rows.append(f"| {r['rater']} | {fmt(r['vs_user'])} | {fmt(r['test_vs_user'])} | {fmt(r['vs_judge']['haiku'])} | "
                    f"{fmt(r['vs_judge']['opus'])} | {r['cant_pick']} |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    s = sub.add_parser("score", help="score a responses CSV against the key and the stored judge replies")
    s.add_argument("--csv", required=True, type=pathlib.Path)
    s.add_argument("--record", action="store_true", help=f"write reports/metrics/{NAME}.json")
    a = ap.parse_args()
    SOURCE["csv"] = a.csv
    res = cache.run_or_load(NAME, run, rerun=True, **params(a.csv)) if a.record else score(a.csv)
    print(table(res), flush=True)


if __name__ == "__main__":
    main()
