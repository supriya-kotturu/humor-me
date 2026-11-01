"""How far can the R2 judge's labels be trusted? Agreement with 105 hand-checked jokes.

Usage (laptop, no GPU):
    python explorer/hand_check.py [data/labels/rjokes_hand_check.csv] [data/processed/rjokes/modes.jsonl]

The hand sample is stratified by judge mode (15 per mode), so the raw hit rate over-weights rare
modes; the population estimate re-weights each mode's precision by its share of the labeled sample.
Safety notes were written only where something stood out, so judge *misses* are measurable but
judge false alarms are not.
"""
from __future__ import annotations  # cluster venv is Python 3.9: keeps `X | None` hints unevaluated

import collections
import csv
import json
import math
import pathlib
import sys

Z95 = 1.96
HAND = "data/labels/rjokes_hand_check.csv"
MODES = "data/processed/rjokes/modes.jsonl"


def wilson(k: int, n: int) -> tuple[float, float]:
    # Wilson, not normal-approx: ~12 jokes per mode with rates near 1 would give bounds above 100%.
    if n == 0:
        return 0.0, 1.0
    p, z2 = k / n, Z95 ** 2
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = Z95 * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return max(0.0, mid - half), min(1.0, mid + half)


def mode_shares(path: str) -> dict[str, float]:
    c = collections.Counter()
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "mode" in r:
            c[r["mode"]] += 1
    total = sum(c.values())
    return {m: k / total for m, k in sorted(c.items())}


def run(hand: str = HAND, modes: str = MODES) -> dict:
    rows = list(csv.DictReader(pathlib.Path(hand).open(encoding="utf-8", newline="")))
    shares = mode_shares(modes)

    per_mode = {}
    for m in sorted({r["judge_mode"] for r in rows}):
        rs = [r for r in rows if r["judge_mode"] == m]
        v = collections.Counter(r["verdict"] for r in rs)
        judged = v["y"] + v["n"]
        lo, hi = wilson(v["y"], judged)
        per_mode[m] = {"y": v["y"], "n": v["n"], "unsure": v["?"], "skip": v["skip"],
                       "precision": v["y"] / judged if judged else None, "ci": [lo, hi],
                       "share": shares.get(m, 0.0)}

    judged = [r for r in rows if r["verdict"] in ("y", "n")]
    hits = sum(r["verdict"] == "y" for r in judged)
    weighted = sum(s["share"] * s["precision"] for s in per_mode.values() if s["precision"] is not None)
    weighted /= sum(s["share"] for s in per_mode.values() if s["precision"] is not None)
    misses = collections.Counter(f"{r['judge_mode']} → {r['human_mode'] or '?'}"
                                 for r in rows if r["verdict"] == "n")

    flagged = [r for r in rows if r["human_flags"]]
    by_flag = collections.defaultdict(lambda: {"human": 0, "judge_too": 0})
    for r in flagged:
        judge = set(filter(None, r["judge_flags"].split(",")))
        for f in r["human_flags"].split(","):
            key = f if not f.startswith("other:") else "outside schema"
            by_flag[key]["human"] += 1
            by_flag[key]["judge_too"] += f in judge

    return {
        "rows": len(rows),
        "verdicts": dict(collections.Counter(r["verdict"] for r in rows)),
        "sample_precision": hits / len(judged),
        "sample_ci": list(wilson(hits, len(judged))),
        "weighted_precision": weighted,
        "per_mode": per_mode,
        "misses": dict(misses.most_common()),
        "safety": {
            "human_flagged": len(flagged),
            "judge_flagged_any_of_those": sum(bool(r["judge_flags"]) for r in flagged),
            "judge_flagged_total": sum(bool(r["judge_flags"]) for r in rows),
            "by_flag": dict(sorted(by_flag.items())),
        },
    }


def table(res: dict) -> str:
    out = ["| Judge mode | Share of labels | y | n | ? | skipped | Precision | 95% CI |", "|---|---|---|---|---|---|---|---|"]
    for m, s in sorted(res["per_mode"].items(), key=lambda kv: -kv[1]["share"]):
        p = "–" if s["precision"] is None else f"{s['precision']:.0%}"
        out.append(f"| {m} | {s['share']:.1%} | {s['y']} | {s['n']} | {s['unsure']} | {s['skip']} "
                   f"| {p} | {s['ci'][0]:.0%}–{s['ci'][1]:.0%} |")
    lo, hi = res["sample_ci"]
    out.append(f"\nPrecision on judged rows: **{res['sample_precision']:.0%}** ({lo:.0%}–{hi:.0%}); "
               f"re-weighted to the labeled mode mix: **{res['weighted_precision']:.0%}**.")
    out.append("Misses (judge → human): " + ", ".join(f"{k} ×{v}" for k, v in res["misses"].items()))
    sf = res["safety"]
    out.append(f"\n| Human safety note | Rows | Judge flagged it too |\n|---|---|---|")
    for f, s in sf["by_flag"].items():
        out.append(f"| {f} | {s['human']} | {s['judge_too']} |")
    out.append(f"\nJudge flagged {sf['judge_flagged_total']} rows; of the {sf['human_flagged']} rows a human "
               f"flagged, it flagged {sf['judge_flagged_any_of_those']}.")
    return "\n".join(out)


def main() -> None:
    print(table(run(*sys.argv[1:3])))


if __name__ == "__main__":
    main()
