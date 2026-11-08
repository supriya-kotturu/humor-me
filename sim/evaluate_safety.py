"""Exp 19: does the rule-based safety filter catch the hard-rule red-team lines without rejecting clean ones?

Usage:
    python -m sim.evaluate_safety

Pass criterion, fixed in docs/plans/live-loop.md (T8) before any rule was written: catch 100% of the hard-rule
lines AND reject at most 10% of clean lines. The red-team list is split in the file: rules were written while
looking at `tune` lines only, so the verdict uses the `held` lines. A miss is reported, not re-tuned.
"""
import hashlib
import math
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
RED_TEAM = ROOT / "tests" / "red_team.yaml"
CLEAN = ROOT / "tests" / "clean_lines.yaml"
BAR_CATCH, BAR_FALSE_REJECT = 1.0, 0.10
CLEAN_AGES = ("mixed", "teens")

sys.path.insert(0, str(ROOT / "backend"))
from app import safety  # noqa: E402


def load(path: pathlib.Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def digest() -> str:
    """Part of the cache key, so editing a fixture file recomputes the experiment."""
    return hashlib.sha256(RED_TEAM.read_bytes() + CLEAN.read_bytes()).hexdigest()[:12]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """95% interval for a rate; with a handful of lines it is wide, which is the honest picture."""
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def rate(rows: list[dict]) -> dict:
    k = sum(r["rejected"] for r in rows)
    return {"n": len(rows), "caught": k, "rate": k / len(rows) if rows else 0.0, "ci95": wilson(k, len(rows))}


def run(fixtures: str = "", bar_catch: float = BAR_CATCH, bar_false_reject: float = BAR_FALSE_REJECT) -> dict:
    assert fixtures in ("", digest()), "fixture files changed since this call was built"
    red = [dict(r, rejected=not safety.check(r["text"], r["age"]).ok, reason=safety.check(r["text"], r["age"]).reason)
           for r in load(RED_TEAM)]
    hard = [r for r in red if r["hard"]]
    cats = sorted({r["cat"] for r in red})
    per_cat = {c: {s: rate([r for r in red if r["cat"] == c and r["split"] == s]) for s in ("tune", "held")}
               for c in cats}
    overall = {s: rate([r for r in hard if r["split"] == s]) for s in ("tune", "held")}

    clean = load(CLEAN)
    false_reject = {}
    for age in CLEAN_AGES:
        rows = [dict(c, rejected=not (v := safety.check(c["text"], age)).ok, reason=v.reason) for c in clean]
        false_reject[age] = {"all": rate(rows), "rejected": [{"id": r["id"], "split": r["split"], "text": r["text"],
                                                              "reason": r["reason"]} for r in rows if r["rejected"]],
                             **{s: rate([r for r in rows if r["split"] == s]) for s in ("tune", "held")}}

    # Teens must be strictly tighter: never looser on any line, and tighter on at least one.
    lines = [(r["text"]) for r in red] + [c["text"] for c in clean]
    looser = [t for t in lines if safety.check(t, "teens").ok and not safety.check(t, "mixed").ok]
    tighter = [t for t in lines if not safety.check(t, "teens").ok and safety.check(t, "mixed").ok]
    # Every non-teen hard line must also be rejected under teens.
    adult_hard_leaks = [r["id"] for r in hard if r["age"] == "mixed" and r["rejected"]
                        and safety.check(r["text"], "teens").ok]

    missed = [{"id": r["id"], "cat": r["cat"], "split": r["split"], "text": r["text"]} for r in hard
              if not r["rejected"]]
    soft = rate([r for r in red if not r["hard"]])
    held_catch = overall["held"]["rate"]
    fr = {a: false_reject[a]["all"]["rate"] for a in CLEAN_AGES}
    return {
        "n_red_team": len(red), "n_hard": len(hard), "n_clean": len(clean), "per_category": per_cat,
        "hard_overall": overall, "soft": soft, "missed_hard": missed, "false_reject": false_reject,
        "teens": {"looser": looser, "tighter_count": len(tighter), "adult_hard_leaks": adult_hard_leaks},
        "criteria": {"bar_catch": bar_catch, "bar_false_reject": bar_false_reject, "held_catch": held_catch,
                     "false_reject": fr, "catch_pass": held_catch >= bar_catch,
                     "false_reject_pass": all(v <= bar_false_reject for v in fr.values())},
    }


def table(res: dict) -> str:
    def cell(r: dict) -> str:
        return f"{r['caught']}/{r['n']} ({r['rate']:.0%})"

    rows = ["| Category | Tuning: caught | Held out: caught | Held out 95% CI |", "|---|---|---|---|"]
    for c, v in res["per_category"].items():
        lo, hi = v["held"]["ci95"]
        rows.append(f"| {c} | {cell(v['tune'])} | {cell(v['held'])} | {lo:.0%} to {hi:.0%} |")
    h = res["hard_overall"]
    lo, hi = h["held"]["ci95"]
    rows.append(f"| **All hard-rule lines** | {cell(h['tune'])} | {cell(h['held'])} | {lo:.0%} to {hi:.0%} |")
    fr = res["false_reject"]
    rows += ["", "| Clean lines | Rejected, all | Tuning half | Held-out half |", "|---|---|---|---|"]
    for age, v in fr.items():
        rows.append(f"| age {age} | {cell(v['all'])} | {cell(v['tune'])} | {cell(v['held'])} |")
    return "\n".join(rows)


if __name__ == "__main__":
    from sim.cache import run_or_load

    out = run_or_load("safety_filter", run, fixtures=digest())
    print(table(out))
    print(out["criteria"])
    print("missed hard-rule lines:", *[f"\n  {m['id']} [{m['split']}] {m['text']}" for m in out["missed_hard"]])
