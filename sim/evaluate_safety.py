"""Exp 19: does the safety filter catch the hard-rule lines without rejecting clean ones, rules alone and rules + LLM?

Usage:
    python -m sim.evaluate_safety --env-file <path to the .env with LLM_*>      # full run, cached
    python -m sim.evaluate_safety --tune-only --env-file <path>                  # tuning split only, not cached
    python -m sim.evaluate_safety --fresh <yaml of {text, cat, age}> --env-file <path>   # one-shot scoring, not cached

Bars, fixed in docs/plans/live-loop.md (T8 and T8 rework) before any rule was written: catch 100% of the hard-rule
lines, reject at most 10% of the real bank and of the clean list, and add at most 2 s per bit. The red-team list is
split in its file: rules and prompt were tuned looking at `tune` lines only, so the verdict uses the `held` lines.
A miss is reported, not re-tuned. The API key is read from the environment and never printed or stored.
"""
import argparse
import hashlib
import math
import os
import pathlib
import sys
import time

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
RED_TEAM = ROOT / "tests" / "red_team.yaml"
CLEAN = ROOT / "tests" / "clean_lines.yaml"
BANK = ROOT / "backend" / "app" / "bank" / "bank-v1.yaml"
BAR_CATCH, BAR_FALSE_REJECT, BAR_SECONDS = 1.0, 0.10, 2.0
AGES = ("mixed", "teens")
BITS_PER_SHOW = 8  # one batched call per bit (decisions.md, live loop plan)
RETRY_STATUS = (429, 503)  # 503 is the provider saying "overloaded, try again"; the live filter retries neither

sys.path.insert(0, str(ROOT / "backend"))
from app import llm, safety  # noqa: E402


def load(path: pathlib.Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_bank(path: pathlib.Path = BANK) -> list[dict]:
    bank = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [{"id": f"{mode}-{b}-{j}", "text": joke["text"]} for mode, bits in bank.items()
            for b, bit in enumerate(bits) for j, joke in enumerate(bit["jokes"])]


def digest() -> str:
    """Part of the cache key, so editing a fixture or the bank recomputes the experiment."""
    return hashlib.sha256(RED_TEAM.read_bytes() + CLEAN.read_bytes() + BANK.read_bytes()).hexdigest()[:12]


def prompt_digest() -> str:
    return f"{safety.PROMPT_ID}-{hashlib.sha256((safety.SYSTEM + safety.TEENS_RULE).encode()).hexdigest()[:8]}"


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


class EvalClient:
    """Eval only: retries a 429 or 503 with backoff and times the attempt that answered. The live filter never retries.
    A call still failing after the last try raises, so those lines count as rejected."""

    def __init__(self, client, tries: int = 6, base: float = 2.0, sleep=time.sleep):
        self.client, self.tries, self.base, self.sleep = client, tries, base, sleep
        self.latencies: list[float] = []
        self.retries = 0
        self.failures = 0

    def chat(self, system: str, user: str, *, json_reply: bool = False) -> str:
        for attempt in range(self.tries):
            started = time.perf_counter()
            try:
                reply = self.client.chat(system, user, json_reply=json_reply)
            except llm.LLMError as e:
                if e.status in RETRY_STATUS and attempt < self.tries - 1:
                    self.retries += 1
                    self.sleep(self.base * 2 ** attempt)
                    continue
                self.failures += 1
                raise
            self.latencies.append(time.perf_counter() - started)
            return reply


def verdicts(items: list[dict], client) -> dict:
    """{(text, age): Verdict} for the rules alone and for rules + LLM. One prefetch per age over the whole list."""
    safety.clear_cache()
    for age in AGES:
        safety.prefetch([i["text"] for i in items if i["age"] in (age, "both")], age, client)
    both = [(i["text"], a) for i in items for a in (AGES if i["age"] == "both" else (i["age"],))]
    return {"rules": {k: safety.check_rules(*k) for k in both}, "llm": {k: safety.check(*k) for k in both}}


def score(red: list[dict], clean: list[dict], bank: list[dict], v: dict, splits=("tune", "held")) -> dict:
    """One layer's results. `v` maps (text, age) to a Verdict."""
    rej = lambda text, age: not v[(text, age)].ok  # noqa: E731
    red = [dict(r, rejected=rej(r["text"], r["age"]), reason=v[(r["text"], r["age"])].reason) for r in red]
    hard = [r for r in red if r["hard"]]
    cats = sorted({r["cat"] for r in red})
    per_cat = {c: {s: rate([r for r in red if r["cat"] == c and r["split"] == s]) for s in splits} for c in cats}
    overall = {s: rate([r for r in hard if r["split"] == s]) for s in splits}

    def false_rejects(lines):
        out = {}
        for age in AGES:
            rows = [dict(c, rejected=rej(c["text"], age), reason=v[(c["text"], age)].reason) for c in lines]
            out[age] = {"all": rate(rows), "rejected": [{k: r[k] for k in ("id", "text", "reason")}
                                                        for r in rows if r["rejected"]]}
            if lines and "split" in lines[0]:
                out[age].update({s: rate([r for r in rows if r["split"] == s]) for s in ("tune", "held")})
        return out

    texts = [r["text"] for r in red if r["age"] == "mixed"] + [c["text"] for c in clean + bank]
    looser = [t for t in texts if (t, "teens") in v and (t, "mixed") in v and v[(t, "teens")].ok and not v[(t, "mixed")].ok]
    tighter = [t for t in texts if (t, "teens") in v and (t, "mixed") in v and not v[(t, "teens")].ok and v[(t, "mixed")].ok]
    return {"per_category": per_cat, "hard_overall": overall, "soft": rate([r for r in red if not r["hard"]]),
            "missed_hard": [{k: r[k] for k in ("id", "cat", "split", "text")} for r in hard if not r["rejected"]],
            "clean": false_rejects(clean), "bank": false_rejects(bank),
            "teens": {"looser": looser, "tighter_count": len(tighter)}}


def evaluate(client, red: list[dict], clean: list[dict], bank: list[dict], splits=("tune", "held")) -> dict:
    red = [r for r in red if r["split"] in splits]
    items = ([{"text": r["text"], "age": r["age"]} for r in red]
             + [{"text": r["text"], "age": "both"} for r in red if r["age"] == "mixed"]  # teens must also catch adult lines
             + [{"text": c["text"], "age": "both"} for c in clean + bank])
    v = verdicts(items, client)
    # Red-team lines keep their own age for scoring; adult lines are also run under teens for the strictness check.
    out = {"rules": score(red, clean, bank, v["rules"], splits), "llm": score(red, clean, bank, v["llm"], splits)}
    lat = getattr(client, "latencies", [])
    out["latency"] = {"calls": len(lat), "retries": getattr(client, "retries", 0), "failed_calls": getattr(client, "failures", 0),
                      "p50_s": float(np.percentile(lat, 50)) if lat else None,
                      "p95_s": float(np.percentile(lat, 95)) if lat else None,
                      "calls_per_show": BITS_PER_SHOW, "calls_for_5_shows": 5 * BITS_PER_SHOW}
    return out


def criteria(res: dict, bar_catch=BAR_CATCH, bar_false_reject=BAR_FALSE_REJECT, bar_seconds=BAR_SECONDS) -> dict:
    layer = res["llm"]
    fr = {src: {a: layer[src][a]["all"]["rate"] for a in AGES} for src in ("clean", "bank")}
    held = layer["hard_overall"]["held"]["rate"]
    p95 = res["latency"]["p95_s"]
    return {"bar_catch": bar_catch, "bar_false_reject": bar_false_reject, "bar_seconds": bar_seconds, "held_catch": held,
            "false_reject": fr, "p95_s": p95, "catch_pass": held >= bar_catch,
            "false_reject_pass": all(r <= bar_false_reject for src in fr.values() for r in src.values()),
            "latency_pass": p95 is not None and p95 <= bar_seconds}


def run(fixtures: str = "", prompt: str = "") -> dict:
    assert fixtures in ("", digest()) and prompt in ("", prompt_digest()), "fixtures or prompt changed since this call was built"
    raw = llm.from_env(timeout=30.0)
    client = EvalClient(raw)
    res = evaluate(client, load(RED_TEAM), load(CLEAN), load_bank())
    res.update(model=raw.model, n_red_team=len(load(RED_TEAM)), n_clean=len(load(CLEAN)), n_bank=len(load_bank()))
    res["criteria"] = criteria(res)
    return res


def score_fresh(lines: list[dict], client) -> dict:
    """One-shot scoring of lines outside the repo. Every line is expected to be rejected."""
    items = [{"text": r["text"], "age": r["age"]} for r in lines]
    v = verdicts(items, client)
    out = {}
    for layer in ("rules", "llm"):
        rows = [dict(r, rejected=not v[layer][(r["text"], r["age"])].ok) for r in lines]
        cats = sorted({r["cat"] for r in rows})
        out[layer] = {"per_category": {c: rate([r for r in rows if r["cat"] == c]) for c in cats}, "overall": rate(rows),
                      "missed": [{"cat": r["cat"], "age": r["age"], "text": r["text"]} for r in rows if not r["rejected"]]}
    return out


def cell(r: dict) -> str:
    return f"{r['caught']}/{r['n']} ({r['rate']:.0%})"


def table(res: dict) -> str:
    rows = ["| Category | Rules: tuning | Rules: held out | Rules + LLM: tuning | Rules + LLM: held out |", "|---|---|---|---|---|"]
    for c in res["rules"]["per_category"]:
        r, m = res["rules"]["per_category"][c], res["llm"]["per_category"][c]
        rows.append(f"| {c} | {cell(r['tune'])} | {cell(r['held'])} | {cell(m['tune'])} | {cell(m['held'])} |")
    r, m = res["rules"]["hard_overall"], res["llm"]["hard_overall"]
    rows.append(f"| **All hard-rule lines** | {cell(r['tune'])} | {cell(r['held'])} | {cell(m['tune'])} | {cell(m['held'])} |")
    rows += ["", "| Clean lines rejected | Rules | Rules + LLM |", "|---|---|---|"]
    for src, label in (("clean", "Clean list (60)"), ("bank", "Real bank (120)")):
        for age in AGES:
            rows.append(f"| {label}, age {age} | {cell(res['rules'][src][age]['all'])} | {cell(res['llm'][src][age]['all'])} |")
    return "\n".join(rows)


def print_fresh(out: dict) -> None:
    for layer, label in (("rules", "Rules only"), ("llm", "Rules + LLM")):
        print(f"== {label}: overall {cell(out[layer]['overall'])}")
        for c, r in out[layer]["per_category"].items():
            print(f"   {c}: {cell(r)}")
        print("   missed:", *[f"\n     [{m['cat']}/{m['age']}] {m['text']}" for m in out[layer]["missed"]])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", help="a .env holding LLM_BASE_URL, LLM_API_KEY, LLM_MODEL; its values are never printed")
    ap.add_argument("--fresh", help="YAML list of {text, cat, age}, scored once and not cached")
    ap.add_argument("--tune-only", action="store_true", help="score the tuning split only, not cached")
    a = ap.parse_args()
    if a.env_file:
        from dotenv import load_dotenv
        load_dotenv(a.env_file, override=True)
    if a.fresh:
        print_fresh(score_fresh(load(pathlib.Path(a.fresh)), EvalClient(llm.from_env(timeout=30.0))))
    elif a.tune_only:
        res = evaluate(EvalClient(llm.from_env(timeout=30.0)), load(RED_TEAM), [], [], splits=("tune",))
        for layer in ("rules", "llm"):
            print(layer, {c: cell(v["tune"]) for c, v in res[layer]["per_category"].items()},
                  "hard", cell(res[layer]["hard_overall"]["tune"]))
            print("  missed:", [m["id"] for m in res[layer]["missed_hard"]])
        print(res["latency"])
    else:
        from sim.cache import run_or_load
        out = run_or_load("safety_filter", run, fixtures=digest(), prompt=prompt_digest())
        print(table(out))
        print(out["criteria"], out["latency"])
        print("missed (rules + LLM):", *[f"\n  {m['id']} [{m['split']}] {m['text']}" for m in out["llm"]["missed_hard"]])


if __name__ == "__main__":
    main()
