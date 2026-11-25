"""Exp 24: which model should write the live show's bits? Flash-Lite (the baseline), Flash, or Claude Haiku 5.5.

Usage (needs keys in backend/.env, see backend/README.md "Comparing generator models"):
    python -m sim.compare_generators run --env-file backend/.env            # generate + safety, resumable, paced
    python -m sim.compare_generators export                                  # blind ballot for the user, local only
    python -m sim.compare_generators vote                                    # rank each prompt in the terminal
    python -m sim.compare_generators score [--rerun]                         # numbers -> reports/metrics/model_comparison.json

Each prompt is one bit, generated exactly as a show does it (`generate.GeminiGenerator`, one call, 3 threads of 5
jokes, the production timeout, retries and deadline), then judged by the same safety filter for every model (Flash-Lite
through the OpenAI-compatible endpoint, as in a show). Latency, failures, safety pass rate, tokens and estimated cost
are measured; funniness is the user's blind ranking of one safe thread per model per prompt. Generated text and votes
stay under `data/model_comparison/` (gitignored); only numbers reach `reports/metrics`.

The bar is fixed in docs/plans/live-loop.md (Exp 24) before any run:
  eligible = p95 latency of successful bit generations <= the generator's deadline AND failure rate <= 5%;
  among the eligible, the best mean rank wins if the paired 95% CI of (next - best) excludes 0;
  otherwise funniness is called inconclusive and the cheaper (then faster) of the two is picked.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import pathlib
import random
import sys
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "model_comparison"
sys.path.insert(0, str(ROOT / "backend"))
from app import generate, llm, safety  # noqa: E402
from sim import cache  # noqa: E402
from sim.env import AGES, ENERGIES, VENUES  # noqa: E402

DEADLINE_S = generate.GENERATOR_TIMING[1]  # wall time a bit may take over all attempts: what the bridge lines cover
BAR_FAILURE = 0.05
N_PROMPTS = 30  # power reasoning in docs/agent/decisions-detail.md D-88
SEED = 24
N_BOOT = 10_000
BITS_PER_5_SHOWS = 40  # 8 bits x 5 shows, the demo's horizon
SAFETY_MODEL = "gemini-flash-lite-latest"  # the filter is fixed across candidates: only the generator varies
METRICS_NAME = "model_comparison"


@dataclass(frozen=True)
class Spec:
    key: str
    provider: str  # app.llm.from_env provider
    model: str
    price_in: float  # USD per million tokens
    price_out: float
    bucket: str  # requests that share a quota, so they are spaced together
    gap_s: float  # minimum seconds between the end of one request and the start of the next in the bucket


# Claude's price is from the user (D-88). The Gemini prices are ASSUMED list prices for the paid tier (the free tier
# costs $0 and is what a show uses today); `-latest` aliases move, so check them against the key's billing page before
# reading the cost column. They are part of the cache key of the scored result.
MODELS = {
    "gemini-flash-lite": Spec("gemini-flash-lite", "gemini", "gemini-flash-lite-latest", 0.10, 0.40, "gemini-lite", 7.0),
    "gemini-flash": Spec("gemini-flash", "gemini", "gemini-flash-latest", 0.30, 2.50, "gemini-flash", 7.0),
    "claude-haiku-5-5": Spec("claude-haiku-5-5", "anthropic", "claude-haiku-5-5", 0.10, 0.50, "anthropic", 1.0),
}
BASELINE = "gemini-flash-lite"
SAFETY_BUCKET = MODELS[BASELINE].bucket
SAFETY_GAP_S = MODELS[BASELINE].gap_s
MODE_ORDER = tuple(generate.MODES)


# --- prompts and cache -------------------------------------------------------------------------------------------

def prompts(n: int = N_PROMPTS, seed: int = SEED) -> list[dict]:
    """The same list for every model, and a prefix of a longer list: modes round-robin, each context drawn from its own
    seeded stream and never repeated for a mode. The first bit of a show, so `premises_already_told` is empty."""
    out, seen = [], set()
    for i in range(n):
        mode, rng = MODE_ORDER[i % len(MODE_ORDER)], random.Random(f"{seed}:{i}")
        while True:
            ctx = {"venue": rng.choice(VENUES), "energy": rng.choice(ENERGIES), "age": rng.choice(AGES)}
            if (mode, *ctx.values()) not in seen:
                break
        seen.add((mode, *ctx.values()))
        out.append({"id": f"p{i:02d}", "mode": mode, "context": ctx})
    return out


def gen_digest(seed: int = SEED) -> str:
    """Part of the cache path: editing the generator's prompt or timing starts a fresh comparison, not a mixed one."""
    parts = [generate.PROMPT_ID, generate.SYSTEM, generate.TEENS_RULE, json.dumps(generate.MODES, sort_keys=True),
             str((generate.K, generate.JOKES, generate.TEMPERATURE, generate.GENERATOR_TIMING, generate.RETRIES)),
             safety.PROMPT_ID, str(seed)]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:8]


def run_dir(digest: str | None = None, root: pathlib.Path | None = None) -> pathlib.Path:
    return (root or DATA) / f"gen-{digest or gen_digest()}"


def read_record(path: pathlib.Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def write_record(path: pathlib.Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)  # a Ctrl-C mid-write must not leave a half file that resume would trust


# --- generating --------------------------------------------------------------------------------------------------

class Pacer:
    """Spaces requests that share a quota. Waiting happens before the timed call, never inside it."""

    def __init__(self, sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        self.sleep, self.clock, self.last = sleep, clock, {}

    def wait(self, bucket: str, gap_s: float) -> None:
        if bucket in self.last and (left := self.last[bucket] + gap_s - self.clock()) > 0:
            self.sleep(left)

    def mark(self, bucket: str) -> None:
        self.last[bucket] = self.clock()


class CredentialError(RuntimeError):
    """The run stops instead of caching a pile of failures that only mean a missing or wrong key."""


def _is_auth(e: Exception) -> bool:
    return isinstance(e, llm.LLMError) and (e.status in (401, 403) or "credentials" in str(e))


def generate_bit(spec: Spec, prompt: dict, client, pacer: Pacer, now=time.perf_counter) -> dict:
    """One call as a show makes it. Latency is wall time over all attempts, retries and backoff included."""
    pacer.wait(spec.bucket, spec.gap_s)
    gen = generate.GeminiGenerator(client)
    before = (client.attempts, client.input_tokens, client.output_tokens)
    start = now()
    try:
        texts = [gen.generate(prompt["mode"], 0, j, prompt["context"]).texts for j in range(generate.JOKES)]
        threads, failure = [[texts[j][i] for j in range(generate.JOKES)] for i in range(generate.K)], None
    except Exception as e:  # the generator stores LLMError and GenerationError, and re-raises them per joke
        threads, failure = None, e
    latency = now() - start
    pacer.mark(spec.bucket)
    if failure is not None and _is_auth(failure):
        raise CredentialError(f"{spec.key}: {failure}")
    return {"model": spec.key, "prompt": prompt["id"], "latency_s": latency, "ok": failure is None,
            "reason": None if failure is None else f"{type(failure).__name__}: {failure}"[:200],
            "attempts": client.attempts - before[0], "input_tokens": client.input_tokens - before[1],
            "output_tokens": client.output_tokens - before[2], "threads": threads, "safety": None, "safety_tries": 0}


def judge_bit(rec: dict, prompt: dict, safety_client, pacer: Pacer, now=time.perf_counter) -> dict:
    """Run the safety filter over all 15 lines. `safety[i][j]` is [ok, reason] for joke j of thread i. A verdict that
    only says the filter's own call failed is infrastructure, flagged so resume retries it."""
    age = prompt["context"]["age"]
    flat = [t for thread in rec["threads"] for t in thread]
    pacer.wait(SAFETY_BUCKET, SAFETY_GAP_S)
    start = now()
    safety.prefetch(flat, age, safety_client)
    rec["safety_latency_s"] = now() - start
    pacer.mark(SAFETY_BUCKET)
    verdicts = [[list(safety.check(t, age)) for t in thread] for thread in rec["threads"]]
    rec["safety"] = verdicts
    rec["safety_infra_error"] = any(r.startswith("llm:error") for thread in verdicts for ok, r in thread if not ok)
    rec["safety_tries"] += 1
    return rec


MAX_SAFETY_TRIES = 3


def needs_safety(rec: dict) -> bool:
    return rec["ok"] and (rec["safety"] is None or (rec.get("safety_infra_error") and rec["safety_tries"] < MAX_SAFETY_TRIES))


def run_comparison(n: int, models: list[str], make_client: Callable[[Spec], object], safety_client, *,
                   redo_failed: bool = False, directory: pathlib.Path | None = None, pacer: Pacer | None = None,
                   log: Callable[[str], None] = print) -> dict:
    """Prompt-major, so an interrupted run leaves every model with the same prompts. Returns counts of what was done."""
    directory, pacer = directory or run_dir(), pacer or Pacer()
    clients = {m: make_client(MODELS[m]) for m in models}
    done = {"generated": 0, "judged": 0, "cached": 0}
    for prompt in prompts(n):
        for m in models:
            path = directory / m / f"{prompt['id']}.json"
            rec = read_record(path)
            if rec is not None and not rec["ok"] and redo_failed:
                rec = None
            if rec is None:
                rec = generate_bit(MODELS[m], prompt, clients[m], pacer)
                done["generated"] += 1
                write_record(path, rec)
            else:
                done["cached"] += 1
            if needs_safety(rec):
                judge_bit(rec, prompt, safety_client, pacer)
                done["judged"] += 1
                write_record(path, rec)
            log(f"{prompt['id']} {m}: {'ok' if rec['ok'] else 'FAILED ' + str(rec['reason'])} {rec['latency_s']:.1f}s")
    return done


def make_client(spec: Spec):
    """The production generator's timing and retries, so latency and failures are what a show would see."""
    timeout, deadline = generate.GENERATOR_TIMING
    return llm.from_env(timeout, provider=spec.provider, model=spec.model, retries=generate.RETRIES, deadline_s=deadline)


def make_safety_client():
    timeout, deadline = generate.FILTER_TIMING
    return llm.from_env(timeout, provider="gemini", role="safety_check", model=SAFETY_MODEL, retries=generate.RETRIES, deadline_s=deadline)


# --- blind ballot ------------------------------------------------------------------------------------------------

LETTERS = "ABC"


def playable_thread(rec: dict | None) -> int | None:
    """The first thread whose five lines all passed the filter, with the filter's own calls not failing: the thread a
    show would play unchanged. None when the model has no such thread for this prompt."""
    if not rec or not rec["ok"] or rec["safety"] is None or rec.get("safety_infra_error"):
        return None
    return next((i for i, thread in enumerate(rec["safety"]) if all(ok for ok, _ in thread)), None)


def read_votes(path: pathlib.Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {r["prompt"]: r["ranking"].strip().upper() for r in csv.DictReader(f)}


def write_votes(path: pathlib.Path, votes: dict[str, str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt", "ranking"])
        w.writerows(sorted(votes.items()))


def export_ballot(models: list[str], n: int, directory: pathlib.Path | None = None, seed: int = SEED) -> dict:
    """Writes ballot.md, key.json and votes.csv under the run directory. Incremental: prompts already in the key keep
    their letters and votes, so raising `n` later only adds. Prompts where any model lacks a playable thread are
    dropped and listed, since a vote needs all of them."""
    directory = directory or run_dir()
    directory.mkdir(parents=True, exist_ok=True)
    key_path, votes_path = directory / "key.json", directory / "votes.csv"
    key = json.loads(key_path.read_text(encoding="utf-8")) if key_path.exists() else {}
    votes, dropped = read_votes(votes_path), {}
    by_id = {p["id"]: p for p in prompts(n, seed)}
    for pid, prompt in by_id.items():
        if pid in key:
            continue
        picks = {m: playable_thread(read_record(directory / m / f"{pid}.json")) for m in models}
        if any(p is None for p in picks.values()):
            dropped[pid] = [m for m, p in picks.items() if p is None]
            continue
        order = list(models)
        random.Random(f"{seed}:ballot:{pid}").shuffle(order)
        key[pid] = {"letters": dict(zip(LETTERS, order)), "threads": picks}
        votes.setdefault(pid, "")
    lines = ["# Blind ballot: rank the options for each setting, funniest first (e.g. BAC)\n"]
    for pid in sorted(key):
        ctx = by_id[pid]["context"]
        lines.append(f"\n## {pid}: {ctx['venue']}, {ctx['energy']} energy, {ctx['age']} audience\n")
        for letter in LETTERS:
            m = key[pid]["letters"][letter]
            thread = read_record(directory / m / f"{pid}.json")["threads"][key[pid]["threads"][m]]
            lines.append(f"\n**{letter}**\n" + "\n".join(f"{j + 1}. {t}" for j, t in enumerate(thread)) + "\n")
    key_path.write_text(json.dumps(key, indent=1, sort_keys=True), encoding="utf-8")
    (directory / "ballot.md").write_text("".join(lines), encoding="utf-8")
    write_votes(votes_path, votes)
    return {"in_ballot": len(key), "dropped": dropped}


def parse_ranking(text: str) -> str | None:
    """'bac' -> 'BAC' when it names each letter once; anything else is None."""
    s = "".join(c for c in text.upper() if c.isalpha())
    return s if sorted(s) == sorted(LETTERS) else None


def vote_interactively(directory: pathlib.Path | None = None, read=input, out=print) -> int:
    """Shows each unranked prompt and saves after every answer, so quitting loses nothing."""
    directory = directory or run_dir()
    blocks, current = {}, None
    for line in (directory / "ballot.md").read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("## "):
            current = line[3:].split(":")[0]
        if current:
            blocks[current] = blocks.get(current, "") + line
    votes_path, count = directory / "votes.csv", 0
    votes = read_votes(votes_path)
    todo = [p for p, r in sorted(votes.items()) if not parse_ranking(r)]
    for i, pid in enumerate(todo, 1):
        out(f"\n[{i}/{len(todo)}]\n{blocks[pid]}")
        while True:
            answer = read("ranking, best first (e.g. BAC), s to skip, q to quit: ").strip().lower()
            if answer in ("s", "q") or parse_ranking(answer):
                break
            out("use each of A, B, C once")
        if answer == "q":
            break
        if answer != "s":
            votes[pid] = parse_ranking(answer)
            write_votes(votes_path, votes)
            count += 1
    return count


# --- scoring -----------------------------------------------------------------------------------------------------

def percentile(xs: list[float], q: float) -> float | None:
    return float(np.percentile(xs, q)) if xs else None


def call_stats(records: list[dict], spec: Spec) -> dict:
    """Per-model numbers from its saved calls. Latency is over successful calls (a failure is counted as a failure, not
    also as a slow success). Cost covers every call's tokens, failed ones included, since a provider bills them."""
    ok = [r for r in records if r["ok"]]
    judged = [r for r in ok if r["safety"] is not None and not r.get("safety_infra_error")]
    lines = [v for r in judged for thread in r["safety"] for v in thread]
    threads = [t for r in judged for t in r["safety"]]
    tin, tout = sum(r["input_tokens"] for r in records), sum(r["output_tokens"] for r in records)
    cost = (tin * spec.price_in + tout * spec.price_out) / 1e6
    n = len(records)
    reasons: dict[str, int] = {}
    for r in records:
        if not r["ok"]:
            reasons[r["reason"][:60]] = reasons.get(r["reason"][:60], 0) + 1
    return {"model": spec.model, "n_calls": n, "n_failed": n - len(ok), "failure_rate": (n - len(ok)) / n if n else None,
            "failure_reasons": reasons, "latency_p50_s": percentile([r["latency_s"] for r in ok], 50),
            "latency_p95_s": percentile([r["latency_s"] for r in ok], 95),
            "latency_max_s": max((r["latency_s"] for r in ok), default=None),
            "attempts_per_call": sum(r["attempts"] for r in records) / n if n else None,
            "n_judged": len(judged), "line_pass_rate": sum(v[0] for v in lines) / len(lines) if lines else None,
            "thread_pass_rate": sum(all(v[0] for v in t) for t in threads) / len(threads) if threads else None,
            "playable_rate": (sum(any(all(v[0] for v in t) for t in r["safety"]) for r in judged) / len(judged)
                              if judged else None),
            "input_tokens_per_call": tin / n if n else None, "output_tokens_per_call": tout / n if n else None,
            "price_per_mtok": [spec.price_in, spec.price_out],
            "cost_per_bit_usd": cost / n if n else None,
            "cost_5_shows_usd": cost / n * BITS_PER_5_SHOWS if n else None}


def boot_idx(n_items: int, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """One resampling of prompts shared by every statistic, so models and gaps are bootstrapped on the same draws."""
    return np.random.default_rng(seed).integers(0, n_items, (n_boot, n_items))


def boot_ci(x: np.ndarray, idx: np.ndarray) -> list[float]:
    means = x[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def ranks_from_votes(votes: dict[str, str], key: dict, models: list[str]) -> dict[str, dict[str, int]]:
    """{prompt: {model: rank}} (1 = funniest) for prompts with a valid ranking and a key."""
    out = {}
    for pid, ranking in votes.items():
        if pid in key and parse_ranking(ranking):
            letters = key[pid]["letters"]
            r = {letters[c]: i for i, c in enumerate(parse_ranking(ranking), 1) if letters[c] in models}
            if len(r) == len(models):
                out[pid] = r
    return out


def restrict(ranks: dict[str, dict[str, int]], keep: list[str]) -> dict[str, dict[str, int]]:
    """Re-rank each prompt among the models in `keep`, order preserved."""
    return {pid: {m: i for i, m in enumerate(sorted(keep, key=lambda m: r[m]), 1)} for pid, r in ranks.items()}


def rank_stats(ranks: dict[str, dict[str, int]], models: list[str], n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    pids = sorted(ranks)
    if not pids:
        return {"n": 0, "mean_rank": {}, "ci95": {}, "diff": {}}
    idx = boot_idx(len(pids), n_boot, seed)
    arr = {m: np.array([ranks[p][m] for p in pids], float) for m in models}
    return {"n": len(pids), "mean_rank": {m: float(arr[m].mean()) for m in models},
            "ci95": {m: boot_ci(arr[m], idx) for m in models},
            "diff": {f"{a}|{b}": {"mean": float((arr[a] - arr[b]).mean()), "ci95": boot_ci(arr[a] - arr[b], idx)}
                     for a in models for b in models if a < b}}


def diff_ci(ranks: dict[str, dict[str, int]], worse: str, better: str, n_boot: int = N_BOOT, seed: int = SEED):
    """Paired bootstrap over prompts of mean(rank worse - rank better): positive means `better` is funnier."""
    pids = sorted(ranks)
    d = np.array([ranks[p][worse] - ranks[p][better] for p in pids], float)
    return float(d.mean()), boot_ci(d, boot_idx(len(pids), n_boot, seed))


def eligibility(stats: dict, deadline_s: float = DEADLINE_S, bar_failure: float = BAR_FAILURE) -> dict[str, dict]:
    """Per model: the two bar terms and whether both hold. A model with no successful call is not eligible."""
    out = {}
    for m, s in stats.items():
        p95, fr = s["latency_p95_s"], s["failure_rate"]
        out[m] = {"latency_ok": p95 is not None and p95 <= deadline_s, "failure_ok": fr is not None and fr <= bar_failure}
        out[m]["eligible"] = out[m]["latency_ok"] and out[m]["failure_ok"]
    return out


def decide(stats: dict, ranks: dict[str, dict[str, int]], n_boot: int = N_BOOT, seed: int = SEED,
           deadline_s: float = DEADLINE_S, bar_failure: float = BAR_FAILURE) -> dict:
    """The pre-fixed bar. Ranks are redone among the eligible models before comparing them."""
    elig = eligibility(stats, deadline_s, bar_failure)
    ok = sorted(m for m, e in elig.items() if e["eligible"])
    out = {"eligibility": elig, "eligible": ok, "pick": None, "best": None, "second": None, "funniness_clear": None,
           "diff": None, "reason": ""}
    if not ok:
        out["reason"] = "no model meets the latency and failure bar; the generator stays as it is"
        return out
    if len(ok) == 1:
        out.update(pick=ok[0], reason="the only eligible model")
        return out
    if not ranks:
        out["reason"] = "waiting for the user's votes"
        return out
    re_ranked = restrict(ranks, ok)
    mean = {m: float(np.mean([r[m] for r in re_ranked.values()])) for m in ok}
    best, second = sorted(ok, key=lambda m: (mean[m], m))[:2]
    d, ci = diff_ci(re_ranked, second, best, n_boot, seed)
    clear = ci[0] > 0
    out.update(best=best, second=second, diff={"mean": d, "ci95": ci}, funniness_clear=clear, mean_rank_eligible=mean)
    if clear:
        out.update(pick=best, reason=f"{best} is funnier than {second}: the paired CI of the rank gap excludes 0")
    else:
        cheaper = min((best, second), key=lambda m: (stats[m]["cost_per_bit_usd"], stats[m]["latency_p50_s"], m))
        out.update(pick=cheaper, reason=f"funniness is inconclusive between {best} and {second} (the paired CI "
                                        "includes 0), so the cheaper, then faster, of the two")
    return out


# --- power -------------------------------------------------------------------------------------------------------

RANK_SD_NO_SIGNAL = math.sqrt(2 / 3)  # sd of one model's rank among 3 when every order is equally likely


def halfwidth(n: int, sd: float = RANK_SD_NO_SIGNAL) -> float:
    """Half-width of a 95% interval for one model's mean rank over n prompts (normal approximation)."""
    return 1.96 * sd / math.sqrt(n)


def simulate_gap_power(n: int, gap: float, reps: int = 4000, seed: int = SEED) -> float:
    """P(the paired 95% interval of (next - best) excludes 0) when the best model sits `gap` latent sd above the other
    two (Thurstone: score = mean + N(0, 1) per model per prompt; a t interval stands in for the bootstrap)."""
    rng = np.random.default_rng(seed)
    mu = np.array([gap, 0.0, 0.0])
    hits = 0
    for _ in range(reps):
        ranks = np.argsort(np.argsort(-(mu + rng.standard_normal((n, 3))), axis=1), axis=1) + 1
        d = ranks[:, 1] - ranks[:, 0]  # second model minus best
        hits += d.mean() - 1.96 * d.std(ddof=1) / math.sqrt(n) > 0
    return hits / reps


def expected_rank_gap(gap: float, reps: int = 200_000, seed: int = SEED) -> float:
    """Mean rank of a model at the others' level minus the best's, under the same model."""
    rng = np.random.default_rng(seed)
    ranks = np.argsort(np.argsort(-(np.array([gap, 0.0, 0.0]) + rng.standard_normal((reps, 3))), axis=1), axis=1) + 1
    return float(ranks[:, 1].mean() - ranks[:, 0].mean())


# --- the experiment ----------------------------------------------------------------------------------------------

def load_records(directory: pathlib.Path, models: list[str], n: int) -> dict[str, list[dict]]:
    return {m: [r for p in prompts(n) if (r := read_record(directory / m / f"{p['id']}.json"))] for m in models}


def votes_digest(directory: pathlib.Path) -> str:
    """The votes and key, never the text: a changed vote recomputes the experiment."""
    body = b"".join((directory / name).read_bytes() for name in ("key.json", "votes.csv") if (directory / name).exists())
    return hashlib.sha256(body.replace(b"\r\n", b"\n")).hexdigest()[:12]


def current_params(n: int = N_PROMPTS, models: tuple[str, ...] = tuple(MODELS)) -> dict:
    return {"gen": gen_digest(), "votes": votes_digest(run_dir()), "n_prompts": n, "models": list(models), "seed": SEED,
            "n_boot": N_BOOT, "deadline_s": DEADLINE_S, "bar_failure": BAR_FAILURE,
            "prices": {m: [MODELS[m].price_in, MODELS[m].price_out] for m in models}}


def run(gen: str = "", votes: str = "", n_prompts: int = N_PROMPTS, models: list[str] | None = None, seed: int = SEED,
        n_boot: int = N_BOOT, deadline_s: float = DEADLINE_S, bar_failure: float = BAR_FAILURE,
        prices: dict | None = None) -> dict:
    """Scores the saved generations and votes. Numbers only: no joke text and no letters leave the data directory."""
    models = models or list(MODELS)
    assert gen == gen_digest(seed), "the generator prompt or timing changed since these calls were made"
    directory = run_dir(gen)
    assert votes == votes_digest(directory), "votes or key changed since this call was built"
    records = load_records(directory, models, n_prompts)
    stats = {m: call_stats(records[m], MODELS[m]) for m in models}
    key_path = directory / "key.json"
    key = json.loads(key_path.read_text(encoding="utf-8")) if key_path.exists() else {}
    ranks = ranks_from_votes(read_votes(directory / "votes.csv"), key, models)
    by_id = {p["id"]: p for p in prompts(n_prompts, seed)}
    return {"models": stats, "n_prompts": n_prompts, "n_ballot": len(key), "n_voted": len(ranks),
            "votes": rank_stats(ranks, models, n_boot, seed),
            "per_prompt": [{"prompt": p, "mode": by_id[p]["mode"], "ranks": ranks[p]} for p in sorted(ranks)],
            "decision": decide(stats, ranks, n_boot, seed, deadline_s, bar_failure)}


def stored() -> dict | None:
    path = cache.METRICS / f"{METRICS_NAME}.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else None


def fmt(x, spec="{:.2f}", none="-"):
    return none if x is None else spec.format(x)


def table(res: dict) -> str:
    rows = ["| Model | Calls | Failed | p50 | p95 | Line pass | Playable | Mean rank (95% CI) | Cost, 5 shows | Eligible |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    mean, ci = res["votes"].get("mean_rank", {}), res["votes"].get("ci95", {})
    for m, s in res["models"].items():
        rank = f"{mean[m]:.2f} ({ci[m][0]:.2f} to {ci[m][1]:.2f})" if m in mean else "-"
        rows.append(f"| {m} | {s['n_calls']} | {s['n_failed']} | {fmt(s['latency_p50_s'], '{:.1f} s')} | "
                    f"{fmt(s['latency_p95_s'], '{:.1f} s')} | {fmt(s['line_pass_rate'], '{:.0%}')} | "
                    f"{fmt(s['playable_rate'], '{:.0%}')} | {rank} | {fmt(s['cost_5_shows_usd'], '${:.3f}')} | "
                    f"{'yes' if res['decision']['eligibility'][m]['eligible'] else 'no'} |")
    return "\n".join(rows)


def verdict(res: dict) -> str:
    d = res["decision"]
    return f"Pick: {d['pick'] or 'none'}. {d['reason']}."


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=["run", "export", "vote", "score"])
    ap.add_argument("--env-file")
    ap.add_argument("--n", type=int, default=N_PROMPTS)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--redo-failed", action="store_true", help="drop cached failed calls and call again")
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    if a.env_file:
        from dotenv import load_dotenv
        load_dotenv(a.env_file, override=True)
    models = a.models.split(",")
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        sys.exit(f"unknown models {unknown}; choose from {list(MODELS)}")
    if a.command == "run":
        if "claude-haiku-5-5" in models and not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set: add a paid key to backend/.env (or pass --models without it)")
        try:
            print(run_comparison(a.n, models, make_client, make_safety_client(), redo_failed=a.redo_failed))
        except CredentialError as e:
            sys.exit(f"stopped, nothing cached for the failing call: {e}")
    elif a.command == "export":
        print(export_ballot(models, a.n))
        print(f"ballot: {run_dir() / 'ballot.md'}  votes: {run_dir() / 'votes.csv'}")
    elif a.command == "vote":
        print(f"{vote_interactively()} votes saved to {run_dir() / 'votes.csv'}")
    else:
        res = cache.run_or_load(METRICS_NAME, run, rerun=a.rerun, **current_params(a.n, tuple(models)))
        print(table(res))
        print(verdict(res))


if __name__ == "__main__":
    main()
