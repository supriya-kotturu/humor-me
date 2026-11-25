"""Exp 20: does a generated show keep its timing, and does it fit the free quota?

Usage:
    python -m sim.evaluate_live --env-file backend/.env      # full run, cached; needs Kokoro (backend venv, Python 3.12)

Whole shows with `SimRoom`, real Kokoro and the real model, played in real time. The dead air before each line is
measured from the later of the previous audio's end and the previous hold's end, split into within-bit (render-ahead
has to hide the voice) and between-bit (a bridge line has to hide generation and the filter's call).

Bars, fixed in docs/plans/live-loop.md (T9) before the first run: within-bit p95 <= 1 s; no between-bit silence over
2 s; API requests for 5 shows within the free quota. A miss is reported, not re-tuned. The key is read from the
environment and never printed or stored.
"""
import argparse
import asyncio
import hashlib
import os
import pathlib
import re
import sys
import tempfile
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
BAR_WITHIN_P95_S, BAR_BETWEEN_MAX_S = 1.0, 2.0
sys.path.insert(0, str(ROOT / "backend"))
from app import generate, llm, safety  # noqa: E402
from app.bank import BankGenerator  # noqa: E402
from app.director import load_director  # noqa: E402
from app.log import SessionLog, read_session  # noqa: E402
from app.rooms import REACTION, RealClock, SimRoom  # noqa: E402
from app import show as show_module  # noqa: E402
from app.show import UniformSelector, run_show  # noqa: E402
from app.tts import wav_seconds  # noqa: E402


class RealTimePlayer:
    """Plays in real time: sleeps the audio's length, so the show runs at the pace a room would hear."""

    async def play(self, wav: bytes, text: str) -> None:
        await asyncio.sleep(wav_seconds(wav))


def timed(client: llm.Client, calls: list[dict]) -> None:
    """Record when each call started and how long it took, retries included."""
    inner = client.chat

    def chat(*args, **kwargs):
        start = time.time()
        try:
            return inner(*args, **kwargs)
        finally:
            calls.append({"start": start, "seconds": time.time() - start})
    client.chat = chat


def silences(rows: list[dict]) -> dict:
    """Dead air before every line but the show's first, as {"within": [...], "between": [...]}, read from the log.

    Within-bit is jokes 1 to 4 of a bit. Between-bit is joke 0 of a bit and every bridge line: both sit across a bit
    boundary. The silence runs from the later of the previous audio's end and the end of the last joke's hold (its
    `gap` row). Rows are in the order things happened, so this pairs a hold with its joke by position, never by text
    (a repeated line would pair with the wrong hold) and never by comparing a timestamp with another source's (a
    coarse clock makes them equal). Bridge lines are `bridge` rows, so a real show's log answers this by itself.
    """
    out = {"within": [], "between": []}
    prev_end = last_hold = None
    for r in rows:
        if r["event"] in ("joke_played", "bridge"):
            if prev_end is not None:
                last = max(prev_end, last_hold if last_hold is not None else prev_end)
                out["within" if r["event"] == "joke_played" and r["joke"] > 0 else "between"].append(
                    max(0.0, r["audio_start_at"] - last))
            prev_end = r["audio_end_at"]
        elif r["event"] == "gap":
            last_hold = r["at"]
    return out


def bank_plays(rows: list[dict]) -> list[str]:
    """For every joke the show played from the bank, why: the bank's items say so in their reason, and `chosen`
    indexes into the joke's one `candidates` row."""
    items = {(r["bit"], r["joke"]): r["items"] for r in rows if r["event"] == "candidates"}
    played = [items[(r["bit"], r["joke"])][r["chosen"]]["reason"] for r in rows if r["event"] == "joke_played"]
    return [bank_reason(x) for x in played if x.startswith("hand-checked bank")]


def bank_reason(reason: str) -> str:
    """Why the bank was used, from a bank item's reason: 'hand-checked bank: <why> (<bank id>); rules: <verdict>'."""
    m = re.match(r"hand-checked bank: (.*) \([^()]*\); rules:", reason)
    return m.group(1) if m else reason


def one_show(tts, age: str, seed: int, n_bits: int) -> dict:
    gen, flt, fallback, bridges = generate.llm_stack("sim")
    gen_calls, safe_calls, player = [], [], RealTimePlayer()
    timed(gen.client, gen_calls)
    timed(flt.client, safe_calls)
    context = {"venue": "bar", "energy": "high", "age": age}
    with tempfile.TemporaryDirectory() as tmp:  # never data/sessions: these are not real shows
        log = SessionLog(pathlib.Path(tmp))
        started = time.time()
        asyncio.run(run_show(load_director([], seed=seed), "sim", context, generator=gen, safety=flt,
                             selector=UniformSelector(seed), tts=tts, player=player, source=SimRoom(context, seed),
                             clock=RealClock(), log=log, n_bits=n_bits, config=REACTION, fallback=fallback,
                             bridge_lines=bridges))
        rows = read_session(log.path)
    bank = bank_plays(rows)
    texts = [r["text"] for r in rows if r["event"] == "joke_played"]
    first = min(r["audio_start_at"] for r in rows if r["event"] in ("joke_played", "bridge"))
    return {"age": age, "seconds": time.time() - started, "first_line_s": first - rows[0]["at"],
            **silences(rows),
            "bridges": sum(r["event"] == "bridge" for r in rows),
            "jokes_played": sum(r["event"] == "joke_played" for r in rows),
            "bank_jokes": len(bank), "bank_reasons": sorted(set(bank)),
            "repeated_lines": len(texts) - len(set(texts)),
            "generator": {"attempts": gen.client.attempts, "retried": gen.client.retried, "failed": gen.client.failed,
                          "seconds": [c["seconds"] for c in gen_calls]},
            "filter": {"attempts": flt.client.attempts, "retried": flt.client.retried, "failed": flt.client.failed,
                       "bad_replies": flt.client.bad_replies,
                       "seconds": [c["seconds"] for c in safe_calls]},
            "request_starts": [c["start"] for c in gen_calls + safe_calls]}


def digest() -> str:
    """Part of the cache key, so a changed prompt, bridge line or timing constant recomputes the experiment."""
    parts = [generate.PROMPT_ID, generate.SYSTEM, generate.TEENS_RULE, repr(generate.BRIDGES), repr(generate.MODES),
             str((generate.K, generate.JOKES, generate.TEMPERATURE, generate.GENERATOR_TIMING, generate.FILTER_TIMING,
                  generate.RETRIES, generate.FILTER_RETRIES, show_module.MAX_BRIDGES, llm.MIN_ATTEMPT_S))]
    return hashlib.sha256(chr(10).join(parts).encode()).hexdigest()[:12]


def peak_per_minute(starts: list[float]) -> int:
    starts = sorted(starts)
    return max((sum(1 for t in starts if s <= t < s + 60) for s in starts), default=0)


def run(ages: tuple = ("mixed", "teens", "mixed"), n_bits: int = 8, seed: int = 0,
        quota_rpd: int = 0, quota_rpm: int = 0, model: str = "", config: str = "") -> dict:
    """`quota_rpd` and `quota_rpm` are the provider's free-tier requests per day and per minute for this model,
    assumed by whoever runs this; they are inputs, not measured. `model` and `config` only key the cache."""
    assert config in ("", digest()), "prompt, bridge lines or timing constants changed since this call was built"
    assert model in ("", llm.setting("JOKE_WRITER_MODEL")), "JOKE_WRITER_MODEL is not the model this call was keyed to"
    from app.tts import KokoroTTS, SerialTTS
    tts = SerialTTS(KokoroTTS())
    tts.synthesize("Warm up.")  # the first call loads weights; a real show starts with it warm
    safety.clear_cache()
    shows = [one_show(tts, age, seed + i, n_bits) for i, age in enumerate(ages)]
    within = [x for s in shows for x in s["within"]]
    between = [x for s in shows for x in s["between"]]
    requests = [s["generator"]["attempts"] + s["filter"]["attempts"] for s in shows]
    retried = sum(s["generator"]["retried"] + s["filter"]["retried"] for s in shows)
    failed = sum(s["generator"]["failed"] + s["filter"]["failed"] for s in shows)
    model = llm.from_env(role="joke_writer").model
    per_day = 5 * float(np.mean(requests))
    peak = peak_per_minute([t for s in shows for t in s["request_starts"]])
    return {"model": model, "shows": shows, "n_within": len(within), "n_between": len(between),
            "within_p50_s": float(np.percentile(within, 50)), "within_p95_s": float(np.percentile(within, 95)),
            "within_max_s": max(within), "between_p95_s": float(np.percentile(between, 95)),
            "between_max_s": max(between), "requests_per_show": requests, "requests_for_5_shows": per_day,
            "retried_requests": retried, "failed_calls": failed, "peak_requests_per_minute": peak,
            "quota": {"rpd": quota_rpd, "rpm": quota_rpm},
            "criteria": {"within_p95_pass": float(np.percentile(within, 95)) <= BAR_WITHIN_P95_S,
                         "between_pass": max(between) <= BAR_BETWEEN_MAX_S,
                         "quota_pass": bool(quota_rpd) and per_day <= quota_rpd and (not quota_rpm or peak <= quota_rpm)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-file", help="a .env holding GEMINI_BASE_URL, GEMINI_API_KEY, JOKE_WRITER_MODEL, SAFETY_CHECK_MODEL (backend/sample.env); its values are never printed")
    ap.add_argument("--quota-rpd", type=int, required=True, help="free-tier requests per day for the model, from its dashboard")
    ap.add_argument("--quota-rpm", type=int, required=True, help="free-tier requests per minute for the model")
    ap.add_argument("--rerun", action="store_true")
    a = ap.parse_args()
    if a.env_file:
        from dotenv import load_dotenv
        load_dotenv(a.env_file, override=True)
    from sim.cache import run_or_load
    out = run_or_load("live_generator", run, rerun=a.rerun, quota_rpd=a.quota_rpd, quota_rpm=a.quota_rpm,
                      model=llm.setting("JOKE_WRITER_MODEL"), config=digest())
    print({k: v for k, v in out.items() if k != "shows"})


if __name__ == "__main__":
    main()
