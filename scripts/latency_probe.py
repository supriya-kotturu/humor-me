"""Measure LLM time-to-first-token and total time for a one-joke request.

Comedy timing budget is ~1s from laugh decay to next line, so TTFT is the number that
decides providers. Works with any OpenAI-compatible endpoint (see backend/.env.example).

Usage:
    python scripts/latency_probe.py [--runs 5]
"""
import argparse
import json
import os
import statistics
import time

import httpx
from dotenv import load_dotenv

PROMPT = (
    "You are a self-aware, slightly anxious robot doing stand-up at a project showcase. "
    "Give one short opening line about being on display. One or two sentences, no preamble."
)


def probe(client: httpx.Client, base: str, key: str, model: str) -> tuple[float, float, str]:
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    body = {"model": model, "stream": True, "messages": [{"role": "user", "content": PROMPT}]}
    start = time.perf_counter()
    ttft, text = None, []
    with client.stream("POST", f"{base}/chat/completions", json=body, headers=headers) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choices = json.loads(line[6:]).get("choices") or []
            delta = choices[0].get("delta", {}).get("content") if choices else None
            if delta:
                ttft = ttft or time.perf_counter() - start
                text.append(delta)
    return ttft or float("nan"), time.perf_counter() - start, "".join(text)


def main() -> None:
    load_dotenv("backend/.env")
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    args = ap.parse_args()
    # role names first (backend/sample.env), then the pre-rename ones
    base = os.getenv("GEMINI_BASE_URL") or os.environ["LLM_BASE_URL"]
    key = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY", "")
    model = os.getenv("JOKE_WRITER_MODEL") or os.environ["LLM_MODEL"]
    if model.startswith("claude") or os.getenv("JOKE_WRITER_PROVIDER") == "anthropic":
        # This probe speaks the OpenAI-compatible protocol; Claude is not on that endpoint.
        raise SystemExit(f"{model}: Claude is not reachable through GEMINI_BASE_URL. Probe a Gemini model, e.g. "
                         "JOKE_WRITER_MODEL=gemini-flash-lite-latest, or time Claude with `sim.compare_generators`.")

    ttfts, totals = [], []
    with httpx.Client(timeout=60) as client:
        for i in range(args.runs):
            ttft, total, text = probe(client, base, key, model)
            ttfts.append(ttft)
            totals.append(total)
            print(f"run {i}: ttft={ttft:.3f}s total={total:.3f}s | {text.strip()[:100]}")
    # First run includes connection setup / cold model load; report it but exclude from median.
    warm = ttfts[1:] or ttfts
    print(f"\n{model} @ {base}\nmedian warm TTFT={statistics.median(warm):.3f}s "
          f"median total={statistics.median(totals[1:] or totals):.3f}s")


if __name__ == "__main__":
    main()
