"""Rank a list of jokes with the LLM judge (all pairs + Bradley-Terry) and print the result.

Every "which joke is funnier?" question goes to a local LLM in Ollama (http://localhost:11434),
so no API key is needed. Start the server first:  ollama serve   (and: ollama pull llama3.1:8b)

Usage (from the humor-me root):
    python -m judge.run_llm
    python -m judge.run_llm --jokes my_jokes.json --model llama3.1:8b

my_jokes.json format:
    {"prompt": "the headline", "jokes": ["joke 1", "joke 2", ...]}
"""
import argparse
import json
import math

from .judges import LLMJudge  # NEW
from .rank import rank  # NEW

EXAMPLE = {
    "prompt": "Study finds cats recognize their own names but often choose to ignore them",
    "jokes": [
        "The cats did respond instantly to one word. Researchers believe it was 'treat'. The cats believe it's their real name.",
        "In related news, my cat has read the study and declined to comment.",
        "Cats ignore their names. Cats are animals.",
        "The cats reportedly responded to the findings by knocking the paper off the table.",
        "Finally, science confirms what my cat and my teenage son have in common.",
        "Cat.",
        "Researchers also found the cats recognize the word 'study' and chose to ignore that too.",
        "The study took three years because the cats kept ignoring the consent forms.",
    ],
}


def binomial_p(k, T):
    """Two-sided exact test of 'shown-first wins with probability 1/2'."""
    probs = [math.comb(T, i) / 2 ** T for i in range(T + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] * (1 + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jokes", help="JSON file with 'prompt' and 'jokes' (default: built-in example)")
    ap.add_argument("--model", default="llama3.1:8b")  # NEW
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default="llm_judgments.jsonl")
    a = ap.parse_args()

    data = EXAMPLE
    if a.jokes:
        with open(a.jokes, encoding="utf-8") as f:
            data = json.load(f)
    prompt, jokes = data["prompt"], data["jokes"]
    n = len(jokes)
    print(f"{n} jokes -> {n * (n - 1) // 2} judge calls to {a.model}\n")

    open(a.log, "w").close()                                  # LLMJudge appends; start fresh  # NEW
    judge = LLMJudge(a.model, log_path=a.log)  # NEW
    r = rank(prompt, jokes, judge, method="all_pairs", budget=None, alpha=a.alpha, seed=a.seed)  # NEW

    print(f"Headline: {prompt}\n")
    print(" rank  fitted    se   joke")
    order = r.order()  # NEW
    for k, i in enumerate(order):  # NEW
        joke = jokes[i] if len(jokes[i]) <= 70 else jokes[i][:67] + "..."
        print(f"{k + 1:5d} {r.theta[i]:7.2f} {r.se[i]:5.2f}   {joke}")  # NEW

    best, worst = int(order[0]), int(order[-1])  # NEW
    lo, hi = r.win_prob_interval(best, worst)  # NEW
    print(f"\nBradley-Terry: P(rank 1 beats rank {n}) = {r.win_prob(best, worst):.2f}  (rough band {lo:.2f}-{hi:.2f})")  # NEW
    T = r.n_calls  # NEW
    first_wins = sum(c.shown_first == c.winner for c in r.comparisons)  # NEW
    print(f"shown-first win rate = {first_wins}/{T} = {first_wins / T:.2f}"  # NEW
          f"   (binomial p = {binomial_p(first_wins, T):.3f}; p < 0.05 means position bias)")
    print(f"every answer logged to {a.log}")


if __name__ == "__main__":
    main()  # NEW
