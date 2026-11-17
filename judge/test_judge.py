"""
LLM-judge test for the judge ranking (all-pairs + Bradley–Terry).

Every "which joke is funnier?" question is answered by a local LLM
running in Ollama (http://localhost:11434), so no API key is needed.
Start the server first:  ollama serve

Usage (from the humor-me root, with Diane active):
    python -m judge.test_judge
    python -m judge.test_judge --jokes my_jokes.json --model deepseek-r1:latest

my_jokes.json format:
    {"prompt": "the headline", "jokes": ["joke 1", "joke 2", ...]}
"""
import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

import numpy as np


# rank / fit_bt / bt_se live in this file (judge/functions.py has no rank)
def rank(prompt, jokes, judge, alpha=1.0, seed=0):
    n = len(jokes)
    rng = np.random.default_rng(seed)
    W = np.zeros((n, n))                      # W[i, j] = wins of i over j      (1.1)
    shown_first_wins = 0

    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for k in rng.permutation(len(pairs)):
        i, j = pairs[k]
        if rng.random() < 0.5:                # coin flip: who is shown first
            i, j = j, i
        first_won = judge.compare(prompt, jokes[i], jokes[j]) == 0
        shown_first_wins += first_won
        winner, loser = (i, j) if first_won else (j, i)
        W[winner, loser] += 1

    theta = fit_bt(W, alpha)                  # uncentered: dummy sits at 0
    se = bt_se(theta, W + W.T, alpha)
    theta = theta - theta.mean()              # centering, for display only
    return theta, se, np.argsort(-theta), shown_first_wins, len(pairs)


def fit_bt(W, alpha, tol=1e-8, max_iter=1000):
    N = W + W.T                               # n_ij
    wins = W.sum(axis=1) + alpha              # W_i + alpha
    pi = np.ones(len(W))
    for _ in range(max_iter):
        D = (N / (pi[:, None] + pi[None, :])).sum(axis=1) + 2 * alpha / (pi + 1)  # (5.8)
        new = wins / D                                                             # (5.9)
        done = np.max(np.abs(np.log(new / pi))) < tol
        pi = new
        if done:
            break
    return np.log(pi)


def bt_se(theta, N, alpha):
    P = 1 / (1 + np.exp(theta[None, :] - theta[:, None]))  # p_ij = sigma(theta_i - theta_j)
    V = N * P * (1 - P)
    p0 = 1 / (1 + np.exp(-theta))                           # p_i0 = sigma(theta_i)
    I = -V                                                  # off-diagonal of -H   (6.5)
    np.fill_diagonal(I, V.sum(axis=1) + 2 * alpha * p0 * (1 - p0))  # diagonal   (6.4)
    return np.sqrt(np.diag(np.linalg.inv(I)))               # (6.7)


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

SYSTEM = "You are a careful judge of comedy. You compare two jokes written for the same news headline."
TEMPLATE = """Headline: {prompt}

Joke A: {first}

Joke B: {second}

Which joke is funnier as a response to this headline? Answer with exactly one letter: A or B."""


class LLMJudge:
    """compare(prompt, first, second) -> 0 if `first` is funnier, 1 if `second` is."""

    def __init__(self, model, log_path, retries=3):
        self.url = "http://localhost:11434/api/chat"                             # FIXED
        self.model, self.retries = model, retries
        self.log = open(log_path, "w", encoding="utf-8")
        self.calls = 0

    def ask(self, text):
        body = json.dumps({                                                       # FIXED
            "model": self.model,                                                  # FIXED
            "messages": [{"role": "system", "content": SYSTEM},                   # FIXED
                         {"role": "user", "content": text}],                      # FIXED
            "stream": False,                                                      # FIXED
            "options": {"temperature": 0, "num_predict": 2048},                   # FIXED
        }).encode()                                                               # FIXED
        req = urllib.request.Request(self.url, data=body,                         # FIXED
                                     headers={"Content-Type": "application/json"})  # FIXED
        with urllib.request.urlopen(req, timeout=300) as r:                       # FIXED
            return json.load(r)["message"]["content"] or ""                       # FIXED

    def compare(self, prompt, first, second):
        text = TEMPLATE.format(prompt=prompt, first=first, second=second)
        reply = ""
        for attempt in range(self.retries):
            try:
                reply = self.ask(text)
            except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as e:
                reply = f"<error: {e!r}>"
            final = re.sub(r"<think>.*?(</think>|$)", "", reply, flags=re.S)       # FIXED
            m = re.findall(r"\b([AB])\b", final.upper())                         # FIXED
            if m:                                                                 # FIXED
                answer = 0 if m[-1] == "A" else 1                                 # FIXED
                self.calls += 1
                self.log.write(json.dumps({"prompt": prompt, "shown_first": first, "shown_second": second,
                                           "answer": "A" if answer == 0 else "B", "raw": reply,
                                           "model": self.model}) + "\n")
                self.log.flush()
                return answer
            time.sleep(2 ** attempt)  # wait 1 s, 2 s, 4 s before retrying
        raise RuntimeError(f"No usable A/B answer after {self.retries} tries; last reply: {reply!r}")


def binomial_p(k, T):
    """Two-sided exact test of 'shown-first wins with probability 1/2'."""
    probs = [math.comb(T, i) / 2 ** T for i in range(T + 1)]
    return min(1.0, sum(p for p in probs if p <= probs[k] * (1 + 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jokes", help="JSON file with 'prompt' and 'jokes' (default: built-in example)")
    ap.add_argument("--model", default="deepseek-r1:latest")  # FIXED
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

    judge = LLMJudge(a.model, a.log)
    theta, se, order, first_wins, T = rank(prompt, jokes, judge, a.alpha, a.seed)

    print(f"Headline: {prompt}\n")
    print(" rank  fitted    se   joke")
    for r, i in enumerate(order):
        joke = jokes[i] if len(jokes[i]) <= 70 else jokes[i][:67] + "..."
        print(f"{r + 1:5d} {theta[i]:7.2f} {se[i]:5.2f}   {joke}")
    print(f"\nshown-first win rate = {first_wins}/{T} = {first_wins / T:.2f}"
          f"   (binomial p = {binomial_p(first_wins, T):.3f}; p < 0.05 means position bias)")
    print(f"every answer logged to {a.log}")


if __name__ == "__main__":
    main()