"""Exp 24, part 2: an LLM judge calibrated on the user's votes ranks the prompts the user does not have time to vote on.

Usage (after `sim.compare_generators` run, export and vote):
    # default judge: Qwen2.5-7B-Instruct on Explorer, neither candidate's family (explorer/README.md "Exp 24 judge")
    python -m sim.compare_generators judge --export-bundle data/model_comparison/judge     # upload calls.jsonl, run the job
    python -m sim.compare_generators judge --import-replies <downloaded replies.jsonl>      # read it back
    # an API judge instead (needs its key): python -m sim.compare_generators judge --judge-model claude-haiku-5-5 --env-file ...

The judge is shown the user's ranked prompts (the bits under the same blind letters the user saw, and the user's order)
and ranks a new prompt the same way. Every prompt is judged once per position the bits can take (a cyclic rotation of one
seeded shuffle: with three bits, three orders, each bit shown first, second and last exactly once), combined by mean
rank, so position bias cancels exactly; it is also reported. The judge may only join the user's votes if it first proves it
agrees with them, leave-one-out (each of the user's prompts judged seeing only the user's OTHER rankings):

  1. top-pick match with the user at least 60% AND the 95% Wilson interval's lower end above chance (mean of 1 / bits shown);
  2. pairwise agreement (the share of bit pairs ordered as the user ordered them) with the interval's lower end above 50%;
  3. per model, the mean judge-minus-user rank gap with a bootstrap 95% CI over prompts must not lie beyond 0.3 ranks.

(3 is the reviewer's addition, open to the user's veto, D-93.) A run in which any call was not answered for an outage,
a bad key or a missing reply is `incomplete`: it joins nothing and the gate is not decided. Replies are cached on disk beside
the generations (gitignored), keyed by a hash of the judge, its exact system and user messages (so a changed vote, example
or bit asks again). Only numbers and prompt ids are stored in `judge_result.json`.
"""
import hashlib
import json
import os
import pathlib
import random
import re
import sys
import uuid
from typing import Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app import llm  # noqa: E402
from sim import compare_generators as cg  # noqa: E402
from sim.evaluate_safety import wilson  # noqa: E402

JUDGE_MODEL = "qwen2.5-7b-instruct"  # why this one: docs/agent/decisions-detail.md D-93
JUDGES = {
    # Run on Explorer from a bundle (explorer/judge_generators.py); no API, no price, no live client.
    JUDGE_MODEL: cg.Spec(JUDGE_MODEL, "explorer", "Qwen/Qwen2.5-7B-Instruct", 0.0, 0.0, "explorer", 0.0),
    "claude-haiku-5-5": cg.MODELS["claude-haiku-5-5"],
    "gemini-flash-lite": cg.MODELS["gemini-flash-lite"],
}
OLLAMA_URL = "http://127.0.0.1:11434/v1"  # Ollama's OpenAI-compatible endpoint: the same llm.Client, no key
OLLAMA_TIMING = (900.0, 1500.0)  # a 27B model reading 15k tokens is slow: per attempt and over all attempts
CHARS_PER_TOKEN = 3.5  # a conservative estimate: a real prompt has fewer tokens than chars / 3.5
SHORTFALL = 0.5  # Ollama must report reading at least this share of the estimated prompt tokens, or it truncated
THINK = re.compile(r"<think>.*?</think>", re.S)
SCHEME = "latin-rotation-v1"  # how many orders and which: part of what makes a result current
BAR_AGREEMENT = 0.60
BAR_PAIRWISE = 0.50  # the lower end of the pairwise agreement's interval must be above this
BAR_BIAS = 0.3  # ranks: a per-model gap whose 95% CI lies beyond this on either side blocks the judge
JUDGE_TIMING = (60.0, 90.0)  # API judges: per attempt and over all attempts, a prompt of 29 ranked examples is long
JUDGE_RETRIES = 2

SYSTEM = """You are helping to pick the funniest stand-up material for a live show performed by a self-aware, slightly anxious robot. One person, whose taste you must learn, has ranked several sets of alternative bits that were written for different rooms.

You are given their ranked examples. Each example shows a setting, then several options (each is a bit: five lines in playing order, under a letter) and the order that person put them in, best first. Study what they prefer: what makes a bit land for them, what falls flat.

Then you get a new setting with its own options. The letters A, B and C are only labels, reused from one setting to the next; they carry no meaning. Rank the new options the way that person would, best first. Use every letter exactly once. Judge the bit, never the letter or its position.

Reply with JSON only: {"ranking": "<letters, best first>"}"""
REPLY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["ranking"],
                "properties": {"ranking": {"type": "string"}}}


def prompt_digest() -> str:
    """What a result depends on besides the votes: the judge's instructions, reply shape and order scheme."""
    return hashlib.sha256((SYSTEM + json.dumps(REPLY_SCHEMA, sort_keys=True) + SCHEME).encode()).hexdigest()[:8]


# --- what the judge sees -----------------------------------------------------------------------------------------

def setting_line(prompt: dict) -> str:
    c = prompt["context"]
    return f"{c['venue']}, {c['energy']} energy, {c['age']} audience"


def block(setting: str, options: dict[str, list[str]]) -> str:
    out = [f"Setting: {setting}"]
    for letter, lines in options.items():
        out.append(f"Option {letter}:\n" + "\n".join(f"{j + 1}. {t}" for j, t in enumerate(lines)))
    return "\n".join(out)


def build_prompt(examples: list[dict], target: str) -> tuple[str, str]:
    """`examples`: {setting, options: {letter: lines}, ranking}. The letters are the ones the user saw. Nothing but the
    letters identifies a model."""
    parts = [f"## Example {i}\n{block(e['setting'], e['options'])}\nTheir ranking, best first: {e['ranking']}"
             for i, e in enumerate(examples, 1)]
    parts.append(f"## New setting to rank\n{target}\nRank these options as that person would.")
    return SYSTEM, "\n\n".join(parts)


def require_complete(directory: pathlib.Path, models: list[str], n: int) -> None:
    """Every prompt up to n needs a record for every model, as `export` does: a missing one is never an automatic tie."""
    pending = [p["id"] for p in cg.prompts(n) if any(
        not (r := cg.read_record(directory / m / f"{p['id']}.json")) or not r["attempts"] for m in models)]
    if pending:
        raise SystemExit(f"{len(pending)} prompts have no record for some model yet (e.g. {pending[:3]}): finish `run` first")


def shown_threads(directory: pathlib.Path, models: list[str], pid: str, key: dict) -> tuple[dict[str, list[str]], list[str]]:
    """({model: the thread's lines}, models with no safe thread): the ballot's own choice when the prompt is on it, else
    the same rule export uses, the first fully safe thread of the attempt the ballot uses."""
    recs = {m: cg.read_record(directory / m / f"{pid}.json") for m in models}
    if any(r is None or not r["attempts"] for r in recs.values()):
        raise SystemExit(f"{pid} has no record for every model yet: finish `run` first")
    if pid in key:
        picks = {m: i for m, i in key[pid]["threads"].items() if m in models}
    else:
        picks = {m: t for m in models if (t := cg.playable_thread(recs[m])) is not None}
    return ({m: cg.final(recs[m])["threads"][t] for m, t in picks.items()},
            [m for m in models if m not in picks])


def user_examples(directory: pathlib.Path, models: list[str], key: dict, votes: dict, prompts_by_id: dict) -> dict[str, dict]:
    """Every prompt the user ranked among `models` (two or more bits): its setting, its options under the user's own
    letters, and their ranking. With a subset of the ballot's models the bits of the others are left out and the user's
    order is kept for the rest."""
    out = {}
    for pid, k in key.items():
        all_letters = "".join(k["letters"])
        ranking = cg.parse_ranking(votes.get(pid, ""), all_letters) if len(k["letters"]) >= 2 else None
        if not ranking:
            continue
        kept = [c for c in ranking if k["letters"][c] in models]
        if len(kept) >= 2:
            threads, _ = shown_threads(directory, models, pid, key)
            out[pid] = {"setting": setting_line(prompts_by_id[pid]), "ranking": "".join(kept),
                        "options": {c: threads[k["letters"][c]] for c in sorted(kept)}}
    return out


def orders_for(pid: str, shown: list[str], seed: int = cg.SEED) -> list[dict[str, str]]:
    """The letter assignments a prompt is judged under: one seeded shuffle of the bits, then each cyclic rotation of it.
    With s bits shown there are s orders and every bit takes every position exactly once, so a judge's liking for a
    position (the first bit, say) cancels in the mean rank instead of being averaged down by chance. (The alternative, two
    orders with the second the reverse of the first, balances the first and last positions but leaves a three-way
    prompt's middle bit always in the middle, and costs a third fewer calls.)"""
    base = list(shown)
    random.Random(f"{seed}:judge:{pid}").shuffle(base)
    return [dict(zip(cg.LETTERS, base[k:] + base[:k])) for k in range(len(base))]


def call_id(judge: str, kind: str, pid: str, order: int, system: str, user: str) -> str:
    """Names the call by a hash of everything the judge is shown, so the cache never answers a different question."""
    h = hashlib.sha256(f"{judge}\x00{system}\x00{user}".encode()).hexdigest()[:16]
    return f"{kind}-{pid}-o{order}-{h}"


def calls_for(judge: str, kind: str, pid: str, item: dict, examples: list[dict], seed: int = cg.SEED) -> list[dict]:
    out = []
    for order, letters in enumerate(orders_for(pid, list(item["shown"]), seed)):
        system, user = build_prompt(examples, block(item["setting"], {L: item["shown"][m] for L, m in letters.items()}))
        out.append({"id": call_id(judge, kind, pid, order, system, user), "kind": kind, "pid": pid, "order": order,
                    "letters": letters, "system": system, "user": user})
    return out


def plan(directory: pathlib.Path, models: list[str], n: int, n_ballot: int, judge: str, seed: int = cg.SEED) -> dict:
    """Everything the judge step will ask, and the context to score the answers. Needs every ballot prompt with two or
    more bits shown voted on, and every generated prompt present for every model."""
    require_complete(directory, models, n)
    key = {p: k for p, k in json.loads(cg.key_path(directory).read_text(encoding="utf-8")).items()
           if p in {q["id"] for q in cg.prompts(n_ballot)}}
    votes = cg.read_votes(directory / "votes.csv")
    missing = [p for p, k in key.items() if len(k["letters"]) >= 2 and not cg.parse_ranking(votes.get(p, ""), "".join(k["letters"]))]
    if missing:
        raise SystemExit(f"vote on every ballot prompt first ({len(missing)} left, e.g. {missing[:3]}): the judge learns from all of them")
    by_id = {p["id"]: p for p in cg.prompts(n)}
    ex = user_examples(directory, models, key, votes, by_id)

    def item(pid):
        shown, last = shown_threads(directory, models, pid, key)
        return {"setting": setting_line(by_id[pid]), "shown": shown, "last": last}

    items, calls, auto = {pid: item(pid) for pid in ex}, [], {}
    for pid in sorted(ex):
        calls += calls_for(judge, "loo", pid, items[pid], [ex[k] for k in sorted(ex) if k != pid], seed)
    for p in cg.prompts(n)[n_ballot:]:
        it = items[p["id"]] = item(p["id"])
        if len(it["shown"]) < 2:  # one bit or none: nothing to rank, as on the ballot
            auto[p["id"]] = cg.assemble_ranks({m: float(i) for i, m in enumerate(it["shown"], 1)}, it["last"])
        else:
            calls += calls_for(judge, "new", p["id"], it, [ex[k] for k in sorted(ex)], seed)
    return {"calls": calls, "items": items, "examples": ex, "auto": auto, "key": key,
            "user_ranks": cg.ranks_from_votes(votes, key, models), "models": models, "n_ballot": n_ballot}


# --- asking, with a cache ----------------------------------------------------------------------------------------

def _parse(reply: str, letters: dict[str, str]) -> str:
    reply = THINK.sub("", reply).split("</think>")[-1]  # a thinking model's reasoning is not the answer
    ranking = cg.parse_ranking(str(json.loads(llm.strip_fence(reply))["ranking"]), "".join(letters))
    if not ranking:
        raise ValueError("not a ranking of the letters")
    return ranking


def cache_path(directory: pathlib.Path, judge: str, cid: str) -> pathlib.Path:
    return directory / "judge_cache" / judge / f"{cid}.json"


def ask(client, spec: cg.Spec, call: dict, directory: pathlib.Path, pacer: cg.Pacer, first: set) -> dict:
    """One judge call, cached. Returns {ok, order: [models best first] or None, kind, reason, tokens, latency, live}. A
    config failure on the FIRST LIVE call of a run raises `ConfigError` (the guard is only used up by a call that reached
    the judge, not by a cache hit). An outage or config failure is not cached: it is not the judge's answer."""
    path = cache_path(directory, spec.key, call["id"])
    if path.exists():
        return {**json.loads(path.read_text(encoding="utf-8")), "live": False}
    letters = call["letters"]
    pacer.wait(spec.bucket, spec.gap_s)
    before = (client.input_tokens, client.output_tokens) if hasattr(client, "input_tokens") else (0, 0)
    start = cg.time.perf_counter()
    failure = ranking = kind = None
    try:
        reply = client.chat(call["system"], call["user"], json_reply=REPLY_SCHEMA, validate=lambda r: _parse(r, letters))
    except Exception as e:  # the client could not get an answer: an LLMError, or something unexpected (a config problem)
        failure, kind = e, cg.failure_kind(e)
    else:
        try:
            ranking = _parse(reply, letters)
        except Exception as e:  # answered, and still not a ranking after the re-ask: the judge's answer, a miss
            failure, kind = e, "model"
    latency = cg.time.perf_counter() - start
    pacer.mark(spec.bucket)
    if failure is not None and spec.key in first and kind == "config":
        raise cg.ConfigError(f"judge {spec.key}: {type(failure).__name__}: {failure}")
    if kind != "infra" or failure is None:
        first.discard(spec.key)  # the judge answered (even badly): the key and model are real
    rec = {"ok": ranking is not None, "order": None if ranking is None else [letters[c] for c in ranking],
           "reason": None if failure is None else f"{type(failure).__name__}: {failure}"[:200], "latency_s": latency,
           "input_tokens": getattr(client, "input_tokens", 0) - before[0], "output_tokens": getattr(client, "output_tokens", 0) - before[1],
           "kind": kind}
    if kind not in ("config", "infra"):
        cg.write_record(path, rec)
    return {**rec, "live": True}


def estimate_tokens(system: str, user: str) -> int:
    return int((len(system) + len(user)) / CHARS_PER_TOKEN)


def context_length() -> int:
    """The context the Ollama server was started with (`OLLAMA_CONTEXT_LENGTH`), 32768 by default here: the few-shot
    prompts are 12 to 20k tokens and Ollama's own default is far smaller."""
    return int(os.environ.get("OLLAMA_CONTEXT_LENGTH") or 32768)


def check_context(calls: list[dict], context: int) -> None:
    """Refuse to run when a prompt would not fit: Ollama truncates silently, so the judge would rank bits it never read."""
    longest = max(calls, key=lambda c: estimate_tokens(c["system"], c["user"]), default=None)
    if longest and estimate_tokens(longest["system"], longest["user"]) > context:
        raise SystemExit(f"{longest['id']}: the prompt is about {estimate_tokens(longest['system'], longest['user'])} tokens but the "
                         f"context is {context}: start Ollama with OLLAMA_CONTEXT_LENGTH higher (and a model that has the room), or use fewer examples")


class OllamaGuard:
    """Wraps the llm.Client that talks to Ollama's /v1. Ollama truncates a prompt to its context silently and still
    answers, so the guard checks the reply's `usage.prompt_tokens` (Ollama's prompt_eval_count) against the prompt's
    estimated size: a big shortfall, or no usage at all, is an infrastructure failure (not cached, run incomplete), never an
    answer. A per-call nonce at the start of the system message stops Ollama reusing a cached prefix, which would make the
    count look like truncation."""

    def __init__(self, client, context: int | None = None):
        self.client, self.context = client, context or context_length()
        self.model = client.model

    @property
    def input_tokens(self):
        return self.client.input_tokens

    @property
    def output_tokens(self):
        return self.client.output_tokens

    def chat(self, system, user, *, json_reply=False, temperature=0.0, validate=None):
        est = estimate_tokens(system, user)
        if est > self.context:
            raise llm.LLMError(f"prompt of about {est} tokens is over the {self.context}-token context", retryable=True)
        before = self.client.input_tokens
        reply = self.client.chat(system + f"\n\n(request {uuid.uuid4().hex[:8]})", user, json_reply=json_reply,
                                 temperature=temperature, validate=validate)
        read = self.client.input_tokens - before
        if read < SHORTFALL * est:
            raise llm.LLMError(f"Ollama read {read} tokens of a prompt of about {est}: truncated, or no usage reported", retryable=True)
        return reply


def resolve_judge(model: str | None, provider: str | None) -> cg.Spec:
    """The judge to use. `provider` ollama builds a spec for any Ollama model (the first to try is qwen3.8:27b); otherwise
    `model` names one of JUDGES (default Qwen2.5-7B on Explorer). The key is filesystem-safe (no colon)."""
    if provider == "ollama":
        model = model or "qwen3.8:27b"
        return cg.Spec("ollama-" + re.sub(r"[^A-Za-z0-9.]+", "-", model), "ollama", model, 0.0, 0.0, "ollama", 0.0)
    if provider not in (None, "explorer", "anthropic", "gemini"):
        raise SystemExit(f"unknown judge provider {provider!r}")
    judge = model or JUDGE_MODEL
    if judge not in JUDGES:
        raise SystemExit(f"unknown judge {judge!r}; choose from {list(JUDGES)} or use --judge-provider ollama")
    return JUDGES[judge]


class ReplayClient:
    """The judge when its answers were produced elsewhere (Explorer): everything must already be in the cache, so a call
    that is not is reported as missing (an infrastructure failure), never made up."""
    input_tokens = output_tokens = 0
    model = "replay"

    def chat(self, system, user, *, json_reply=False, temperature=0.0, validate=None):
        raise llm.LLMError("no reply for this call in the imported file", retryable=True)


# --- scoring the answers -----------------------------------------------------------------------------------------

def combine(item: dict, recs: list[dict]) -> dict:
    """One prompt's judge ranks from its answered orders (mean rank among the shown bits, the models with no safe thread
    tied last); ranks None when no order answered."""
    answered = [{m: float(i) for i, m in enumerate(r["order"], 1)} for r in recs if r["ok"]]
    if not answered:
        return {"ranks": None}
    mean = {m: float(np.mean([o[m] for o in answered])) for m in item["shown"]}
    return {"ranks": cg.assemble_ranks(mean, item["last"])}


def top_credit(judge_ranks: dict[str, float], shown: list[str], user_top: str) -> float:
    """1 when the judge's top pick is the user's, 1 / k when k bits tie for it and the user's is among them, else 0."""
    best = min(judge_ranks[m] for m in shown)
    tops = [m for m in shown if judge_ranks[m] == best]
    return 1.0 / len(tops) if user_top in tops else 0.0


def agreement(credits: list[float], chances: list[float]) -> dict:
    """Top-pick match with a Wilson 95% interval, against chance (the mean of 1 / bits shown per prompt)."""
    n, k = len(credits), float(sum(credits))
    ci = wilson(k, n)
    chance = float(np.mean(chances)) if chances else 0.0
    rate = k / n if n else 0.0
    return {"n": n, "matches": k, "rate": rate, "ci95": ci, "chance": chance,
            "pass": n > 0 and rate >= BAR_AGREEMENT and ci[0] > chance}


def pairwise_agreement(judge_ranks: dict[str, dict[str, float]], user_ranks: dict[str, dict[str, float]],
                       shown: dict[str, list[str]]) -> dict:
    """The share of bit pairs (within a prompt, among the bits shown) that the judge orders as the user did; a judge tie
    earns half. Chance is 50%. The pairs of one prompt are not independent, so the interval is optimistic; it is a second
    test beside the top pick, which only looks at one bit per prompt."""
    credits = []
    for pid in sorted(judge_ranks):
        ms = shown[pid]
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                u = user_ranks[pid][a] - user_ranks[pid][b]
                j = judge_ranks[pid][a] - judge_ranks[pid][b]
                credits.append(0.5 if j == 0 else float((u < 0) == (j < 0)))
    n, k = len(credits), float(sum(credits))
    ci = wilson(k, n)
    return {"n": n, "matches": k, "rate": k / n if n else 0.0, "ci95": ci, "chance": 0.5, "pass": n > 0 and ci[0] > BAR_PAIRWISE}


def model_bias(judge_ranks: dict[str, dict[str, float]], user_ranks: dict[str, dict[str, float]], models: list[str],
               n_boot: int = cg.N_BOOT, seed: int = cg.SEED) -> dict:
    """Per model: the mean of (judge rank - user rank) over the prompts both ranked, a bootstrap 95% CI over prompts, and
    whether the CI lies beyond +-BAR_BIAS. Positive: the judge puts the model lower than the user does. What this
    guarantees: no CI that excludes a gap of 0.3 ranks passes; a bias smaller than the CI is wide enough to hide passes."""
    pids = sorted(set(judge_ranks) & set(user_ranks))
    if not pids:
        return {"n": 0, "models": {}, "pass": False}
    idx = cg.boot_idx(len(pids), n_boot, seed)
    out = {}
    for m in models:
        d = np.array([judge_ranks[p][m] - user_ranks[p][m] for p in pids], float)
        ci = cg.boot_ci(d, idx)
        out[m] = {"mean": float(d.mean()), "ci95": ci, "blocks": ci[1] < -BAR_BIAS or ci[0] > BAR_BIAS}
    return {"n": len(pids), "models": out, "pass": not any(v["blocks"] for v in out.values())}


def position_bias(calls: list[dict]) -> dict:
    """How often the bit shown first (letter A) is ranked top, against chance (the mean of 1 / bits shown)."""
    ok = [c for c in calls if c["ok"]]
    if not ok:
        return {"n": 0, "first_top": None, "chance": None}
    return {"n": len(ok), "first_top": float(np.mean([c["order"][0] == c["first_shown"] for c in ok])),
            "chance": float(np.mean([1 / c["shown"] for c in ok]))}


def gate(agree: dict, pair: dict, bias: dict, incomplete: int = 0) -> tuple[bool, str]:
    if incomplete:
        return False, (f"incomplete: {incomplete} calls were not answered (an outage, a bad key or replies missing from the "
                       "imported file), so the judge is not decided and nothing joins; run it again")
    if not agree["pass"]:
        return False, (f"top-pick agreement {agree['rate']:.0%} (95% CI {agree['ci95'][0]:.0%} to {agree['ci95'][1]:.0%}) "
                       f"does not clear {BAR_AGREEMENT:.0%} with a lower end above chance {agree['chance']:.0%}: the user's votes decide alone")
    if not pair["pass"]:
        return False, (f"pairwise agreement {pair['rate']:.0%} (95% CI {pair['ci95'][0]:.0%} to {pair['ci95'][1]:.0%}) "
                       f"does not have a lower end above {BAR_PAIRWISE:.0%}: the user's votes decide alone")
    if not bias["pass"]:
        bad = ", ".join(m for m, v in bias["models"].items() if v["blocks"])
        return False, f"the judge agrees on top picks but is biased on {bad} (CI beyond {BAR_BIAS} ranks): the user's votes decide alone"
    return True, "the judge agrees with the user and shows no model-specific bias it could be told from: its rankings of the later prompts join the votes"


def assess(pl: dict, recs: dict[str, dict]) -> dict:
    """Score the answers (`recs[call id]`) against the plan: leave-one-out agreement and bias, then the later prompts."""
    by_pid: dict[tuple[str, str], list] = {}
    for c in pl["calls"]:
        by_pid.setdefault((c["kind"], c["pid"]), []).append((c, recs[c["id"]]))
    loo_calls, new_calls, rows, credits, chances = [], [], {}, [], []
    judge_r, user_r, shown_of, new_ranks, unjudged = {}, {}, {}, {}, []
    for (kind, pid), pairs in sorted(by_pid.items()):
        item = pl["items"][pid]
        info = [{**r, "shown": len(item["shown"]), "first_shown": c["letters"]["A"]} for c, r in pairs]
        ranks = combine(item, [r for _, r in pairs])["ranks"]
        if kind == "loo":
            loo_calls += info
            shown = list(item["shown"])
            user_top = min(shown, key=lambda m: pl["user_ranks"][pid][m])
            chances.append(1 / len(shown))
            if ranks is None:
                credits.append(0.0)  # an unjudged prompt counts against the judge
                rows[pid] = {"judged": False}
            else:
                credits.append(top_credit(ranks, shown, user_top))
                judge_r[pid], user_r[pid], shown_of[pid] = ranks, pl["user_ranks"][pid], shown
                rows[pid] = {"judged": True, "credit": credits[-1], "shown": len(shown)}
        else:
            new_calls += info
            if ranks is None:
                unjudged.append(pid)
            else:
                new_ranks[pid] = ranks
    new_ranks.update(pl["auto"])
    agree = agreement(credits, chances)
    pair = pairwise_agreement(judge_r, user_r, shown_of)
    bias = model_bias(judge_r, user_r, pl["models"])
    missing = [r for ps in by_pid.values() for _, r in ps if not r["ok"] and r["kind"] in ("config", "infra")]
    ok, reason = gate(agree, pair, bias, len(missing))
    calls = loo_calls + new_calls
    return {"agreement": agree, "pairwise": pair, "bias": bias,
            "position_bias": {"leave_one_out": position_bias(loo_calls), "new": position_bias(new_calls), "all": position_bias(calls)},
            "incomplete": bool(missing), "n_unanswered": len(missing), "joined": ok, "reason": reason,
            "n_calls": len(calls), "n_failed_calls": sum(not c["ok"] for c in calls), "unjudged_new": unjudged,
            "auto_ranked_new": sorted(pl["auto"]), "loo": rows, "judge_ranks": new_ranks,
            "input_tokens": sum(c["input_tokens"] for c in calls), "output_tokens": sum(c["output_tokens"] for c in calls)}


def run_judge(models: list[str], n: int, n_ballot: int, make_client: Callable[[cg.Spec], object], judge: str,
              directory: pathlib.Path | None = None, pacer: cg.Pacer | None = None, log=print,
              spec: cg.Spec | None = None) -> dict:
    """Ask (or read from the cache) every call of the plan, score them, and write `judge_result.json` (numbers and prompt
    ids only). An incomplete run is written too, marked `incomplete`, and `cg.judge_ranks` ignores it."""
    directory, pacer = directory or cg.run_dir(), pacer or cg.Pacer()
    spec = spec or JUDGES[judge]
    pl = plan(directory, models, n, n_ballot, judge)
    client, first = make_client(spec), {spec.key}
    if spec.provider == "ollama" and not isinstance(client, ReplayClient):
        check_context(pl["calls"], context_length())
    recs = {}
    for i, call in enumerate(pl["calls"], 1):
        recs[call["id"]] = ask(client, spec, call, directory, pacer, first)
        if i % 20 == 0:
            log(f"{i}/{len(pl['calls'])} calls")
    res = assess(pl, recs)
    res.update(judge_model=spec.model, prompt_digest=prompt_digest(), scheme=SCHEME, ballot_digest=cg.ballot_digest(directory),
               n_ballot=n_ballot, n_prompts=n, n_examples=len(pl["examples"]), n_planned_calls=len(pl["calls"]),
               cost_usd=(res["input_tokens"] * spec.price_in + res["output_tokens"] * spec.price_out) / 1e6)
    cg.write_record(cg.judge_path(directory), res)
    return res


# --- the Explorer bundle -----------------------------------------------------------------------------------------

def manifest_path(directory: pathlib.Path) -> pathlib.Path:
    return directory / "judge_manifest.json"


def export_bundle(models: list[str], n: int, n_ballot: int, judge: str, out_dir: pathlib.Path,
                  directory: pathlib.Path | None = None, smoke: int = 0) -> dict:
    """Write `calls.jsonl` (id, system, user, letters: nothing that names a model, no key) for the cluster, and keep the
    letters-to-models manifest here, in the run folder, never in the bundle. `smoke` keeps only the calls of the first
    N leave-one-out prompts."""
    directory = directory or cg.run_dir()
    pl = plan(directory, models, n, n_ballot, judge)
    calls = pl["calls"]
    if smoke:
        keep = sorted({c["pid"] for c in calls if c["kind"] == "loo"})[:smoke]
        calls = [c for c in calls if c["kind"] == "loo" and c["pid"] in keep]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "calls.jsonl", "w", encoding="utf-8") as f:
        for c in calls:
            f.write(json.dumps({"id": c["id"], "system": c["system"], "user": c["user"], "letters": sorted(c["letters"])}) + "\n")
    cg.write_record(manifest_path(directory), {c["id"]: {"kind": c["kind"], "pid": c["pid"], "order": c["order"],
                                                           "letters": c["letters"]} for c in pl["calls"]})
    return {"calls": len(calls), "loo": sum(c["kind"] == "loo" for c in calls), "new": sum(c["kind"] == "new" for c in calls),
            "bundle": str(out_dir / "calls.jsonl")}


def import_replies(path: pathlib.Path, judge: str, directory: pathlib.Path | None = None) -> dict:
    """Read the cluster's `replies.jsonl` into the judge cache, using the local manifest to turn letters back into models.
    Replies for ids the manifest does not know (a stale bundle) are ignored and counted."""
    directory = directory or cg.run_dir()
    manifest = json.loads(manifest_path(directory).read_text(encoding="utf-8"))
    taken = unknown = bad = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        m = manifest.get(r["id"])
        if m is None:
            unknown += 1
            continue
        ranking = cg.parse_ranking(r.get("ranking") or "", "".join(m["letters"]))
        if not ranking:
            bad += 1
            continue
        cg.write_record(cache_path(directory, judge, r["id"]), {"ok": True, "order": [m["letters"][c] for c in ranking], "reason": None,
                                                                 "latency_s": 0.0, "input_tokens": 0, "output_tokens": 0, "kind": None})
        taken += 1
    return {"imported": taken, "unknown_ids": unknown, "not_a_ranking": bad}


def summary(res: dict) -> str:
    if res["incomplete"]:
        return f"judge {res['judge_model']}: INCOMPLETE, {res['n_unanswered']} of {res['n_calls']} calls unanswered.\n{res['reason']}."
    a, pw, b, p = res["agreement"], res["pairwise"], res["bias"], res["position_bias"]["all"]
    biases = ", ".join(f"{m} {v['mean']:+.2f} ({v['ci95'][0]:+.2f} to {v['ci95'][1]:+.2f})" for m, v in b["models"].items())
    first = "n/a" if p["first_top"] is None else f"{p['first_top']:.0%} (chance {p['chance']:.0%})"
    return (f"judge {res['judge_model']}: top-pick agreement {a['rate']:.0%} of {a['n']} (95% CI {a['ci95'][0]:.0%} to "
            f"{a['ci95'][1]:.0%}, chance {a['chance']:.0%}); pairwise {pw['rate']:.0%} of {pw['n']} pairs (CI {pw['ci95'][0]:.0%} to "
            f"{pw['ci95'][1]:.0%}); judge minus user rank by model: {biases}; first-shown bit ranked top {first}; "
            f"{res['n_calls']} calls, ${res['cost_usd']:.2f}.\n{res['reason']}.")


def main(models: list[str], n: int, n_ballot: int, judge: str | None, export_to: str | None = None,
         import_from: str | None = None, smoke: int = 0, provider: str | None = None) -> None:
    spec = resolve_judge(judge, provider)
    judge = spec.key
    if export_to:
        print(export_bundle(models, n, n_ballot, judge, pathlib.Path(export_to), smoke=smoke))
        return
    if import_from or spec.provider == "explorer":
        if not import_from:
            sys.exit(f"{judge} runs on Explorer: `judge --export-bundle <dir>`, upload and run explorer/judge_generators.sbatch "
                     "(or explorer/judge_ollama.sbatch for Ollama), download replies.jsonl, then `judge --import-replies <file>` "
                     "(explorer/README.md)")
        print(import_replies(pathlib.Path(import_from), judge))
        make = lambda s: ReplayClient()  # noqa: E731
    elif spec.provider == "ollama":  # an Ollama server you can reach (a laptop one, or a tunnel to the cluster)
        def make(s):
            timeout, deadline = OLLAMA_TIMING
            return OllamaGuard(llm.Client(os.environ.get("OLLAMA_URL", OLLAMA_URL), "", s.model, timeout, retries=1, deadline_s=deadline))
    else:
        if spec.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("ANTHROPIC_API_KEY is not set: add a paid key to backend/.env, or use the Explorer judge")

        def make(s):
            timeout, deadline = JUDGE_TIMING
            return llm.from_env(timeout, role="joke_writer", provider=s.provider, model=s.model, retries=JUDGE_RETRIES,
                                deadline_s=deadline)
    try:
        print(summary(run_judge(models, n, n_ballot, make, judge, spec=spec)))
    except cg.ConfigError as e:
        sys.exit(f"stopped, nothing cached for the failing call: {e}")
