"""Exp 24, part 2: an LLM judge calibrated on the user's votes ranks the prompts the user does not have time to vote on.

Usage (after `sim.compare_generators` run, export and vote; needs the judge model's key):
    python -m sim.compare_generators judge --env-file backend/.env

The judge is shown the user's ranked prompts (the bits under the same blind letters the user saw, and the user's order)
and ranks a new prompt the same way. Every prompt is judged in `ORDERS` shuffled orders, combined by mean rank, so the
judge's liking for whichever bit comes first cancels out (and is reported). It may only join the user's votes if it first
proves it agrees with them:

  Leave-one-out check: each of the user's prompts is judged seeing only the user's OTHER rankings. The judge's top pick
  must match the user's top pick at least 60% of the time AND the 95% Wilson interval's lower end must be above chance
  (the mean of 1 / bits shown). Then, per model, the paired gap between the judge's rank and the user's rank (bootstrap
  over prompts) must not exclude 0 by more than 0.3 ranks for any model: top-pick agreement can pass while the judge
  favors one model, which is what a Gemini or Claude judge liking its own family would look like. (The bias check is a
  reviewer's addition to the user's bar, see D-92.)

If both hold, the judge's rankings of prompts after the ballot join the user's votes; otherwise the user's votes decide
alone and the judge is reported as unreliable. Replies are cached on disk beside the generations (gitignored), keyed by
the judge model, its prompt, the examples shown and the order. Only numbers are stored in `judge_result.json`.
"""
import hashlib
import json
import pathlib
import random
import sys
from typing import Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app import llm  # noqa: E402
from sim import compare_generators as cg  # noqa: E402
from sim.evaluate_safety import wilson  # noqa: E402

JUDGE_MODEL = "claude-haiku-5-5"  # why this one: docs/agent/decisions-detail.md D-92
ORDERS = 2  # shuffled orders per prompt
BAR_AGREEMENT = 0.60
BAR_BIAS = 0.3  # ranks: a per-model gap whose 95% CI lies beyond this on either side blocks the judge
JUDGE_TIMING = (60.0, 90.0)  # per attempt and over all attempts: a prompt of 29 ranked examples is long
JUDGE_RETRIES = 2

SYSTEM = """You are helping to pick the funniest stand-up material for a live show performed by a self-aware, slightly anxious robot. One person, whose taste you must learn, has ranked several sets of alternative bits that were written for different rooms.

You are given their ranked examples. Each example shows a setting, then several options (each is a bit: five lines in playing order, under a letter) and the order that person put them in, best first. Study what they prefer: what makes a bit land for them, what falls flat.

Then you get a new setting with options under new letters. Rank them the way that person would, best first. Use every letter exactly once. Judge the bit, never the letter or its position.

Reply with JSON only: {"ranking": "<letters, best first>"}"""
REPLY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["ranking"],
                "properties": {"ranking": {"type": "string"}}}


def prompt_digest() -> str:
    return hashlib.sha256((SYSTEM + json.dumps(REPLY_SCHEMA, sort_keys=True) + str(ORDERS)).encode()).hexdigest()[:8]


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


def shown_threads(directory: pathlib.Path, models: list[str], pid: str, key: dict) -> tuple[dict[str, list[str]], list[str]]:
    """({model: the thread's lines}, models with no safe thread): the ballot's own choice when the prompt is on it,
    else the same rule export uses, the first fully safe thread of the attempt the ballot uses."""
    if pid in key:
        picks = {m: i for m, i in key[pid]["threads"].items() if m in models}
        last = [m for m in models if m not in picks]
    else:
        picks = {m: t for m in models if (t := cg.playable_thread(cg.read_record(directory / m / f"{pid}.json"))) is not None}
        last = [m for m in models if m not in picks]
    return {m: cg.final(cg.read_record(directory / m / f"{pid}.json"))["threads"][t] for m, t in picks.items()}, last


def user_examples(directory: pathlib.Path, models: list[str], key: dict, votes: dict, prompts_by_id: dict) -> dict[str, dict]:
    """Every prompt the user ranked (two or more bits shown): its setting, its options under the user's own letters, and
    their ranking."""
    out = {}
    for pid, k in key.items():
        ranking = cg.parse_ranking(votes.get(pid, ""), "".join(k["letters"])) if len(k["letters"]) >= 2 else None
        if ranking:
            threads, _ = shown_threads(directory, models, pid, key)
            out[pid] = {"setting": setting_line(prompts_by_id[pid]), "ranking": ranking,
                        "options": {letter: threads[m] for letter, m in k["letters"].items()}}
    return out


def target_letters(pid: str, order: int, shown: list[str], seed: int = cg.SEED) -> dict[str, str]:
    """Letters for a prompt's bits in one judging order, shuffled apart from the user's letters and from other orders."""
    models = list(shown)
    random.Random(f"{seed}:judge:{pid}:{order}").shuffle(models)
    return dict(zip(cg.LETTERS, models))


# --- asking, with a cache ----------------------------------------------------------------------------------------

def cache_file(directory: pathlib.Path, judge: str, kind: str, pid: str, order: int, example_ids: list[str],
               examples_digest: str) -> pathlib.Path:
    key = hashlib.sha256(f"{judge}\x00{prompt_digest()}\x00{examples_digest}\x00{pid}\x00{order}".encode()).hexdigest()[:16]
    return directory / "judge_cache" / judge / f"{kind}-{pid}-o{order}-{key}.json"


def examples_digest(examples: list[dict]) -> str:
    """What the judge was shown, so a changed vote or a different set of examples never reuses an old reply."""
    return hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()[:12]


def ask(client, spec: cg.Spec, system: str, user: str, letters: dict[str, str], pacer: cg.Pacer,
        path: pathlib.Path, first: bool) -> dict:
    """One judge call, cached. Returns {ok, order: [models best first] or None, reason, tokens, latency}. A config failure
    on the first call of a run raises `ConfigError`; other failures are recorded (and cached)."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    pacer.wait(spec.bucket, spec.gap_s)
    before = (client.input_tokens, client.output_tokens)
    start = cg.time.perf_counter()
    reply = failure = None
    try:
        reply = client.chat(system, user, json_reply=REPLY_SCHEMA, validate=lambda r: _parse(r, letters))
        ranking = _parse(reply, letters)
    except Exception as e:  # LLMError from the client, or a reply that is still not a ranking after the re-ask
        ranking, failure = None, e
    latency = cg.time.perf_counter() - start
    pacer.mark(spec.bucket)
    if failure is not None and first and cg.failure_kind(failure) == "config":
        raise cg.ConfigError(f"judge {spec.key}: {type(failure).__name__}: {failure}")
    rec = {"ok": ranking is not None, "order": None if ranking is None else [letters[c] for c in ranking],
           "reason": None if failure is None else f"{type(failure).__name__}: {failure}"[:200], "latency_s": latency,
           "input_tokens": client.input_tokens - before[0], "output_tokens": client.output_tokens - before[1],
           "kind": None if failure is None else cg.failure_kind(failure)}
    if rec["kind"] not in ("config", "infra"):  # an outage is not the judge's answer: ask again next run
        cg.write_record(path, rec)
    return rec


def _parse(reply: str, letters: dict[str, str]) -> str:
    ranking = cg.parse_ranking(str(json.loads(llm.strip_fence(reply))["ranking"]), "".join(letters))
    if not ranking:
        raise ValueError("not a ranking of the letters")
    return ranking


def judge_prompt(client, spec, directory, judge: str, pid: str, item: dict, examples: dict[str, dict], kind: str,
                 pacer: cg.Pacer, first: set, seed: int = cg.SEED) -> dict:
    """The judge's combined view of one prompt: {ranks: {model: mean rank}, calls: [...]}, ranks None when no order
    produced a ranking. `item` is {setting, shown: {model: lines}, last: [models]}."""
    ex = [examples[k] for k in sorted(examples)]
    digest = examples_digest(ex)
    calls, per_order = [], []
    for order in range(ORDERS):
        letters = target_letters(pid, order, list(item["shown"]), seed)
        system, user = build_prompt(ex, block(item["setting"], {L: item["shown"][m] for L, m in letters.items()}))
        path = cache_file(directory, judge, kind, pid, order, sorted(examples), digest)
        call = ask(client, spec, system, user, letters, pacer, path, first=spec.key in first)
        first.discard(spec.key)
        calls.append({**call, "shown": len(letters), "first_shown": letters["A"]})
        if call["ok"]:
            per_order.append({m: float(i) for i, m in enumerate(call["order"], 1)})
    if not per_order:
        return {"ranks": None, "calls": calls}
    mean = {m: float(np.mean([o[m] for o in per_order])) for m in item["shown"]}
    return {"ranks": cg.assemble_ranks(mean, item["last"]), "calls": calls}


# --- the leave-one-out check -------------------------------------------------------------------------------------

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


def model_bias(judge_ranks: dict[str, dict[str, float]], user_ranks: dict[str, dict[str, float]], models: list[str],
               n_boot: int = cg.N_BOOT, seed: int = cg.SEED) -> dict:
    """Per model: the mean of (judge rank - user rank) over the prompts both ranked, a bootstrap 95% CI over prompts, and
    whether the CI lies beyond +-BAR_BIAS. Positive: the judge puts the model lower than the user does."""
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


def position_bias(calls: list[dict], order_key: str = "order") -> dict:
    """How often the bit shown first (letter A) is ranked top, against chance (the mean of 1 / bits shown)."""
    ok = [c for c in calls if c["ok"]]
    if not ok:
        return {"n": 0, "first_top": None, "chance": None}
    return {"n": len(ok), "first_top": float(np.mean([c["order"][0] == c["first_shown"] for c in ok])),
            "chance": float(np.mean([1 / c["shown"] for c in ok]))}


def gate(agree: dict, bias: dict) -> tuple[bool, str]:
    if not agree["pass"]:
        return False, (f"top-pick agreement {agree['rate']:.0%} (95% CI {agree['ci95'][0]:.0%} to {agree['ci95'][1]:.0%}) "
                       f"does not clear {BAR_AGREEMENT:.0%} with a lower end above chance {agree['chance']:.0%}: the user's votes decide alone")
    if not bias["pass"]:
        bad = ", ".join(m for m, v in bias["models"].items() if v["blocks"])
        return False, f"the judge agrees on top picks but is biased on {bad} (CI beyond {BAR_BIAS} ranks): the user's votes decide alone"
    return True, "the judge agrees with the user and shows no model-specific bias: its rankings of the later prompts join the votes"


def leave_one_out(client, spec, directory, judge, items, user_ex, key, votes_ranks, models, pacer, first) -> dict:
    """Judge each of the user's ranked prompts seeing only the others. `items[pid]` as in `judge_prompt`."""
    credits, chances, judged, calls, judge_r, user_r, rows = [], [], {}, [], {}, {}, {}
    for pid in sorted(user_ex):
        others = {k: v for k, v in user_ex.items() if k != pid}
        res = judge_prompt(client, spec, directory, judge, pid, items[pid], others, "loo", pacer, first)
        calls += res["calls"]
        shown = list(items[pid]["shown"])
        user_top = min(shown, key=lambda m: votes_ranks[pid][m])
        chances.append(1 / len(shown))
        if res["ranks"] is None:
            credits.append(0.0)  # an unjudged prompt counts against the judge
            rows[pid] = {"judged": False}
            continue
        credits.append(top_credit(res["ranks"], shown, user_top))
        judge_r[pid], user_r[pid] = res["ranks"], votes_ranks[pid]
        rows[pid] = {"judged": True, "credit": credits[-1], "shown": len(shown)}
    return {"credits": credits, "chances": chances, "judge_ranks": judge_r, "user_ranks": user_r, "calls": calls, "rows": rows}


# --- the run -----------------------------------------------------------------------------------------------------

def run_judge(models: list[str], n: int, n_ballot: int, make_client: Callable[[cg.Spec], object], judge: str,
              directory: pathlib.Path | None = None, pacer: cg.Pacer | None = None, log=print) -> dict:
    """Needs every ballot prompt that has two or more bits shown to be voted on. Writes `judge_result.json` (numbers and
    prompt ids only) and returns it."""
    directory, pacer = directory or cg.run_dir(), pacer or cg.Pacer()
    spec = cg.MODELS[judge]
    kp = cg.key_path(directory)
    key = {p: k for p, k in json.loads(kp.read_text(encoding="utf-8")).items()
           if p in {q["id"] for q in cg.prompts(n_ballot)}}
    votes = cg.read_votes(directory / "votes.csv")
    missing = [p for p, k in key.items() if len(k["letters"]) >= 2 and not cg.parse_ranking(votes.get(p, ""), "".join(k["letters"]))]
    if missing:
        raise SystemExit(f"vote on every ballot prompt first ({len(missing)} left, e.g. {missing[:3]}): the judge learns from all of them")
    by_id = {p["id"]: p for p in cg.prompts(n)}
    ex = user_examples(directory, models, key, votes, by_id)
    user_r = cg.ranks_from_votes(votes, key, models)
    client, first = make_client(spec), {spec.key}

    def item(pid):
        shown, last = shown_threads(directory, models, pid, key)
        return {"setting": setting_line(by_id[pid]), "shown": shown, "last": last}

    items = {pid: item(pid) for pid in ex}
    loo = leave_one_out(client, spec, directory, judge, items, ex, key, user_r, models, pacer, first)
    log(f"leave-one-out: {sum(loo['credits']):.1f} of {len(loo['credits'])} top picks")
    agree = agreement(loo["credits"], loo["chances"])
    bias = model_bias(loo["judge_ranks"], loo["user_ranks"], models)
    ok, reason = gate(agree, bias)
    new_ids = [p["id"] for p in cg.prompts(n)[n_ballot:]]
    new_ranks, new_calls, unjudged, auto = {}, [], [], []
    for pid in new_ids:  # judged either way (it is cheap next to the check), joined only if the gate passed
        it = item(pid)
        if len(it["shown"]) < 2:  # one bit or none: nothing to rank, as on the ballot
            new_ranks[pid] = cg.assemble_ranks({m: float(i) for i, m in enumerate(it["shown"], 1)}, it["last"])
            auto.append(pid)
            continue
        res = judge_prompt(client, spec, directory, judge, pid, it, ex, "new", pacer, first)
        new_calls += res["calls"]
        if res["ranks"] is None:
            unjudged.append(pid)
        else:
            new_ranks[pid] = res["ranks"]
    calls = loo["calls"] + new_calls
    tin, tout = sum(c["input_tokens"] for c in calls), sum(c["output_tokens"] for c in calls)
    result = {"judge_model": spec.model, "prompt_digest": prompt_digest(), "ballot_digest": cg.ballot_digest(directory),
              "n_ballot": n_ballot, "n_examples": len(ex), "orders": ORDERS, "agreement": agree, "bias": bias,
              "position_bias": {"leave_one_out": position_bias(loo["calls"]), "new": position_bias(new_calls),
                                "all": position_bias(calls)},
              "joined": ok, "reason": reason, "n_calls": len(calls), "n_failed_calls": sum(not c["ok"] for c in calls),
              "unjudged_new": unjudged, "auto_ranked_new": auto,
              "input_tokens": tin, "output_tokens": tout,
              "cost_usd": (tin * spec.price_in + tout * spec.price_out) / 1e6,
              "loo": loo["rows"], "judge_ranks": new_ranks}
    cg.write_record(cg.judge_path(directory), result)
    return result


def summary(res: dict) -> str:
    a, b, p = res["agreement"], res["bias"], res["position_bias"]["all"]
    biases = ", ".join(f"{m} {v['mean']:+.2f} ({v['ci95'][0]:+.2f} to {v['ci95'][1]:+.2f})" for m, v in b["models"].items())
    first = "n/a" if p["first_top"] is None else f"{p['first_top']:.0%} (chance {p['chance']:.0%})"
    return (f"judge {res['judge_model']}: top-pick agreement {a['rate']:.0%} of {a['n']} (95% CI {a['ci95'][0]:.0%} to "
            f"{a['ci95'][1]:.0%}, chance {a['chance']:.0%}); judge minus user rank by model: {biases}; first-shown bit ranked "
            f"top {first}; {res['n_calls']} calls, ${res['cost_usd']:.2f}.\n{res['reason']}.")


def main(models: list[str], n: int, n_ballot: int, judge: str | None) -> None:
    judge = judge or JUDGE_MODEL
    if judge not in cg.MODELS:
        sys.exit(f"unknown judge {judge!r}; choose from {list(cg.MODELS)}")
    import os
    if judge == "claude-haiku-5-5" and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set: add a paid key to backend/.env, or pass --judge-model gemini-flash-lite")

    def make(spec):
        timeout, deadline = JUDGE_TIMING
        return llm.from_env(timeout, role="joke_writer", provider=spec.provider, model=spec.model, retries=JUDGE_RETRIES,
                            deadline_s=deadline)

    try:
        print(summary(run_judge(models, n, n_ballot, make, judge)))
    except cg.ConfigError as e:
        sys.exit(f"stopped, nothing cached for the failing call: {e}")
