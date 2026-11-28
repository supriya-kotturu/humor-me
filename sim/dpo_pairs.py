"""Exp 36 (D-121): does the pairwise judge agree with the user on two threads written by the same model?

DPO needs pairs of threads labelled "which is funnier". The user's taste cannot label thousands, so a judge will, but only
if it first agrees with the user on same-model pairs, which are much closer than the cross-model ballot of Exp 24 (D-102).
This module writes the candidate pairs, gives the user the ranking tool, and runs that gate.

Usage (Ollama up with `OLLAMA_NUM_PARALLEL=3` and `OLLAMA_CONTEXT_LENGTH=8192`; do NOT run `write` during a live show, it
shares the one GPU and the show's 12 s bit deadline would suffer):
    python -m sim.dpo_pairs pool                              # once: freeze the prompts (reads data/tonight/**/fetch.json)
    python -m sim.dpo_pairs write --split ranking --env-file backend/.env [--n N]   # in the background; resumable
    python -m sim.dpo_pairs write --split train   --env-file backend/.env [--n N]   # can run while you rank
    python -m sim.dpo_pairs topup --split ranking --env-file backend/.env   # after write: bits with under 2 clean threads get up to 2 more batches
    python -m sim.dpo_pairs rank [--limit 20]                 # you pick the funnier thread of a pair, in sittings
    python -m sim.dpo_pairs judge --env-file backend/.env     # the gate: 120 calls (60 test pairs, both orders)
    python -m sim.dpo_pairs score --record                    # numbers -> reports/metrics/dpo_judge_gate.json
    python -m sim.dpo_pairs judge --model opus --env-file backend/.env   # D-122, Exp 37: same pairs, Opus 5.5, own cache
    python -m sim.dpo_pairs score --model opus --record       # -> reports/metrics/dpo_judge_gate_opus.json
    python -m sim.dpo_pairs pool --split train --rebuild      # later, with 3+ days stored, before any judge labelling

D-123 (the pool for training, see sim/dpo_train.py and sim/dpo_eval.py; the full run order is in explorer/README.md):
    python -m sim.dpo_pairs write --split train --writer qwen3:4b-instruct --env-file backend/.env   # -> data/dpo/bits/train-qwen3/
    python -m sim.dpo_pairs label --split train --writer qwen3:4b-instruct --env-file backend/.env   # Haiku, both orders
    python -m sim.dpo_pairs export --writer qwen3:4b-instruct                                         # -> data/dpo/train/
    python -m sim.dpo_pairs eval write|plan|power|bar|vote|judge|score ...                             # Exp 38
`--writer` on write and topup names the model that writes (default: the live writer, JOKE_WRITER_MODEL or bonsai-2:4b); its bits
go to their own folder and every record carries its name, so one writer's bits never answer for another's.

Prompts. `pool` writes `data/dpo/pool.json` once (gitignored, it holds headline gists): a seeded pool of bit requests built the
way the live show builds them (`ThreadGenerator`, gen-v5, material off, tonight's context on for half). Two disjoint
splits, fixed before any data: RANKING_N prompts for the 120 pairs you rank, TRAIN_N for DPO later. Half of every split
carries tonight's context (market line, and an approved headline gist as the topic on bits 2 and 6, as in a show), half
does not. Modes are round-robin, rooms are seeded, the bit number runs 0 to 7 (a later bit has premises already told).
No prompt equals one of Exp 24's ballot or of Exp 26, 27 or 30 (`assert_disjoint`). Tonight prompts use only gists you
approved in `sim.tonight fetch`; with few days of them a gist is reused by several prompts (round-robin, seeded).

Writing. One prompt is one bit is 3 parallel calls on the live writer, then the show's safety check on every line. A thread
with any rejected line is kept in the record (`safe: false`) and never enters a pair. One bit at a time. A bit with no
thread is written again on the next run, and three in a row stop the writer (Ollama is down).

Ranking. Each ranking prompt is assigned to teach or test now, by seed, before any answer: per tonight half, 30 teach and
30 test pairs (30 in each of 4 cells); prompts sharing a headline gist share a role. One pair per prompt: two clean
threads of the bit, sides random. Answer 1, 2, t (tie) or s (skip). Ties and skips are not counted and the prompt is
replaced by the next one of its cell, so you end with 120 decided pairs; both counts are recorded. Neither the order of
questions nor the progress line tells you which pairs are test.

Gate (D-102's bar, D-121's rule for pairs the judge answers inconsistently; both set before any data). The judge is taught
with the 60 teach pairs as examples and asked on each of the 60 test pairs in both orders. A pair whose two orders name
different threads gets no label (and is dropped from DPO later). On the consistent pairs: let c be how many there are and a
how many agree with you. Pass iff c >= 30 (at least half of 60) AND the Wilson 95% lower bound of a/c is above 50%.
Reported: a, c, the share of pairs whose orders disagree, agreement per tonight half; the strict count (a of 60) and
the half-credit rate are beside it, never deciding. Generated text lives only under `data/dpo/`.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import random
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))
from app import generate, llm, safety  # noqa: E402
from app import tonight as app_tonight  # noqa: E402
from sim import cache  # noqa: E402
from sim import compare_generators as cg  # noqa: E402
from sim import judge_calibration as jc  # noqa: E402
from sim import thread_coherence as tc  # noqa: E402
from sim import thread_timing as tt  # noqa: E402
from sim.blind import plain  # noqa: E402
from sim.env import AGES, ENERGIES, VENUES  # noqa: E402

NAME = "dpo_judge_gate"
SEED = "dpo-pool-v1"
RANK_SEED = "dpo-rank-v1"
RERANK_SEED = "dpo-rerank-v1"
OPUS_MODEL = "claude-opus-5-5"
# D-122: the second look at the same 60 test pairs. Two looks, so the bar is Bonferroni-corrected: 97.5% instead of
# 95% (z = NormalDist().inv_cdf(0.9875), pinned so the record's key is stable). Judgment call on the correction method.
Z = {"haiku": 1.96, "opus": 2.2414}
LEVEL = {"haiku": "95", "opus": "975"}
RECORD = {"haiku": "dpo_judge_gate", "opus": "dpo_judge_gate_opus"}
OPUS_TIMING = (90.0, 240.0)  # adaptive thinking is slower than the safety check's 12 s / 25 s; not a quality setting
RERANK_PER_CELL = 5  # 20 pairs, 10 teach and 10 test: the user's choice (2026-11-26); reported only, no bar
RANKING_N = 240  # 120 decided pairs plus room for ties, skips and bits with fewer than 2 clean threads: judgment call, not measured
TRAIN_N = 800  # judgment call, not measured; bounded by the free (mode, room) combinations of a first bit, see build_pool
HALVES = ("tonight", "plain")
ROLES = ("teach", "test")
CELLS = tuple((r, h) for h in HALVES for r in ROLES)
PER_CELL = 30
N_PAIRS = PER_CELL * len(CELLS)  # 120
N_TEST = PER_CELL * len(HALVES)  # 60
MIN_CONSISTENT = N_TEST // 2  # D-121: the gate needs at least half of the 60 test pairs answered the same way in both orders
BITS_PER_SHOW = 8
MAX_TOLD = tc.MAX_TOLD
DATA = ROOT / "data" / "dpo"
ORDERS = ("ab", "ba")  # "ab": the judge's A is the thread you saw on the left

# Placeholder premises for "already told", none from Exp 27's or Exp 30's lists.
TOPICS = ("a hotel minibar", "a dentist's chair", "a smart doorbell", "a wedding seating chart", "a car wash",
          "a tax form", "an escape room", "a farmers market", "a ride share", "a hospital parking lot",
          "a laundromat", "a coat check", "a trivia night", "a phone upgrade", "a fire drill", "a pet groomer",
          "a gift exchange", "a train delay", "a conference badge", "a tip jar", "a rental car counter",
          "a potluck", "a library book sale", "a weather app")

JUDGE_SYSTEM = """You judge stand-up material. You get two bits, A and B. Each is five lines performed in order by a self-aware, slightly anxious robot. Both were written by the same model for the same request, so they are close in quality.

One person's taste decides which is funnier. Below are {n} pairs they ranked, with their pick. Learn their taste from them, then pick the funnier bit of the new pair the way they would. Judge funniness only.

{examples}

Reply with JSON only: {{"funnier": "A"}} or {{"funnier": "B"}}."""
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["funnier"],
                "properties": {"funnier": {"type": "string", "enum": ["A", "B"]}}}


def stamp() -> str:
    """When an answer was given (UTC): the ranking took far longer than planned and nothing recorded by how much."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _write(path: pathlib.Path, rec: dict) -> None:
    tc._write(path, rec)


def _read(path: pathlib.Path) -> dict | None:
    return tc._read(path)


def rerank_path(root: pathlib.Path) -> pathlib.Path:
    return root / "rerank.json"


def pool_path(root: pathlib.Path | None = None) -> pathlib.Path:
    return (root or DATA) / "pool.json"


def live_writer() -> str:
    return tc.writer_model()


def writer_slug(writer: str) -> str:
    """The short name a writer's bits folder carries: `qwen3:4b-instruct` is `qwen3` (the record keeps the full name and every
    read checks it, so two models with one slug cannot share a folder unnoticed)."""
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in writer.split(":")[0])


def bits_folder(split: str, writer: str | None = None) -> str:
    """`train` for the first writer (bonsai, the folders written before D-123), `train-qwen3` for another (D-123)."""
    return split if writer in (None, llm.OLLAMA_MODEL) else f"{split}-{writer_slug(writer)}"


def bit_path(root: pathlib.Path, split: str, pid: str, writer: str | None = None) -> pathlib.Path:
    return root / "bits" / bits_folder(split, writer) / f"{pid}.json"


def rank_path(root: pathlib.Path) -> pathlib.Path:
    return root / "rank.json"


# --- tonight's stored days -----------------------------------------------------------------------------------------------

def load_days(directory: pathlib.Path | None = None) -> list[dict]:
    """One entry per day found under `directory` (default data/tonight): the newest fetch of each date, only its approved gists.
    An older fetch of the same day (before a rules change) is ignored, never merged."""
    best: dict[str, dict] = {}
    for path in sorted((directory or app_tonight.DATA).rglob("fetch.json")):
        rec = _read(path)
        if not rec or not rec.get("approved") or not rec.get("date"):
            continue
        if rec["date"] not in best or rec.get("fetched_at", "") > best[rec["date"]].get("fetched_at", ""):
            best[rec["date"]] = rec
    days = []
    for date, rec in sorted(best.items()):
        approved = list(rec["approved"])
        days.append({"date": date, "approved": approved, "facts": list(rec.get("facts", [])),
                     "market": (rec.get("market") or {}).get("line"),
                     "names": [n for g in approved for n in rec.get("keep", {}).get(g, [])],
                     "people": [n for g in approved for n in rec.get("people", {}).get(g, [])]})
    return days


# --- the prompt pool -----------------------------------------------------------------------------------------------------

def earlier_keys() -> set[tuple]:
    """The `prompt_key` of every prompt Exp 24 (ballot and spares), 26, 27 and 30 sent, all with
    no tonight context."""
    out = set()
    sets = [cg.prompts(65), tt.prompts(20), tc.prompts(tc.N_EXT), jc.prompts(jc.N_PROMPTS)]
    for prompts in sets:
        for p in prompts:
            out.add((p["mode"], p["context"]["venue"], p["context"]["energy"], p["context"]["age"], tuple(p.get("told", ())), "plain", None, None))
    return out


def prompt_key(p: dict) -> tuple:
    """What the request is made of. The bit number is not in it: it reaches the request only through the premises told
    and, for the tonight half, whether a gist is the topic; the tonight half's background is its day's."""
    c = p["context"]
    return (p["mode"], c["venue"], c["energy"], c["age"], tuple(p["told"]), p["half"], p["day"], p["gist"])


def build_pool(days: list[dict], n_ranking: int = RANKING_N, n_train: int = TRAIN_N, seed: str = SEED,
               keep_ranking: list[dict] | None = None) -> dict:
    """The frozen prompt pool. Split, half, role, mode, bit and gist are fixed by position and seed; only a room that would
    repeat an earlier prompt is redrawn. Raises when a half cannot be built (no approved gist, no free room).
    `keep_ranking` (a frozen ranking split) builds only the training split around it, disjoint from it."""
    if not days or not any(d["approved"] for d in days):
        raise SystemExit("no approved gists under data/tonight: the tonight half needs `python -m sim.tonight fetch` output "
                         "(junction data/tonight from the checkout that has it)")
    gists = [(d["date"], g) for d in days for g in d["approved"]]
    random.Random(f"{seed}:gists").shuffle(gists)
    headline = 0  # next gist: round-robin over the shuffled list
    taken = earlier_keys()
    modes = tuple(generate.MODES)
    prompts: list[dict] = [dict(p) for p in keep_ranking or []]
    taken |= {prompt_key(p) for p in prompts}
    bumped = 0
    for split, n in (("train", n_train),) if keep_ranking else (("ranking", n_ranking), ("train", n_train)):
        if n % (2 * len(ROLES) if split == "ranking" else 2):
            raise ValueError(f"{split} size {n} must split evenly into halves" + (" and roles" if split == "ranking" else ""))
        cells = {h: 0 for h in HALVES}
        halves = [h for _ in range(n // 2) for h in HALVES]
        random.Random(f"{seed}:{split}:halves").shuffle(halves)
        for i, half in enumerate(halves):
            k = cells[half]
            cells[half] += 1
            bit = k % BITS_PER_SHOW
            p = {"id": f"{'r' if split == 'ranking' else 't'}{i:04d}", "split": split, "half": half, "role": None,
                 "mode": modes[(k // BITS_PER_SHOW + bit) % len(modes)], "bit": bit, "day": None, "gist": None}
            if half == "tonight" and bit % app_tonight.HEADLINE_EVERY == app_tonight.HEADLINE_REMAINDER:
                p["day"], p["gist"] = gists[headline % len(gists)]
                headline += 1
            elif half == "tonight":
                p["day"] = days[(k // BITS_PER_SHOW) % len(days)]["date"]
            r = random.Random(f"{seed}:{p['id']}")
            while True:
                for _ in range(500):
                    p["context"] = {"venue": r.choice(VENUES), "energy": r.choice(ENERGIES), "age": r.choice(AGES)}
                    p["told"] = r.sample(TOPICS, min(p["bit"], MAX_TOLD))
                    if prompt_key(p) not in taken:
                        break
                else:
                    # Only a plain first bit (nothing told) can run out: Exp 24 and 26 already used most of its 27 rooms per
                    # mode. It becomes a second bit (one premise told) rather than repeat an earlier request.
                    if p["bit"] != 0 or half != "plain":
                        raise SystemExit(f"{split} pool too big: no free room for {p['id']} (bit {p['bit']}); lower the size")
                    p["bit"], bumped = 1, bumped + 1
                    continue
                break
            taken.add(prompt_key(p))
            prompts.append(p)
    # Roles: per tonight half, half of the ranking prompts teach and half test, shuffled once now, before any answer.
    # Prompts sharing a headline gist go to one side together, so no test pair has a teach example's topic.
    for half in () if keep_ranking else HALVES:
        mine = [p for p in prompts if p["split"] == "ranking" and p["half"] == half]
        random.Random(f"{RANK_SEED}:roles:{half}").shuffle(mine)
        groups: dict[str, list[dict]] = {}
        for p in mine:
            groups.setdefault(p["gist"] or p["id"], []).append(p)
        count = dict.fromkeys(ROLES, 0)
        for group in sorted(groups.values(), key=len, reverse=True):  # stable: shuffled order among equal sizes
            role = min(ROLES, key=lambda r: (count[r], ROLES.index(r)))
            for p in group:
                p["role"] = role
            count[role] += len(group)
        if len(set(count.values())) != 1:
            raise SystemExit(f"ranking roles do not split evenly in the {half} half: {count}")
    pool = {"seed": seed, "rank_seed": RANK_SEED, "days": days, "bumped_first_bits": bumped, "prompts": prompts}
    assert_disjoint(pool)
    pool["digest"] = pool_digest(pool)
    return pool


def pool_digest(pool: dict) -> str:
    body = json.dumps({"seed": pool["seed"], "days": pool["days"], "prompts": pool["prompts"]}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()[:8]


def assert_disjoint(pool: dict) -> None:
    """Ranking and train share no prompt, and neither shares one with Exp 24's ballot or Exp 26, 27, 30."""
    seen: dict[str, set] = {"ranking": set(), "train": set()}
    ids = set()
    for p in pool["prompts"]:
        assert p["id"] not in ids, f"duplicate id {p['id']}"
        ids.add(p["id"])
        seen[p["split"]].add(prompt_key(p))
    assert not seen["ranking"] & seen["train"], "a ranking prompt equals a train prompt"
    earlier = earlier_keys()
    for split, keys in seen.items():
        assert not keys & earlier, f"a {split} prompt equals one of Exp 24, 26, 27 or 30"
    ranking = [p for p in pool["prompts"] if p["split"] == "ranking"]
    for h in HALVES:
        for r in ROLES:
            assert sum(p["half"] == h and p["role"] == r for p in ranking) >= PER_CELL, f"cell {r}/{h} has too few prompts"


def ensure_pool(root: pathlib.Path | None = None, tonight_dir: pathlib.Path | None = None, rebuild: bool = False) -> dict:
    path = pool_path(root)
    existing = None if rebuild else _read(path)
    if existing is not None:
        return existing
    pool = build_pool(load_days(tonight_dir))
    _write(path, pool)
    return pool


def rebuild_train(root: pathlib.Path | None = None, tonight_dir: pathlib.Path | None = None) -> dict:
    """Rebuilds only the training split, from the days now stored (more days, D-121), around the frozen ranking split.
    Refuses once any training bit has been labelled (`data/dpo/labels/`). Training bits written for the old split are moved
    aside, not deleted (their ids may now name other prompts); a bit whose prompt and day are unchanged is kept. The
    ranking prompts' days must be unchanged. Stop any `write --split train` first."""
    root = root or DATA
    old = _read(pool_path(root))
    if old is None:
        raise SystemExit("no pool yet: run `python -m sim.dpo_pairs pool` first")
    labels = root / "labels"
    if labels.exists() and any(labels.iterdir()):
        raise SystemExit("training bits are already labelled (data/dpo/labels): the training split is frozen")
    other = sorted(f.name for f in (root / "bits").glob("train-*")) if (root / "bits").exists() else []
    if other:
        raise SystemExit(f"bits were written for the training split by another writer ({', '.join(other)}): the split is frozen")
    days = load_days(tonight_dir)
    now = {d["date"]: d for d in days}
    ranking = [p for p in old["prompts"] if p["split"] == "ranking"]
    for d in old["days"]:
        if any(p["day"] == d["date"] for p in ranking) and now.get(d["date"]) != d:
            raise SystemExit(f"the stored fetch for {d['date']} is missing or changed: the ranking split was written with it")
    pool = build_pool(days, keep_ranking=ranking)
    bits = root / "bits" / "train"
    if bits.exists():
        aside = bits.with_name(f"train.before-{old['digest']}")
        k = 1
        while aside.exists():
            k += 1
            aside = bits.with_name(f"train.before-{old['digest']}-{k}")
        bits.rename(aside)
        # A prompt the rebuild left unchanged (the whole plain half, and tonight prompts of an unchanged day) keeps its bit.
        before = {p["id"]: p for p in old["prompts"] if p["split"] == "train"}
        old_days = {d["date"]: d for d in old["days"]}
        for p in (q for q in pool["prompts"] if q["split"] == "train"):
            src = aside / f"{p['id']}.json"
            if before.get(p["id"]) != p or (p["day"] is not None and old_days.get(p["day"]) != now.get(p["day"])):
                continue
            if same_prompt(_read(src), p):
                dest = bit_path(root, "train", p["id"])
                dest.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dest)
    _write(pool_path(root), pool)
    return pool


def sizes(pool: dict) -> dict:
    return {f"{s}/{h}": sum(p["split"] == s and p["half"] == h for p in pool["prompts"]) for s in ("ranking", "train") for h in HALVES}


# --- writing -------------------------------------------------------------------------------------------------------------

def tonight_for(prompt: dict, days: dict[str, dict]):
    """The show's context for a tonight prompt (its gist first, so `draw` gives it on a headline bit), None for plain."""
    if prompt["half"] != "tonight":
        return None
    day = days[prompt["day"]]
    gists = ([prompt["gist"]] if prompt["gist"] else []) + [g for g in day["approved"] if g != prompt["gist"]]
    return app_tonight.TonightContext(day["facts"], day["market"], gists, allowed_names=day["names"], allowed_people=day["people"])


def new_generator(client, prompt: dict, days: dict[str, dict]):
    """The generator a show would hold when it reaches this bit: live prompt, material off, premises told, and for the tonight
    half the day's context."""
    gen = generate.ThreadGenerator(client, prompt_id=generate.PROMPT_ID, told=prompt["told"])
    if (t := tonight_for(prompt, days)) is not None:
        gen.set_tonight(t)
    return gen


def set_allow(live_safety, prompt: dict, days: dict[str, dict]) -> None:
    """One LiveSafety serves the whole split, so its allow lists are set for every bit: the day's for a tonight bit, none
    for a plain one (a plain bit after a tonight bit would otherwise pass names a plain show rejects)."""
    t = tonight_for(prompt, days)
    live_safety.allow_caps = t.allow_caps if t else frozenset()
    live_safety.allow_people = t.allow_people if t else frozenset()


def request_digest(request: str, generator_id: str, writer: str | None = None) -> str:
    parts = [request, generator_id, str((generate.K, generate.JOKES, generate.TEMPERATURE, generate.GENERATOR_TIMING,
                                         generate.RETRIES, generate.WRITER_CONTEXT)), writer or tc.writer_model()]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:12]


class Capture:
    """Wraps a writer client to keep every reply it returns, so a thread can be stored with the exact text the model wrote
    (D-123: SFT and DPO train on that text, not on a re-serialisation of the parsed lines). Everything else passes through."""

    def __init__(self, client):
        self._client, self.replies, self._lock = client, [], threading.Lock()

    def chat(self, *args, **kwargs):
        reply = self._client.chat(*args, **kwargs)
        with self._lock:
            self.replies.append(str(reply))
        return reply

    def __getattr__(self, name):
        return getattr(self._client, name)


def attach_raw(threads: list[dict], replies: list[str]) -> None:
    """Gives each thread the reply that parses to it. A reply that does not parse (a bad shape that was asked again) matches no
    thread; a thread with no match keeps no `raw`, and export skips it."""
    parsed = {}
    for text in replies:
        try:
            premise, lines = generate.parse(text)
        except generate.GenerationError:
            continue
        parsed.setdefault((premise, tuple(lines)), text)
    for t in threads:
        if (raw := parsed.get((t["premise"], tuple(t["lines"])))) is not None:
            t["raw"] = raw


def check_threads(threads: list[dict], live_safety, age: str) -> bool:
    """Marks every thread `safe` (all five lines pass) and keeps the rejected lines' reasons. Returns whether the safety
    layer itself failed (its own call, not a verdict about the lines)."""
    live_safety.prefetch([line for t in threads for line in t["lines"]], age)
    infra = False
    for t in threads:
        verdicts = [live_safety.check(line, age) for line in t["lines"]]
        t["safe"] = all(v.ok for v in verdicts)
        t["rejected"] = [v.reason for v in verdicts if not v.ok]
        infra = infra or any(r.startswith("llm:error") or r == "llm:no verdict" for r in t["rejected"])
    return infra


def write_bit(client, live_safety, prompt: dict, days: dict[str, dict]) -> dict:
    cap = Capture(client)
    gen = new_generator(cap, prompt, days)
    set_allow(live_safety, prompt, days)
    bad0 = client.bad_replies
    start = time.perf_counter()
    rec = {"id": prompt["id"], "split": prompt["split"], "half": prompt["half"], "mode": prompt["mode"], "bit": prompt["bit"],
           "context": prompt["context"], "told": prompt["told"], "day": prompt["day"], "gist": prompt["gist"],
           "generator_id": gen.generator_id, "writer": client.model, "request": None, "threads": [], "lost": [], "failure": None}
    try:
        gen.generate(prompt["mode"], prompt["bit"], 0, prompt["context"])
    except Exception as e:  # noqa: BLE001 - a bit with no thread is a result, not a crash
        rec["failure"] = type(e).__name__
        rec["lost"] = list(getattr(e, "threads_lost", [])) or [f"{type(e).__name__}: {e}"[:100]]
    got = gen._bits.get(prompt["bit"])
    if got is not None and not isinstance(got, Exception):
        request, lines_major, premises, lost = got
        rec["request"], rec["lost"] = request, list(lost)
        rec["threads"] = [{"premise": premises[i], "lines": [lines_major[j][i] for j in range(generate.JOKES)]}
                          for i in range(len(premises))]
        attach_raw(rec["threads"], cap.replies)
    rec["wall_s"] = time.perf_counter() - start
    rec["bad_replies"] = client.bad_replies - bad0
    rec["digest"] = request_digest(rec["request"] or "", gen.generator_id, client.model)
    return finish(rec, live_safety, prompt["context"]["age"])


def finish(rec: dict, live_safety, age: str) -> dict:
    """The show's safety check on every line; stamps the rules version and says when the layer itself failed."""
    infra = check_threads(rec["threads"], live_safety, age) if rec["threads"] else False
    rec.update(safety_rules=safety.RULES_VERSION, safety_infra_error=infra,
               written_at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"))
    return rec


PROMPT_FIELDS = ("split", "half", "mode", "bit", "context", "told", "day", "gist")
MAX_EMPTY_IN_A_ROW = 3  # judgment call, not measured: three bits with no thread in a row is an outage, not bad luck


def done(rec: dict | None) -> bool:
    """Written for good: threads came back and the safety layer gave a verdict. A bit with no thread is retried."""
    return rec is not None and bool(rec.get("threads")) and not rec.get("safety_infra_error")


def same_prompt(rec: dict | None, p: dict, writer: str | None = None) -> bool:
    """The record was written for this prompt, not for another pool's prompt with the same id (a rebuild while a writer ran),
    and by `writer` when one is given (a record from before D-123 has no writer: it is the first one's, bonsai's)."""
    return (rec is not None and all(rec.get(k) == p[k] for k in PROMPT_FIELDS)
            and (writer is None or rec.get("writer", llm.OLLAMA_MODEL) == writer))


def write(split: str, client, live_safety, pool: dict, root: pathlib.Path | None = None, n: int | None = None,
          out=print) -> int:
    """Writes the missing bits of a split, one at a time, in pool order; a bit whose safety layer failed is re-checked, not
    regenerated, and a bit with no thread is written again. Stops after `n` new bits (None: all), or when
    MAX_EMPTY_IN_A_ROW bits in a row return no thread (the writer is down). Returns how many it wrote."""
    root = root or DATA
    days = {d["date"]: d for d in pool["days"]}
    mine = [p for p in pool["prompts"] if p["split"] == split]
    total = len(mine)
    w = client.model  # the writer: its bits have their own folder and carry its name (D-123)
    ready = sum(done(r) and same_prompt(r, p, w) for p in mine if (r := _read(bit_path(root, split, p["id"], w))) is not None)
    wrote = empty = 0
    for i, p in enumerate(mine, 1):
        path = bit_path(root, split, p["id"], w)
        rec = _read(path)
        if not same_prompt(rec, p, w):
            rec = None
        if done(rec):
            continue
        if n is not None and wrote >= n:
            break
        set_allow(live_safety, p, days)
        if rec is not None and rec["threads"]:  # only the safety layer failed: redo it on the stored threads
            rec = finish(rec, live_safety, p["context"]["age"])
        else:
            rec = write_bit(client, live_safety, p, days)
        _write(path, rec)
        wrote += 1
        ready += done(rec)
        clean = sum(t["safe"] for t in rec["threads"])
        out(f"[{i}/{total}] {p['id']} {p['mode']} {p['half']} bit {p['bit']}: {len(rec['threads'])} threads, {clean} clean, "
            f"{rec['wall_s']:.1f} s ({ready} of {total} done)")
        empty = 0 if rec["threads"] else empty + 1
        if empty >= MAX_EMPTY_IN_A_ROW:
            raise SystemExit(f"{empty} bits in a row came back with no thread (last: {rec['lost'][-1:]}): is Ollama up? "
                             "Finished bits are saved; rerun to resume")
    return wrote


MAX_TOPUPS = 2  # extra batches of 3 threads per bit: judgment call, not measured


def clean_count(rec: dict) -> int:
    return sum(bool(t.get("safe")) for t in rec["threads"])


def topup_bit(client, live_safety, rec: dict, prompt: dict, days: dict[str, dict]) -> dict | None:
    """One more batch of 3 threads for a written bit, on the request stored in the record. Appends the threads that came
    back (the originals stay, in front) and returns the new record; None when no thread came back or the safety layer
    failed (nothing is stored, so a rerun tries again). Raises if the request is not the stored one."""
    cap = Capture(client)
    gen = new_generator(cap, prompt, days)
    set_allow(live_safety, prompt, days)
    start = time.perf_counter()
    try:
        gen.generate(prompt["mode"], prompt["bit"], 0, prompt["context"])
    except Exception:  # noqa: BLE001 - no thread this time is not a crash
        return None
    request, lines_major, premises, lost = gen._bits[prompt["bit"]]
    if request != rec["request"] or gen.generator_id != rec["generator_id"] or request_digest(request, gen.generator_id, client.model) != rec["digest"]:
        raise SystemExit(f"{rec['id']}: the request rebuilt from the pool is not the stored one; nothing was changed")
    batch = len(rec.get("topups", [])) + 1
    new = [{"premise": premises[i], "lines": [lines_major[j][i] for j in range(generate.JOKES)], "batch": batch}
           for i in range(len(premises))]
    attach_raw(new, cap.replies)
    if check_threads(new, live_safety, prompt["context"]["age"]):
        return None
    out = dict(rec)
    out["threads"] = [{**t, "batch": t.get("batch", 0)} for t in rec["threads"]] + new
    out["topups"] = list(rec.get("topups", [])) + [{
        "batch": batch, "digest": request_digest(request, gen.generator_id, client.model), "threads": len(new), "lost": list(lost),
        "wall_s": time.perf_counter() - start, "rules": safety.RULES_VERSION,
        "at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}]
    return out


def topup(split: str, client, live_safety, pool: dict, root: pathlib.Path | None = None, cap: int = MAX_TOPUPS,
          out=print) -> int:
    """For every written bit with fewer than 2 clean threads, writes up to `cap` more batches on the stored request until
    it has 2. Bits with 2 or more are never touched, so a pair already shown stays the pair shown. Same stop as `write`
    (MAX_EMPTY_IN_A_ROW bits with no thread). Returns the number of batches added."""
    root = root or DATA
    days = {d["date"]: d for d in pool["days"]}
    mine = [p for p in pool["prompts"] if p["split"] == split]
    w = client.model
    todo = [(p, r) for p in mine if (r := _read(bit_path(root, split, p["id"], w))) is not None and same_prompt(r, p, w)
            and done(r) and clean_count(r) < 2 and len(r.get("topups", [])) < cap]
    added = empty = 0
    for i, (p, rec) in enumerate(todo, 1):
        while clean_count(rec) < 2 and len(rec.get("topups", [])) < cap:
            new = topup_bit(client, live_safety, rec, p, days)
            if new is None:
                empty += 1
                out(f"[{i}/{len(todo)}] {p['id']}: no usable batch")
                if empty >= MAX_EMPTY_IN_A_ROW:
                    raise SystemExit(f"{empty} batches in a row came back with no thread: is Ollama up? "
                                     "Finished bits are saved; rerun to resume")
                break
            empty = 0
            rec = new
            _write(bit_path(root, split, p["id"], w), rec)
            added += 1
            out(f"[{i}/{len(todo)}] {p['id']} batch {len(rec['topups'])}: {clean_count(rec)} clean of {len(rec['threads'])}")
    return added


# --- ranking -------------------------------------------------------------------------------------------------------------

def read_state(root: pathlib.Path) -> dict:
    return _read(rank_path(root)) or {"seed": RANK_SEED, "answers": {}}


def pair_for(rec: dict | None) -> tuple[int, int] | None:
    """Two clean threads of a bit as (left, right) thread indices, picked and ordered by seed; None with fewer than two clean."""
    if rec is None:
        return None
    clean = [i for i, t in enumerate(rec["threads"]) if t.get("safe")]
    if len(clean) < 2:
        return None
    left, right = random.Random(f"{RANK_SEED}:pair:{rec['id']}").sample(clean, 2)
    return left, right


def decided(state: dict, pool: dict | None = None) -> dict[tuple, int]:
    """Decided pairs (answer 1 or 2) per (role, half) cell."""
    roles = {p["id"]: (p["role"], p["half"]) for p in pool["prompts"] if p["split"] == "ranking"} if pool else {}
    out = {c: 0 for c in CELLS}
    for pid, a in state["answers"].items():
        if a["choice"] in ("1", "2") and pid in roles:
            out[roles[pid]] += 1
    return out


def next_pair(pool: dict, state: dict, root: pathlib.Path) -> tuple[dict, dict, tuple[int, int]] | None:
    """The next prompt to ask about: from a cell with the fewest decided pairs (so a sitting stays balanced; among equal
    cells a seeded draw, so the order of questions does not tell you which are test pairs), the first unanswered prompt
    of its pool order whose bit is written and has two clean threads."""
    counts = decided(state, pool)
    r = random.Random(f"{RANK_SEED}:next:{len(state['answers'])}")
    open_cells = [c for c in CELLS if counts[c] < PER_CELL]
    draws = {c: r.random() for c in open_cells}
    for cell in sorted(open_cells, key=lambda c: (counts[c], draws[c])):
        for p in pool["prompts"]:
            if p["split"] != "ranking" or (p["role"], p["half"]) != cell or p["id"] in state["answers"]:
                continue
            rec = _read(bit_path(root, "ranking", p["id"]))
            if (pair := pair_for(rec)) is not None and done(rec):
                return p, rec, pair
    return None


def counts_line(state: dict, pool: dict) -> str:
    """The progress you see while ranking: no teach / test split, which would tell you which questions are test pairs."""
    d, ans = decided(state, pool), state["answers"].values()
    return (f"{sum(d.values())} of {N_PAIRS} done (ties {sum(a['choice'] == 't' for a in ans)}, "
            f"skips {sum(a['choice'] == 's' for a in ans)})")


def rank(pool: dict, root: pathlib.Path | None = None, limit: int | None = None, read=input, out=print) -> int:
    """Asks about pairs until 120 are decided, `limit` questions are asked, you quit, or the writer has no pair ready.
    Saves after every answer. Returns the number of decided pairs."""
    root = root or DATA
    state = read_state(root)
    state["seed"] = RANK_SEED
    asked = 0
    while sum(decided(state, pool).values()) < N_PAIRS and (limit is None or asked < limit):
        nxt = next_pair(pool, state, root)
        if nxt is None:
            out("No pair ready: the writer has not finished more ranking bits (`write --split ranking`).")
            break
        p, rec, (left, right) = nxt
        out(f"\nPair {sum(decided(state, pool).values()) + 1} of {N_PAIRS}")
        for side, i in (("1", left), ("2", right)):
            out(f"--- {side}\n" + lines_text(rec["threads"][i]["lines"]))
        while True:
            a = read("Which is funnier? [1/2/t=tie/s=skip/q=stop] ").strip().lower()
            if a in ("1", "2", "t", "s", "q"):
                break
        if a == "q":
            break
        state["answers"][p["id"]] = {"choice": a, "left": left, "right": right, "at": stamp()}
        _write(rank_path(root), state)
        asked += 1
    out(counts_line(state, pool) + ".")
    return sum(decided(state, pool).values())


def lines_text(lines: list[str]) -> str:
    return "\n".join(f"{i}. {plain(x)}" for i, x in enumerate(lines, 1))


# --- re-rank: your agreement with yourself ------------------------------------------------------------------------------

def read_rerank(root: pathlib.Path) -> dict:
    return _read(rerank_path(root)) or {"seed": RERANK_SEED, "answers": {}}


def rerank_picks(pool: dict, state: dict) -> list[str]:
    """The 20 decided pairs to ask again, by fixed seed: 5 per (role, half) cell, in a seeded mixed order."""
    cells = {c: [] for c in CELLS}
    for p in pool["prompts"]:
        a = state["answers"].get(p["id"])
        if p["split"] == "ranking" and a is not None and a["choice"] in ("1", "2"):
            cells[(p["role"], p["half"])].append(p["id"])
    picks = []
    for cell in CELLS:
        picks += random.Random(f"{RERANK_SEED}:pick:{cell[0]}:{cell[1]}").sample(sorted(cells[cell]), RERANK_PER_CELL)
    random.Random(f"{RERANK_SEED}:order").shuffle(picks)
    return picks


def rerank_sides(pid: str, a: dict) -> tuple[int, int]:
    """The same two threads with sides re-drawn by a new seed, so position cannot give the old answer away."""
    swap = random.Random(f"{RERANK_SEED}:side:{pid}").random() < 0.5
    return (a["right"], a["left"]) if swap else (a["left"], a["right"])


def rerank(pool: dict, root: pathlib.Path | None = None, limit: int | None = None, read=input, out=print) -> int:
    """Asks the 20 picked pairs again (same threads, new sides, no skip, earlier answer never shown), saving after every
    answer into rerank.json; rank.json is never written. Returns the number answered so far."""
    root = root or DATA
    state = read_state(root)
    if sum(decided(state, pool).values()) < N_PAIRS:
        raise SystemExit(f"rank first: {counts_line(state, pool)}")
    picks, mine = rerank_picks(pool, state), read_rerank(root)
    mine["seed"] = RERANK_SEED
    asked = 0
    for pid in picks:
        if pid in mine["answers"]:
            continue
        if limit is not None and asked >= limit:
            break
        left, right = rerank_sides(pid, state["answers"][pid])
        rec = _read(bit_path(root, "ranking", pid))
        out(f"\nRe-rank {len(mine['answers']) + 1} of {len(picks)}")
        for side, i in (("1", left), ("2", right)):
            out(f"--- {side}\n" + lines_text(rec["threads"][i]["lines"]))
        while True:
            a = read("Which is funnier? [1/2/t=tie/q=stop] ").strip().lower()
            if a in ("1", "2", "t", "q"):
                break
        if a == "q":
            break
        mine["answers"][pid] = {"choice": a, "left": left, "right": right, "at": stamp()}
        _write(rerank_path(root), mine)
        asked += 1
    out(f"{len(mine['answers'])} of {len(picks)} re-ranked.")
    return len(mine["answers"])


def self_agreement(pool: dict, state: dict, mine: dict, pairs: list[dict], letters: dict) -> dict:
    """Reported only, no bar (set before any re-rank answer exists): of the re-ranked pairs, how many got the same pick.
    A tie on the re-rank counts as a disagreement and is reported separately. It is a ceiling for any judge on these
    pairs. The judge's agreement is on the test pairs among them only (teach pairs were shown to the judge)."""
    ids = set(rerank_picks(pool, state))
    got = {pid: a for pid, a in mine["answers"].items() if pid in ids}
    if not got:
        return {"status": "not run"}
    same = ties = 0
    for pid, a in got.items():
        was = state["answers"][pid]
        before = was["left"] if was["choice"] == "1" else was["right"]
        ties += a["choice"] == "t"
        same += a["choice"] in ("1", "2") and (a["left"] if a["choice"] == "1" else a["right"]) == before
    w = jc.wilson(same, len(got))
    test = [p for p in pairs if p["role"] == "test" and p["id"] in got]
    rows = [[picked_thread(letters.get((p["id"], o)), o) for o in ORDERS] for p in test]
    judge = None
    if test and all(None not in r for r in rows):
        judge = {"pairs": len(test), "agree": sum(r[0] == r[1] == p["winner"] for r, p in zip(rows, test)),
                 "flip": sum(r[0] != r[1] for r in rows)}
    return {"status": "done" if len(got) == len(ids) else "partial", "answered": len(got), "of": len(ids), "same": same,
            "ties": ties, "wilson95": w and [round(x, 4) for x in w], "judge_same_pairs": judge}


def rerank_digest(mine: dict) -> str:
    return hashlib.sha256(json.dumps(mine["answers"], sort_keys=True).encode()).hexdigest()[:8] if mine["answers"] else ""


# --- the gate ------------------------------------------------------------------------------------------------------------

def decided_pairs(pool: dict, state: dict, root: pathlib.Path) -> list[dict]:
    """Every decided ranking pair with its role, half, the two threads' lines as you saw them and your pick (0 left, 1 right)."""
    out = []
    for p in pool["prompts"]:
        a = state["answers"].get(p["id"])
        if p["split"] != "ranking" or a is None or a["choice"] not in ("1", "2"):
            continue
        rec = _read(bit_path(root, "ranking", p["id"]))
        t = rec["threads"]
        out.append({"id": p["id"], "role": p["role"], "half": p["half"], "left": t[a["left"]]["lines"],
                    "right": t[a["right"]]["lines"], "winner": 0 if a["choice"] == "1" else 1})
    return out


def teach_prompt(pairs: list[dict]) -> str:
    """The judge's system message: only teach pairs, in a seeded order, each as the user saw it (left is A)."""
    teach = [p for p in pairs if p["role"] == "teach"]
    random.Random(f"{RANK_SEED}:teach-order").shuffle(teach)
    blocks = [f"Pair {i}\nBit A:\n{lines_text(p['left'])}\nBit B:\n{lines_text(p['right'])}\nTheir pick: {'A' if p['winner'] == 0 else 'B'}"
              for i, p in enumerate(teach, 1)]
    system = JUDGE_SYSTEM.format(n=len(teach), examples="\n\n".join(blocks))
    # The gate's whole point: the judge never sees a test pair.
    for p in pairs:
        if p["role"] == "test":
            assert lines_text(p["left"]) not in system and lines_text(p["right"]) not in system, "a test pair is in the teach prompt"
    return system


def judge_user(pair: dict, order: str) -> str:
    a, b = (pair["left"], pair["right"]) if order == "ab" else (pair["right"], pair["left"])
    return f"Bit A:\n{lines_text(a)}\nBit B:\n{lines_text(b)}"


def parse_pick(reply: str) -> str:
    v = json.loads(llm.strip_fence(reply))["funnier"]
    if v not in ("A", "B"):
        raise ValueError("funnier is not A or B")
    return v


def judge_model(which: str = "haiku") -> tuple[str, str, bool]:
    """(provider, model, thinking). Opus 5.5 cannot run with thinking off, so its thinking is on at effort low."""
    return tc.judge_model() if which == "haiku" else ("anthropic", OPUS_MODEL, True)


def make_judge_client(which: str = "haiku"):
    if which == "haiku":
        return tc.make_judge_client()
    # Same prompt, schema, effort and retries as the Haiku run; only the model, the timeouts and prompt caching differ.
    provider, model, thinking = judge_model(which)
    return llm.from_env(OPUS_TIMING[0], role="safety_check", provider=provider, model=model, retries=generate.FILTER_RETRIES,
                        deadline_s=OPUS_TIMING[1], thinking=thinking, cache_system=True)


def judge_digest(system: str, which: str = "haiku") -> str:
    provider, model, thinking = judge_model(which)
    body = "\x00".join([system, JUDGE_SCHEMA and json.dumps(JUDGE_SCHEMA, sort_keys=True), provider, model, str(thinking)])
    return hashlib.sha256(body.encode()).hexdigest()[:8]


def judge_dir(root: pathlib.Path, system: str, which: str = "haiku") -> pathlib.Path:
    return root / "judge" / judge_digest(system, which)


def judge_all(client, pairs: list[dict], root: pathlib.Path | None = None, out=print, which: str = "haiku") -> int:
    """Asks the judge about every test pair in both orders, cached per call and resumable. Returns calls made."""
    root = root or DATA
    system = teach_prompt(pairs)
    directory, made = judge_dir(root, system, which), 0
    for pair in (p for p in pairs if p["role"] == "test"):
        for order in ORDERS:
            path = directory / f"{pair['id']}-{order}.json"
            if _read(path) is not None:
                continue
            try:
                letter = parse_pick(client.chat(system, judge_user(pair, order), json_reply=JUDGE_SCHEMA, validate=parse_pick))
            except (llm.LLMError, ValueError, KeyError, TypeError, AttributeError):
                continue  # an outage is not the judge's answer: not cached, counted as missing by score
            _write(path, {"funnier": letter})
            made += 1
            out(f"judged {pair['id']} {order} ({made} calls)")
    return made


def picked_thread(letter: str | None, order: str) -> int | None:
    """The side (0 left, 1 right, as you saw it) a judge's letter names in this order."""
    if letter is None:
        return None
    return (0 if letter == "A" else 1) if order == "ab" else (1 if letter == "A" else 0)


def read_letters(pairs: list[dict], root: pathlib.Path | None = None, which: str = "haiku") -> dict[tuple[str, str], str | None]:
    directory = judge_dir(root or DATA, teach_prompt(pairs), which)
    return {(p["id"], o): (_read(directory / f"{p['id']}-{o}.json") or {}).get("funnier")
            for p in pairs if p["role"] == "test" for o in ORDERS}


def summarize(pairs: list[dict], letters: dict[tuple[str, str], str | None], ties: int = 0, skips: int = 0,
              which: str = "haiku") -> dict:
    test = [p for p in pairs if p["role"] == "test"]
    rows = []
    for p in test:
        picks = [picked_thread(letters.get((p["id"], o)), o) for o in ORDERS]
        complete = None not in picks
        rows.append({"half": p["half"], "complete": complete, "agree": complete and picks[0] == picks[1] == p["winner"],
                     "flip": complete and picks[0] != picks[1],
                     "order_agree": [pk == p["winner"] for pk in picks if pk is not None]})
    n, agree = len(rows), sum(r["agree"] for r in rows)
    missing = sum(not r["complete"] for r in rows)
    z, lv = Z[which], LEVEL[which]
    w = jc.wilson(agree, n, z)

    def by(half: str) -> dict:
        mine = [r for r in rows if r["half"] == half]
        return {"pairs": len(mine), "agree": sum(r["agree"] for r in mine),
                "consistent": sum(r["complete"] and not r["flip"] for r in mine),
                "flip_rate": rate(sum(r["flip"] for r in mine), sum(r["complete"] for r in mine))}

    singles = [x for r in rows for x in r["order_agree"]]
    cons = [r for r in rows if r["complete"] and not r["flip"]]
    c, a_ = len(cons), sum(r["agree"] for r in cons)
    wc = jc.wilson(a_, c, z)
    if n != N_TEST or missing:
        verdict = "incomplete"
    else:
        verdict = "pass" if passes(a_, c, z) else "fail"
    return {"verdict": verdict, "n_test": n, "consistent": c, "agree_consistent": a_, "min_consistent": MIN_CONSISTENT,
            f"wilson{lv}": wc and [round(x, 4) for x in wc],
            "agree": agree, f"wilson{lv}_strict": w and [round(x, 4) for x in w],
            "missing_pairs": missing, "flip_rate": rate(sum(r["flip"] for r in rows), n - missing),
            # beside the bar, never deciding: D-95's convention (a judge tie earns half), the strict count above, one order
            "agree_flip_half_credit": rate(2 * agree + sum(r["flip"] for r in rows), 2 * n),
            "agree_single_order": rate(sum(singles), len(singles)),
            "by_half": {h: by(h) for h in HALVES}, "ties": ties, "skips": skips,
            "judge": {"provider": judge_model(which)[0], "model": judge_model(which)[1], "teach_pairs": sum(p["role"] == "teach" for p in pairs)}}


def judge_vs_judge(pairs: list[dict], root: pathlib.Path | None = None) -> dict:
    """Exp 37, reported only, no bar: on test pairs both judges answered the same way in both orders, how often they pick
    the same thread, and how often each picks yours. Reads both stored caches; no calls. Teach pairs are in the prompt,
    so neither cache holds an answer for them."""
    sides = {w: read_letters(pairs, root, w) for w in ("haiku", "opus")}
    picks = {w: {} for w in sides}
    for w, letters in sides.items():
        for p in pairs:
            if p["role"] != "test":
                continue
            two = [picked_thread(letters.get((p["id"], o)), o) for o in ORDERS]
            if None not in two and two[0] == two[1]:
                picks[w][p["id"]] = two[0]
    both = [p for p in pairs if p["id"] in picks["haiku"] and p["id"] in picks["opus"]]
    same = sum(picks["haiku"][p["id"]] == picks["opus"][p["id"]] for p in both)
    z = Z["opus"]
    out = {"pairs": len(both), "same": same, "same_range975": jc.wilson(same, len(both), z) if both else None}
    for w in sides:
        k = sum(picks[w][p["id"]] == p["winner"] for p in both)
        out[f"{w}_agrees_with_you"] = k
        out[f"{w}_range975"] = jc.wilson(k, len(both), z) if both else None
    out["teach_pairs_answered"] = sum(p["role"] == "teach" and any(letters.get((p["id"], o)) for letters in sides.values() for o in ORDERS)
                                      for p in pairs)
    return out


def passes(a: int, c: int, z: float = Z["haiku"]) -> bool:
    """D-121's gate: enough pairs answered consistently, and among them the Wilson lower bound of agreement above 50%
    (95% for the first look, 97.5% for D-122's second look: pass `Z["opus"]`)."""
    return c >= MIN_CONSISTENT and jc.wilson(a, c, z)[0] > 0.5


def rate(k: int, n: int) -> float | None:
    return round(k / n, 4) if n else None


def need_pairs(root: pathlib.Path | None = None) -> tuple[dict, dict, list[dict]]:
    root = root or DATA
    pool = _read(pool_path(root))
    if pool is None:
        raise SystemExit("run `python -m sim.dpo_pairs pool` first")
    state = read_state(root)
    pairs = decided_pairs(pool, state, root)
    if len(pairs) < N_PAIRS:
        raise SystemExit(f"rank first: {counts_line(state, pool)}")
    return pool, state, pairs


def answers_digest(state: dict) -> str:
    return hashlib.sha256(json.dumps(state["answers"], sort_keys=True).encode()).hexdigest()[:8]


def params(pool: dict, state: dict, pairs: list[dict], root: pathlib.Path | None = None, which: str = "haiku") -> dict:
    """The record's key: the pool, every answer (the test answers are not in the teach prompt), the teach prompt and the bar,
    plus the re-rank answers when present so the record reflects them (the gate verdict never depends on them)."""
    key = {"pool_digest": pool["digest"], "rank_seed": RANK_SEED, "answers_digest": answers_digest(state),
           "judge_digest_": judge_digest(teach_prompt(pairs), which), "min_consistent": MIN_CONSISTENT,
           "rerank_digest": rerank_digest(read_rerank(root or DATA))}
    return key if which == "haiku" else {**key, "model": judge_model(which)[1], "z": Z[which]}


def score(root: pathlib.Path | None = None, which: str = "haiku") -> dict:
    pool, state, pairs = need_pairs(root)
    ans = state["answers"].values()
    letters = read_letters(pairs, root, which)
    res = summarize(pairs, letters, sum(a["choice"] == "t" for a in ans), sum(a["choice"] == "s" for a in ans), which)
    res["self_agreement"] = self_agreement(pool, state, read_rerank(root or DATA), pairs, letters)
    return res


def run(pool_digest: str = "", rank_seed: str = "", answers_digest: str = "", judge_digest_: str = "",
        min_consistent: int = 0, rerank_digest: str = "") -> dict:
    """Exp 36's record: the numbers from your answers and the judge's cached replies (the params pin which pool, which
    teach prompt and which bar they are for)."""
    return score()


def run_opus(pool_digest: str = "", rank_seed: str = "", answers_digest: str = "", judge_digest_: str = "",
             min_consistent: int = 0, rerank_digest: str = "", model: str = "", z: float = 0.0) -> dict:
    """Exp 37's record (D-122): the same numbers for the Opus 5.5 judge on the same pairs, under the corrected bar."""
    return score(which="opus")


def stored(which: str = "haiku") -> dict | None:
    """The committed record (numbers only), for the notebook; None until `score --record` has run."""
    path = cache.METRICS / f"{RECORD[which]}.json"
    return json.loads(path.read_text())["results"] if path.exists() else None


def table(res: dict, which: str = "haiku") -> str:
    w = res[f"wilson{LEVEL[which]}"]
    chance = "95%" if which == "haiku" else "97.5%"
    rows = ["| Check | Result | Bar |", "|---|---|---|",
            f"| Test pairs the judge answered the same way in both orders (c) | {res['consistent']} of {res['n_test']} | at least {res['min_consistent']} |",
            f"| Of those, the judge picks your thread (a) | {res['agree_consistent']} of {res['consistent']} | lower end of the range above 50% |",
            f"| Likely range of a / c (Wilson {chance}) | {'n/a' if w is None else f'{w[0]:.0%} to {w[1]:.0%}'} | lower end above 50% |",
            f"| Pairs where the two orders disagree | {res['flip_rate']} | beside the bar |"]
    for h, v in res["by_half"].items():
        rows.append(f"| {h} half: consistent, agreeing | {v['consistent']} of {v['pairs']}, {v['agree']} | beside the bar |")
    rows.append(f"| Strict count (agrees in both orders, of all pairs) | {res['agree']} of {res['n_test']} | beside the bar |")
    rows.append(f"| Same, a disagreeing pair earning half | {res['agree_flip_half_credit']} | beside the bar |")
    rows.append(f"| Ties / skips while ranking | {res['ties']} / {res['skips']} | beside the bar |")
    sa = res.get("self_agreement", {"status": "not run"})
    if sa["status"] == "not run":
        rows.append("| Your agreement with yourself (re-rank of 20) | not run | reported only |")
    else:
        sw = sa["wilson95"]
        rng = "n/a" if sw is None else f"{sw[0]:.0%} to {sw[1]:.0%}"
        rows.append(f"| Your agreement with yourself ({sa['answered']} of {sa['of']} re-ranked) | {sa['same']} of {sa['answered']} same pick, "
                    f"likely range {rng}; {sa['ties']} ties counted as disagreements | reported only; a ceiling for any judge on these pairs |")
        j = sa["judge_same_pairs"]
        rows.append("| Judge on the test pairs among them | " + ("n/a" if j is None else
                    f"{j['agree']} of {j['pairs']} agree in both orders, {j['flip']} with the orders disagreeing") + " | reported only |")
    rows.append(f"| Verdict | {res['verdict']} | the bar |")
    return "\n".join(rows)


# --- command line --------------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("pool", "write", "topup", "rank", "rerank", "judge", "score", "label", "export", "modelfile", "eval"))
    ap.add_argument("sub", nargs="?", help="eval: write, plan, power, bar, vote, judge or score")
    ap.add_argument("--writer", help="write, topup, label, export: the model that writes (default: the live writer); eval write: A, B, C or a model name")
    ap.add_argument("--workers", type=int, default=4, help="label: parallel Haiku calls after the first (which writes the prompt cache)")
    ap.add_argument("--base", help="modelfile: the text of `ollama show qwen3:4b-instruct --modelfile`")
    ap.add_argument("--gguf", help="modelfile: the trained model's GGUF file")
    ap.add_argument("--out", help="modelfile: where to write the Modelfile")
    ap.add_argument("--wins", type=int, help="eval bar: wins of 15 a trained writer needs")
    ap.add_argument("--split", choices=("ranking", "train"), help="write: which split; pool --rebuild --split train rebuilds only the training split")
    ap.add_argument("--n", type=int, help="write: stop after this many new bits")
    ap.add_argument("--limit", type=int, help="rank, rerank, eval vote: ask at most this many questions this sitting; label: at most this many calls")
    ap.add_argument("--rebuild", action="store_true", help="pool: replace the frozen pool (invalidates every written bit)")
    ap.add_argument("--tonight-dir", help="pool: where the stored `sim.tonight fetch` folders are (default data/tonight)")
    ap.add_argument("--record", action="store_true", help="score: save the result as the Exp 36 record (Exp 37 with --model opus)")
    ap.add_argument("--model", choices=("haiku", "opus"), default="haiku",
                    help="judge, score: the judge model; haiku (Exp 36, D-121) is the default, opus is Exp 37 (D-122)")
    ap.add_argument("--env-file")
    a = ap.parse_args()
    if a.env_file:
        from dotenv import load_dotenv
        if not pathlib.Path(a.env_file).is_file():
            raise SystemExit(f"--env-file {a.env_file} not found; from a worktree use the main checkout's backend/.env")
        load_dotenv(a.env_file, override=True)
    os.environ.setdefault("JOKE_WRITER_PROVIDER", "ollama")
    tonight_dir = pathlib.Path(a.tonight_dir) if a.tonight_dir else None
    if a.command == "pool":
        if a.rebuild and a.split == "train":
            pool = rebuild_train(tonight_dir=tonight_dir)
            print(f"pool {pool['digest']}: {json.dumps(sizes(pool))}; tonight days: {[d['date'] for d in pool['days']]}")
            return
        if a.rebuild and DATA.joinpath("bits").exists():
            raise SystemExit("bits are already written for this pool: move data/dpo aside before rebuilding")
        pool = ensure_pool(tonight_dir=tonight_dir, rebuild=a.rebuild)
        print(f"pool {pool['digest']}: {json.dumps(sizes(pool))}; tonight days: {[d['date'] for d in pool['days']]}")
        return
    if a.command in ("write", "topup"):
        if not a.split:
            raise SystemExit(f"{a.command} needs --split ranking or --split train")
        pool = ensure_pool(tonight_dir=tonight_dir)
        client = generate.writer_client(a.writer)
        if not getattr(client, "local", False):
            raise SystemExit("the pair writer is the local writer: set JOKE_WRITER_PROVIDER=ollama")
        print(f"Writer {client.model}, bits in data/dpo/bits/{bits_folder(a.split, client.model)}/. "
              "Do not run this during a live show: it shares the GPU. Ctrl-C is safe; rerun to resume.", flush=True)
        generate.warm_up(generate.ThreadGenerator(client))
        live_safety = generate.LiveSafety(llm.from_env(generate.FILTER_TIMING[0], role="safety_check", retries=generate.FILTER_RETRIES,
                                                       deadline_s=generate.FILTER_TIMING[1]))
        try:
            if a.command == "topup":
                print(f"added {topup(a.split, client, live_safety, pool)} batches")
                return
            wrote = write(a.split, client, live_safety, pool, n=a.n)
        except KeyboardInterrupt:
            raise SystemExit("stopped; finished bits are saved, rerun to resume")
        print(f"wrote {wrote} new bits")
        return
    if a.command == "modelfile":
        from sim import dpo_train
        if not (a.base and a.gguf and a.out):
            raise SystemExit("modelfile needs --base, --gguf and --out")
        pathlib.Path(a.out).write_text(dpo_train.modelfile_for(pathlib.Path(a.base).read_text(encoding="utf-8"), a.gguf), encoding="utf-8")
        print(f"wrote {a.out}: `ollama create <name> -f {a.out}`")
        return
    if a.command == "export":
        from sim import dpo_train
        dpo_train.export(a.writer or live_writer())
        return
    if a.command == "eval":
        from sim import dpo_eval
        dpo_eval.main(a)
        return
    if a.command == "label":
        from sim import dpo_train
        if a.split != "train":
            raise SystemExit("label --split train")
        pool, state, pairs = need_pairs()
        writer = a.writer or live_writer()
        dpo_train.label_all(dpo_train.make_label_client(), a.split, writer, pairs, pool, workers=a.workers, limit=a.limit)
        return
    if a.command == "rank":
        pool = _read(pool_path())
        if pool is None:
            raise SystemExit("run `python -m sim.dpo_pairs pool` first")
        try:
            rank(pool, limit=a.limit)
        except (KeyboardInterrupt, EOFError):
            print("\nstopped; every answer so far is saved.")
        return
    if a.command == "rerank":
        pool = _read(pool_path())
        if pool is None:
            raise SystemExit("run `python -m sim.dpo_pairs pool` first")
        try:
            rerank(pool, limit=a.limit)
        except (KeyboardInterrupt, EOFError):
            print("\nstopped; every answer so far is saved.")
        return
    pool, state, pairs = need_pairs()
    if a.command == "judge":
        client = make_judge_client(a.model)
        n = judge_all(client, pairs, which=a.model)
        print(f"{n} calls made; tokens in {client.input_tokens}, out {client.output_tokens}, "
              f"cache read {getattr(client, 'cache_read_tokens', 0)}, cache write {getattr(client, 'cache_write_tokens', 0)}")
        print(f"now `score{'' if a.model == 'haiku' else ' --model ' + a.model} --record`")
        return
    res = score(which=a.model)
    if a.record:
        fn = run if a.model == "haiku" else run_opus
        res = cache.run_or_load(RECORD[a.model], fn, rerun=True, **params(pool, state, pairs, which=a.model))
    print(table(res, a.model), flush=True)


if __name__ == "__main__":
    main()
