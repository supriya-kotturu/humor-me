"""Exp 36 (D-121): does the pairwise judge agree with the user on two threads written by the same model?

DPO needs pairs of threads labelled "which is funnier". The user's taste cannot label thousands, so a judge will, but only
if it first agrees with the user on same-model pairs, which are much closer than the cross-model ballot of Exp 24 (D-102).
This module writes the candidate pairs, gives the user the ranking tool, and runs that gate.

Usage (Ollama up with `OLLAMA_NUM_PARALLEL=3` and `OLLAMA_CONTEXT_LENGTH=8192`; do NOT run `write` during a live show, it
shares the one GPU and the show's 12 s bit deadline would suffer):
    python -m sim.dpo_pairs pool                              # once: freeze the prompts (reads data/tonight/**/fetch.json)
    python -m sim.dpo_pairs write --split ranking --env-file backend/.env [--n N]   # in the background; resumable
    python -m sim.dpo_pairs write --split train   --env-file backend/.env [--n N]   # can run while you rank
    python -m sim.dpo_pairs rank [--limit 20]                 # you pick the funnier thread of a pair, in sittings
    python -m sim.dpo_pairs judge --env-file backend/.env     # the gate: 120 calls (60 test pairs, both orders)
    python -m sim.dpo_pairs score --record                    # numbers -> reports/metrics/dpo_judge_gate.json
    python -m sim.dpo_pairs pool --split train --rebuild      # later, with 3+ days stored, before any judge labelling

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


def _write(path: pathlib.Path, rec: dict) -> None:
    tc._write(path, rec)


def _read(path: pathlib.Path) -> dict | None:
    return tc._read(path)


def pool_path(root: pathlib.Path | None = None) -> pathlib.Path:
    return (root or DATA) / "pool.json"


def bit_path(root: pathlib.Path, split: str, pid: str) -> pathlib.Path:
    return root / "bits" / split / f"{pid}.json"


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


def request_digest(request: str, generator_id: str) -> str:
    parts = [request, generator_id, str((generate.K, generate.JOKES, generate.TEMPERATURE, generate.GENERATOR_TIMING,
                                         generate.RETRIES, generate.WRITER_CONTEXT)), tc.writer_model()]
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()[:12]


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
    gen = new_generator(client, prompt, days)
    set_allow(live_safety, prompt, days)
    bad0 = client.bad_replies
    start = time.perf_counter()
    rec = {"id": prompt["id"], "split": prompt["split"], "half": prompt["half"], "mode": prompt["mode"], "bit": prompt["bit"],
           "context": prompt["context"], "told": prompt["told"], "day": prompt["day"], "gist": prompt["gist"],
           "generator_id": gen.generator_id, "request": None, "threads": [], "lost": [], "failure": None}
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
    rec["wall_s"] = time.perf_counter() - start
    rec["bad_replies"] = client.bad_replies - bad0
    rec["digest"] = request_digest(rec["request"] or "", gen.generator_id)
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


def same_prompt(rec: dict | None, p: dict) -> bool:
    """The record was written for this prompt, not for another pool's prompt with the same id (a rebuild while a writer ran)."""
    return rec is not None and all(rec.get(k) == p[k] for k in PROMPT_FIELDS)


def write(split: str, client, live_safety, pool: dict, root: pathlib.Path | None = None, n: int | None = None,
          out=print) -> int:
    """Writes the missing bits of a split, one at a time, in pool order; a bit whose safety layer failed is re-checked, not
    regenerated, and a bit with no thread is written again. Stops after `n` new bits (None: all), or when
    MAX_EMPTY_IN_A_ROW bits in a row return no thread (the writer is down). Returns how many it wrote."""
    root = root or DATA
    days = {d["date"]: d for d in pool["days"]}
    mine = [p for p in pool["prompts"] if p["split"] == split]
    total = len(mine)
    ready = sum(done(r) and same_prompt(r, p) for p in mine if (r := _read(bit_path(root, split, p["id"]))) is not None)
    wrote = empty = 0
    for i, p in enumerate(mine, 1):
        path = bit_path(root, split, p["id"])
        rec = _read(path)
        if not same_prompt(rec, p):
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
        state["answers"][p["id"]] = {"choice": a, "left": left, "right": right}
        _write(rank_path(root), state)
        asked += 1
    out(counts_line(state, pool) + ".")
    return sum(decided(state, pool).values())


def lines_text(lines: list[str]) -> str:
    return "\n".join(f"{i}. {plain(x)}" for i, x in enumerate(lines, 1))


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


def judge_digest(system: str) -> str:
    provider, model, thinking = tc.judge_model()
    body = "\x00".join([system, JUDGE_SCHEMA and json.dumps(JUDGE_SCHEMA, sort_keys=True), provider, model, str(thinking)])
    return hashlib.sha256(body.encode()).hexdigest()[:8]


def judge_dir(root: pathlib.Path, system: str) -> pathlib.Path:
    return root / "judge" / judge_digest(system)


def judge_all(client, pairs: list[dict], root: pathlib.Path | None = None, out=print) -> int:
    """Asks the judge about every test pair in both orders, cached per call and resumable. Returns calls made."""
    root = root or DATA
    system = teach_prompt(pairs)
    directory, made = judge_dir(root, system), 0
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


def read_letters(pairs: list[dict], root: pathlib.Path | None = None) -> dict[tuple[str, str], str | None]:
    directory = judge_dir(root or DATA, teach_prompt(pairs))
    return {(p["id"], o): (_read(directory / f"{p['id']}-{o}.json") or {}).get("funnier")
            for p in pairs if p["role"] == "test" for o in ORDERS}


def summarize(pairs: list[dict], letters: dict[tuple[str, str], str | None], ties: int = 0, skips: int = 0) -> dict:
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
    w = jc.wilson(agree, n)

    def by(half: str) -> dict:
        mine = [r for r in rows if r["half"] == half]
        return {"pairs": len(mine), "agree": sum(r["agree"] for r in mine),
                "consistent": sum(r["complete"] and not r["flip"] for r in mine),
                "flip_rate": rate(sum(r["flip"] for r in mine), sum(r["complete"] for r in mine))}

    singles = [x for r in rows for x in r["order_agree"]]
    cons = [r for r in rows if r["complete"] and not r["flip"]]
    c, a_ = len(cons), sum(r["agree"] for r in cons)
    wc = jc.wilson(a_, c)
    if n != N_TEST or missing:
        verdict = "incomplete"
    else:
        verdict = "pass" if passes(a_, c) else "fail"
    return {"verdict": verdict, "n_test": n, "consistent": c, "agree_consistent": a_, "min_consistent": MIN_CONSISTENT,
            "wilson95": wc and [round(x, 4) for x in wc],
            "agree": agree, "wilson95_strict": w and [round(x, 4) for x in w],
            "missing_pairs": missing, "flip_rate": rate(sum(r["flip"] for r in rows), n - missing),
            # beside the bar, never deciding: D-95's convention (a judge tie earns half), the strict count above, one order
            "agree_flip_half_credit": rate(2 * agree + sum(r["flip"] for r in rows), 2 * n),
            "agree_single_order": rate(sum(singles), len(singles)),
            "by_half": {h: by(h) for h in HALVES}, "ties": ties, "skips": skips,
            "judge": {"provider": tc.judge_model()[0], "model": tc.judge_model()[1], "teach_pairs": sum(p["role"] == "teach" for p in pairs)}}


def passes(a: int, c: int) -> bool:
    """D-121's gate: enough pairs answered consistently, and among them the Wilson 95% lower bound of agreement above 50%."""
    return c >= MIN_CONSISTENT and jc.wilson(a, c)[0] > 0.5


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


def params(pool: dict, state: dict, pairs: list[dict]) -> dict:
    """The record's key: the pool, every answer (the test answers are not in the teach prompt), the teach prompt and the bar."""
    return {"pool_digest": pool["digest"], "rank_seed": RANK_SEED, "answers_digest": answers_digest(state),
            "judge_digest_": judge_digest(teach_prompt(pairs)), "min_consistent": MIN_CONSISTENT}


def score(root: pathlib.Path | None = None) -> dict:
    pool, state, pairs = need_pairs(root)
    ans = state["answers"].values()
    return summarize(pairs, read_letters(pairs, root), sum(a["choice"] == "t" for a in ans), sum(a["choice"] == "s" for a in ans))


def run(pool_digest: str = "", rank_seed: str = "", answers_digest: str = "", judge_digest_: str = "",
        min_consistent: int = 0) -> dict:
    """Exp 36's record: the numbers from your answers and the judge's cached replies (the params pin which pool, which
    teach prompt and which bar they are for)."""
    return score()


def stored() -> dict | None:
    """The committed record (numbers only), for the notebook; None until `score --record` has run."""
    path = cache.METRICS / f"{NAME}.json"
    return json.loads(path.read_text())["results"] if path.exists() else None


def table(res: dict) -> str:
    w = res["wilson95"]
    rows = ["| Check | Result | Bar |", "|---|---|---|",
            f"| Test pairs the judge answered the same way in both orders (c) | {res['consistent']} of {res['n_test']} | at least {res['min_consistent']} |",
            f"| Of those, the judge picks your thread (a) | {res['agree_consistent']} of {res['consistent']} | lower end of the range above 50% |",
            f"| Likely range of a / c (Wilson 95%) | {'n/a' if w is None else f'{w[0]:.0%} to {w[1]:.0%}'} | lower end above 50% |",
            f"| Pairs where the two orders disagree | {res['flip_rate']} | beside the bar |"]
    for h, v in res["by_half"].items():
        rows.append(f"| {h} half: consistent, agreeing | {v['consistent']} of {v['pairs']}, {v['agree']} | beside the bar |")
    rows.append(f"| Strict count (agrees in both orders, of all pairs) | {res['agree']} of {res['n_test']} | beside the bar |")
    rows.append(f"| Same, a disagreeing pair earning half | {res['agree_flip_half_credit']} | beside the bar |")
    rows.append(f"| Ties / skips while ranking | {res['ties']} / {res['skips']} | beside the bar |")
    rows.append(f"| Verdict | {res['verdict']} | the bar |")
    return "\n".join(rows)


# --- command line --------------------------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("pool", "write", "rank", "judge", "score"))
    ap.add_argument("--split", choices=("ranking", "train"), help="write: which split; pool --rebuild --split train rebuilds only the training split")
    ap.add_argument("--n", type=int, help="write: stop after this many new bits")
    ap.add_argument("--limit", type=int, help="rank: ask at most this many questions this sitting")
    ap.add_argument("--rebuild", action="store_true", help="pool: replace the frozen pool (invalidates every written bit)")
    ap.add_argument("--tonight-dir", help="pool: where the stored `sim.tonight fetch` folders are (default data/tonight)")
    ap.add_argument("--record", action="store_true", help="score: save the result as the Exp 36 record")
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
    if a.command == "write":
        if not a.split:
            raise SystemExit("write needs --split ranking or --split train")
        pool = ensure_pool(tonight_dir=tonight_dir)
        client = generate.writer_client()
        if not getattr(client, "local", False):
            raise SystemExit("the pair writer is the local writer: set JOKE_WRITER_PROVIDER=ollama")
        print("Do not run this during a live show: it shares the GPU. Ctrl-C is safe; rerun to resume.", flush=True)
        generate.warm_up(generate.ThreadGenerator(client))
        live_safety = generate.LiveSafety(llm.from_env(generate.FILTER_TIMING[0], role="safety_check", retries=generate.FILTER_RETRIES,
                                                       deadline_s=generate.FILTER_TIMING[1]))
        try:
            wrote = write(a.split, client, live_safety, pool, n=a.n)
        except KeyboardInterrupt:
            raise SystemExit("stopped; finished bits are saved, rerun to resume")
        print(f"wrote {wrote} new bits")
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
    pool, state, pairs = need_pairs()
    if a.command == "judge":
        n = judge_all(tc.make_judge_client(), pairs)
        print(f"{n} calls made; now `score --record`")
        return
    res = score()
    if a.record:
        res = cache.run_or_load(NAME, run, rerun=True, **params(pool, state, pairs))
    print(table(res), flush=True)


if __name__ == "__main__":
    main()
