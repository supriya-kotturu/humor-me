"""Exp 38 (D-123): is a trained writer funnier than the base, by the user's blind votes? And can the laptop serve it live?

Usage (laptop; one writer served at a time, `ollama stop` the others; `OLLAMA_NUM_PARALLEL=3`, `OLLAMA_CONTEXT_LENGTH=8192`):
    python -m sim.dpo_pairs eval write --writer qwen3:4b-instruct --env-file backend/.env   # A, the base
    python -m sim.dpo_pairs eval write --writer humor-qwen3-B --env-file backend/.env       # B: SFT then DPO on Haiku's labels
    python -m sim.dpo_pairs eval write --writer humor-qwen3-C --env-file backend/.env       # C: DPO on the user's own pairs
    python -m sim.dpo_pairs eval plan                          # which of the 30 vote pairs are ready
    python -m sim.dpo_pairs eval power                         # the table behind the bar (no data needed)
    python -m sim.dpo_pairs eval bar --wins 10                 # the user fixes the bar BEFORE voting; `vote` refuses without it
    python -m sim.dpo_pairs eval vote [--limit 10]             # 30 blind pairs: 15 B vs A, 15 C vs A
    python -m sim.dpo_pairs eval judge --model opus --env-file backend/.env   # reported only; also --model haiku (circular for B)
    python -m sim.dpo_pairs eval score --record                # -> reports/metrics/dpo_eval.json

Prompts: `data/dpo/train/eval_prompts.jsonl` (written by `export`, 60 new prompts, half tonight, none in any earlier set). One
bit per prompt per writer, written through the live request path (D-123), the first clean thread of each bit is the one compared.
Votes: 30 pairs, 15 B against A and 15 C against A, on different prompts, sides seeded, no skip, ties allowed (1, 2, t, q),
each answer stamped. B against C is judged by Opus only. The win bar is fixed by the user with `eval bar` before the first vote
(D-123 proposes one and computes its power); a trained writer passes when it wins at least that many pairs of its 15 and nothing
else. Judges (Opus 5.5 with the 60 teach pairs, both orders, every eval prompt) are reported only; Haiku labelled B's training
pairs, so its read of B is circular and says so. Latency: each writer's bit time (median, slowest 5%) against the live
per-bit budget; B and C pass when their slowest 5% is within it and at most 5% of their threads were lost to the clock (a slow
call is dropped at the budget, so bit time alone would hide a writer too slow to finish).
"""
import datetime
import hashlib
import itertools
import json
import math
import pathlib
import random
import statistics

from sim import cache
from sim import compare_generators as cg
from sim import dpo_pairs as dp
from sim import dpo_train as dt

from app import generate  # noqa: E402

NAME = "dpo_eval"
WRITERS = {"A": "qwen3:4b-instruct", "B": "humor-qwen3-B", "C": "humor-qwen3-C"}
# (first, second): "first" is the one a win is counted for; B against C is judged only, never voted.
JUDGED = (("B", "A"), ("C", "A"), ("B", "C"))
VOTED = ("B", "C")  # each against A
N_VOTE = 15  # pairs per voted comparison: the user's choice (30 shared pairs over two comparisons)
VOTE_SEED = "dpo-eval-vote-v1"
PATTERN = ("B", "C", "C", "B")  # comparison of the next ready prompt, so each half is shared between B and C
PROPOSED_WINS = 10  # D-123's proposal, NOT in force until the user runs `eval bar`
LEVELS = (0.65, 0.70, 0.75)  # how often the trained writer would truly win a decided pair
LATENCY_BAR = ("p95 bit time within generate.GENERATOR_TIMING[1] (the live per-bit budget) and at most 5% of threads lost to time "
               "(a call that runs out of budget is dropped, so the bit time alone can never exceed the budget)")
MAX_TIMED_OUT = 0.05  # judgment call, not measured: share of a writer's threads that may be lost to the clock
TIME_MARKS = ("outlived the bit's budget", "deadline passed", "Timeout", "no time left")


def eval_dir(root: pathlib.Path) -> pathlib.Path:
    return root / "eval"


def stamp() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def load_prompts(root: pathlib.Path) -> list[dict]:
    path = root / dt.TRAIN_DIR / "eval_prompts.jsonl"
    if not path.exists():
        raise SystemExit("no eval prompts: run `python -m sim.dpo_pairs export` first")
    return dt.read_jsonl(path)


def eval_pool(root: pathlib.Path) -> dict:
    """The pool `dpo_pairs.write` takes, holding the eval prompts as a split of their own."""
    pool = dp._read(dp.pool_path(root))
    if pool is None:
        raise SystemExit("run `python -m sim.dpo_pairs pool` first")
    return {"days": pool["days"], "prompts": load_prompts(root), "digest": pool["digest"]}


def bit_of(root: pathlib.Path, writer: str, pid: str) -> dict | None:
    rec = dp._read(dp.bit_path(root, "eval", pid, writer))
    return rec if dp.done(rec) else None


def first_clean(rec: dict | None) -> list[str] | None:
    """The lines of the bit's first clean thread (by call index, which carries no order: the calls are independent samples)."""
    if rec is None:
        return None
    return next((t["lines"] for t in rec["threads"] if t.get("safe")), None)


def side_lines(root: pathlib.Path, pid: str) -> dict[str, list[str] | None]:
    return {k: first_clean(bit_of(root, w, pid)) for k, w in WRITERS.items()}


# --- power and the bar ---------------------------------------------------------------------------------------------------

def p_at_least(k: int, n: int, p: float) -> float:
    """P(at least k wins in n decided pairs) when each is won with probability p."""
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def power_table(n: int = N_VOTE, wins: tuple[int, ...] = (9, 10, 11, 12)) -> dict:
    """Chance a bar of `wins` of `n` is met, for a writer that truly wins a decided pair with probability 0.5 (no skill) or one of
    LEVELS. Ties are left out of the count: fewer than n decided pairs lower every number, so the same table at n = 12 and 13
    is in `power_ties`. Pinned by a test."""
    return {k: {p: p_at_least(k, n, p) for p in (0.5, *LEVELS)} for k in wins}


def power_ties(wins: int = PROPOSED_WINS) -> dict:
    """The proposed bar when ties shrink the decided pairs: the wins stay at `wins`, so it gets harder, not softer."""
    return {n: {p: p_at_least(wins, n, p) for p in (0.5, *LEVELS)} for n in (15, 14, 13, 12)}


def either_passes(wins: int, n: int = N_VOTE) -> float:
    """Chance that at least one of the two voted comparisons meets the bar by luck alone (independent prompts, coin flips)."""
    return 1 - (1 - p_at_least(wins, n, 0.5)) ** len(VOTED)


def bar_path(root: pathlib.Path) -> pathlib.Path:
    return eval_dir(root) / "bar.json"


def read_bar(root: pathlib.Path) -> dict | None:
    return dp._read(bar_path(root))


def set_bar(root: pathlib.Path, wins: int) -> dict:
    """The user's confirmed bar, stamped. Refused once a vote exists: a bar set after the data is not a bar."""
    if dp._read(vote_path(root)) and dp._read(vote_path(root))["answers"]:
        raise SystemExit("votes already exist: the bar cannot change now")
    if not 1 <= wins <= N_VOTE:
        raise SystemExit(f"--wins must be between 1 and {N_VOTE}")
    rec = {"wins": wins, "of": N_VOTE, "at": stamp(), "proposed": PROPOSED_WINS}
    dp._write(bar_path(root), rec)
    return rec


def power_text() -> str:
    rows = ["| Bar (wins of 15 decided) | True win rate 50% (coin flip) | 65% | 70% | 75% |", "|---|---|---|---|---|"]
    for k, v in power_table().items():
        rows.append(f"| at least {k} | " + " | ".join(f"{v[p]:.1%}" for p in (0.5, *LEVELS)) + " |")
    rows.append(f"\nEither of two comparisons meets the bar by luck: {either_passes(PROPOSED_WINS):.1%} at {PROPOSED_WINS}, "
                f"{either_passes(PROPOSED_WINS + 1):.1%} at {PROPOSED_WINS + 1}.")
    rows += ["", "| Decided pairs | at least 10 wins: coin flip | 65% | 70% | 75% |", "|---|---|---|---|---|"]
    for n, v in power_ties().items():
        rows.append(f"| {n} | " + " | ".join(f"{v[p]:.1%}" for p in (0.5, *LEVELS)) + " |")
    return "\n".join(rows)


# --- the vote ------------------------------------------------------------------------------------------------------------

def vote_path(root: pathlib.Path) -> pathlib.Path:
    return eval_dir(root) / "vote.json"


def plan(prompts: list[dict], root: pathlib.Path) -> list[dict]:
    """The 30 pairs, in the order they are asked. Prompts come in a seeded order that alternates the tonight and plain halves,
    and are dealt to B against A and C against A in PATTERN, skipping a prompt where A or that writer has no clean thread.
    Each pair has the trained writer on a seeded side. Fewer than 30 rows means more bits must be written first."""
    halves = {h: [p["id"] for p in prompts if p["half"] == h] for h in dp.HALVES}
    for h, ids in halves.items():
        random.Random(f"{VOTE_SEED}:order:{h}").shuffle(ids)
    order = [pid for pair in itertools.zip_longest(*(halves[h] for h in dp.HALVES)) for pid in pair if pid is not None]
    got, items, pos = dict.fromkeys(VOTED, 0), [], 0
    for pid in order:
        if all(n >= N_VOTE for n in got.values()):
            break
        comp = PATTERN[pos % len(PATTERN)]
        if got[comp] >= N_VOTE:
            comp = next(c for c in VOTED if c != comp)
        lines = side_lines(root, pid)
        if lines["A"] is None or lines[comp] is None:
            continue
        got[comp] += 1
        pos += 1
        trained_left = random.Random(f"{VOTE_SEED}:side:{pid}:{comp}").random() < 0.5
        items.append({"pid": pid, "comp": comp, "trained_left": trained_left,
                      "left": lines[comp] if trained_left else lines["A"], "right": lines["A"] if trained_left else lines[comp]})
    random.Random(f"{VOTE_SEED}:present").shuffle(items)
    return items


def vote_key(item: dict) -> str:
    return f"{item['pid']}:{item['comp']}"


def read_votes(root: pathlib.Path) -> dict:
    return dp._read(vote_path(root)) or {"seed": VOTE_SEED, "answers": {}}


def frozen_plan(items: list[dict]) -> list[dict]:
    """What the first vote fixes: each pair's key, side and a hash of the lines shown."""
    return [{"key": vote_key(i), "trained_left": i["trained_left"],
             "sha": hashlib.sha256(json.dumps([i["left"], i["right"]], ensure_ascii=False).encode()).hexdigest()[:8]} for i in items]


def plan_changed(items: list[dict], state: dict) -> bool:
    return "plan" in state and state["plan"] != frozen_plan(items)


def vote(items: list[dict], root: pathlib.Path, limit: int | None = None, read=input, out=print) -> int:
    """Asks the unanswered pairs in order: 1, 2, t (tie) or q. No skip. Saves after every answer (stamped), never shows which
    writer is which or the running tally. Returns the number answered so far."""
    if read_bar(root) is None:
        raise SystemExit("fix the bar first: `python -m sim.dpo_pairs eval power`, then `eval bar --wins N`")
    if len(items) < 2 * N_VOTE:
        raise SystemExit(f"only {len(items)} of {2 * N_VOTE} pairs are ready: write the eval bits of every writer first (`eval plan`)")
    state = read_votes(root)
    # The plan is rebuilt from the bits on every run; a bit written or rewritten after voting began would re-deal prompts
    # between B and C or change the lines behind an answer. So the first vote fixes it, and any later change stops the vote.
    if plan_changed(items, state):
        raise SystemExit("the eval bits changed after voting began (the plan of pairs is not the one voted on): restore them")
    state.setdefault("plan", frozen_plan(items))
    asked = 0
    for item in items:
        key = vote_key(item)
        if key in state["answers"]:
            continue
        if limit is not None and asked >= limit:
            break
        out(f"\nPair {len(state['answers']) + 1} of {len(items)}")
        for side in ("left", "right"):
            out(f"--- {'1' if side == 'left' else '2'}\n" + dp.lines_text(item[side]))
        while True:
            a = read("Which is funnier? [1/2/t=tie/q=stop] ").strip().lower()
            if a in ("1", "2", "t", "q"):
                break
        if a == "q":
            break
        state["answers"][key] = {"choice": a, "trained_left": item["trained_left"], "at": stamp()}
        dp._write(vote_path(root), state)
        asked += 1
    out(f"{len(state['answers'])} of {len(items)} answered.")
    return len(state["answers"])


def tally(items: list[dict], state: dict, bar: dict | None) -> dict:
    out = {}
    for comp in VOTED:
        mine = [state["answers"][vote_key(i)] for i in items if i["comp"] == comp and vote_key(i) in state["answers"]]
        wins = sum(a["choice"] in ("1", "2") and (a["choice"] == "1") == a["trained_left"] for a in mine)
        ties = sum(a["choice"] == "t" for a in mine)
        decided = len(mine) - ties
        row = {"answered": len(mine), "of": N_VOTE, "wins": wins, "losses": decided - wins, "ties": ties, "decided": decided}
        row["verdict"] = ("pending" if bar is None or len(mine) < N_VOTE else "met" if wins >= bar["wins"] else "not met")
        out[comp] = row
    return out


# --- judges (reported only) ----------------------------------------------------------------------------------------------

def judge_key(comp: tuple[str, str], pid: str, order: str) -> str:
    return f"{comp[0]}{comp[1]}-{pid}-{order}.json"


def judge_dir(root: pathlib.Path, system: str, which: str) -> pathlib.Path:
    return eval_dir(root) / "judge" / f"{which}-{dp.judge_digest(system, which)}"


def make_client(which: str):
    return dt.make_label_client() if which == "haiku" else dp.make_judge_client(which)


def judge_all(client, pairs: list[dict], prompts: list[dict], root: pathlib.Path, which: str, out=print) -> int:
    """Every comparison of JUDGED on every eval prompt where both writers have a clean thread, in both orders (the first writer
    of the comparison is A in order `ab`). Cached per call, resumable. Returns calls made."""
    system = dp.teach_prompt(pairs)
    directory, made = judge_dir(root, system, which), 0
    for p in prompts:
        lines = side_lines(root, p["id"])
        for comp in JUDGED:
            if lines[comp[0]] is None or lines[comp[1]] is None:
                continue
            pair = {"left": lines[comp[0]], "right": lines[comp[1]]}
            for order in dp.ORDERS:
                path = directory / judge_key(comp, p["id"], order)
                if dp._read(path) is not None:
                    continue
                try:
                    letter = dp.parse_pick(client.chat(system, dp.judge_user(pair, order), json_reply=dp.JUDGE_SCHEMA,
                                                       validate=dp.parse_pick))
                except Exception:  # noqa: BLE001 - an outage is not the judge's answer; score counts it missing
                    continue
                dp._write(path, {"funnier": letter})
                made += 1
                out(f"judged {comp[0]}{comp[1]} {p['id']} {order} ({made} calls)")
    return made


def judge_tally(pairs: list[dict], prompts: list[dict], root: pathlib.Path, which: str) -> dict:
    """Per comparison: prompts where the judge named the same writer in both orders, how many named the first writer, the share
    whose orders disagreed, and the prompts still missing an answer."""
    directory = judge_dir(root, dp.teach_prompt(pairs), which)
    out = {}
    for comp in JUDGED:
        first = consistent = flips = missing = 0
        for p in prompts:
            lines = side_lines(root, p["id"])
            if lines[comp[0]] is None or lines[comp[1]] is None:
                continue
            picks = [dp.picked_thread((dp._read(directory / judge_key(comp, p["id"], o)) or {}).get("funnier"), o) for o in dp.ORDERS]
            if None in picks:
                missing += 1
            elif picks[0] != picks[1]:
                flips += 1
            else:
                consistent += 1
                first += picks[0] == 0
        out["".join(comp)] = {"consistent": consistent, "first_wins": first, "flips": flips, "missing": missing}
    return out


# --- latency -------------------------------------------------------------------------------------------------------------

def reference_latency() -> dict:
    """Bonsai's numbers on the same laptop, from the committed records: Exp 26 (D-110, one thread per call) and Exp 24."""
    ref = {}
    try:
        t = json.loads((cache.METRICS / "thread_timing.json").read_text(encoding="utf-8"))["results"]
        ref["bonsai_bit_p50_s"], ref["bonsai_bit_p95_s"] = t["bit_p50_s"], t["bit_p95_s"]
    except (OSError, ValueError, KeyError):
        pass
    try:
        m = json.loads((cache.METRICS / "model_comparison.json").read_text(encoding="utf-8"))["results"]["models"]
        ref["exp24_call_p50_s"] = {k: m[k]["latency_p50_s"] for k in ("bonsai-2-4b", "qwen3-4b-instruct")}
    except (OSError, ValueError, KeyError):
        pass
    return ref


def latency(prompts: list[dict], root: pathlib.Path) -> dict:
    """Each writer's bit wall time (the writer's calls only, as Exp 26 timed it) over its eval bits: median and the slowest 5%."""
    budget = generate.GENERATOR_TIMING[1]
    out = {"budget_s": budget, "bar": LATENCY_BAR, **reference_latency(), "writers": {}}
    for k, w in WRITERS.items():
        recs = [r for p in prompts if (r := bit_of(root, w, p["id"])) is not None]
        walls = [r["wall_s"] for r in recs]
        row = {"bits": len(recs)}
        timed_out = sum(any(m in reason for m in TIME_MARKS) for r in recs for reason in r.get("lost", []))
        if walls:
            row |= {"p50_s": round(statistics.median(walls), 2), "p95_s": round(cg.percentile(walls, 95), 2),
                    "max_s": round(max(walls), 2), "bits_losing_a_thread": sum(len(r["threads"]) < generate.K for r in recs),
                    "threads_lost_to_time": timed_out, "share_lost_to_time": round(timed_out / (len(recs) * generate.K), 4)}
            row["within_budget"] = row["p95_s"] <= budget and row["share_lost_to_time"] <= MAX_TIMED_OUT
        out["writers"][k] = row
    return out


# --- the record ----------------------------------------------------------------------------------------------------------

def file_digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8] if path.exists() else ""


def params(root: pathlib.Path | None = None) -> dict:
    """What the record is for: the prompts, the bar, the votes and the bits (a hash of each writer's bits)."""
    root = root or dp.DATA
    prompts = load_prompts(root)
    bits = {k: hashlib.sha256("".join(f"{r['id']}:{r['digest']}:{len(r['threads'])}" for p in prompts
                                      if (r := bit_of(root, w, p["id"])) is not None).encode()).hexdigest()[:8] for k, w in WRITERS.items()}
    judges = {}
    for which in ("opus", "haiku"):
        d = eval_dir(root) / "judge"
        judges[which] = sum(1 for _ in d.glob(f"{which}-*/*.json")) if d.exists() else 0
    return {"prompts_digest": file_digest(root / dt.TRAIN_DIR / "eval_prompts.jsonl"), "bar": (read_bar(root) or {}).get("wins"),
            "votes_digest": hashlib.sha256(json.dumps(read_votes(root)["answers"], sort_keys=True).encode()).hexdigest()[:8],
            "bits": bits, "judge_calls": judges}


def score(root: pathlib.Path | None = None) -> dict:
    root = root or dp.DATA
    prompts = load_prompts(root)
    bar, state = read_bar(root), read_votes(root)
    items = plan(prompts, root)
    pool, _, pairs = dp.need_pairs(root)
    res = {"status": "pending" if len(state["answers"]) < 2 * N_VOTE or bar is None else "done", "bar": bar,
           "plan_changed_after_voting": plan_changed(items, state),
           "proposed_wins": PROPOSED_WINS, "votes": tally(items, state, bar) if items else None, "latency": latency(prompts, root),
           "judges": {w: judge_tally(pairs, prompts, root, w) for w in ("opus", "haiku")},
           "judge_note": "reported only; haiku labelled B's training pairs, so its read of B is circular"}
    return res


def run(prompts_digest: str = "", bar=None, votes_digest: str = "", bits=None, judge_calls=None) -> dict:
    """Exp 38's record (params pin the prompts, the bar, the votes and the bits it was scored on)."""
    return score()


def stored() -> dict | None:
    path = cache.METRICS / f"{NAME}.json"
    return json.loads(path.read_text())["results"] if path.exists() else None


def table(res: dict) -> str:
    rows = ["| Check | Result | Bar |", "|---|---|---|"]
    bar = res["bar"]["wins"] if res["bar"] else None
    for comp, v in (res["votes"] or {}).items():
        rows.append(f"| {comp} against A, your blind votes | {v['wins']} wins, {v['losses']} losses, {v['ties']} ties "
                    f"({v['answered']} of {v['of']} answered) | at least {bar if bar else 'not fixed yet'} wins: {v['verdict']} |")
    for w, row in res["latency"]["writers"].items():
        if row.get("bits"):
            rows.append(f"| {w} bit time, median / slowest 5% | {row['p50_s']} s / {row['p95_s']} s ({row['bits']} bits, "
                        f"{row['threads_lost_to_time']} threads lost to time) | "
                        f"{'within' if row['within_budget'] else 'over'} the {res['latency']['budget_s']:.0f} s budget |")
    for which, comps in res["judges"].items():
        for comp, v in comps.items():
            if v["consistent"] or v["flips"]:
                rows.append(f"| {which} judge, {comp}: first writer wins | {v['first_wins']} of {v['consistent']} consistent "
                            f"({v['flips']} disagreed) | reported only{', circular for B' if which == 'haiku' and 'B' in comp else ''} |")
    rows.append(f"| Status | {res['status']} | |")
    return "\n".join(rows)


# --- command line --------------------------------------------------------------------------------------------------------

def writer_name(name: str) -> str:
    return WRITERS.get(name.upper(), name) if len(name) == 1 else name


def main(a, root: pathlib.Path | None = None) -> None:
    from app import llm
    root = root or dp.DATA
    sub = a.sub
    if sub == "power":
        print(power_text())
    elif sub == "write":
        if not a.writer:
            raise SystemExit("eval write needs --writer (A, B, C or a model name)")
        pool = eval_pool(root)
        client = generate.writer_client(writer_name(a.writer))
        if not getattr(client, "local", False):
            raise SystemExit("the eval writer is a local model: set JOKE_WRITER_PROVIDER=ollama")
        print("One writer at a time, nothing else loaded (`ollama stop` the others): bit times are the latency measure.", flush=True)
        generate.warm_up(generate.ThreadGenerator(client))
        live_safety = generate.LiveSafety(llm.from_env(generate.FILTER_TIMING[0], role="safety_check", retries=generate.FILTER_RETRIES,
                                                       deadline_s=generate.FILTER_TIMING[1]))
        try:
            print(f"wrote {dp.write('eval', client, live_safety, pool, root, n=a.n)} new bits")
        except KeyboardInterrupt:
            raise SystemExit("stopped; finished bits are saved, rerun to resume")
    elif sub == "plan":
        items = plan(load_prompts(root), root)
        print(f"{len(items)} of {2 * N_VOTE} vote pairs ready ({', '.join(f'{c}: {sum(i['comp'] == c for i in items)}' for c in VOTED)})")
        for k, w in WRITERS.items():
            print(f"{k} {w}: {sum(bit_of(root, w, p['id']) is not None for p in load_prompts(root))} of {len(load_prompts(root))} bits")
    elif sub == "bar":
        if a.wins is None:
            raise SystemExit("eval bar --wins N")
        print(set_bar(root, a.wins))
    elif sub == "vote":
        try:
            vote(plan(load_prompts(root), root), root, limit=a.limit)
        except (KeyboardInterrupt, EOFError):
            print("\nstopped; every answer so far is saved.")
    elif sub == "judge":
        pool, _, pairs = dp.need_pairs(root)
        client = make_client(a.model)
        n = judge_all(client, pairs, load_prompts(root), root, a.model)
        print(f"{n} calls made; tokens in {client.input_tokens}, out {client.output_tokens}, "
              f"cache read {getattr(client, 'cache_read_tokens', 0)}, cache write {getattr(client, 'cache_write_tokens', 0)}")
    elif sub == "score":
        res = score(root)
        if a.record:
            res = cache.run_or_load(NAME, run, rerun=True, **params(root))
        print(table(res))
    else:
        raise SystemExit("eval write | plan | power | bar | vote | judge | score")
