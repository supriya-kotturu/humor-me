"""D-123: label the training pool with Haiku, export the datasets Explorer trains on, build the Modelfile for a trained writer.

Usage (after `python -m sim.dpo_pairs write --split train --writer qwen3:4b-instruct`, see sim/dpo_pairs.py):
    python -m sim.dpo_pairs label  --split train --writer qwen3:4b-instruct --env-file backend/.env   # Haiku, resumable, prints cost
    python -m sim.dpo_pairs export --writer qwen3:4b-instruct                                         # -> data/dpo/train/
    python -m sim.dpo_pairs modelfile --base Modelfile.base --gguf humor-qwen3-B-Q4_K_M.gguf --out Modelfile.B

Labels (Exp 36's configuration, D-121). Claude Haiku 5.5, thinking off, taught by the user's 60 teach pairs in the system message,
asked `{"funnier": "A"|"B"}`. Every pair among the first three clean threads of a bit (up to 3 pairs a bit), each in both orders.
A pair whose two orders name different threads gets no label and is dropped. This is RLAIF: the labels are an AI judge's taste,
and Exp 36 and 37 found neither Haiku nor Opus agrees with the user better than a coin flip (D-121, D-122). The system message is
a cache breakpoint (the 18,000-token teach prefix is read 4,800 times), the cache of answers is its own folder under
`data/dpo/labels/` (so the training split is frozen once labelling starts, `rebuild_train`'s guard), and a rerun asks only what
the cache lacks. The Message Batches API (half price) is not used: the repo's Claude client has no batch call, and adding one
unrun was not worth the risk; the labelling costs about $1.3 with caching (D-123).

Export. `data/dpo/train/` (gitignored): `sft_B.jsonl`, `dpo_B.jsonl`, `dpo_C.jsonl`, `eval_prompts.jsonl`, `manifest.json`. A
row is `{"id", "prompt": [system, user messages], ...}` and the reply texts are the raw replies the writer returned, so training
targets the exact live output format.
"""
import concurrent.futures
import hashlib
import itertools
import json
import pathlib
import random

from sim import dpo_pairs as dp
from sim.env import AGES, ENERGIES, VENUES

from app import generate, llm  # noqa: E402 - sim.dpo_pairs has put backend/ on the path

MAX_THREADS_PER_BIT = 3  # the first three clean threads: up to 3 pairs a bit (user's brief)
FAIL_STOP = 8  # calls that fail in a row: an outage, not bad luck (judgment call, not measured)
# Per million tokens, USD. Input and output are Haiku 5.5's (app.llm_anthropic.PRICE_PER_MTOK); a cache read at 0.1 times and a
# cache write at 1.25 times the input price are the standard multipliers, assumed (D-122 used the same for Opus), not read from a bill.
PRICE = {"input": 0.10, "output": 0.50, "cache_read": 0.01, "cache_write": 0.125}
EVAL_N = 60  # held-out prompts, half tonight: judgment call, see D-123
EVAL_SEED = "dpo-eval-v1"
TRAIN_DIR = "train"
RULE_SFT = ("the clean thread with the most wins among its labelled pairs, then the fewest losses, then the lowest thread index; "
            "a bit with no win gets no SFT row")


# --- labels --------------------------------------------------------------------------------------------------------------

def labels_dir(root: pathlib.Path) -> pathlib.Path:
    return root / "labels"


def label_file(root: pathlib.Path, split: str, writer: str) -> pathlib.Path:
    return labels_dir(root) / f"{dp.bits_folder(split, writer)}.jsonl"


def usage_file(root: pathlib.Path, split: str, writer: str) -> pathlib.Path:
    return labels_dir(root) / f"usage-{dp.bits_folder(split, writer)}.json"


def thread_pairs(rec: dict) -> list[tuple[int, int]]:
    """Every pair among the bit's first MAX_THREADS_PER_BIT clean threads, as thread indices (i < j)."""
    clean = [i for i, t in enumerate(rec["threads"]) if t.get("safe")][:MAX_THREADS_PER_BIT]
    return list(itertools.combinations(clean, 2))


def written_bits(split: str, writer: str, pool: dict, root: pathlib.Path) -> list[tuple[dict, dict]]:
    """(prompt, record) of every bit of the split that `writer` finished for exactly that prompt, in pool order."""
    out = []
    for p in (q for q in pool["prompts"] if q["split"] == split):
        rec = dp._read(dp.bit_path(root, split, p["id"], writer))
        if dp.done(rec) and dp.same_prompt(rec, p, writer):
            out.append((p, rec))
    return out


def cache_dir(root: pathlib.Path, split: str, writer: str, system: str) -> pathlib.Path:
    """One folder per teach prompt and judge (the digest), then per writer's bits: two writers' bits share ids, not threads."""
    return labels_dir(root) / "cache" / dp.judge_digest(system, "haiku") / dp.bits_folder(split, writer)


def call_path(directory: pathlib.Path, rec: dict, i: int, j: int, order: str) -> pathlib.Path:
    """The file name carries a hash of the two threads' lines, so an answer is only ever read back for the text it judged."""
    text = json.dumps([rec["threads"][i]["lines"], rec["threads"][j]["lines"]], ensure_ascii=False)
    return directory / f"{rec['id']}-{i}-{j}-{order}-{hashlib.sha256(text.encode()).hexdigest()[:8]}.json"


def make_label_client():
    """Exp 36's judge client (Haiku 5.5, thinking off, the safety check's timing and retries) with the teach prefix cached."""
    timeout, deadline = generate.FILTER_TIMING
    provider, model, thinking = dp.judge_model("haiku")
    return llm.from_env(timeout, role="safety_check", provider=provider, model=model, retries=generate.FILTER_RETRIES,
                        deadline_s=deadline, thinking=thinking, cache_system=True)


def cost_usd(usage: dict) -> float:
    return sum(usage.get(k, 0) * PRICE[k] for k in PRICE) / 1e6


def usage_of(client) -> dict:
    return {"input": client.input_tokens, "output": client.output_tokens, "cache_read": getattr(client, "cache_read_tokens", 0),
            "cache_write": getattr(client, "cache_write_tokens", 0)}


def add_usage(path: pathlib.Path, now: dict) -> dict:
    """Adds this run's token counts to the file's running total (a resumed labelling keeps its cost), returns the total."""
    total = dp._read(path) or {}
    out = {k: total.get(k, 0) + now.get(k, 0) for k in ("input", "output", "cache_read", "cache_write")}
    out["calls"] = total.get("calls", 0) + now.get("calls", 0)
    dp._write(path, out)
    return out


def estimate(todo: int, system: str) -> str:
    prefix = len(system) // 3  # characters / 3: D-122 measured this prefix at 24,074 tokens, a third more than characters / 4
    cached = todo * (prefix * PRICE["cache_read"] + 300 * PRICE["input"] + 12 * PRICE["output"]) / 1e6 + prefix * PRICE["cache_write"] / 1e6
    return f"{todo} calls to make, about ${cached:.2f} with the teach prefix cached (${todo * (prefix + 300) * PRICE['input'] / 1e6:.2f} without)"


def label_all(client, split: str, writer: str, pairs: list[dict], pool: dict, root: pathlib.Path | None = None, workers: int = 4,
              limit: int | None = None, out=print) -> dict:
    """Asks Haiku about every pair of every written bit in both orders, caching each answer; then writes the labels file.
    The first call runs alone so it writes the prompt cache, the rest run on `workers` threads. Stops after FAIL_STOP failed calls
    in a row, or `limit` new calls. Returns {"calls", "cost_usd", "total_cost_usd"}."""
    root = root or dp.DATA
    system = dp.teach_prompt(pairs)
    directory = cache_dir(root, split, writer, system)
    todo = [(p["id"], rec, i, j, order) for p, rec in written_bits(split, writer, pool, root) for i, j in thread_pairs(rec)
            for order in dp.ORDERS if dp._read(call_path(directory, rec, i, j, order)) is None]
    if limit is not None:
        todo = todo[:limit]
    out(estimate(len(todo), system))
    made = failed = 0

    def ask(item):
        bit, rec, i, j, order = item
        pair = {"left": rec["threads"][i]["lines"], "right": rec["threads"][j]["lines"]}
        try:
            letter = dp.parse_pick(client.chat(system, dp.judge_user(pair, order), json_reply=dp.JUDGE_SCHEMA, validate=dp.parse_pick))
        except (llm.LLMError, ValueError, KeyError, TypeError, AttributeError):
            return item, None  # an outage is not the judge's answer: not cached
        return item, letter

    def take(item, letter) -> bool:
        nonlocal made, failed
        if letter is None:
            failed += 1
            return failed < FAIL_STOP
        failed = 0
        _, rec, i, j, order = item
        dp._write(call_path(directory, rec, i, j, order), {"funnier": letter})
        made += 1
        if made % 50 == 0:
            out(f"{made} calls, ${cost_usd(usage_of(client)):.3f} so far")
        return True

    try:
        if todo:
            ok = take(*ask(todo[0]))
            rest = todo[1:]
            with concurrent.futures.ThreadPoolExecutor(max(1, workers)) as pool_:
                for k in range(0, len(rest) if ok else 0, max(1, workers) * 4):
                    chunk = rest[k:k + max(1, workers) * 4]
                    if not all(take(*r) for r in pool_.map(ask, chunk)):
                        out(f"{FAIL_STOP} calls failed in a row: is the API reachable? Finished answers are saved; rerun to resume")
                        break
    finally:
        now = {**usage_of(client), "calls": made}
        total = add_usage(usage_file(root, split, writer), now)
        write_labels(split, writer, pool, root, pairs)
    res = {"calls": made, "cost_usd": cost_usd(usage_of(client)), "total_cost_usd": cost_usd(total)}
    out(f"{made} calls this run, ${res['cost_usd']:.3f}; ${res['total_cost_usd']:.3f} in all for these labels "
        f"(tokens in {client.input_tokens}, out {client.output_tokens}, cache read {getattr(client, 'cache_read_tokens', 0)}, "
        f"cache write {getattr(client, 'cache_write_tokens', 0)})")
    return res


def collect_labels(split: str, writer: str, pool: dict, root: pathlib.Path, pairs: list[dict]) -> list[dict]:
    """One row per pair: its winner when Haiku named the same thread in both orders, `dropped` when the orders disagree,
    `missing` when an order has no answer yet. Rows are in pool order, then thread order, so the file is deterministic."""
    directory = cache_dir(root, split, writer, dp.teach_prompt(pairs))
    rows = []
    for p, rec in written_bits(split, writer, pool, root):
        for i, j in thread_pairs(rec):
            letters = {o: (dp._read(call_path(directory, rec, i, j, o)) or {}).get("funnier") for o in dp.ORDERS}
            picks = [dp.picked_thread(letters[o], o) for o in dp.ORDERS]
            if None in picks:
                status, winner = "missing", None
            elif picks[0] != picks[1]:
                status, winner = "dropped", None
            else:
                status, winner = "label", (i, j)[picks[0]]
            rows.append({"bit": p["id"], "i": i, "j": j, "status": status, "winner": winner, "ab": letters["ab"], "ba": letters["ba"]})
    return rows


def write_labels(split: str, writer: str, pool: dict, root: pathlib.Path, pairs: list[dict]) -> list[dict]:
    rows = collect_labels(split, writer, pool, root, pairs)
    path = label_file(root, split, writer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)
    return rows


def read_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def label_summary(rows: list[dict]) -> dict:
    n = len(rows)
    return {"pairs": n, "labelled": sum(r["status"] == "label" for r in rows), "dropped": sum(r["status"] == "dropped" for r in rows),
            "missing": sum(r["status"] == "missing" for r in rows)}


# --- export --------------------------------------------------------------------------------------------------------------

def split_request(request: str) -> list[dict]:
    """The system and user messages a bit's stored request was sent as. The request is `system + "\\n\\n" + user`, and the user
    message is one `json.dumps` line (newlines escaped), so the last blank line is the join."""
    system, user = request.rsplit("\n\n", 1)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def reply_text(thread: dict) -> str | None:
    """The raw reply the writer returned for a thread, or None when none was stored."""
    return thread.get("raw")


def canonical_reply(thread: dict) -> str:
    """For a thread with no stored reply (the bonsai ranking bits, written before replies were kept): the reply as the schema
    shapes it. Judgment call, not measured: Ollama's own spacing and escapes are unknown, so this is not byte-equal to what bonsai
    returned (D-123, writer C)."""
    return json.dumps({"premise": thread["premise"], "jokes": list(thread["lines"])}, ensure_ascii=False)


def thread_scores(rows: list[dict]) -> dict[str, dict[int, list[int]]]:
    """bit -> thread -> [wins, losses] over its labelled pairs."""
    out: dict[str, dict[int, list[int]]] = {}
    for r in rows:
        if r["status"] != "label":
            continue
        s = out.setdefault(r["bit"], {})
        loser = r["j"] if r["winner"] == r["i"] else r["i"]
        s.setdefault(r["winner"], [0, 0])[0] += 1
        s.setdefault(loser, [0, 0])[1] += 1
    return out


def best_thread(scores: dict[int, list[int]]) -> int | None:
    """RULE_SFT: most wins, then fewest losses, then lowest index; None when no thread has a win."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1][0], kv[1][1], kv[0]))
    return ranked[0][0] if ranked and ranked[0][1][0] > 0 else None


def export_b(rows: list[dict], bits: dict[str, dict]) -> tuple[list[dict], list[dict], dict]:
    """(sft rows, dpo rows, counts) for writer B from the labels. A thread with no stored reply cannot be a target: its rows are
    skipped and counted."""
    sft, dpo = [], []
    no_raw = same = 0
    for r in rows:
        if r["status"] != "label":
            continue
        rec = bits[r["bit"]]
        win, lose = (rec["threads"][k] for k in (r["winner"], r["j"] if r["winner"] == r["i"] else r["i"]))
        if reply_text(win) is None or reply_text(lose) is None:
            no_raw += 1
            continue
        if reply_text(win) == reply_text(lose):  # two samples that came out the same: no preference to learn
            same += 1
            continue
        dpo.append({"id": f"{r['bit']}-{r['i']}-{r['j']}", "prompt": split_request(rec["request"]), "chosen": reply_text(win),
                    "rejected": reply_text(lose)})
    for bit, scores in thread_scores(rows).items():
        k = best_thread(scores)
        if k is None:
            continue
        if reply_text(bits[bit]["threads"][k]) is None:
            no_raw += 1
            continue
        sft.append({"id": bit, "prompt": split_request(bits[bit]["request"]), "completion": reply_text(bits[bit]["threads"][k])})
    return sft, dpo, {"sft": len(sft), "dpo": len(dpo), "skipped_no_reply": no_raw, "skipped_identical": same}


def export_c(pool: dict, state: dict, root: pathlib.Path) -> tuple[list[dict], dict]:
    """The user's decided ranking pairs as DPO rows: chosen is the thread they picked. The bits are bonsai's (off-policy for a
    qwen3 base, D-123) and carry no stored reply, so both replies are `canonical_reply` of the parsed thread."""
    rows, ties = [], 0
    for p in pool["prompts"]:
        a = state["answers"].get(p["id"])
        if p["split"] != "ranking" or a is None:
            continue
        if a["choice"] not in ("1", "2"):
            ties += a["choice"] == "t"
            continue
        rec = dp._read(dp.bit_path(root, "ranking", p["id"]))
        win, lose = (a["left"], a["right"]) if a["choice"] == "1" else (a["right"], a["left"])
        t = rec["threads"]
        rows.append({"id": p["id"], "role": p["role"], "half": p["half"], "prompt": split_request(rec["request"]),
                     "chosen": reply_text(t[win]) or canonical_reply(t[win]), "rejected": reply_text(t[lose]) or canonical_reply(t[lose])})
    return rows, {"dpo": len(rows), "ties_left_out": ties}


def build_eval_prompts(pool: dict, n: int = EVAL_N, seed: str = EVAL_SEED) -> list[dict]:
    """New prompts from the pool's own distribution (D-123), disjoint from the ranking and train splits and from Exp 24, 26, 27
    and 30. Half carry tonight's context. Built like `dpo_pairs.build_pool`: modes round-robin, rooms seeded, a bit number 0 to 7,
    the headline gist round-robin on the bits that take one. The tonight half shares the stored day's headlines with training
    (one day exists, D-121): held out by prompt, not by topic."""
    if n % 2:
        raise ValueError("eval size must split evenly into halves")
    taken = dp.earlier_keys() | {dp.prompt_key(p) for p in pool["prompts"]}
    gists = [(d["date"], g) for d in pool["days"] for g in d["approved"]]
    random.Random(f"{seed}:gists").shuffle(gists)
    modes, cells, headline, prompts = tuple(generate.MODES), {h: 0 for h in dp.HALVES}, 0, []
    halves = [h for _ in range(n // 2) for h in dp.HALVES]
    random.Random(f"{seed}:halves").shuffle(halves)
    for i, half in enumerate(halves):
        k = cells[half]
        cells[half] += 1
        bit = k % dp.BITS_PER_SHOW
        p = {"id": f"e{i:04d}", "split": "eval", "half": half, "role": None, "mode": modes[(k // dp.BITS_PER_SHOW + bit) % len(modes)],
             "bit": bit, "day": None, "gist": None}
        if half == "tonight" and bit % dp.app_tonight.HEADLINE_EVERY == dp.app_tonight.HEADLINE_REMAINDER:
            p["day"], p["gist"] = gists[headline % len(gists)]
            headline += 1
        elif half == "tonight":
            p["day"] = pool["days"][(k // dp.BITS_PER_SHOW) % len(pool["days"])]["date"]
        r = random.Random(f"{seed}:{p['id']}")
        while True:
            for _ in range(500):
                p["context"] = {"venue": r.choice(VENUES), "energy": r.choice(ENERGIES), "age": r.choice(AGES)}
                p["told"] = r.sample(dp.TOPICS, min(p["bit"], dp.MAX_TOLD))
                if dp.prompt_key(p) not in taken:
                    break
            else:
                if p["bit"] != 0 or half != "plain":
                    raise SystemExit(f"eval too big: no free room for {p['id']}; lower the size")
                p["bit"] = 1
                continue
            break
        taken.add(dp.prompt_key(p))
        prompts.append(p)
    return prompts


def assert_eval_disjoint(eval_prompts: list[dict], pool: dict) -> None:
    keys = {dp.prompt_key(p) for p in eval_prompts}
    assert len(keys) == len(eval_prompts), "two eval prompts are the same request"
    assert not keys & {dp.prompt_key(p) for p in pool["prompts"]}, "an eval prompt equals a ranking or train prompt"
    assert not keys & dp.earlier_keys(), "an eval prompt equals one of Exp 24, 26, 27 or 30"
    assert sum(p["half"] == "tonight" for p in eval_prompts) * 2 == len(eval_prompts), "the eval is not half tonight"


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:8]


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def export(writer: str, root: pathlib.Path | None = None, split: str = "train", out=print) -> dict:
    """Writes `data/dpo/train/`. Needs the pool, the user's ranking and the finished labels. Overwrites its own files only."""
    root = root or dp.DATA
    pool = dp._read(dp.pool_path(root))
    if pool is None:
        raise SystemExit("run `python -m sim.dpo_pairs pool` first")
    state = dp.read_state(root)
    path = label_file(root, split, writer)
    if not path.exists():
        raise SystemExit(f"no labels at {path}: run `label` first")
    rows = read_jsonl(path)
    summary = label_summary(rows)
    if summary["missing"]:
        raise SystemExit(f"{summary['missing']} pairs are not labelled yet: rerun `label` until none is missing")
    bits = {p["id"]: rec for p, rec in written_bits(split, writer, pool, root)}
    sft, dpo_b, counts_b = export_b(rows, bits)
    dpo_c, counts_c = export_c(pool, state, root)
    eval_prompts = build_eval_prompts(pool)
    assert_eval_disjoint(eval_prompts, pool)
    d = root / TRAIN_DIR
    files = {"sft_B.jsonl": sft, "dpo_B.jsonl": dpo_b, "dpo_C.jsonl": dpo_c, "eval_prompts.jsonl": eval_prompts}
    for name, data in files.items():
        write_jsonl(d / name, data)
    manifest = {"writer": writer, "pool_digest": pool["digest"], "rank_answers_digest": dp.answers_digest(state),
                "labels_digest": sha(path), "labels": summary, "sft_rule": RULE_SFT, "B": counts_b, "C": counts_c,
                "eval_prompts": len(eval_prompts), "eval_seed": EVAL_SEED,
                "files": {name: {"rows": len(data), "sha8": sha(d / name)} for name, data in files.items()},
                "c_note": "off-policy: bonsai-2:4b's threads, ranked by the user; no stored reply, so replies are canonical JSON"}
    dp._write(d / "manifest.json", manifest)
    out(json.dumps(manifest, indent=1))
    return manifest


# --- the laptop's Modelfile ----------------------------------------------------------------------------------------------

def modelfile_for(base: str, gguf: str) -> str:
    """The base model's Modelfile (`ollama show qwen3:4b-instruct --modelfile`) with its weights replaced by `gguf`: the template
    and every parameter stay, so the trained writer is served exactly as the base is (D-124)."""
    out, replaced = [], False
    for line in base.splitlines():
        if line.startswith("FROM "):
            if not replaced:
                out.append(f"FROM ./{pathlib.PurePath(gguf).name}")
            replaced = True
            continue
        out.append(line)
    if not replaced:
        raise ValueError("the base Modelfile has no FROM line")
    return "\n".join(out) + "\n"
