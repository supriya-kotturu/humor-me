"""Exp 40 (D-125): did the name-check fix (rules-v4) loosen safety, and how many threads does it free?

Rules only: no API, no Ollama, local files read-only. Re-runs the rules on every stored generated line twice, with the
rules as they were (rules-v3, read from git) and as they are, and compares per pool. Pools: the DPO ranking bits (bonsai),
the training bits (qwen3) and the generated candidate lines of logged live shows (never audience speech).

Numbers only in the record: counts, and a word only when it is one of the plain words the fix added. No joke text is ever
stored. `review` shows every freed thread whose first rejected word was not a plain word, then a seeded 50 of the other
freed threads, one by one, and keeps your yes/no in a gitignored file; bar (b) of D-125 is that file.

    python -m sim.name_check run --ranking DIR --train DIR [--sessions DIR] [--record]
    python -m sim.name_check review --ranking DIR --train DIR [--sessions DIR]
"""
import argparse
import collections
import json
import pathlib
import random
import subprocess
import sys
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from app import safety  # noqa: E402
from sim import cache  # noqa: E402

NAME = "name_check"
BEFORE_REF = "3cc32b8"  # the last commit with rules-v3
SEED = 40
SAMPLE = 50
REVIEW_FILE = ROOT / "data" / "name_check" / "review.json"

# Bar (a): every line here must still be rejected, as a named person, by the rules.
MUST_STILL_REJECT = [
    "I told Dave that my battery was fading.",
    "My coworker Priya keeps unplugging me.",
    "Then Sarah laughed at my cable.",
    "Mr Smith unplugged me again.",
    "Dr. Lee asked about my firmware.",
    "She said 'Sarah, please stop unplugging me.'",
    "She said, “Sarah is unplugging me.”",
    "She said \"Dave is unplugging me.\"",
    "Here is the culprit: Kevin unplugged me.",
    "My boss said - Dave unplugged me.",
    "He sighed… Kevin unplugged me again.",
    "Wait… Linda is unplugging me!",
    "Who unplugged me? Tom did, I think.",
    "Oh, Dave did it again.",
    "That's what Priya said to me yesterday.",
    "My landlord Gregor unplugged me.",  # not a listed first name: a mid-sentence capital
    # added in review: surnames that are not in FIRST_NAMES, right after a clause opener
    "Breaking news: Musk unplugged me.",
    "My boss said - Obama unplugged me.",
    "My boss said — Bezos unplugged me.",
    "He sighed… Gregor unplugged me again.",
    "He said, 'Swift's the one who unplugged me.'",
]
PLAIN = safety.SENTENCE_WORDS | safety.CAPS_OK  # the only words a record may list: none is a person


def plain(word: str) -> bool:
    return word in PLAIN or (word.endswith("'s") and word[:-2] in PLAIN)


def load_before() -> types.ModuleType:
    """The rules as they were at BEFORE_REF, run from the same path so the blocklist files resolve the same way."""
    src = subprocess.run(["git", "show", f"{BEFORE_REF}:backend/app/safety.py"], cwd=ROOT, check=True,
                         capture_output=True, text=True, encoding="utf-8").stdout
    mod = types.ModuleType("safety_before")
    mod.__file__ = safety.__file__
    exec(compile(src, "safety_before", "exec"), mod.__dict__)
    assert mod.RULES_VERSION == "rules-v3"
    return mod


def read_bits(directory: pathlib.Path) -> list[dict]:
    """One unit per generated thread: key, age, lines. Threads that came back empty are not units."""
    units = []
    for path in sorted(pathlib.Path(directory).glob("*.json")):
        rec = json.loads(path.read_text(encoding="utf-8"))
        for i, t in enumerate(rec.get("threads") or []):
            units.append({"key": f"{rec['id']}:{i}", "age": rec["context"]["age"], "lines": list(t["lines"])})
    return units


def read_sessions(directory: pathlib.Path) -> list[dict]:
    """One unit per distinct generated candidate or played line of a logged show. Nothing the audience said is read."""
    units = []
    for path in sorted(pathlib.Path(directory).glob("*.jsonl")):
        age, seen = "mixed", set()
        for raw in path.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(raw)
            except ValueError:
                continue
            kind = ev.get("event")
            if kind == "show_start":
                age = (ev.get("context") or {}).get("age", "mixed")
            texts = [i.get("text") for i in ev.get("items", [])] if kind == "candidates" else [ev.get("text")] if kind == "joke_played" else []
            for text in texts:
                if isinstance(text, str) and text not in seen:
                    seen.add(text)
                    units.append({"key": f"{path.stem}:{len(units)}", "age": age, "lines": [text]})
    return units


def first_reason(mod, unit: dict) -> str | None:
    """The reason of the first line the rules reject, or None when every line passes."""
    for line in unit["lines"]:
        v = mod.check_rules(line, unit["age"])
        if not v.ok:
            return v.reason
    return None


def kind(reason: str | None) -> str:
    return "clean" if reason is None else reason.split(":", 1)[0]


def count_pool(units: list[dict], before, after) -> tuple[dict, list[dict]]:
    """Counts for one pool, and the units that flip from rejected to clean."""
    rb, ra = ({u["key"]: first_reason(m, u) for u in units} for m in (before, after))
    flipped = [u for u in units if rb[u["key"]] and ra[u["key"]] is None]
    words_before = collections.Counter(r.split(":", 1)[1] for r in rb.values() if r and r.startswith("named_person:"))
    words_after = collections.Counter(r.split(":", 1)[1] for r in ra.values() if r and r.startswith("named_person:"))
    n = len(units)
    rej = lambda r: sum(v is not None for v in r.values())  # noqa: E731
    res = {"units": n, "rejected_v3": rej(rb), "rejected_v4": rej(ra),
           "share_v3": round(rej(rb) / n, 4) if n else None, "share_v4": round(rej(ra) / n, 4) if n else None,
           "reasons_v3": dict(collections.Counter(kind(v) for v in rb.values() if v)),
           "reasons_v4": dict(collections.Counter(kind(v) for v in ra.values() if v)),
           "flipped": len(flipped),
           "named_person_v3": sum(words_before.values()), "named_person_v4": sum(words_after.values()),
           # the first rejected word per thread; only plain words are listed, the rest are only counted
           "top_named_v3": [[w, c] for w, c in words_before.most_common() if plain(w)][:25],
           "other_named_v3": sum(c for w, c in words_before.items() if not plain(w)),
           "top_flipped": [[w, c] for w, c in collections.Counter(
               rb[u["key"]].split(":", 1)[1] for u in flipped if rb[u["key"]].startswith("named_person:")
           ).most_common() if plain(w)][:25],
           # counted from every freed thread, not from the capped list above
           "flipped_unlisted_words": len(risky(flipped, rb))}
    return res, flipped


def risky(flipped: list[dict], reasons: dict) -> list[dict]:
    """Freed threads whose first word rejected by rules-v3 is not a plain word: the ones a person must read."""
    return [u for u in flipped if not (r := reasons[u["key"]]).startswith("named_person:") or not plain(r.split(":", 1)[1])]


def must_still_reject(after) -> dict:
    out = [after.check_rules(line, "mixed") for line in MUST_STILL_REJECT]
    return {"lines": len(out), "rejected": sum(not v.ok for v in out),
            "named_person": sum(v.reason.startswith("named_person") for v in out)}


def pools(ranking, train, sessions) -> dict[str, list[dict]]:
    out = {"bonsai ranking": read_bits(ranking), "qwen3 training": read_bits(train)}
    if sessions and pathlib.Path(sessions).exists():
        out["live shows"] = read_sessions(sessions)
    return out


def sample_flipped(flipped: dict[str, list[dict]], seed: int = SEED, n: int = SAMPLE,
                   must: frozenset = frozenset()) -> list[tuple[str, dict]]:
    """Every freed thread in `must` (pool|key, the risky ones), then a seeded n of the rest, all pools together."""
    allf = sorted(((p, u) for p, us in flipped.items() for u in us), key=lambda x: (x[0], x[1]["key"]))
    forced = [x for x in allf if key_of(*x) in must]
    rest = [x for x in allf if key_of(*x) not in must]
    return forced + random.Random(seed).sample(rest, min(n, len(rest)))


def freed(units_by_pool: dict[str, list[dict]], before, after) -> tuple[dict, dict, frozenset]:
    """Per pool: counts, the freed threads, and the pool|key of every risky freed thread."""
    per_pool, flipped, must = {}, {}, set()
    for name, units in units_by_pool.items():
        per_pool[name], flipped[name] = count_pool(units, before, after)
        reasons = {u["key"]: first_reason(before, u) for u in flipped[name]}
        must |= {key_of(name, u) for u in risky(flipped[name], reasons)}
    return per_pool, flipped, frozenset(must)


def key_of(pool: str, unit: dict) -> str:
    return f"{pool}|{unit['key']}"


def review_state(path: pathlib.Path = REVIEW_FILE) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def review_summary(state: dict, size: int) -> dict:
    answers = [a for a in state.values() if a in ("yes", "no")]
    return {"sample": size, "reviewed": len(answers), "unsafe_or_real_name": answers.count("yes")}


def verdict(bar_a: dict, rev: dict) -> str:
    if bar_a["rejected"] != bar_a["lines"]:
        return "fail"
    if rev["unsafe_or_real_name"]:
        return "fail"
    return "adopt" if rev["reviewed"] == rev["sample"] else "pending"


def run(before_ref: str = BEFORE_REF, seed: int = SEED, sample: int = SAMPLE, rules: str = "") -> dict:
    """Exp 40's record. Reads the paths in SOURCE; the params pin the rules and the sample, not the machine's folders."""
    before = load_before()
    per_pool, flipped, must = freed(pools(**SOURCE), before, safety)
    picked = sample_flipped(flipped, seed, sample, must)
    bar_a = must_still_reject(safety)
    mine = {key_of(p, u) for p, u in picked}
    rev = review_summary({k: v for k, v in review_state().items() if k in mine}, len(picked))
    return {"rules_before": before.RULES_VERSION, "rules_after": safety.RULES_VERSION, "pools": per_pool,
            "bar_a": bar_a, "bar_b": {**rev, "risky_all_read": len(must)}, "verdict": verdict(bar_a, rev)}


SOURCE = {"ranking": ROOT / "data/dpo/bits/ranking", "train": ROOT / "data/dpo/bits/train-qwen3",
          "sessions": ROOT / "data/sessions"}


def stored() -> dict | None:
    path = cache.METRICS / f"{NAME}.json"
    return json.loads(path.read_text(encoding="utf-8"))["results"] if path.exists() else None


def table(res: dict) -> str:
    rows = ["| Pool | Threads | Rejected by rules-v3 | Rejected by rules-v4 | Freed |", "|---|---|---|---|---|"]
    for name, p in res["pools"].items():
        rows.append(f"| {name} | {p['units']} | {p['rejected_v3']} ({p['share_v3']:.1%}) | "
                    f"{p['rejected_v4']} ({p['share_v4']:.1%}) | {p['flipped']} |")
    return "\n".join(rows)


def review(picked: list[tuple[str, dict]], path: pathlib.Path = REVIEW_FILE, ask=input, out=print) -> dict:
    """Shows each sampled thread and keeps yes (a real name or anything unsafe) or no. Resumes; 'q' stops."""
    state = review_state(path)
    for n, (pool, unit) in enumerate(picked, 1):
        key = key_of(pool, unit)
        if key in state:
            continue
        out(f"\n--- {n} of {len(picked)} ({pool}, {unit['age']}) ---")
        for line in unit["lines"]:
            out(f"  {line}")
        while (a := ask("Any real person's name or anything unsafe? [y/n/q] ").strip().lower()) not in ("y", "n", "q"):
            pass
        if a == "q":
            break
        state[key] = "yes" if a == "y" else "no"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    out(f"\n{json.dumps(review_summary({k: state[k] for k in state if k in {f'{p}|{u['key']}' for p, u in picked}}, len(picked)))}")
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "review"):
        s = sub.add_parser(name)
        s.add_argument("--ranking", type=pathlib.Path, default=SOURCE["ranking"])
        s.add_argument("--train", type=pathlib.Path, default=SOURCE["train"])
        s.add_argument("--sessions", type=pathlib.Path, default=SOURCE["sessions"])
        if name == "run":
            s.add_argument("--record", action="store_true", help=f"write reports/metrics/{NAME}.json")
    a = ap.parse_args()
    SOURCE.update(ranking=a.ranking, train=a.train, sessions=a.sessions)
    if a.cmd == "run":
        params = {"before_ref": BEFORE_REF, "seed": SEED, "sample": SAMPLE, "rules": safety.RULES_VERSION}
        res = cache.run_or_load(NAME, run, rerun=True, **params) if a.record else run()
        print(table(res))
        print(f"bar (a): {res['bar_a']}; bar (b): {res['bar_b']}; verdict: {res['verdict']}")
    else:
        _, flipped, must = freed(pools(a.ranking, a.train, a.sessions), load_before(), safety)
        review(sample_flipped(flipped, must=must))


if __name__ == "__main__":
    main()
