"""Label the R2 sample (the jokes in modes.jsonl) with the room each joke suits best: venue, energy,
age, in the sim's vocabulary (a test keeps them equal; sim isn't imported so the cluster venv needs no yaml). For DPO experiments (D-89); the labels are the LLM's guess,
not anything measured in a room.

Usage (on an Explorer GPU node, from the repo root; -m so the explorer package imports):
    python -m explorer.label_state --n 50 --out /tmp/state_smoke.jsonl     # smoke test
    python -m explorer.label_state                                          # all of modes.jsonl
    python -m explorer.label_state --export data/processed/rjokes/dpo_state.jsonl   # join, CPU only

Resumable like label_modes.py. The label file holds no joke text; --export writes text + mode + state
in one file to hand over, which stays local (rJokes text is never committed).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

from explorer.label_modes import JSON_OBJ, chunks, load_done, make_generator

VENUE = {
    "bar": "late-night bar crowd, a few drinks in, came for a night out",
    "company_event": "colleagues at a work event, bosses in the room, must stay office-safe",
    "showcase": "comedy-club showcase, people who came for comedy and like clever or odd material",
}
ENERGY = {
    "low": "tired or quiet room: the joke must land easily, gentle, short setup",
    "normal": "an ordinary attentive room",
    "high": "lively, loud room ready for fast, edgy or big material",
}
AGE = {
    "teens": "13 to 17 years old: nothing sexual, no drugs or alcohol",
    "young_adults": "18 to 30",
    "mixed": "all ages together, including older people",
}

SYSTEM = "You classify jokes for a research dataset. Answer with one JSON object and nothing else."


def _block(d: dict) -> str:
    return "\n".join(f'- "{k}": {v}' for k, v in d.items())


PROMPT = f"""Joke:
<<<{{joke}}>>>

A comedian wants to tell this joke on stage. Which room would it work best in?
1. venue, pick one:
{_block(VENUE)}
2. energy of the room, pick one:
{_block(ENERGY)}
3. audience age, pick one:
{_block(AGE)}

Return JSON: {{"venue": "...", "energy": "...", "age": "..."}}"""

FIELDS = {"venue": VENUE, "energy": ENERGY, "age": AGE}


def parse(text: str) -> dict | None:
    m = JSON_OBJ.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    out = {k: str(d.get(k, "")).lower().strip() for k in FIELDS}
    return out if all(out[k] in FIELDS[k] for k in FIELDS) else None


def sample_ids(modes_path: pathlib.Path) -> list[str]:
    """R2's ids in their labeling order, which is the seed's shuffle, so --n takes a uniform prefix."""
    return [json.loads(line)["id"] for line in open(modes_path, encoding="utf-8")]


def export(modes_path, state_path, input_path, out_path) -> tuple[int, int]:
    """One row per joke with a valid mode and state label. Returns (written, skipped)."""
    states = {r["id"]: r for r in map(json.loads, open(state_path, encoding="utf-8")) if "venue" in r}
    modes = {r["id"]: r for r in map(json.loads, open(modes_path, encoding="utf-8")) if r.get("mode")}
    keep = states.keys() & modes.keys()
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for line in open(input_path, encoding="utf-8"):
            r = json.loads(line)
            if r["id"] not in keep:
                continue
            m, s = modes[r["id"]], states[r["id"]]
            f.write(json.dumps({
                "id": r["id"], "text": r["text"], "score": m["score"], "pct": m["pct"],
                "mode": m["mode"], "mode2": m.get("mode2"), "safety": m.get("safety"),
                "venue": s["venue"], "energy": s["energy"], "age": s["age"],
            }) + "\n")
            written += 1
    return written, len(modes.keys() | states.keys()) - written


EXPORT_PATH = "data/processed/rjokes/dpo_state.jsonl"
STATE_PATH = "data/processed/rjokes/state.jsonl"
MODES_PATH = "data/processed/rjokes/modes.jsonl"
INFO_BAR = 0.80  # a field whose top value covers more than this tells the judge almost nothing (D-89, fixed before the run)


def run(export: str = EXPORT_PATH, state: str = STATE_PATH, modes: str = MODES_PATH) -> dict:
    """Exp 25: how the room labels spread. Counts only, so the result can be committed; the files stay on the cluster."""
    rows = [json.loads(line) for line in open(export, encoding="utf-8")]
    labeled = [json.loads(line) for line in open(state, encoding="utf-8")]
    n_modes = sum(1 for _ in open(modes, encoding="utf-8"))
    fields = {}
    for k, vocab in FIELDS.items():
        counts = {v: 0 for v in vocab}
        for r in rows:
            counts[r[k]] += 1
        fields[k] = sorted(counts.items(), key=lambda kv: -kv[1])
    return {"n": len(rows), "sample": n_modes, "left_out": n_modes - len(rows),
            "parse_failures": sum("raw" in r for r in labeled), "fields": fields, "info_bar": INFO_BAR,
            "top_share": {k: v[0][1] / len(rows) for k, v in fields.items()}}


def table(r: dict) -> str:
    out = ["| Field | Value | Jokes | Share |", "|---|---|---|---|"]
    for k, counts in r["fields"].items():
        for i, (v, c) in enumerate(counts):
            out.append(f"| {k if i == 0 else ''} | {v} | {c:,} | {c / r['n']:.1%} |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/processed/rjokes/train.jsonl")
    ap.add_argument("--modes", default="data/processed/rjokes/modes.jsonl")
    ap.add_argument("--out", default="data/processed/rjokes/state.jsonl")
    ap.add_argument("--n", type=int, default=None, help="label only the first n of the R2 sample")
    ap.add_argument("--export", default=None, help="write text + mode + state to this path and exit")
    ap.add_argument("--spread", action="store_true", help="Exp 25: count the labels in the export and record them")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--dtype", default="auto", help="auto | half | bfloat16 (use half on V100)")
    ap.add_argument("--backend", default="transformers", choices=["transformers", "vllm"])
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    if args.spread:  # Exp 25, on the cluster after the export: writes reports/metrics/rjokes_state.json (numbers only)
        from sim.cache import run_or_load
        print(table(run_or_load("rjokes_state", run, rerun=True, export=EXPORT_PATH, state=STATE_PATH, modes=MODES_PATH)))
        return
    if args.export:
        written, skipped = export(args.modes, args.out, args.input, args.export)
        print(f"wrote {written} rows to {args.export}; {skipped} without both labels left out")
        return

    out = pathlib.Path(args.out)
    done, models = load_done(out)
    if models - {args.model}:
        sys.exit(f"{out} was labeled by {sorted(models)}; refusing to mix in {args.model}. Use another --out.")
    ids = sample_ids(pathlib.Path(args.modes))[:args.n]
    want = {i for i in ids if i not in done}
    text = {}
    for line in open(args.input, encoding="utf-8"):
        r = json.loads(line)
        if r["id"] in want:
            text[r["id"]] = r["text"]
    todo = [i for i in ids if i in text]
    print(f"{len(done)} already labeled; {len(todo)} to go", flush=True)
    if not todo:
        return

    generate = make_generator(args)
    t0, labeled, failed = time.time(), 0, 0
    for batch in chunks(todo, args.chunk):
        convs = [[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": PROMPT.replace("{joke}", text[i])}] for i in batch]
        outs = generate(convs)
        with open(out, "a", encoding="utf-8") as f:
            for i, reply in zip(batch, outs):
                rec = {"id": i, "model": args.model, "seed": args.seed}
                lab = parse(reply)
                if lab:
                    rec.update(lab)
                else:
                    failed += 1
                    rec["raw"] = reply[:200]
                f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        labeled += len(batch)
        print(f"{len(done) + labeled}/{len(ids)} labeled, {labeled / (time.time() - t0):.1f}/s, "
              f"parse failures {failed}", flush=True)


if __name__ == "__main__":
    main()
