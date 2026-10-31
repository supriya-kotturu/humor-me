"""R2: label a uniform sample of rJokes with a HumorGen mode + safety flags, using a local LLM (vLLM).

Usage (on an Explorer GPU node, from the repo root):
    python explorer/label_modes.py --input data/processed/rjokes/train.jsonl --n 30000
    python explorer/label_modes.py --input ... --n 50 --out /tmp/smoke.jsonl     # smoke test

Resumable: labels go to --out in checkpointed chunks; rerunning the same command skips what's
done, on any node. Extending the sample = same seed, larger --n.

Uniform (not stratified) sample so per-mode score distributions need no reweighting.
Writes one JSON line per joke: id, pct, score, mode, mode2, safety{...}, raw on parse failure.
No joke text in the output, so results can be copied back and committed if useful.
"""
import argparse
import json
import os
import pathlib
import random
import re
import sys
import time

MODES = {
    "neurotic": "internal anxiety, overthinking, social insecurity (tension release)",
    "cynic": "hypocrisy, biting sarcasm, moral contradictions (superiority, social critique)",
    "observer": "mundane minutiae, unwritten awkward social norms (incongruity)",
    "wordsmith": "puns, double meanings, sound play (linguistic ambiguity)",
    "optimist": "wholesome reinterpretation of something negative (benign violation)",
    "absurdist": "non-sequiturs, dream logic, broken cause and effect (surrealism)",
}
SAFETY = {
    "targets_group": "the punchline depends on a race, religion, nationality, gender, sexuality or disability",
    "about_bodies": "mocks appearance, weight, or body parts",
    "sexual": "sexual content or innuendo",
    "alcohol_drugs": "alcohol or drug use",
    "slur": "contains a slur",
}

SYSTEM = "You classify jokes for a research dataset. Answer with one JSON object and nothing else."
TEMPLATE = """Joke:
<<<{joke}>>>

1. Which comedic mode does this joke mainly use? Pick one, and optionally a second:
{modes}
Use "none" if no mode fits.

2. Safety flags, true or false:
{safety}

Return JSON: {{"mode": "...", "mode2": "...|null", {keys}}}"""

PROMPT = TEMPLATE.replace("{modes}", "\n".join(f'- "{k}": {v}' for k, v in MODES.items())) \
    .replace("{safety}", "\n".join(f'- "{k}": {v}' for k, v in SAFETY.items())) \
    .replace("{keys}", ", ".join(f'"{k}": false' for k in SAFETY))
JSON_OBJ = re.compile(r"\{.*\}", re.S)


def parse(text: str) -> dict | None:
    m = JSON_OBJ.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    mode = str(d.get("mode", "")).lower()
    if mode not in MODES and mode != "none":
        return None
    mode2 = str(d.get("mode2") or "").lower()
    return {
        "mode": mode,
        "mode2": mode2 if mode2 in MODES and mode2 != mode else None,
        "safety": {k: bool(d.get(k, False)) for k in SAFETY},
    }


def load_done(path: pathlib.Path) -> tuple[set[str], set[str]]:
    """Ids already labeled, and the models that labeled them. Drops a torn last line (a job
    killed mid-write) by rewriting the file without it, so appending stays valid JSONL."""
    if not path.exists():
        return set(), set()
    good, torn = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            good.append(json.loads(line))
        except json.JSONDecodeError:
            torn = True
    if torn:
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in good), encoding="utf-8")
        os.replace(tmp, path)
    return {r["id"] for r in good}, {r["model"] for r in good}


def pending(rows: list[dict], done: set[str], n: int, seed: int) -> list[dict]:
    """The first n jokes of the seed's fixed shuffle that aren't labeled yet. Same seed → same
    order, so an interrupted run resumes exactly, and a larger n extends the same sample."""
    order = rows[:]
    random.Random(seed).shuffle(order)
    return [r for r in order[:n] if r["id"] not in done]


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="data/processed/rjokes/modes.jsonl")
    ap.add_argument("--n", type=int, default=30000, help="total sample size (grow it to extend the same sample)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=2000, help="jokes per checkpoint")
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    done, models = load_done(out)
    if models - {args.model}:
        sys.exit(f"{out} was labeled by {sorted(models)}; refusing to mix in {args.model}. Use another --out.")
    rows = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    todo = pending(rows, done, args.n, args.seed)
    print(f"{len(done)} already labeled; {len(todo)} to go (n={args.n}, seed={args.seed})", flush=True)
    if not todo:
        return

    from vllm import LLM, SamplingParams  # imported late so --help and resume checks work without a GPU

    llm = LLM(model=args.model, max_model_len=args.max_model_len, seed=args.seed)
    params = SamplingParams(temperature=0.0, max_tokens=120)
    t0, labeled, failed = time.time(), 0, 0
    for batch in chunks(todo, args.chunk):
        convs = [[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": PROMPT.replace("{joke}", r["text"])}] for r in batch]
        outs = llm.chat(convs, params)
        with open(out, "a", encoding="utf-8") as f:
            for r, o in zip(batch, outs):
                text = o.outputs[0].text
                lab = parse(text)
                rec = {"id": r["id"], "score": r["score"], "pct": r["pct"], "model": args.model, "seed": args.seed}
                if lab:
                    rec.update(lab)
                else:
                    failed += 1
                    rec["raw"] = text[:200]
                f.write(json.dumps(rec) + "\n")
            # The checkpoint: once a chunk is on disk, a killed job never redoes it.
            f.flush()
            os.fsync(f.fileno())
        labeled += len(batch)
        dt = time.time() - t0
        print(f"{len(done) + labeled}/{args.n} labeled, {labeled / dt:.1f}/s, parse failures {failed}", flush=True)


if __name__ == "__main__":
    main()
