"""S4+S5: bit boundaries and annotation in one LLM pass per set.

Usage:
    python -m pipeline.bits <set_id> [<set_id> ...] [--force]
    python -m pipeline.bits --all [--force]

Reads  data/processed/<dataset>/lines.jsonl, backend/.env (LLM_BASE_URL, LLM_API_KEY, PIPELINE_MODEL).
Writes data/interim/<set>.bits.raw.json (validated model output, cache) and
       data/processed/<dataset>/bits.jsonl (all sets processed so far).

The model sees numbered lines and answers with line numbers plus *paraphrased* premises and
hooks, so nothing it returns quotes the transcript; a verbatim-overlap check enforces that.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time

import httpx
from dotenv import load_dotenv

from pipeline.callbacks import candidates, corpus_df, render_hints
from pipeline.common import INTERIM, PROCESSED, ROOT

MODES = ["Neurotic", "Cynic", "Observer", "Wordsmith", "Optimist", "Absurdist"]
TYPES = ["prepared", "crowd_work", "callback", "segue"]
TECHNIQUES = ["escalation", "act_out", "tag_run", "misdirection", "wordplay", "self_deprecation",
              "rule_of_three", "analogy", "exaggeration", "deadpan", "story", "list"]
VERBATIM_NGRAM = 6  # a premise/hook sharing this many consecutive words with the set is a quote
MAX_ATTEMPTS = 3
HTTP_RETRIES = 6

SYSTEM = f"""You segment stand-up comedy transcripts into bits and annotate each bit.

Definitions:
- A bit is the SMALLEST unit that works on its own without earlier setup: one premise with its
  setup, punchlines, tags and act-outs. If a stretch could be performed alone, it is its own bit.
- Consecutive bits on the same topic form a thread: set "continues": true on each bit that
  stays on the previous bit's topic (a new angle on it). The first bit of a topic has false.
- A callback bit returns to an earlier bit's premise or image after OTHER material in between.
  It is a NEW bit with callback_to = the start line of the earlier bit, which must be at least
  2 bits back. Continuing the immediately previous topic is the same bit or a new bit, never a
  callback.
- Crowd work (talking with audience members) is its own bit, type "crowd_work".
- Greetings, thank-yous and transitions with no premise are type "segue".

Input: numbered lines "[NNNN] text", optionally followed by a laugh tag like ⟨L2⟩
(level of the audience laugh after that line; use it as evidence of punchlines).

Output: ONLY a JSON object:
{{"bits": [{{"start": <line number where the bit begins>,
            "premise": "<your own paraphrase, max 15 words>",
            "mode": one of {MODES},
            "type": one of {TYPES},
            "techniques": subset of {TECHNIQUES},
            "continues": <true if same topic as the previous bit, else false>,
            "callback_to": <start line of an earlier bit, or null>,
            "hook": "<memorable image a later callback could reuse, your own words, max 10 words, or null>"}}]}}

Work in this order:
1. Topic map: find every point where the subject changes, and every segue line that bridges
   two subjects.
2. Bits: within each subject, start a new bit wherever a new premise starts. Test each
   candidate: could a comedian perform this stretch on its own, as a complete joke, without the
   lines before it? Long stories usually hold several bits; split at each new premise even
   inside one story. Tags and act-outs on the same premise stay in that bit.
3. continues: true only when the bit stays on the previous bit's subject. A new subject is
   false, even if a segue links the two.
4. Callbacks: for EVERY bit after the second, ask whether it brings back a specific image,
   character, phrase, or premise from a bit at least 2 bits earlier, after other material.
   Callbacks can be lexical (the same object or name returns) or thematic (the same idea or
   self-image returns). Use the candidate list below the transcript if present, but verify
   each one; many are coincidences. A bit can both continue and call back.

Rules:
- The first bit starts at the first line number. Starts strictly increasing. Every line
  belongs to exactly one bit (a bit runs until the next start).
- NEVER copy phrases from the transcript into premise or hook; paraphrase in neutral words.
- Mode = the dominant comic stance of the bit: Neurotic (anxiety, overthinking),
  Cynic (sarcasm, social critique), Observer (mundane details, social norms),
  Wordsmith (puns, language play), Optimist (wholesome reframing), Absurdist (surreal logic).
"""


PROMPT_SHA = hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()[:12]


def load_env() -> tuple[str, str, str]:
    load_dotenv(ROOT / "backend" / ".env")
    try:
        # role names first (backend/sample.env), then the pre-rename ones
        get = lambda new, old: os.environ.get(new) or os.environ[old]  # noqa: E731
        return (get("GEMINI_BASE_URL", "LLM_BASE_URL").rstrip("/"), get("GEMINI_API_KEY", "LLM_API_KEY"),
                get("TRANSCRIPT_PIPELINE_MODEL", "PIPELINE_MODEL"))
    except KeyError as e:
        sys.exit(f"backend/.env is missing {e}; see backend/sample.env")


def load_set_lines(set_id: str) -> tuple[str, list[dict]]:
    manifest = json.loads((PROCESSED / "datasets.json").read_text(encoding="utf-8"))
    dataset = next((d for d in ("laugh_labeled", "structure_only") if set_id in manifest[d]), None)
    if dataset is None:
        sys.exit(f"{set_id} not in datasets.json")
    with open(PROCESSED / dataset / "lines.jsonl", encoding="utf-8") as f:
        lines = [r for r in map(json.loads, f) if r["set_id"] == set_id]
    return dataset, lines


def render(lines: list[dict]) -> str:
    return "\n".join(f"[{ln['line_idx']:04d}] {ln['text']}" + (f" ⟨L{ln['label']}⟩" if ln["label"] else "")
                     for ln in lines)


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def validate(data: object, lines: list[dict], source_ngrams: set) -> list[str]:
    """Return a list of problems (empty = valid). Messages never include transcript text."""
    if not isinstance(data, dict) or not isinstance(data.get("bits"), list) or not data["bits"]:
        return ['top level must be {"bits": [ ... ]} with at least one bit']
    idx = [ln["line_idx"] for ln in lines]
    valid_idx, errs, starts = set(idx), [], []
    for i, b in enumerate(data["bits"]):
        if not isinstance(b, dict):
            errs.append(f"bit {i}: not an object")
            continue
        s = b.get("start")
        if not isinstance(s, int) or s not in valid_idx:
            errs.append(f"bit {i}: start {s!r} is not a line number in this set")
        elif starts and s <= starts[-1]:
            errs.append(f"bit {i}: start {s} not after previous start {starts[-1]}")
        else:
            starts.append(s)
        if b.get("mode") not in MODES:
            errs.append(f"bit {i}: mode {b.get('mode')!r} not in {MODES}")
        if b.get("type") not in TYPES:
            errs.append(f"bit {i}: type {b.get('type')!r} not in {TYPES}")
        tech = b.get("techniques", [])
        if not isinstance(tech, list) or any(t not in TECHNIQUES for t in tech):
            errs.append(f"bit {i}: techniques must be a subset of {TECHNIQUES}")
        cb = b.get("callback_to")
        # The immediately preceding bit is a continuation, not a callback (design.md, Callbacks).
        if cb is not None and (not isinstance(cb, int) or cb not in starts[:-2]):
            errs.append(f"bit {i}: callback_to {cb!r} must be the start of a bit at least 2 bits "
                        f"earlier; returning to the previous bit is a continuation, not a callback")
        if b.get("type") == "callback" and cb is None:
            errs.append(f"bit {i}: type 'callback' needs callback_to")
        if not isinstance(b.get("continues", False), bool):
            errs.append(f"bit {i}: continues must be true or false")
        elif i == 0 and b.get("continues"):
            errs.append("bit 0 cannot continue a previous bit")
        for field, limit in (("premise", 15), ("hook", 10)):
            v = b.get(field)
            if field == "premise" and not (isinstance(v, str) and v.strip()):
                errs.append(f"bit {i}: premise missing")
            if isinstance(v, str):
                if len(v.split()) > limit + 5:
                    errs.append(f"bit {i}: {field} too long (max {limit} words)")
                if ngrams(v, VERBATIM_NGRAM) & source_ngrams:
                    errs.append(f"bit {i}: {field} copies {VERBATIM_NGRAM}+ consecutive words from the "
                                f"transcript; paraphrase it")
    if starts and starts[0] != idx[0]:
        errs.append(f"first bit must start at line {idx[0]}")
    return errs


def call_llm(base: str, key: str, model: str, messages: list[dict]) -> str:
    body = {"model": model, "messages": messages, "response_format": {"type": "json_object"}}
    for attempt in range(HTTP_RETRIES):
        try:
            r = httpx.post(f"{base}/chat/completions", json=body, timeout=600,
                           headers={"Authorization": f"Bearer {key}"})
        except httpx.TransportError as e:
            wait = 2 ** attempt * 5
            print(f"  network error ({type(e).__name__}); retry in {wait}s")
            time.sleep(wait)
            continue
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        if r.status_code in (429, 500, 502, 503, 504):
            wait = int(r.headers.get("retry-after", 0)) or 2 ** attempt * 10
            print(f"  HTTP {r.status_code}; retry in {wait}s")
            time.sleep(wait)
            continue
        sys.exit(f"HTTP {r.status_code}: {r.text[:300]}")
    sys.exit("gave up after repeated rate limits / server errors")


def annotate(set_id: str, lines: list[dict], force: bool, hints: str = "") -> dict:
    cache = INTERIM / f"{set_id}.bits.raw.json"
    source = ngrams(" ".join(ln["text"] for ln in lines), VERBATIM_NGRAM)
    if cache.exists() and not force:
        cached = json.loads(cache.read_text(encoding="utf-8"))
        # Rules and definitions change; a cache from another prompt or failing current rules is
        # re-requested, not trusted.
        if cached.get("prompt_sha") != PROMPT_SHA:
            print(f"{set_id}: cached output is from an older prompt; re-requesting")
        elif validate(cached, lines, source):
            print(f"{set_id}: cached output fails current rules; re-requesting")
        else:
            print(f"{set_id}: cached")
            return cached
    base, key, model = load_env()
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": render(lines) + hints}]
    for attempt in range(1, MAX_ATTEMPTS + 1):
        t0 = time.time()
        content = call_llm(base, key, model, messages)
        try:
            data = json.loads(content)
            errs = validate(data, lines, source)
        except json.JSONDecodeError as e:
            data, errs = None, [f"invalid JSON: {e.msg}"]
        print(f"{set_id}: attempt {attempt} ({time.time() - t0:.0f}s) -> "
              + ("valid" if not errs else f"{len(errs)} problems"))
        if not errs:
            data["model"] = model
            data["prompt_sha"] = PROMPT_SHA
            cache.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
            return data
        # Feed problems back; the conversation keeps the model's own answer, not transcript copies.
        messages += [{"role": "assistant", "content": content},
                     {"role": "user", "content": "Fix these problems and return the full corrected JSON:\n- "
                                                 + "\n- ".join(errs[:30])}]
    sys.exit(f"{set_id}: still invalid after {MAX_ATTEMPTS} attempts:\n- " + "\n- ".join(errs[:15]))


def to_bits(set_id: str, lines: list[dict], raw: dict, labeled: bool) -> list[dict]:
    by_idx = {ln["line_idx"]: ln for ln in lines}
    order = [ln["line_idx"] for ln in lines]
    pos = {li: i for i, li in enumerate(order)}
    starts = [b["start"] for b in raw["bits"]]
    bit_of_start = {s: i for i, s in enumerate(starts)}
    # Opener/closer are the first/last bits with a premise; hellos and goodnights are segues.
    content = [i for i, b in enumerate(raw["bits"]) if b["type"] != "segue"] or list(range(len(starts)))
    out, thread = [], -1
    for i, b in enumerate(raw["bits"]):
        lo = pos[b["start"]]
        hi = pos[starts[i + 1]] if i + 1 < len(starts) else len(order)
        members = [by_idx[li] for li in order[lo:hi]]
        levels = [m["label"] for m in members]
        continues = bool(b.get("continues")) and i > 0
        thread += not continues
        out.append({
            "set_id": set_id,
            "bit_idx": i,
            "thread_idx": thread,
            "continues": continues,
            "start_line": order[lo],
            "end_line": order[hi - 1],
            "n_lines": hi - lo,
            "words": sum(m["words"] for m in members),
            "position": ("opener" if i == content[0] else "closer" if i == content[-1]
                         else "edge" if i < content[0] or i > content[-1] else "body"),
            "type": b["type"],
            "mode": b["mode"],
            "techniques": b.get("techniques", []),
            "premise": b["premise"].strip(),
            "hook": (b.get("hook") or "").strip() or None,
            "callback_to": bit_of_start[b["callback_to"]] if b.get("callback_to") is not None else None,
            "laugh_lines": sum(lv >= 2 for lv in levels) if labeled else None,
            "max_level": max(levels) if labeled else None,
            "laugh_rate": round(sum(lv >= 2 for lv in levels) / len(levels), 3) if labeled else None,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("set_ids", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true", help="ignore cached model output")
    args = ap.parse_args()
    manifest = json.loads((PROCESSED / "datasets.json").read_text(encoding="utf-8"))
    targets = manifest["laugh_labeled"] + manifest["structure_only"] if args.all else args.set_ids
    if not targets:
        sys.exit("give set_ids or --all")

    df, n_sets = corpus_df()
    for set_id in targets:
        dataset, lines = load_set_lines(set_id)
        raw = annotate(set_id, lines, args.force, render_hints(candidates(lines, df, n_sets)))
        bits = to_bits(set_id, lines, raw, labeled=dataset == "laugh_labeled")
        cbs = sum(b["callback_to"] is not None for b in bits)
        sizes = sorted(b["n_lines"] for b in bits)
        print(f"  {len(bits)} bits, median {sizes[len(sizes) // 2]} lines, {cbs} callbacks, "
              f"types={ {t: sum(b['type'] == t for b in bits) for t in TYPES} }")

    # Rebuild each dataset's bits.jsonl from every cached set, so partial runs accumulate.
    for dataset in ("laugh_labeled", "structure_only"):
        rows = []
        for set_id in manifest[dataset]:
            cache = INTERIM / f"{set_id}.bits.raw.json"
            if cache.exists():
                _, lines = load_set_lines(set_id)
                rows += to_bits(set_id, lines, json.loads(cache.read_text(encoding="utf-8")),
                                labeled=dataset == "laugh_labeled")
        with open(PROCESSED / dataset / "bits.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{dataset}: {len(rows)} bits from {len({r['set_id'] for r in rows})} sets")


if __name__ == "__main__":
    main()
