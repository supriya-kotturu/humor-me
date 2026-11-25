"""Exp 24 judge on Explorer: Qwen2.5-7B-Instruct ranks the bits of prepared prompts, from a bundle the laptop wrote.

Usage (on an Explorer GPU node, from the repo root; see explorer/README.md "Exp 24 judge"):
    python explorer/judge_generators.py --input data/model_comparison/judge/calls.jsonl \\
        --out data/model_comparison/judge/replies.jsonl --dtype half
    python explorer/judge_generators.py ... --limit 3        # smoke test: the first 3 calls

The input is one JSON line per call: {id, system, user, letters}. `user` holds the few-shot examples (the user's ranked
prompts, bits under blind letters, no model names) and the new prompt; `letters` lists the letters to rank. Nothing in
the bundle says which model wrote which bit, and no key file is ever uploaded. Our own generated jokes only.

Backends: `transformers` (default, Qwen2.5-7B-Instruct in float16, the fallback that is known to work here) and `ollama`
(`--backend ollama --model qwen3.8:27b --url http://127.0.0.1:11434`, a server started by explorer/judge_ollama.sbatch).
Ollama silently truncates a prompt to its context window, so every call is guarded: the prompt's size is estimated (chars /
3.5) and refused above `--context`, the request itself asks for that context (`num_ctx`), and the reply's `prompt_eval_count` must
be at least half the estimate or the call counts as truncated. Neither is ever written as an answer. A per-call nonce at the
start of the system message stops Ollama reusing a cached prefix, which would make `prompt_eval_count` look like truncation.

Output, one line per call answered: {id, ranking} with the letters best first. Resumable like label_modes.py: replies
are appended in checkpointed chunks and a rerun skips the ids already there, on any node. Greedy decoding. A reply that
is not a ranking is asked once more with a nudge; if that fails too the call is NOT written as an answer (it goes to
`<out>.failed.jsonl` for a look) and is tried again on the next run: a parse failure is infrastructure, never the judge's
answer. The prompts are long (about 12 to 20k tokens), so the batch size is 1, attention is SDPA (no flash-attn on V100)
and an out-of-memory call is split like label_modes does, then skipped (and retried next run) if it still does not fit.
"""
from __future__ import annotations  # cluster venv is Python 3.9: keeps `X | None` hints unevaluated

import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.request
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from explorer.label_modes import chunks, length_order, split_on_oom  # noqa: E402

NUDGE = 'Reply with only this JSON and nothing else: {"ranking": "<the letters, best first>"}'
JSON_OBJ = re.compile(r"\{.*?\}", re.S)
RANKING_KEY = re.compile(r'"?ranking"?\s*[:=]\s*"?([A-Za-z][A-Za-z ,>\-]*)')
MAX_NEW_TOKENS = 40
CHARS_PER_TOKEN = 3.5  # a conservative estimate for English: a real prompt has fewer tokens than chars / 3.5
SHORTFALL = 0.5  # Ollama must report at least this share of the estimated prompt tokens, or it truncated
THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_ranking(text: str, letters: list[str]) -> str | None:
    """A ranking of exactly `letters` from the model's reply, or None. Tolerant, since transformers has no JSON mode: a
    JSON object with a "ranking" field, a stray `ranking: BAC`, or bare letters ("B > A > C") are all accepted; letters
    are upper-cased, separators dropped, and every letter must appear once."""
    text = THINK.sub("", text)
    text = text.split("</think>")[-1]  # a thinking model's reasoning, closed or cut off, is not the answer
    candidates = []
    for m in JSON_OBJ.finditer(text):
        try:
            value = json.loads(m.group(0)).get("ranking")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(value, str):
            candidates.append(value)
    if m := RANKING_KEY.search(text):
        candidates.append(m.group(1))
    candidates.append(text.strip())  # the whole reply, for "BAC" or "B > A > C"
    want = sorted(letters)
    for c in candidates:
        got = [ch for ch in c.upper() if ch.isalpha()]
        if sorted(got) == want:
            return "".join(got)
    return None


def load_done(path: pathlib.Path) -> set[str]:
    """Ids already answered. Drops a torn last line (a job killed mid-write) so appending stays valid JSONL."""
    if not path.exists():
        return set()
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
    return {r["id"] for r in good}


def conv(call: dict, nudge: str | None = None) -> list[dict]:
    msgs = [{"role": "system", "content": call["system"]}, {"role": "user", "content": call["user"]}]
    return msgs if nudge is None else msgs + [{"role": "user", "content": nudge}]


def answer_calls(calls: list[dict], generate, out: pathlib.Path, chunk: int = 8, log=print) -> dict:
    """Ask `generate(convs) -> replies` for every call not in `out`, a chunk at a time, appending what parsed. Returns
    counts. The failed ones are written to `<out>.failed.jsonl` and left for the next run."""
    done = load_done(out)
    todo = [c for c in calls if c["id"] not in done]
    log(f"{len(done)} already answered; {len(todo)} to go")
    answered = failed = 0
    failed_path = out.with_name(out.name + ".failed.jsonl")
    for batch in chunks(todo, chunk):
        replies = generate([conv(c) for c in batch])
        rankings = [parse_ranking(r, c["letters"]) for r, c in zip(replies, batch)]
        retry = [i for i, r in enumerate(rankings) if r is None]
        if retry:  # once more, with a nudge: the same prompt would give the same greedy reply
            again = generate([conv(batch[i], NUDGE) for i in retry])
            for i, text in zip(retry, again):
                rankings[i] = parse_ranking(text, batch[i]["letters"])
                replies[i] = text if rankings[i] is None else replies[i]
        with open(out, "a", encoding="utf-8") as f:
            for c, r in zip(batch, rankings):
                if r is not None:
                    f.write(json.dumps({"id": c["id"], "ranking": r}) + "\n")
            f.flush()
            os.fsync(f.fileno())  # the checkpoint: a chunk on disk is never redone
        bad = [(c, t) for c, r, t in zip(batch, rankings, replies) if r is None]
        if bad:
            with open(failed_path, "a", encoding="utf-8") as f:
                for c, t in bad:
                    f.write(json.dumps({"id": c["id"], "reason": "not a ranking", "raw": t[:200]}) + "\n")
        answered += len(batch) - len(bad)
        failed += len(bad)
        log(f"{len(done) + answered}/{len(done) + len(todo)} answered, {failed} not a ranking (retried next run)")
    return {"answered": answered, "failed": failed, "already": len(done)}


def estimate_tokens(convs: list[dict]) -> int:
    return int(sum(len(m["content"]) for m in convs) / CHARS_PER_TOKEN)


def ollama_reply(post, model: str, context: int, conv_: list[dict]) -> str:
    """One call to Ollama's native /api/chat, guarded against silent truncation. Returns the reply text, or a bracketed
    reason (which never parses as a ranking, so the call is listed as failed and retried next run) when the prompt is too
    big for `context` or Ollama read far fewer tokens than we sent."""
    est = estimate_tokens(conv_)
    if est > context:
        return f"[refused: the prompt is about {est} tokens, over the {context}-token context]"
    msgs = [dict(m) for m in conv_]
    msgs[0]["content"] += f"\n\n(request {uuid.uuid4().hex[:8]})"  # unique start: no cached prefix, so prompt_eval_count is the whole prompt
    resp = post({"model": model, "messages": msgs, "stream": False, "think": False,
                 "options": {"temperature": 0, "num_ctx": context, "num_predict": 200}})
    read = int(resp.get("prompt_eval_count") or 0)
    if read < SHORTFALL * est:
        return f"[truncated or unverifiable: Ollama read {read} tokens of about {est}]"
    return (resp.get("message") or {}).get("content", "")


def http_post(url: str):
    def post(payload: dict) -> dict:
        req = urllib.request.Request(url.rstrip("/") + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=3600) as r:
            return json.loads(r.read())
    return post


def make_ollama_generator(args, post=None):
    post = post or http_post(args.url)
    return lambda convs: [ollama_reply(post, args.model, args.context, c) for c in convs]


def make_generator(args):
    """generate(convs) -> reply texts, in input order. Heavy imports happen here so --help and the resume check run
    without a GPU. Greedy; SDPA attention; one prompt per forward pass (they are 12 to 20k tokens)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"half": torch.float16, "bfloat16": torch.bfloat16}.get(args.dtype)
    if dtype is None:  # auto: bfloat16 only where the GPU supports it (not V100/P100)
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
    torch.manual_seed(0)

    def is_oom(e):
        if isinstance(e, torch.OutOfMemoryError):
            torch.cuda.empty_cache()
            return True
        return False

    def generate(convs):
        texts = [tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in convs]

        def run(idx):
            enc = tok([texts[j] for j in idx], return_tensors="pt", padding=True).to("cuda")
            if enc["input_ids"].shape[1] > args.max_len:
                return [""] * len(idx)  # too long for the context we budgeted: no answer, listed in the failed file
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            return tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

        replies = {}
        order = length_order(texts)
        for i in range(0, len(order), args.batch_size):
            replies |= split_on_oom(run, order[i:i + args.batch_size], is_oom)
        return [replies[j] for j in range(len(texts))]

    return generate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="calls.jsonl from `compare_generators judge --export-bundle`")
    ap.add_argument("--out", default="data/model_comparison/judge/replies.jsonl")
    ap.add_argument("--backend", default="transformers", choices=["transformers", "ollama"])
    ap.add_argument("--url", default="http://127.0.0.1:11434", help="ollama backend: the server's address")
    ap.add_argument("--context", type=int, default=32768, help="ollama backend: the context to ask for and the longest prompt to send")
    ap.add_argument("--model", default=None, help="default Qwen/Qwen2.5-7B-Instruct (transformers) or qwen3.8:27b (ollama)")
    ap.add_argument("--dtype", default="auto", help="auto | half | bfloat16 (use half on V100)")
    ap.add_argument("--batch-size", type=int, default=1, help="prompts per forward pass (they are very long)")
    ap.add_argument("--chunk", type=int, default=8, help="calls per checkpoint")
    ap.add_argument("--max-len", type=int, default=30000, help="longest prompt in tokens (Qwen2.5 supports 32k)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: only the first N calls")
    args = ap.parse_args()
    args.model = args.model or ("qwen3.8:27b" if args.backend == "ollama" else "Qwen/Qwen2.5-7B-Instruct")
    calls = [json.loads(line) for line in open(args.input, encoding="utf-8")]
    if args.limit:
        calls = calls[:args.limit]
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not [c for c in calls if c["id"] not in load_done(out)]:
        print("nothing to do")
        return
    generate = make_ollama_generator(args) if args.backend == "ollama" else make_generator(args)
    t0 = time.time()
    counts = answer_calls(calls, generate, out, args.chunk)
    print(f"{counts} in {time.time() - t0:.0f} s; replies in {out}", flush=True)
    if counts["failed"]:
        sys.exit(f"{counts['failed']} calls were not rankings: see {out}.failed.jsonl, and submit the job again to retry them")


if __name__ == "__main__":
    main()
