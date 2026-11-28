"""Shared by train_sft.py and train_dpo.py: data, model, LoRA, the stop-on-NaN and time-limit callbacks, version printing."""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pathlib
import sys
import time

from explorer.chatml import IM_END, render_hf, render_ollama

LORA = dict(r=16, lora_alpha=32, lora_dropout=0.05, target_modules="all-linear", bias="none", task_type="CAUSAL_LM")
DEFAULT_BASE = "Qwen/Qwen3-4B-Instruct-2507"  # D-124
RUNS_DIR = "data/dpo/train/runs"
MARKER = "humor_checkpoint.json"  # written beside every checkpoint this job saves; resume refuses a folder without it
NAN_STEPS = 5  # consecutive skipped fp16 steps (non-finite grad norm, finite loss) before a run is called unstable
# Contractions, digits, accents, runs of spaces and ChatML tokens: where a changed pre-tokenizer regex would split differently.
TOKEN_SAMPLE = ("<|im_start|>user\nI'm 37, can't stop: it's 3:45 p.m. and we'll see -- café mañana 12345.67!\n\n"
                "  two  spaces\tand a tab<|im_end|>\n<|im_start|>assistant\n{\"premise\": \"x\", \"jokes\": [\"A.\"]}")


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    return [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sft_rows(rows: list[dict]) -> list[dict]:
    """{"prompt": Ollama-rendered text, "completion": the raw reply}. TRL adds the end token to the completion."""
    return [{"prompt": render_ollama(r["prompt"]), "completion": r["completion"]} for r in rows]


def dpo_rows(rows: list[dict]) -> list[dict]:
    return [{"prompt": render_ollama(r["prompt"]), "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]


def longest(tok, rows: list[dict], replies: tuple[str, ...]) -> int:
    """Tokens of the longest prompt + reply + end token among `rows` (rendered rows: prompt text and reply columns)."""
    n = lambda text: len(tok(text, add_special_tokens=False)["input_ids"])  # noqa: E731
    return max((n(r["prompt"]) + max(n(r[k]) for k in replies) + 1 for r in rows), default=0)


def check_lengths(tok, rows: list[dict], replies: tuple[str, ...], max_length: int) -> None:
    """Refuses to train when any row is longer than `max_length`: TRL would truncate it silently (the system text or the
    reply's end), and a writer trained on cut prompts or replies with no end token is not the one the show serves."""
    top = longest(tok, rows, replies)
    print(f"[data] longest row {top} tokens (max_length {max_length})", flush=True)
    if top > max_length:
        raise SystemExit(f"a row is {top} tokens, over --max-length {max_length}: raise --max-length")


def versions() -> str:
    import accelerate
    import peft
    import torch
    import transformers
    import trl
    return (f"torch {torch.__version__} cuda={torch.cuda.is_available()} transformers {transformers.__version__} trl {trl.__version__} "
            f"peft {peft.__version__} accelerate {accelerate.__version__} python {sys.version.split()[0]}")


def config_kwargs(cls, wanted: dict) -> dict:
    """Only the fields this TRL version's config class has: argument names moved between releases (max_prompt_length, ...), and a
    renamed one must not stop an 8 hour job on its first line. The dropped names are printed."""
    fields = {f.name for f in dataclasses.fields(cls)}
    dropped = sorted(set(wanted) - fields)
    if dropped:
        print(f"[config] {cls.__name__} has no field {dropped}: left out", flush=True)
    return {k: v for k, v in wanted.items() if k in fields}


def tokenizer_source(base: str) -> str:
    """Where the tokenizer is loaded from: the base model id, never a merged model's folder. Merging changes weights only, and
    transformers 4.57.x warns about a "Mistral regex" for a local folder whose config.json a newer 4.57 release wrote, Qwen
    included (its early return for non-Mistral models covers only configs from older releases). The suggested
    fix_mistral_regex=True would swap Qwen's pre-tokenizer for Mistral's and change how contractions split, so it is never set;
    the smoke job checks the ids."""
    return DEFAULT_BASE if os.path.isdir(base) else base


def load_model(base: str, dtype: str = "fp16"):
    """fp16 weights on the V100 (no bfloat16), SDPA attention (no flash-attn on Volta). PEFT keeps the LoRA weights in fp32.
    `fp32` is the fallback when fp16 keeps overflowing (NAN_STEPS skipped steps in a row): plain fp32, no autocast and no loss
    scaler (`amp` is False); 16 GB of weights still fit the 32 GB card, but steps are several times slower.
    The output layer's logits are cast to fp32 in a forward hook, so the log-softmax over 151k entries never runs in fp16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    src = tokenizer_source(base)
    tok = AutoTokenizer.from_pretrained(src)
    end_token_check(tok)
    print(f"[tokenizer] from {src}; eos {tok.eos_token!r}, bos {tok.bos_token!r} (Qwen adds no BOS, so TRL's "
          f"'Updated tokens: bos_token_id None' is expected)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype={"fp16": torch.float16, "fp32": torch.float32}[dtype],
                                                 attn_implementation="sdpa")
    model.get_output_embeddings().register_forward_hook(lambda _m, _i, out: out.float())
    return model, tok


def amp(dtype: str) -> bool:
    """fp16 mixed precision (autocast plus the loss scaler) only on fp16 weights; the fp32 fallback trains in plain fp32."""
    return dtype == "fp16"


def end_token_check(tok) -> None:
    """TRL appends tok.eos_token to every reply; it must be <|im_end|>, the token Ollama's template closes a turn with and the
    GGUF converter records as the end token, or the trained writer learns to stop on a token the show never stops on."""
    if tok.eos_token != IM_END:
        raise SystemExit(f"tokenizer eos_token is {tok.eos_token!r}, not {IM_END!r}")


def same_tokens(tok, ref, label: str, text: str = TOKEN_SAMPLE) -> None:
    """`tok` must give `ref`'s ids for TOKEN_SAMPLE (smoke job: the merged folder's and the training tokenizer against the base's)."""
    a, b = list(tok(text, add_special_tokens=False)["input_ids"]), list(ref(text, add_special_tokens=False)["input_ids"])
    if a != b:
        raise SystemExit(f"[tokenizer] {label} tokenizes the sample differently from the base: {a[:20]} vs {b[:20]}")
    print(f"[tokenizer] {label} == base tokenizer ({len(a)} ids on the sample)", flush=True)


def check_trainable_fp32(model) -> None:
    import torch
    bad = [n for n, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if bad:
        raise SystemExit(f"trainable weights must be fp32 under fp16 mixed precision, found {bad[:3]}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n / 1e6:.1f}M trainable parameters, all fp32", flush=True)


def finite(v) -> bool:
    return not isinstance(v, (int, float)) or math.isfinite(v)


class NanRule:
    """When a NaN or infinite value stops a run. Under fp16 the loss scaler starts at 65536 and halves on every step whose
    gradients overflow; that step is skipped (no weight changes, the learning-rate schedule does not move) and Hugging Face
    logs its grad norm as nan or inf with a finite loss. A few at the start, and after a resume, are normal, so only `k` in a
    row stop the run. A non-finite loss stops it at once: the forward pass broke, and no scaling fixes that. In plain fp32
    (k=1) there is no scaler, a NaN gradient would reach the weights, so the first one stops."""

    def __init__(self, k: int = NAN_STEPS):
        self.k, self.run = k, 0

    def update(self, loss, grad_norm) -> tuple[bool, str | None]:
        """(stop, message) for one logged step."""
        if not finite(loss):
            return True, f"loss is {loss}"
        if grad_norm is None:
            return False, None
        if finite(grad_norm):
            self.run = 0
            return False, None
        self.run += 1
        if self.run >= self.k:
            return True, f"grad_norm is {grad_norm} for {self.run} logged steps in a row"
        return False, f"grad_norm is {grad_norm} with a finite loss: the fp16 loss scaler skipped this step ({self.run} of {self.k} allowed in a row)"


def unstable_advice(dtype: str) -> str:
    if amp(dtype):
        return "[stop] fp16 unstable, rerun with DTYPE=fp32 (plain fp32: no loss scaler, slower, about 20 GB)"
    return "[stop] NaN in plain fp32, so not an fp16 overflow: lower LR= or check the data"


def run_id() -> str:
    return os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"


def write_marker(ckpt_dir: str | os.PathLike, rid: str) -> None:
    pathlib.Path(ckpt_dir, MARKER).write_text(json.dumps({"written_by": "humor-me explorer", "run_id": rid}), encoding="utf-8")


def own_checkpoint(path: str | os.PathLike, runs: str | os.PathLike = RUNS_DIR) -> bool:
    """True only for a folder inside `runs` (symlinks and `..` resolved) holding this job's marker with a run id."""
    p, root = pathlib.Path(path).resolve(), pathlib.Path(runs).resolve()
    if root != p and root not in p.parents:
        return False
    try:
        meta = json.loads((p / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return isinstance(meta, dict) and bool(meta.get("run_id")) and meta.get("written_by") == "humor-me explorer"


def patch_torch_load_guard(modules: dict | None = None) -> list[str]:
    """Replaces every `check_torch_load_is_safe` that transformers 4.57.x's Trainer may call with a no-op, in the modules that
    define or import it. Returns what was patched. Which one the Trainer uses depends on the release, so all are tried."""
    mods = sys.modules if modules is None else modules
    done = []
    for name in ("transformers.trainer", "transformers.utils.import_utils", "transformers.utils"):
        mod = mods.get(name)
        if mod is not None and hasattr(mod, "check_torch_load_is_safe"):
            mod.check_torch_load_is_safe = lambda: None
            done.append(f"{name}.check_torch_load_is_safe")
    return done


def resume_point(ckpt: str | None, runs: str | os.PathLike = RUNS_DIR, modules: dict | None = None) -> str | None:
    """The checkpoint to resume from (None: start fresh). transformers 4.57 refuses torch.load on torch < 2.6 (CVE-2025-32434,
    about unpickling untrusted files) when it restores optimizer.pt, scheduler.pt and rng_state.pth, and the cluster's CUDA 12.3
    driver rules out the cu124+ builds of torch 2.6. Those files were written by this job on this cluster, so the guard is
    bypassed, but only for a checkpoint inside `runs` that carries this job's marker; anything else stops the run. The adapter
    weights are safetensors and never went through the guard."""
    if not ckpt:
        return None
    if not own_checkpoint(ckpt, runs):
        raise SystemExit(f"[resume] refusing {ckpt}: not inside {runs} with a {MARKER} written by this job (delete it to start over)")
    patched = patch_torch_load_guard(modules)
    print(f"[resume] from {ckpt} (own checkpoint, torch.load guard bypassed for optimizer/scheduler/rng state)", flush=True)
    print(f"[resume] guard patched in: {patched}" if patched else "[resume] no check_torch_load_is_safe found to patch; resume may fail on torch < 2.6", flush=True)
    return ckpt


def callbacks(max_hours: float, dtype: str = "fp16"):
    """NanRule (exit code 3 after the trainer returns), a line for every step the loss scaler skips, and save and stop before
    the job's time limit so a resubmit continues from the checkpoint."""
    from transformers import TrainerCallback

    class Guard(TrainerCallback):
        def __init__(self):
            self.start, self.nan, self.rule = time.time(), False, NanRule(NAN_STEPS if amp(dtype) else 1)

        def on_save(self, args, state, control, **kw):
            write_marker(os.path.join(args.output_dir, f"checkpoint-{state.global_step}"), run_id())

        def on_step_end(self, args, state, control, optimizer=None, **kw):
            # Runs before on_log for the same step; accelerate's optimizer wrapper knows whether the scaler skipped it.
            if getattr(optimizer, "step_was_skipped", False):
                scaler = getattr(optimizer, "scaler", None)
                scale = scaler.get_scale() if scaler is not None else "?"
                print(f"[scaler] step {state.global_step} skipped (fp16 gradient overflow); loss scale now {scale}", flush=True)
            if (time.time() - self.start) / 3600 > max_hours:
                print(f"[stop] {max_hours} h used at step {state.global_step}: saving, resubmit to continue", flush=True)
                control.should_save, control.should_training_stop = True, True

        def on_log(self, args, state, control, logs=None, **kw):
            logs = logs or {}
            if "loss" not in logs and "grad_norm" not in logs:
                return  # the end-of-training summary (train_loss, train_runtime)
            stop, msg = self.rule.update(logs.get("loss"), logs.get("grad_norm"))
            if msg:
                print(f"[{'stop' if stop else 'scaler'}] {msg} at step {state.global_step}", flush=True)
            if stop:
                print(unstable_advice(dtype), flush=True)
                self.nan, control.should_training_stop = True, True

    return Guard()


def report_peak() -> None:
    import torch
    if torch.cuda.is_available():
        print(f"[memory] peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB of {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}", flush=True)


def last_checkpoint(out: str) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint
    return get_last_checkpoint(out) if os.path.isdir(out) else None


def inspect_tokens(tok, ids, label: str, ends: bool = True) -> None:
    """Prints the head and tail of a tokenized example, so a smoke run shows no stray BOS, and refuses one that should end a
    reply (`ends`) but does not end in exactly one <|im_end|>: TRL appends it once, and a writer trained without it never stops."""
    ids, end = list(ids), tok.convert_tokens_to_ids(IM_END)
    print(f"[inspect] {label}: {len(ids)} tokens, first {tok.decode(ids[:8])!r}, last {tok.decode(ids[-6:])!r}, "
          f"{ids.count(end)} x {IM_END}", flush=True)
    if ends and (not ids or ids[-1] != end or (len(ids) > 1 and ids[-2] == end)):
        raise SystemExit(f"{label} does not end in exactly one {IM_END}")


def template_check(tok, messages: list[dict]) -> None:
    """The smoke job's template test: the tokenizer's chat template must render the same text as `render_hf` (so `render_ollama`,
    which differs from it only by the blank line after the system text, is the one the show sees)."""
    got = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    want = render_hf(messages)
    if got != want:
        raise SystemExit(f"the tokenizer's chat template differs from explorer/chatml.render_hf:\n{got!r}\n{want!r}")
    print("[template] tokenizer == render_hf; render_ollama adds only the blank line after the system text", flush=True)
    print(f"[template] Ollama form: {render_ollama(messages)!r}", flush=True)
