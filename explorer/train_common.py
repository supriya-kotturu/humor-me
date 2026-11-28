"""Shared by train_sft.py and train_dpo.py: data, model, LoRA, the stop-on-NaN and time-limit callbacks, version printing."""
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


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    return [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sft_rows(rows: list[dict]) -> list[dict]:
    """{"prompt": Ollama-rendered text, "completion": the raw reply}. TRL adds the end token to the completion."""
    return [{"prompt": render_ollama(r["prompt"]), "completion": r["completion"]} for r in rows]


def dpo_rows(rows: list[dict]) -> list[dict]:
    return [{"prompt": render_ollama(r["prompt"]), "chosen": r["chosen"], "rejected": r["rejected"]} for r in rows]


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


def load_model(base: str):
    """fp16 weights on the V100 (no bfloat16), SDPA attention (no flash-attn on Volta). PEFT keeps the LoRA weights in fp32."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, attn_implementation="sdpa")
    return model, tok


def check_trainable_fp32(model) -> None:
    import torch
    bad = [n for n, p in model.named_parameters() if p.requires_grad and p.dtype != torch.float32]
    if bad:
        raise SystemExit(f"trainable weights must be fp32 under fp16 mixed precision, found {bad[:3]}")
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n / 1e6:.1f}M trainable parameters, all fp32", flush=True)


def callbacks(max_hours: float):
    """Stop on a NaN or infinite loss or gradient norm (exit code 3 after the trainer returns), and save and stop before the
    job's time limit so a resubmit continues from the checkpoint."""
    from transformers import TrainerCallback

    class Guard(TrainerCallback):
        def __init__(self):
            self.start, self.nan = time.time(), False

        def on_log(self, args, state, control, logs=None, **kw):
            for k in ("loss", "grad_norm"):
                v = (logs or {}).get(k)
                if isinstance(v, (int, float)) and not math.isfinite(v):
                    print(f"[stop] {k} is {v} at step {state.global_step}", flush=True)
                    self.nan, control.should_training_stop = True, True

        def on_step_end(self, args, state, control, **kw):
            if (time.time() - self.start) / 3600 > max_hours:
                print(f"[stop] {max_hours} h used at step {state.global_step}: saving, resubmit to continue", flush=True)
                control.should_save, control.should_training_stop = True, True

    return Guard()


def report_peak() -> None:
    import torch
    if torch.cuda.is_available():
        print(f"[memory] peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB of {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}", flush=True)


def last_checkpoint(out: str) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint
    return get_last_checkpoint(out) if os.path.isdir(out) else None


def inspect_tokens(tok, ids, label: str) -> None:
    """Prints the tail of a tokenized example so a smoke run shows the end token once and no stray BOS."""
    ids = list(ids)
    print(f"[inspect] {label}: {len(ids)} tokens, first {tok.decode(ids[:8])!r}, last {tok.decode(ids[-6:])!r}, "
          f"{ids.count(tok.convert_tokens_to_ids(IM_END))} x {IM_END}", flush=True)


def template_check(tok, messages: list[dict]) -> None:
    """The smoke job's template test: the tokenizer's chat template must render the same text as `render_hf` (so `render_ollama`,
    which differs from it only by the blank line after the system text, is the one the show sees)."""
    got = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    want = render_hf(messages)
    if got != want:
        raise SystemExit(f"the tokenizer's chat template differs from explorer/chatml.render_hf:\n{got!r}\n{want!r}")
    print("[template] tokenizer == render_hf; render_ollama adds only the blank line after the system text", flush=True)
    print(f"[template] Ollama form: {render_ollama(messages)!r}", flush=True)
