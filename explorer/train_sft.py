"""SFT of writer B (D-123): LoRA on the Haiku-chosen threads, one row per bit. Run by explorer/sft_B.sbatch (and smoke.sbatch).

    python -m explorer.train_sft --data data/dpo/train/sft_B.jsonl --out data/dpo/train/runs/sft_B [--max-steps 10]

Prompt and completion are separate columns, so the loss is on the reply only. Resumes from the last checkpoint in --out.
"""
import argparse
import sys

from explorer import train_common as tcm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dpo/train/sft_B.jsonl")
    ap.add_argument("--out", default="data/dpo/train/runs/sft_B")
    ap.add_argument("--base", default=tcm.DEFAULT_BASE)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--max-steps", type=int, default=-1, help="the smoke test sets 10")
    ap.add_argument("--max-hours", type=float, default=7.5, help="save and stop this long after the start (the job limit is 8 h)")
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "fp32"), help="base weights; fp32 if fp16 stops on a NaN")
    a = ap.parse_args()
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer
    print(tcm.versions(), flush=True)
    model, tok = tcm.load_model(a.base, a.dtype)
    rows = tcm.sft_rows(tcm.read_jsonl(a.data))
    tcm.check_lengths(tok, rows, ("completion",), a.max_length)
    data = Dataset.from_list(rows)
    print(f"[data] {len(data)} rows; prompt tail {data[0]['prompt'][-60:]!r}", flush=True)
    cfg = SFTConfig(**tcm.config_kwargs(SFTConfig, dict(
        output_dir=a.out, learning_rate=a.lr, num_train_epochs=a.epochs, max_steps=a.max_steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=a.accum, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True, logging_steps=1, save_steps=a.save_steps, save_total_limit=2, report_to="none", seed=0, max_grad_norm=1.0,
        warmup_ratio=0.05, lr_scheduler_type="cosine", max_length=a.max_length, completion_only_loss=True,
        dataloader_num_workers=0)))
    guard = tcm.callbacks(a.max_hours)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=data, processing_class=tok, peft_config=LoraConfig(**tcm.LORA),
                         callbacks=[guard])
    tcm.check_trainable_fp32(trainer.model)
    tcm.inspect_tokens(tok, trainer.train_dataset[0]["input_ids"], "first example")
    trainer.train(resume_from_checkpoint=tcm.last_checkpoint(a.out))
    tcm.report_peak()
    trainer.save_model(a.out)
    if guard.nan:
        sys.exit(3)


if __name__ == "__main__":
    main()
