"""DPO of a writer (D-123): TRL DPOTrainer on a LoRA adapter. Run by explorer/dpo_B.sbatch, dpo_C.sbatch and smoke.sbatch.

    python -m explorer.train_dpo --data data/dpo/train/dpo_C.jsonl --out data/dpo/train/runs/dpo_C [--base <model or merged dir>]

The reference model is the same network with the adapter switched off (`ref_model=None` with a PEFT config), and its
log-probabilities are computed once before training (`precompute_ref_log_probs`), so only one copy of the 4B model is ever in
memory. Checkpoints every --save-steps; the same command resumes. Stops on a NaN loss or gradient norm (exit code 3).
"""
import argparse
import sys

from explorer import train_common as tcm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=tcm.DEFAULT_BASE, help="B: the merged SFT model's folder; C: the base model")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1536)
    ap.add_argument("--max-steps", type=int, default=-1, help="the smoke test sets 10")
    ap.add_argument("--max-hours", type=float, default=7.5)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--dtype", default="fp16", choices=("fp16", "fp32"), help="base weights; fp32 if fp16 stops on a NaN")
    a = ap.parse_args()
    from datasets import Dataset
    from peft import LoraConfig
    from trl import DPOConfig, DPOTrainer
    print(tcm.versions(), flush=True)
    model, tok = tcm.load_model(a.base, a.dtype)
    rows = tcm.dpo_rows(tcm.read_jsonl(a.data))
    tcm.check_lengths(tok, rows, ("chosen", "rejected"), a.max_length)
    data = Dataset.from_list(rows)
    print(f"[data] {len(data)} pairs", flush=True)
    cfg = DPOConfig(**tcm.config_kwargs(DPOConfig, dict(
        output_dir=a.out, learning_rate=a.lr, beta=a.beta, num_train_epochs=a.epochs, max_steps=a.max_steps,
        per_device_train_batch_size=1, gradient_accumulation_steps=a.accum, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, fp16=True, logging_steps=1, save_steps=a.save_steps,
        save_total_limit=2, report_to="none", seed=0, max_grad_norm=1.0, warmup_ratio=0.1, lr_scheduler_type="cosine",
        # TRL 0.24's max_prompt_length defaults to 512 and cuts the prompt's start (the system text); check_lengths has
        # already made sure no row needs cutting, so only max_length bounds a row.
        max_length=a.max_length, max_prompt_length=None, max_completion_length=None, precompute_ref_log_probs=True, loss_type="sigmoid", dataloader_num_workers=0)))
    guard = tcm.callbacks(a.max_hours)
    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=data, processing_class=tok,
                         peft_config=LoraConfig(**tcm.LORA), callbacks=[guard])
    tcm.check_trainable_fp32(trainer.model)
    row = trainer.train_dataset[0]
    for key in ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids"):
        if key in row:
            tcm.inspect_tokens(tok, row[key], key)
    trainer.train(resume_from_checkpoint=tcm.last_checkpoint(a.out))
    tcm.report_peak()
    trainer.save_model(a.out)
    if guard.nan:
        sys.exit(3)


if __name__ == "__main__":
    main()
