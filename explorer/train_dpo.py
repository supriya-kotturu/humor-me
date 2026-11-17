import argparse, os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--data_dir", default="/scratch/tingyu.z/dpo")
p.add_argument("--out_dir", default=os.path.join(os.environ["COURSE_DIR"], "runs", "dpo_judge_v1"))
p.add_argument("--train_records", type=int, default=20000)
p.add_argument("--dev_records", type=int, default=2000)
p.add_argument("--max_steps", type=int, default=-1)            # -1 = run the full epoch
p.add_argument("--fp16", action="store_true")
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)

def to_chat(rec):
    # TRL applies the model's chat template when the fields are lists of messages
    return {"prompt":   [{"role": "user",      "content": rec["prompt"]}],
            "chosen":   [{"role": "assistant", "content": rec["chosen"]}],
            "rejected": [{"role": "assistant", "content": rec["rejected"]}]}

def load(name, n):
    ds = load_dataset("json", data_files=os.path.join(args.data_dir, name), split="train")
    ds = ds.select(range(min(n, len(ds))))
    ds = ds.filter(lambda r: len(tok(r["prompt"]).input_ids) <= 448)   # drop very long joke pairs
    return ds.map(to_chat, remove_columns=ds.column_names)

train = load("dpo_train.jsonl", args.train_records)
dev = load("dpo_dev.jsonl", args.dev_records)
print(f"train {len(train)} records, dev {len(dev)} records")

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if args.fp16 else torch.bfloat16)

cfg = DPOConfig(
    output_dir=args.out_dir,
    beta=0.1,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,                  # 8 x 4 = 32 records per update
    num_train_epochs=1,
    max_steps=args.max_steps,
    max_length=512,
    bf16=not args.fp16,
    fp16=args.fp16, 
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    report_to="none",
)

lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                  target_modules="all-linear", task_type="CAUSAL_LM")

trainer = DPOTrainer(model=model, ref_model=None, args=cfg,
                     train_dataset=train, eval_dataset=dev,
                     processing_class=tok, peft_config=lora)

if args.fp16:                                                  
    for prm in trainer.model.parameters():                 
        if prm.requires_grad:                                 
            prm.data = prm.data.float()                         

last = get_last_checkpoint(args.out_dir) if os.path.isdir(args.out_dir) else None
trainer.train(resume_from_checkpoint=last)              # resumes if a checkpoint exists
trainer.save_model(os.path.join(args.out_dir, "final"))