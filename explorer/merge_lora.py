"""Merges a LoRA adapter into its base model and saves fp16 weights (merge_and_unload). The base is the original model or, for
writer B's DPO, the merged SFT model. Runs on the CPU (explorer/merge_gguf.sbatch) or, with --device cuda, on the GPU node.

    python -m explorer.merge_lora --adapter data/dpo/train/runs/sft_B --out data/dpo/train/runs/merged_sft_B [--base ...]
"""
import argparse

from explorer import train_common as tcm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=tcm.DEFAULT_BASE)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    a = ap.parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.float16, device_map="cuda" if a.device == "cuda" else None)
    merged = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    merged.save_pretrained(a.out, safe_serialization=True)
    # The base model's tokenizer, also when --base is a merged folder (B's DPO): merging never changes it (tcm.tokenizer_source).
    AutoTokenizer.from_pretrained(tcm.tokenizer_source(a.base)).save_pretrained(a.out)
    print(f"merged {a.adapter} into {a.base} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
