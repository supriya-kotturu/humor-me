import argparse, json, math, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

p = argparse.ArgumentParser()
p.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct")
p.add_argument("--adapter", default=None)                  # leave out for the untrained baseline
p.add_argument("--data", default="/scratch/tingyu.z/dpo/dpo_test.jsonl")
p.add_argument("--n", type=int, default=-1)               # -1 = all records
p.add_argument("--out", required=True)
p.add_argument("--fp16", action="store_true")
args = p.parse_args()

tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.float16 if args.fp16 else torch.bfloat16, device_map="cuda")
if args.adapter:
    model = PeftModel.from_pretrained(model, args.adapter)
model.eval()
ID_A = tok.encode("A", add_special_tokens=False)[0]
ID_B = tok.encode("B", add_special_tokens=False)[0]

rows = [json.loads(l) for l in open(args.data)]
if args.n > 0:
    rows = rows[:args.n]

correct, said_a, probs = 0, 0, []
for i, r in enumerate(rows):
    msgs = [{"role": "user", "content": r["prompt"]}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        z = model(**ids).logits[0, -1]                       # scores for the first answer token
    p_a = torch.sigmoid((z[ID_A] - z[ID_B]).float()).item()  # P(A) when only A or B is allowed
    pick = "A" if p_a > 0.5 else "B"
    correct += pick == r["chosen"]
    said_a += pick == "A"
    probs.append(p_a)
    if (i + 1) % 1000 == 0:
        print(f"{i+1}/{len(rows)}  accuracy so far {correct/(i+1):.3f}", flush=True)

n = len(rows)
acc = correct / n
half = 1.96 * math.sqrt(acc * (1 - acc) / n)
result = {"adapter": args.adapter, "n": n, "accuracy": acc,
          "ci95": [acc - half, acc + half], "share_answered_A": said_a / n}
print(json.dumps(result, indent=2))
os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({**result, "p_a": probs}, open(args.out, "w"))