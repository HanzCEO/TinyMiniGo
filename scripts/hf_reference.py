"""Reference: run tiny-random/minicpm5 through HF transformers, dump greedy generation + first-step logits."""
import json
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = "/tmp/tinyminicpm5"
PROMPT = "What is the capital of France?"
MAX_NEW = 40

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
model.eval()

enc = tok.apply_chat_template(
    [{"role": "user", "content": PROMPT}],
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
)
ids = enc["input_ids"]
print("prompt ids:", ids[0].tolist(), file=sys.stderr)

with torch.no_grad():
    out = model.generate(
        ids,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        return_dict_in_generate=True,
        output_scores=True,
    )

gen_ids = out.sequences[0][ids.shape[1]:].tolist()
first_logits = out.scores[0][0].tolist()

# top-5 of first step
top5 = torch.tensor(first_logits).topk(5)
result = {
    "prompt_ids": ids[0].tolist(),
    "generated_ids": gen_ids,
    "first_step_top5_ids": top5.indices.tolist(),
    "first_step_top5_vals": [round(v, 6) for v in top5.values.tolist()],
    "first_logits_first8": [round(v, 6) for v in first_logits[:8]],
    "decoded": tok.decode(out.sequences[0][ids.shape[1]:], skip_special_tokens=True),
}
with open("/tmp/hf_reference.json", "w") as f:
    json.dump(result, f, indent=1)
print(json.dumps({k: v for k, v in result.items() if k != "decoded"}, indent=1))
print("decoded:", repr(result["decoded"])[:400], file=sys.stderr)
