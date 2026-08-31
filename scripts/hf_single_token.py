"""Dump HF intermediate activations for a single-token forward (BOS id=0)."""
import json
import torch
from transformers import AutoModelForCausalLM

MODEL_DIR = "/tmp/tinyminicpm5"
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32)
model.eval()

ids = torch.tensor([[0]])
with torch.no_grad():
    out = model(ids, output_hidden_states=True)

hs = out.hidden_states
logits = out.logits[0].float()
top5 = logits[0].topk(5)

def r8(vals):
    return [round(float(v), 8) for v in vals]

d = {
    "embed_first8": r8(hs[0][0, 0, :8]),
    "after_layer0_first8": r8(hs[1][0, 0, :8]),
    "after_layer1_first8": r8(hs[2][0, 0, :8]),
    "final_norm_first8": r8(model.model.norm(hs[2][0].float())[0, :8]),
    "logits_first8": r8(logits[0, :8]),
    "top5_ids": top5.indices.tolist(),
    "top5_vals": r8(top5.values),
}
with open("/tmp/hf_single_token.json", "w") as f:
    json.dump(d, f, indent=1)
print(json.dumps(d, indent=1))
