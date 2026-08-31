import json
import torch
from transformers import AutoModelForCausalLM

m = AutoModelForCausalLM.from_pretrained('/tmp/tinyminicpm5', dtype=torch.float32)
m.eval()
captured = {}
def hook(name):
    def f(mod, inp, out):
        o = out[0] if isinstance(out, tuple) else out
        captured[name] = o.detach().float()[0, 0, :8].clone()
    return f

layer0 = m.model.layers[0]
layer0.input_layernorm.register_forward_hook(hook('ln1_out'))
layer0.self_attn.q_proj.register_forward_hook(hook('q_out'))
layer0.self_attn.o_proj.register_forward_hook(hook('o_out'))
layer0.post_attention_layernorm.register_forward_hook(hook('ln2_out'))
layer0.mlp.down_proj.register_forward_hook(hook('mlp_out'))
layer0.register_forward_hook(hook('layer0_out'))

with torch.no_grad():
    out = m(torch.tensor([[0]]), output_hidden_states=True)

for k in ['ln1_out','q_out','o_out','ln2_out','mlp_out','layer0_out']:
    print(k, [round(float(v),8) for v in captured[k].tolist()])
