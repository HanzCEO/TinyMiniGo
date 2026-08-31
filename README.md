# TinyMiniGo

A minimal, from-scratch inference engine for **MiniCPM5**, written in pure Rust.

MiniCPM5 uses the standard `LlamaForCausalLM` decoder architecture — RMSNorm,
grouped-query attention (GQA) with RoPE, SwiGLU MLPs — so this engine implements
exactly that, with all tensor math hand-rolled on plain `f32` slices (no ML
framework, no BLAS). Helper crates are used only for the plumbing: `safetensors`
(file parsing), `tokenizers` (tokenizer.json), `minijinja` + `minijinja-contrib`
(HF chat templates with Python string-method emulation), and `clap` (CLI).

Correctness is validated against the tiny debug model
[tiny-random/minicpm5](https://huggingface.co/tiny-random/minicpm5): this
engine's logits and greedy generations match `transformers` reference output
numerically (see `tests/numerical.rs` and `scripts/`).

## Build

```bash
cargo build --release
```

## Get the debug model

```bash
huggingface-cli download tiny-random/minicpm5 --local-dir /tmp/tinyminicpm5
# or manually: config.json, model.safetensors, tokenizer.json,
#              tokenizer_config.json, chat_template.jinja
```

## Usage

```bash
./target/release/tinyminigo -m model.safetensors --template chat_template.jinja "What is the capital of France?"
```

Options:

| Flag | Default | Description |
|---|---|---|
| `-m, --model <FILE>` | (required) | Model weights (safetensors) |
| `--template <FILE>` | (required) | Jinja chat template |
| `--tokenizer <FILE>` | `<model_dir>/tokenizer.json` | Tokenizer file |
| `--max-tokens <N>` | 256 | Max new tokens |
| `--temperature <T>` | 0 (greedy) | Sampling temperature |
| `--top-k <K>` | 0 (off) | Top-k sampling |
| `--top-p <P>` | 1.0 (off) | Nucleus sampling |
| `--repetition-penalty <R>` | 1.0 (off) | CTRL-style repetition penalty |
| `--seed <SEED>` | 42 | Sampling seed |

## Architecture notes

MiniCPM5 = Llama-style decoder:

- **RMSNorm** (`rms_norm_eps` from config, applied pre-attention and pre-MLP, plus final norm)
- **GQA attention**: `num_attention_heads` query heads share `num_key_value_heads` KV heads; RoPE uses the HF `rotate_half` (half-split) layout with `rope_theta` (may live in `rope_parameters.rope_theta`)
- **SwiGLU MLP**: `down(silu(gate(x)) * up(x))`
- **Weights**: `model.embed_tokens.weight`, per-layer `self_attn.{q,k,v,o}_proj.weight`, `mlp.{gate,up,down}_proj.weight`, `{input,post_attention}_layernorm.weight`, `model.norm.weight`, `lm_head.weight` (falls back to tied embeddings when absent). HF `nn.Linear` convention `y = xW^T` with weights stored `[out, in]`.
- **KV cache**: per layer, per timestep; incremental decode with causal masking

All math is computed in `f32` (weights are converted from BF16/F16 at load).

## Verification

```bash
cargo test            # unit tests + numerical cross-checks (needs debug model in /tmp/tinyminicpm5)
```

The numerical tests compare first-step logits (top-5 ids, magnitudes), layer-0
attention output, and a 40-step greedy generation against `transformers`
reference values captured by `scripts/hf_reference.py`, `scripts/hf_single_token.py`,
and `scripts/hf_layer_debug.py`.
