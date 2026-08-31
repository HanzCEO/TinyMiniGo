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
- **KV cache**: per layer, one contiguous flat buffer `[pos, n_kv*head_dim]`; incremental decode with causal masking, zero per-token allocations

## Performance

Kernels are hand-rolled AVX2+FMA (runtime-detected, scalar fallback) with std-thread
parallelism over output rows: batched prefill gemm uses all cores, decode gemv uses
core count / 4 (decode is DRAM-bandwidth-bound — more threads measurably hurt).
Weights stay in raw BF16 and are dequantized in-register during the dot products,
halving memory traffic; the conversion is bit-exact.

On a Ryzen 7 5800H (MiniCPM5-1B, greedy, `scripts/bench.sh 64`):

| phase  | before | after | speedup |
|--------|-------:|------:|--------:|
| prefill | 1.4 tok/s | ~62 tok/s | ~44× |
| decode  | 1.39 tok/s | ~12.3 tok/s | ~8.8× |
| total (47 tokens) | 49 s | ~10 s | ~5× |

See `BENCHMARK.md` for the step-by-step optimization log.

## Verification

```bash
cargo test            # unit tests + numerical cross-checks (needs debug model in /tmp/tinyminicpm5)
```

The numerical tests compare first-step logits (top-5 ids, magnitudes), layer-0
attention output, and a 40-step greedy generation against `transformers`
reference values captured by `scripts/hf_reference.py`, `scripts/hf_single_token.py`,
and `scripts/hf_layer_debug.py`.

Weight matrices are stored as raw BF16 (`tensor::WMat`) and dequantized
in-register inside the AVX2 kernels; F16/F32 inputs are converted to f32 at load.
