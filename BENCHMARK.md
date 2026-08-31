# Benchmarks

Hardware: AMD Ryzen 7 5800H (8C/16T, AVX2+FMA, no AVX-512/AMX), 16 GB RAM, `powersave` governor (untuned).
Command: `scripts/bench.sh 64 "Who are you?"` (MiniCPM5-1B, BF16 safetensors, greedy).
Correctness gate: stdout md5 must stay `a14a3514dc482111095d83baa764a9f7`
(13-token prompt, greedy sampling; identical text output). The HF-reference
numerical tests (`cargo test`) additionally pin greedy generation to
`transformers` output.

## Baseline (scalar f32, single-thread, per-token prefill)

| phase  | throughput | notes |
|--------|-----------:|-------|
| load   | ~10 s      | BF16→f32 conversion of 2.16 GB weights |
| prefill| 1.4 tok/s  | one token at a time through the whole model |
| decode | 1.39 tok/s | 46 tokens in 33.0 s |
| total  | 49 s wall  | 47 tokens generated |
| RSS    | ~4.3 GB    | f32-converted weights |

## Optimization log

| step | prefill tok/s | decode tok/s | notes |
|------|--------------:|-------------:|-------|
| baseline | 1.4 | 1.39 | scalar, single-thread, per-token prefill |
| batched prefill (forward_seq) | 1.8 | 1.29 | weights streamed once per prefill instead of once per token; now compute-bound scalar |
| AVX2+FMA dot + RoPE tables + fused MLP | 10.9 | 6.47 | fixed a broken horizontal-sum in the SIMD reduce; decode now purely bandwidth-bound |
| std threads (gemm all cores, gemv 1/4) | 55.7 | 7.59 | gemv with all 16 cores was SLOWER (5.7) than 4 threads (7.6) — DRAM contention |
| contiguous flat KV cache | ~56 | ~7.6 | no per-token KV allocations; attention reads flat cache rows |
| BF16 weights end-to-end (in-kernel dequant) | 62.2 | 12.28 | gemv/gemm stream half the bytes; RSS 2.57 GB; bit-identical math |
| register-tile prefill GEMM (tinyBLAS-style) | 85-97 (589-tok) | 12.0 | 4-tok × 2-row m256 accumulator tiles; weight row dequantized once per 4 tokens; bit-identical |
| decode scratch buffers | ~same | ~same | zero per-layer allocs (hygiene; decode is DRAM-bound so neutral) |
| **TMG_I8=1** int8 weights (opt-in) | 92-94 (589-tok) | 15.6-18.3 | per-row symmetric int8 + exact i16 madd integer dots + f32 scale epilogue; lm_head stays BF16; ~1.1 GB; 5/6 prompts token-identical, 1 benign divergence |

## Key findings

1. **Decode is DRAM-bound, not compute-bound.** SIMD gave 4.7× on decode only
   because the scalar loop wasn't even saturating bandwidth; after that, adding
   threads made decode *slower* (16 threads: 5.7 tok/s vs 4 threads: 7.6).
   The real decode win was halving weight bytes (BF16 in-kernel dequant).
2. **Prefill is compute-bound.** Batching (weights streamed once) × SIMD × all
   cores compounded multiplicatively: 1.4 → 62 tok/s (~44×).
3. **bf16→f32 is exact** (bit shift), so keeping raw BF16 weights is numerically
   identical to pre-converting — free 2× bandwidth.
4. A broken AVX2 horizontal-sum produced plausible-but-wrong logits; the
   md5-of-output gate caught it immediately. Always keep an end-to-end
   correctness gate when doing SIMD work.

## Remaining ideas (not implemented)

- int4 weight-only quantization (group-wise scales) — predicted another
  ~1.5-1.8× decode if the dequant kernel stays bandwidth-bound; quality
  risk at 1B scale needs the same prompt-battery gate.
- mmap weights to cut load time (load is 3-5 s, not dominant).
- GQA attention parallelism across heads (small gains expected — decode is
  DRAM-bound, not overhead-bound).
- L2-tiled weight layout in consumption order for prefill.
- Runtime CPU governor tuning (`performance`) for stable clocks — numbers here
  are on `powersave` with visible frequency variance.

## Speed modes

- default: BF16 weights, bit-identical to the verified reference output.
- `TMG_I8=1`: int8 weight-only quantization (per-row scales, BF16 lm_head).
  Decode ~+40-55%, ~1.1 GB RSS; 5/6 prompt battery token-identical, 1
  benign divergence (see research report). Not bit-identical — opt in.
- `TMG_THREADS=N`: manual worker-thread override (default: all cores).
  Decode is DRAM-bound: more threads do not help once saturated.
