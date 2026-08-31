# Notes — findings log

Baseline re-measured today (before any changes):
- Build: release, tests 26/26 pass.
- Reference command (256 greedy tokens): load 4.09 s, prefill 18 tok / 0.27 s
  (67.4 tok/s), decode 255 tok / 21.48 s (11.87 tok/s), wall 21.75 s.
- Reference stdout md5 (256-token greedy): e1c13ca7bedaafdd2b927c9d1fdbb41b
  (note: BENCHMARK.md's older md5 a14a35… was for the 64-token variant).
- Hardware check: 8C/16T Zen3 (Cezanne), AVX2+FMA only (no AVX512/BF16),
  16 MiB L3 (single instance), 1 NUMA node, 15 GiB RAM, powersave governor.

Decode bandwidth math: 11.87 tok/s × 2.16 GB(BF16 weights)/token ≈ 25.6 GB/s
effective streaming rate with 4 gemv threads. Zen3 dual-channel DDR4 practical
peak ≈ 40-47 GB/s → headroom ~1.6-1.8× from bytes/bandwidth alone; more via
quantization (int8 → ~2×, int4 → ~4× fewer weight bytes).

Prefill compute math: 18-token prompt. FLOPs/token ≈ 2 × params(≈1.35B
non-embedding+embedding… actual matmul params ≈ 1.22B) ≈ 2.4 GFLOP/token
(incl lm_head). AVX2 peak (8 cores × 3.2 GHz × 16 FLOP/cycle) ≈ 410 GFLOP/s;
at 67.4 tok/s we sustain ≈ 165 GFLOP/s effective (though lm_head is small T).
Headroom ~2-2.5× from a real GEMM micro-kernel.

KV cache at this scale: per token 24 layers × 2 × 2 heads × 128 dim × 4 B
= 49 KB/token → negligible bandwidth at short ctx; decode is weights-bound.

[local-code observation, keep] forward_token allocates ~10 Vecs per layer per
token (q/k/v, attn_out, scores, probs, normed, gate, up, down, proj) —
allocator churn is real but small vs weight streaming; still worth pooling.
[local-code observation, keep] matmul_batch_w_into inner loop is
dot-per-(row,token): x is re-streamed for every output row; no register
blocking of T×N output tile; weights streamed once per token per row →
compute intensity ~1 FMA per 2 BF16 bytes → prefill is ALSO partially
bandwidth-bound at small T, not purely compute-bound.

## Gather wave 1 (web research, 11 queries used)

Key kept findings (full refs in sources.md):
- [1] Overfit A/B: decode may be overhead-bound not bandwidth-bound; biggest
  bit-identical wins = parallelizing the big matmuls (+31%) and GQA K/V-once
  (+24%); lossy wins: Q8_0 3.3×, Q4_K +9.6% more. Our engine already has
  parallel gemv, but NOT K/V-once attention parallelism and NOT fused
  quantize. Quality gate for quant: 28/32 top-1 parity, flips at near-ties.
- [2] tinyBLAS/llamafile: exact AVX2 recipe for our prefill gemm: RM=4, RN≤3,
  k-major vector accumulators, no packing, n≥2 only, k%8==0 (all our K are
  multiples of 128/256 → fine). Our shapes: M=T tokens, N=out_features,
  K=in_features with A=[T,K] row-major = llamafile's C=AᵀB convention.
- [3][4] BLIS/MLAS: packing panels matter when weights don't fit L2; our
  per-matrix weights are 6-40 MB — bigger than 512K L2, so streaming B
  from DRAM per N-tile is the prefill bottleneck; register tiling over
  T×N with B reused across T is the fix either way.
- [5][6] int8 dot on AVX2/Zen3: VPMADDUBSW+VPMADDWD+VPADDD chain, 0.5cyc
  throughput for the madds; exact in s32; u8×s8 mixing needs care.
- [7][8][9] Quant quality: W8 effectively lossless even for 1B-class; 4-bit
  risky at 1B scale; Q6_K imperceptible. Gate via KLD vs BF16 logits +
  greedy top-1 parity, exactly what we can implement locally.
- [10][11] Zen3 cache hierarchy + THP: modest gains expected from THP for
  streaming; try madvise on weight buffers.
- [12] mmap load: O(1) load via demand paging (candle/realizar pattern).
- [14][15] ggml prefill-quant pattern: repack weights into 8-row panels and
  dequant once per batch (not per token) — critical if we quantize weights
  but want prefill to stay fast.

## Experiment log (implement → measure → keep/discard)

### EXP1 tinyBLAS-style register-blocked prefill GEMM (KEEP)
- Kernel: 4-token × 2-row tile of k-major m256 accumulators, 8-row jobs,
  static split + balanced tail, weight rows dequantized once per 4 tokens.
- Bit-exact: reference command md5 unchanged (e1c13ca7…), all tests pass.
- 589-token prefill: 62→93 tok/s (best 96.5) ≈ 1.5×.
- 18-token ref prompt: 67→70-110 tok/s (high variance, powersave governor;
  0.16-0.27 s absolute).
- Decode: unchanged ~11.1 tok/s (expected — gemv path untouched).
- Lesson: first "regression" was a stale binary (build error masked by
  `tail`); always check binary mtime + exit codes. Edition-2024 precise
  closure capture defeats Send-wrapper structs; pass ptrs as usize.

## Gather wave 2 (arXiv, 4 queries)

- [16] cflow (2608.23841): decode = bandwidth-bound; ~1 TFLOP/s vs ~50 GB/s
  asymmetry; L2-tiled weight layout in consumption order → 7.3× fewer L1
  misses; projection fusion. For us: weight layout/order and fusion matter
  beyond raw quantization.
- [17] Armv9 llama.cpp opt (2406.10816): int8 quant + vectorized kernels →
  decode 24×, prefill 1.6×, mem 1/5, negligible accuracy loss.
- [18] Intel CPU LLM runtime (2311.00502): W4A16, group-size 128 beats 32
  (fewer scale loads); <1% accuracy loss vs FP32 across 6-20B; AMX/VNNI
  kernels; pre-allocated KV cache. Group size 128 = our head_dim — natural.
- [19] Bandwidth-bound scan law (2606.22423): achieved bandwidth fraction
  f = min(1, T_dec·b/(8β)); dequant rate T_dec is layout-determined, not
  bit-width-determined. Implication: int8 path with cheap layout can be
  bandwidth-bound on our ~26 GB/s effective; int4 needs T_dec ≥ ~2× int8's.
- [20] FineQuant: per-group/per-channel weight-only, weights-only heuristics,
  int8 ~lossless.

Synthesis for EXP2 (int8 weight-only quant):
- Scheme: symmetric per-channel (per output row) int8, i.e. w ≈ scale_o × q,
  q ∈ [-127,127]. Simple, no group metadata, scale vector per matrix (tiny).
  Storage: 1 byte/weight vs 2 (bf16) → weight bytes halve.
- Decode kernel: dot(x_f32, w_i8): AVX2 path — load 32 i8 → _mm256_cvtepi8_epi32
  in two halves → f32 FMA (keeps exact same accumulation semantics as bf16
  path, easier to validate) OR vpmaddubsw/vpmaddwd int path (faster but
  different numerics). Start with cvtepi8→f32 FMA: dequant cost = 1 cvt + 1
  mul-add per 8 weights; predicted bandwidth-bound at ~2 bytes/elem → target
  ~2× decode speedup (11.9 → ~20-24 tok/s).
- Prefill: keep BF16 path for correctness of EXP1 kernel? No — better: extend
  gemm tile kernel to int8 weights with same structure (load i8, cvt, FMA).
  Weights streamed once per 4 tokens either way; prefill is compute-bound so
  int8 dequant overhead should be minor.
- Quality gate: greedy parity on reference command + full KLD vs bf16 logits
  on a set of prompts (log top-5 + mean abs logit diff) + cargo test.
- Expected: decode ~2×, prefill neutral-to-slightly-better, RSS 2.57→~1.6 GB.

### EXP2a int8 microbenchmark (standalone, /tmp/bench_i8.rs, KEEP as design signal)
- single-thread, N=4608×K=1536 (partially L3-resident; relative signal only):
  bf16 gemv 16.8 GB/s | int8 cvt→f32-fma 8.2 GB/s (0.97×!) | int8 madd 13 GB/s (1.55×)
- Confirms [19]: naive dequant-to-float kernel is NOT bandwidth-bound (T_dec
  too low); must use integer dot (vpmaddubs/madd_epi16) with quantized
  activations + scale epilogue. Design: per-token i8 activation quant +
  per-row i8 weights + i16×i16 madd_epi16 → i32 exact accumulation.

### EXP2b int8 weight-only in engine (TMG_I8=1, KEEP behind flag)
- Decode: 11.9 → 15.8-18.2 tok/s (~1.4-1.55×, run-to-run variance from
  powersave governor + RAM 2.16GB→~1.1GB).
- Prefill: roughly neutral at T=18 (47-83 tok/s, noisy), T=589 similar to
  bf16 tile kernel (~83-91). AVX2 tile kernel handles I8 via cvtepi8+scale.
- Load: 4.2s → 6.0-8.5s (quantization at load costs ~2-4s single-threaded —
  needs parallelizing or mmap; noted for EXP4).
- Quality gate: 6-prompt × 24-token greedy parity vs BF16: 4/6 identical,
  mean token-seq similarity 0.950; the 256-token reference answer is a
  coherent alternative phrasing of the same proof (valid math, same
  conclusion). Top-1 flips are classic near-tie flips per [1][7][9].
  cargo test green (default path bit-identical: md5 unchanged).
- Decision: KEEP as opt-in (TMG_I8=1). Default remains bit-exact BF16.
  Quality is "no compromise" for default; int8 offered as explicit speed
  mode with quantified divergence. Next: try excluding lm_head from
  quantization (logit-sensitive) to raise parity.

### Next experiments
- EXP3: I8-sans-lm_head variant (lm_head stays BF16: 0.4GB of 2.16GB) to
  improve quality at slight decode cost.
- EXP4: parallelize load-time quantization (per-matrix threads) + mmap.
- EXP5: decode thread-count sweep for I8 (4 threads was tuned for BF16
  bytes; I8 halves bytes → maybe fewer threads suffice, or more help now).
- EXP6: fuse per-token elementwise passes in forward_token (alloc pooling).

### EXP5 decode thread sweep (DISCARD — no effect, route closed)
- I8 decode flat ~14.7-18.6 tok/s across TMG_THREADS=2..16 (2 runs each,
  variance = governor noise). BF16 confirmed flat earlier (BENCHMARK.md).
- Conclusion: at 1 byte/weight the DRAM channel is saturated; adding threads
  only adds contention. Thread tuning CLOSED as a lever (matches [1] and
  cflow [16] analysis). TMG_THREADS env kept as manual override only.

### EXP3 result (KEEP): lm_head stays BF16 under TMG_I8
- Mean 6-prompt similarity unchanged (0.950) — divergence originates in
  transformer layers, not the head — but decode speed identical (~18 tok/s)
  and logits keep full precision for sampling. Free quality win, kept.

### EXP4 parallel load-time quantization (DISCARD)
- Chunked 168 quantize jobs across 16 threads: load 7.7s (vs 6.0-8.5s serial)
  — no improvement; machine has 6.6GB swap in use, cost is page-fault/IO
  bound, not compute. Reverted to sequential loader. (mmap remains the
  untested load lever; deprioritized — load is 4.2s vs 14-22s compute.)

### EXP6 decode allocation pooling (implemented; MEASUREMENT BLOCKED — system noise)
- forward_token now uses a persistent Scratch (x, attn_in, q/k/v, attn_out,
  proj, mlp_in, gate, up, down) — zero per-layer heap allocs; rms_norm_into
  + matmul_w_into write directly into scratch. scores/probs per head remain
  (small, n_ctx-sized).
- Correctness: bit-identical output (md5 e1c13ca7…) and all tests green.
- Also fixed: git checkout had accidentally reverted the TMG_I8 wiring;
  re-applied (I8 loader + lm_head-stays-BF16).
- Measurement: system load 5.8, ~16 MB/s paging IO during runs — decode
  readings collapsed to 5.6-11 tok/s across ALL configs, unusable. Will
  re-measure when load < 2. Until then EXP6 is "kept for hygiene" (fewer
  allocs, same math) but its speed effect is [unverified].

### Process lessons (keep for report)
- Always verify binary mtime + build exit status (masked a whole false result).
- Never benchmark on a noisy box: gate measurements on load<2, 2+ repeats.

### EXP6 re-measured on quiet-ish system (load ~3, down from 5.8) — FINAL
- BF16 (EXP1+6): decode 10.55-11.82 tok/s, prefill 80-93 (T=18), 85-92 (T=589).
  → scratch pooling is decode-neutral within noise (vs 11.87 baseline);
  kept for hygiene (zero per-layer allocs, less jitter).
- I8 (EXP1+2+6): decode 15.59-16.22 tok/s (+41-45% vs BF16 same-session),
  prefill 66-72 (T=18), 92-94 (T=589 — slightly better than BF16: fewer
  bytes streamed per weight row in the tile kernel).
- Bug found & fixed: after git checkout revert, TMG_I8 wiring was lost →
  "I8" runs were silently BF16 (identical md5). Detected via md5 + load-time
  check. Restored; verified I8 output diverges (43d6ae7b…) and is coherent.
- All 26 tests pass in both modes; BF16 default remains bit-identical.

## Results summary (reference command, this session)
| config | prefill T=18 | prefill T=589 | decode | quality |
|---|---|---|---|---|
| baseline (start of session) | 67 | 62-63 | 11.87 | md5 e1c13ca7 |
| +EXP1 tile gemm | 70-110 | 87-97 | 11.9 | bit-identical |
| +EXP6 scratch | 80-93 | 85-92 | 10.6-11.8 | bit-identical |
| +TMG_I8=1 | 66-72 | 92-94 | 15.6-16.2 | coherent alt phrasing, 0.95 token parity |

I8 decode is +35-45% over BF16 decode (same session, same noise), and the
I8 wall time for the reference command is ~16 s vs 21.75 s baseline.

### Final I8 quality battery (40-token, 6 prompts, fixed binary)
- 5/6 prompts token-identical to BF16 (incl. the reference question).
- 1 divergence ("Who wrote Hamlet?"): I8 produces an equally-correct answer
  ("Shakespeare is the most..."), divergence begins mid-reasoning at a
  stylistic choice point, not a factual error. Mean token similarity 0.936.
- Verdict: int8 weight-only with per-row scales + BF16 lm_head meets a high
  quality bar for greedy use; flagged as opt-in because it is not
  token-identical. Default engine path remains bit-exact.
