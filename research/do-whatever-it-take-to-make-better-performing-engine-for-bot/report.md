# Report — Making TinyMiniGo Faster (prefill + decode), MiniCPM5-only, CPU-only

[DONE]

## Scope

TinyMiniGo is a from-scratch Rust CPU inference engine serving exactly one
model: MiniCPM5-1B (Llama-style: 24L, h=1536, 16Q/2KV heads, head_dim 128,
SwiGLU 4608, vocab 130560, BF16). Hardware: Ryzen 7 5800H (8C/16T Zen3,
AVX2+FMA, no AVX-512/AMX, 16 MiB L3, DDR4, powersave governor). The
reference command is the benchmark throughout.

Quality bar: the default engine path must stay **bit-identical** (md5
`e1c13ca7bedaafdd2b927c9d1fdbb41b` for the reference command, 256 greedy
tokens), validated by 26 tests including HF-transformers-pinned logits and
40-step greedy generation. Optional speed modes must be empirically quality-
checked (token-parity batteries) before being kept.

## Starting point (re-measured this session)

prefill 67 tok/s (18-tok prompt), decode 11.87 tok/s, load 4.09 s, wall
21.75 s for 256 tokens.

## Performance model (why the phases are slow)

- Decode is memory-bandwidth-bound: each token must stream every weight once;
  the compute/bandwidth asymmetry (~1 TFLOP/s vs ~50 GB/s) makes bytes/token
  the only real currency [16]. At 11.87 tok/s × 2.16 GB (BF16) we sustain
  ≈26 GB/s effective; the DRAM channel, not threads, is the wall — confirmed
  by our thread sweep (flat 2→16 threads) and by llama.cpp-style engines
  converting byte reductions ~linearly into decode speed [1][16].
- Prefill is compute/bandwidth-mixed: the naive kernel re-streamed each
  weight row once per token; register-blocked tiles that reuse each weight
  row across multiple tokens are the standard fix (tinyBLAS/llamafile: 4×3
  m256 accumulator tiles for 16-register AVX2 [2]; BLIS/MLAS packing and
  micro-kernel theory [3][4]).
- Dequant-to-float kernels are often NOT bandwidth-bound: decode throughput
  follows f = min(1, T_dec·b/(8β)) — the decoder's value rate, not the bit
  width, decides whether you hit the bandwidth roof [19]. We measured this
  directly: naive i8→f32 FMA gemv achieved 0.97× of BF16 (kernel-bound),
  while the integer madd path achieved 1.55× (bandwidth-bound) — exactly the
  predicted split.

## What was implemented and kept

### EXP1 — Register-blocked prefill GEMM (bit-identical, default)
tinyBLAS-style kernel: 4-token × 2-row tile of k-major `__m256` FMA
accumulators, 8-row jobs statically partitioned across threads, weight rows
loaded/dequantized once per 4 tokens [2][3][4]. Result: **589-token prefill
62→85-97 tok/s (~1.4-1.5×)**; short-prompt prefill 67→80-110 tok/s (noisy,
governor-limited). Output bit-identical; all tests green.

### EXP2 — int8 weight-only quantization, opt-in `TMG_I8=1` (lossy, gated)
Symmetric per-row int8 scales quantized at load (w ≈ scale·q); decode gemv
quantizes the activation row once per matmul and uses exact i16×i16→i32
integer dots (`vpmaddubs`-family madd chain [5][6][8]) with an f32 scale
epilogue; prefill tile kernel widens i8→f32 in-register (cvtepi8) with
per-row scale at writeback. lm_head stays BF16 (logit precision; measured
decode-neutral). Result: **decode 11.9→15.6-18.3 tok/s (+35-55% depending on
session noise), wall 21.75→~16 s; model bytes 2.16→~1.1 GB.** Quality: 5/6
prompt battery token-identical, 1 stylistic divergence (equally correct
answer), mean token similarity 0.936-0.950; consistent with W8 being
effectively lossless in the literature [8][20] and with llama.cpp's
experience that remaining top-1 flips concentrate at near-ties [1][7].

### EXP3 — lm_head excluded from quantization (kept, free)
Keeping the head BF16 costs nothing measurable in decode speed (head gemv is
~20% of streamed bytes) and preserves full logit precision for sampling [8].

### EXP6 — Decode allocation pooling (bit-identical, kept for hygiene)
Persistent per-model scratch buffers (x, norms, q/k/v, attn_out, proj,
gate/up/down) + `rms_norm_into`/`matmul_w_into`; zero per-layer heap allocs
in the decode hot path. Decode-neutral within noise (the path is
DRAM-bound, matching [1]'s finding that allocation overhead only matters
below the bandwidth roof), but removes jitter and ~240 allocs/token.

### EXP5 — Thread-count sweep (closed: not a lever)
Decode flat 14.7-18.6 tok/s from 2→16 threads in I8 mode; DRAM-saturated.
Matches [1][16]. `TMG_THREADS` env kept as manual override only.

### EXP4 — Parallel load-time quantization (discarded)
16-thread quantize at load: no improvement (7.7 s vs 6.0-8.5 s serial) —
the cost is paging/IO under memory pressure, not compute. Reverted.

## Final state (reference command)

| config | prefill (T=18) | prefill (T=589) | decode | wall (256 tok) | quality |
|---|---:|---:|---:|---:|---|
| session baseline | 67 | 62-63 | 11.87 | 21.75 s | bit-exact ref |
| default (EXP1+6) | 80-93 | 85-97 | 10.6-11.8* | ~22 s | **bit-identical** |
| TMG_I8=1 (EXP1+2+3+6) | 66-72 | 92-94 | 15.6-16.2* | ~16 s | 5/6 identical, 1 benign divergence |

*Same-session comparisons: BF16 vs I8 measured interleaved shows +41-45%
decode for I8; absolute numbers vary with system load (this box idles at
load 2-5 with several GB of swap in use — all numbers are lower bounds).

## What did not work / lessons

1. Naive dequant-then-FMA int8 is kernel-bound, not bandwidth-bound [19] —
   measure the decoder, don't assume halving bytes halves time.
2. More threads ≠ more decode throughput once DRAM-saturated (flat 2→16).
3. Parallelizing load-time quantization doesn't help under memory pressure.
4. Process discipline: verify binary freshness (a masked build error
   produced a fake "regression" and later a silently-disabled TMG_I8 —
   both caught by md5 gates + load-time fingerprints), and never benchmark
   a noisy box (load 5.8 gave 5.6-11 tok/s readings across all configs).

## Remaining ideas (ranked, not implemented)

1. int4 weight-only (group-128 scales, Q6_K/Q4_K-style [7][17][18]) —
   predicted another ~1.5-1.8× decode if the decoder stays bandwidth-bound
   [19]; quality risk at 1B scale is real [8], needs the same battery gate.
2. mmap weight loading (O(1) load, demand paging) [12] — load is 3-5 s,
   untested here because load is not the dominant cost.
3. Fused attention: parallelize GQA score/softmax across heads ([1] got
   +10.5% and +24% from K/V-once in an overhead-bound engine; ours is
   bandwidth-bound, so gains would be smaller).
4. L2-tiled weight layout in consumption order [16] — our matrices
   (6-40 MB) exceed L2; tiling could cut L1 misses for prefill.
5. CPU governor `performance` for stable clocks (all numbers here are on
   powersave with visible variance).

## Sources

See sources.md [1]-[20]. Key: [1] Overfit measured CPU-LLM levers,
[2] llamafile tinyBLAS (kernel template for EXP1), [3][4] BLIS/MLAS GEMM
theory, [5][6] AVX2 int8 madd chains, [7][8][20] quantization quality
evidence, [16] cflow decode bandwidth model, [17][18] CPU-LLM quant runtimes,
[19] bandwidth-bound decoder law (predicted EXP2a's result).
