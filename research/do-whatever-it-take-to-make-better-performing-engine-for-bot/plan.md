# Research Plan — Faster TinyMiniGo (prefill + decode) for MiniCPM5 only

## Scope (restate)

TinyMiniGo is a from-scratch, **CPU-only** Rust inference engine that serves
**exactly one model: MiniCPM5-1B** (LlamaForCausalLM-style: 24 layers, hidden
1536, 16 Q heads / 2 KV heads, head_dim 128, SwiGLU 4608, RMSNorm, RoPE θ=5e6,
vocab 130560, BF16 safetensors). Because the model is fixed, the engine may
hard-specialize every kernel to these exact shapes (dims 1536/4608/128, GQA
group 8, vocab 130560) — generalization is explicitly NOT a goal. Quality must
not be compromised: greedy generation must stay token-identical (or numerically
justified) vs the current verified baseline.

**Hard constraints**
- CPU only (AMD Ryzen 7 5800H: 8C/16T Zen3, AVX2+FMA, no AVX-512/BF16 ISA,
  16 MiB L3, single NUMA node, ~15 GB RAM). No GPU, no accelerators.
- Reference command (the benchmark):
  `cargo run -r -- -m ~/Documents/Models/openbmb/MiniCPM5-1B/model-00000-of-00001.safetensors --template ~/Documents/Models/openbmb/MiniCPM5-1B/chat_template.jinja "Why is the root of 2 irrational?"`
- Correctness gates: `cargo test` (26 tests, HF-pinned numerics) + reference
  command md5 `e1c13ca7bedaafdd2b927c9d1fdbb41b` (256 greedy tokens).

**Current baseline (measured today on this machine)**
- load: 4.09 s — decode: 11.87 tok/s — prefill: 67.4 tok/s — 256 tok in 21.75 s.

## Performance model (why each phase is slow)

- **Decode (gemv, memory-bound):** every token streams all ~2.16 GB of BF16
  weights through 4 threads. At 11.87 tok/s that's ≈26 GB/s effective vs
  ~40-50 GB/s practical peak on this box → we are ~50-60% of achievable.
  Levers: (1) shrink bytes (int8/int4 weight-only quantization ≈2×/4×),
  (2) thread/memory tuning, (3) fusing the many small passes (RoPE, norms,
  elementwise) that re-stream activations, (4) hog Wildeans like hugepages.
- **Prefill (gemm, compute-bound):** current "gemm" is T separate gemv dot
  loops per weight row — no register blocking, no packed panels, FMA
  utilization far below peak. Levers: real [T,K]×[K,N] micro-kernel with
  register tiling + packed B (à la MLAS/GGML/BLIS), better thread partitioning.
- **Load:** full-file read + per-element conversion; memmap would cut it.

## Perspectives (STORM-style)

1. **Kernel engineering** (GEMM micro-kernels, register blocking, packing,
   fused epilogues) — prefill focus.
2. **Data movement / quantization** (weight-only int8/int4, dequant-in-register,
   memory layout, hugepages, NUMA/CCX topology) — decode focus.
3. **Runtime & scheduling** (thread pool persistence, thread affinity, core
   parking, governor, allocator behavior, memmap load, avoiding per-token allocs).
4. **Numerical quality** (what does "no quality compromise" mean per lever:
   bit-exact vs tolerance-bounded; validation methodology beyond greedy md5).

## Sub-questions (Phase 1)

- Q1. What GEMM micro-kernel structure (tile shapes, packing layout, epilogue
  fusion) is optimal for [T≤~512, K∈{1536,4608,128}] × [N∈{1536,4608,2048,vocab 130560}] on Zen3 AVX2+FMA, and what fraction of peak FMA should be reachable?
- Q2. What weight-only quantization scheme (int8 per-channel? int4 group-wise?
  which matrices?) maximizes decode tok/s on Zen3 while keeping greedy
  generation token-identical or within a defensible tolerance for MiniCPM5?
- Q3. How should decode parallelism be tuned (thread count vs DRAM bandwidth,
  core topology on Cezanne: 4 CCX of 2 cores sharing L3) and what secondary
  wins exist (persistent thread pool, hugepages/THP, prefetch, software
  pipelining of dequant+FMA)?
- Q4. Which fusion/allocation cleanups matter (per-token Vec churn in
  forward_token, RoPE table reuse, KV layout) and how much load-time can
  memmap + lazy conversion save?
- Q5. What do established CPU engines (llama.cpp/GGML, MLAS, whisper.cpp,
  MLC, K-quants literature) do for exactly these two regimes, and which of
  their techniques transfer to a fixed-shape 1B BF16 model on AVX2-only
  hardware?

## Method

Karpathy-style keep/discard loop: implement one lever → run reference command
+ `cargo test` → record prefill/decode/load deltas in BENCHMARK-style table →
keep if strictly better and quality-gated, else revert. Web research
(gather phase) feeds Q1-Q5 with primary sources (llama.cpp source/docs,
GGML PRs, MLAS docs, papers). Budget: 15 queries. Stop when saturated.

## Success criteria

- Decode ≥ ~2× baseline (≥ 22 tok/s) without quality regression on the gate.
- Prefill ≥ ~1.5× baseline (≥ 100 tok/s).
- Load ≤ 2 s.
- `cargo test` green; reference-command output md5 unchanged (or change is
  explicitly numerically justified and re-pinned).
