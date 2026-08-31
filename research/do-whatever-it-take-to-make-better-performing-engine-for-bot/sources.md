# Sources — kept only (fetched + verified)

Local evidence (code/tests, not web):
- TinyMiniGo repo (src/*, BENCHMARK.md, tests/*) — T1 for implementation facts.
- Reference run on this machine, 2026-02-21: prefill 67.4 tok/s, decode 11.87
  tok/s, load 4.09 s, md5 e1c13ca7bedaafdd2b927c9d1fdbb41b (256 greedy tok).

## Web sources

[16] cflow: Pipeline-Native Transformers — CPU co-design for bandwidth-efficient decode (arXiv 2608.23841) | https://arxiv.org/abs/2608.23841 | T1 (paper) | KEY: quantifies the CPU decode regime (~1 TFLOP/s compute vs ~50 GB/s DRAM; every token streams every weight once); L2-sized tiles in compute-consumption order cut L1 read misses 7.3×; fusing projections matters; validates that decode speed ≈ bytes/token / effective bandwidth for dense models.

[17] Optimization of Armv9 llama.cpp inference (arXiv 2406.10816) | https://arxiv.org/abs/2406.10816 | T1 (paper) | KEY: int8 quantization + operator vectorization + compile flags on Arm (Yitian 710): prefill +1.6×, decode +24×, memory 1/5, negligible accuracy loss — quantization is THE decode lever on CPU; vectorizing the dequant path is where the speed comes from.

[18] Efficient LLM Inference on CPUs (Intel, arXiv 2311.00502, NeurIPS ENLSP) | https://arxiv.org/abs/2311.00502 | T1 (paper) | KEY: W4A16 weight-only runtime with special CPU kernels; automatic int4 flow; shows the design pattern of weight-only quant + optimized dequant-GEMV kernels for CPU decode.

[19] When Is a Columnar Scan Bandwidth-Bound? (arXiv 2606.22423) | https://arxiv.org/abs/2606.22423 | T1 (paper) | KEY: decode throughput law f = min(1, T_dec·b/(8β)) — dequant throughput T_dec (values/s) is set by decode layout, NOT bit-width; achieving bandwidth-bound int4 requires a decoder ≥ 8β/4 values/s. Directly predicts whether our int8/int4 kernels will actually hit the bandwidth ceiling.

[20] FineQuant (arXiv 2308.09723, Meta) | https://arxiv.org/abs/2308.09723 | T1 (paper) | KEY: fine-grained (per-channel/per-group) weight-only quant heuristics using weights alone; int8 nearly lossless; on-the-fly dequant GEMM pattern.

[1] Overfit: llama.cpp CPU analysis (DevOnBike) | https://github.com/DevOnBike/Overfit/blob/main/docs/llamacpp-cpu-analysis.md | T2 (engineering blog, A/B measured) | KEY: measured lever-by-lever CPU LLM decode optimization: parallelize FFN+lm_head matmul +31%, head-parallel attention +10.5%, Q8_0 3.3×, Q4_K +9.6% more, GQA K/V-once +24% bit-identical; decode can be overhead-bound, not bandwidth-bound; llama.cpp converts byte-reduction ~linearly to speed when bandwidth-bound.

[2] llama.cpp llamafile sgemm.cpp (tinyBLAS) | https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/llamafile/sgemm.cpp | T1 (primary source code) | KEY: production register-blocked GEMM for C=AᵀB on AVX2: RM=4, RN≤3 (16 regs), KN=8 (m256), accumulators span k, no packing/malloc, work-stealing over N×BN jobs, requires k%8==0, n≥2 (prefill only), m%4 handling via BM∈{4,2,1}.

[3] BLIS papers (Analytical Models for BLIS; BLIS TOMS) | https://www.cs.utexas.edu/~flame/pubs/flawn74.pdf | T1 (peer-reviewed) | KEY: GEMM = pack A into mc×kc panels, micro-kernel mr×nr register tile, 5 parameters (mr, nr, kc, mc, nc); loop order jc→pc→nc; edge sizes matter for small GEMM.

[4] MLAS sgemm.cpp + architecture (microsoft/onnxruntime) | https://github.com/microsoft/onnxruntime/blob/master/onnxruntime/core/mlas/lib/sgemm.cpp | T1 (primary source) | KEY: MLAS packs A (and B) into panels, MlasSgemmKernelLoop over CountM×CountN tiles, 2D thread partitioning (ThreadCountM × ThreadCountN) — the standard production pattern ONNX uses for transformer GEMMs incl. lm_head.

[5] oneDNN int8 nuances doc | https://uxlfoundation.github.io/oneDNN/v3.11/dev_guide_int8_computations.html | T1 (official docs) | KEY: canonical AVX2 int8 path = VPMADDUBSW (u8×s8→s16, saturating) → VPMADDWD (s16 pairs→s32) → VPADDD; sum exact at s32 stage; saturation risk is only in the s16 stage.

[6] uops.info VPMADDWD / VPMADDUBSW / VPADDD Zen 3 entries | https://uops.info/html-instr/VPMADDWD_XMM_XMM_XMM.html | T1 (measured) | KEY: on Zen 3 VPMADDWD and VPMADDUBSW run 0.5 cyc throughput (2 ports), VPADDD 0.25; int8 dot chains are throughput-fine on Zen3; VPMADDUBSW mixes signed/unsigned (needs offset trick or unsigned weights).

[7] "Which Quantization Should I Use? Unified Evaluation of llama.cpp quant formats" | https://arxiv.org/html/2601.14277v1 | T1 (paper) | KEY: Q8_0 ~essentially lossless, Q6_K imperceptible, quality cliffs begin at ≤4bpw; evaluates on Llama-3.1-8B with official tooling; validates KLD methodology.

[8] "Give Me BF16 or Give Me Death" (Llama-3.1 family quant study) | https://arxiv.org/html/2411.02355v4 | T1 (paper) | KEY: W8 weight-only quantization has no measurable quality effect across scales; INT4 weight-only starts to hurt smaller models (1B/3B) more than large — argues for int8 (or 6-bit) rather than 4-bit for a 1B model.

[9] llama.cpp perplexity tool (KLD methodology) | https://github.com/ggml-org/llama.cpp/tree/master/tools/perplexity | T1 | KEY: standard quality gate for quant: perplexity delta + KL divergence vs FP16 logits on Wikitext — we'll adapt this as logits-KLD vs BF16 engine on our own prompts.

[10] AMD Ryzen Software Optimization (GDC 2024, gpuopen) | https://gpuopen.com/download/GDC2024_AMD_Ryzen_Processor_Software_Optimization.pdf | T1 (AMD) | KEY: Zen3 per-core: 32K L1d, 512K L2/core, shared L3 per CCX; FMA dual-port; guidance for cache blocking & thread placement.

[11] Linux kernel THP docs + Ubuntu THP-madvise benchmark thread | https://docs.kernel.org/admin-guide/mm/transhuge.html + https://lists.ubuntu.com/archives/kernel-team/2017-November/088430.html | T1/T2 | KEY: THP reduces TLB misses for large streaming working sets; madvise-style opt-in is the safe pattern; benefit larger for pointer-chasing, modest for pure sequential streams.

[12] fastsafetensors paper + candle MmapedSafetensors | https://arxiv.org/html/2505.23072v1 + https://git.v0l.io/huggingface/candle | T1/T1 | KEY: mmap makes safetensors load O(1) (demand paging), candle's production pattern for zero-copy weight loading; RSS grows as pages are touched.

[13] 85.30 GFLOPS single-core FP32 Zen3 matmul project | https://github.com/houslast3/85.30-GFLOPS-Single-Core-FP32-Matrix-Multiplication-on-AMD-Zen-3 | T2 | KEY: 28-iteration optimization progression reaching ~63% of theoretical peak on Zen3 single core; documents which levers actually mattered (layout, blocking, unrolling, prefetch).

[14] ggml Q6_K block-interleaving PR #15275 | https://github.com/ggml-org/llama.cpp/pull/15275 | T1 | KEY: for prefill, repacking quantized weights into 8-row interleaved blocks (block_q6_Kx8) made batch GEMM efficient — same packing idea applies to int8: pack for gemm separately from gemv layout.

[15] ggml IQ AVX2 GEMM panel PR #27402 | https://github.com/ggml-org/llama.cpp/pull/27402 | T1 | KEY: for batch (prefill) with quantized weights: pre-decode 8-row × 256-col panels once per batch instead of per token — the "dequant once, reuse across T" pattern that fixes quantized prefill.
