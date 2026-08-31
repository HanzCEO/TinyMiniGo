# Sources

Tier legend: T1 = primary (official docs, papers, kernel man pages, first-party repos/PRs) · T2 = reputable secondary (maintainer discussions, engineering write-ups) · T3 = weak/blog.

## Kept sources

[1] safetensors Rust crate docs — SafeTensors (borrowed buffer, "No Tensor allocation", mmap example) · https://docs.rs/safetensors/latest/safetensors/tensor/struct.SafeTensors.html · T1 · zero-copy design; TensorView::data() -> &'data [u8] borrowed slice.

[2] TensorView in safetensors::tensor · https://docs.rs/safetensors/latest/safetensors/tensor/struct.TensorView.html · T1 · tensor data is a reference into the full byte-buffer.

[3] candle-core src/safetensors.rs (huggingface/candle) · https://github.com/huggingface/candle/blob/main/candle-core/src/safetensors.rs · T1 · MmapedSafetensors = memmap2 + header deserialize, avoids full allocation.

[4] MmapedSafetensors docs · https://docs.rs/candle-core/latest/candle_core/safetensors/struct.MmapedSafetensors.html · T1 · "wrapper around a memory mapped file and deserialize the safetensors header"; unsafe inherited from memmap2.

[5] vLLM PR #24469 — --safetensors-load-strategy (lazy/mmap default; eager for Lustre/NFS) · https://github.com/vllm-project/vllm/pull/24469 · T1 · mmap default is "highly efficient for local storage (like SSDs)"; eager sequential read for network FS: 94 min → 14 min; eager raises CPU memory (130 GiB → 542 GiB/node in their test).

[6] vLLM LoadConfig docs · https://docs.vllm.ai/en/latest/api/vllm/config/load/ · T1 · load_format auto→safetensors; npcache stores converted torch format on disk (preprocessing precedent).

[7] ggerganov/llama.cpp issue #91 "Should use mmap for model loading" · https://github.com/ggerganov/llama.cpp/issues/91 · T2 · MADV_SEQUENTIAL evicts read pages (bad: model re-read every token) → use MADV_WILLNEED; jart: pre-shape/align data in a conversion step so runtime structures map directly; mmap may need benchmarking (TLB shootdowns).

[8] justine.lol/mmap — "Edge AI Just Got Faster" (llama.cpp mmap switch) · https://justine.lol/mmap/ · T2 (first-party engineering post) · mmap instead of stdio: "load LLaMA 100x faster using half as much memory"; multiple processes share the page cache.

[9] llama.cpp PR #18012 — Async DirectIO model loading on Linux · https://github.com/ggml-org/llama.cpp/pull/18012 · T1 · mmap fast when cache-hot; uncached read gives consistent load at raw disk speed (GPT-OSS-120B: ~10.5 s consistent vs 67-110 s mmap); direct-io for cold one-shot loads.

[10] ServerlessLLM (OSDI'24) — arXiv 2401.14351 · https://arxiv.org/abs/2401.14351 · T1 · loading-optimized checkpoint: per-partition chunked sequential reads + tensor index file + word-aligned tensors; 3.6-8.2× faster than safetensors loading (NVMe); 112K page faults cold-start for LLaMA-2-7B with plain safetensors; O_DIRECT + multi-thread I/O.

[11] fastsafetensors — arXiv 2505.23072 (IBM) · https://arxiv.org/abs/2505.23072 · T1 · current safetensors deserialization underutilizes NVMe (max 5/28 GB/s), high kernel CPU, one-by-one tensor instantiation; aggregate contiguous tensor groups into large I/O + raw-buffer instantiation + parallel I/O → 4.8-7.5× faster; vLLM Llama-2-7B startup 7.29 s → 3.67 s.

[12] GGUF spec (ggml docs) · https://github.com/ggerganov/ggml/blob/master/docs/gguf.md · T1 · "mmap compatibility: models can be loaded using mmap for fast loading and saving"; single-file.

[13] gguf.cpp — gguf_tensor_info.offset must be multiple of ALIGNMENT · https://github.com/ggml-org/llama.cpp/blob/fc2b0053/ggml/src/gguf.cpp · T1 · aligned tensor offsets enable direct mmap addressing.

[14] mmap(2) man page · https://man7.org/linux/man-pages/man2/mmap.2.html · T1 · MAP_POPULATE: prefault + readahead (best-effort for file mappings); MAP_LOCKED.

[15] madvise(2) man page · https://man7.org/linux/man-pages/man2/madvise.2.html · T1 · MADV_SEQUENTIAL: aggressive readahead, pages "may be freed soon after they are accessed"; MADV_WILLNEED: prefetch.

[16] kernel patch: madvise(MADV_POPULATE_READ) ra_pages = device max request size → latency 1/10 over default 128 KB readahead (1 GB file) · https://lists.openwall.net/linux-kernel/2024/02/02/74 · T1 · MADV_POPULATE_READ is sequential IO for file-backed VMA.

[17] kernel-internals.org — Memory-Mapped I/O · https://kernel-internals.org/io/mmap-io/ · T2 · read() = syscall+copy_to_user; mmap = page faults + PTE/TLB cost; mmap wins for random access, shared reads (page cache shared), read-heavy workloads; read() often wins for one-pass sequential streaming.

[18] Long Wang — mmap() vs read(): When Zero-Copy Is Actually Slower · https://wanglong.cv/articles/mmap-vs-read-performance/ · T2 · decision table: large one-pass sequential scan → buffered read; hot/reused/shared data → mmap; model weights = textbook mmap case; MAP_POPULATE is not a general fast switch; hints are best-effort.

[19] transformers issue #44262 — from_pretrained no longer uses mmap for CPU weights in 5.x → full materialization, 17 s for 30B · https://github.com/huggingface/transformers/issues/44262 · T2 · mmap removal → materialization cost; backs "copies dominate".

[20] stevhliu.com — The Transformers loading pipeline · https://www.stevhliu.com/2026/transformers-loading-pipeline · T2 · basic flow materializes weights twice (random init + real copy) → peak ~2× model size.

[21] memmap2 docs — MmapOptions::populate() · https://docs.rs/memmap2/latest/memmap2/struct.MmapOptions.html · T1 · "Populate (prefault) page tables… causes read-ahead… corresponds to MAP_POPULATE on Linux."

[22] llama.cpp PR #20834 / issue #26110 — load-mode mlock/mmap/directio refactor · https://github.com/ggml-org/llama.cpp/pull/20834 · T1 · mlock pins mapped weights (no swap), used for CPU-offloaded tensors.

## Discarded sources
- NVIDIA forum "from_pretrained SUPER slow on DGX Spark" (T3; superseded by [19][20]).
- Level Up Coding safetensors/GGUF blog (T3 aggregator; facts covered by T1 sources).
- strongly.ai blog (T3; format overview covered by T1).
- DeepWiki pages (T3 auto-generated; only used as index to T1 primary sources).
- ndarrow / tpt-safetensors-io / ndarray-safetensors crates (not needed; vanilla safetensors crate already exposes borrowed slices [1][2]).
- IEEE "De-Quantization Penalties" (GPU-specific; out of scope for CPU-only).

## Local measurement evidence (not web sources; recorded in notes.md)
- M1: cold dd read of model file = 0.95 s @ 2.3 GB/s.
- M2: load-stage breakdown bench (fs::read 1.2-4.5 s; header parse 0.7 ms; BF16 chunks copy ~1.05 s; second wmat copy ~2.9 s; f32 conversions ~0.61 s).
- M3: machine RAM pressure (7.3 GB used / 15 GB; 6.2 GB swap used; page cache 4.5 GB).
- Binary load: 8.59 s cold / 8.49-8.98 s warm.
