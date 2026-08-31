# Findings log — Make model loading faster on CPU-only (MiniCPM5-1B, TinyMiniGo)

Method: web+arXiv research, verified fetches, plus read-only local measurements
(no repo files edited). All timings on this machine: AMD Ryzen 7 5800H (8C/16T,
AVX2+FMA, no AVX-512), 16 GB RAM, NVMe SSD, Linux 6.12, powersave governor,
THP=madvise, HugePages_Total=0.

## Phase 0-1 (planning)
- plan.md written: 4 perspectives (OS/VM, file-format, engine/runtime,
  quantization), 6 sub-questions. Query budget 15; 11 used; saturation after
  2 empty waves (waves 4-6 reinforcing only) → stop gathering.

## Local measurements (read-only; scratch code in /tmp/loadbench)

### M1 — Raw I/O is NOT the bottleneck
- Cold sequential read of the 2,161,290,912-byte file (dd, 1 MiB blocks):
  **0.95 s ≈ 2.3 GB/s** (page cache dropped). KEEP.
- Warm `fs::read` (standalone bench): **~1.2-4.5 s** (varies with RAM pressure;
  warm page-cache read of 2.16 GB is fast).
- Full binary cold load: 8.59 s; warm: 8.49-8.98 s. Load is NOT I/O-bound.

### M2 — Load-time breakdown (standalone bench, warm cache, release build)
| stage | time |
|---|---|
| fs::read whole file | 1.2-4.5 s (RAM-pressure dependent) |
| SafeTensors::deserialize (header JSON parse) | 0.7 ms — negligible |
| per-tensor BF16 chunks_exact copy → Vec<u16> | ~1.05 s |
| wmat() to_vec() SECOND full copy | ~2.9 s (!!) |
| embed+norms bf16→f32 conversion (200.6M elems) | ~0.61 s |
- 219 tensors, all BF16, 2,161,265,664 tensor bytes. The loader streams the
  file ≥3× (fs::read → BF16Raw Vec → WMat Vec + embed f32). KEEP.

### M3 — Memory-pressure hypothesis (consistent with M2 + machine state)
- 15 GB RAM, 7.3 GB used before run, 6.2 GB swap USED, 4.5 GB page cache.
- Current loader peak: 2.16 GB (fs::read) + 2.16 GB (BF16Raw) + 2.16 GB (WMat)
  + 0.8 GB (embed f32) ≈ 7.3 GB anonymous → exceeds available RAM → page-cache
  eviction + swap churn → warm load still 8.5-9 s. KEEP (measured machine
  state; peak-RSS probe raced, [unverified] exact peak).

## Verified web sources (keep; see sources.md for URLs/tiers)

### S1 — safetensors Rust core is zero-copy / mmap-friendly (T1)
`SafeTensors<'data>` owns metadata, borrows the byte buffer; `deserialize`
does "No Tensor allocation"; official example mmaps the file via memmap2 and
deserializes the mapping; `TensorView::data()` returns `&'data [u8]` borrowed
from the buffer. The crate's design assumption is mmap+borrow — TinyMiniGo
currently does fs::read + owned copies instead.

### S2 — candle's MmapedSafetensors (T1)
candle-core safetensors.rs: `MmapedSafetensors` wraps memmap2 mmap +
header deserialize, avoiding full allocation. Explicit precedent for
zero-copy weight loading in Rust.

### S3 — vLLM defaults to lazy/mmap safetensors; eager is the fallback (T1)
vLLM `--safetensors-load-strategy` defaults "lazy" (mmap); "eager" reads the
whole file into memory, recommended only for network FS (Lustre/NFS) where
mmap's small random reads are pathological. On local NVMe, mmap is the
default and preferred. Their Lustre case: 94 min → 14 min with eager. On
local NVMe the opposite ordering applies: mmap/lazy wins.

### S4 — llama.cpp history: mmap → 100× faster loading, half the memory (T1/T2)
- justine.lol/mmap: llama.cpp switched to mmap instead of C++ stdio: "load
  LLaMA 100x faster using half as much memory"; concurrent processes share
  page cache.
- Issue #91 (T2, maintainer discussion): do NOT use MADV_SEQUENTIAL (evicts
  pages after read; model is re-read every token → thrash); use
  MADV_WILLNEED on the whole model to kick off paging; mmap can be slower
  than regular I/O on some setups due to TLB shootdowns — benchmark.
- Async DirectIO PR #18012 (T2): mmap is fast when page-cache-hot, but
  uncached read gives *consistent* load times at raw disk speed (~10.5 s vs
  67-110 s for GPT-OSS-120B on DGX Spark); direct-io is for cold/one-shot
  loads. For TinyMiniGo (fixed file, want fast warm restart), mmap wins.

### S5 — ServerlessLLM (T1, USENIX OSDI'24, arXiv 2401.14351)
Loading-optimized checkpoint: tensors grouped per-GPU into partitions,
sequential chunk-based reads, separate tensor index file (name→offset,size),
word-aligned tensors for direct addressing. 3.6-8.2× faster than safetensors
loading on NVMe; 112K page faults for LLaMA-2-7B cold start with plain
safetensors. Uses O_DIRECT + parallel I/O threads. Implication: offline
repacking of the fixed checkpoint into a load-optimized native layout is a
proven, big-win pattern.

### S6 — fastsafetensors (T1, arXiv 2505.23072, IBM)
Analysis: current safetensors deserialization instantiates tensors one by
one → underutilizes NVMe (max 5 GB/s of 28 GB/s), high kernel CPU from page
cache, mmap + host copies. Fix: aggregate contiguous tensor groups into one
large I/O, instantiate via raw-buffer views (DLPack), parallel I/O threads;
4.8-7.5× faster model loading (vLLM: 7.29 s → 3.67 s for Llama-2-7B).
Directly validates: (a) aggregate I/O over per-tensor deserialization,
(b) fewer/bigger buffers, (c) avoid one-by-one tensor object creation.

### S7 — GGUF: mmap-compatible format with aligned tensors (T1)
GGUF designed for mmap: single file, KV metadata, tensor descriptors;
`gguf_tensor_info.offset` must be a multiple of `ALIGNMENT`; llama.cpp maps
tensor data directly from the file. A converted/repacked format removes
per-load conversion entirely (jart in #91: "introduce a third conversion
step that creates a new file format, where the data is in the appropriate
shape and alignment ahead of time" — exactly ServerlessLLM's + GGUF's
approach).

### S8 — OS-level mmap knobs (T1)
- mmap(2)/madvise(2) man pages: MAP_POPULATE prefaults (file-backed:
  readahead, best-effort); MADV_POPULATE_READ blocks until faulted;
  MADV_WILLNEED prefetches; MADV_SEQUENTIAL evicts after read.
- Kernel patch (2024): MADV_POPULATE_READ with ra_pages=device max reduced
  latency to 1/10 for a 1 GB file vs default 128 KB readahead.
- kernel-internals.org + wanglong.cv mmap-vs-read (T2): for one-pass
  sequential scan, buffered read is competitive; mmap wins for hot/reused
  data and shared-across-processes data (model weights = textbook case).
  mmap costs: page faults, TLB pressure, PTE memory; MAP_POPULATE
  (memmap2 `.populate()`) moves fault cost to load time.

### S9 — transformers/stevhliu + transformers issue #44262 (T2)
transformers loading materializes weights twice in CPU RAM (random-init
copy + real copy) → peak ~2× model size; transformers 5.x dropped mmap for
CPU weights causing full materialization + 17 s load for a 30B. Reinforces:
copies dominate load time; mmap avoids the double materialization.

### S10 — memmap2 Rust (T1)
`MmapOptions::populate()` = MAP_POPULATE on Linux ("prefault page tables…
read-ahead… reduce blocking on page faults later"). Direct Rust
implementation path.

## Synthesis → recommendations (for report.md)
1. Remove redundant copies in the current loader (biggest, easiest win):
   - mmap the file (memmap2, `MmapOptions::populate()`), `SafeTensors::deserialize(&map)`.
   - Convert `TensorData::BF16Raw(Vec<u16>)` → `&[u16]` borrowed from the
     mapping (zero-copy: BF16 stays in file, no Vec, no chunks loop).
   - Drop the second copy in `wmat()` (`to_vec()` → borrow or construct WMat
     by moving the single per-tensor buffer, no double full-file stream).
   - This alone removes ≥2 of 3 full-file streams (~1-4 s) and ~2.2-4.4 GB
     of anonymous allocations → less page-cache eviction → warm loads drop
     from ~8.5-9 s toward ~2-3 s (raw read + one conversion pass).
   - Conversion (embed/norms bf16→f32, ~0.6 s) can run once, not per load,
     if the offline repack keeps f32 copies — see next.
2. Offline-preprocess the FIXED checkpoint once (file never changes!):
   - Repack into a load-optimized native layout (GGUF-style or
     ServerlessLLM-style chunked + index): tensor data in consumption order,
     ALIGNMENT-aligned offsets (AVX2 alignment for kernels), optionally with
     f32-converted embed/norms and int8/int4-quantized weights.
   - Then load = mmap + parse tiny index + point WMat at mapped offsets:
     ~0.95 s (disk read) to ~1.2-1.5 s total; second and later runs can even
     skip population (page cache warm). This is the ceiling: you can't beat
     the disk read.
3. Keep BF16 raw (bit-identical path) — no conversion at load.
4. int8 (TMG_I8) / future int4 quantization shrinks bytes → load time scales
   with bytes; Q4 ≈ 0.54 GB file → ~0.3-0.6 s load. Quality tradeoff must
   keep the existing token-parity gate.
5. Do NOT use MADV_SEQUENTIAL (evicts pages; decode re-reads every token);
   use MADV_WILLNEED / MAP_POPULATE instead.
6. Don't bother with header parse optimization (0.7 ms) or multiple
   processes (single-process CLI app).

## Discarded
- justine.lol/mmap fetch failed initially (T2 note kept from search snippet:
  100× faster claim is first-party llama.cpp blog — kept as T2 with
  caveat).
- Thread-parallel loading (ServerlessLLM uses it, but our bottleneck is
  copies not I/O; disk read alone is 0.95 s) — mentioned as minor.
- HuggingFace 'from_pretrained SUPER slow on DGX Spark' forum post (T3) —
  not needed; transformers issue + stevhliu cover it.
