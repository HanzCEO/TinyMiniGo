# Report — Making Model Loading Faster on CPU-Only (TinyMiniGo + MiniCPM5-1B)

[DONE]

## TL;DR

The load is **not disk-bound** — it is **copy-bound and RAM-pressure-bound**.
On this machine (NVMe, 16 GB RAM, Linux 6.12) a raw cold sequential read of
the 2.16 GB checkpoint takes **0.95 s**, yet the engine takes **8.5-9 s to
load — even with a warm page cache**. The current loader streams the file
≥3× into fresh allocations (fs::read → per-tensor `Vec<u16>` → `wmat()`
`to_vec()` copy → f32 conversions), and the ~7 GB of anonymous memory
evicts the page cache and pushes into swap on a 15 GB box.

Measured stage breakdown (standalone bench, release build, warm cache):

| stage | time |
|---|---|
| header JSON parse (`SafeTensors::deserialize`) | **0.7 ms** (negligible) |
| `fs::read` whole file | 1.2-4.5 s (RAM-pressure dependent) |
| per-tensor BF16 `chunks_exact` → `Vec<u16>` | ~1.05 s |
| `wmat()` second full copy (`to_vec()`) | **~2.9 s** |
| embed + norms bf16→f32 conversion (200.6M elems) | ~0.61 s |

**Fix order: (1) kill the redundant copies (expect ~2-4 s), (2) mmap with
`MAP_POPULATE` + borrow (keeps bit-identical BF16 math, zero conversion),
(3) offline-repack the fixed checkpoint into an aligned, consumption-ordered
native format (ServerlessLLM/GGUF pattern) to approach the ~0.95-1.5 s disk
floor, (4) int8/int4 to shrink bytes further. Do not use `MADV_SEQUENTIAL`.**

---

## 1. Why the loader is slow (measured, not guessed)

### 1.1 The file and machine
- `model-00000-of-00001.safetensors`: 2,161,290,912 bytes; 219 tensors, all
  BF16; 2,161,265,664 tensor bytes. Header JSON ≈ 0.7 ms to parse [1][2].
- Host: Ryzen 7 5800H, 16 GB RAM (7.3 GB already in use, 6.2 GB swap used,
  4.5 GB page cache), NVMe SSD (~2.3 GB/s measured), powersave governor.
- Cold sequential read of the entire file: **0.95 s** (2.3 GB/s).

### 1.2 The current loader makes ≥3 full-file passes
1. `fs::read` pulls all 2.16 GB into a heap buffer (copy #1).
2. Per tensor, `chunks_exact(2)` + `from_le_bytes` materializes a fresh
   `Vec<u16>` (copy #2, ~1.05 s) — despite BF16 bits on disk being exactly
   what the kernels consume (dequant is in-kernel; BENCHMARK.md).
3. `wmat()` then does `to_vec()` into `WMat::BF16` (copy #3, **~2.9 s**);
   `f32vec()` converts embed + norms to f32 (~0.61 s, 200.6M elements).
4. Peak anonymous memory ≈ 2.16 + 2.16 + 2.16 + 0.8 ≈ **7.3 GB** — beyond
   available RAM → page-cache eviction + swap churn → warm loads still
   8.5-9 s. This matches the machine state (6.2 GB swap already used).
   (Exact peak RSS probe raced; magnitude [unverified], mechanism consistent
   with M1-M3 and with the literature on copy-dominated loading [19][20].)

The safetensors crate itself is designed for zero-copy: `SafeTensors<'data>`
borrows the byte buffer, `TensorView::data()` returns `&'data [u8]`, and the
canonical example mmaps the file [1][2]. TinyMiniGo ignores all of that.

## 2. What the ecosystem does (verified)

### 2.1 Rust: candle `MmapedSafetensors` (T1)
candle-core wraps `memmap2` + header deserialize and borrows tensors from
the mapping — "avoiding full allocation" [3][4]. This is the direct
in-repo-adjacent precedent: same crate, same file format, zero-copy.

### 2.2 vLLM: mmap is the default; eager only for network FS (T1)
vLLM's default safetensors loading is lazy mmap — "highly efficient for
local storage (like SSDs)" [5][6]. They added an "eager" mode (read whole
file) only because Lustre/NFS make mmap's small random reads pathological
(94 min → 14 min on Lustre) [5]. On local NVMe the recommendation is the
opposite: **mmap/lazy wins**. Eager mode also raised CPU memory
130 GiB → 542 GiB/node in their benchmark [5] — exactly the kind of
materialization TinyMiniGo should avoid.

### 2.3 llama.cpp/GGUF: mmap + aligned layout = the format answer (T1/T2)
- llama.cpp's switch from stdio to mmap: "load LLaMA 100x faster using half
  as much memory"; multiple processes share the same page cache [8].
- Maintainer guidance: don't use `MADV_SEQUENTIAL` on weights — it evicts
  pages after reading, and the model is re-read every decode token →
  thrash; use `MADV_WILLNEED` on the whole model instead [7].
- GGUF is mmap-compatible by design: single file, tensor offsets aligned to
  `ALIGNMENT`, so tensor data is addressed directly from the mapping [12][13].
- The deeper lesson (jart, issue #91): the real fix for startup is a
  conversion step that pre-shapes and pre-aligns data so runtime structures
  map directly — i.e., **offline repacking**, not faster parsing [7].
- Caveat: plain mmap gives fast *warm* loads; cold loads are page-fault
  driven. llama.cpp's DirectIO path exists for *consistent* cold loads at
  raw disk speed (GPT-OSS-120B: ~10.5 s consistent vs 67-110 s mmap) [9].
  TinyMiniGo's goal (fast restart of a fixed local file) is the warm case.

### 2.4 ServerlessLLM (T1, OSDI'24): loading-optimized checkpoint format
ServerlessLLM repacks checkpoints into a format with (a) tensors grouped
into partitions enabling **sequential chunk-based reads**, (b) a separate
tensor index (name → offset,size) for direct addressing, (c) word-aligned
tensors. Result: **3.6-8.2× faster than safetensors loading** on NVMe; plain
safetensors caused 112K page faults on cold start for LLaMA-2-7B [10]. They
also use O_DIRECT + parallel I/O threads. This is the strongest published
evidence that **format + I/O strategy, not raw device speed, decides load
time** [10][11].

### 2.5 fastsafetensors (T1, IBM 2025): aggregate, don't instantiate one-by-one
IBM's analysis of the default safetensors path: one-by-one tensor
instantiation underutilizes NVMe (max 5 GB/s of 28 GB/s), drives high
kernel CPU via page cache, and copies through host bounce buffers.
Their fix — aggregate contiguous tensor groups into few large I/O
transfers and instantiate tensors as raw-buffer views (DLPack) with
parallel I/O threads — gives **4.8-7.5× faster loading** (vLLM Llama-2-7B
startup 7.29 s → 3.67 s) [11]. For TinyMiniGo the CPU-only analog is:
fewer, bigger buffers; no per-tensor Vec construction; no intermediate
copies.

### 2.6 transformers: copies are the enemy (T2)
transformers' basic loading materializes weights twice (random init +
real copy) → peak ~2× model size [20]; dropping mmap in 5.x caused full
materialization (17 s for a 30B model) [19]. Same lesson: **every extra
full-size copy costs seconds and RAM**.

## 3. OS-level knobs (verified)

| knob | effect | use here |
|---|---|---|
| `MAP_POPULATE` (memmap2 `.populate()`) | prefault + readahead at map time; best-effort on file mappings | yes — moves fault cost to load time [14][21] |
| `MADV_POPULATE_READ` | blocks until pages faulted; sequential readahead; kernel patch cut latency to 1/10 (1 GB file, default 128 KB readahead) | yes — stronger guarantee than MAP_POPULATE [16] |
| `MADV_WILLNEED` | prefetch entire range | yes — alternative/complement [7][15] |
| `MADV_SEQUENTIAL` | readahead **+ evict pages after read** | **no** — decode re-reads every weight each token [7][15] |
| `mlock` | pin mapped weights against swap | optional; llama.cpp `--load-mode mlock` precedent [22] |
| `O_DIRECT` / direct-io | bypass page cache, consistent cold loads | only for cold one-shot loads; not the warm-restart goal [9][10] |
| THP / huge pages | reduce PTE/TLB cost of large mappings | minor; THP=madvise already available [unverified benefit on this box] |

General mmap-vs-read guidance: mmap wins for hot, reused, shared data —
"static indexes, lookup tables, **model weights**" — exactly this workload;
buffered read wins for one-pass sequential scans [17][18]. MAP_POPULATE is
not a magic switch; benchmark [18].

## 4. Recommendations for TinyMiniGo (in priority order)

### R1 — Kill the redundant copies (biggest, easiest; bit-identical)
- `mmap` the file (`memmap2`, `MmapOptions::new().populate()`), then
  `SafeTensors::deserialize(&map)` [1][21].
- Make `TensorData::BF16Raw` hold a borrowed `&'a [u16]` (or store
  `(&'a [u8], offset)` views) instead of an owned `Vec<u16>`; drop the
  `chunks_exact` loop entirely — BF16 bits on disk are the runtime bits.
- In `model.rs`, remove the second `to_vec()` in `wmat()` (borrow or move;
  construct each `WMat` from the single per-tensor buffer, no duplicate
  full-file stream).
- Keep embed/norms conversion (it's needed as f32) but do it once over the
  mmap slice with SIMD-friendly bulk conversion, not per-element
  `chunks_exact` (currently ~0.6 s).
- **Expected: load ~8.5-9 s → ~2-4 s warm** (raw read 1.2-2 s + one f32
  conversion pass ~0.6 s), RSS drops by ~2.2-4.4 GB, page cache stays hot
  for decode. Same bit-identical math (BF16 dequant is already in-kernel;
  BENCHMARK.md "bf16→f32 is exact").

### R2 — Offline-repack the fixed checkpoint (ceiling: ~0.95-1.5 s)
The file never changes — pay conversion once, not per launch:
- Convert (a one-time script) to a ServerlessLLM-style/GGUF-style native
  layout: tensor data in consumption order, offsets aligned (e.g., 64 B for
  AVX2), a tiny binary index (name → offset,size,dtype), f32 embed/norms
  pre-converted, optional int8/int4 weights pre-quantized [7][10][12][13].
- Load becomes: mmap + parse small index + point `WMat`s at mapped offsets.
  Cold load ≈ disk read ≈ **~1 s**; warm load near-instant.
- This mirrors what llama.cpp/GGUF (mmap + aligned offsets) [7][12][13] and
  ServerlessLLM (chunked + index) [10] prove in production.

### R3 — Quantization shrinks load time (scales with bytes)
- `TMG_I8=1` (already in repo): 2.16 GB → ~1.1 GB → proportionally less
  load I/O and RAM; Q4-style would go to ~0.54 GB (~0.3-0.6 s load) [11][12].
- Keep the existing token-parity gate for any lossy path.

### R4 — Runtime knobs
- `MADV_WILLNEED` (or `MADV_POPULATE_READ` for a hard guarantee) after
  mapping [7][16]; never `MADV_SEQUENTIAL` [7][15].
- Optional `mlock` if swap is a concern [22].
- Header parse, thread-parallel loading, and other micro-optimizations are
  NOT worth it here (0.7 ms parse; bottleneck is copies, not I/O) [1][11].

## 5. Expected outcomes summary

| lever | load time (warm, est.) | RSS peak | bit-identical? |
|---|---|---|---|
| baseline (current) | ~8.5-9 s | ~7.3 GB (thrash) | yes |
| R1 zero-copy mmap + drop copies | ~2-4 s | ~3-4 GB | yes |
| R2 offline repack (aligned native format) | ~1-1.5 s cold, <1 s warm | ~2.6-3 GB | yes (f32-converted parts) |
| R3 + int8 (existing) | ~1-2 s | ~1.8 GB | no (opt-in, token-parity gated) |
| R3 + int4 (future) | ~0.5-1 s | ~1 GB | no (opt-in) |

The disk floor is ~0.95 s for a cold read; R2 approaches it. R1 alone
captures most of the win for a few hours of work and stays bit-identical.

## Sources
See sources.md — 22 kept sources (T1/T2) with URLs; 3 local measurement
notes (M1-M3) in notes.md. No claims above lack a citation except the
marked [unverified] peak-RSS magnitude.
