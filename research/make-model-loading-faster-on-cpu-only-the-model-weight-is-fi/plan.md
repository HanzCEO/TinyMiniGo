# Plan: Make model loading faster on CPU-only

## Topic restated

TinyMiniGo (Rust, CPU-only llama-style engine) loads a **fixed 2.16 GB BF16
safetensors checkpoint** (`MiniCPM5-1B/model-00000-of-00001.safetensors`,
2,161,265,664 bytes total) at startup. Current load path:
`fs::read` (whole 2.16 GB into RAM) → parse header JSON → per-tensor
element-wise conversion loops (`chunks_exact` + `from_le_bytes`) into
`Vec<f32>` / `Vec<u16>`. Load currently takes ~3–5 s (per BENCHMARK.md;
~10 s at the unoptimized baseline). Goal: minimize time from process start
to first token on CPU-only hardware (no GPU, no device offload).

Constraints: the weight file is fixed (already downloaded, will not change);
the model must stay numerically compatible with the verified BF16 reference
output (bit-identical path is the default mode); CPU-only (no CUDA/Metal).

## Perspectives (STORM)

1. **OS / virtual-memory perspective** — I/O and page-cache tricks: mmap vs
   `read()`, `MAP_POPULATE`, `madvise` (SEQUENTIAL/WILLNEED), `posix_fadvise`,
   readahead, huge pages/THP, page-cache warmup, parallel reader threads,
   NUMA interleave. Question: can we avoid the 2.16 GB copy entirely?
2. **File-format / serialization perspective** — safetensors vs GGUF vs a
   custom preprocessed layout; zero-copy tensor views (borrow from mmap)
   vs eager conversion; parallel deserialization; alignment; endianness.
   The file is fixed, but *offline* preprocessing into a load-optimized
   format is allowed.
3. **Engine / runtime perspective** — how fast CPU-first engines already
   solve this: llama.cpp (GGUF + mmap), candle (MmapedSafetensors), vLLM
   (mmap safetensors load), transformers (`low_cpu_mem_usage`),
   mistral.rs / llama-rs.
4. **Data-size / quantization perspective** — load time scales with bytes;
   int8/int4 weight-only quantization (already partially in repo via
   `TMG_I8=1`), BF16 kept raw, and whether halving bytes halves load time.

## Sub-questions

1. What dominates the current load time: 2.16 GB disk read, page-cache miss,
   header JSON parse, per-element conversion loops, or allocation?
   *(measure locally, read-only)*
2. What zero-copy / mmap capabilities does the Rust `safetensors` crate
   expose (`SafeTensors::deserialize(&[u8])`, borrows), and what does
   candle's `MmapedSafetensors` do differently?
3. How do llama.cpp/GGUF, vLLM, and transformers cut CPU load time
   (mmap defaults, lazy weight loading, pread, parallel decode)?
4. Which OS techniques (MAP_POPULATE, MADV_SEQUENTIAL/WILLNEED, fadvise,
   THP, read threads, page-cache warmup) measurably speed up loading a
   single fixed 2.16 GB file on Linux?
5. What does the academic literature say about fast checkpoint loading
   (ServerlessLLM chunked checkpoints, Mmap Attention, LLM in a Flash,
   CPU inference startup)?
6. Does quantization (int8 per-row already in repo; int4 group-wise) or a
   converted native format reduce load time further, and at what
   correctness/quality cost?

## Query budget

15 total queries. Wave plan: 4 wide (web×3 + arXiv×1) → narrow on mmap
engines, safetensors internals, OS knobs, arXiv papers → verify with
targeted fetches. Saturation after 2 empty waves.

## Deliverables

- `sources.md` — kept, fetched sources with tiers (T1 primary / T2 reputable / T3 weak)
- `notes.md` — verified findings log with keep/discard decisions
- `report.md` — cited synthesis with concrete recommendations for TinyMiniGo
