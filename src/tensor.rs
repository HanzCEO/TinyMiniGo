//! Core tensor math for Llama-style (MiniCPM5) inference, hand-rolled on f32 slices.
//!
//! Hot kernels (gemv / batched gemm) have AVX2+FMA paths selected at runtime
//! (std::arch::is_x86_feature_detected!) with a portable scalar fallback.

use rand::{Rng, distr::Distribution};

/// Cached SIMD support flags (detected once).
fn has_avx2_fma() -> bool {
    use std::sync::OnceLock;
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    })
}

/// Dot product a·b with the best available SIMD path.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // safety: feature detected at runtime
            return unsafe { dot_f32_avx2(a, b) };
        }
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Dot product of f32 activation with raw-BF16 weight row.
/// BF16 -> f32 dequantization happens in-register (u16 -> u32 << 16),
/// so the arithmetic is bit-identical to the pre-converted f32 path.
#[inline]
pub(crate) fn dot_bf16(a: &[f32], b: &[u16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx2_fma() {
            // safety: feature detected at runtime
            return unsafe { dot_bf16_avx2(a, b) };
        }
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, w)| x * f32::from_bits((*w as u32) << 16))
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 { unsafe {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let chunks = n / 8;
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
    // horizontal sum
    let mut sum = reduce_add(acc);
    for i in chunks * 8..n {
        sum += a[i] * b[i];
    }
    sum
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(crate) unsafe fn dot_bf16_avx2(a: &[f32], b: &[u16]) -> f32 { unsafe {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_ps();
    let chunks = n / 8;
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        // load 8 bf16 u16s, zero-extend to 8 x u32, shift left 16 -> f32 bits
        let w128 = _mm_loadu_si128(b.as_ptr().add(i * 8) as *const __m128i);
        let w256 = _mm256_cvtepu16_epi32(w128);
        let wf = _mm256_castsi256_ps(_mm256_slli_epi32(w256, 16));
        acc = _mm256_fmadd_ps(va, wf, acc);
    }
    let mut sum = reduce_add(acc);
    for i in chunks * 8..n {
        sum += a[i] * f32::from_bits((b[i] as u32) << 16);
    }
    sum
}}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn reduce_add(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    // 256 -> 128: add the two 128-bit halves
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    // 128 -> scalar: [s0+s2, s1+s3, ..] then [.., (s0+s2)+(s1+s3), ..]
    let s2 = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s3 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 0x55));
    _mm_cvtss_f32(s3)
}

/// Exact i32 horizontal sum of 8 i32 lanes.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_add_i32(v: std::arch::x86_64::__m256i) -> i32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256(v, 1);
    let s = _mm_add_epi32(lo, hi);
    let s2 = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x0E)); // [s0+s2, s1+s3, _, _]
    let s3 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0x01)); // [all, _, _, _]
    _mm_cvtsi128_si32(s3)
}

/// Quantize one activation vector to i8 with a single symmetric scale.
/// Returns (q, scale) where x ≈ q * scale. Scalar implementation: runs once
/// per activation row (K elements), negligible vs the N-row weight stream.
pub(crate) fn quantize_row_i8(x: &[f32]) -> (Vec<i8>, f32) {
    let maxabs = x.iter().fold(0f32, |m, &v| v.abs().max(m));
    if maxabs == 0.0 {
        return (vec![0i8; x.len()], 0.0);
    }
    let s = maxabs / 127.0;
    let inv = 1.0 / s;
    let mut q = vec![0i8; x.len()];
    for (qi, &xv) in q.iter_mut().zip(x.iter()) {
        *qi = (xv * inv).round().clamp(-127.0, 127.0) as i8;
    }
    (q, s)
}

/// Integer dot: sum_i a_i * b_i for i8 slices (exact, accumulated in i32 via
/// i16 pairwise madd). AVX2 path: 32 i8 lanes per iteration.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let n = a.len();
    let mut acc = _mm256_setzero_si256();
    let chunks = n / 32;
    for c in 0..chunks {
        let i = c * 32;
        let av = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let bv = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);
        // widen 32 i8 -> 16 i16 pairs in two registers
        let a01 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(av));   // a[0..16]
        let a23 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(av, 1)); // a[16..32]
        let b01 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(bv));
        let b23 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bv, 1));
        let m01 = _mm256_madd_epi16(a01, b01); // 8 x i32
        let m23 = _mm256_madd_epi16(a23, b23);
        acc = _mm256_add_epi32(acc, m01);
        acc = _mm256_add_epi32(acc, m23);
    }
    let mut sum = reduce_add_i32(acc);
    for i in chunks * 32..n {
        sum += a[i] as i32 * b[i] as i32;
    }
    sum
}

/// Number of worker threads for parallel matmuls (0 = auto).
static N_THREADS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Set worker-thread count for parallel matmuls. Called at startup from
/// the TMG_THREADS env var (default 0 = all logical cores).
pub fn set_num_threads(n: usize) {
    N_THREADS.store(n, std::sync::atomic::Ordering::Relaxed);
}

fn num_threads() -> usize {
    let n = N_THREADS.load(std::sync::atomic::Ordering::Relaxed);
    if n == 0 {
        std::thread::available_parallelism().map(|v| v.get()).unwrap_or(1)
    } else {
        n
    }
}

/// Minimum output rows before threading pays for itself (spawn overhead).
const MIN_ROWS_FOR_THREADS: usize = 64;

/// Weight matrix storage: raw BF16 bits (2 bytes/element), pre-converted
/// f32, or int8-quantized (1 byte/element) with a per-row f32 scale.
/// I8 halves the bytes streamed during memory-bound decode; quantization is
/// symmetric per-row (w ≈ scale[o] · q), applied at load time.
#[derive(Clone)]
pub enum WMat {
    F32(Vec<f32>),
    BF16(Vec<u16>),
    /// (quantized rows, per-row scale)
    I8(Vec<i8>, Vec<f32>),
    /// Borrowed zero-copy view into a TMB mmap: (byte offset, byte len) of
    /// BF16 raw bits. Resolved via `WMatView::bytes()` during matmul.
    View(u64, u64),
}

/// Marker trait: types that can resolve a borrowed `WMat::View` into raw
/// bytes (the TMB mmap). Implemented by `&[u8]` so tests can use a plain
/// slice. Kernels that match `WMat::View` require `R: WMatView`.
pub trait WMatView: Send + Sync {
    fn wmat_bytes(&self, off: u64, len: u64) -> Option<&[u8]>;
}

impl WMatView for &[u8] {
    fn wmat_bytes(&self, off: u64, len: u64) -> Option<&[u8]> {
        let start = off as usize;
        let end = start + len as usize;
        self.get(start..end)
    }
}

impl WMatView for Option<memmap2::Mmap> {
    fn wmat_bytes(&self, off: u64, len: u64) -> Option<&[u8]> {
        match self {
            Some(m) => {
                let start = off as usize;
                let end = start + len as usize;
                m.get(start..end)
            }
            None => None,
        }
    }
}

impl WMat {
    pub fn len(&self) -> usize {
        match self {
            WMat::F32(v) => v.len(),
            WMat::BF16(v) => v.len(),
            WMat::I8(q, _) => q.len(),
            WMat::View(_, len) => (*len as usize) / 2, // BF16: 2 bytes/elem
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Resolve a `WMat::View` into a borrowed `[u16]` BF16 slice.
    pub fn bf16_view<'m, R: WMatView>(&self, m: &'m R) -> Option<&'m [u16]> {
        match self {
            WMat::View(off, len) => {
                let b = m.wmat_bytes(*off, *len)?;
                if b.len() % 2 != 0 {
                    return None;
                }
                // Safe: TMB offsets are 8-byte aligned and mmap base is
                // page-aligned, so this slice start is at least 2-aligned.
                Some(bytemuck::cast_slice(b))
            }
            _ => None,
        }
    }

    /// Extract the (offset, len) of a `WMat::View` (for tests / tools).
    pub fn view_range(&self) -> Option<(u64, u64)> {
        match self {
            WMat::View(off, len) => Some((*off, *len)),
            _ => None,
        }
    }

    /// Symmetric per-row int8 quantization from an f32 row-major matrix.
    /// scale[o] = max|w| / 127; q = round(w / scale).
    pub fn quantize_i8(w: &[f32], out_features: usize, in_features: usize) -> WMat {
        let mut q = vec![0i8; w.len()];
        let mut s = vec![0f32; out_features];
        for o in 0..out_features {
            let row = &w[o * in_features..(o + 1) * in_features];
            let maxabs = row.iter().fold(0f32, |m, &v| v.abs().max(m));
            let scale = if maxabs > 0.0 { maxabs / 127.0 } else { 1.0 };
            s[o] = scale;
            for (qi, &wv) in q[o * in_features..(o + 1) * in_features]
                .iter_mut()
                .zip(row.iter())
            {
                *qi = (wv / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
        WMat::I8(q, s)
    }

    /// Row slice for output row o (row-major [out, in]).
    #[allow(dead_code)] // debugging helper
    pub fn row_f32(&self, o: usize, in_features: usize) -> Vec<f32> {
        match self {
            WMat::F32(v) => v[o * in_features..(o + 1) * in_features].to_vec(),
            WMat::BF16(v) => v[o * in_features..(o + 1) * in_features]
                .iter()
                .map(|w| f32::from_bits((*w as u32) << 16))
                .collect(),
            WMat::I8(q, s) => q[o * in_features..(o + 1) * in_features]
                .iter()
                .map(|&w| w as f32 * s[o])
                .collect(),
            WMat::View(_, _) => vec![], // no owned data to slice; use bf16_view()
        }
    }
}

/// y = x @ W^T where W is [out, in] row-major (HF convention: nn.Linear stores weight [out, in]).
/// (gemv path: dot products via SIMD, output rows split across threads.)
pub fn matmul_w<R: WMatView>(x: &[f32], w: &WMat, out_features: usize, in_features: usize, m: &R) -> Vec<f32> {
    let mut y = vec![0.0f32; out_features];
    matmul_w_into(x, w, out_features, in_features, &mut y, m);
    y
}

/// f32 gemv (used by tests and the f32 fallback path).
#[allow(dead_code)] // f32 fallback path, used by tests
#[allow(dead_code)] // f32 scalar fallback / used by tests

pub fn matmul(x: &[f32], w: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_features];
    matmul_into(x, w, out_features, in_features, &mut y);
    y
}

pub fn matmul_w_into<R: WMatView>(
    x: &[f32],
    w: &WMat,
    out_features: usize,
    in_features: usize,
    y: &mut [f32],
    m: &R,
) {
    debug_assert_eq!(y.len(), out_features);
    // int8 fast path: quantize activation once, integer dots per row, scale
    if let WMat::I8(q, scales) = w {
        #[cfg(target_arch = "x86_64")]
        if has_avx2_fma() {
            let (xq, xs) = quantize_row_i8(x);
            let nthreads = (num_threads() / 4).max(1).min(out_features);
            if nthreads <= 1 || out_features < MIN_ROWS_FOR_THREADS {
                unsafe {
                    for o in 0..out_features {
                        y[o] = dot_i8(&xq, &q[o * in_features..(o + 1) * in_features]) as f32
                            * xs
                            * scales[o];
                    }
                }
                return;
            }
            let chunk = out_features.div_ceil(nthreads);
            std::thread::scope(|s| {
                for (ti, ys) in y.chunks_mut(chunk).enumerate() {
                    let start = ti * chunk;
                    let xq = &xq;
                    s.spawn(move || {
                        unsafe {
                            for (i, y_o) in ys.iter_mut().enumerate() {
                                let o = start + i;
                                *y_o = dot_i8(xq, &q[o * in_features..(o + 1) * in_features])
                                    as f32
                                    * xs
                                    * scales[o];
                            }
                        }
                    });
                }
            });
            return;
        }
        // scalar fallback (no AVX2): dequant-accumulate
        for o in 0..out_features {
            let mut acc = 0f32;
            let row = &q[o * in_features..(o + 1) * in_features];
            for (xv, wq) in x.iter().zip(row.iter()) {
                acc += xv * (*wq as f32);
            }
            y[o] = acc * scales[o];
        }
        return;
    }
    let nthreads = (num_threads() / 4).max(1).min(out_features);
    if nthreads <= 1 || out_features < MIN_ROWS_FOR_THREADS {
        match w {
            WMat::F32(wf) => {
                for (o, y_o) in y.iter_mut().enumerate() {
                    let row = &wf[o * in_features..(o + 1) * in_features];
                    *y_o = dot(x, row);
                }
            }
            WMat::BF16(wb) => {
                for (o, y_o) in y.iter_mut().enumerate() {
                    let row = &wb[o * in_features..(o + 1) * in_features];
                    *y_o = dot_bf16(x, row);
                }
            }
            WMat::View(off, len) => {
                if let Some(wb) = m.wmat_bytes(*off, *len) {
                    let wb: &[u16] = bytemuck::cast_slice(wb);
                    for (o, y_o) in y.iter_mut().enumerate() {
                        let row = &wb[o * in_features..(o + 1) * in_features];
                        *y_o = dot_bf16(x, row);
                    }
                }
            }
            // I8 already handled above (AVX2) — scalar fallback reached on
            // non-AVX2 only, handled before threading; keep for exhaustiveness
            WMat::I8(_, _) => unreachable!("I8 scalar path handled earlier"),
        }
        return;
    }
    let chunk = out_features.div_ceil(nthreads);
    match w {
        WMat::I8(_, _) => unreachable!("I8 handled before threading"),
        WMat::F32(wf) => std::thread::scope(|s| {
            for (ti, ys) in y.chunks_mut(chunk).enumerate() {
                let start = ti * chunk;
                s.spawn(move || {
                    for (i, y_o) in ys.iter_mut().enumerate() {
                        let o = start + i;
                        let row = &wf[o * in_features..(o + 1) * in_features];
                        *y_o = dot(x, row);
                    }
                });
            }
        }),
        WMat::BF16(wb) => std::thread::scope(|s| {
            for (ti, ys) in y.chunks_mut(chunk).enumerate() {
                let start = ti * chunk;
                s.spawn(move || {
                    for (i, y_o) in ys.iter_mut().enumerate() {
                        let o = start + i;
                        let row = &wb[o * in_features..(o + 1) * in_features];
                        *y_o = dot_bf16(x, row);
                    }
                });
            }
        }),
        WMat::View(off, len) => std::thread::scope(|s| {
            if let Some(wb) = m.wmat_bytes(*off, *len) {
                let wb: &[u16] = bytemuck::cast_slice(wb);
                for (ti, ys) in y.chunks_mut(chunk).enumerate() {
                    let start = ti * chunk;
                    s.spawn(move || {
                        for (i, y_o) in ys.iter_mut().enumerate() {
                            let o = start + i;
                            let row = &wb[o * in_features..(o + 1) * in_features];
                            *y_o = dot_bf16(x, row);
                        }
                    });
                }
            }
        }),
    }
}

/// Writes y = x @ W^T into a caller-provided buffer (no allocation when y is reused).
/// Decode (gemv) is memory-bandwidth-bound: many threads contending on DRAM
/// actually hurt (measured 6.5 -> 5.7 tok/s with 16 threads on a 5800H),
/// so gemv uses a quarter of the cores while batched gemm (compute-bound
/// prefill) uses all of them.
#[allow(dead_code)] // f32 scalar fallback / used by tests

pub fn matmul_into(x: &[f32], w: &[f32], out_features: usize, in_features: usize, y: &mut [f32]) {
    assert_eq!(x.len(), in_features, "matmul input dim mismatch");
    debug_assert_eq!(y.len(), out_features);
    let nthreads = (num_threads() / 4).max(1).min(out_features);
    if nthreads <= 1 || out_features < MIN_ROWS_FOR_THREADS {
        for (o, y_o) in y.iter_mut().enumerate() {
            let row = &w[o * in_features..(o + 1) * in_features];
            *y_o = dot(x, row);
        }
        return;
    }
    if std::env::var("TMG_TRACE").is_ok() { eprintln!("matmul_into threaded: out={out_features} threads={nthreads}"); }
    let chunk = out_features.div_ceil(nthreads);
    std::thread::scope(|s| {
        for (ti, ys) in y.chunks_mut(chunk).enumerate() {
            let start = ti * chunk;
            s.spawn(move || {
                for (i, y_o) in ys.iter_mut().enumerate() {
                    let o = start + i;
                    let row = &w[o * in_features..(o + 1) * in_features];
                    *y_o = dot(x, row);
                }
            });
        }
    });
}

/// Batched matmul: x is [T, in_features] row-major, W is [out, in]; returns [T, out].
/// For each output row o (weight row streamed once), computes T dot products.
/// The weight row is the large operand, so this maximizes cache reuse.
#[allow(dead_code)] // f32 scalar fallback / used by tests

pub fn matmul_batch(x: &[f32], w: &[f32], out_features: usize, in_features: usize, n_tokens: usize) -> Vec<f32> {
    assert_eq!(x.len(), n_tokens * in_features, "batched matmul input dim mismatch");
    let mut y = vec![0.0f32; n_tokens * out_features];
    matmul_batch_into(x, w, out_features, in_features, n_tokens, &mut y);
    y
}

/// Writes [T, out] = x[T,in] @ W^T into a caller-provided buffer.
#[allow(dead_code)] // f32 scalar fallback / used by tests

pub fn matmul_batch_into(x: &[f32], w: &[f32], out_features: usize, in_features: usize, n_tokens: usize, y: &mut [f32]) {
    assert_eq!(x.len(), n_tokens * in_features, "batched matmul input dim mismatch");
    debug_assert_eq!(y.len(), n_tokens * out_features);
    let nthreads = num_threads().min(out_features);
    if nthreads <= 1 || out_features < MIN_ROWS_FOR_THREADS {
        for o in 0..out_features {
            let row = &w[o * in_features..(o + 1) * in_features];
            for t in 0..n_tokens {
                let xv = &x[t * in_features..(t + 1) * in_features];
                y[t * out_features + o] = dot(xv, row);
            }
        }
        return;
    }
    let chunk = out_features.div_ceil(nthreads);
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for ti in 0..nthreads {
            let start = ti * chunk;
            let rows = chunk.min(out_features - start);
            if rows == 0 { continue; }
            handles.push((start, rows, s.spawn(move || {
                // local [T, rows] buffer, scattered into y afterwards
                let mut local = vec![0.0f32; n_tokens * rows];
                for o in 0..rows {
                    let row = &w[(start + o) * in_features..(start + o + 1) * in_features];
                    for t in 0..n_tokens {
                        let xv = &x[t * in_features..(t + 1) * in_features];
                        local[t * rows + o] = dot(xv, row);
                    }
                }
                local
            })));
        }
        for (start, rows, h) in handles {
            let local = h.join().unwrap();
            for t in 0..n_tokens {
                y[t * out_features + start..t * out_features + start + rows]
                    .copy_from_slice(&local[t * rows..(t + 1) * rows]);
            }
        }
    });
}

/// Batched matmul: x is [T, in_features] row-major, W is [out, in]; returns [T, out].
/// For each output row o (weight row streamed once), computes T dot products.
/// The weight row is the large operand, so this maximizes cache reuse.
pub fn matmul_batch_w<R: WMatView>(
    x: &[f32],
    w: &WMat,
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    m: &R,
) -> Vec<f32> {
    assert_eq!(x.len(), n_tokens * in_features, "batched matmul input dim mismatch");
    let mut y = vec![0.0f32; n_tokens * out_features];
    matmul_batch_w_into(x, w, out_features, in_features, n_tokens, &mut y, m);
    y
}

/// Writes [T, out] = x[T,in] @ W^T into a caller-provided buffer.
pub fn matmul_batch_w_into<R: WMatView>(
    x: &[f32],
    w: &WMat,
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    y: &mut [f32],
    m: &R,
) {
    assert_eq!(x.len(), n_tokens * in_features, "batched matmul input dim mismatch");
    debug_assert_eq!(y.len(), n_tokens * out_features);
    #[cfg(target_arch = "x86_64")]
    if has_avx2_fma() && n_tokens >= 2 {
        // safety: feature detected at runtime
        unsafe { matmul_batch_w_avx2(x, w, out_features, in_features, n_tokens, y, m) };
        return;
    }
    let nthreads = num_threads().min(out_features);
    if nthreads <= 1 || out_features < MIN_ROWS_FOR_THREADS {
        match w {
            WMat::F32(wf) => {
                for o in 0..out_features {
                    let row = &wf[o * in_features..(o + 1) * in_features];
                    for t in 0..n_tokens {
                        let xv = &x[t * in_features..(t + 1) * in_features];
                        y[t * out_features + o] = dot(xv, row);
                    }
                }
            }
            WMat::BF16(wb) => {
                for o in 0..out_features {
                    let row = &wb[o * in_features..(o + 1) * in_features];
                    for t in 0..n_tokens {
                        let xv = &x[t * in_features..(t + 1) * in_features];
                        y[t * out_features + o] = dot_bf16(xv, row);
                    }
                }
            }
            WMat::View(off, len) => {
                if let Some(wb) = m.wmat_bytes(*off, *len) {
                    let wb: &[u16] = bytemuck::cast_slice(wb);
                    for o in 0..out_features {
                        let row = &wb[o * in_features..(o + 1) * in_features];
                        for t in 0..n_tokens {
                            let xv = &x[t * in_features..(t + 1) * in_features];
                            y[t * out_features + o] = dot_bf16(xv, row);
                        }
                    }
                }
            }
            WMat::I8(q, s) => {
                for o in 0..out_features {
                    let row = &q[o * in_features..(o + 1) * in_features];
                    for t in 0..n_tokens {
                        let xv = &x[t * in_features..(t + 1) * in_features];
                        let mut acc = 0f32;
                        for (a, wq) in xv.iter().zip(row.iter()) {
                            acc += a * (*wq as f32);
                        }
                        y[t * out_features + o] = acc * s[o];
                    }
                }
            }
        }
        return;
    }
    // NOTE: scalar/threaded fallback mirrors the AVX2 kernel's partitioning
    // (row-pair blocks) only loosely; it exists for non-AVX2 machines and
    // correctness, not speed.
    let chunk = out_features.div_ceil(nthreads);
    match w {
        WMat::I8(_, _) => unreachable!("I8 handled in single-thread path"),
        WMat::F32(wf) => std::thread::scope(|s| {
            let mut handles = Vec::new();
            for ti in 0..nthreads {
                let start = ti * chunk;
                let rows = chunk.min(out_features - start);
                if rows == 0 { continue; }
                handles.push((start, rows, s.spawn(move || {
                    let mut local = vec![0.0f32; n_tokens * rows];
                    for o in 0..rows {
                        let row = &wf[(start + o) * in_features..(start + o + 1) * in_features];
                        for t in 0..n_tokens {
                            let xv = &x[t * in_features..(t + 1) * in_features];
                            local[t * rows + o] = dot(xv, row);
                        }
                    }
                    local
                })));
            }
            for (start, rows, h) in handles {
                let local = h.join().unwrap();
                for t in 0..n_tokens {
                    y[t * out_features + start..t * out_features + start + rows]
                        .copy_from_slice(&local[t * rows..(t + 1) * rows]);
                }
            }
        }),
        WMat::BF16(wb) => std::thread::scope(|s| {
            let mut handles = Vec::new();
            for ti in 0..nthreads {
                let start = ti * chunk;
                let rows = chunk.min(out_features - start);
                if rows == 0 { continue; }
                handles.push((start, rows, s.spawn(move || {
                    let mut local = vec![0.0f32; n_tokens * rows];
                    for o in 0..rows {
                        let row = &wb[(start + o) * in_features..(start + o + 1) * in_features];
                        for t in 0..n_tokens {
                            let xv = &x[t * in_features..(t + 1) * in_features];
                            local[t * rows + o] = dot_bf16(xv, row);
                        }
                    }
                    local
                })));
            }
            for (start, rows, h) in handles {
                let local = h.join().unwrap();
                for t in 0..n_tokens {
                    y[t * out_features + start..t * out_features + start + rows]
                        .copy_from_slice(&local[t * rows..(t + 1) * rows]);
                }
            }
        }),
        WMat::View(off, len) => std::thread::scope(|s| {
            if let Some(wb) = m.wmat_bytes(*off, *len) {
                let wb: &[u16] = bytemuck::cast_slice(wb);
                let mut handles = Vec::new();
                for ti in 0..nthreads {
                    let start = ti * chunk;
                    let rows = chunk.min(out_features - start);
                    if rows == 0 { continue; }
                    handles.push((start, rows, s.spawn(move || {
                        let mut local = vec![0.0f32; n_tokens * rows];
                        for o in 0..rows {
                            let row = &wb[(start + o) * in_features..(start + o + 1) * in_features];
                            for t in 0..n_tokens {
                                let xv = &x[t * in_features..(t + 1) * in_features];
                                local[t * rows + o] = dot_bf16(xv, row);
                            }
                        }
                        local
                    })));
                }
                for (start, rows, h) in handles {
                    let local = h.join().unwrap();
                    for t in 0..n_tokens {
                        y[t * out_features + start..t * out_features + start + rows]
                            .copy_from_slice(&local[t * rows..(t + 1) * rows]);
                    }
                }
            }
        }),
    }
}

/// Register-blocked batched GEMM for prefill, in the style of llama.cpp's
/// tinyBLAS (llamafile sgemm): a 4-token × 2-output-row tile of k-major
/// `__m256` accumulators. Each weight row is dequantized/loaded ONCE per
/// 4 tokens (instead of once per token in the naive loop), halving+ the
/// weight bytes streamed per token pair and tripling FMA reuse.
/// Partitioning: 8 output rows (4 row-pairs) per job, jobs stolen by threads.
/// Numerics: identical accumulation order per (t, o) as the scalar dot
/// (sequential over k in 8-wide FMA chains), so results are bit-identical
/// to the old kernel on AVX2 machines. I8 weights: i8→f32 widening in-register
/// (cvtepi8_epi32×2→cvtepi32_ps×2) per 8-lane chunk; scales applied at writeback.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn matmul_batch_w_avx2<R: WMatView>(
    x: &[f32],
    w: &WMat,
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    y: &mut [f32],
    m: &R,
) { unsafe {
    use std::arch::x86_64::*;
    debug_assert_eq!(x.len(), n_tokens * in_features);
    debug_assert_eq!(y.len(), n_tokens * out_features);

    let rowblock = 8; // output rows per job (4 row-pairs of RN=2)
    let njobs = out_features.div_ceil(rowblock);
    let nthreads = num_threads().min(njobs).max(1);

    if nthreads <= 1 {
        gemm_tile_avx2(x, w, out_features, in_features, n_tokens, y, 0, out_features, m);
        return;
    }

    std::thread::scope(|s| {
        // static split with tail balanced onto earlier threads; each thread
        // writes disjoint rows of y (split_at_mut on disjoint row ranges)
        let base = njobs / nthreads;
        let extra = njobs % nthreads;
        let mut job_start = 0usize;
        let mut handles = Vec::with_capacity(nthreads);
        for ti in 0..nthreads {
            let jobs = base + if ti < extra { 1 } else { 0 };
            if jobs == 0 { break; }
            let start_row = job_start * rowblock;
            let end_row = ((job_start + jobs) * rowblock).min(out_features);
            job_start += jobs;
            // disjoint row band of y for this thread: [start_row, end_row)
            // y layout is [t, o] so a row band is a strided set of slices;
            // pass the pointer as a usize (plain integers are Send) and
            // rebuild the slice inside the thread — disjointness is
            // guaranteed by construction (bands partition out_features).
            let yp = y.as_mut_ptr() as usize;
            let ylen = y.len();
            handles.push(s.spawn(move || {
                let yband = unsafe { std::slice::from_raw_parts_mut(yp as *mut f32, ylen) };
                gemm_tile_avx2(x, w, out_features, in_features, n_tokens, yband, start_row, end_row, m)
            }));
        }
        for h in handles {
            let _ = h.join();
        }
    });
}}

/// Compute y[t, o] = dot(x[t, :], w[o, :]) for o in [start_row, end_row)
/// using a 4-token × 2-row register tile of k-major m256 accumulators.
/// Writes go straight into y (rows are disjoint across threads).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_tile_avx2<R: WMatView>(
    x: &[f32],
    w: &WMat,
    out_features: usize,
    in_features: usize,
    n_tokens: usize,
    y: &mut [f32],
    start_row: usize,
    end_row: usize,
    m: &R,
) { unsafe {
    use std::arch::x86_64::*;
    let kchunks = in_features / 8;
    let ktail = in_features % 8;

    let mut o = start_row;
    while o + 1 < end_row {
        // process tokens in blocks of 4; tail (<4) handled by scalar dots
        let tblock = n_tokens - (n_tokens % 4);
        for tbase in (0..tblock).step_by(4) {
            // accumulators: acc[token][row] — 8 m256 registers (4 tokens × 2 rows)
            let mut acc = [[_mm256_setzero_ps(); 2]; 4];
            for k in 0..kchunks {
                let ki = k * 8;
                let w0 = load_w_m256(w, m, in_features, o, ki);
                let w1 = load_w_m256(w, m, in_features, o + 1, ki);
                for ti in 0..4 {
                    let xv = _mm256_loadu_ps(x.as_ptr().add((tbase + ti) * in_features + ki));
                    acc[ti][0] = _mm256_fmadd_ps(xv, w0, acc[ti][0]);
                    acc[ti][1] = _mm256_fmadd_ps(xv, w1, acc[ti][1]);
                }
            }
            for ti in 0..4 {
                let mut s0 = reduce_add(acc[ti][0]);
                let mut s1 = reduce_add(acc[ti][1]);
                for k in (kchunks * 8)..(kchunks * 8 + ktail) {
                    let xk = x[(tbase + ti) * in_features + k];
                    s0 += xk * wf_elem(w, m, in_features, o, k);
                    s1 += xk * wf_elem(w, m, in_features, o + 1, k);
                }
                y[(tbase + ti) * out_features + o] = s0 * w_row_scale(w, o);
                y[(tbase + ti) * out_features + o + 1] = s1 * w_row_scale(w, o + 1);
            }
        }
        // token tail (<4): reuse the same weight rows via scalar dots
        for t in tblock..n_tokens {
            let xv = &x[t * in_features..(t + 1) * in_features];
            y[t * out_features + o] = dot_w_row_avx2(xv, w, m, in_features, o);
            y[t * out_features + o + 1] = dot_w_row_avx2(xv, w, m, in_features, o + 1);
        }
        o += 2;
    }
    // odd trailing row
    if o < end_row {
        for t in 0..n_tokens {
            let xv = &x[t * in_features..(t + 1) * in_features];
            y[t * out_features + o] = dot_w_row_avx2(xv, w, m, in_features, o);
        }
    }
}}

/// dot(x, w[o,:]) with per-row scale applied (I8), via existing SIMD dots.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_w_row_avx2<R: WMatView>(x: &[f32], w: &WMat, m: &R, in_features: usize, o: usize) -> f32 {
    match w {
        WMat::F32(wf) => dot_f32_avx2(x, &wf[o * in_features..(o + 1) * in_features]),
        WMat::BF16(wb) => dot_bf16_avx2(x, &wb[o * in_features..(o + 1) * in_features]),
        WMat::View(off, len) => {
            if let Some(wb) = m.wmat_bytes(*off, *len) {
                let wb: &[u16] = bytemuck::cast_slice(wb);
                dot_bf16_avx2(x, &wb[o * in_features..(o + 1) * in_features])
            } else {
                0.0
            }
        }
        WMat::I8(q, s) => {
            // rare tail path; scalar is fine
            let mut acc = 0f32;
            for (xv, wq) in x.iter().zip(&q[o * in_features..(o + 1) * in_features]) {
                acc += xv * (*wq as f32);
            }
            acc * s[o]
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn w_row_scale(w: &WMat, o: usize) -> f32 {
    match w {
        WMat::F32(_) | WMat::BF16(_) | WMat::View(_, _) => 1.0,
        WMat::I8(_, s) => s[o],
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn load_w_m256<R: WMatView>(w: &WMat, m: &R, in_features: usize, o: usize, ki: usize) -> std::arch::x86_64::__m256 { unsafe {
    use std::arch::x86_64::*;
    match w {
        WMat::F32(wf) => _mm256_loadu_ps(wf.as_ptr().add(o * in_features + ki)),
        WMat::BF16(wb) => {
            let w128 = _mm_loadu_si128(wb.as_ptr().add(o * in_features + ki) as *const __m128i);
            let w256 = _mm256_cvtepu16_epi32(w128);
            _mm256_castsi256_ps(_mm256_slli_epi32(w256, 16))
        }
        WMat::View(off, len) => {
            if let Some(wb) = m.wmat_bytes(*off, *len) {
                let wb: &[u16] = bytemuck::cast_slice(wb);
                let w128 = _mm_loadu_si128(wb.as_ptr().add(o * in_features + ki) as *const __m128i);
                let w256 = _mm256_cvtepu16_epi32(w128);
                _mm256_castsi256_ps(_mm256_slli_epi32(w256, 16))
            } else {
                _mm256_setzero_ps()
            }
        }
        WMat::I8(q, _) => {
            // 8 x i8 -> 8 x i32 -> 8 x f32 (scale applied at writeback)
            let q64 = _mm_loadl_epi64(q.as_ptr().add(o * in_features + ki) as *const __m128i);
            let q32 = _mm256_cvtepi8_epi32(q64);
            _mm256_cvtepi32_ps(q32)
        }
    }
}}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn wf_elem<R: WMatView>(w: &WMat, m: &R, in_features: usize, o: usize, k: usize) -> f32 {
    match w {
        WMat::F32(wf) => wf[o * in_features + k],
        WMat::BF16(wb) => f32::from_bits((wb[o * in_features + k] as u32) << 16),
        WMat::View(off, len) => {
            if let Some(wb) = m.wmat_bytes(*off, *len) {
                let wb: &[u16] = bytemuck::cast_slice(wb);
                f32::from_bits((wb[o * in_features + k] as u32) << 16)
            } else {
                0.0
            }
        }
        WMat::I8(q, s) => q[o * in_features + k] as f32 * s[o],
    }
}

/// RMSNorm over each row of [T, n].
pub fn rms_norm_batch(x: &mut [f32], weight: &[f32], eps: f32, n: usize) {
    for row in x.chunks_exact_mut(n) {
        let sq_sum: f32 = row.iter().map(|v| v * v).sum();
        let inv = 1.0 / (sq_sum / n as f32 + eps).sqrt();
        for (v, w) in row.iter_mut().zip(weight.iter()) {
            *v *= inv * w;
        }
    }
}

/// RMSNorm: x / sqrt(mean(x^2) + eps) * weight
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let sq_sum: f32 = x.iter().map(|v| v * v).sum();
    let mean = sq_sum / x.len() as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    x.iter()
        .zip(weight.iter())
        .map(|(v, w)| v * inv * w)
        .collect()
}

/// RMSNorm writing into a caller-provided buffer (decode hot path, no alloc).
/// Same math and order as `rms_norm`.
pub fn rms_norm_into(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());
    let sq_sum: f32 = x.iter().map(|v| v * v).sum();
    let mean = sq_sum / x.len() as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    for (o, (v, w)) in out.iter_mut().zip(x.iter().zip(weight.iter())) {
        *o = v * inv * w;
    }
}

/// Apply RoPE rotation in-place to a single head's q or k vector at position `pos`.
/// Uses the Llama/HF "rotate_half" layout: pairs are (i, i + d/2).
#[allow(dead_code)] // f32 scalar fallback / used by tests

pub fn rope_rotate(v: &mut [f32], pos: usize, theta: f32) {
    let half = v.len() / 2;
    for i in 0..half {
        let freq = theta.powf(2.0 * i as f32 / v.len() as f32);
        let angle = pos as f32 * freq.recip();
        let (sin, cos) = angle.sin_cos();
        let a = v[i];
        let b = v[i + half];
        v[i] = a * cos - b * sin;
        v[i + half] = a * sin + b * cos;
    }
}

/// Precomputed RoPE tables: inverse frequencies + lazily-grown per-position cos/sin rows.
pub struct Rope {
    inv_freq: Vec<f32>,
    cos: Vec<f32>, // [pos][half] flattened, grown as positions are needed
    sin: Vec<f32>,
    half: usize,
}

impl Rope {
    pub fn new(head_dim: usize, theta: f32) -> Self {
        let half = head_dim / 2;
        let inv_freq = (0..half)
            .map(|i| theta.powf(2.0 * i as f32 / head_dim as f32).recip())
            .collect();
        Self { inv_freq, cos: Vec::new(), sin: Vec::new(), half }
    }

    /// Ensure cos/sin tables cover positions 0..pos_needed.
    fn ensure(&mut self, pos_needed: usize) {
        while self.cos.len() / self.half <= pos_needed {
            let pos = self.cos.len() / self.half;
            for i in 0..self.half {
                let (s, c) = (pos as f32 * self.inv_freq[i]).sin_cos();
                self.cos.push(c);
                self.sin.push(s);
            }
        }
    }

    /// Rotate a head vector in-place at position `pos` using precomputed tables.
    pub fn rotate(&mut self, v: &mut [f32], pos: usize) {
        self.ensure(pos);
        let half = self.half;
        let c = &self.cos[pos * half..(pos + 1) * half];
        let s = &self.sin[pos * half..(pos + 1) * half];
        for i in 0..half {
            let a = v[i];
            let b = v[i + half];
            v[i] = a * c[i] - b * s[i];
            v[i + half] = a * s[i] + b * c[i];
        }
    }
}

/// SwiGLU MLP: down( silu(gate(x)) * up(x) ) computed per-element; returns silu(gate)*up (caller applies down proj).
#[allow(dead_code)] // f32 scalar fallback / used by tests

pub fn silu(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x / (1.0 + (-x).exp())).collect()
}

/// elementwise multiply
#[allow(dead_code)] // f32 scalar fallback / used by tests

pub fn mul_elem(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Numerically stable softmax over a slice.
pub fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Argmax (ties broken by lowest index).
pub fn argmax(x: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, v) in x.iter().enumerate() {
        if *v > best_val {
            best_val = *v;
            best = i;
        }
    }
    best
}

/// Scale-and-mask softmax attention for one query row over cached keys.
/// scores[i] = dot(q, k_i) / sqrt(d); masked positions given -inf; returns attention output vector.
#[allow(dead_code)] // f32 scalar fallback / used by tests

pub fn attention_row(q: &[f32], keys: &[Vec<f32>], values: &[Vec<f32>], mask_len: usize, scale: f32) -> Vec<f32> {
    let scores: Vec<f32> = keys
        .iter()
        .take(mask_len)
        .map(|k| q.iter().zip(k.iter()).map(|(a, b)| a * b).sum::<f32>() * scale)
        .collect();
    let probs = softmax(&scores);
    let mut out = vec![0.0f32; q.len()];
    for (p, v) in probs.iter().zip(values.iter()) {
        for (o, vv) in out.iter_mut().zip(v.iter()) {
            *o += p * vv;
        }
    }
    out
}

/// Causal attention for a batch of T query rows (positions pos..pos+T) over cached keys
/// (all previous + these T). q_flat is [T, n_heads*head_dim]; keys/values are flat
/// [pos+T, n_kv*head_dim] cache slices (keys already RoPE'd and appended by the caller).
/// Returns [T, n_heads*head_dim].
pub fn attention_batch(
    q_flat: &[f32],
    keys_flat: &[f32],
    values_flat: &[f32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    start_pos: usize,
) -> Vec<f32> {
    let t = q_flat.len() / (n_heads * head_dim);
    let group = n_heads / n_kv_heads;
    let kv_dim = n_kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; t * n_heads * head_dim];
    for ti in 0..t {
        let pos = start_pos + ti;
        let q_row = &q_flat[ti * n_heads * head_dim..(ti + 1) * n_heads * head_dim];
        for hd in 0..n_heads {
            let kv_head = hd / group;
            let s = kv_head * head_dim;
            let q = &q_row[hd * head_dim..(hd + 1) * head_dim];
            // scores over positions 0..=pos
            let n_ctx = pos + 1;
            let mut scores = vec![0.0f32; n_ctx];
            for cpos in 0..n_ctx {
                let k = &keys_flat[cpos * kv_dim + s..cpos * kv_dim + s + head_dim];
                scores[cpos] = dot(q, k) * scale;
            }
            let probs = softmax(&scores);
            let o = &mut out[ti * n_heads * head_dim + hd * head_dim
                ..ti * n_heads * head_dim + (hd + 1) * head_dim];
            for (cpos, p) in probs.iter().enumerate() {
                let v = &values_flat[cpos * kv_dim + s..cpos * kv_dim + s + head_dim];
                for (oo, vv) in o.iter_mut().zip(v.iter()) {
                    *oo += p * vv;
                }
            }
        }
    }
    out
}

/// Apply repetition penalty in-place to logits (CTRL-style: divide positive, multiply negative).
pub fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
    if penalty == 1.0 {
        return;
    }
    let mut seen = std::collections::HashSet::new();
    for &t in generated {
        seen.insert(t as usize);
    }
    for &t in &seen {
        let l = &mut logits[t];
        *l = if *l > 0.0 { *l / penalty } else { *l * penalty };
    }
}

/// Sample a token from logits given sampling parameters. temperature==0 -> greedy argmax.
pub fn sample(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut impl Rng,
) -> usize {
    if temperature <= 0.0 {
        return argmax(logits);
    }
    // temperature scaling
    let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();
    let mut idx: Vec<usize> = (0..scaled.len()).collect();

    // top-k
    if top_k > 0 && top_k < idx.len() {
        idx.sort_unstable_by(|&a, &b| scaled[b].partial_cmp(&scaled[a]).unwrap());
        idx.truncate(top_k);
    }

    // softmax over candidates
    let cand: Vec<f32> = idx.iter().map(|&i| scaled[i]).collect();
    let mut probs = softmax(&cand);

    // top-p (nucleus)
    if top_p < 1.0 {
        let mut order: Vec<usize> = (0..idx.len()).collect();
        order.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let mut cum = 0.0f32;
        let mut cutoff = order.len();
        for (rank, &i) in order.iter().enumerate() {
            cum += probs[i];
            if cum > top_p {
                cutoff = rank + 1;
                break;
            }
        }
        let keep: Vec<usize> = order[..cutoff].to_vec();
        let mut new_probs = vec![0.0f32; idx.len()];
        let sum: f32 = keep.iter().map(|&i| probs[i]).sum();
        for &i in &keep {
            new_probs[i] = probs[i] / sum;
        }
        probs = new_probs;
    }

    // sample
    let r: f32 = rand::distr::StandardUniform.sample(rng);
    let mut cum = 0.0f32;
    for (i, p) in probs.iter().enumerate() {
        cum += p;
        if r < cum {
            return idx[i];
        }
    }
    idx[probs.iter().position(|&p| p > 0.0).unwrap_or(0)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() < tol, "a={a} b={b}");
    }

    #[test]
    fn matmul_identity() {
        // W = I (2x2) stored [out=2, in=2]
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let x = vec![3.0, -4.0];
        let y = matmul(&x, &w, 2, 2);
        assert_eq!(y, vec![3.0, -4.0]);
    }

    #[test]
    fn matmul_linear_weights() {
        // HF nn.Linear: y = x W^T, weight [out, in].
        // out0 = 1*x0 + 2*x1 ; out1 = 3*x0 + 4*x1
        let w = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let y = matmul(&x, &w, 2, 2);
        assert_eq!(y, vec![3.0, 7.0]);
    }

    #[test]
    fn rms_norm_known() {
        // x = [3, 4]: mean(x^2) = 12.5; inv = 1/sqrt(12.5+0)
        let w = vec![1.0, 1.0];
        let y = rms_norm(&[3.0, 4.0], &w, 0.0);
        let inv = 1.0 / 12.5f32.sqrt();
        approx(y[0], 3.0 * inv, 1e-6);
        approx(y[1], 4.0 * inv, 1e-6);
    }

    #[test]
    fn rms_norm_weighted() {
        let w = vec![2.0, 0.5];
        let y = rms_norm(&[3.0, 4.0], &w, 1e-6);
        let inv = 1.0 / (12.5f32 + 1e-6).sqrt();
        approx(y[0], 3.0 * inv * 2.0, 1e-6);
        approx(y[1], 4.0 * inv * 0.5, 1e-6);
    }

    #[test]
    fn rope_preserves_norm() {
        let mut v: Vec<f32> = (0..8).map(|i| (i as f32) * 0.5 - 1.75).collect();
        let before: f32 = v.iter().map(|x| x * x).sum();
        rope_rotate(&mut v, 17, 5000000.0);
        let after: f32 = v.iter().map(|x| x * x).sum();
        approx(before, after, 1e-4);
    }

    #[test]
    fn rope_zero_pos_identity() {
        let mut v = vec![1.0, 2.0, 3.0, 4.0];
        rope_rotate(&mut v, 0, 10000.0);
        assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn rope_matches_reference() {
        // Half-split layout (Llama rotate_half): pairs are (i, i+half).
        // pos=1, theta=10000, dim=4: angle_0 = 1, angle_1 = 10000^(-2/4)=0.01
        let mut v = vec![1.0, 0.0, 0.0, 1.0];
        rope_rotate(&mut v, 1, 10000.0);
        let a0 = 1.0f32.cos();
        let b0 = 1.0f32.sin();
        // pair 0: (v[0], v[2]) = (1, 0)
        approx(v[0], a0, 1e-5);
        approx(v[2], b0, 1e-5);
        // pair 1: (v[1], v[3]) = (0, 1)
        let ang = 0.01f32;
        approx(v[1], 0.0 * ang.cos() - 1.0 * ang.sin(), 1e-5);
        approx(v[3], 0.0 * ang.sin() + 1.0 * ang.cos(), 1e-5);
    }

    #[test]
    fn silu_known() {
        approx(silu(&[0.0])[0], 0.0, 1e-6);
        approx(silu(&[1.0])[0], 1.0 / (1.0 + (-1.0f32).exp()), 1e-6);
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        approx(p.iter().sum(), 1.0, 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn argmax_basic() {
        assert_eq!(argmax(&[0.1, 3.0, 2.0]), 1);
        // tie broken by lowest index
        assert_eq!(argmax(&[2.0, 2.0]), 0);
    }

    #[test]
    fn attention_row_shapes() {
        let q = vec![1.0, 0.0];
        let keys = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let values = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let out = attention_row(&q, &keys, &values, 2, 1.0 / 2f32.sqrt());
        // dot scores: [1, 0] -> probs [~0.731, ~0.269]
        let p = softmax(&[1.0f32 / 2f32.sqrt(), 0.0]);
        approx(out[0], p[0] * 1.0 + p[1] * 3.0, 1e-5);
        approx(out[1], p[0] * 2.0 + p[1] * 4.0, 1e-5);
    }

    #[test]
    fn repetition_penalty_applied() {
        let mut logits = vec![2.0, -2.0, 1.0];
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0);
        assert_eq!(logits[0], 1.0); // 2/2
        assert_eq!(logits[1], -4.0); // -2*2
        assert_eq!(logits[2], 1.0); // untouched
    }

    #[test]
    fn greedy_sample_is_argmax() {
        let mut rng = rand::rng();
        assert_eq!(sample(&[0.1, 5.0, 3.0], 0.0, 0, 1.0, &mut rng), 1);
    }

    #[test]
    fn temperature_sample_deterministic_with_seed() {
        use rand::SeedableRng;
        let mut r1 = rand::rngs::StdRng::seed_from_u64(123);
        let mut r2 = rand::rngs::StdRng::seed_from_u64(123);
        let logits = vec![0.1, 0.2, 0.3, 0.4, 5.0];
        let a = sample(&logits, 1.0, 0, 1.0, &mut r1);
        let b = sample(&logits, 1.0, 0, 1.0, &mut r2);
        assert_eq!(a, b);
        // strongly peaked -> should pick 4 most of the time
        assert_eq!(a, 4);
    }

    #[test]
    fn top_k_restricts_candidates() {
        use rand::SeedableRng;
        // top_k=2: only indices 1 and 3 (values 5.0 and 4.0) reachable
        let logits = vec![0.0, 5.0, 1.0, 4.0, 2.0];
        for seed in 0..50u64 {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let t = sample(&logits, 1.0, 2, 1.0, &mut rng);
            assert!(t == 1 || t == 3, "got {t}");
        }
    }
}
