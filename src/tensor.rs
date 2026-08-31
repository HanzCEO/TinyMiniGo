//! Core tensor math for Llama-style (MiniCPM5) inference, hand-rolled on f32 slices.

use rand::{Rng, distr::Distribution};

/// y = x @ W^T where W is [out, in] row-major (HF convention: nn.Linear stores weight [out, in]).
pub fn matmul(x: &[f32], w: &[f32], out_features: usize, in_features: usize) -> Vec<f32> {
    assert_eq!(x.len(), in_features, "matmul input dim mismatch");
    let mut y = vec![0.0f32; out_features];
    for (o, y_o) in y.iter_mut().enumerate() {
        let row = &w[o * in_features..(o + 1) * in_features];
        let mut acc = 0.0f32;
        for (xi, wi) in x.iter().zip(row.iter()) {
            acc += xi * wi;
        }
        *y_o = acc;
    }
    y
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

/// Apply RoPE rotation in-place to a single head's q or k vector at position `pos`.
/// Uses the Llama/HF "rotate_half" layout: pairs are (i, i + d/2).
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

/// SwiGLU MLP: down( silu(gate(x)) * up(x) ) computed per-element; returns silu(gate)*up (caller applies down proj).
pub fn silu(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| x / (1.0 + (-x).exp())).collect()
}

/// elementwise multiply
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
