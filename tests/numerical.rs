#[path = "../src/config.rs"] pub mod config;
#[path = "../src/safetensors_loader.rs"] pub mod safetensors_loader;
#[path = "../src/model.rs"] pub mod model;
#[path = "../src/tensor.rs"] pub mod tensor;

use model::{KvCache, Model};
use std::path::Path;

fn model_available() -> bool {
    Path::new("/tmp/tinyminicpm5/model.safetensors").exists()
}

#[test]
fn layer0_o_proj_matches_reference() {
    if !model_available() {
        eprintln!("debug model not downloaded; skipping");
        return;
    }
    let m = Model::load(Path::new("/tmp/tinyminicpm5/model.safetensors")).unwrap();
    let c = &m.w.config;
    let h = c.hidden_size;
    let lw = &m.w.layers[0];

    let x0: Vec<f32> = m.w.embed[0..h].to_vec();
    let ln1 = tensor::rms_norm(&x0, &lw.input_layernorm, c.rms_norm_eps);
    let q = tensor::matmul(&ln1, &lw.wq, c.num_attention_heads * c.head_dim, h);
    let k = tensor::matmul(&ln1, &lw.wk, c.num_key_value_heads * c.head_dim, h);
    let v = tensor::matmul(&ln1, &lw.wv, c.num_key_value_heads * c.head_dim, h);

    let n_q = c.num_attention_heads;
    let n_kv = c.num_key_value_heads;
    let mut cache = KvCache::new(1);
    let mut k_flat = k.clone();
    for hd in 0..n_kv {
        let kv = &mut k_flat[hd * c.head_dim..(hd + 1) * c.head_dim];
        tensor::rope_rotate(kv, 0, c.rope_theta);
    }
    cache.push(0, k_flat, v);

    let scale = 1.0 / (c.head_dim as f32).sqrt();
    let group = n_q / n_kv;
    let mut attn = vec![0.0f32; n_q * c.head_dim];
    for hd in 0..n_q {
        let mut qv = q[hd * c.head_dim..(hd + 1) * c.head_dim].to_vec();
        tensor::rope_rotate(&mut qv, 0, c.rope_theta);
        let kv_head = hd / group;
        let s = kv_head * c.head_dim;
        let keys: Vec<Vec<f32>> = cache.keys[0]
            .iter()
            .map(|k| k[s..s + c.head_dim].to_vec())
            .collect();
        let vals: Vec<Vec<f32>> = cache.values[0]
            .iter()
            .map(|vv| vv[s..s + c.head_dim].to_vec())
            .collect();
        let out = tensor::attention_row(&qv, &keys, &vals, keys.len(), scale);
        attn[hd * c.head_dim..(hd + 1) * c.head_dim].copy_from_slice(&out);
    }
    let o = tensor::matmul(&attn, &lw.wo, h, n_q * c.head_dim);

    // HF reference (scripts/hf_layer_debug.py): o_out first 8
    let hf = [0.363_092_1f32, -0.230_118_9, 0.818_211, 0.340_060_1,
              -1.512_215_6, 1.409_458_2, 2.533_720_7, 0.920_836_9];
    for i in 0..8 {
        assert!(
            (o[i] - hf[i]).abs() < 1e-3,
            "o[{i}] = {} vs hf {} (diff {})",
            o[i], hf[i], (o[i] - hf[i]).abs()
        );
    }
}

#[test]
fn full_prompt_first_step_logits_match_hf() {
    if !model_available() {
        eprintln!("debug model not downloaded; skipping");
        return;
    }
    let mut m = Model::load(Path::new("/tmp/tinyminicpm5/model.safetensors")).unwrap();
    let mut cache = KvCache::new(m.w.config.num_hidden_layers);
    // Prompt ids from HF apply_chat_template for "What is the capital of France?"
    let ids: [u32; 16] = [0, 130072, 8448, 220, 2928, 357, 285, 4894, 304, 6918, 52, 130073, 220, 130072, 130071, 220];
    let mut logits = vec![];
    for &t in &ids {
        logits = m.forward_token(t, &mut cache).unwrap();
    }
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    // HF reference top-5 for this prompt
    let hf_top5 = [104125usize, 109925, 91900, 73380, 58475];
    assert_eq!(&idx[..5], &hf_top5[..]);
    let hf_first8 = [0.122487f32, -0.088185, 0.180103, 0.016682, -0.053285, 0.201443, -0.055636, -0.195765];
    for i in 0..8 {
        assert!((logits[i] - hf_first8[i]).abs() < 1e-5, "logit[{i}] {} vs {}", logits[i], hf_first8[i]);
    }
}

#[test]
fn greedy_generation_multi_step_matches_hf() {
    if !model_available() {
        eprintln!("debug model not downloaded; skipping");
        return;
    }
    let mut m = Model::load(Path::new("/tmp/tinyminicpm5/model.safetensors")).unwrap();
    let ids: Vec<u32> = vec![0, 130072, 8448, 220, 2928, 357, 285, 4894, 304, 6918, 52, 130073, 220, 130072, 130071, 220];
    let params = model::GenParams {
        max_tokens: 40,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 42,
    };
    let out = model::generate(&mut m, &ids, &params, |_| {}).unwrap();
    // HF greedy generation, 40 steps
    let hf_gen: [u32; 40] = [104125, 65126, 93, 104125, 65126, 37023, 23022, 49608, 65126, 37023,
        23022, 26077, 65126, 37023, 37023, 109925, 65126, 37023, 37023, 37023,
        37023, 37023, 109925, 65126, 37023, 37023, 37023, 37023, 109925, 65126,
        37023, 37023, 37023, 37023, 37023, 109925, 65126, 37023, 37023, 37023];
    assert_eq!(out.tokens.len(), 40);
    for (i, (a, b)) in out.tokens.iter().zip(hf_gen.iter()).enumerate() {
        assert_eq!(a, b, "step {i}: rust {a} vs hf {b}");
    }
}
