#[path = "../src/config.rs"] pub mod config;
#[path = "../src/safetensors_loader.rs"] pub mod safetensors_loader;
#[path = "../src/model.rs"] pub mod model;
#[path = "../src/tensor.rs"] pub mod tensor;
#[path = "../src/tmb.rs"] pub mod tmb;

use model::{KvCache, Model};
use std::path::Path;

#[test]
fn forward_seq_matches_forward_token() {
    if !Path::new("/tmp/tinyminicpm5/model.safetensors").exists() {
        eprintln!("debug model not downloaded; skipping");
        return;
    }
    let mut m = Model::load(Path::new("/tmp/tinyminicpm5/model.safetensors")).unwrap();
    let ids: Vec<u32> = vec![0, 130072, 8448, 220, 2928, 357, 285, 4894, 304, 6918, 52, 130073, 220, 130072, 130071, 220];

    // path A: token-by-token
    let mut c1 = KvCache::new(m.w.config.num_hidden_layers);
    c1.set_kv_dim(m.w.config.num_key_value_heads * m.w.config.head_dim);
    let mut last_a = Vec::new();
    for &t in &ids {
        last_a = m.forward_token(t, &mut c1).unwrap();
    }

    // path B: batched
    let mut c2 = KvCache::new(m.w.config.num_hidden_layers);
    c2.set_kv_dim(m.w.config.num_key_value_heads * m.w.config.head_dim);
    let last_b = m.forward_seq(&ids, &mut c2).unwrap();

    let mut max_diff: f32 = 0.0;
    for (a, b) in last_a.iter().zip(last_b.iter()) {
        max_diff = max_diff.max((a - b).abs());
    }
    eprintln!("max logit diff: {max_diff}");
    // AVX2+FMA reassociates the accumulation, so paths differ by ~5e-3 on the
    // random-weight debug model; greedy outputs still match HF exactly.
    assert!(max_diff < 1e-2, "max diff {max_diff}");
}

#[test]
fn argmax_debug() {
    if !Path::new("/tmp/tinyminicpm5/model.safetensors").exists() { return; }
    let mut m = Model::load(Path::new("/tmp/tinyminicpm5/model.safetensors")).unwrap();
    let ids: Vec<u32> = vec![0, 130072, 8448, 220, 2928, 357, 285, 4894, 304, 6918, 52, 130073, 220, 130072, 130071, 220];
    let mut c = KvCache::new(m.w.config.num_hidden_layers);
    c.set_kv_dim(m.w.config.num_key_value_heads * m.w.config.head_dim);
    let logits = m.forward_seq(&ids, &mut c).unwrap();
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
    eprintln!("top5: {:?}", &idx[..5]);
    eprintln!("top1 val: {}", logits[idx[0]]);
    // also via forward_token path
    let mut c2 = KvCache::new(m.w.config.num_hidden_layers);
    c2.set_kv_dim(m.w.config.num_key_value_heads * m.w.config.head_dim);
    let mut l2 = Vec::new();
    for &t in &ids { l2 = m.forward_token(t, &mut c2).unwrap(); }
    let mut idx2: Vec<usize> = (0..l2.len()).collect();
    idx2.sort_by(|&a, &b| l2[b].partial_cmp(&l2[a]).unwrap());
    eprintln!("tok-path top5: {:?}", &idx2[..5]);
}

#[test]
fn generate_replica() {
    if !Path::new("/tmp/tinyminicpm5/model.safetensors").exists() { return; }
    let mut m = Model::load(Path::new("/tmp/tinyminicpm5/model.safetensors")).unwrap();
    let ids: Vec<u32> = vec![0, 130072, 8448, 220, 2928, 357, 285, 4894, 304, 6918, 52, 130073, 220, 130072, 130071, 220];
    let params = model::GenParams {
        max_tokens: 40, temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: 42,
    };
    let out = model::generate(&mut m, &ids, &params, |_| {}).unwrap();
    eprintln!("first 5 tokens: {:?}", &out.tokens[..5]);
}
