//! Model configuration (LlamaForCausalLM-style, as used by MiniCPM5).

use anyhow::{Context, Result, anyhow};
use std::path::Path;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub vocab_size: usize,
    pub eos_token_ids: Vec<u32>,
    #[allow(dead_code)] // parsed for completeness (BOS embedding index)
    pub bos_token_id: u32,
}

fn get_usize(cfg: &serde_json::Value, key: &str) -> Result<usize> {
    cfg.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .ok_or_else(|| anyhow!("config missing integer field `{key}`"))
}

fn get_f32(cfg: &serde_json::Value, key: &str, default: f32) -> f32 {
    cfg.get(key)
        .and_then(|v| v.as_f64())
        .map(|v| v as f32)
        .unwrap_or(default)
}

impl ModelConfig {
    /// Build a config from an already-parsed JSON value (used by the TMB
    /// loader, which embeds the config in the repacked file).
    pub fn from_json(cfg: &serde_json::Value) -> Result<Self> {
        let hidden_size = get_usize(cfg, "hidden_size")?;
        let num_attention_heads = get_usize(cfg, "num_attention_heads")?;
        let num_key_value_heads = cfg
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);
        let head_dim = cfg
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_attention_heads);

        let rope_theta = cfg
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .or_else(|| {
                cfg.get("rope_parameters")
                    .and_then(|r| r.get("rope_theta"))
                    .and_then(|v| v.as_f64())
            })
            .map(|v| v as f32)
            .unwrap_or(10000.0);

        let eos_token_ids = match cfg.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => {
                vec![n.as_u64().context("bad eos_token_id")? as u32]
            }
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect(),
            _ => vec![],
        };

        Ok(Self {
            hidden_size,
            num_hidden_layers: get_usize(cfg, "num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size: get_usize(cfg, "intermediate_size")?,
            rms_norm_eps: get_f32(cfg, "rms_norm_eps", 1e-5),
            rope_theta,
            vocab_size: get_usize(cfg, "vocab_size")?,
            eos_token_ids,
            bos_token_id: cfg
                .get("bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(0),
        })
    }

    /// Serialize the runtime-relevant fields to JSON (used by the TMB
    /// repack tool; the loader parses it back via `from_json`).
    pub fn to_json(&self) -> Result<serde_json::Value> {
        Ok(serde_json::json!({
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "vocab_size": self.vocab_size,
            "eos_token_ids": self.eos_token_ids,
            "bos_token_id": self.bos_token_id,
        }))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let cfg: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;

        let hidden_size = get_usize(&cfg, "hidden_size")?;
        let num_attention_heads = get_usize(&cfg, "num_attention_heads")?;
        let num_key_value_heads = cfg
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(num_attention_heads);
        let head_dim = cfg
            .get("head_dim")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_attention_heads);

        // rope_theta may be top-level or nested under rope_parameters
        let rope_theta = cfg
            .get("rope_theta")
            .and_then(|v| v.as_f64())
            .or_else(|| {
                cfg.get("rope_parameters")
                    .and_then(|r| r.get("rope_theta"))
                    .and_then(|v| v.as_f64())
            })
            .map(|v| v as f32)
            .unwrap_or(10000.0);

        // eos may be a single id or a list
        let eos_token_ids = match cfg.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => {
                vec![n.as_u64().context("bad eos_token_id")? as u32]
            }
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .filter_map(|v| v.as_u64().map(|x| x as u32))
                .collect(),
            _ => vec![],
        };

        Ok(Self {
            hidden_size,
            num_hidden_layers: get_usize(&cfg, "num_hidden_layers")?,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size: get_usize(&cfg, "intermediate_size")?,
            rms_norm_eps: get_f32(&cfg, "rms_norm_eps", 1e-5),
            rope_theta,
            vocab_size: get_usize(&cfg, "vocab_size")?,
            eos_token_ids,
            bos_token_id: cfg
                .get("bos_token_id")
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .unwrap_or(0),
        })
    }

    pub fn config_path_for_model(model_path: &Path) -> Option<std::path::PathBuf> {
        let dir = model_path.parent()?;
        let candidates = ["config.json"];
        for c in candidates {
            let p = dir.join(c);
            if p.exists() {
                return Some(p);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tiny_random_minicpm5_config() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 0,
            "eos_token_id": [1, 130073],
            "head_dim": 32,
            "hidden_size": 16,
            "intermediate_size": 64,
            "model_type": "llama",
            "num_attention_heads": 16,
            "num_hidden_layers": 2,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_parameters": {"rope_theta": 5000000.0, "rope_type": "default"},
            "tie_word_embeddings": false,
            "vocab_size": 130560
        }"#;
        let path = std::env::temp_dir().join("tinyminigo-test-config.json");
        std::fs::write(&path, json).unwrap();
        let c = ModelConfig::load(&path).unwrap();
        assert_eq!(c.hidden_size, 16);
        assert_eq!(c.num_hidden_layers, 2);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_key_value_heads, 2);
        assert_eq!(c.head_dim, 32);
        assert_eq!(c.intermediate_size, 64);
        assert_eq!(c.rms_norm_eps, 1e-6);
        assert_eq!(c.rope_theta, 5_000_000.0);
        assert_eq!(c.vocab_size, 130560);
        assert_eq!(c.eos_token_ids, vec![1, 130073]);
        assert_eq!(c.bos_token_id, 0);
        // GQA group size: 16 Q heads / 2 KV heads = 8
        assert_eq!(c.num_attention_heads / c.num_key_value_heads, 8);
    }
}
