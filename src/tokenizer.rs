use anyhow::{Context, Result};
use std::path::Path;

/// Wraps the `tokenizers` crate for encode/decode.
pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
}

impl Tokenizer {
    pub fn from_file(path: &Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer from {}: {e}", path.display()))?;
        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        self.inner
            .encode(text, add_special_tokens)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("detokenization failed: {e}"))
    }

    #[allow(dead_code)] // vocab_size kept for completeness/debugging
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }
}

/// Resolve tokenizer path: explicit --tokenizer flag, or tokenizer.json next to the model file.
pub fn resolve_tokenizer_path(explicit: Option<&str>, model_path: &Path) -> Result<std::path::PathBuf> {
    if let Some(p) = explicit {
        return Ok(std::path::PathBuf::from(p));
    }
    let dir = model_path
        .parent()
        .with_context(|| format!("model path {} has no parent dir", model_path.display()))?;
    Ok(dir.join("tokenizer.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_explicit() {
        let p = resolve_tokenizer_path(Some("/x/t.json"), Path::new("/y/m.safetensors")).unwrap();
        assert_eq!(p, std::path::PathBuf::from("/x/t.json"));
    }
}
