//! TMB (TinyMiniGo Model Binary) — load-optimized repack of the fixed checkpoint.
//!
//! Rationale: the model weights never change, so conversion cost is paid once
//! offline instead of every launch. A TMB file is a single binary blob:
//!   magic "TMB1" | config (JSON, length-prefixed) | weights section | index
//!   | metadata trailer (index offsets, lengths, tensor count, section sizes)
//! Weights are laid out in *consumption order* so the loader maps the file and
//! borrows every tensor directly from the mapping — zero per-tensor copies,
//! and embed/norms are stored pre-converted to f32 (the only conversion the
//! runtime needs), also borrowed in place.
//!
//! Alignment: all tensors are 8-byte aligned in the file (and 16-byte
//! relative to the page-mapped base, which is itself page-aligned) so BF16 and
//! f32 views are always correctly aligned.

use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use crate::config::ModelConfig;
use crate::safetensors_loader::Tensor;

/// Magic + version for the repacked format.
pub const MAGIC: &[u8; 4] = b"TMB1";
/// Tensors are aligned to 8 bytes in the file; page-mapped base adds 16.
pub const TENSOR_ALIGN: usize = 8;

/// Describes one tensor's location in the mapped file (borrowed view).
#[derive(Clone, Copy, Debug)]
pub struct TensorRef {
    /// Byte offset of the tensor data within the file.
    pub offset: u64,
    /// Byte length of the tensor data.
    pub len: u64,
}

impl TensorRef {
    /// Borrow the tensor data as raw bytes from a memory map.
    pub fn bytes<'m>(&self, map: &'m [u8]) -> Result<&'m [u8]> {
        let start = self.offset as usize;
        let end = start + self.len as usize;
        map.get(start..end)
            .ok_or_else(|| anyhow!("TMB tensor range out of bounds"))
    }

    /// Borrow the tensor data as a `[u16]` BF16 view (data is 2-byte aligned).
    pub fn bf16<'m>(&self, map: &'m [u8]) -> Result<&'m [u16]> {
        let b = self.bytes(map)?;
        if b.len() % 2 != 0 {
            bail!("TMB BF16 tensor has odd byte length");
        }
        // Safe: file offsets are 8-byte aligned (TENSOR_ALIGN) and the map
        // base is page-aligned, so the slice start is at least 2-aligned.
        Ok(bytemuck::cast_slice(b))
    }

    /// Borrow the tensor data as a `[f32]` view (data is 4-byte aligned).
    pub fn f32<'m>(&self, map: &'m [u8]) -> Result<&'m [f32]> {
        let b = self.bytes(map)?;
        if b.len() % 4 != 0 {
            bail!("TMB f32 tensor has non-multiple-of-4 byte length");
        }
        Ok(bytemuck::cast_slice(b))
    }
}

/// The parsed in-memory TMB index (borrows nothing; plain offsets).
#[derive(Debug, Default)]
pub struct TmbIndex {
    pub tensors: HashMap<String, TensorRef>,
}

/// Parsed TMB header: config + tensor index (no tensor data copied).
pub struct TmbHeader {
    pub config: ModelConfig,
    pub index: TmbIndex,
}

/// Parse the TMB file header + index from a memory map.
pub fn parse_tmb_header(map: &[u8]) -> Result<TmbHeader> {
    if map.len() < 16 {
        bail!("TMB file too small");
    }
    if &map[0..4] != MAGIC {
        bail!("not a TMB1 file (wrong magic)");
    }
    let cfg_len = u64::from_le_bytes(map[4..12].try_into()?) as usize;
    let n_tensors = u32::from_le_bytes(map[12..16].try_into()?) as usize;

    let mut pos = 16usize;
    let cfg_end = pos + cfg_len;
    let cfg_json: serde_json::Value =
        serde_json::from_slice(&map[pos..cfg_end]).context("TMB config JSON")?;
    pos = cfg_end;

    // Weights section: weights_len field, then padding to TENSOR_ALIGN, then
    // the weights blob (skipped; tensors are addressed by absolute offsets).
    let weights_len = u64::from_le_bytes(map[pos..pos + 8].try_into()?);
    let weights_off = align_up(pos + 8, TENSOR_ALIGN);
    pos = weights_off + weights_len as usize;

    let mut tensors = HashMap::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        // name: (len u32, utf8)
        let name_len = u32::from_le_bytes(map[pos..pos + 4].try_into()?) as usize;
        pos += 4;
        let name = std::str::from_utf8(&map[pos..pos + name_len])
            .context("TMB tensor name utf8")?
            .to_string();
        pos += name_len;
        let offset = u64::from_le_bytes(map[pos..pos + 8].try_into()?);
        let len = u64::from_le_bytes(map[pos + 8..pos + 16].try_into()?);
        pos += 16;
        tensors.insert(name, TensorRef { offset, len });
    }
    Ok(TmbHeader {
        config: ModelConfig::from_json(&cfg_json)?,
        index: TmbIndex { tensors },
    })
}

// ---------------------------------------------------------------------------
// Offline repack tool
// ---------------------------------------------------------------------------

/// Load the source safetensors (all tensors into memory) — used only by the
/// one-time repack tool, never by the runtime load path.
pub fn load_safetensors_for_repack(path: &Path) -> Result<HashMap<String, Tensor>> {
    crate::safetensors_loader::load_safetensors(path)
}

/// Write a TMB1 file from the source safetensors + config.
/// Weights are emitted in consumption order: embed, layer by layer
/// (q,k,v,o,gate,up,down, input_norm, post_norm), final_norm, lm_head.
/// BF16 weights are copied raw; embed/norms are converted to f32 once here.
pub fn write_tmb(
    src: &Path,
    dst: &Path,
    config: &ModelConfig,
) -> Result<()> {
    let tensors = load_safetensors_for_repack(src)?;
    let mut weights: Vec<u8> = Vec::with_capacity(2_200_000_000);
    let mut index: Vec<(String, TensorRef)> = Vec::with_capacity(256);

    let push_bf16 = |name: &str, t: &Tensor, weights: &mut Vec<u8>| -> Result<(String, TensorRef)> {
        let b = t.bf16_bytes();
        let off = align_up(weights.len(), TENSOR_ALIGN);
        weights.resize(off, 0);
        weights.extend_from_slice(b);
        Ok((name.to_string(), TensorRef { offset: off as u64, len: b.len() as u64 }))
    };
    let push_f32 = |name: &str, t: &Tensor, weights: &mut Vec<u8>| -> Result<(String, TensorRef)> {
        let v: Vec<f32> = if t.is_bf16() {
            t.bf16_bits().iter().map(|w| f32::from_bits((*w as u32) << 16)).collect()
        } else {
            t.as_f32().to_vec()
        };
        let bytes = bytemuck::cast_slice(&v);
        let off = align_up(weights.len(), TENSOR_ALIGN);
        weights.resize(off, 0);
        weights.extend_from_slice(bytes);
        Ok((name.to_string(), TensorRef { offset: off as u64, len: bytes.len() as u64 }))
    };

    // embed (f32)
    let t = tensors.get("model.embed_tokens.weight").context("missing embed_tokens.weight")?;
    index.push(push_f32("model.embed_tokens.weight", t, &mut weights)?);

    let n_layers = config.num_hidden_layers;
    for l in 0..n_layers {
        let p = |s: &str| format!("model.layers.{l}.{s}");
        for (name, bf16) in [
            ("self_attn.q_proj.weight", true),
            ("self_attn.k_proj.weight", true),
            ("self_attn.v_proj.weight", true),
            ("self_attn.o_proj.weight", true),
            ("mlp.gate_proj.weight", true),
            ("mlp.up_proj.weight", true),
            ("mlp.down_proj.weight", true),
            ("input_layernorm.weight", false),
            ("post_attention_layernorm.weight", false),
        ] {
            let full = p(name);
            let t = tensors.get(&full).with_context(|| format!("missing {full}"))?;
            if bf16 {
                index.push(push_bf16(&full, t, &mut weights)?);
            } else {
                index.push(push_f32(&full, t, &mut weights)?);
            }
        }
    }

    // final norm (f32)
    let t = tensors.get("model.norm.weight").context("missing model.norm.weight")?;
    index.push(push_f32("model.norm.weight", t, &mut weights)?);

    // lm_head (BF16 raw; present in MiniCPM5)
    if let Some(t) = tensors.get("lm_head.weight") {
        index.push(push_bf16("lm_head.weight", t, &mut weights)?);
    }

    // ---- Assemble file ----
    let config_json = config.to_json()?;
    let cfg_bytes = serde_json::to_string(&config_json)?.into_bytes();

    let mut file = std::fs::File::create(dst).with_context(|| format!("creating {}", dst.display()))?;
    file.write_all(MAGIC)?;
    file.write_all(&(cfg_bytes.len() as u64).to_le_bytes())?;
    file.write_all(&(index.len() as u32).to_le_bytes())?;
    file.write_all(&cfg_bytes)?;

    // weights section
    // Layout: magic(4) + cfg_len(8) + n(4) + cfg + weights_len(8) + [pad] +
    // weights + index + trailer. The pad aligns the weights section (and thus
    // every tensor) to TENSOR_ALIGN in the file; the mmap base is
    // page-aligned, so f32/u16 views are correctly aligned.
    // The reader computes the same alignment from cfg_len.
    let weights_len_field_off = (16 + cfg_bytes.len()) as u64;
    let weights_off = align_up((weights_len_field_off + 8) as usize, TENSOR_ALIGN) as u64;
    // The tensor offsets in the index are relative to the FILE start; they
    // were recorded relative to the weights Vec, so rebase them here.
    for (_, tr) in index.iter_mut() {
        tr.offset += weights_off;
    }
    let weights_len = weights.len() as u64;
    file.write_all(&weights_len.to_le_bytes())?;
    let pad = (weights_off - (weights_len_field_off + 8)) as usize;
    for _ in 0..pad {
        file.write_all(&[0u8])?;
    }
    file.write_all(&weights)?;

    // index section
    let index_off = weights_off + weights_len;
    for (name, tr) in &index {
        file.write_all(&(name.len() as u32).to_le_bytes())?;
        file.write_all(name.as_bytes())?;
        file.write_all(&tr.offset.to_le_bytes())?;
        file.write_all(&tr.len.to_le_bytes())?;
    }

    // metadata trailer
    let meta_off = index_off + index
        .iter()
        .map(|(n, _)| 4 + n.len() + 16)
        .sum::<usize>() as u64;
    let file_size = meta_off + 32;
    file.write_all(&index_off.to_le_bytes())?;
    file.write_all(&meta_off.to_le_bytes())?;
    file.write_all(&weights_len.to_le_bytes())?;
    file.write_all(&file_size.to_le_bytes())?;

    eprintln!(
        "TMB1 repack: {} tensors, weights {} bytes, file {} bytes -> {}",
        index.len(),
        weights_len,
        file_size,
        dst.display()
    );
    Ok(())
}

fn align_up(n: usize, a: usize) -> usize {
    (n + a - 1) & !(a - 1)
}
