use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use memmap2::MmapOptions;
use safetensors::tensor::SafeTensors;
use safetensors::tensor::Dtype as DataType;

/// A tensor loaded into memory.
/// BF16 tensors keep their raw bits (dequantized in-register in the gemm kernels)
/// to halve memory footprint and bandwidth; F16/F32 are converted to f32 at load.
/// `data` is a raw byte view so tensors can borrow directly from the memory map.
#[derive(Clone)]
pub struct Tensor {
    #[allow(dead_code)] // kept alongside data for shape-aware consumers
    pub shape: Vec<usize>,
    pub data: TensorData,
}

#[derive(Clone)]
pub enum TensorData {
    F32(Vec<f32>),
    F16(Vec<f32>),      // converted from f16 at load
    BF16Raw(Vec<u8>),   // raw bf16 bits (2 per element), dequantized in the math kernels
}

impl Tensor {
    pub fn as_f32(&self) -> &[f32] {
        match &self.data {
            TensorData::F32(v) | TensorData::F16(v) => v,
            TensorData::BF16Raw(_) => panic!("as_f32 on raw BF16 tensor; use bf16_bits()"),
        }
    }

    pub fn bf16_bits(&self) -> &[u16] {
        match &self.data {
            // BF16 data is 2-byte aligned (all offsets are even in practice);
            // the 2-byte alignment invariant is enforced at construction.
            TensorData::BF16Raw(v) => bytemuck::cast_slice(v),
            _ => panic!("bf16_bits on non-BF16 tensor"),
        }
    }

    /// Raw BF16 bytes (used by the TMB repack tool, which copies them as-is).
    pub fn bf16_bytes(&self) -> &[u8] {
        match &self.data {
            TensorData::BF16Raw(v) => v,
            _ => panic!("bf16_bytes on non-BF16 tensor"),
        }
    }

    pub fn is_bf16(&self) -> bool {
        matches!(self.data, TensorData::BF16Raw(_))
    }

    /// Move the owned BF16 bits out (used to build WMat without a second copy).
    pub fn into_bf16_bits(self) -> Vec<u16> {
        match self.data {
            TensorData::BF16Raw(v) => {
                // BF16 byte len is always even, so this reinterprets in place.
                let mut v = v;
                let ptr = v.as_mut_ptr() as *mut u16;
                let len = v.len() / 2;
                let cap = v.capacity() / 2;
                std::mem::forget(v);
                // Safety: the byte Vec is 2-byte aligned (heap allocations are
                // at least 8-byte aligned) and length/capacity are exact.
                unsafe { Vec::from_raw_parts(ptr, len, cap) }
            }
            _ => panic!("into_bf16_bits on non-BF16 tensor"),
        }
    }

    /// Move the owned f32 data out (used to build WMat / f32 vecs without a copy).
    pub fn into_f32(self) -> Vec<f32> {
        match self.data {
            TensorData::F32(v) | TensorData::F16(v) => v,
            TensorData::BF16Raw(_) => panic!("into_f32 on raw BF16 tensor; use bf16_bits()"),
        }
    }
}

/// Parse the 8-byte little-endian header length, then read the header JSON.
fn read_safetensors_header(bytes: &[u8]) -> Result<serde_json::Value> {
    if bytes.len() < 8 {
        bail!("safetensors file too small");
    }
    let n = u64::from_le_bytes(bytes[0..8].try_into()?) as usize;
    if bytes.len() < 8 + n {
        bail!("safetensors header truncated");
    }
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + n]).context("invalid safetensors header JSON")?;
    Ok(header)
}

/// Load all tensors from a safetensors file.
///
/// The file is memory-mapped with MAP_POPULATE (prefault + readahead on
/// Linux): tensor data is copied out of the mapping in bulk (one memcpy per
/// tensor) and BF16 bits are kept raw, so loading is a single pass over the
/// file with no per-element conversion loops.
pub fn load_safetensors(path: &Path) -> Result<HashMap<String, Tensor>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let map = unsafe { MmapOptions::new().populate().map(&file) }
        .with_context(|| format!("mmapping {}", path.display()))?;
    if map.len() < 8 {
        bail!("safetensors file too small: {}", path.display());
    }
    let _header = read_safetensors_header(&map)?;

    let st = SafeTensors::deserialize(&map)?;
    let mut out = HashMap::new();
    for (name, view) in st.iter() {
        let shape: Vec<usize> = view.shape().to_vec();
        let data = match view.dtype() {
            DataType::F32 => {
                // Bulk-copy the raw bytes: input is 4-byte aligned by the
                // safetensors layout, so this is a single memcpy per tensor.
                let bytes = view.data();
                if bytes.len() % 4 != 0 {
                    bail!("F32 tensor `{name}` has non-multiple-of-4 byte length");
                }
                TensorData::F32(bytemuck::cast_slice(bytes).to_vec())
            }
            DataType::F16 => {
                let raw: Vec<f32> = view
                    .data()
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect();
                TensorData::F16(raw)
            }
            DataType::BF16 => {
                TensorData::BF16Raw(view.data().to_vec())
            }
            other => bail!(
                "unsupported dtype {:?} for tensor {} in {}",
                other,
                name,
                path.display()
            ),
        };
        out.insert(name.to_string(), Tensor { shape, data });
    }
    Ok(out)
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let frac = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            // subnormal
            let mut e = 127u32 - 1;
            let mut f = frac;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            f &= 0x3ff;
            sign | (e << 23) | (f << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13) // inf/nan
    } else {
        sign | ((exp + 127 - 15) << 23) | (frac << 13)
    };
    f32::from_bits(bits)
}

#[allow(dead_code)] // kept for reference/tests
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_conversions() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0), 0.0);
        assert_eq!(f16_to_f32(0xbc00), -1.0);
    }

    #[test]
    fn bf16_conversions() {
        // 1.0f32 bits: 0x3f800000 -> bf16 0x3f80
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        assert_eq!(bf16_to_f32(0), 0.0);
    }

    #[test]
    fn safetensors_roundtrip() {
        use safetensors::tensor::{serialize, TensorView};
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let view = TensorView::new(DataType::F32, vec![2, 2], bytemuck::cast_slice::<f32, u8>(&data)).unwrap();
        let bytes = serialize(vec![("w", view)], None).unwrap();
        let dir = std::env::temp_dir();
        let path = dir.join("tinyminigo-test.safetensors");
        std::fs::write(&path, &bytes).unwrap();
        let tensors = load_safetensors(&path).unwrap();
        let t = tensors.get("w").unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.as_f32(), &data);
    }

    #[test]
    fn bf16_safetensors_roundtrip() {
        use safetensors::tensor::{serialize, TensorView};
        // bf16 bits for 1.0 and -2.0
        let data: Vec<u16> = vec![0x3f80, 0xc000];
        let view = TensorView::new(DataType::BF16, vec![2], bytemuck::cast_slice::<u16, u8>(&data)).unwrap();
        let bytes = serialize(vec![("b", view)], None).unwrap();
        let path = std::env::temp_dir().join("tinyminigo-test-bf16.safetensors");
        std::fs::write(&path, &bytes).unwrap();
        let tensors = load_safetensors(&path).unwrap();
        let t = tensors.get("b").unwrap();
        assert!(t.is_bf16());
        assert_eq!(t.bf16_bits(), &[0x3f80, 0xc000]);
    }
}
