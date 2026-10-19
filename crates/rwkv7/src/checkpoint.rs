//! Read the pinned RWKV-7 checkpoint into a [`LayerSpec`].
//!
//! Only layer zero is supported: it has the pre-norm and no v-LoRA. The
//! checkpoint stores BF16; the gate converts to F32 exactly as the harness
//! reference does.

use std::path::Path;

use safetensors::{Dtype, SafeTensors};

use crate::layer::{LayerSpec, G_LORA_RANK, HEADS, HEAD_SIZE, HIDDEN, INTERMEDIATE, LORA_RANK};

pub struct Checkpoint {
    data: Vec<u8>,
}

impl Checkpoint {
    pub fn open(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
        SafeTensors::deserialize(&data)
            .map_err(|error| format!("not a safetensors file: {error}"))?;
        Ok(Self { data })
    }

    fn read_range(&self, name: &str, start: usize, count: usize) -> Result<Vec<f32>, String> {
        let tensors = SafeTensors::deserialize(&self.data)
            .map_err(|error| format!("safetensors: {error}"))?;
        let view = tensors
            .tensor(name)
            .map_err(|_| format!("checkpoint is missing `{name}`"))?;
        let shape = view.shape().to_vec();
        let elements = shape.iter().product::<usize>();
        if start + count > elements {
            return Err(format!(
                "`{name}` has {elements} elements; range {start}+{count} is out of bounds"
            ));
        }
        let bytes = view.data();
        match view.dtype() {
            Dtype::F32 => {
                let begin = start * 4;
                let end = (start + count) * 4;
                Ok(bytes[begin..end]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            Dtype::BF16 => {
                let begin = start * 2;
                let end = (start + count) * 2;
                Ok(bytes[begin..end]
                    .chunks_exact(2)
                    .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                    .collect())
            }
            other => Err(format!("`{name}` has unsupported dtype {other:?}")),
        }
    }

    fn flat(&self, name: &str, expected: usize) -> Result<Vec<f32>, String> {
        let values = self.read_range(name, 0, expected)?;
        if values.len() != expected {
            return Err(format!(
                "`{name}` has {} elements, expected {expected}",
                values.len()
            ));
        }
        Ok(values)
    }

    pub fn embedding(&self, token: usize) -> Result<Vec<f32>, String> {
        self.read_range("model.embeddings.weight", token * HIDDEN, HIDDEN)
    }

    /// Layer-zero weights, with `hidden` set to the embedding row for `token`.
    pub fn layer_zero_spec(&self, token: usize) -> Result<LayerSpec, String> {
        let p = "model.layers.0";
        Ok(LayerSpec {
            pre_norm_w: self.flat(&format!("{p}.pre_norm.weight"), HIDDEN)?,
            pre_norm_b: self.flat(&format!("{p}.pre_norm.bias"), HIDDEN)?,
            attn_norm_w: self.flat(&format!("{p}.attn_norm.weight"), HIDDEN)?,
            attn_norm_b: self.flat(&format!("{p}.attn_norm.bias"), HIDDEN)?,
            ffn_norm_w: self.flat(&format!("{p}.ffn_norm.weight"), HIDDEN)?,
            ffn_norm_b: self.flat(&format!("{p}.ffn_norm.bias"), HIDDEN)?,
            g_norm_w: self.flat(&format!("{p}.attn.g_norm.weight"), HIDDEN)?,
            g_norm_b: self.flat(&format!("{p}.attn.g_norm.bias"), HIDDEN)?,
            x_r: self.flat(&format!("{p}.attn.x_r"), HIDDEN)?,
            x_w: self.flat(&format!("{p}.attn.x_w"), HIDDEN)?,
            x_k: self.flat(&format!("{p}.attn.x_k"), HIDDEN)?,
            x_v: self.flat(&format!("{p}.attn.x_v"), HIDDEN)?,
            x_a: self.flat(&format!("{p}.attn.x_a"), HIDDEN)?,
            x_g: self.flat(&format!("{p}.attn.x_g"), HIDDEN)?,
            r_proj: self.flat(&format!("{p}.attn.r_proj.weight"), HIDDEN * HIDDEN)?,
            k_proj: self.flat(&format!("{p}.attn.k_proj.weight"), HIDDEN * HIDDEN)?,
            v_proj: self.flat(&format!("{p}.attn.v_proj.weight"), HIDDEN * HIDDEN)?,
            o_proj: self.flat(&format!("{p}.attn.o_proj.weight"), HIDDEN * HIDDEN)?,
            w_lora0: self.flat(
                &format!("{p}.attn.w_lora.lora.0.weight"),
                LORA_RANK * HIDDEN,
            )?,
            w_lora2: self.flat(
                &format!("{p}.attn.w_lora.lora.2.weight"),
                HIDDEN * LORA_RANK,
            )?,
            w_lora2_b: self.flat(&format!("{p}.attn.w_lora.lora.2.bias"), HIDDEN)?,
            a_lora0: self.flat(
                &format!("{p}.attn.a_lora.lora.0.weight"),
                LORA_RANK * HIDDEN,
            )?,
            a_lora2: self.flat(
                &format!("{p}.attn.a_lora.lora.2.weight"),
                HIDDEN * LORA_RANK,
            )?,
            a_lora2_b: self.flat(&format!("{p}.attn.a_lora.lora.2.bias"), HIDDEN)?,
            g_lora0: self.flat(
                &format!("{p}.attn.g_lora.lora.0.weight"),
                G_LORA_RANK * HIDDEN,
            )?,
            g_lora2: self.flat(
                &format!("{p}.attn.g_lora.lora.2.weight"),
                HIDDEN * G_LORA_RANK,
            )?,
            // Layer zero has no v-LoRA. The materializer still needs tensors;
            // these zeros are never read on the layer-zero path.
            v_lora0: vec![0.0; LORA_RANK * HIDDEN],
            v_lora2: vec![0.0; HIDDEN * LORA_RANK],
            v_lora2_b: vec![0.0; HIDDEN],
            k_k: self.flat(&format!("{p}.attn.k_k"), HIDDEN)?,
            k_a: self.flat(&format!("{p}.attn.k_a"), HIDDEN)?,
            r_k: self.flat(&format!("{p}.attn.r_k"), HEADS * HEAD_SIZE)?,
            ffn_x_k: self.flat(&format!("{p}.ffn.x_k"), HIDDEN)?,
            ffn_key: self.flat(&format!("{p}.ffn.key.weight"), INTERMEDIATE * HIDDEN)?,
            ffn_value: self.flat(&format!("{p}.ffn.value.weight"), HIDDEN * INTERMEDIATE)?,
            hidden: self.embedding(token)?,
        })
    }
}
