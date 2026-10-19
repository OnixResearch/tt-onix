//! One RWKV-7 layer, generic over a Burn backend.
//!
//! The equations follow `pkgs/rwkv-layer-harness/reference/rwkv7_torch_equation_reference.py`
//! at the pinned checkpoint identity: layer norm, token shift, the time-mixing
//! LoRA projections, the WKV7 recurrence, group norm, the value correction, the
//! output projection, and the squared-ReLU channel mixer.
//!
//! The gate compares `Flex` (CPU, FP32) with `TtBackend` on identical synthetic
//! weights. It is a layer-graph and state-carry check, not a real-weight model
//! result.

use burn::tensor::{activation, backend::Backend, Tensor, TensorData};

pub const HIDDEN: usize = 768;
pub const INTERMEDIATE: usize = 3072;
pub const HEADS: usize = 12;
pub const HEAD_SIZE: usize = 64;
pub const LORA_RANK: usize = 64;
const LAYER_NORM_EPS: f32 = 1.0e-5;
const GROUP_NORM_EPS: f32 = HEAD_SIZE as f32 * LAYER_NORM_EPS;
const NEGATIVE_INVERSE_SQRT_E: f32 = -0.606_530_67;
const NORMALIZATION_FLOOR: f32 = 1.0e-12;

#[derive(Clone, Debug)]
pub struct LayerSpec {
    pub pre_norm_w: Vec<f32>,
    pub pre_norm_b: Vec<f32>,
    pub attn_norm_w: Vec<f32>,
    pub attn_norm_b: Vec<f32>,
    pub ffn_norm_w: Vec<f32>,
    pub ffn_norm_b: Vec<f32>,
    pub g_norm_w: Vec<f32>,
    pub g_norm_b: Vec<f32>,
    pub x_r: Vec<f32>,
    pub x_w: Vec<f32>,
    pub x_k: Vec<f32>,
    pub x_v: Vec<f32>,
    pub x_a: Vec<f32>,
    pub x_g: Vec<f32>,
    pub r_proj: Vec<f32>,
    pub k_proj: Vec<f32>,
    pub v_proj: Vec<f32>,
    pub o_proj: Vec<f32>,
    pub w_lora0: Vec<f32>,
    pub w_lora2: Vec<f32>,
    pub w_lora2_b: Vec<f32>,
    pub a_lora0: Vec<f32>,
    pub a_lora2: Vec<f32>,
    pub a_lora2_b: Vec<f32>,
    pub g_lora0: Vec<f32>,
    pub g_lora2: Vec<f32>,
    pub v_lora0: Vec<f32>,
    pub v_lora2: Vec<f32>,
    pub v_lora2_b: Vec<f32>,
    pub k_k: Vec<f32>,
    pub k_a: Vec<f32>,
    pub r_k: Vec<f32>,
    pub ffn_x_k: Vec<f32>,
    pub ffn_key: Vec<f32>,
    pub ffn_value: Vec<f32>,
    pub hidden: Vec<f32>,
}

pub struct LayerWeights<B: Backend> {
    pub pre_norm_w: Tensor<B, 2>,
    pub pre_norm_b: Tensor<B, 2>,
    pub attn_norm_w: Tensor<B, 2>,
    pub attn_norm_b: Tensor<B, 2>,
    pub ffn_norm_w: Tensor<B, 2>,
    pub ffn_norm_b: Tensor<B, 2>,
    pub g_norm_w: Tensor<B, 2>,
    pub g_norm_b: Tensor<B, 2>,
    pub x_r: Tensor<B, 2>,
    pub x_w: Tensor<B, 2>,
    pub x_k: Tensor<B, 2>,
    pub x_v: Tensor<B, 2>,
    pub x_a: Tensor<B, 2>,
    pub x_g: Tensor<B, 2>,
    pub r_proj: Tensor<B, 2>,
    pub k_proj: Tensor<B, 2>,
    pub v_proj: Tensor<B, 2>,
    pub o_proj: Tensor<B, 2>,
    pub w_lora0: Tensor<B, 2>,
    pub w_lora2: Tensor<B, 2>,
    pub w_lora2_b: Tensor<B, 2>,
    pub a_lora0: Tensor<B, 2>,
    pub a_lora2: Tensor<B, 2>,
    pub a_lora2_b: Tensor<B, 2>,
    pub g_lora0: Tensor<B, 2>,
    pub g_lora2: Tensor<B, 2>,
    pub v_lora0: Tensor<B, 2>,
    pub v_lora2: Tensor<B, 2>,
    pub v_lora2_b: Tensor<B, 2>,
    pub k_k: Tensor<B, 2>,
    pub k_a: Tensor<B, 2>,
    pub r_k: Tensor<B, 2>,
    pub ffn_x_k: Tensor<B, 2>,
    pub ffn_key: Tensor<B, 2>,
    pub ffn_value: Tensor<B, 2>,
}

#[derive(Clone)]
pub struct LayerState<B: Backend> {
    pub attention_previous: Tensor<B, 2>,
    pub ffn_previous: Tensor<B, 2>,
    pub matrix: Tensor<B, 3>,
}

impl<B: Backend> LayerState<B> {
    pub fn zeros(device: &B::Device) -> Self {
        Self {
            attention_previous: tensor2(&vec![0.0; HIDDEN], 1, HIDDEN, device),
            ffn_previous: tensor2(&vec![0.0; HIDDEN], 1, HIDDEN, device),
            matrix: tensor3(&vec![0.0; HEADS * HEAD_SIZE * HEAD_SIZE], device),
        }
    }
}

fn tensor2<B: Backend>(data: &[f32], rows: usize, cols: usize, device: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(data.to_vec(), [rows, cols]), device)
}

fn tensor3<B: Backend>(data: &[f32], device: &B::Device) -> Tensor<B, 3> {
    Tensor::from_data(
        TensorData::new(data.to_vec(), [HEADS, HEAD_SIZE, HEAD_SIZE]),
        device,
    )
}

impl LayerSpec {
    pub fn materialize<B: Backend>(&self, device: &B::Device) -> LayerWeights<B> {
        let v = |data: &Vec<f32>, rows: usize, cols: usize| tensor2(data, rows, cols, device);
        LayerWeights {
            pre_norm_w: v(&self.pre_norm_w, 1, HIDDEN),
            pre_norm_b: v(&self.pre_norm_b, 1, HIDDEN),
            attn_norm_w: v(&self.attn_norm_w, 1, HIDDEN),
            attn_norm_b: v(&self.attn_norm_b, 1, HIDDEN),
            ffn_norm_w: v(&self.ffn_norm_w, 1, HIDDEN),
            ffn_norm_b: v(&self.ffn_norm_b, 1, HIDDEN),
            g_norm_w: v(&self.g_norm_w, HEADS, HEAD_SIZE),
            g_norm_b: v(&self.g_norm_b, HEADS, HEAD_SIZE),
            x_r: v(&self.x_r, 1, HIDDEN),
            x_w: v(&self.x_w, 1, HIDDEN),
            x_k: v(&self.x_k, 1, HIDDEN),
            x_v: v(&self.x_v, 1, HIDDEN),
            x_a: v(&self.x_a, 1, HIDDEN),
            x_g: v(&self.x_g, 1, HIDDEN),
            r_proj: v(&self.r_proj, HIDDEN, HIDDEN),
            k_proj: v(&self.k_proj, HIDDEN, HIDDEN),
            v_proj: v(&self.v_proj, HIDDEN, HIDDEN),
            o_proj: v(&self.o_proj, HIDDEN, HIDDEN),
            w_lora0: v(&self.w_lora0, LORA_RANK, HIDDEN),
            w_lora2: v(&self.w_lora2, HIDDEN, LORA_RANK),
            w_lora2_b: v(&self.w_lora2_b, 1, HIDDEN),
            a_lora0: v(&self.a_lora0, LORA_RANK, HIDDEN),
            a_lora2: v(&self.a_lora2, HIDDEN, LORA_RANK),
            a_lora2_b: v(&self.a_lora2_b, 1, HIDDEN),
            g_lora0: v(&self.g_lora0, LORA_RANK, HIDDEN),
            g_lora2: v(&self.g_lora2, HIDDEN, LORA_RANK),
            v_lora0: v(&self.v_lora0, LORA_RANK, HIDDEN),
            v_lora2: v(&self.v_lora2, HIDDEN, LORA_RANK),
            v_lora2_b: v(&self.v_lora2_b, 1, HIDDEN),
            k_k: v(&self.k_k, 1, HIDDEN),
            k_a: v(&self.k_a, 1, HIDDEN),
            r_k: v(&self.r_k, 1, HIDDEN),
            ffn_x_k: v(&self.ffn_x_k, 1, HIDDEN),
            ffn_key: v(&self.ffn_key, INTERMEDIATE, HIDDEN),
            ffn_value: v(&self.ffn_value, HIDDEN, INTERMEDIATE),
        }
    }
}

fn linear<B: Backend>(x: Tensor<B, 2>, weight: Tensor<B, 2>) -> Tensor<B, 2> {
    x.matmul(weight.transpose())
}

fn linear_b<B: Backend>(x: Tensor<B, 2>, weight: Tensor<B, 2>, bias: Tensor<B, 2>) -> Tensor<B, 2> {
    linear(x, weight) + bias
}

fn layer_norm<B: Backend>(
    x: Tensor<B, 2>,
    weight: Tensor<B, 2>,
    bias: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let (var, mean) = x.clone().var_mean_bias(1);
    (x - mean) / (var + LAYER_NORM_EPS).sqrt() * weight + bias
}

fn group_norm<B: Backend>(
    heads: Tensor<B, 2>,
    weight: Tensor<B, 2>,
    bias: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let (var, mean) = heads.clone().var_mean_bias(1);
    (heads - mean) / (var + GROUP_NORM_EPS).sqrt() * weight + bias
}

fn mixed<B: Backend>(
    current: Tensor<B, 2>,
    previous: Tensor<B, 2>,
    factor: Tensor<B, 2>,
) -> Tensor<B, 2> {
    current.clone() + (previous - current) * factor
}

/// Run one layer token. `layer_zero` selects the pre-norm and omits the v-LoRA
/// (the checkpoint has no v-LoRA at layer 0). `anchor` is required for nonzero
/// layers; layer zero returns the anchor it established.
pub fn layer_step<B: Backend>(
    weights: &LayerWeights<B>,
    hidden: Tensor<B, 2>,
    state: &LayerState<B>,
    layer_zero: bool,
    anchor: Option<Tensor<B, 2>>,
) -> (Tensor<B, 2>, LayerState<B>, Option<Tensor<B, 2>>) {
    let residual = if layer_zero {
        layer_norm(
            hidden,
            weights.pre_norm_w.clone(),
            weights.pre_norm_b.clone(),
        )
    } else {
        hidden
    };
    let attention_input = layer_norm(
        residual.clone(),
        weights.attn_norm_w.clone(),
        weights.attn_norm_b.clone(),
    );

    let x_r = mixed(
        attention_input.clone(),
        state.attention_previous.clone(),
        weights.x_r.clone(),
    );
    let x_w = mixed(
        attention_input.clone(),
        state.attention_previous.clone(),
        weights.x_w.clone(),
    );
    let x_k = mixed(
        attention_input.clone(),
        state.attention_previous.clone(),
        weights.x_k.clone(),
    );
    let x_v = mixed(
        attention_input.clone(),
        state.attention_previous.clone(),
        weights.x_v.clone(),
    );
    let x_a = mixed(
        attention_input.clone(),
        state.attention_previous.clone(),
        weights.x_a.clone(),
    );
    let x_g = mixed(
        attention_input.clone(),
        state.attention_previous.clone(),
        weights.x_g.clone(),
    );

    let receptance = linear(x_r, weights.r_proj.clone());
    let raw_key = linear(x_k.clone(), weights.k_proj.clone());
    let projected_value = linear(x_v.clone(), weights.v_proj.clone());

    let (value, anchor) = if layer_zero {
        (projected_value.clone(), Some(projected_value))
    } else {
        let anchor = anchor.expect("a nonzero layer needs the layer-zero value anchor");
        let value_hidden = linear(x_v, weights.v_lora0.clone());
        let value_gate = activation::sigmoid(linear_b(
            value_hidden,
            weights.v_lora2.clone(),
            weights.v_lora2_b.clone(),
        ));
        let value = projected_value.clone() + (anchor.clone() - projected_value) * value_gate;
        (value, Some(anchor))
    };

    let decay_hidden = linear(x_w, weights.w_lora0.clone()).tanh();
    let decay = (NEGATIVE_INVERSE_SQRT_E
        * activation::sigmoid(linear_b(
            decay_hidden,
            weights.w_lora2.clone(),
            weights.w_lora2_b.clone(),
        )))
    .exp();
    let adaptation_hidden = linear(x_a, weights.a_lora0.clone());
    let adaptation = activation::sigmoid(linear_b(
        adaptation_hidden,
        weights.a_lora2.clone(),
        weights.a_lora2_b.clone(),
    ));
    let gate_hidden = activation::sigmoid(linear(x_g, weights.g_lora0.clone()));
    let gate = linear(gate_hidden, weights.g_lora2.clone());

    let key_key_heads: Tensor<B, 2> =
        (raw_key.clone() * weights.k_k.clone()).reshape([HEADS, HEAD_SIZE]);
    let key_key_norm = (key_key_heads.clone() * key_key_heads.clone())
        .sum_dim(1)
        .sqrt()
        .clamp_min(NORMALIZATION_FLOOR);
    let key_key = key_key_heads / key_key_norm;

    let key: Tensor<B, 2> = raw_key * (1.0 + (adaptation.clone() - 1.0) * weights.k_a.clone());
    let receptance_heads: Tensor<B, 2> = receptance.clone().reshape([HEADS, HEAD_SIZE]);
    let key_heads: Tensor<B, 2> = key.reshape([HEADS, HEAD_SIZE]);
    let value_heads: Tensor<B, 2> = value.reshape([HEADS, HEAD_SIZE]);
    let adaptation_heads: Tensor<B, 2> = adaptation.reshape([HEADS, HEAD_SIZE]);
    let decay_heads: Tensor<B, 2> = decay.reshape([HEADS, HEAD_SIZE]);

    let state_dot_a = state
        .matrix
        .clone()
        .matmul((key_key.clone() * -1.0).unsqueeze_dim::<3>(2));
    let next_matrix = state.matrix.clone() * decay_heads.unsqueeze_dim::<3>(1)
        + state_dot_a * (key_key * adaptation_heads).unsqueeze_dim::<3>(1)
        + value_heads.clone().unsqueeze_dim::<3>(2) * key_heads.clone().unsqueeze_dim::<3>(1);

    let recurrent_output = next_matrix
        .clone()
        .matmul(receptance_heads.clone().unsqueeze_dim::<3>(2))
        .squeeze_dim::<2>(2);
    let normalized = group_norm(
        recurrent_output,
        weights.g_norm_w.clone(),
        weights.g_norm_b.clone(),
    );
    let r_k_heads: Tensor<B, 2> = weights.r_k.clone().reshape([HEADS, HEAD_SIZE]);
    let correction = (receptance_heads * key_heads * r_k_heads).sum_dim(1);
    let corrected = (normalized + correction * value_heads).reshape([1, HIDDEN]) * gate;
    let attention_output = linear(corrected, weights.o_proj.clone());
    let after_attention = residual + attention_output;

    let ffn_input = layer_norm(
        after_attention.clone(),
        weights.ffn_norm_w.clone(),
        weights.ffn_norm_b.clone(),
    );
    let ffn_mixed = mixed(
        ffn_input.clone(),
        state.ffn_previous.clone(),
        weights.ffn_x_k.clone(),
    );
    let ffn_key = activation::relu(linear(ffn_mixed, weights.ffn_key.clone()));
    let ffn_key = ffn_key.clone() * ffn_key;
    let ffn_output = linear(ffn_key, weights.ffn_value.clone());
    let next_hidden = after_attention + ffn_output;

    let next_state = LayerState {
        attention_previous: attention_input,
        ffn_previous: ffn_input,
        matrix: next_matrix,
    };
    (next_hidden, next_state, anchor)
}

/// Deterministic synthetic weights for the differential gate.
pub fn synthetic_spec(seed: u64) -> LayerSpec {
    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 40) as f32) / ((1u32 << 24) as f32)
    }
    fn fill(state: &mut u64, n: usize, scale: f32, base: f32) -> Vec<f32> {
        (0..n)
            .map(|_| base + (lcg(state) * 2.0 - 1.0) * scale)
            .collect()
    }
    let mut state = seed;
    LayerSpec {
        pre_norm_w: fill(&mut state, HIDDEN, 0.02, 1.0),
        pre_norm_b: fill(&mut state, HIDDEN, 0.02, 0.0),
        attn_norm_w: fill(&mut state, HIDDEN, 0.02, 1.0),
        attn_norm_b: fill(&mut state, HIDDEN, 0.02, 0.0),
        ffn_norm_w: fill(&mut state, HIDDEN, 0.02, 1.0),
        ffn_norm_b: fill(&mut state, HIDDEN, 0.02, 0.0),
        g_norm_w: fill(&mut state, HIDDEN, 0.02, 1.0),
        g_norm_b: fill(&mut state, HIDDEN, 0.02, 0.0),
        x_r: fill(&mut state, HIDDEN, 0.1, 0.5),
        x_w: fill(&mut state, HIDDEN, 0.1, 0.5),
        x_k: fill(&mut state, HIDDEN, 0.1, 0.5),
        x_v: fill(&mut state, HIDDEN, 0.1, 0.5),
        x_a: fill(&mut state, HIDDEN, 0.1, 0.5),
        x_g: fill(&mut state, HIDDEN, 0.1, 0.5),
        r_proj: fill(&mut state, HIDDEN * HIDDEN, 0.02, 0.0),
        k_proj: fill(&mut state, HIDDEN * HIDDEN, 0.02, 0.0),
        v_proj: fill(&mut state, HIDDEN * HIDDEN, 0.02, 0.0),
        o_proj: fill(&mut state, HIDDEN * HIDDEN, 0.02, 0.0),
        w_lora0: fill(&mut state, LORA_RANK * HIDDEN, 0.02, 0.0),
        w_lora2: fill(&mut state, HIDDEN * LORA_RANK, 0.02, 0.0),
        w_lora2_b: fill(&mut state, HIDDEN, 0.02, 0.0),
        a_lora0: fill(&mut state, LORA_RANK * HIDDEN, 0.02, 0.0),
        a_lora2: fill(&mut state, HIDDEN * LORA_RANK, 0.02, 0.0),
        a_lora2_b: fill(&mut state, HIDDEN, 0.02, 0.0),
        g_lora0: fill(&mut state, LORA_RANK * HIDDEN, 0.02, 0.0),
        g_lora2: fill(&mut state, HIDDEN * LORA_RANK, 0.02, 0.0),
        v_lora0: fill(&mut state, LORA_RANK * HIDDEN, 0.02, 0.0),
        v_lora2: fill(&mut state, HIDDEN * LORA_RANK, 0.02, 0.0),
        v_lora2_b: fill(&mut state, HIDDEN, 0.02, 0.0),
        k_k: fill(&mut state, HIDDEN, 0.2, 0.5),
        k_a: fill(&mut state, HIDDEN, 0.2, 0.5),
        r_k: fill(&mut state, HIDDEN, 0.2, 0.5),
        ffn_x_k: fill(&mut state, HIDDEN, 0.1, 0.5),
        ffn_key: fill(&mut state, INTERMEDIATE * HIDDEN, 0.02, 0.0),
        ffn_value: fill(&mut state, HIDDEN * INTERMEDIATE, 0.02, 0.0),
        hidden: fill(&mut state, HIDDEN, 0.1, 0.0),
    }
}
