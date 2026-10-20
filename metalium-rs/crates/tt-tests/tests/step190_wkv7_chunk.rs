//! A prompt chunk through one RWKV-7 layer, batch projections against the
//! token-at-a-time path and Flex.
//!
//! The layer is `rwkv7::layer::layer_step_chunk`: norms, token shifts,
//! projections, LoRA and the FFN run as `[tokens, HIDDEN]` batch matmuls; the
//! WKV7 recurrence stays sequential over rows and carries the device state.
//! The reference is four `layer_step_fused` calls on the same hidden rows and
//! four Flex `layer_step` calls.
//!
//! Claims: the chunk's rows, final recurrent matrix and shift vectors match the
//! token path within `TOLERANCE`; the chunk also matches Flex.
//!
//! ttsim and card 0: `with_device` picks the engine.

use burn::tensor::{backend::Backend, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use rwkv7::layer::{
    layer_step, layer_step_chunk, layer_step_fused, synthetic_spec, LayerSpec, LayerState, HIDDEN,
};
use tt_tests::burn_device::{with_device, Config};

const TOLERANCE: f32 = 2.5e-3;
/// Four fast-matmul tokens drift from Flex by about 3.2e-3 in the matrix; the
/// chunk is bit-identical to the token path, which is the implementation gate.
const STATE_TOLERANCE: f32 = 4.0e-3;
const TOKENS: usize = 4;

fn to_vec<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().expect("f32 tensor")
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

/// Distinct, deterministic layer inputs for a prompt chunk.
fn token_values(spec: &LayerSpec) -> Vec<f32> {
    let mut values = Vec::with_capacity(TOKENS * HIDDEN);
    for token in 0..TOKENS {
        let scale = 1.0 + 0.05 * token as f32;
        let offset = 0.002 * token as f32;
        for value in &spec.hidden {
            values.push(value * scale + offset);
        }
    }
    values
}

fn row<B: Backend>(values: &[f32], token: usize, device: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(
        TensorData::new(
            values[token * HIDDEN..(token + 1) * HIDDEN].to_vec(),
            [1, HIDDEN],
        ),
        device,
    )
}

fn flex_tokens(spec: &LayerSpec, values: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let device = FlexDevice;
    let weights = spec.materialize::<Flex>(&device);
    let mut state = LayerState::<Flex>::zeros(&device);
    let mut hidden = Vec::with_capacity(TOKENS);
    for token in 0..TOKENS {
        let (out, next, _) = layer_step(&weights, row(values, token, &device), &state, true, None);
        hidden.push(to_vec(out));
        state = next;
    }
    (
        hidden,
        to_vec(state.matrix),
        to_vec(state.attention_previous),
        to_vec(state.ffn_previous),
    )
}

#[test]
fn four_token_chunk_matches_eager_tokens_and_flex() {
    let spec = synthetic_spec(29);
    let values = token_values(&spec);
    let (flex_hidden, flex_matrix, flex_attention, flex_ffn) = flex_tokens(&spec, &values);

    with_device(Config::default(), |device| {
        let weights = spec.materialize::<TtBackend>(&device);
        let mut token_state = LayerState::<TtBackend>::zeros(&device);
        let mut token_hidden = Vec::with_capacity(TOKENS);
        for token in 0..TOKENS {
            let (out, next, _) = layer_step_fused(
                &weights,
                row(&values, token, &device),
                &token_state,
                true,
                None,
            );
            token_hidden.push(to_vec(out));
            token_state = next;
        }

        let chunk_state = LayerState::<TtBackend>::zeros(&device);
        let input = Tensor::<TtBackend, 2>::from_data(
            TensorData::new(values.clone(), [TOKENS, HIDDEN]),
            &device,
        );
        let (chunk_out, chunk_state, _) =
            layer_step_chunk(&weights, input, &chunk_state, true, None);
        let chunk_values = to_vec(chunk_out);

        for token in 0..TOKENS {
            let got = &chunk_values[token * HIDDEN..(token + 1) * HIDDEN];
            let token_gap = max_abs(got, &token_hidden[token]);
            let flex_gap = max_abs(got, &flex_hidden[token]);
            println!(
                "token {token}: max |chunk - token path| {token_gap:.3e}, |chunk - Flex| {flex_gap:.3e}"
            );
            assert!(
                token_gap.is_finite() && token_gap <= TOLERANCE,
                "token {token}: chunk differs from the token path by {token_gap:.3e}"
            );
            assert!(
                flex_gap.is_finite() && flex_gap <= TOLERANCE,
                "token {token}: chunk differs from Flex by {flex_gap:.3e}"
            );
        }

        for (name, got, token, flex) in [
            (
                "matrix",
                to_vec(chunk_state.matrix),
                to_vec(token_state.matrix),
                flex_matrix,
            ),
            (
                "attention_previous",
                to_vec(chunk_state.attention_previous),
                to_vec(token_state.attention_previous),
                flex_attention,
            ),
            (
                "ffn_previous",
                to_vec(chunk_state.ffn_previous),
                to_vec(token_state.ffn_previous),
                flex_ffn,
            ),
        ] {
            let token_gap = max_abs(&got, &token);
            let flex_gap = max_abs(&got, &flex);
            println!(
                "{name}: max |chunk - token path| {token_gap:.3e}, |chunk - Flex| {flex_gap:.3e}"
            );
            assert!(
                token_gap.is_finite() && token_gap <= TOLERANCE,
                "{name}: chunk differs from the token path by {token_gap:.3e}"
            );
            assert!(
                flex_gap.is_finite() && flex_gap <= STATE_TOLERANCE,
                "{name}: chunk differs from Flex by {flex_gap:.3e}"
            );
        }
    });
}
