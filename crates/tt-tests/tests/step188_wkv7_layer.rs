//! One RWKV-7 layer on the fused WKV7 recurrence, Flex against `burn-tt`.
//!
//! The layer graph is the same one `rwkv7-layer-gate` checks; the only change
//! is the recurrence: `rwkv7::layer::fused_tt_recurrence` slices the twelve
//! heads out of the projections and runs `burn_tt::wkv7_step` once per head,
//! instead of the composed matmuls. Everything else -- norms, token shift,
//! projections, LoRA, group norm, the value correction and the mixer -- is
//! `layer_step_with`'s shared graph.
//!
//! Claims: two synthetic tokens, the second starting from the first's device
//! state, match Flex within `TOLERANCE`; zeroing the receptance projection
//! moves the layer's output (the recurrence is not dead code).
//!
//! ttsim and card 0: `with_device` picks the engine.

use burn::tensor::{backend::Backend, Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::TtBackend;
use rwkv7::layer::{layer_step, layer_step_fused, synthetic_spec, LayerSpec, LayerState, HIDDEN};
use tt_tests::burn_device::{with_device, Config};

const TOLERANCE: f32 = 1.0e-3;

fn hidden<B: Backend>(values: &[f32], device: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(values.to_vec(), [1, HIDDEN]), device)
}

fn to_vec<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().expect("f32 tensor")
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn specs() -> (LayerSpec, LayerSpec) {
    let first = synthetic_spec(29);
    let mut second = first.clone();
    second.hidden = first.hidden.iter().map(|value| -0.5 * value).collect();
    (first, second)
}

/// Flex's two tokens: both hidden states, then both recurrent states.
fn run_flex(first: &LayerSpec, second: &LayerSpec) -> [Vec<f32>; 4] {
    let device = FlexDevice;
    let weights = first.materialize::<Flex>(&device);
    let mut state = LayerState::<Flex>::zeros(&device);
    let one = layer_step(&weights, hidden(&first.hidden, &device), &state, true, None);
    let first_matrix = to_vec(one.1.matrix.clone());
    state = one.1;
    let two = layer_step(
        &weights,
        hidden(&second.hidden, &device),
        &state,
        true,
        None,
    );
    [
        to_vec(one.0),
        to_vec(two.0),
        first_matrix,
        to_vec(two.1.matrix),
    ]
}

#[test]
fn one_layer_on_the_fused_recurrence_matches_flex() {
    let (first, second) = specs();
    let want = run_flex(&first, &second);

    // The negative control: no receptance projection, so `out` is zero.
    let mut control_first = first.clone();
    let mut control_second = second.clone();
    control_first.r_proj = vec![0.0; control_first.r_proj.len()];
    control_second.r_proj = vec![0.0; control_second.r_proj.len()];
    let control = run_flex(&control_first, &control_second);
    let control_gap = max_abs(&control[0], &want[0]);
    println!("control: max |layer without r_proj - layer| {control_gap:.3e}");
    assert!(
        control_gap > 1.0e-4,
        "zeroing r_proj moved the layer by only {control_gap:.3e}"
    );

    with_device(Config::default(), |device| {
        // Default matmul mode: the layer's 768x768 products are TF32 here and
        // their rounding dominates the gap (about 5e-4 against Flex). The
        // kernel's recurrence is gated at 1e-5 in step186/187, and the
        // accurate card layer gate lands at 3.6e-7.
        let weights = first.materialize::<TtBackend>(&device);
        let mut state = LayerState::<TtBackend>::zeros(&device);
        let one = layer_step_fused(&weights, hidden(&first.hidden, &device), &state, true, None);
        let first_matrix = to_vec(one.1.matrix.clone());
        state = one.1;
        let two = layer_step_fused(
            &weights,
            hidden(&second.hidden, &device),
            &state,
            true,
            None,
        );
        let got = [
            to_vec(one.0),
            to_vec(two.0),
            first_matrix,
            to_vec(two.1.matrix),
        ];
        for (name, got, want) in [
            ("token 1 hidden", &got[0], &want[0]),
            ("token 2 hidden", &got[1], &want[1]),
            ("token 1 state", &got[2], &want[2]),
            ("token 2 state", &got[3], &want[3]),
        ] {
            let gap = max_abs(got, want);
            println!("{name}: max |gap| {gap:.3e}");
            assert!(
                gap.is_finite() && gap <= TOLERANCE,
                "{name}: gap {gap:.3e} exceeds {TOLERANCE:.1e}"
            );
        }
    });
}
