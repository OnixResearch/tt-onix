//! One RWKV-7 layer, Flex (CPU FP32) against TtBackend on identical synthetic
//! weights. Two tokens carry the recurrent and token-shift state.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{accurate_matmul, TtBackend, TtConfig, TtDevice};
use rwkv7::layer::{layer_step, synthetic_spec, LayerSpec, LayerState, HIDDEN};

const TOLERANCE: f32 = 1.0e-3;

fn hidden<B: burn::tensor::backend::Backend>(values: &[f32], device: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(values.to_vec(), [1, HIDDEN]), device)
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn to_vec<const D: usize, B: burn::tensor::backend::Backend>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().expect("f32 tensor")
}

fn run_flex(spec: &LayerSpec) -> ([Vec<f32>; 2], [Vec<f32>; 2]) {
    let device = FlexDevice;
    let weights = spec.materialize::<Flex>(&device);
    let mut state = LayerState::<Flex>::zeros(&device);
    let first = layer_step(&weights, hidden(&spec.hidden, &device), &state, true, None);
    let first_matrix = to_vec(first.1.matrix.clone());
    state = first.1;
    let second = layer_step(
        &weights,
        hidden(
            &spec.hidden.iter().map(|v| -0.5 * v).collect::<Vec<_>>(),
            &device,
        ),
        &state,
        true,
        None,
    );
    (
        [to_vec(first.0), to_vec(second.0)],
        [first_matrix, to_vec(second.1.matrix)],
    )
}

fn run_tt(spec: &LayerSpec, card: u16) -> ([Vec<f32>; 2], [Vec<f32>; 2]) {
    let device = TtDevice::new(card);
    burn_tt::init(device, TtConfig::default()).expect("attach card");
    let weights = spec.materialize::<TtBackend>(&device);
    accurate_matmul(|| {
        let mut state = LayerState::<TtBackend>::zeros(&device);
        let first = layer_step(&weights, hidden(&spec.hidden, &device), &state, true, None);
        let first_matrix = to_vec(first.1.matrix.clone());
        state = first.1;
        let second = layer_step(
            &weights,
            hidden(
                &spec.hidden.iter().map(|v| -0.5 * v).collect::<Vec<_>>(),
                &device,
            ),
            &state,
            true,
            None,
        );
        (
            [to_vec(first.0), to_vec(second.0)],
            [first_matrix, to_vec(second.1.matrix)],
        )
    })
}

fn main() -> std::process::ExitCode {
    let mut card = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--card" => {
                let Some(value) = args.next() else {
                    eprintln!("--card needs a device index");
                    return std::process::ExitCode::FAILURE;
                };
                match value.parse::<u16>() {
                    Ok(value) => card = Some(value),
                    Err(_) => {
                        eprintln!("--card {value}: not a number");
                        return std::process::ExitCode::FAILURE;
                    }
                }
            }
            "-h" | "--help" => {
                println!("usage: rwkv7-layer-gate [--card N]");
                return std::process::ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument {other}\n\nusage: rwkv7-layer-gate [--card N]");
                return std::process::ExitCode::FAILURE;
            }
        }
    }

    let spec = synthetic_spec(29);
    let (flex_hidden, flex_state) = run_flex(&spec);

    let mut control = spec.clone();
    control.k_k = vec![0.0; control.k_k.len()];
    let (control_hidden, _) = run_flex(&control);
    let control_gap = max_abs(&control_hidden[1], &flex_hidden[1]);
    println!(
        "flex: token1 max {:.6}, token2 max {:.6}, state max {:.6}; control gap {:.6}",
        flex_hidden[0].iter().map(|v| v.abs()).fold(0.0, f32::max),
        flex_hidden[1].iter().map(|v| v.abs()).fold(0.0, f32::max),
        flex_state[1].iter().map(|v| v.abs()).fold(0.0, f32::max),
        control_gap
    );
    if control_gap <= 1.0e-4 {
        eprintln!("negative control did not move the layer output: {control_gap}");
        return std::process::ExitCode::FAILURE;
    }

    let Some(card) = card else {
        println!("flex layer PASS; pass --card N to run the device gate");
        return std::process::ExitCode::SUCCESS;
    };
    let (tt_hidden, tt_state) = run_tt(&spec, card);
    let gaps = [
        ("token1 hidden", max_abs(&tt_hidden[0], &flex_hidden[0])),
        ("token2 hidden", max_abs(&tt_hidden[1], &flex_hidden[1])),
        ("token1 state", max_abs(&tt_state[0], &flex_state[0])),
        ("token2 state", max_abs(&tt_state[1], &flex_state[1])),
    ];
    for (name, gap) in gaps {
        println!("card {card}: {name} gap {gap:.3e}");
        if !gap.is_finite() || gap > TOLERANCE {
            eprintln!("{name}: gap {gap:.3e} exceeds {TOLERANCE:.1e}");
            return std::process::ExitCode::FAILURE;
        }
    }
    println!("layer gate PASS");
    std::process::ExitCode::SUCCESS
}
