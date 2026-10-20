//! One RWKV-7 layer, Flex (CPU FP32) against TtBackend.
//!
//! Without `--checkpoint`, the weights are deterministic synthetic values.
//! With `--checkpoint`, layer zero is read from the pinned safetensors file and
//! the two token inputs are embedding rows 1 and 2.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{accurate_matmul, TileChoice, TtBackend, TtConfig, TtDevice};
use rwkv7::checkpoint::Checkpoint;
use rwkv7::layer::{layer_step, layer_step_fused, synthetic_spec, LayerSpec, LayerState, HIDDEN};

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

fn run_flex(first: &LayerSpec, second: &LayerSpec) -> ([Vec<f32>; 2], [Vec<f32>; 2]) {
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
    (
        [to_vec(one.0), to_vec(two.0)],
        [first_matrix, to_vec(two.1.matrix)],
    )
}

fn run_tt(
    first: &LayerSpec,
    second: &LayerSpec,
    card: u16,
    fused: bool,
    tiles: Option<usize>,
) -> ([Vec<f32>; 2], [Vec<f32>; 2]) {
    let device = TtDevice::new(card);
    let config = match tiles {
        Some(n) => TtConfig::default().tiles(TileChoice::Count(n)),
        None => TtConfig::default(),
    };
    burn_tt::init(device, config).expect("attach card");
    let weights = first.materialize::<TtBackend>(&device);
    accurate_matmul(|| {
        let mut state = LayerState::<TtBackend>::zeros(&device);
        let one = if fused {
            layer_step_fused(&weights, hidden(&first.hidden, &device), &state, true, None)
        } else {
            layer_step(&weights, hidden(&first.hidden, &device), &state, true, None)
        };
        let first_matrix = to_vec(one.1.matrix.clone());
        state = one.1;
        let two = if fused {
            layer_step_fused(
                &weights,
                hidden(&second.hidden, &device),
                &state,
                true,
                None,
            )
        } else {
            layer_step(
                &weights,
                hidden(&second.hidden, &device),
                &state,
                true,
                None,
            )
        };
        (
            [to_vec(one.0), to_vec(two.0)],
            [first_matrix, to_vec(two.1.matrix)],
        )
    })
}

fn main() -> std::process::ExitCode {
    let mut card = None;
    let mut checkpoint = None;
    let mut fused = false;
    let mut tiles = Some(12);
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
            "--fused" => fused = true,
            "--tiles" => {
                let Some(value) = args.next() else {
                    eprintln!("--tiles needs a count");
                    return std::process::ExitCode::FAILURE;
                };
                match value.parse::<usize>() {
                    Ok(value) if value > 0 => tiles = Some(value),
                    _ => {
                        eprintln!("--tiles {value}: expected a positive count");
                        return std::process::ExitCode::FAILURE;
                    }
                }
            }
            "--checkpoint" => {
                let Some(value) = args.next() else {
                    eprintln!("--checkpoint needs a safetensors path");
                    return std::process::ExitCode::FAILURE;
                };
                checkpoint = Some(value);
            }
            "-h" | "--help" => {
                println!(
                    "usage: rwkv7-layer-gate [--checkpoint PATH] [--card N] [--fused] [--tiles N]"
                );
                println!("  --fused  the WKV7 recurrence through the fused kernel");
                println!("  --tiles  Tensix tiles to open (default 12, the model's heads)");
                return std::process::ExitCode::SUCCESS;
            }
            other => {
                eprintln!(
                    "unknown argument {other}\n\nusage: rwkv7-layer-gate [--checkpoint PATH] [--card N] [--fused] [--tiles N]"
                );
                return std::process::ExitCode::FAILURE;
            }
        }
    }

    let (first, second, source) = if let Some(path) = checkpoint {
        let checkpoint = match Checkpoint::open(std::path::Path::new(&path)) {
            Ok(checkpoint) => checkpoint,
            Err(error) => {
                eprintln!("checkpoint: {error}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let first = match checkpoint.layer_zero_spec(1) {
            Ok(spec) => spec,
            Err(error) => {
                eprintln!("layer zero (token 1): {error}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let second = match checkpoint.layer_zero_spec(2) {
            Ok(spec) => spec,
            Err(error) => {
                eprintln!("layer zero (token 2): {error}");
                return std::process::ExitCode::FAILURE;
            }
        };
        (first, second, "checkpoint layer 0")
    } else {
        let first = synthetic_spec(29);
        let mut second = first.clone();
        second.hidden = first.hidden.iter().map(|value| -0.5 * value).collect();
        (first, second, "synthetic layer 0")
    };

    let (flex_hidden, flex_state) = run_flex(&first, &second);

    let mut control_first = first.clone();
    let mut control_second = second.clone();
    control_first.r_proj = vec![0.0; control_first.r_proj.len()];
    control_second.r_proj = vec![0.0; control_second.r_proj.len()];
    let (control_hidden, _) = run_flex(&control_first, &control_second);
    let control_gap = max_abs(&control_hidden[0], &flex_hidden[0]);
    println!(
        "{source}: flex token1 max {:.6}, token2 max {:.6}, state max {:.6}; control gap {:.6}",
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
    let (tt_hidden, tt_state) = run_tt(&first, &second, card, fused, tiles);
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
