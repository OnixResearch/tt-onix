//! All 12 RWKV-7 layers with the real checkpoint, Flex FP32 against TtBackend.
//!
//! Token inputs are embedding rows 1 and 2. The gate compares the final
//! normalized hidden vector and every layer's post-token-2 recurrent matrix.
//! Logits are not evaluated: the 65536x768 head is outside this gate.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{accurate_matmul, TileChoice, TtBackend, TtConfig, TtDevice};
use rwkv7::checkpoint::Checkpoint;
use rwkv7::layer::{
    layer_norm, layer_step, layer_step_fused, LayerSpec, LayerState, LayerWeights, HIDDEN,
};

const LAYER_COUNT: usize = 12;
const RELATIVE_TOLERANCE: f32 = 1.0e-4;
const EXPECTED_TOKEN: u32 = 2;
const EXPECTED_LOGIT: f32 = 2.864_108_3;
const EXPECTED_RUNNER: u32 = 33;
const EXPECTED_RUNNER_LOGIT: f32 = 0.896_408_86;
const EXPECTED_MARGIN: f32 = 1.967_699_5;
const TOKEN_TOLERANCE: f32 = 1.0e-2;

/// One layer token, with the WKV7 recurrence the caller chose.
type Step<B> = fn(
    &LayerWeights<B>,
    Tensor<B, 2>,
    &LayerState<B>,
    bool,
    Option<Tensor<B, 2>>,
) -> (Tensor<B, 2>, LayerState<B>, Option<Tensor<B, 2>>);

fn vector<B: burn::tensor::backend::Backend>(values: &[f32], device: &B::Device) -> Tensor<B, 2> {
    Tensor::from_data(TensorData::new(values.to_vec(), [1, HIDDEN]), device)
}

fn to_vec<const D: usize, B: burn::tensor::backend::Backend>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor.into_data().to_vec::<f32>().expect("f32 tensor")
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn top_two(logits: &[f32]) -> (u32, f32, u32, f32) {
    let mut first = (0u32, f32::NEG_INFINITY);
    let mut second = (0u32, f32::NEG_INFINITY);
    for (index, &value) in logits.iter().enumerate() {
        if value > first.1 {
            second = first;
            first = (index as u32, value);
        } else if value > second.1 {
            second = (index as u32, value);
        }
    }
    (first.0, first.1, second.0, second.1)
}

fn check_greedy(name: &str, logits: &[f32]) -> bool {
    let (token, logit, runner, runner_logit) = top_two(logits);
    let margin = logit - runner_logit;
    println!(
        "{name}: token {token} logit {logit:.7}, runner {runner} logit {runner_logit:.7}, margin {margin:.7}"
    );
    let ok = token == EXPECTED_TOKEN
        && runner == EXPECTED_RUNNER
        && (logit - EXPECTED_LOGIT).abs() <= TOKEN_TOLERANCE
        && (runner_logit - EXPECTED_RUNNER_LOGIT).abs() <= TOKEN_TOLERANCE
        && (margin - EXPECTED_MARGIN).abs() <= TOKEN_TOLERANCE;
    if !ok {
        eprintln!(
            "{name}: expected token {EXPECTED_TOKEN} logit {EXPECTED_LOGIT}, runner {EXPECTED_RUNNER} logit {EXPECTED_RUNNER_LOGIT}, margin {EXPECTED_MARGIN}"
        );
    }
    ok
}

fn materialize<B: burn::tensor::backend::Backend>(
    specs: &[LayerSpec],
    norm: &(Vec<f32>, Vec<f32>),
    lm_head: &[f32],
    device: &B::Device,
) -> (
    Vec<LayerWeights<B>>,
    Tensor<B, 2>,
    Tensor<B, 2>,
    Tensor<B, 2>,
) {
    let weights = specs
        .iter()
        .map(|spec| spec.materialize::<B>(device))
        .collect();
    (
        weights,
        vector(&norm.0, device),
        vector(&norm.1, device),
        Tensor::from_data(TensorData::new(lm_head.to_vec(), [65_536, HIDDEN]), device),
    )
}

fn run<B: burn::tensor::backend::Backend>(
    weights: &[LayerWeights<B>],
    norm: &(Tensor<B, 2>, Tensor<B, 2>),
    lm_head: &Tensor<B, 2>,
    embeddings: &[Vec<f32>; 2],
    device: &B::Device,
    step: Step<B>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut states: Vec<LayerState<B>> = (0..LAYER_COUNT)
        .map(|_| LayerState::<B>::zeros(device))
        .collect();
    let mut final_hidden = Vec::new();
    for (token, values) in embeddings.iter().enumerate() {
        let mut hidden = vector(values, device);
        let mut anchor = None;
        for (layer, layer_weights) in weights.iter().enumerate() {
            let (next, state, next_anchor) =
                step(layer_weights, hidden, &states[layer], layer == 0, anchor);
            hidden = next;
            states[layer] = state;
            anchor = next_anchor;
        }
        if token == embeddings.len() - 1 {
            final_hidden = to_vec(layer_norm(hidden, norm.0.clone(), norm.1.clone()));
        }
    }
    let mut recurrent = Vec::with_capacity(LAYER_COUNT * 12 * 64 * 64);
    for state in states {
        recurrent.extend(to_vec(state.matrix));
    }
    let logits = to_vec(vector(&final_hidden, device).matmul(lm_head.clone().transpose()));
    (final_hidden, logits, recurrent)
}

fn run_flex(
    specs: &[LayerSpec],
    norm_values: &(Vec<f32>, Vec<f32>),
    lm_head: &[f32],
    embeddings: &[Vec<f32>; 2],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let device = FlexDevice;
    let (weights, norm_w, norm_b, head) = materialize::<Flex>(specs, norm_values, lm_head, &device);
    run(
        &weights,
        &(norm_w, norm_b),
        &head,
        embeddings,
        &device,
        layer_step::<Flex>,
    )
}

fn run_tt(
    specs: &[LayerSpec],
    norm_values: &(Vec<f32>, Vec<f32>),
    lm_head: &[f32],
    embeddings: &[Vec<f32>; 2],
    card: u16,
    fused: bool,
    tiles: Option<usize>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let device = TtDevice::new(card);
    let config = match tiles {
        Some(n) => TtConfig::default().tiles(TileChoice::Count(n)),
        None => TtConfig::default(),
    };
    burn_tt::init(device, config).expect("attach card");
    let step: Step<TtBackend> = if fused {
        layer_step_fused
    } else {
        layer_step::<TtBackend>
    };
    accurate_matmul(|| {
        let (weights, norm_w, norm_b, head) =
            materialize::<TtBackend>(specs, norm_values, lm_head, &device);
        run(
            &weights,
            &(norm_w, norm_b),
            &head,
            embeddings,
            &device,
            step,
        )
    })
}

fn read_f32(path: &std::path::Path) -> Result<Vec<f32>, String> {
    let bytes = std::fs::read(path).map_err(|error| format!("{}: {error}", path.display()))?;
    if bytes.len() % 4 != 0 {
        return Err(format!("{} is not a multiple of 4 bytes", path.display()));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn main() -> std::process::ExitCode {
    let mut card = None;
    let mut checkpoint_path = None;
    let mut oracle = None;
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
            "--checkpoint" => {
                let Some(value) = args.next() else {
                    eprintln!("--checkpoint needs a safetensors path");
                    return std::process::ExitCode::FAILURE;
                };
                checkpoint_path = Some(value);
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
            "--oracle" => {
                let Some(value) = args.next() else {
                    eprintln!("--oracle needs a reference directory");
                    return std::process::ExitCode::FAILURE;
                };
                oracle = Some(value);
            }
            "-h" | "--help" => {
                println!(
                    "usage: rwkv7-model-gate --checkpoint PATH [--oracle DIR] [--card N] [--fused] [--tiles N]"
                );
                println!("  --fused  the WKV7 recurrence through the fused kernel");
                println!("  --tiles  Tensix tiles to open (default 12, the model's heads)");
                return std::process::ExitCode::SUCCESS;
            }
            other => {
                eprintln!(
                    "unknown argument {other}\n\nusage: rwkv7-model-gate --checkpoint PATH [--oracle DIR] [--card N] [--fused] [--tiles N]"
                );
                return std::process::ExitCode::FAILURE;
            }
        }
    }
    let Some(path) = checkpoint_path else {
        eprintln!("--checkpoint PATH is required");
        return std::process::ExitCode::FAILURE;
    };

    let checkpoint = match Checkpoint::open(std::path::Path::new(&path)) {
        Ok(checkpoint) => checkpoint,
        Err(error) => {
            eprintln!("checkpoint: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let specs = match (0..LAYER_COUNT)
        .map(|layer| checkpoint.layer_spec(layer, None))
        .collect::<Result<Vec<_>, String>>()
    {
        Ok(specs) => specs,
        Err(error) => {
            eprintln!("checkpoint layers: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let embeddings = match (checkpoint.embedding(1), checkpoint.embedding(2)) {
        (Ok(first), Ok(second)) => [first, second],
        (Err(error), _) | (_, Err(error)) => {
            eprintln!("checkpoint embeddings: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let norm_values = match checkpoint.final_norm() {
        Ok(norm) => norm,
        Err(error) => {
            eprintln!("checkpoint final norm: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let lm_head = match checkpoint.lm_head() {
        Ok(head) => head,
        Err(error) => {
            eprintln!("checkpoint lm_head: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };

    let (flex_hidden, flex_logits, flex_states) =
        run_flex(&specs, &norm_values, &lm_head, &embeddings);

    let mut control = specs.clone();
    control[0].r_proj = vec![0.0; control[0].r_proj.len()];
    let (control_hidden, _, _) = run_flex(&control, &norm_values, &lm_head, &embeddings);
    let control_gap = max_abs(&control_hidden, &flex_hidden);
    println!(
        "flex: final hidden max {:.6}, logits max {:.6}, recurrent max {:.6}; control gap {:.6}",
        flex_hidden.iter().map(|v| v.abs()).fold(0.0, f32::max),
        flex_logits.iter().map(|v| v.abs()).fold(0.0, f32::max),
        flex_states.iter().map(|v| v.abs()).fold(0.0, f32::max),
        control_gap
    );
    if control_gap <= 1.0e-4 {
        eprintln!("negative control did not move the final hidden: {control_gap}");
        return std::process::ExitCode::FAILURE;
    }
    if !check_greedy("flex greedy", &flex_logits) {
        return std::process::ExitCode::FAILURE;
    }

    if let Some(directory) = oracle.as_deref() {
        let root = std::path::Path::new(directory);
        let reference_hidden = match read_f32(&root.join("final_hidden.f32")) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("oracle hidden: {error}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let reference_logits = match read_f32(&root.join("logits.f32")) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("oracle logits: {error}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let reference_states = match read_f32(&root.join("states.f32")) {
            Ok(values) => values,
            Err(error) => {
                eprintln!("oracle states: {error}");
                return std::process::ExitCode::FAILURE;
            }
        };
        for (name, ours, reference, bound) in [
            (
                "flex hidden vs oracle",
                &flex_hidden,
                &reference_hidden,
                0.5,
            ),
            (
                "flex logits vs oracle",
                &flex_logits,
                &reference_logits,
                1.0,
            ),
            (
                "flex states vs oracle",
                &flex_states,
                &reference_states,
                1.0,
            ),
        ] {
            if ours.len() != reference.len() {
                eprintln!("{name}: length {} != {}", ours.len(), reference.len());
                return std::process::ExitCode::FAILURE;
            }
            let gap = max_abs(ours, reference);
            println!("{name}: gap {gap:.3e} (bound {bound:.1e})");
            if !gap.is_finite() || gap > bound {
                eprintln!("{name}: gap {gap:.3e} exceeds {bound:.1e}");
                return std::process::ExitCode::FAILURE;
            }
        }
    }

    let Some(card) = card else {
        println!("flex model PASS; pass --card N to run the device gate");
        return std::process::ExitCode::SUCCESS;
    };
    let (tt_hidden, tt_logits, tt_states) = run_tt(
        &specs,
        &norm_values,
        &lm_head,
        &embeddings,
        card,
        fused,
        tiles,
    );
    let scale = |values: &[f32]| values.iter().map(|v| v.abs()).fold(0.0, f32::max).max(1.0);
    let gaps = [
        (
            "final hidden",
            max_abs(&tt_hidden, &flex_hidden),
            scale(&flex_hidden),
        ),
        (
            "logits",
            max_abs(&tt_logits, &flex_logits),
            scale(&flex_logits),
        ),
        (
            "recurrent state",
            max_abs(&tt_states, &flex_states),
            scale(&flex_states),
        ),
    ];
    for (name, gap, scale) in gaps {
        let bound = RELATIVE_TOLERANCE * scale;
        println!(
            "card {card}: {name} gap {gap:.3e} (max {scale:.3e}, relative {:.3e})",
            gap / scale
        );
        if !gap.is_finite() || gap > bound {
            eprintln!("{name}: gap {gap:.3e} exceeds {bound:.3e}");
            return std::process::ExitCode::FAILURE;
        }
    }
    if !check_greedy("tt greedy", &tt_logits) {
        return std::process::ExitCode::FAILURE;
    }
    if let Some(directory) = oracle.as_deref() {
        let root = std::path::Path::new(directory);
        let reference_hidden = read_f32(&root.join("final_hidden.f32")).expect("hidden");
        let reference_logits = read_f32(&root.join("logits.f32")).expect("logits");
        let reference_states = read_f32(&root.join("states.f32")).expect("states");
        for (name, ours, reference, bound) in [
            ("tt hidden vs oracle", &tt_hidden, &reference_hidden, 0.5),
            ("tt logits vs oracle", &tt_logits, &reference_logits, 1.0),
            ("tt states vs oracle", &tt_states, &reference_states, 1.0),
        ] {
            let gap = max_abs(ours, reference);
            println!("{name}: gap {gap:.3e} (bound {bound:.1e})");
            if !gap.is_finite() || gap > bound {
                eprintln!("{name}: gap {gap:.3e} exceeds {bound:.1e}");
                return std::process::ExitCode::FAILURE;
            }
        }
    }
    println!("model gate PASS");
    std::process::ExitCode::SUCCESS
}
