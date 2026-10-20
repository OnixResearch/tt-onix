//! Stateful decode chain against the harness decode checkpoint.
//!
//! Seed token 1, three greedy steps. The harness expects token 1 at every
//! step with logits 0.04493069, 6.9543834, 6.486726.

use burn::tensor::{Tensor, TensorData};
use burn_flex::{Flex, FlexDevice};
use burn_tt::{accurate_matmul, TileChoice, TtBackend, TtConfig, TtDevice};
use rwkv7::checkpoint::Checkpoint;
use rwkv7::layer::{
    layer_norm, layer_step, layer_step_chunk, layer_step_fused, LayerSpec, LayerState,
    LayerWeights, HIDDEN,
};
use rwkv7::tokenizer::Tokenizer;

const LAYER_COUNT: usize = 12;
const SEED_TOKEN: u32 = 1;
const STEPS: usize = 3;
const EXPECTED_TOKENS: [u32; STEPS] = [1, 1, 1];
const EXPECTED_LOGITS: [f32; STEPS] = [0.044_930_69, 6.954_383_4, 6.486_726];
const TOKEN_TOLERANCE: f32 = 1.0e-2;
/// The fast matmul path's logit bound: the accurate path meets 1e-2, the fast
/// one's largest measured step difference is 1.9e-2 (step 0, logit 0.026
/// against 0.045), and its tokens are the same at every step.
const FAST_TOLERANCE: f32 = 5.0e-2;
const FIXED_PROMPT: &str = "<|rwkv_tokenizer_end_of_text|>User: Hi\n\nAssistant:";
const EXPECTED_TEXT: &str = " Hi, I";
const EXPECTED_STEPS: usize = 8;
const EXPECTED_TEXT_8: &str = " Hi, I'm sorry to hear that";

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

fn token_step<B: burn::tensor::backend::Backend>(
    weights: &[LayerWeights<B>],
    states: &mut [LayerState<B>],
    embedding: &[f32],
    device: &B::Device,
    step: Step<B>,
) -> Tensor<B, 2> {
    let mut hidden = vector(embedding, device);
    let mut anchor = None;
    for (layer, layer_weights) in weights.iter().enumerate() {
        let (next, state, next_anchor) =
            step(layer_weights, hidden, &states[layer], layer == 0, anchor);
        hidden = next;
        states[layer] = state;
        anchor = next_anchor;
    }
    hidden
}

fn eval_logits<B: burn::tensor::backend::Backend>(
    hidden: Tensor<B, 2>,
    norm: &(Tensor<B, 2>, Tensor<B, 2>),
    lm_head: &Tensor<B, 2>,
) -> Vec<f32> {
    let normalized = layer_norm(hidden, norm.0.clone(), norm.1.clone());
    to_vec(normalized.matmul(lm_head.clone().transpose()))
}

#[allow(clippy::too_many_arguments)]
fn generate<B: burn::tensor::backend::Backend>(
    checkpoint: &Checkpoint,
    weights: &[LayerWeights<B>],
    norm: &(Tensor<B, 2>, Tensor<B, 2>),
    lm_head: &Tensor<B, 2>,
    prompt_ids: &[u32],
    steps: usize,
    device: &B::Device,
    step: Step<B>,
) -> Result<Vec<u32>, String> {
    let mut states: Vec<LayerState<B>> = (0..LAYER_COUNT)
        .map(|_| LayerState::<B>::zeros(device))
        .collect();
    let mut hidden = None;
    for &id in prompt_ids {
        let embedding = checkpoint.embedding(id as usize)?;
        hidden = Some(token_step(weights, &mut states, &embedding, device, step));
        B::sync(device).map_err(|error| format!("{error:?}"))?;
    }
    let Some(mut hidden) = hidden else {
        return Err("prompt has no tokens".into());
    };
    let mut generated = Vec::with_capacity(steps);
    for index in 0..steps {
        let logits = eval_logits(hidden.clone(), norm, lm_head);
        let (token, _, _, _) = top_two(&logits);
        generated.push(token);
        if index + 1 < steps {
            let embedding = checkpoint.embedding(token as usize)?;
            hidden = token_step(weights, &mut states, &embedding, device, step);
            B::sync(device).map_err(|error| format!("{error:?}"))?;
        }
    }
    Ok(generated)
}

/// The prompt as batch layer steps: every token-wise projection and the FFN is
/// a `[chunk, HIDDEN]` matmul, the WKV7 recurrence stays sequential. The
/// generated tokens then use the one-token fused step.
#[allow(clippy::too_many_arguments)]
fn generate_chunked(
    checkpoint: &Checkpoint,
    weights: &[LayerWeights<TtBackend>],
    norm: &(Tensor<TtBackend, 2>, Tensor<TtBackend, 2>),
    lm_head: &Tensor<TtBackend, 2>,
    prompt_ids: &[u32],
    steps: usize,
    device: &TtDevice,
    chunk: usize,
) -> Result<Vec<u32>, String> {
    let mut states: Vec<LayerState<TtBackend>> = (0..LAYER_COUNT)
        .map(|_| LayerState::<TtBackend>::zeros(device))
        .collect();
    let mut last = None;
    for group in prompt_ids.chunks(chunk) {
        let mut values = Vec::with_capacity(group.len() * HIDDEN);
        for &id in group {
            values.extend(checkpoint.embedding(id as usize)?);
        }
        let mut hidden = Tensor::from_data(TensorData::new(values, [group.len(), HIDDEN]), device);
        let mut anchor = None;
        for (layer, layer_weights) in weights.iter().enumerate() {
            let (next, state, next_anchor) =
                layer_step_chunk(layer_weights, hidden, &states[layer], layer == 0, anchor);
            hidden = next;
            states[layer] = state;
            anchor = next_anchor;
        }
        last = Some(hidden);
    }
    let Some(last) = last else {
        return Err("prompt has no tokens".into());
    };
    let rows = last.dims()[0];
    let mut hidden = last.slice([rows - 1..rows, 0..HIDDEN]);
    let mut generated = Vec::with_capacity(steps);
    for index in 0..steps {
        let logits = eval_logits(hidden.clone(), norm, lm_head);
        let (token, _, _, _) = top_two(&logits);
        generated.push(token);
        if index + 1 < steps {
            let embedding = checkpoint.embedding(token as usize)?;
            hidden = token_step(weights, &mut states, &embedding, device, layer_step_fused);
        }
    }
    Ok(generated)
}

fn decode<B: burn::tensor::backend::Backend>(
    checkpoint: &Checkpoint,
    weights: &[LayerWeights<B>],
    norm: &(Tensor<B, 2>, Tensor<B, 2>),
    lm_head: &Tensor<B, 2>,
    device: &B::Device,
    step: Step<B>,
) -> Result<Vec<(u32, f32)>, String> {
    let mut states: Vec<LayerState<B>> = (0..LAYER_COUNT)
        .map(|_| LayerState::<B>::zeros(device))
        .collect();
    let mut input = SEED_TOKEN;
    let mut output = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let embedding = checkpoint.embedding(input as usize)?;
        let mut hidden = vector(&embedding, device);
        let mut anchor = None;
        for (layer, layer_weights) in weights.iter().enumerate() {
            let (next, state, next_anchor) =
                step(layer_weights, hidden, &states[layer], layer == 0, anchor);
            hidden = next;
            states[layer] = state;
            anchor = next_anchor;
        }
        let normalized = layer_norm(hidden, norm.0.clone(), norm.1.clone());
        let logits = to_vec(normalized.matmul(lm_head.clone().transpose()));
        let (token, logit, _, _) = top_two(&logits);
        output.push((token, logit));
        input = token;
    }
    Ok(output)
}

fn run_flex(
    checkpoint: &Checkpoint,
    specs: &[LayerSpec],
    norm: &(Vec<f32>, Vec<f32>),
    lm_head: &[f32],
) -> Result<Vec<(u32, f32)>, String> {
    let device = FlexDevice;
    let (weights, norm_w, norm_b, head) = materialize::<Flex>(specs, norm, lm_head, &device);
    decode::<Flex>(
        checkpoint,
        &weights,
        &(norm_w, norm_b),
        &head,
        &device,
        layer_step::<Flex>,
    )
}

#[allow(clippy::too_many_arguments)]
fn run_tt(
    checkpoint: &Checkpoint,
    specs: &[LayerSpec],
    norm: &(Vec<f32>, Vec<f32>),
    lm_head: &[f32],
    card: u16,
    fused: bool,
    tiles: Option<usize>,
    fast: bool,
) -> Result<Vec<(u32, f32)>, String> {
    let device = TtDevice::new(card);
    let config = match tiles {
        Some(n) => TtConfig::default().tiles(TileChoice::Count(n)),
        None => TtConfig::default(),
    };
    burn_tt::init(device, config).map_err(|error| error.to_string())?;
    let step: Step<TtBackend> = if fused {
        layer_step_fused
    } else {
        layer_step::<TtBackend>
    };
    let run = || {
        let (weights, norm_w, norm_b, head) =
            materialize::<TtBackend>(specs, norm, lm_head, &device);
        decode::<TtBackend>(
            checkpoint,
            &weights,
            &(norm_w, norm_b),
            &head,
            &device,
            step,
        )
    };
    if fast {
        run()
    } else {
        accurate_matmul(run)
    }
}

fn check(name: &str, steps: &[(u32, f32)], tolerance: f32) -> bool {
    if steps.len() != STEPS {
        eprintln!("{name}: {} steps, expected {STEPS}", steps.len());
        return false;
    }
    let mut ok = true;
    for (index, &(token, logit)) in steps.iter().enumerate() {
        let expected = EXPECTED_LOGITS[index];
        println!(
            "{name} step {index}: token {token} logit {logit:.8} (expected token {} logit {expected:.8})",
            EXPECTED_TOKENS[index]
        );
        if token != EXPECTED_TOKENS[index] || (logit - expected).abs() > tolerance {
            ok = false;
        }
    }
    ok
}

fn main() -> std::process::ExitCode {
    let mut card = None;
    let mut checkpoint_path = None;
    let mut vocab = None;
    let mut prompt = None;
    let mut steps = STEPS;
    let mut fused = true;
    let mut tiles = Some(12);
    let mut fast = false;
    let mut chunk = 8usize;
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
            "--composed" => fused = false,
            "--fast" => fast = true,
            "--chunk" => {
                let Some(value) = args.next() else {
                    eprintln!("--chunk needs a token count");
                    return std::process::ExitCode::FAILURE;
                };
                match value.parse::<usize>() {
                    Ok(value) if value > 0 => chunk = value,
                    _ => {
                        eprintln!("--chunk {value}: expected a positive count");
                        return std::process::ExitCode::FAILURE;
                    }
                }
            }
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
            "--vocab" => {
                let Some(value) = args.next() else {
                    eprintln!("--vocab needs a vocabulary path");
                    return std::process::ExitCode::FAILURE;
                };
                vocab = Some(value);
            }
            "--prompt" => {
                let Some(value) = args.next() else {
                    eprintln!("--prompt needs a string");
                    return std::process::ExitCode::FAILURE;
                };
                prompt = Some(value);
            }
            "--generate" => {
                let Some(value) = args.next() else {
                    eprintln!("--generate needs a count");
                    return std::process::ExitCode::FAILURE;
                };
                match value.parse::<usize>() {
                    Ok(value) if value > 0 => steps = value,
                    _ => {
                        eprintln!("--generate {value}: expected a positive count");
                        return std::process::ExitCode::FAILURE;
                    }
                }
            }
            "-h" | "--help" => {
                println!(
                    "usage: rwkv7-generate-gate --checkpoint PATH [--vocab PATH --prompt TEXT --generate N] [--card N] [--fused | --composed] [--tiles N] [--fast] [--chunk N]"
                );
                println!("  --fused  the WKV7 recurrence through the fused kernel (default)");
                println!("  --composed  the Burn-op recurrence (the reference)");
                println!("  --tiles  Tensix tiles to open (default 12, the model's heads)");
                println!("  --fast   decode with fast matmuls (the prompt path already does)");
                return std::process::ExitCode::SUCCESS;
            }
            other => {
                eprintln!(
                    "unknown argument {other}\n\nusage: rwkv7-generate-gate --checkpoint PATH [--vocab PATH --prompt TEXT --generate N] [--card N] [--fused | --composed] [--tiles N] [--fast]"
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
    let norm = match checkpoint.final_norm() {
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

    if let Some(prompt) = prompt {
        let Some(vocab) = vocab else {
            eprintln!("--prompt needs --vocab PATH");
            return std::process::ExitCode::FAILURE;
        };
        let tokenizer = match Tokenizer::from_file(std::path::Path::new(&vocab)) {
            Ok(tokenizer) => tokenizer,
            Err(error) => {
                eprintln!("vocab: {error}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let prompt_ids = match tokenizer.encode_wrapper(&prompt) {
            Ok(ids) => ids,
            Err(error) => {
                eprintln!("prompt tokenization: {error}");
                return std::process::ExitCode::FAILURE;
            }
        };
        println!("prompt ids: {prompt_ids:?}");
        let run = |card: Option<u16>| -> Result<Vec<u32>, String> {
            match card {
                None => {
                    let device = FlexDevice;
                    let (weights, norm_w, norm_b, head) =
                        materialize::<Flex>(&specs, &norm, &lm_head, &device);
                    generate::<Flex>(
                        &checkpoint,
                        &weights,
                        &(norm_w, norm_b),
                        &head,
                        &prompt_ids,
                        steps,
                        &device,
                        layer_step::<Flex>,
                    )
                }
                Some(card) => {
                    let device = TtDevice::new(card);
                    let config = match tiles {
                        Some(n) => TtConfig::default().tiles(TileChoice::Count(n)),
                        None => TtConfig::default(),
                    };
                    burn_tt::init(device, config).map_err(|error| error.to_string())?;
                    let (weights, norm_w, norm_b, head) =
                        materialize::<TtBackend>(&specs, &norm, &lm_head, &device);
                    let step: Step<TtBackend> = if fused {
                        layer_step_fused
                    } else {
                        layer_step::<TtBackend>
                    };
                    let started = std::time::Instant::now();
                    let result = if fused && chunk > 1 {
                        generate_chunked(
                            &checkpoint,
                            &weights,
                            &(norm_w, norm_b),
                            &head,
                            &prompt_ids,
                            steps,
                            &device,
                            chunk,
                        )
                    } else {
                        if !fused && chunk > 1 {
                            eprintln!("--chunk needs the fused recurrence");
                            return Err("chunked prompt needs --fused".into());
                        }
                        generate::<TtBackend>(
                            &checkpoint,
                            &weights,
                            &(norm_w, norm_b),
                            &head,
                            &prompt_ids,
                            steps,
                            &device,
                            step,
                        )
                    };
                    println!("prompt generation: {:?} (chunk {chunk})", started.elapsed());
                    result
                }
            }
        };
        let flex_ids = match run(None) {
            Ok(ids) => ids,
            Err(error) => {
                eprintln!("flex generation: {error}");
                return std::process::ExitCode::FAILURE;
            }
        };
        let flex_bytes = tokenizer.decode_wrapper(&flex_ids).unwrap();
        let flex_text = String::from_utf8_lossy(&flex_bytes);
        println!("flex generated ids: {flex_ids:?}");
        println!("flex generated text: {flex_text:?}");
        let expected = match steps {
            STEPS => Some(EXPECTED_TEXT),
            EXPECTED_STEPS => Some(EXPECTED_TEXT_8),
            _ => None,
        };
        if let (FIXED_PROMPT, Some(expected)) = (prompt.as_str(), expected) {
            if flex_text != expected {
                eprintln!("flex text differs from the harness: expected {expected:?}");
                return std::process::ExitCode::FAILURE;
            }
        }
        if let Some(card) = card {
            let tt_ids = match run(Some(card)) {
                Ok(ids) => ids,
                Err(error) => {
                    eprintln!("device generation: {error}");
                    return std::process::ExitCode::FAILURE;
                }
            };
            let tt_bytes = tokenizer.decode_wrapper(&tt_ids).unwrap();
            let tt_text = String::from_utf8_lossy(&tt_bytes);
            println!("tt generated ids: {tt_ids:?}");
            println!("tt generated text: {tt_text:?}");
            let expected = match steps {
                STEPS => Some(EXPECTED_TEXT),
                EXPECTED_STEPS => Some(EXPECTED_TEXT_8),
                _ => None,
            };
            if let (FIXED_PROMPT, Some(expected)) = (prompt.as_str(), expected) {
                if tt_text != expected {
                    eprintln!("tt text differs from the harness: expected {expected:?}");
                    return std::process::ExitCode::FAILURE;
                }
            }
            if tt_ids != flex_ids {
                eprintln!("tt and flex selected different tokens");
                return std::process::ExitCode::FAILURE;
            }
        }
        println!("prompt gate PASS");
        return std::process::ExitCode::SUCCESS;
    }

    let flex = match run_flex(&checkpoint, &specs, &norm, &lm_head) {
        Ok(steps) => steps,
        Err(error) => {
            eprintln!("flex decode: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    if !check("flex", &flex, TOKEN_TOLERANCE) {
        return std::process::ExitCode::FAILURE;
    }

    let Some(card) = card else {
        println!("flex decode PASS; pass --card N to run the device gate");
        return std::process::ExitCode::SUCCESS;
    };
    let tt = match run_tt(
        &checkpoint,
        &specs,
        &norm,
        &lm_head,
        card,
        fused,
        tiles,
        fast,
    ) {
        Ok(steps) => steps,
        Err(error) => {
            eprintln!("device decode: {error}");
            return std::process::ExitCode::FAILURE;
        }
    };
    let tolerance = if fast {
        FAST_TOLERANCE
    } else {
        TOKEN_TOLERANCE
    };
    if !check("tt", &tt, tolerance) {
        return std::process::ExitCode::FAILURE;
    }
    println!("decode gate PASS");
    std::process::ExitCode::SUCCESS
}
