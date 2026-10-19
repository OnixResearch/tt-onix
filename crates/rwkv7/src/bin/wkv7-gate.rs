//! WKV7 correctness gate.
//!
//! Host reference always runs. With `--card N`, the same two-token sequence runs
//! on the card through `burn-tt` and is compared element by element.

use std::process::ExitCode;

use burn_tt::{TtConfig, TtDevice};
use rwkv7::{step, step_device_attached, Inputs, HEAD_COUNT, HEAD_SIZE};

const TOLERANCE: f32 = 1e-5;

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    let unit = ((*seed >> 40) as f32) / ((1u32 << 24) as f32);
    unit * 2.0 - 1.0
}

fn inputs(seed: u64, state: Option<Vec<f32>>) -> Inputs {
    let mut seed = seed;
    let mut values =
        |n: usize, scale: f32| (0..n).map(|_| lcg(&mut seed) * scale).collect::<Vec<_>>();
    Inputs {
        r: values(HEAD_COUNT * HEAD_SIZE, 0.5),
        w: values(HEAD_COUNT * HEAD_SIZE, 0.5),
        k: values(HEAD_COUNT * HEAD_SIZE, 0.25),
        v: values(HEAD_COUNT * HEAD_SIZE, 0.25),
        a: values(HEAD_COUNT * HEAD_SIZE, 0.25),
        b: values(HEAD_COUNT * HEAD_SIZE, 0.25),
        state: state.unwrap_or_else(|| values(HEAD_COUNT * HEAD_SIZE * HEAD_SIZE, 0.25)),
    }
}

fn max_abs(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn main() -> ExitCode {
    let mut card = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--card" => {
                let Some(value) = args.next() else {
                    eprintln!("--card needs a device index");
                    return ExitCode::FAILURE;
                };
                match value.parse::<u16>() {
                    Ok(value) => card = Some(value),
                    Err(_) => {
                        eprintln!("--card {value}: not a number");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "-h" | "--help" => {
                println!("usage: wkv7-gate [--card N]");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument {other}\n\nusage: wkv7-gate [--card N]");
                return ExitCode::FAILURE;
            }
        }
    }

    let first = inputs(7, None);
    let host_first = match step(&first) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("host step 1 failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let second = inputs(13, Some(host_first.state.clone()));
    let host_second = match step(&second) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("host step 2 failed: {error}");
            return ExitCode::FAILURE;
        }
    };

    let mut zero_a = first.clone();
    zero_a.a = vec![0.0; zero_a.a.len()];
    let control = step(&zero_a).expect("control step");
    let control_gap = max_abs(&control.out, &host_first.out);
    if control_gap <= 1e-3 {
        eprintln!("negative control did not move the output: {control_gap}");
        return ExitCode::FAILURE;
    }
    println!(
        "host reference: step1 out {:.6}, state {:.6}; step2 out {:.6}; control gap {:.6}",
        host_first.out.iter().map(|v| v.abs()).fold(0.0, f32::max),
        host_first.state.iter().map(|v| v.abs()).fold(0.0, f32::max),
        host_second.out.iter().map(|v| v.abs()).fold(0.0, f32::max),
        control_gap
    );

    let Some(card) = card else {
        println!("host reference PASS; pass --card N to run the device gate");
        return ExitCode::SUCCESS;
    };

    let device = TtDevice::new(card);
    if let Err(error) = burn_tt::init(device, TtConfig::default()) {
        eprintln!("attach card {card}: {error}");
        return ExitCode::FAILURE;
    }
    let device_first = match step_device_attached(device, &first) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("device step 1 failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let device_second = match step_device_attached(device, &second) {
        Ok(output) => output,
        Err(error) => {
            eprintln!("device step 2 failed: {error}");
            return ExitCode::FAILURE;
        }
    };

    let out_gap = max_abs(&device_first.out, &host_first.out);
    let state_gap = max_abs(&device_first.state, &host_first.state);
    let carry_gap = max_abs(&device_second.out, &host_second.out);
    let carry_state_gap = max_abs(&device_second.state, &host_second.state);
    println!(
        "card {card}: step1 out gap {out_gap:.3e}, state gap {state_gap:.3e}; \
         step2 out gap {carry_gap:.3e}, state gap {carry_state_gap:.3e}"
    );
    for (name, gap) in [
        ("step1 out", out_gap),
        ("step1 state", state_gap),
        ("step2 out (carried)", carry_gap),
        ("step2 state (carried)", carry_state_gap),
    ] {
        if !gap.is_finite() || gap > TOLERANCE {
            eprintln!("{name}: gap {gap:.3e} exceeds {TOLERANCE:.1e}");
            return ExitCode::FAILURE;
        }
    }
    println!("device gate PASS");
    ExitCode::SUCCESS
}
