//! WKV7 through `burn-tt`: the routed op, not the raw role kernel.
//!
//! `burn_tt::wkv7_step` sends one device job -- gather, the three-role
//! `tt_kernels::wkv7` kernel, scatter -- through the attached engine and
//! returns one `[64, 96]` tensor. `rwkv7::step_head_device_attached` builds the
//! head's tensors and splits the result; `step_device_fused_attached` runs all
//! twelve heads, one launch a head. Both gates compare with the host
//! recurrence across a carried second token; `step186_wkv7` covers the same
//! kernel's role streams and its `Session` form directly.
//!
//! ttsim and card 0: `with_device` picks the engine, so one gate proves both.

use burn_tt::TtDevice;
use rwkv7::{
    step, step_device_fused_attached, step_head_device_attached, Inputs, HEAD_COUNT, HEAD_SIZE,
};
use tt_tests::burn_device::{with_device, Config};

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
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

fn check(what: &str, got: &[f32], want: &[f32]) {
    let gap = max_abs(got, want);
    println!("{what}: max |gap| {gap:.3e}");
    assert!(
        gap.is_finite() && gap <= TOLERANCE,
        "{what}: gap {gap:.3e} exceeds {TOLERANCE:.1e}"
    );
}

fn run(device: TtDevice, inputs: &Inputs) -> rwkv7::HeadOutputs {
    step_head_device_attached(device, inputs, 0).expect("fused head step")
}

#[test]
fn one_head_wkv7_through_burn_tt_matches_the_host_reference() {
    let first = inputs(7, None);
    let host_first = step(&first).expect("host step 1");
    with_device(Config::default(), |device| {
        let got = run(device, &first);
        check(
            "step 1 state",
            &got.state,
            &host_first.state[..HEAD_SIZE * HEAD_SIZE],
        );
        check("step 1 out", &got.out, &host_first.out[..HEAD_SIZE]);

        // The second token starts from the state the routed op produced.
        let mut second = inputs(13, None);
        second.state[..HEAD_SIZE * HEAD_SIZE].copy_from_slice(&got.state);
        let host_second = step(&second).expect("host step 2");
        let got = run(device, &second);
        check(
            "step 2 state (carried)",
            &got.state,
            &host_second.state[..HEAD_SIZE * HEAD_SIZE],
        );
        check(
            "step 2 out (carried)",
            &got.out,
            &host_second.out[..HEAD_SIZE],
        );
    });
}

/// The model-level path: all twelve heads, one kernel launch a head, with every
/// head's state carried into the second token.
#[test]
fn all_twelve_heads_through_burn_tt_match_the_host_reference() {
    let first = inputs(7, None);
    let host_first = step(&first).expect("host step 1");
    with_device(Config::default(), |device| {
        let got = step_device_fused_attached(device, &first).expect("fused token 1");
        check(
            "12 heads: step 1 state",
            &got.state,
            &host_first.state[..HEAD_COUNT * HEAD_SIZE * HEAD_SIZE],
        );
        check(
            "12 heads: step 1 out",
            &got.out,
            &host_first.out[..HEAD_COUNT * HEAD_SIZE],
        );

        // Every head starts the next token from what the kernel produced.
        let mut second = inputs(13, None);
        second
            .state
            .copy_from_slice(&got.state[..HEAD_COUNT * HEAD_SIZE * HEAD_SIZE]);
        let host_second = step(&second).expect("host step 2");
        let got = step_device_fused_attached(device, &second).expect("fused token 2");
        check(
            "12 heads: step 2 state (carried)",
            &got.state,
            &host_second.state[..HEAD_COUNT * HEAD_SIZE * HEAD_SIZE],
        );
        check(
            "12 heads: step 2 out (carried)",
            &got.out,
            &host_second.out[..HEAD_COUNT * HEAD_SIZE],
        );
    });
}
