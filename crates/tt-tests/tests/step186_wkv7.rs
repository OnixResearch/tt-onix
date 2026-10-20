//! One-head WKV7: the fused SFPU kernel against the host reference.
//!
//! `crates/tt-kernels/src/wkv7.rs` computes one head's token step in one
//! Tensix programme: the two matrix-vector products are in-tile column folds,
//! and the state update is two replayed row loops. The oracle is the host
//! recurrence in `rwkv7::step`, run on twelve heads; this gate stages one of
//! them and holds the device's head to the reference's.
//!
//! Claims:
//! - the device's updated state and output equal the host reference within
//!   `TOLERANCE`, on random one-head data; every element is checked, not a
//!   summary;
//! - **state carries through the kernel**: a second token, staged with the
//!   state the device produced, matches the host recurrence started from that
//!   same state;
//! - **the rank-1 term reaches the arithmetic**: zeroing `a` moves the output
//!   by more than `CONTROL_GAP` and the device follows the zeroed host
//!   reference (a kernel that ignored `a` would fail both).
//!
//! The same gate runs on ttsim and on card 0 (`cargo xtask silicon --filter
//! step186_wkv7`). The burn-tt routing is the next gate and is not claimed
//! here.

use tt_isa::dm::TILE_DATA;
use tt_kernels::matmul::{detilize_packed, tilize_into, TILE_IMAGE_BYTES};
use tt_kernels::wkv7::{self, Layout};
use tt_tests::harness::{self, Roles, Run};

use rwkv7::{step, Inputs, HEAD_COUNT, HEAD_SIZE};

/// The host reference's recurrence is left-to-right F32; the kernel's folds
/// are a tree and its update has two fused multiply-adds. With these inputs
/// the measured gap is far below this bound.
const TOLERANCE: f32 = 1e-5;
/// A kernel that dropped the rank-1 term would differ by at least this much on
/// the control data.
const CONTROL_GAP: f32 = 1e-3;

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

/// The tile images of a row-major `[rows, cols]` F32 matrix.
fn tiles(values: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    let count = rows.div_ceil(32) * cols.div_ceil(32);
    let mut out = vec![0u8; count * TILE_IMAGE_BYTES];
    tilize_into(
        |i| values[i].to_bits(),
        rows,
        cols,
        0..count,
        &mut out,
        TILE_IMAGE_BYTES,
    );
    out
}

/// A 32x32 tile with every column equal to `value`.
fn column_tile(value: &[f32]) -> Vec<u8> {
    let values: Vec<f32> = (0..1024).map(|i| value[i / 32]).collect();
    tiles(&values, 32, 32)
}

/// One head's operands, as the host reference holds them.
struct Head {
    r: Vec<f32>,
    w: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    state: Vec<f32>,
}

impl Head {
    fn of(inputs: &Inputs) -> Self {
        let d = HEAD_SIZE;
        Head {
            r: inputs.r[..d].to_vec(),
            w: inputs.w[..d].to_vec(),
            k: inputs.k[..d].to_vec(),
            v: inputs.v[..d].to_vec(),
            a: inputs.a[..d].to_vec(),
            b: inputs.b[..d].to_vec(),
            state: inputs.state[..d * d].to_vec(),
        }
    }
}

/// What a kernel run produced: the updated state, row-major `[64, 64]`, and the
/// 64 outputs.
struct Device {
    state: Vec<f32>,
    out: Vec<f32>,
}

/// Stage `head` into `layout` and run the three roles once.
fn run_head(dev: &mut harness::Dev<'_>, layout: &Layout, head: &Head) -> Device {
    let state = tiles(&head.state, 64, 64);
    let mut images: Vec<(u64, Vec<u8>)> = Vec::new();
    for (k, slot) in layout.state_in.iter().enumerate() {
        images.push((
            *slot,
            state[k * TILE_IMAGE_BYTES..(k + 1) * TILE_IMAGE_BYTES].to_vec(),
        ));
    }
    let mut vector = |slot: &[u64; 2], value: &[f32], columns: bool| {
        for (half, at) in slot.iter().enumerate() {
            let part = &value[half * 32..(half + 1) * 32];
            images.push((
                *at,
                if columns {
                    column_tile(part)
                } else {
                    // A row vector: one tile with its 32 values in row 0.
                    tiles(part, 1, 32)
                },
            ));
        }
    };
    vector(&layout.a, &head.a, false);
    vector(&layout.w, &head.w, false);
    vector(&layout.b, &head.b, false);
    vector(&layout.k, &head.k, false);
    vector(&layout.r, &head.r, false);
    vector(&layout.v, &head.v, true);

    let stage: Vec<(u64, &[u8])> = images.iter().map(|(at, image)| (*at, &image[..])).collect();
    let mut reads: Vec<(u64, usize)> = Vec::new();
    for slot in layout.state_out.iter().chain(layout.out.iter()) {
        reads.push((slot + TILE_DATA, 4096));
    }
    let roles = wkv7::roles(layout);
    let output = harness::run(
        dev,
        &Run::roles(Roles {
            unpack: &roles[0],
            math: &roles[1],
            pack: &roles[2],
        })
        .concurrent(&layout.init)
        .stage(&stage)
        .dump_rows(0)
        .read_back(&reads),
    );

    let mut packed = Vec::new();
    for bytes in &output.l1[..4] {
        packed.extend_from_slice(bytes);
    }
    let state = detilize_packed(&packed, 64, 64);
    let mut packed = Vec::new();
    for bytes in &output.l1[4..] {
        packed.extend_from_slice(bytes);
    }
    let columns = detilize_packed(&packed, 64, 32);
    let out = (0..64).map(|row| columns[row * 32]).collect();
    Device { state, out }
}

fn check(what: &str, got: &[f32], want: &[f32]) {
    let gap = max_abs(got, want);
    println!("{what}: max |gap| {gap:.3e}");
    assert!(
        gap.is_finite() && gap <= TOLERANCE,
        "{what}: gap {gap:.3e} exceeds {TOLERANCE:.1e}"
    );
}

/// The same step through the GDDR-resident path a backend uses: upload the
/// head as tensors, one `Session::wkv7_step` (gather, kernel, scatter), and
/// download the combined result.
#[test]
fn the_session_op_matches_the_host_reference() {
    use tt_kernels::session::TileChoice;
    use tt_tests::backend::{with_session, GATE_TILE};

    let first = inputs(7, None);
    let host = step(&first).expect("host step");
    let head = Head::of(&first);
    with_session(TileChoice::Exactly(GATE_TILE.0, GATE_TILE.1), |s| {
        s.enable_dram(tt_firmware_images::DM_B.1, tt_firmware_images::DM_NC.1)
            .expect("enable GDDR");
        let state = s.upload(&head.state, 64, 64).expect("state");
        let r = s.upload(&head.r, 1, 64).expect("r");
        let w = s.upload(&head.w, 1, 64).expect("w");
        let k = s.upload(&head.k, 1, 64).expect("k");
        let v = s.upload(&head.v, 64, 1).expect("v");
        let a = s.upload(&head.a, 1, 64).expect("a");
        let b = s.upload(&head.b, 1, 64).expect("b");
        let combined = s
            .wkv7_step(&state, &r, &w, &k, &v, &a, &b)
            .expect("one-head WKV7 step");
        assert_eq!([combined.rows, combined.cols], [64, 96]);
        let values = s.download(&combined).expect("download");
        let (state, out) = wkv7::split_step(&values);
        check(
            "session state",
            &state,
            &host.state[..HEAD_SIZE * HEAD_SIZE],
        );
        check("session out", &out, &host.out[..HEAD_SIZE]);
    });
}

#[test]
fn one_head_wkv7_step_matches_the_host_reference_and_carries_its_state() {
    let layout = wkv7::plan_layout().expect("one head fits the data arena");
    let first = inputs(7, None);
    let host_first = step(&first).expect("host step 1");
    let head = Head::of(&first);

    // The control: `a` zeroed. Its output must move, and the device must move
    // with the host reference.
    let mut no_rank = Head::of(&first);
    no_rank.a = vec![0.0; HEAD_SIZE];
    let mut control_inputs = first.clone();
    control_inputs.a[..HEAD_SIZE].fill(0.0);
    let host_control = step(&control_inputs).expect("host control");

    harness::in_device(|dev| {
        let device_first = run_head(dev, &layout, &head);
        check(
            "step 1 state",
            &device_first.state,
            &host_first.state[..HEAD_SIZE * HEAD_SIZE],
        );
        check(
            "step 1 out",
            &device_first.out,
            &host_first.out[..HEAD_SIZE],
        );

        // The second token starts from what the device produced.
        let mut second = inputs(13, None);
        second.state[..HEAD_SIZE * HEAD_SIZE].copy_from_slice(&device_first.state);
        let host_second = step(&second).expect("host step 2");
        let device_second = run_head(dev, &layout, &Head::of(&second));
        check(
            "step 2 state (carried)",
            &device_second.state,
            &host_second.state[..HEAD_SIZE * HEAD_SIZE],
        );
        check(
            "step 2 out (carried)",
            &device_second.out,
            &host_second.out[..HEAD_SIZE],
        );

        let device_control = run_head(dev, &layout, &no_rank);
        let moved = max_abs(&device_control.out, &device_first.out);
        println!("control: max |device - correct out| {moved:.3e}");
        assert!(
            moved > CONTROL_GAP,
            "zeroing a moved the output by only {moved:.3e}: the rank-1 term is not reaching \
             the arithmetic"
        );
        check(
            "control state",
            &device_control.state,
            &host_control.state[..HEAD_SIZE * HEAD_SIZE],
        );
        check(
            "control out",
            &device_control.out,
            &host_control.out[..HEAD_SIZE],
        );
    });
}
