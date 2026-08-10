//! Pure fused-equation reference and capability contract for the RWKV-7 WKV
//! kernel.
//!
//! r[impl tenstorrent.rwkv7_p150x2.fused_wkv_kernel]
//! r[impl tenstorrent.rwkv7_p150x2.fused_wkv_kernel.valid]
//! r[impl tenstorrent.rwkv7_p150x2.fused_wkv_kernel.invalid_input]
//!
//! This module is pure and device-free. It owns the accepted fused WKV
//! equation as a deterministic host reference, and it validates the typed
//! capability contract that gates fused dispatch. The device shell reads the
//! capability decision and never re-derives the equation.

use serde::{Deserialize, Serialize};

pub const WKV_HEAD_SIZE: usize = 64;
pub const WKV_HEAD_COUNT: usize = 12;
pub const WKV_HIDDEN_SIZE: usize = WKV_HEAD_COUNT * WKV_HEAD_SIZE;
pub const FUSED_ARCHITECTURE: &str = "blackhole";
pub const FUSED_STATE_DTYPE: &str = "float32";
pub const FUSED_VECTOR_DTYPE: &str = "bfloat16";
pub const FUSED_OUTPUT_DTYPE: &str = "bfloat16";
const FP32_BYTE_WIDTH: usize = 4;
const BF16_BYTE_WIDTH: usize = 2;
const MAXIMUM_SUPPORTED_BATCH: usize = 64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rejection {
    pub code: &'static str,
    pub detail: String,
}

impl Rejection {
    fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for Rejection {}

/// The typed capability contract for fused WKV dispatch.
///
/// The device shell validates this before creating any kernel program. An
/// unsupported field rejects the fused path and the runtime keeps the
/// admitted generic path.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FusedWkvCapability {
    pub architecture: String,
    pub head_count: usize,
    pub head_size: usize,
    pub batch_size: usize,
    pub state_dtype: String,
    pub vector_dtype: String,
    pub output_dtype: String,
    pub state_layout: String,
    pub state_ownership_plan: StateOwnershipPlan,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StateOwnershipPlan {
    /// One writer per (request, head) tile.
    pub single_writer_per_state_tile: bool,
    /// The plan names each live state tile exactly once.
    pub unique_state_tile_owners: bool,
    /// No request may share a state tile with another request.
    pub no_cross_request_state_sharing: bool,
}

/// The fused per-token reference inputs for one request.
///
/// Vectors are supplied in head-major row order: `[head][D]` for
/// `r, w, k, v, a, b`; state is `[head][D][D]`. All values are the exact
/// FP32 values the device would see after the accepted BF16->FP32 cast at
/// the state boundary. The pure core performs no cast itself; the shell
/// owns the cast boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct FusedWkvInputs {
    pub head_count: usize,
    pub head_size: usize,
    pub r: Vec<f32>,
    pub w: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub state: Vec<f32>,
}

impl FusedWkvInputs {
    fn rows(&self) -> Result<usize, Rejection> {
        if self.head_count == 0 || self.head_size == 0 {
            return Err(Rejection::new(
                "invalid_head_geometry",
                "head count and head size must be positive",
            ));
        }
        Ok(self.head_count * self.head_size)
    }

    fn validate_vector(&self, name: &str, values: &[f32]) -> Result<(), Rejection> {
        let expected = self.rows()?;
        if values.len() != expected {
            return Err(Rejection::new(
                "invalid_vector_shape",
                format!(
                    "{name} has {} elements, expected {expected} (heads x head size)",
                    values.len()
                ),
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Rejection::new(
                "non_finite_reference_input",
                format!("{name} contains a non-finite value"),
            ));
        }
        Ok(())
    }
}

/// The fused per-token reference result for one request.
#[derive(Clone, Debug, PartialEq)]
pub struct FusedWkvResult {
    /// `[head][D][D]`, FP32.
    pub next_state: Vec<f32>,
    /// `[head][D]`, the pre-correction attention output values.
    pub output: Vec<f32>,
}

/// The accepted fused WKV equation, per head, per token:
///
/// ```text
/// sa_i      = sum_j a[j] * S[i,j]
/// S'[i,j]   = S[i,j] * w[j] + sa_i * b[j] + k[j] * v[i]
/// out[i]    = sum_j S'[i,j] * r[j]
/// ```
///
/// This is the token-at-a-time 64-wide recurrence from the Goose paper
/// (Eq. 17-22) and the canonical `wkv7_cuda.cu` forward kernel. It fuses
/// decay, the two rank-update products, the outer product, the state write,
/// and the readout into one pure step.
///
/// r[verify tenstorrent.rwkv7_p150x2.fused_wkv_kernel.valid]
pub fn fused_wkv_step(inputs: &FusedWkvInputs) -> Result<FusedWkvResult, Rejection> {
    let rows = inputs.rows()?;
    for (name, values) in [
        ("r", &inputs.r),
        ("w", &inputs.w),
        ("k", &inputs.k),
        ("v", &inputs.v),
        ("a", &inputs.a),
        ("b", &inputs.b),
    ] {
        inputs.validate_vector(name, values)?;
    }
    if inputs.state.len() != rows * inputs.head_size {
        return Err(Rejection::new(
            "invalid_state_shape",
            format!(
                "state has {} elements, expected {} (heads x head size x head size)",
                inputs.state.len(),
                rows * inputs.head_size
            ),
        ));
    }
    if inputs.state.iter().any(|value| !value.is_finite()) {
        return Err(Rejection::new(
            "non_finite_reference_input",
            "state contains a non-finite value",
        ));
    }

    let d = inputs.head_size;
    let mut next_state = vec![0.0_f32; inputs.state.len()];
    let mut output = vec![0.0_f32; rows];

    for head in 0..inputs.head_count {
        let head_base = head * d;
        let state_head = &inputs.state[head * d * d..(head + 1) * d * d];
        for i in 0..d {
            // sa_i = sum_j a[j] * S[i,j]
            let mut sa = 0.0_f32;
            for j in 0..d {
                sa += inputs.a[head_base + j] * state_head[i * d + j];
            }
            // S'[i,j] = S[i,j] * w[j] + sa * b[j] + k[j] * v[i]
            for j in 0..d {
                let next =
                    state_head[i * d + j] * inputs.w[head_base + j]
                        + sa * inputs.b[head_base + j]
                        + inputs.k[head_base + j] * inputs.v[head_base + i];
                next_state[head * d * d + i * d + j] = next;
            }
            // out[i] = sum_j S'[i,j] * r[j]
            let mut acc = 0.0_f32;
            for j in 0..d {
                acc += next_state[head * d * d + i * d + j] * inputs.r[head_base + j];
            }
            output[head_base + i] = acc;
        }
    }
    Ok(FusedWkvResult {
        next_state,
        output,
    })
}

/// Validate the typed capability contract. Every check runs before any
/// device dispatch; an unsupported field rejects the fused path.
///
/// r[verify tenstorrent.rwkv7_p150x2.fused_wkv_kernel.invalid_input]
pub fn validate_fused_capability(capability: &FusedWkvCapability) -> Result<(), Rejection> {
    if capability.architecture != FUSED_ARCHITECTURE {
        return Err(Rejection::new(
            "unsupported_architecture",
            format!(
                "fused WKV requires {FUSED_ARCHITECTURE}, got {}",
                capability.architecture
            ),
        ));
    }
    if capability.head_size != WKV_HEAD_SIZE {
        return Err(Rejection::new(
            "unsupported_head_width",
            format!(
                "fused WKV requires head size {WKV_HEAD_SIZE}, got {}",
                capability.head_size
            ),
        ));
    }
    if capability.head_count != WKV_HEAD_COUNT {
        return Err(Rejection::new(
            "unsupported_head_count",
            format!(
                "fused WKV requires {WKV_HEAD_COUNT} heads, got {}",
                capability.head_count
            ),
        ));
    }
    if capability.batch_size == 0 || capability.batch_size > MAXIMUM_SUPPORTED_BATCH {
        return Err(Rejection::new(
            "unsupported_batch_size",
            format!(
                "batch size {} is outside 1..={MAXIMUM_SUPPORTED_BATCH}",
                capability.batch_size
            ),
        ));
    }
    if capability.state_dtype != FUSED_STATE_DTYPE {
        return Err(Rejection::new(
            "unsupported_state_dtype",
            format!(
                "fused WKV requires FP32 state, got {}",
                capability.state_dtype
            ),
        ));
    }
    if capability.vector_dtype != FUSED_VECTOR_DTYPE {
        return Err(Rejection::new(
            "unsupported_vector_dtype",
            format!(
                "fused WKV requires BF16 model vectors, got {}",
                capability.vector_dtype
            ),
        ));
    }
    if capability.output_dtype != FUSED_OUTPUT_DTYPE {
        return Err(Rejection::new(
            "unsupported_output_dtype",
            format!(
                "fused WKV requires BF16 output, got {}",
                capability.output_dtype
            ),
        ));
    }
    if capability.state_layout != "tile" {
        return Err(Rejection::new(
            "unsupported_state_layout",
            format!(
                "fused WKV requires tile-layout state, got {}",
                capability.state_layout
            ),
        ));
    }
    if !capability.state_ownership_plan.single_writer_per_state_tile {
        return Err(Rejection::new(
            "state_ownership_not_single_writer",
            "every state tile must have exactly one writer",
        ));
    }
    if !capability.state_ownership_plan.unique_state_tile_owners {
        return Err(Rejection::new(
            "state_ownership_not_unique",
            "state tile ownership must be unique",
        ));
    }
    if !capability.state_ownership_plan.no_cross_request_state_sharing {
        return Err(Rejection::new(
            "state_ownership_cross_request",
            "state tiles must not be shared across requests",
        ));
    }
    Ok(())
}

/// Checked state byte budget for one fused WKV invocation.
/// The shell uses this before allocating any device state.
pub fn fused_state_bytes(capability: &FusedWkvCapability) -> Result<usize, Rejection> {
    validate_fused_capability(capability)?;
    let tiles_per_request = capability.head_count;
    let bytes = capability
        .batch_size
        .checked_mul(tiles_per_request)
        .and_then(|tiles| tiles.checked_mul(capability.head_size * capability.head_size))
        .and_then(|elements| elements.checked_mul(FP32_BYTE_WIDTH))
        .ok_or_else(|| {
            Rejection::new(
                "state_byte_overflow",
                "fused WKV state byte budget overflows usize",
            )
        })?;
    Ok(bytes)
}

/// Checked vector byte budget for one fused WKV invocation.
pub fn fused_vector_bytes(capability: &FusedWkvCapability) -> Result<usize, Rejection> {
    validate_fused_capability(capability)?;
    let per_request = capability
        .head_count
        .checked_mul(capability.head_size)
        .and_then(|rows| rows.checked_mul(BF16_BYTE_WIDTH))
        .ok_or_else(|| {
            Rejection::new(
                "vector_byte_overflow",
                "fused WKV vector byte budget overflows usize",
            )
        })?;
    let vectors = 6_usize; // r, w, k, v, a, b
    capability
        .batch_size
        .checked_mul(per_request)
        .and_then(|bytes| bytes.checked_mul(vectors))
        .ok_or_else(|| {
            Rejection::new(
                "vector_byte_overflow",
                "fused WKV vector byte budget overflows usize",
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capability(batch: usize) -> FusedWkvCapability {
        FusedWkvCapability {
            architecture: FUSED_ARCHITECTURE.to_owned(),
            head_count: WKV_HEAD_COUNT,
            head_size: WKV_HEAD_SIZE,
            batch_size: batch,
            state_dtype: FUSED_STATE_DTYPE.to_owned(),
            vector_dtype: FUSED_VECTOR_DTYPE.to_owned(),
            output_dtype: FUSED_OUTPUT_DTYPE.to_owned(),
            state_layout: "tile".to_owned(),
            state_ownership_plan: StateOwnershipPlan {
                single_writer_per_state_tile: true,
                unique_state_tile_owners: true,
                no_cross_request_state_sharing: true,
            },
        }
    }

    fn make_inputs(head_count: usize, head_size: usize, fill: f32) -> FusedWkvInputs {
        let rows = head_count * head_size;
        let state = vec![fill; head_count * head_size * head_size];
        FusedWkvInputs {
            head_count,
            head_size,
            r: vec![fill; rows],
            w: vec![fill; rows],
            k: vec![fill; rows],
            v: vec![fill; rows],
            a: vec![fill; rows],
            b: vec![fill; rows],
            state,
        }
    }

    #[test]
    fn zero_state_produces_outer_product_output() {
        // With S = 0, the fused step reduces to the outer-product write:
        // S'[i,j] = k[j] * v[i]; out[i] = sum_j S'[i,j] * r[j].
        let mut zero = make_inputs(WKV_HEAD_COUNT, WKV_HEAD_SIZE, 0.0);
        zero.r = vec![0.25; WKV_HIDDEN_SIZE];
        zero.k = vec![2.0; WKV_HIDDEN_SIZE];
        zero.v = vec![3.0; WKV_HIDDEN_SIZE];
        let result = fused_wkv_step(&zero).expect("valid step");
        let d = WKV_HEAD_SIZE;
        for head in 0..WKV_HEAD_COUNT {
            for i in 0..d {
                // S'[i,j] = 2.0 * 3.0 = 6.0
                for j in 0..d {
                    assert_eq!(result.next_state[head * d * d + i * d + j], 6.0);
                }
                // out[i] = sum_j 6.0 * 0.25 = 6.0 * 64 * 0.25 = 96.0
                assert_eq!(result.output[head * d + i], 96.0);
            }
        }
    }

    #[test]
    fn nonzero_state_matches_decomposed_reference_manually() {
        // D = 2 mini-world to hand-verify the equation.
        let inputs = FusedWkvInputs {
            head_count: 1,
            head_size: 2,
            r: vec![1.0, 2.0],
            w: vec![0.5, 0.25],
            k: vec![2.0, 3.0],
            v: vec![4.0, 5.0],
            a: vec![0.1, 0.2],
            b: vec![0.3, 0.4],
            state: vec![1.0, 2.0, 3.0, 4.0],
        };
        let result = fused_wkv_step(&inputs).expect("valid step");
        // sa_0 = 0.1*1 + 0.2*2 = 0.5;  S'[0,0] = 1*0.5 + 0.5*0.3 + 2*4 = 8.65
        // S'[0,1] = 2*0.25 + 0.5*0.4 + 3*4 = 12.7; out[0] = 8.65 + 2*12.7 = 34.05
        assert!((result.next_state[0] - 8.65).abs() < 1e-6);
        assert!((result.next_state[1] - 12.7).abs() < 1e-6);
        // sa_1 = 0.1*3 + 0.2*4 = 1.1; S'[1,0] = 3*0.5 + 1.1*0.3 + 2*5 = 11.83
        // S'[1,1] = 4*0.25 + 1.1*0.4 + 3*5 = 16.44; out[1] = 11.83 + 2*16.44 = 44.71
        assert!((result.next_state[2] - 11.83).abs() < 1e-6);
        assert!((result.next_state[3] - 16.44).abs() < 1e-6);
        assert!((result.output[0] - 34.05).abs() < 1e-6);
        assert!((result.output[1] - 44.71).abs() < 1e-6);
    }

    #[test]
    fn repeated_state_carry_is_deterministic() {
        let inputs = make_inputs(WKV_HEAD_COUNT, WKV_HEAD_SIZE, 1.5);
        // Clone a fresh state for the second step; the vectors are reused.
        let first = fused_wkv_step(&inputs).expect("first step");
        let first_state = first.next_state.clone();
        let second = fused_wkv_step(&FusedWkvInputs {
            state: first_state,
            ..inputs.clone()
        })
        .expect("second step");
        // Re-running the same step from the same state is deterministic.
        let second_again = fused_wkv_step(&FusedWkvInputs {
            state: first.next_state.clone(),
            ..inputs.clone()
        })
        .expect("second step again");
        assert_eq!(second.next_state, second_again.next_state);
        assert_eq!(second.output, second_again.output);
    }

    #[test]
    fn valid_capability_passes() {
        validate_fused_capability(&capability(4)).expect("valid capability");
    }

    #[test]
    fn rejects_unsupported_architecture() {
        let mut c = capability(4);
        c.architecture = "wormhole".to_owned();
        let rejection = validate_fused_capability(&c).expect_err("must reject");
        assert_eq!(rejection.code, "unsupported_architecture");
    }

    #[test]
    fn rejects_wrong_head_width_and_count() {
        let mut wide = capability(4);
        wide.head_size = 128;
        assert_eq!(
            validate_fused_capability(&wide).expect_err("wide head").code,
            "unsupported_head_width"
        );
        let mut many = capability(4);
        many.head_count = 16;
        assert_eq!(
            validate_fused_capability(&many).expect_err("many heads").code,
            "unsupported_head_count"
        );
    }

    #[test]
    fn rejects_non_fp32_state() {
        let mut c = capability(4);
        c.state_dtype = "bfloat16".to_owned();
        assert_eq!(
            validate_fused_capability(&c).expect_err("bf16 state").code,
            "unsupported_state_dtype"
        );
    }

    #[test]
    fn rejects_unsupported_layout() {
        let mut c = capability(4);
        c.state_layout = "row_major".to_owned();
        assert_eq!(
            validate_fused_capability(&c).expect_err("layout").code,
            "unsupported_state_layout"
        );
    }

    #[test]
    fn rejects_state_ownership_violations() {
        let mut shared = capability(4);
        shared.state_ownership_plan.no_cross_request_state_sharing = false;
        assert_eq!(
            validate_fused_capability(&shared).expect_err("shared").code,
            "state_ownership_cross_request"
        );
        let mut multi = capability(4);
        multi.state_ownership_plan.single_writer_per_state_tile = false;
        assert_eq!(
            validate_fused_capability(&multi).expect_err("multi").code,
            "state_ownership_not_single_writer"
        );
    }

    #[test]
    fn rejects_non_finite_reference_input() {
        let mut inputs = make_inputs(WKV_HEAD_COUNT, WKV_HEAD_SIZE, 1.0);
        inputs.r[0] = f32::NAN;
        assert_eq!(
            fused_wkv_step(&inputs).expect_err("nan").code,
            "non_finite_reference_input"
        );
    }

    #[test]
    fn rejects_malformed_shapes() {
        let mut inputs = make_inputs(WKV_HEAD_COUNT, WKV_HEAD_SIZE, 1.0);
        inputs.state = vec![0.0; 1];
        assert_eq!(
            fused_wkv_step(&inputs).expect_err("state").code,
            "invalid_state_shape"
        );
        inputs = make_inputs(WKV_HEAD_COUNT, WKV_HEAD_SIZE, 1.0);
        inputs.k = vec![0.0; 1];
        assert_eq!(
            fused_wkv_step(&inputs).expect_err("k").code,
            "invalid_vector_shape"
        );
    }

    #[test]
    fn state_byte_budget_scales_with_batch() {
        let b4 = fused_state_bytes(&capability(4)).expect("b4 bytes");
        let b16 = fused_state_bytes(&capability(16)).expect("b16 bytes");
        assert_eq!(b16, b4 * 4);
        assert!(fused_vector_bytes(&capability(4)).expect("vec bytes") > 0);
    }
}
