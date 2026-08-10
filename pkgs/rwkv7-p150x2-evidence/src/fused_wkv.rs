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

/// Inputs for the attention boundary that follows the WKV recurrence.
///
/// The fused operator covers state readout (in `fused_wkv_step`), attention
/// correction, and the BF16 output boundary. The output projection at the
/// end stays outside the operator per the design decision.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionCorrectionInputs {
    pub head_count: usize,
    pub head_size: usize,
    /// The pre-correction readout values, `[head][D]`.
    pub output: Vec<f32>,
    /// The removal-key rows, `[head][D]`, used for the r_k bonus.
    pub k_rows: Vec<f32>,
    /// The value rows, `[head][D]`, used for the r_k bonus.
    pub v_rows: Vec<f32>,
    /// The rwkv gate rows, `[head][D]`.
    pub gate: Vec<f32>,
    /// The scalar `r_k` coefficient (per request, broadcast).
    pub r_k: f32,
    /// Group-norm affinity weight, `[head][D]`.
    pub group_norm_weight: Vec<f32>,
    /// Group-norm affinity bias, `[head][D]`.
    pub group_norm_bias: Vec<f32>,
}

/// The attention-corrected output rows before the output projection.
#[derive(Clone, Debug, PartialEq)]
pub struct AttentionCorrectionResult {
    /// Corrected rows, `[head][D]`, BF16-bounded values.
    pub corrected_rows: Vec<f32>,
}

fn validate_row_vector(name: &str, values: &[f32], rows: usize) -> Result<(), Rejection> {
    if values.len() != rows {
        return Err(Rejection::new(
            "invalid_correction_shape",
            format!(
                "{name} has {} elements, expected {rows} (heads x head size)",
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

/// Pure attention correction: group-norm the readout rows, add the r_k
/// bonus weighted by the value rows, scale by the gate, and bound the
/// result to the BF16 finite range. The output projection is outside.
///
/// r[verify tenstorrent.rwkv7_p150x2.fused_wkv_kernel.valid]
pub fn attention_correction(
    inputs: &AttentionCorrectionInputs,
    group_norm_epsilon: f32,
) -> Result<AttentionCorrectionResult, Rejection> {
    if inputs.head_count == 0 || inputs.head_size == 0 {
        return Err(Rejection::new(
            "invalid_head_geometry",
            "head count and head size must be positive",
        ));
    }
    let rows = inputs.head_count * inputs.head_size;
    for (name, values) in [
        ("output", &inputs.output),
        ("k_rows", &inputs.k_rows),
        ("v_rows", &inputs.v_rows),
        ("gate", &inputs.gate),
        ("group_norm_weight", &inputs.group_norm_weight),
        ("group_norm_bias", &inputs.group_norm_bias),
    ] {
        validate_row_vector(name, values, rows)?;
    }
    if !inputs.r_k.is_finite() || !group_norm_epsilon.is_finite() || group_norm_epsilon <= 0.0 {
        return Err(Rejection::new(
            "non_finite_reference_input",
            "attention correction scalars must be finite and epsilon positive",
        ));
    }
    let d = inputs.head_size;
    let mut corrected_rows = Vec::with_capacity(rows);
    let mut corrections = Vec::new();
    for head in 0..inputs.head_count {
        let base = head * d;
        let mut mean = 0.0_f32;
        for i in 0..d {
            mean += inputs.output[base + i];
        }
        mean /= d as f32;
        let mut variance = 0.0_f32;
        for i in 0..d {
            let diff = inputs.output[base + i] - mean;
            variance += diff * diff;
        }
        variance /= d as f32;
        let inv_std = 1.0_f32 / (variance + group_norm_epsilon).sqrt();
        let scalar = inputs.r_k
            * (0..d)
                .map(|i| inputs.k_rows[base + i] * inputs.v_rows[base + i])
                .sum::<f32>();
        for i in 0..d {
            let normalized = (inputs.output[base + i] - mean) * inv_std;
            let scaled = normalized * inputs.group_norm_weight[base + i]
                + inputs.group_norm_bias[base + i];
            let corrected =
                (scaled + scalar * inputs.v_rows[base + i]) * inputs.gate[base + i];
            if corrected.abs() > f32::MAX / 2.0 {
                corrections.push((head, i));
            }
            corrected_rows.push(corrected);
        }
    }
    if !corrections.is_empty() {
        return Err(Rejection::new(
            "correction_overflow",
            format!(
                "{} corrected rows exceed the BF16 finite bound",
                corrections.len()
            ),
        ));
    }
    Ok(AttentionCorrectionResult { corrected_rows })
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

/// Per-request dataflow plan for one fused WKV decode step.
///
/// K1: this is the bounded data-movement contract. Each live request owns
/// one FP32 state tile per head and six BF16 vectors. The plan names exact
/// byte counts so the device shell can reserve buffers, and it never exceeds
/// the capability's batch bound.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FusedDataflowPlan {
    pub batch_size: usize,
    pub head_count: usize,
    pub head_size: usize,
    /// FP32 state bytes for the whole batch (batch x heads x D x D x 4).
    pub state_bytes: usize,
    /// BF16 vector bytes for the whole batch (six vectors per request).
    pub vector_bytes: usize,
    /// BF16 output bytes (one vector per request head row).
    pub output_bytes: usize,
    pub state_tiles_per_request: usize,
    pub works_per_request: usize,
}

/// Plan the bounded data movement for one fused decode step.
///
/// r[verify tenstorrent.rwkv7_p150x2.fused_wkv_kernel.valid]
pub fn plan_fused_dataflow(
    capability: &FusedWkvCapability,
) -> Result<FusedDataflowPlan, Rejection> {
    validate_fused_capability(capability)?;
    let d = capability.head_size;
    let per_request_state = capability
        .head_count
        .checked_mul(d)
        .and_then(|area| area.checked_mul(d))
        .and_then(|elements| elements.checked_mul(FP32_BYTE_WIDTH))
        .ok_or_else(|| {
            Rejection::new(
                "dataflow_byte_overflow",
                "per-request state bytes overflow usize",
            )
        })?;
    let per_request_vector = capability
        .head_count
        .checked_mul(d)
        .and_then(|rows| rows.checked_mul(6_usize)) // r,w,k,v,a,b
        .and_then(|vectors| vectors.checked_mul(BF16_BYTE_WIDTH))
        .ok_or_else(|| {
            Rejection::new(
                "dataflow_byte_overflow",
                "per-request vector bytes overflow usize",
            )
        })?;
    let per_request_output = capability
        .head_count
        .checked_mul(d)
        .and_then(|rows| rows.checked_mul(BF16_BYTE_WIDTH))
        .ok_or_else(|| {
            Rejection::new(
                "dataflow_byte_overflow",
                "per-request output bytes overflow usize",
            )
        })?;
    Ok(FusedDataflowPlan {
        batch_size: capability.batch_size,
        head_count: capability.head_count,
        head_size: d,
        state_bytes: capability.batch_size.checked_mul(per_request_state).ok_or_else(||
            Rejection::new("dataflow_byte_overflow", "batch state bytes overflow usize")
        )?,
        vector_bytes: capability.batch_size.checked_mul(per_request_vector).ok_or_else(||
            Rejection::new("dataflow_byte_overflow", "batch vector bytes overflow usize")
        )?,
        output_bytes: capability.batch_size.checked_mul(per_request_output).ok_or_else(||
            Rejection::new("dataflow_byte_overflow", "batch output bytes overflow usize")
        )?,
        state_tiles_per_request: capability.head_count,
        works_per_request: capability.head_count,
    })
}

/// A deterministic (request, head) work item.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Serialize)]
pub struct FusedWorkItem {
    pub request_index: usize,
    pub head_index: usize,
}

/// One core's assigned work and its synchronous-count constant.
#[derive(Clone, Copy, Debug, Eq, PartialEq, PartialOrd, Ord, Serialize)]
pub struct CoreAssignment {
    pub core_index: usize,
    pub work_count: usize,
}

/// Deterministic state-tile ownership plan.
///
/// K3: heads and request tiles map to fixed core groups. Every
/// (request, head) state tile has exactly one writer core, and no tile is
/// shared across requests. The plan uses a stable round-robin so it is
/// reproducible without hardware.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FusedOwnershipPlan {
    pub core_count: usize,
    /// Sorted (request, head) work items owned by each core.
    pub per_core_work: Vec<Vec<FusedWorkItem>>,
    /// A stable per-core completion/semaphore count.
    pub per_core_completion_count: Vec<usize>,
}

/// Deterministically assign (request, head) tiles to cores, one writer per
/// tile.
///
/// r[verify tenstorrent.rwkv7_p150x2.fused_wkv_kernel.invalid_input]
pub fn plan_fused_ownership(
    capability: &FusedWkvCapability,
    core_count: usize,
) -> Result<FusedOwnershipPlan, Rejection> {
    validate_fused_capability(capability)?;
    if core_count == 0 {
        return Err(Rejection::new(
            "invalid_core_count",
            "fused WKV core count must be positive",
        ));
    }
    let total_works = capability
        .batch_size
        .checked_mul(capability.head_count)
        .ok_or_else(|| {
            Rejection::new(
                "ownership_works_overflow",
                "total fused work items overflow usize",
            )
        })?;
    let mut per_core_work = vec![Vec::new(); core_count];
    let mut completion = vec![0_usize; core_count];
    for request_index in 0..capability.batch_size {
        for head_index in 0..capability.head_count {
            let core = (request_index * capability.head_count + head_index) % core_count;
            per_core_work[core].push(FusedWorkItem {
                request_index,
                head_index,
            });
            completion[core] += 1;
        }
    }
    let mut per_core = per_core_work;
    for work in per_core.iter_mut() {
        work.sort();
    }
    // Verify single-writer and uniqueness (the round-robin construction is
    // deterministic, so this is an internal invariant check).
    let mut seen = Vec::with_capacity(total_works);
    for work in per_core.iter().flatten() {
        if seen.contains(work) {
            return Err(Rejection::new(
                "ownership_not_unique",
                "state tile assigned to more than one core",
            ));
        }
        seen.push(*work);
    }
    if seen.len() != total_works {
        return Err(Rejection::new(
            "ownership_incomplete",
            "state tile ownership does not cover every work item",
        ));
    }
    Ok(FusedOwnershipPlan {
        core_count,
        per_core_work: per_core,
        per_core_completion_count: completion,
    })
}

/// The completed per-request plan consumed by the runtime shell and the
/// later device op source tree.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FusedPlanSet {
    pub capability: FusedWkvCapability,
    pub dataflow: FusedDataflowPlan,
    pub ownership: FusedOwnershipPlan,
    pub state_bytes: usize,
    pub vector_bytes: usize,
    pub output_bytes: usize,
}

/// Assemble the full device-free plan set for one fused decode step.
///
/// r[verify tenstorrent.rwkv7_p150x2.fused_wkv_kernel.valid]
pub fn plan_fused_step(
    capability: &FusedWkvCapability,
    core_count: usize,
) -> Result<FusedPlanSet, Rejection> {
    let dataflow = plan_fused_dataflow(capability)?;
    let ownership = plan_fused_ownership(capability, core_count)?;
    Ok(FusedPlanSet {
        capability: capability.clone(),
        dataflow: dataflow.clone(),
        ownership: ownership.clone(),
        state_bytes: dataflow.state_bytes,
        vector_bytes: dataflow.vector_bytes,
        output_bytes: dataflow.output_bytes,
    })
}


/// The capability-selected dispatch decision.
///
/// K4: the runtime shell calls this before choosing the fused path. It is
/// fail-closed: the fused path is selected only when the capability is
/// supported AND the fused path has been physically admitted. Otherwise the
/// generic TT-NN path stays.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FusedDispatchDecision {
    pub candidate_id: String,
    pub capability_supported: bool,
    pub physically_admitted: bool,
    pub use_fused_path: bool,
    pub reason: String,
}

/// Decide whether to use the fused WKV path for a candidate.
///
/// The `physically_admitted` flag is owned by the evidence pipeline; it is
/// set only after stable alternating A/B and strict admission pass. This
/// function never fabricates that evidence.
///
/// r[verify tenstorrent.rwkv7_p150x2.fused_wkv_kernel.valid]
/// r[verify tenstorrent.rwkv7_p150x2.fused_wkv_kernel.no_gain]
pub fn decide_fused_dispatch(
    candidate_id: &str,
    capability: &FusedWkvCapability,
    physically_admitted: bool,
) -> FusedDispatchDecision {
    let capability_supported = validate_fused_capability(capability).is_ok();
    let use_fused_path = capability_supported && physically_admitted;
    let reason = if !capability_supported {
        "capability validation failed; generic path retained".to_owned()
    } else if !physically_admitted {
        "capability valid but fused path not yet physically admitted; generic path retained"
            .to_owned()
    } else {
        "capability supported and fused path physically admitted".to_owned()
    };
    FusedDispatchDecision {
        candidate_id: candidate_id.to_owned(),
        capability_supported,
        physically_admitted,
        use_fused_path,
        reason,
    }
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

    #[test]
    fn dataflow_plan_is_bounded_and_batch_scaled() {
        let p4 = plan_fused_dataflow(&capability(4)).expect("b4 dataflow");
        let p16 = plan_fused_dataflow(&capability(16)).expect("b16 dataflow");
        // FP32 state per request: 12 heads x 64 x 64 x 4 = 196608 bytes.
        assert_eq!(p4.state_bytes, 4 * 196608);
        assert_eq!(p16.state_bytes, p4.state_bytes * 4);
        // Six BF16 vectors per request: 6 x 12 x 64 x 2 = 9216 bytes.
        assert_eq!(p4.vector_bytes, 4 * 9216);
        // One BF16 output per request: 12 x 64 x 2 = 1536 bytes.
        assert_eq!(p4.output_bytes, 4 * 1536);
        assert_eq!(p16.state_tiles_per_request, WKV_HEAD_COUNT);
    }

    #[test]
    fn ownership_plan_is_unique_and_complete() {
        let plan = plan_fused_ownership(&capability(8), 4).expect("ownership");
        assert_eq!(plan.per_core_work.len(), 4);
        let total: usize = plan.per_core_work.iter().map(|work| work.len()).sum();
        assert_eq!(total, 8 * WKV_HEAD_COUNT);
        let mut seen = Vec::new();
        for work in plan.per_core_work.iter().flatten() {
            assert!(!seen.contains(work), "duplicate work item");
            seen.push(*work);
        }
        assert_eq!(plan.per_core_completion_count.len(), 4);
    }

    #[test]
    fn ownership_rejects_zero_cores() {
        let rejection = plan_fused_ownership(&capability(4), 0).expect_err("zero cores");
        assert_eq!(rejection.code, "invalid_core_count");
    }

    #[test]
    fn plan_set_assembles_device_free() {
        let set = plan_fused_step(&capability(8), 4).expect("plan set");
        assert_eq!(set.dataflow.state_bytes, set.state_bytes);
        assert_eq!(set.ownership.core_count, 4);
        assert_eq!(set.vector_bytes, 8 * 9216);
    }

    #[test]
    fn dispatch_fails_closed_without_admission() {
        let decision = decide_fused_dispatch("B16", &capability(4), false);
        assert!(decision.capability_supported);
        assert!(!decision.use_fused_path);
        assert!(decision.reason.contains("not yet physically admitted"));
    }

    #[test]
    fn dispatch_selects_fused_path_only_after_admission() {
        let decision = decide_fused_dispatch("B16", &capability(4), true);
        assert!(decision.use_fused_path);
    }

    #[test]
    fn dispatch_rejects_unsupported_capability_even_if_admitted() {
        let mut bad = capability(4);
        bad.head_size = 128;
        let decision = decide_fused_dispatch("B16", &bad, true);
        assert!(!decision.capability_supported);
        assert!(!decision.use_fused_path);
        assert!(decision.reason.contains("capability validation failed"));
    }

    fn correction_inputs(head_count: usize, head_size: usize, val: f32) -> AttentionCorrectionInputs {
        let rows = head_count * head_size;
        AttentionCorrectionInputs {
            head_count,
            head_size,
            output: vec![val; rows],
            k_rows: vec![1.0; rows],
            v_rows: vec![1.0; rows],
            gate: vec![1.0; rows],
            r_k: 1.0,
            group_norm_weight: vec![1.0; rows],
            group_norm_bias: vec![0.0; rows],
        }
    }

    #[test]
    fn correction_with_constant_rows_is_identity_like() {
        // Constant head rows have zero variance -> normalized row is 0.
        // corrected = (0 + r_k * sum(1*1)=d) * 1 = d for each element.
        let result = attention_correction(&correction_inputs(2, 64, 5.0), 1e-5)
            .expect("valid correction");
        assert_eq!(result.corrected_rows.len(), 2 * 64);
        assert!((result.corrected_rows[0] - 64.0).abs() < 1e-4);
    }

    #[test]
    fn correction_rejects_non_finite_input() {
        let mut inputs = correction_inputs(2, 64, 1.0);
        inputs.output[3] = f32::NAN;
        assert_eq!(
            attention_correction(&inputs, 1e-5).expect_err("nan").code,
            "non_finite_reference_input"
        );
    }

    #[test]
    fn correction_rejects_bad_epsilon() {
        let rejection = attention_correction(&correction_inputs(2, 64, 1.0), 0.0)
            .expect_err("zero epsilon");
        assert_eq!(rejection.code, "non_finite_reference_input");
    }

    #[test]
    fn correction_rejects_overflowing_rows() {
        let mut inputs = correction_inputs(2, 64, 1.0);
        inputs.gate = vec![f32::MAX; inputs.gate.len()];
        let rejection =
            attention_correction(&inputs, 1e-5).expect_err("overflow");
        assert_eq!(rejection.code, "correction_overflow");
    }
}
