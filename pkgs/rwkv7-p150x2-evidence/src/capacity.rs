//! Pure capacity-profile and card-local memory planning for RWKV-7 decode.
//!
//! r[impl tenstorrent.rwkv7_p150x2.scalable_card_local_batching]
//! r[impl tenstorrent.rwkv7_p150x2.scalable_card_local_batching.valid]
//! r[impl tenstorrent.rwkv7_p150x2.scalable_card_local_batching.invalid_memory]
//! r[impl tenstorrent.rwkv7_p150x2.scalable_card_local_batching.invalid_state]
//! r[impl tenstorrent.rwkv7_p150x2.scalable_card_local_batching.target_miss]
//!
//! This module is pure and device-free. It validates typed capacity
//! candidates, derives checked memory requirements before any device opens,
//! and owns deterministic state-slot ownership. The imperative shells (the
//! benchmark launcher, the Rust CLI, and the C++ worker) consume these
//! decisions; they never re-derive them.

use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

pub const CAPACITY_SCHEMA_VERSION: u32 = 1;
pub const SUPPORTED_CAPACITY_IDS: [&str; 4] = ["B4", "B16", "B32", "B64"];
pub const SUPPORTED_SLOT_COUNTS: [usize; 4] = [4, 16, 32, 64];
pub const EXPECTED_CAPACITY_PROFILE_ID: &str = "rwkv7-p150x2-grid-capacity-v1";
const PERCENTILE_P99: f64 = 0.99;
const TOKEN_SHIFT_BUFFER_COUNT: usize = 2;
const MINIMUM_WAVE_COUNT: usize = 1;
const MAXIMUM_WAVE_COUNT: usize = 8;
const TIGHT_FLOAT_TOLERANCE: f64 = 0.000_000_001;
const SAMPLE_COUNT_MINIMUM: usize = 1;

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

/// Pinned model geometry shared by every candidate plan.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelExtent {
    pub layer_count: usize,
    pub hidden_size: usize,
    pub head_count: usize,
    pub head_size: usize,
}

impl ModelExtent {
    fn recurrent_elements_per_slot(&self) -> Result<usize, Rejection> {
        self.head_count
            .checked_mul(self.head_size)
            .and_then(|heads_area| heads_area.checked_mul(self.head_size))
            .and_then(|per_layer| per_layer.checked_mul(self.layer_count))
            .ok_or_else(|| {
                Rejection::new(
                    "capacity_arithmetic_overflow",
                    "recurrent-state element count overflows usize",
                )
            })
    }

    fn token_shift_elements_per_slot(&self) -> Result<usize, Rejection> {
        self.hidden_size
            .checked_mul(TOKEN_SHIFT_BUFFER_COUNT)
            .and_then(|per_layer| per_layer.checked_mul(self.layer_count))
            .ok_or_else(|| {
                Rejection::new(
                    "capacity_arithmetic_overflow",
                    "token-shift element count overflows usize",
                )
            })
    }
}

/// Optimistic window coefficients from the typed profile.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MemoryBounds {
    pub fp32_bytes_per_element: usize,
    pub bf16_bytes_per_element: usize,
    pub temporary_tensor_bytes_per_slot: usize,
    pub device_memory_budget_bytes: usize,
    pub safety_reserve_bytes: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityCandidate {
    pub id: String,
    pub per_card_slot_count: usize,
    pub maximum_wave_count: usize,
    pub minimum_ab_throughput_gain_fraction: f64,
    pub maximum_p99_latency_regression_fraction: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityAdmissionThresholds {
    pub ab_repeat_count: usize,
    pub minimum_candidate_gain_fraction: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityProfile {
    pub schema_version: u32,
    pub profile_id: String,
    pub model_profile_blake3: String,
    pub candidates: Vec<CapacityCandidate>,
    pub memory: MemoryBounds,
    pub admission: CapacityAdmissionThresholds,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MemoryPlan {
    pub recurrent_state_bytes: usize,
    pub token_shift_bytes: usize,
    pub temporary_tensor_bytes: usize,
    pub safety_reserve_bytes: usize,
    pub total_required_bytes: usize,
    pub device_budget_bytes: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CapacityPlan {
    pub candidate_id: String,
    pub per_card_slot_count: usize,
    pub maximum_wave_count: usize,
    pub memory: MemoryPlan,
    pub minimum_ab_throughput_gain_fraction: f64,
    pub maximum_p99_latency_regression_fraction: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityMeasurement {
    pub candidate_id: String,
    pub simulated: bool,
    pub parity_exact: bool,
    pub samples: usize,
    pub aggregate_tokens_per_second: f64,
    /// p99 time per generated token, in seconds. This is the honest
    /// cross-capacity latency metric: per-request wall latency grows with
    /// concurrency and is not comparable across different batch widths.
    pub p99_per_token_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CurrentAdmittedMeasurement {
    pub candidate_id: String,
    pub aggregate_tokens_per_second: f64,
    /// p99 time per generated token, in seconds.
    pub p99_per_token_seconds: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CapacityEvaluation {
    pub candidate_id: String,
    pub promoted: bool,
    pub measured_gain_fraction: f64,
    pub latency_regression_fraction: f64,
    pub details: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SweepOutcome {
    pub admitted_candidate_id: String,
    pub blocker_code: Option<String>,
    pub blocker_detail: Option<String>,
    pub evaluations: Vec<CapacityEvaluation>,
    pub plans: Vec<CapacityPlan>,
}

/// A deterministic, sorted state-slot recycler.
///
/// Slots are integers below the capacity. Allocation returns the lowest free
/// slot. A released slot is not reusable until the owner explicitly zeroes it.
/// r[impl tenstorrent.rwkv7_p150x2.scalable_card_local_batching.invalid_state]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SlotRecycler {
    capacity: usize,
    free: BTreeSet<usize>,
    live: BTreeSet<usize>,
    /// Slots whose last allocation ended without an explicit zero.
    dirty: BTreeSet<usize>,
    /// Every future zeroing step must name a slot that was released dirty.
    zeroed_ever: BTreeSet<usize>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SlotAllocation {
    pub slot: usize,
    pub requires_state_zeroing: bool,
}

impl SlotRecycler {
    pub fn new(capacity: usize) -> Result<Self, Rejection> {
        if capacity == 0 {
            return Err(Rejection::new(
                "invalid_slot_capacity",
                "slot recycler capacity must be positive",
            ));
        }
        let free = (0..capacity).collect::<BTreeSet<_>>();
        Ok(Self {
            capacity,
            free,
            live: BTreeSet::new(),
            dirty: BTreeSet::new(),
            zeroed_ever: BTreeSet::new(),
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn live_count(&self) -> usize {
        self.live.len()
    }

    pub fn free_count(&self) -> usize {
        self.free.len()
    }

    pub fn allocate(&mut self) -> Result<SlotAllocation, Rejection> {
        let slot = *self.free.iter().next().ok_or_else(|| {
            Rejection::new(
                "slot_capacity_exhausted",
                "no free state slot remains on this card",
            )
        })?;
        self.free.remove(&slot);
        let requires_state_zeroing = self.dirty.contains(&slot);
        if !self.live.insert(slot) {
            return Err(Rejection::new(
                "duplicate_slot_ownership",
                format!("slot {slot} is already live"),
            ));
        }
        if self.zeroed_ever.insert(slot) && !requires_state_zeroing {
            // First allocation ever: the state was never touched.
        }
        Ok(SlotAllocation {
            slot,
            requires_state_zeroing,
        })
    }

    /// Recycle a completed slot. The slot stays dirty until zeroed.
    pub fn release(&mut self, slot: usize) -> Result<(), Rejection> {
        if !self.live.remove(&slot) {
            return Err(Rejection::new(
                "unknown_slot_owner",
                format!("slot {slot} is not live and cannot be released"),
            ));
        }
        self.dirty.insert(slot);
        self.free.insert(slot);
        Ok(())
    }

    /// Zero a released dirty slot so a new request can own it without
    /// observing prior state.
    pub fn zero(&mut self, slot: usize) -> Result<(), Rejection> {
        if self.live.contains(&slot) {
            return Err(Rejection::new(
                "zero_live_slot",
                format!("slot {slot} is live and cannot be zeroed in place"),
            ));
        }
        if !self.free.contains(&slot) {
            return Err(Rejection::new(
                "zero_unknown_slot",
                format!("slot {slot} does not belong to this card"),
            ));
        }
        if !self.dirty.remove(&slot) {
            return Err(Rejection::new(
                "zero_clean_slot",
                format!("slot {slot} was already zeroed; double zeroing is an error"),
            ));
        }
        Ok(())
    }
}

fn is_blake3_hex(value: &str) -> bool {
    const BLAKE3_HEX_LENGTH: usize = 64;
    value.len() == BLAKE3_HEX_LENGTH
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn finite_non_negative(value: f64, field: &str) -> Result<f64, Rejection> {
    if !value.is_finite() || value < 0.0 {
        return Err(Rejection::new(
            "invalid_capacity_metric",
            format!("{field} must be finite and non-negative, got {value}"),
        ));
    }
    Ok(value)
}

fn finite_positive(value: f64, field: &str) -> Result<f64, Rejection> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Rejection::new(
            "invalid_capacity_metric",
            format!("{field} must be finite and positive, got {value}"),
        ));
    }
    Ok(value)
}

fn candidate_supported(id: &str, slot_count: usize) -> bool {
    SUPPORTED_CAPACITY_IDS
        .iter()
        .zip(SUPPORTED_SLOT_COUNTS)
        .any(|(known_id, known_slots)| *known_id == id && known_slots == slot_count)
}

/// Validate the typed capacity profile. This runs before any memory
/// arithmetic and before any device access.
pub fn validate_capacity_profile(profile: &CapacityProfile) -> Result<(), Rejection> {
    if profile.schema_version != CAPACITY_SCHEMA_VERSION {
        return Err(Rejection::new(
            "capacity_schema_mismatch",
            format!(
                "capacity profile schema is {}, expected {CAPACITY_SCHEMA_VERSION}",
                profile.schema_version
            ),
        ));
    }
    if profile.profile_id != EXPECTED_CAPACITY_PROFILE_ID
        || !is_blake3_hex(&profile.model_profile_blake3)
        || profile.candidates.is_empty()
    {
        return Err(Rejection::new(
            "invalid_capacity_profile",
            "capacity profile identity or candidate list is malformed",
        ));
    }
    if profile.memory.fp32_bytes_per_element != 4
        || profile.memory.bf16_bytes_per_element != 2
        || profile.memory.temporary_tensor_bytes_per_slot == 0
        || profile.memory.device_memory_budget_bytes == 0
        || profile.memory.safety_reserve_bytes == 0
    {
        return Err(Rejection::new(
            "invalid_memory_bounds",
            "memory bounds must name exact FP32/BF16 widths and positive budgets",
        ));
    }
    if profile.admission.ab_repeat_count < SAMPLE_COUNT_MINIMUM {
        return Err(Rejection::new(
            "invalid_capacity_admission",
            "A/B admission repeat count must be positive",
        ));
    }
    finite_non_negative(
        profile.admission.minimum_candidate_gain_fraction,
        "minimum_candidate_gain_fraction",
    )?;
    let mut seen = BTreeSet::new();
    for candidate in &profile.candidates {
        if !candidate_supported(&candidate.id, candidate.per_card_slot_count) {
            return Err(Rejection::new(
                "unsupported_capacity",
                format!(
                    "candidate '{}/{}' is not a supported grid capacity",
                    candidate.id, candidate.per_card_slot_count
                ),
            ));
        }
        if !seen.insert((candidate.id.clone(), candidate.per_card_slot_count)) {
            return Err(Rejection::new(
                "duplicate_capacity_candidate",
                format!("candidate '{}' appears more than once", candidate.id),
            ));
        }
        if candidate.maximum_wave_count < MINIMUM_WAVE_COUNT
            || candidate.maximum_wave_count > MAXIMUM_WAVE_COUNT
        {
            return Err(Rejection::new(
                "unsupported_wave_count",
                format!(
                    "candidate '{}' wave count is outside {}..={}",
                    candidate.id, MINIMUM_WAVE_COUNT, MAXIMUM_WAVE_COUNT
                ),
            ));
        }
        finite_non_negative(
            candidate.minimum_ab_throughput_gain_fraction,
            "minimum_ab_throughput_gain_fraction",
        )?;
        finite_non_negative(
            candidate.maximum_p99_latency_regression_fraction,
            "maximum_p99_latency_regression_fraction",
        )?;
    }
    Ok(())
}

/// Derive the checked card-local memory plan for one candidate.
///
/// r[verify tenstorrent.rwkv7_p150x2.scalable_card_local_batching.invalid_memory]
pub fn plan_candidate_memory(
    candidate: &CapacityCandidate,
    extent: &ModelExtent,
    bounds: &MemoryBounds,
) -> Result<MemoryPlan, Rejection> {
    let recurrent_bytes = extent
        .recurrent_elements_per_slot()?
        .checked_mul(bounds.fp32_bytes_per_element)
        .and_then(|per_slot| per_slot.checked_mul(candidate.per_card_slot_count))
        .ok_or_else(|| {
            Rejection::new(
                "capacity_arithmetic_overflow",
                "recurrent-state bytes overflow the address space",
            )
        })?;
    let token_shift_bytes = extent
        .token_shift_elements_per_slot()?
        .checked_mul(bounds.bf16_bytes_per_element)
        .and_then(|per_slot| per_slot.checked_mul(candidate.per_card_slot_count))
        .ok_or_else(|| {
            Rejection::new(
                "capacity_arithmetic_overflow",
                "token-shift bytes overflow the address space",
            )
        })?;
    let temporary_bytes = candidate
        .per_card_slot_count
        .checked_mul(bounds.temporary_tensor_bytes_per_slot)
        .ok_or_else(|| {
            Rejection::new(
                "capacity_arithmetic_overflow",
                "temporary tensor estimate overflows the address space",
            )
        })?;
    let total_required_bytes = recurrent_bytes
        .checked_add(token_shift_bytes)
        .and_then(|sum| sum.checked_add(temporary_bytes))
        .and_then(|sum| sum.checked_add(bounds.safety_reserve_bytes))
        .ok_or_else(|| {
            Rejection::new(
                "capacity_arithmetic_overflow",
                "total memory requirement overflows the address space",
            )
        })?;
    if total_required_bytes > bounds.device_memory_budget_bytes {
        return Err(Rejection::new(
            "insufficient_device_memory",
            format!(
                "candidate '{}' requires {total_required_bytes} bytes, which exceeds the {}-byte card-local budget",
                candidate.id, bounds.device_memory_budget_bytes
            ),
        ));
    }
    Ok(MemoryPlan {
        recurrent_state_bytes: recurrent_bytes,
        token_shift_bytes,
        temporary_tensor_bytes: temporary_bytes,
        safety_reserve_bytes: bounds.safety_reserve_bytes,
        total_required_bytes,
        device_budget_bytes: bounds.device_memory_budget_bytes,
    })
}

/// Plan every candidate in the profile. A single unsafe candidate is rejected
/// with its exact bound so the sweep retains the strongest blocker.
///
/// r[verify tenstorrent.rwkv7_p150x2.scalable_card_local_batching.valid]
pub fn plan_capacity(
    profile: &CapacityProfile,
    extent: &ModelExtent,
) -> Result<Vec<CapacityPlan>, Rejection> {
    validate_capacity_profile(profile)?;
    let mut plans = Vec::with_capacity(profile.candidates.len());
    for candidate in &profile.candidates {
        let memory = plan_candidate_memory(candidate, extent, &profile.memory)?;
        plans.push(CapacityPlan {
            candidate_id: candidate.id.clone(),
            per_card_slot_count: candidate.per_card_slot_count,
            maximum_wave_count: candidate.maximum_wave_count,
            memory,
            minimum_ab_throughput_gain_fraction: candidate.minimum_ab_throughput_gain_fraction,
            maximum_p99_latency_regression_fraction: candidate
                .maximum_p99_latency_regression_fraction,
        });
    }
    Ok(plans)
}

/// Build a deterministic card-local plan for one candidate by id.
pub fn plan_selected_capacity(
    profile: &CapacityProfile,
    extent: &ModelExtent,
    candidate_id: &str,
) -> Result<CapacityPlan, Rejection> {
    let plans = plan_capacity(profile, extent)?;
    plans
        .into_iter()
        .find(|plan| plan.candidate_id == candidate_id)
        .ok_or_else(|| {
            Rejection::new(
                "unsupported_capacity",
                format!("candidate '{candidate_id}' is not present in the profile"),
            )
        })
}

/// Evaluate one physical candidate against the current admitted runtime with
/// order-balanced A/B evidence. Simulated evidence never promotes.
///
/// r[impl tenstorrent.rwkv7_p150x2.scalable_card_local_batching.target_miss]
pub fn evaluate_candidate(
    plan: &CapacityPlan,
    measurement: &CapacityMeasurement,
    current: &CurrentAdmittedMeasurement,
) -> Result<CapacityEvaluation, Rejection> {
    finite_non_negative(
        measurement.aggregate_tokens_per_second,
        "aggregate_tokens_per_second",
    )?;
    finite_positive(measurement.p99_per_token_seconds, "p99_per_token_seconds")?;
    finite_positive(
        current.aggregate_tokens_per_second,
        "current aggregate_tokens_per_second",
    )?;
    finite_positive(
        current.p99_per_token_seconds,
        "current p99_per_token_seconds",
    )?;
    if measurement.candidate_id != plan.candidate_id {
        return Err(Rejection::new(
            "capacity_measurement_mismatch",
            format!(
                "candidate '{}' plan evaluated against a '{}' measurement",
                plan.candidate_id, measurement.candidate_id
            ),
        ));
    }
    if measurement.samples < SAMPLE_COUNT_MINIMUM {
        return Err(Rejection::new(
            "insufficient_capacity_samples",
            format!(
                "candidate '{}' has {} samples; minimum is {SAMPLE_COUNT_MINIMUM}",
                plan.candidate_id, measurement.samples
            ),
        ));
    }
    if measurement.simulated {
        return Err(Rejection::new(
            "simulated_capacity_claim",
            format!(
                "candidate '{}' reported simulated throughput; simulator evidence cannot promote",
                plan.candidate_id
            ),
        ));
    }
    if !measurement.parity_exact {
        return Err(Rejection::new(
            "capacity_parity_mismatch",
            format!(
                "candidate '{}' did not preserve exact FP32-reference tokens",
                plan.candidate_id
            ),
        ));
    }
    let gain_fraction = (measurement.aggregate_tokens_per_second
        - current.aggregate_tokens_per_second)
        / current.aggregate_tokens_per_second;
    let latency_regression_fraction = (measurement.p99_per_token_seconds
        - current.p99_per_token_seconds)
        / current.p99_per_token_seconds;
    let mut details = Vec::new();
    let above_minimum_gain =
        gain_fraction + TIGHT_FLOAT_TOLERANCE >= plan.minimum_ab_throughput_gain_fraction;
    let below_latency_bound = latency_regression_fraction
        <= plan.maximum_p99_latency_regression_fraction + TIGHT_FLOAT_TOLERANCE;
    if !above_minimum_gain {
        details.push(format!(
            "throughput gain {gain_fraction} is below the {}-fraction bound",
            plan.minimum_ab_throughput_gain_fraction
        ));
    }
    if !below_latency_bound {
        details.push(format!(
            "p99 per-token latency regression {latency_regression_fraction} is above the {}-fraction bound",
            plan.maximum_p99_latency_regression_fraction
        ));
    }
    Ok(CapacityEvaluation {
        candidate_id: plan.candidate_id.clone(),
        promoted: above_minimum_gain && below_latency_bound,
        measured_gain_fraction: gain_fraction,
        latency_regression_fraction,
        details,
    })
}

/// Compute the p99 time per generated token from paired (latency, token
/// count) samples. This is the honest cross-capacity latency metric: it
/// normalizes per-request wall latency by the work done, so a wider batch
/// that amortizes a fixed device stall is not penalized for serving more
/// tokens.
///
/// r[impl tenstorrent.rwkv7_p150x2.scalable_card_local_batching.valid]
pub fn per_token_p99_seconds(samples: &[(f64, usize)]) -> Result<f64, Rejection> {
    if samples.is_empty() {
        return Err(Rejection::new(
            "empty_per_token_samples",
            "per-token latency samples must not be empty",
        ));
    }
    let mut per_token = Vec::with_capacity(samples.len());
    for (latency_seconds, token_count) in samples {
        finite_positive(*latency_seconds, "latency_seconds")?;
        if *token_count == 0 {
            return Err(Rejection::new(
                "empty_token_count",
                "per-token latency sample has zero tokens",
            ));
        }
        per_token.push(latency_seconds / *token_count as f64);
    }
    per_token.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let rank = (PERCENTILE_P99 * per_token.len() as f64).ceil() as usize;
    let index = rank
        .saturating_sub(1)
        .min(per_token.len().saturating_sub(1));
    let value = per_token[index];
    finite_positive(value, "p99 per-token time")?;
    Ok(value)
}

/// A pure admit-advance-retain step is fully covered by `evaluate_candidate`
/// plus `sweep_outcome`. The imperative launcher iterates candidate plans in
/// profile order and stops after the first rejected plan, mirroring the pure
/// sweep's last-admitted rule without re-deriving any decision.
///
/// Serialize every admitted plan with its measured blocker so the receipt
/// package stays deterministic and device-free. The shell writes the result.
pub fn sweep_outcome(
    profile: &CapacityProfile,
    extent: &ModelExtent,
    measurements: &BTreeMap<String, CapacityMeasurement>,
    current: &CurrentAdmittedMeasurement,
) -> Result<SweepOutcome, Rejection> {
    let plans = plan_capacity(profile, extent)?;
    let mut evaluations = Vec::with_capacity(plans.len());
    let mut blocker_code: Option<String> = None;
    let mut blocker_detail: Option<String> = None;
    for plan in &plans {
        match measurements.get(&plan.candidate_id) {
            Some(measurement) => match evaluate_candidate(plan, measurement, current) {
                Ok(evaluation) => {
                    if !evaluation.promoted && blocker_code.is_none() {
                        blocker_code = Some("capacity_admission_failed".to_owned());
                        blocker_detail = Some(evaluation.details.join("; "));
                    }
                    evaluations.push(evaluation);
                }
                Err(rejection) => {
                    if blocker_code.is_none() {
                        blocker_code = Some(rejection.code.to_owned());
                        blocker_detail = Some(rejection.detail.clone());
                    }
                    evaluations.push(CapacityEvaluation {
                        candidate_id: plan.candidate_id.clone(),
                        promoted: false,
                        measured_gain_fraction: 0.0,
                        latency_regression_fraction: 0.0,
                        details: vec![rejection.to_string()],
                    });
                }
            },
            None => {
                if blocker_code.is_none() {
                    blocker_code = Some("candidate_measurement_missing".to_owned());
                    blocker_detail = Some(format!(
                        "candidate '{}' has no measured evidence",
                        plan.candidate_id
                    ));
                }
            }
        }
    }
    let admitted_candidate_id = plans
        .iter()
        .filter(|plan| {
            evaluations.iter().any(|evaluation| {
                evaluation.candidate_id == plan.candidate_id && evaluation.promoted
            })
        })
        .map(|plan| plan.candidate_id.clone())
        .next_back()
        .unwrap_or_else(|| "B4".to_owned());
    Ok(SweepOutcome {
        admitted_candidate_id,
        blocker_code,
        blocker_detail,
        evaluations,
        plans,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_BLAKE3: &str = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88";

    fn extent() -> ModelExtent {
        ModelExtent {
            layer_count: 12,
            hidden_size: 768,
            head_count: 12,
            head_size: 64,
        }
    }

    fn bounds() -> MemoryBounds {
        MemoryBounds {
            fp32_bytes_per_element: 4,
            bf16_bytes_per_element: 2,
            temporary_tensor_bytes_per_slot: 5 * 1024 * 1024,
            device_memory_budget_bytes: 3 * 1024 * 1024 * 1024,
            safety_reserve_bytes: 256 * 1024 * 1024,
        }
    }

    fn profile() -> CapacityProfile {
        CapacityProfile {
            schema_version: CAPACITY_SCHEMA_VERSION,
            profile_id: EXPECTED_CAPACITY_PROFILE_ID.to_owned(),
            model_profile_blake3: MODEL_BLAKE3.to_owned(),
            candidates: vec![
                CapacityCandidate {
                    id: "B4".to_owned(),
                    per_card_slot_count: 4,
                    maximum_wave_count: 2,
                    minimum_ab_throughput_gain_fraction: 0.0,
                    maximum_p99_latency_regression_fraction: 0.10,
                },
                CapacityCandidate {
                    id: "B16".to_owned(),
                    per_card_slot_count: 16,
                    maximum_wave_count: 2,
                    minimum_ab_throughput_gain_fraction: 0.0,
                    maximum_p99_latency_regression_fraction: 0.10,
                },
                CapacityCandidate {
                    id: "B32".to_owned(),
                    per_card_slot_count: 32,
                    maximum_wave_count: 2,
                    minimum_ab_throughput_gain_fraction: 0.0,
                    maximum_p99_latency_regression_fraction: 0.10,
                },
                CapacityCandidate {
                    id: "B64".to_owned(),
                    per_card_slot_count: 64,
                    maximum_wave_count: 2,
                    minimum_ab_throughput_gain_fraction: 0.0,
                    maximum_p99_latency_regression_fraction: 0.10,
                },
            ],
            memory: bounds(),
            admission: CapacityAdmissionThresholds {
                ab_repeat_count: 3,
                minimum_candidate_gain_fraction: 0.0,
            },
        }
    }

    fn candidate<'a>(profile: &'a CapacityProfile, id: &str) -> &'a CapacityCandidate {
        profile
            .candidates
            .iter()
            .find(|candidate| candidate.id == id)
            .expect("candidate exists")
    }

    #[test]
    fn plans_all_four_supported_capacities_device_free() {
        let plan_set = plan_capacity(&profile(), &extent()).expect("valid profile plans");
        assert_eq!(plan_set.len(), 4);
        assert_eq!(
            plan_set
                .iter()
                .map(|plan| plan.candidate_id.as_str())
                .collect::<Vec<_>>(),
            vec!["B4", "B16", "B32", "B64"]
        );
        for plan in &plan_set {
            assert!(plan.memory.total_required_bytes <= plan.memory.device_budget_bytes);
            assert!(plan.memory.recurrent_state_bytes > 0);
            assert!(plan.memory.token_shift_bytes > 0);
        }
        let b64 = plan_set.last().expect("B64 last");
        let b4 = plan_set.first().expect("B4 first");
        assert!(b64.memory.recurrent_state_bytes > b4.memory.recurrent_state_bytes);
    }

    #[test]
    fn memory_arithmetic_matches_known_shapes() {
        let plan = plan_selected_capacity(&profile(), &extent(), "B4")
            .expect("B4 plan must fit the budget");
        // Recurrent: heads(12) * head_size(64) * head_size(64) * layers(12) *
        // fp32(4) * slots(4).
        assert_eq!(plan.memory.recurrent_state_bytes, 12 * 64 * 64 * 12 * 4 * 4);
        // Token shift: 2 buffers * hidden(768) * layers(12) * bf16(2) * slots(4).
        assert_eq!(plan.memory.token_shift_bytes, 2 * 768 * 12 * 2 * 4);
        // Temporary: slots(4) * 5 MiB.
        assert_eq!(plan.memory.temporary_tensor_bytes, 4 * 5 * 1024 * 1024);
    }

    #[test]
    fn rejects_unsupported_capacity_before_memory_work() {
        let mut profile = profile();
        profile.candidates.push(CapacityCandidate {
            id: "B8".to_owned(),
            per_card_slot_count: 8,
            maximum_wave_count: 2,
            minimum_ab_throughput_gain_fraction: 0.0,
            maximum_p99_latency_regression_fraction: 0.10,
        });
        let rejection = plan_capacity(&profile, &extent()).expect_err("B8 must be unsupported");
        assert_eq!(rejection.code, "unsupported_capacity");
    }

    #[test]
    fn rejects_duplicate_capacity_candidate() {
        let mut profile = profile();
        let duplicated = candidate(&profile, "B16").clone();
        profile.candidates.push(duplicated);
        let rejection =
            validate_capacity_profile(&profile).expect_err("duplicate must be rejected");
        assert_eq!(rejection.code, "duplicate_capacity_candidate");
    }

    #[test]
    fn rejects_arithmetic_overflow_memory() {
        let mut bounds = bounds();
        bounds.device_memory_budget_bytes = usize::MAX;
        let huge = CapacityCandidate {
            id: "B64".to_owned(),
            per_card_slot_count: 64,
            maximum_wave_count: 2,
            minimum_ab_throughput_gain_fraction: 0.0,
            maximum_p99_latency_regression_fraction: 0.10,
        };
        let mut extent = extent();
        extent.layer_count = usize::MAX / 4;
        let rejection =
            plan_candidate_memory(&huge, &extent, &bounds).expect_err("overflow must fail");
        assert_eq!(rejection.code, "capacity_arithmetic_overflow");
    }

    #[test]
    fn rejects_insufficient_device_memory() {
        let mut profile = profile();
        profile.memory.device_memory_budget_bytes = 1024 * 1024;
        let rejection = plan_capacity(&profile, &extent()).expect_err("tiny budget must fail");
        assert_eq!(rejection.code, "insufficient_device_memory");
        assert!(rejection.detail.contains("exceeds the"));
    }

    #[test]
    fn recycler_allocates_lowest_slot_and_zeroes_before_reuse() {
        let mut recycler = SlotRecycler::new(16).expect("valid capacity");
        let first = recycler.allocate().expect("slot 0 free");
        assert_eq!(first.slot, 0);
        assert!(!first.requires_state_zeroing);
        let second = recycler.allocate().expect("slot 1 free");
        assert_eq!(second.slot, 1);
        recycler.release(0).expect("release completes slot 0");
        // A reallocation without an explicit zero still flags the obligation.
        let third = recycler.allocate().expect("slot 0 reallocated");
        assert_eq!(third.slot, 0);
        assert!(third.requires_state_zeroing, "recycled slot must be zeroed");
        recycler.release(1).expect("release completes slot 1");
        let fourth = recycler.allocate().expect("slot 1 reallocated");
        assert_eq!(fourth.slot, 1);
        assert!(fourth.requires_state_zeroing);
        // The owner clears the dirty flag while the slot is not live.
        recycler.release(1).expect("released after flagging");
        recycler.zero(1).expect("explicit zero clears dirty state");
        let clean = recycler.allocate().expect("slot 1 clean");
        assert_eq!(clean.slot, 1);
        assert!(!clean.requires_state_zeroing);
    }

    #[test]
    fn recycler_rejects_live_capacity_exhaustion() {
        let mut recycler = SlotRecycler::new(1).expect("valid capacity");
        let allocation = recycler.allocate().expect("slot 0");
        assert!(!allocation.requires_state_zeroing);
        let rejection = recycler
            .allocate()
            .expect_err("single-slot recycler is full");
        assert_eq!(rejection.code, "slot_capacity_exhausted");
    }

    #[test]
    fn recycler_rejects_release_of_unknown_slot() {
        let mut recycler = SlotRecycler::new(4).expect("valid capacity");
        let rejection = recycler.release(3).expect_err("slot 3 is never live");
        assert_eq!(rejection.code, "unknown_slot_owner");
    }

    #[test]
    fn recycler_rejects_zeroing_a_clean_slot() {
        let mut recycler = SlotRecycler::new(4).expect("valid capacity");
        let allocation = recycler.allocate().expect("slot 0");
        recycler.release(allocation.slot).expect("released");
        let rejection = recycler
            .zero(allocation.slot)
            .and_then(|()| recycler.zero(allocation.slot))
            .expect_err("double zero is an error");
        assert_eq!(rejection.code, "zero_clean_slot");
    }

    #[test]
    fn evaluation_promotes_on_gain_within_latency_bound() {
        let plan = plan_selected_capacity(&profile(), &extent(), "B16").expect("B16 fits");
        let evaluation = evaluate_candidate(
            &plan,
            &CapacityMeasurement {
                candidate_id: "B16".to_owned(),
                simulated: false,
                parity_exact: true,
                samples: 3,
                aggregate_tokens_per_second: 250.0,
                p99_per_token_seconds: 0.003,
            },
            &CurrentAdmittedMeasurement {
                candidate_id: "B4".to_owned(),
                aggregate_tokens_per_second: 200.0,
                p99_per_token_seconds: 0.0068,
            },
        )
        .expect("valid A/B evaluation");
        assert!(evaluation.promoted);
        assert!(evaluation.measured_gain_fraction > 0.20);
        assert!(evaluation.details.is_empty());
    }

    #[test]
    fn evaluation_retains_blocker_on_latency_regression() {
        let plan = plan_selected_capacity(&profile(), &extent(), "B16").expect("B16 fits");
        let evaluation = evaluate_candidate(
            &plan,
            &CapacityMeasurement {
                candidate_id: "B16".to_owned(),
                simulated: false,
                parity_exact: true,
                samples: 3,
                aggregate_tokens_per_second: 400.0,
                p99_per_token_seconds: 0.010,
            },
            &CurrentAdmittedMeasurement {
                candidate_id: "B4".to_owned(),
                aggregate_tokens_per_second: 200.0,
                p99_per_token_seconds: 0.0068,
            },
        )
        .expect("valid A/B evaluation");
        assert!(!evaluation.promoted);
        assert!(
            evaluation
                .details
                .iter()
                .any(|detail| { detail.contains("p99 per-token latency regression") })
        );
    }

    #[test]
    fn evaluation_rejects_simulated_claim() {
        let plan = plan_selected_capacity(&profile(), &extent(), "B32").expect("B32 fits");
        let rejection = evaluate_candidate(
            &plan,
            &CapacityMeasurement {
                candidate_id: "B32".to_owned(),
                simulated: true,
                parity_exact: true,
                samples: 3,
                aggregate_tokens_per_second: 9999.0,
                p99_per_token_seconds: 0.003,
            },
            &CurrentAdmittedMeasurement {
                candidate_id: "B4".to_owned(),
                aggregate_tokens_per_second: 200.0,
                p99_per_token_seconds: 0.0068,
            },
        )
        .expect_err("simulator evidence cannot promote");
        assert_eq!(rejection.code, "simulated_capacity_claim");
    }

    #[test]
    fn evaluation_rejects_parity_mismatch() {
        let plan = plan_selected_capacity(&profile(), &extent(), "B32").expect("B32 fits");
        let rejection = evaluate_candidate(
            &plan,
            &CapacityMeasurement {
                candidate_id: "B32".to_owned(),
                simulated: false,
                parity_exact: false,
                samples: 3,
                aggregate_tokens_per_second: 9999.0,
                p99_per_token_seconds: 0.003,
            },
            &CurrentAdmittedMeasurement {
                candidate_id: "B4".to_owned(),
                aggregate_tokens_per_second: 200.0,
                p99_per_token_seconds: 0.0068,
            },
        )
        .expect_err("parity mismatch cannot promote");
        assert_eq!(rejection.code, "capacity_parity_mismatch");
    }

    #[test]
    fn evaluation_rejects_non_finite_or_malformed_metrics() {
        let plan = plan_selected_capacity(&profile(), &extent(), "B32").expect("B32 fits");
        for (aggregate, p99, expected) in [
            (f64::NAN, 0.05, "invalid_capacity_metric"),
            (100.0, f64::NEG_INFINITY, "invalid_capacity_metric"),
            (-5.0, 0.05, "invalid_capacity_metric"),
        ] {
            let rejection = evaluate_candidate(
                &plan,
                &CapacityMeasurement {
                    candidate_id: "B32".to_owned(),
                    simulated: false,
                    parity_exact: true,
                    samples: 3,
                    aggregate_tokens_per_second: aggregate,
                    p99_per_token_seconds: p99,
                },
                &CurrentAdmittedMeasurement {
                    candidate_id: "B4".to_owned(),
                    aggregate_tokens_per_second: 200.0,
                    p99_per_token_seconds: 0.0068,
                },
            )
            .expect_err("malformed metrics must fail");
            assert_eq!(rejection.code, expected);
        }
    }

    #[test]
    fn sweep_keeps_last_admitted_capacity_on_partial_success() {
        let mut measurements = BTreeMap::new();
        measurements.insert(
            "B4".to_owned(),
            CapacityMeasurement {
                candidate_id: "B4".to_owned(),
                simulated: false,
                parity_exact: true,
                samples: 3,
                aggregate_tokens_per_second: 200.0,
                p99_per_token_seconds: 0.0068,
            },
        );
        measurements.insert(
            "B16".to_owned(),
            CapacityMeasurement {
                candidate_id: "B16".to_owned(),
                simulated: false,
                parity_exact: true,
                samples: 3,
                aggregate_tokens_per_second: 240.0,
                p99_per_token_seconds: 0.004,
            },
        );
        // B32 misses the latency bound; B64 is missing entirely.
        measurements.insert(
            "B32".to_owned(),
            CapacityMeasurement {
                candidate_id: "B32".to_owned(),
                simulated: false,
                parity_exact: true,
                samples: 3,
                aggregate_tokens_per_second: 260.0,
                p99_per_token_seconds: 0.015,
            },
        );
        let outcome = sweep_outcome(
            &profile(),
            &extent(),
            &measurements,
            &CurrentAdmittedMeasurement {
                candidate_id: "B4".to_owned(),
                aggregate_tokens_per_second: 200.0,
                p99_per_token_seconds: 0.0068,
            },
        )
        .expect("deterministic sweep");
        assert_eq!(outcome.admitted_candidate_id, "B16");
        assert_eq!(
            outcome.blocker_code.as_deref(),
            Some("capacity_admission_failed")
        );
        assert!(
            outcome
                .blocker_detail
                .as_deref()
                .is_some_and(|detail| { detail.contains("p99 per-token latency regression") })
        );
    }

    #[test]
    fn sweep_never_claims_simulator_throughput() {
        let mut measurements = BTreeMap::new();
        measurements.insert(
            "B4".to_owned(),
            CapacityMeasurement {
                candidate_id: "B4".to_owned(),
                simulated: true,
                parity_exact: true,
                samples: 3,
                aggregate_tokens_per_second: 1000.0,
                p99_per_token_seconds: 0.003,
            },
        );
        let outcome = sweep_outcome(
            &profile(),
            &extent(),
            &measurements,
            &CurrentAdmittedMeasurement {
                candidate_id: "B4".to_owned(),
                aggregate_tokens_per_second: 200.0,
                p99_per_token_seconds: 0.0068,
            },
        )
        .expect("deterministic sweep");
        assert_eq!(outcome.admitted_candidate_id, "B4");
        assert_eq!(
            outcome.blocker_code.as_deref(),
            Some("simulated_capacity_claim")
        );
    }

    #[test]
    fn per_token_p99_normalizes_latency_by_work_done() {
        // A wide batch with one slow step but many tokens per request must
        // not be penalized for serving more work per unit of wall time.
        let samples = vec![
            (0.10, 4),  // 25 ms/token
            (0.16, 16), // 10 ms/token
            (0.28, 16), // 17.5 ms/token
            (0.12, 16), // 7.5 ms/token
        ];
        let p99 = per_token_p99_seconds(&samples).expect("valid per-token p99");
        assert!(p99 > 0.0 && p99 < 0.04, "p99 per token is {p99}");
    }

    #[test]
    fn per_token_p99_rejects_empty_samples() {
        let rejection = per_token_p99_seconds(&[]).expect_err("empty must fail");
        assert_eq!(rejection.code, "empty_per_token_samples");
    }

    #[test]
    fn per_token_p99_rejects_zero_token_samples() {
        let rejection = per_token_p99_seconds(&[(0.1, 0)]).expect_err("zero tokens must fail");
        assert_eq!(rejection.code, "empty_token_count");
    }
}
