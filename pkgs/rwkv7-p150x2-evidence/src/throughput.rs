//! Pure throughput-profile and scaling-evidence admission.
//!
//! r[impl tenstorrent.rwkv7_p150x2.throughput_profile]
//! r[impl tenstorrent.rwkv7_p150x2.throughput_metrics]
//! r[impl tenstorrent.rwkv7_p150x2.throughput_admission]

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};

use serde::{Deserialize, Serialize};

const THROUGHPUT_SCHEMA_VERSION: u32 = 1;
const THROUGHPUT_RECEIPT_KIND: &str = "rwkv7_p150x2_throughput";
const TWO_CARD_DEVICE_COUNT: usize = 2;
const PERCENTILE_P50: f64 = 0.50;
const PERCENTILE_P99: f64 = 0.99;
const FRACTION_DENOMINATOR: f64 = 1.0;
const FLOAT_COMPARISON_TOLERANCE: f64 = 0.000_000_001;
const MINIMUM_SAMPLE_COUNT: usize = 3;
const MAXIMUM_SCHEDULER_WAVE_COUNT: usize = 2;
const EXPECTED_ONE_CARD_DEVICE_ID: usize = 0;
const EXPECTED_TWO_CARD_DEVICE_IDS: [usize; TWO_CARD_DEVICE_COUNT] = [0, 1];
const EXPECTED_SCHEDULER_MODE: &str = "deterministic_continuous_batching";
const EXPECTED_TIE_BREAK: &str = "least_loaded_card_then_lowest_card_then_lowest_slot";
const EXPECTED_REPLICA_MODE: &str = "independent_card_replicas";
const EXPECTED_STATE_RESIDENCY: &str = "device_fp32";
const EXPECTED_TIMED_REGION: &str = "decode_only";
const EXPECTED_MACHINE_FACT_SOURCE: &str = "tt-smi -s --snapshot_no_tty";

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

impl Display for Rejection {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for Rejection {}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ThroughputProfile {
    pub schema_version: u32,
    pub profile_id: String,
    pub model_profile_blake3: String,
    pub repeat_count: usize,
    pub workload: WorkloadProfile,
    pub scheduler: SchedulerProfile,
    pub runtime: RuntimeProfile,
    pub timing: TimingProfile,
    pub admission: AdmissionProfile,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadProfile {
    pub request_count: usize,
    pub short_prompt_token_count: usize,
    pub long_prompt_token_count: usize,
    pub generated_token_count: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SchedulerProfile {
    pub mode: String,
    pub tie_break: String,
    pub per_card_slot_count: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeProfile {
    pub replica_mode: String,
    pub recurrent_state_residency: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TimingProfile {
    pub timed_region: String,
    pub warmup_token_count: usize,
    pub exclude_state_readback: bool,
    pub exclude_profiler_flush: bool,
    pub exclude_control_replay: bool,
    pub exclude_artifact_writes: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AdmissionProfile {
    pub minimum_dual_card_speedup: f64,
    pub maximum_p99_latency_regression_fraction: f64,
    pub require_physical_hardware: bool,
    pub require_profiler_attribution: bool,
    pub machine_fact_source: String,
    pub one_card_physical_device_id: usize,
    pub two_card_physical_device_ids: [usize; TWO_CARD_DEVICE_COUNT],
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RawThroughputEvidence {
    pub schema_version: u32,
    pub profile_identity_blake3: String,
    pub machine_facts: MachineFactsEvidence,
    pub execution: ExecutionEvidence,
    pub one_card_runs: Vec<ThroughputRun>,
    pub two_card_runs: Vec<ThroughputRun>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MachineFactsEvidence {
    pub source: String,
    pub before_blake3: String,
    pub after_blake3: String,
    pub physical_device_ids: [usize; TWO_CARD_DEVICE_COUNT],
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionEvidence {
    pub scheduler_mode: String,
    pub scheduler_tie_break: String,
    pub replica_mode: String,
    pub recurrent_state_residency: String,
    pub timed_region: String,
    pub warmup_token_count: usize,
    pub program_cache_stable: bool,
    pub dynamic_token_and_state_bindings_validated: bool,
    pub state_readback_during_timing: bool,
    pub profiler_flush_during_timing: bool,
    pub control_replay_during_timing: bool,
    pub artifact_writes_during_timing: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ThroughputRun {
    pub run_id: String,
    pub physical_device_ids: Vec<usize>,
    pub physical_hardware: bool,
    pub profiler_attributed: bool,
    pub elapsed_seconds: f64,
    pub requests: Vec<RequestMeasurement>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RequestMeasurement {
    pub request_id: String,
    pub card_id: usize,
    pub state_slot: usize,
    #[serde(default)]
    pub wave_index: usize,
    pub token_count: usize,
    pub latency_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TopologyMetrics {
    pub device_ids: Vec<usize>,
    pub sample_count: usize,
    pub request_count: usize,
    pub token_count: usize,
    pub aggregate_tokens_per_second: f64,
    pub per_card_tokens_per_second: BTreeMap<usize, f64>,
    pub p50_latency_seconds: f64,
    pub p99_latency_seconds: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ThroughputReceipt {
    pub schema_version: u32,
    pub kind: String,
    pub profile_identity_blake3: String,
    pub one_card: TopologyMetrics,
    pub two_card: TopologyMetrics,
    pub dual_card_speedup: f64,
    pub scaling_efficiency: f64,
    pub p99_latency_regression_fraction: f64,
    pub identity_blake3: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ThroughputAdmission {
    pub identity_blake3: String,
    pub dual_card_speedup: f64,
    pub scaling_efficiency: f64,
    pub p99_latency_regression_fraction: f64,
    pub aggregate_tokens_per_second: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub struct SlotAssignment {
    pub request_index: usize,
    pub card_id: usize,
    pub state_slot: usize,
    pub wave_index: usize,
}

pub fn deterministic_slot_assignments(
    request_count: usize,
    card_ids: &[usize],
    per_card_slot_count: usize,
) -> Result<Vec<SlotAssignment>, Rejection> {
    if request_count == 0 || card_ids.is_empty() || per_card_slot_count == 0 {
        return Err(Rejection::new(
            "invalid_scheduler_input",
            "request, card, and per-card slot counts must be positive",
        ));
    }
    let unique_cards = card_ids.iter().copied().collect::<BTreeSet<_>>();
    if unique_cards.len() != card_ids.len() {
        return Err(Rejection::new(
            "duplicate_scheduler_card",
            "scheduler card IDs must be unique",
        ));
    }
    let active_capacity = card_ids
        .len()
        .checked_mul(per_card_slot_count)
        .ok_or_else(|| {
            Rejection::new(
                "scheduler_capacity_overflow",
                "scheduler capacity does not fit in usize",
            )
        })?;
    let total_capacity = active_capacity
        .checked_mul(MAXIMUM_SCHEDULER_WAVE_COUNT)
        .ok_or_else(|| {
            Rejection::new(
                "scheduler_capacity_overflow",
                "scheduler wave capacity does not fit in usize",
            )
        })?;
    if request_count > total_capacity {
        return Err(Rejection::new(
            "scheduler_capacity_exceeded",
            format!("request count is {request_count}, bounded wave capacity is {total_capacity}"),
        ));
    }
    let mut card_loads = vec![0usize; card_ids.len()];
    let mut assignments = Vec::with_capacity(request_count);
    for request_index in 0..request_count {
        let minimum_load = *card_loads.iter().min().expect("card loads are non-empty");
        let card_index = card_loads
            .iter()
            .position(|load| *load == minimum_load)
            .expect("minimum card load exists");
        let state_slot = card_loads[card_index] % per_card_slot_count;
        let wave_index = card_loads[card_index] / per_card_slot_count;
        assert!(wave_index < MAXIMUM_SCHEDULER_WAVE_COUNT);
        assignments.push(SlotAssignment {
            request_index,
            card_id: card_ids[card_index],
            state_slot,
            wave_index,
        });
        card_loads[card_index] += 1;
    }
    Ok(assignments)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TopologyKind {
    OneCard,
    TwoCard,
}

fn is_blake3_hex(value: &str) -> bool {
    const BLAKE3_HEX_LENGTH: usize = 64;
    value.len() == BLAKE3_HEX_LENGTH
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn finite_positive(value: f64, field: &str) -> Result<f64, Rejection> {
    if !value.is_finite() || value <= 0.0 {
        return Err(Rejection::new(
            "invalid_throughput_metric",
            format!("{field} must be finite and positive, got {value}"),
        ));
    }
    Ok(value)
}

fn finite_non_negative(value: f64, field: &str) -> Result<f64, Rejection> {
    if !value.is_finite() || value < 0.0 {
        return Err(Rejection::new(
            "invalid_throughput_metric",
            format!("{field} must be finite and non-negative, got {value}"),
        ));
    }
    Ok(value)
}

fn validate_profile(profile: &ThroughputProfile) -> Result<(), Rejection> {
    if profile.schema_version != THROUGHPUT_SCHEMA_VERSION {
        return Err(Rejection::new(
            "throughput_schema_mismatch",
            format!(
                "profile schema is {}, expected {THROUGHPUT_SCHEMA_VERSION}",
                profile.schema_version
            ),
        ));
    }
    if profile.profile_id.is_empty() || !is_blake3_hex(&profile.model_profile_blake3) {
        return Err(Rejection::new(
            "invalid_throughput_profile",
            "profile identity fields are empty or malformed",
        ));
    }
    if profile.repeat_count < MINIMUM_SAMPLE_COUNT {
        return Err(Rejection::new(
            "insufficient_throughput_samples",
            format!(
                "repeat_count is {}, minimum is {MINIMUM_SAMPLE_COUNT}",
                profile.repeat_count
            ),
        ));
    }
    let bounded_one_card_capacity = profile
        .scheduler
        .per_card_slot_count
        .saturating_mul(MAXIMUM_SCHEDULER_WAVE_COUNT);
    if profile.workload.request_count < TWO_CARD_DEVICE_COUNT
        || profile.workload.request_count > bounded_one_card_capacity
        || profile.workload.short_prompt_token_count == 0
        || profile.workload.long_prompt_token_count <= profile.workload.short_prompt_token_count
        || profile.workload.generated_token_count == 0
    {
        return Err(Rejection::new(
            "invalid_workload_profile",
            "workload must fit bounded one-card waves and name positive short, long, and generation bounds",
        ));
    }
    if profile.scheduler.mode != EXPECTED_SCHEDULER_MODE
        || profile.scheduler.tie_break != EXPECTED_TIE_BREAK
        || profile.scheduler.per_card_slot_count == 0
    {
        return Err(Rejection::new(
            "invalid_scheduler_profile",
            "scheduler must use deterministic continuous batching with at least one card-local slot",
        ));
    }
    if profile.runtime.replica_mode != EXPECTED_REPLICA_MODE
        || profile.runtime.recurrent_state_residency != EXPECTED_STATE_RESIDENCY
    {
        return Err(Rejection::new(
            "invalid_runtime_profile",
            "runtime must use independent replicas with device-resident FP32 state",
        ));
    }
    if profile.timing.timed_region != EXPECTED_TIMED_REGION
        || profile.timing.warmup_token_count == 0
        || !profile.timing.exclude_state_readback
        || !profile.timing.exclude_profiler_flush
        || !profile.timing.exclude_control_replay
        || !profile.timing.exclude_artifact_writes
    {
        return Err(Rejection::new(
            "invalid_timing_profile",
            "timed region must contain warmed decode work only",
        ));
    }
    finite_positive(
        profile.admission.minimum_dual_card_speedup,
        "minimum_dual_card_speedup",
    )?;
    finite_non_negative(
        profile.admission.maximum_p99_latency_regression_fraction,
        "maximum_p99_latency_regression_fraction",
    )?;
    if !profile.admission.require_physical_hardware
        || !profile.admission.require_profiler_attribution
        || profile.admission.machine_fact_source != EXPECTED_MACHINE_FACT_SOURCE
        || profile.admission.one_card_physical_device_id != EXPECTED_ONE_CARD_DEVICE_ID
        || profile.admission.two_card_physical_device_ids != EXPECTED_TWO_CARD_DEVICE_IDS
    {
        return Err(Rejection::new(
            "invalid_admission_profile",
            "admission must require physical devices 0 and 1 with profiler attribution",
        ));
    }
    Ok(())
}

fn validate_machine_facts(
    facts: &MachineFactsEvidence,
    profile: &ThroughputProfile,
) -> Result<(), Rejection> {
    if facts.source != profile.admission.machine_fact_source
        || !is_blake3_hex(&facts.before_blake3)
        || !is_blake3_hex(&facts.after_blake3)
        || facts.physical_device_ids != profile.admission.two_card_physical_device_ids
    {
        return Err(Rejection::new(
            "invalid_throughput_machine_facts",
            "machine facts lack the admitted source, identities, or physical devices",
        ));
    }
    Ok(())
}

fn validate_execution(
    execution: &ExecutionEvidence,
    profile: &ThroughputProfile,
) -> Result<(), Rejection> {
    if execution.scheduler_mode != profile.scheduler.mode
        || execution.scheduler_tie_break != profile.scheduler.tie_break
        || execution.replica_mode != profile.runtime.replica_mode
        || execution.recurrent_state_residency != profile.runtime.recurrent_state_residency
        || execution.timed_region != profile.timing.timed_region
        || execution.warmup_token_count != profile.timing.warmup_token_count
        || !execution.program_cache_stable
        || !execution.dynamic_token_and_state_bindings_validated
    {
        return Err(Rejection::new(
            "execution_profile_mismatch",
            "raw evidence does not bind the admitted scheduler, replica, state, and timing modes",
        ));
    }
    if execution.state_readback_during_timing
        || execution.profiler_flush_during_timing
        || execution.control_replay_during_timing
        || execution.artifact_writes_during_timing
    {
        return Err(Rejection::new(
            "timed_region_contaminated",
            "timed region includes evidence readback, profiler flush, control replay, or artifact writes",
        ));
    }
    Ok(())
}

fn expected_device_ids(kind: TopologyKind) -> &'static [usize] {
    match kind {
        TopologyKind::OneCard => std::slice::from_ref(&EXPECTED_ONE_CARD_DEVICE_ID),
        TopologyKind::TwoCard => &EXPECTED_TWO_CARD_DEVICE_IDS,
    }
}

fn validate_request_workload(
    run: &ThroughputRun,
    kind: TopologyKind,
    slot_count: usize,
) -> Result<Vec<usize>, Rejection> {
    if run.run_id.is_empty() || run.requests.is_empty() {
        return Err(Rejection::new(
            "empty_throughput_run",
            "run identity and request list must be non-empty",
        ));
    }
    if run.physical_device_ids != expected_device_ids(kind) {
        return Err(Rejection::new(
            "physical_device_mismatch",
            format!(
                "run '{}' used {:?}, expected {:?}",
                run.run_id,
                run.physical_device_ids,
                expected_device_ids(kind)
            ),
        ));
    }
    if !run.physical_hardware || !run.profiler_attributed {
        return Err(Rejection::new(
            "unattributed_throughput_run",
            format!(
                "run '{}' lacks physical hardware or profiler attribution",
                run.run_id
            ),
        ));
    }
    finite_positive(run.elapsed_seconds, "elapsed_seconds")?;

    let mut request_ids = BTreeSet::new();
    let mut used_cards = BTreeSet::new();
    let mut live_state_slots = BTreeSet::new();
    let mut token_counts = Vec::with_capacity(run.requests.len());
    for request in &run.requests {
        if request.request_id.is_empty() || !request_ids.insert(request.request_id.as_str()) {
            return Err(Rejection::new(
                "duplicate_request_identity",
                format!(
                    "run '{}' has an empty or duplicate request identity",
                    run.run_id
                ),
            ));
        }
        if !run.physical_device_ids.contains(&request.card_id) {
            return Err(Rejection::new(
                "request_card_mismatch",
                format!(
                    "request '{}' binds card {} outside run {:?}",
                    request.request_id, request.card_id, run.physical_device_ids
                ),
            ));
        }
        if request.state_slot >= slot_count {
            return Err(Rejection::new(
                "state_slot_out_of_range",
                format!(
                    "request '{}' uses slot {}, maximum is {}",
                    request.request_id,
                    request.state_slot,
                    slot_count.saturating_sub(1)
                ),
            ));
        }
        if request.wave_index >= MAXIMUM_SCHEDULER_WAVE_COUNT {
            return Err(Rejection::new(
                "wave_index_out_of_range",
                format!(
                    "request '{}' uses wave {}, maximum is {}",
                    request.request_id,
                    request.wave_index,
                    MAXIMUM_SCHEDULER_WAVE_COUNT.saturating_sub(1)
                ),
            ));
        }
        if !live_state_slots.insert((request.card_id, request.wave_index, request.state_slot)) {
            return Err(Rejection::new(
                "live_state_slot_reuse",
                format!(
                    "run '{}' assigns more than one live request to card {} wave {} slot {}",
                    run.run_id, request.card_id, request.wave_index, request.state_slot
                ),
            ));
        }
        if request.token_count == 0 {
            return Err(Rejection::new(
                "empty_request_measurement",
                format!("request '{}' has no measured tokens", request.request_id),
            ));
        }
        finite_positive(request.latency_seconds, "request latency_seconds")?;
        used_cards.insert(request.card_id);
        token_counts.push(request.token_count);
    }
    if kind == TopologyKind::TwoCard
        && used_cards != EXPECTED_TWO_CARD_DEVICE_IDS.into_iter().collect()
    {
        return Err(Rejection::new(
            "dual_card_not_independent",
            format!(
                "run '{}' did not assign distinct requests to both physical cards",
                run.run_id
            ),
        ));
    }
    token_counts.sort_unstable();
    Ok(token_counts)
}

fn validate_paired_workloads(
    one_card: &[ThroughputRun],
    two_card: &[ThroughputRun],
    profile: &ThroughputProfile,
) -> Result<(), Rejection> {
    if one_card.len() != profile.repeat_count || two_card.len() != profile.repeat_count {
        return Err(Rejection::new(
            "throughput_repeat_count_mismatch",
            format!(
                "one-card and two-card run counts are {} and {}; expected {}",
                one_card.len(),
                two_card.len(),
                profile.repeat_count
            ),
        ));
    }
    let mut run_ids = BTreeSet::new();
    for (index, (single, dual)) in one_card.iter().zip(two_card).enumerate() {
        if single.requests.len() != profile.workload.request_count
            || dual.requests.len() != profile.workload.request_count
        {
            return Err(Rejection::new(
                "throughput_request_count_mismatch",
                format!(
                    "paired sample {index} does not contain {} requests per topology",
                    profile.workload.request_count
                ),
            ));
        }
        if !run_ids.insert(single.run_id.as_str()) || !run_ids.insert(dual.run_id.as_str()) {
            return Err(Rejection::new(
                "duplicate_run_identity",
                "throughput run identities must be unique",
            ));
        }
        let single_tokens = validate_request_workload(
            single,
            TopologyKind::OneCard,
            profile.scheduler.per_card_slot_count,
        )?;
        let dual_tokens = validate_request_workload(
            dual,
            TopologyKind::TwoCard,
            profile.scheduler.per_card_slot_count,
        )?;
        if single_tokens != dual_tokens {
            return Err(Rejection::new(
                "throughput_workload_mismatch",
                format!("paired sample {index} does not measure the same token workload"),
            ));
        }
    }
    Ok(())
}

fn percentile(values: &mut [f64], percentile: f64) -> f64 {
    assert!(!values.is_empty());
    assert!(percentile > 0.0 && percentile <= FRACTION_DENOMINATOR);
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let rank = (percentile * values.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(values.len().saturating_sub(1));
    values[index]
}

fn summarize_topology(runs: &[ThroughputRun]) -> TopologyMetrics {
    assert!(!runs.is_empty());
    let mut throughput_samples = Vec::with_capacity(runs.len());
    let mut latency_samples = Vec::new();
    let mut per_card_tokens_per_second = runs[0]
        .physical_device_ids
        .iter()
        .map(|card_id| (*card_id, 0.0_f64))
        .collect::<BTreeMap<_, _>>();
    let mut request_count = 0usize;
    let mut token_count = 0usize;
    for run in runs {
        let run_token_count = run
            .requests
            .iter()
            .map(|request| request.token_count)
            .sum::<usize>();
        throughput_samples.push(run_token_count as f64 / run.elapsed_seconds);
        for card_id in &run.physical_device_ids {
            let card_tokens = run
                .requests
                .iter()
                .filter(|request| request.card_id == *card_id)
                .map(|request| request.token_count)
                .sum::<usize>();
            *per_card_tokens_per_second
                .get_mut(card_id)
                .expect("run card exists in topology") += card_tokens as f64 / run.elapsed_seconds;
        }
        request_count += run.requests.len();
        token_count += run_token_count;
        latency_samples.extend(runs_latency(run));
    }
    let sample_count = throughput_samples.len() as f64;
    let aggregate_tokens_per_second = throughput_samples.iter().sum::<f64>() / sample_count;
    for throughput in per_card_tokens_per_second.values_mut() {
        *throughput /= sample_count;
    }
    let p50_latency_seconds = percentile(&mut latency_samples.clone(), PERCENTILE_P50);
    let p99_latency_seconds = percentile(&mut latency_samples, PERCENTILE_P99);
    let metrics = TopologyMetrics {
        device_ids: runs[0].physical_device_ids.clone(),
        sample_count: runs.len(),
        request_count,
        token_count,
        aggregate_tokens_per_second,
        per_card_tokens_per_second,
        p50_latency_seconds,
        p99_latency_seconds,
    };
    assert!(metrics.aggregate_tokens_per_second.is_finite());
    assert!(metrics.p50_latency_seconds.is_finite());
    assert!(metrics.p99_latency_seconds.is_finite());
    metrics
}

fn runs_latency(run: &ThroughputRun) -> impl Iterator<Item = f64> + '_ {
    run.requests.iter().map(|request| request.latency_seconds)
}

fn canonical_identity<T: Serialize>(value: &T) -> Result<String, Rejection> {
    let bytes = serde_json::to_vec(value).map_err(|error| {
        Rejection::new(
            "throughput_serialization_failed",
            format!("failed to serialize throughput value: {error}"),
        )
    })?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

pub fn profile_identity(profile: &ThroughputProfile) -> Result<String, Rejection> {
    validate_profile(profile)?;
    canonical_identity(profile)
}

pub fn summarize_throughput(
    raw: &RawThroughputEvidence,
    profile: &ThroughputProfile,
) -> Result<ThroughputReceipt, Rejection> {
    validate_profile(profile)?;
    if raw.schema_version != THROUGHPUT_SCHEMA_VERSION {
        return Err(Rejection::new(
            "throughput_schema_mismatch",
            format!(
                "raw evidence schema is {}, expected {THROUGHPUT_SCHEMA_VERSION}",
                raw.schema_version
            ),
        ));
    }
    let expected_profile_identity = profile_identity(profile)?;
    if raw.profile_identity_blake3 != expected_profile_identity {
        return Err(Rejection::new(
            "throughput_profile_identity_mismatch",
            "raw evidence does not bind the admitted throughput profile",
        ));
    }
    validate_machine_facts(&raw.machine_facts, profile)?;
    validate_execution(&raw.execution, profile)?;
    validate_paired_workloads(&raw.one_card_runs, &raw.two_card_runs, profile)?;

    let one_card = summarize_topology(&raw.one_card_runs);
    let two_card = summarize_topology(&raw.two_card_runs);
    let dual_card_speedup =
        two_card.aggregate_tokens_per_second / one_card.aggregate_tokens_per_second;
    let scaling_efficiency = dual_card_speedup / TWO_CARD_DEVICE_COUNT as f64;
    let p99_latency_regression_fraction = (two_card.p99_latency_seconds
        - one_card.p99_latency_seconds)
        / one_card.p99_latency_seconds;
    finite_positive(dual_card_speedup, "dual_card_speedup")?;
    finite_non_negative(
        p99_latency_regression_fraction.max(0.0),
        "p99_latency_regression_fraction",
    )?;

    let mut receipt = ThroughputReceipt {
        schema_version: THROUGHPUT_SCHEMA_VERSION,
        kind: THROUGHPUT_RECEIPT_KIND.to_owned(),
        profile_identity_blake3: expected_profile_identity,
        one_card,
        two_card,
        dual_card_speedup,
        scaling_efficiency,
        p99_latency_regression_fraction,
        identity_blake3: String::new(),
    };
    receipt.identity_blake3 = canonical_identity(&receipt)?;
    Ok(receipt)
}

fn metric_close(left: f64, right: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= FLOAT_COMPARISON_TOLERANCE
}

fn topology_metrics_match(left: &TopologyMetrics, right: &TopologyMetrics) -> bool {
    left.device_ids == right.device_ids
        && left.sample_count == right.sample_count
        && left.request_count == right.request_count
        && left.token_count == right.token_count
        && metric_close(
            left.aggregate_tokens_per_second,
            right.aggregate_tokens_per_second,
        )
        && left.per_card_tokens_per_second.len() == right.per_card_tokens_per_second.len()
        && left
            .per_card_tokens_per_second
            .iter()
            .all(|(card_id, value)| {
                right
                    .per_card_tokens_per_second
                    .get(card_id)
                    .is_some_and(|expected| metric_close(*value, *expected))
            })
        && metric_close(left.p50_latency_seconds, right.p50_latency_seconds)
        && metric_close(left.p99_latency_seconds, right.p99_latency_seconds)
}

fn throughput_receipts_match(left: &ThroughputReceipt, right: &ThroughputReceipt) -> bool {
    left.schema_version == right.schema_version
        && left.kind == right.kind
        && left.profile_identity_blake3 == right.profile_identity_blake3
        && topology_metrics_match(&left.one_card, &right.one_card)
        && topology_metrics_match(&left.two_card, &right.two_card)
        && metric_close(left.dual_card_speedup, right.dual_card_speedup)
        && metric_close(left.scaling_efficiency, right.scaling_efficiency)
        && metric_close(
            left.p99_latency_regression_fraction,
            right.p99_latency_regression_fraction,
        )
        && left.identity_blake3 == right.identity_blake3
}

pub fn admit_throughput_receipt(
    receipt: &ThroughputReceipt,
    raw: &RawThroughputEvidence,
    profile: &ThroughputProfile,
) -> Result<ThroughputAdmission, Rejection> {
    let expected = summarize_throughput(raw, profile)?;
    if !throughput_receipts_match(receipt, &expected) {
        return Err(Rejection::new(
            "throughput_receipt_mismatch",
            "throughput receipt does not match the deterministic summary within the named float tolerance",
        ));
    }
    if receipt.dual_card_speedup + FLOAT_COMPARISON_TOLERANCE
        < profile.admission.minimum_dual_card_speedup
    {
        return Err(Rejection::new(
            "dual_card_speedup_below_bound",
            format!(
                "dual-card speedup is {}, minimum is {}",
                receipt.dual_card_speedup, profile.admission.minimum_dual_card_speedup
            ),
        ));
    }
    if receipt.p99_latency_regression_fraction
        > profile.admission.maximum_p99_latency_regression_fraction + FLOAT_COMPARISON_TOLERANCE
    {
        return Err(Rejection::new(
            "p99_latency_regression_above_bound",
            format!(
                "p99 latency regression is {}, maximum is {}",
                receipt.p99_latency_regression_fraction,
                profile.admission.maximum_p99_latency_regression_fraction
            ),
        ));
    }
    Ok(ThroughputAdmission {
        identity_blake3: receipt.identity_blake3.clone(),
        dual_card_speedup: receipt.dual_card_speedup,
        scaling_efficiency: receipt.scaling_efficiency,
        p99_latency_regression_fraction: receipt.p99_latency_regression_fraction,
        aggregate_tokens_per_second: receipt.two_card.aggregate_tokens_per_second,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_PROFILE_BLAKE3: &str =
        "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88";
    const RUN_COUNT: usize = 3;
    const REQUEST_COUNT: usize = 4;
    const TOKEN_COUNT: usize = 16;
    const SINGLE_ELAPSED_SECONDS: f64 = 1.0;
    const DUAL_ELAPSED_SECONDS: f64 = 0.5;
    const SINGLE_LATENCY_SECONDS: f64 = 0.100;
    const DUAL_LATENCY_SECONDS: f64 = 0.105;

    fn profile() -> ThroughputProfile {
        ThroughputProfile {
            schema_version: THROUGHPUT_SCHEMA_VERSION,
            profile_id: "rwkv7-p150x2-throughput-v1".to_owned(),
            model_profile_blake3: MODEL_PROFILE_BLAKE3.to_owned(),
            repeat_count: RUN_COUNT,
            workload: WorkloadProfile {
                request_count: REQUEST_COUNT,
                short_prompt_token_count: TWO_CARD_DEVICE_COUNT,
                long_prompt_token_count: TOKEN_COUNT,
                generated_token_count: TOKEN_COUNT,
            },
            scheduler: SchedulerProfile {
                mode: EXPECTED_SCHEDULER_MODE.to_owned(),
                tie_break: EXPECTED_TIE_BREAK.to_owned(),
                per_card_slot_count: REQUEST_COUNT,
            },
            runtime: RuntimeProfile {
                replica_mode: EXPECTED_REPLICA_MODE.to_owned(),
                recurrent_state_residency: EXPECTED_STATE_RESIDENCY.to_owned(),
            },
            timing: TimingProfile {
                timed_region: EXPECTED_TIMED_REGION.to_owned(),
                warmup_token_count: TOKEN_COUNT,
                exclude_state_readback: true,
                exclude_profiler_flush: true,
                exclude_control_replay: true,
                exclude_artifact_writes: true,
            },
            admission: AdmissionProfile {
                minimum_dual_card_speedup: 1.50,
                maximum_p99_latency_regression_fraction: 0.10,
                require_physical_hardware: true,
                require_profiler_attribution: true,
                machine_fact_source: EXPECTED_MACHINE_FACT_SOURCE.to_owned(),
                one_card_physical_device_id: EXPECTED_ONE_CARD_DEVICE_ID,
                two_card_physical_device_ids: EXPECTED_TWO_CARD_DEVICE_IDS,
            },
        }
    }

    fn run(
        run_id: String,
        dual: bool,
        elapsed_seconds: f64,
        latency_seconds: f64,
    ) -> ThroughputRun {
        let device_ids = if dual { vec![0, 1] } else { vec![0] };
        let requests = (0..REQUEST_COUNT)
            .map(|index| RequestMeasurement {
                request_id: format!("{run_id}-request-{index}"),
                card_id: if dual {
                    index % TWO_CARD_DEVICE_COUNT
                } else {
                    0
                },
                state_slot: if dual {
                    index / TWO_CARD_DEVICE_COUNT
                } else {
                    index
                },
                wave_index: 0,
                token_count: TOKEN_COUNT,
                latency_seconds,
            })
            .collect();
        ThroughputRun {
            run_id,
            physical_device_ids: device_ids,
            physical_hardware: true,
            profiler_attributed: true,
            elapsed_seconds,
            requests,
        }
    }

    fn raw(profile: &ThroughputProfile) -> RawThroughputEvidence {
        RawThroughputEvidence {
            schema_version: THROUGHPUT_SCHEMA_VERSION,
            profile_identity_blake3: profile_identity(profile).expect("profile identity"),
            machine_facts: MachineFactsEvidence {
                source: EXPECTED_MACHINE_FACT_SOURCE.to_owned(),
                before_blake3: MODEL_PROFILE_BLAKE3.to_owned(),
                after_blake3: MODEL_PROFILE_BLAKE3.to_owned(),
                physical_device_ids: EXPECTED_TWO_CARD_DEVICE_IDS,
            },
            execution: ExecutionEvidence {
                scheduler_mode: EXPECTED_SCHEDULER_MODE.to_owned(),
                scheduler_tie_break: EXPECTED_TIE_BREAK.to_owned(),
                replica_mode: EXPECTED_REPLICA_MODE.to_owned(),
                recurrent_state_residency: EXPECTED_STATE_RESIDENCY.to_owned(),
                timed_region: EXPECTED_TIMED_REGION.to_owned(),
                warmup_token_count: TOKEN_COUNT,
                program_cache_stable: true,
                dynamic_token_and_state_bindings_validated: true,
                state_readback_during_timing: false,
                profiler_flush_during_timing: false,
                control_replay_during_timing: false,
                artifact_writes_during_timing: false,
            },
            one_card_runs: (0..RUN_COUNT)
                .map(|index| {
                    run(
                        format!("single-{index}"),
                        false,
                        SINGLE_ELAPSED_SECONDS,
                        SINGLE_LATENCY_SECONDS,
                    )
                })
                .collect(),
            two_card_runs: (0..RUN_COUNT)
                .map(|index| {
                    run(
                        format!("dual-{index}"),
                        true,
                        DUAL_ELAPSED_SECONDS,
                        DUAL_LATENCY_SECONDS,
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn admits_physical_independent_dual_card_scaling() {
        let profile = profile();
        let raw = raw(&profile);
        let receipt = summarize_throughput(&raw, &profile).expect("valid summary");
        let admission =
            admit_throughput_receipt(&receipt, &raw, &profile).expect("valid admission");
        assert_eq!(admission.dual_card_speedup, 2.0);
        assert_eq!(admission.scaling_efficiency, 1.0);
        assert!(admission.p99_latency_regression_fraction <= 0.10);
        assert_eq!(receipt.two_card.device_ids, vec![0, 1]);
        assert_eq!(receipt.two_card.per_card_tokens_per_second.len(), 2);
        assert_eq!(receipt.two_card.p50_latency_seconds, DUAL_LATENCY_SECONDS);
    }

    #[test]
    fn admits_a_serialized_receipt_round_trip() {
        let profile = profile();
        let raw = raw(&profile);
        let receipt = summarize_throughput(&raw, &profile).expect("valid summary");
        let bytes = serde_json::to_vec(&receipt).expect("serialize receipt");
        let parsed: ThroughputReceipt = serde_json::from_slice(&bytes).expect("parse receipt");
        assert_eq!(parsed, receipt);
        admit_throughput_receipt(&parsed, &raw, &profile).expect("valid round trip");
    }

    #[test]
    fn scheduler_balances_cards_with_stable_ties() {
        let assignments = deterministic_slot_assignments(REQUEST_COUNT, &[0, 1], 2)
            .expect("valid scheduler input");
        assert_eq!(
            assignments,
            vec![
                SlotAssignment {
                    request_index: 0,
                    card_id: 0,
                    state_slot: 0,
                    wave_index: 0
                },
                SlotAssignment {
                    request_index: 1,
                    card_id: 1,
                    state_slot: 0,
                    wave_index: 0
                },
                SlotAssignment {
                    request_index: 2,
                    card_id: 0,
                    state_slot: 1,
                    wave_index: 0
                },
                SlotAssignment {
                    request_index: 3,
                    card_id: 1,
                    state_slot: 1,
                    wave_index: 0
                },
            ]
        );
    }

    #[test]
    fn scheduler_reuses_slots_only_in_the_next_wave() {
        let request_count = REQUEST_COUNT * MAXIMUM_SCHEDULER_WAVE_COUNT;
        let assignments = deterministic_slot_assignments(request_count, &[0], REQUEST_COUNT)
            .expect("bounded waves must fit");
        assert_eq!(assignments[0].wave_index, 0);
        assert_eq!(assignments[0].state_slot, 0);
        assert_eq!(assignments[REQUEST_COUNT].wave_index, 1);
        assert_eq!(assignments[REQUEST_COUNT].state_slot, 0);
        assert_eq!(assignments[request_count - 1].wave_index, 1);
        assert_eq!(assignments[request_count - 1].state_slot, REQUEST_COUNT - 1);
    }

    #[test]
    fn scheduler_rejects_capacity_overflow() {
        let rejection = deterministic_slot_assignments(
            (REQUEST_COUNT * MAXIMUM_SCHEDULER_WAVE_COUNT) + 1,
            &[0, 1],
            2,
        )
        .expect_err("must reject overflow");
        assert_eq!(rejection.code, "scheduler_capacity_exceeded");
    }

    #[test]
    fn rejects_unphysical_evidence() {
        let profile = profile();
        let mut raw = raw(&profile);
        raw.two_card_runs[0].physical_hardware = false;
        let rejection = summarize_throughput(&raw, &profile).expect_err("must reject estimate");
        assert_eq!(rejection.code, "unattributed_throughput_run");
    }

    #[test]
    fn rejects_dual_run_without_distinct_card_work() {
        let profile = profile();
        let mut raw = raw(&profile);
        for (state_slot, request) in raw.two_card_runs[0].requests.iter_mut().enumerate() {
            request.card_id = EXPECTED_ONE_CARD_DEVICE_ID;
            request.state_slot = state_slot;
        }
        let rejection = summarize_throughput(&raw, &profile).expect_err("must reject lockstep");
        assert_eq!(rejection.code, "dual_card_not_independent");
    }

    #[test]
    fn rejects_contaminated_timing_region() {
        let profile = profile();
        let mut raw = raw(&profile);
        raw.execution.state_readback_during_timing = true;
        let rejection = summarize_throughput(&raw, &profile).expect_err("must reject readback");
        assert_eq!(rejection.code, "timed_region_contaminated");
    }

    #[test]
    fn rejects_live_state_slot_reuse() {
        let profile = profile();
        let mut raw = raw(&profile);
        raw.one_card_runs[0].requests[1].state_slot = 0;
        let rejection = summarize_throughput(&raw, &profile).expect_err("must reject live reuse");
        assert_eq!(rejection.code, "live_state_slot_reuse");
    }

    #[test]
    fn rejects_a_wave_outside_the_scheduler_bound() {
        let profile = profile();
        let mut raw = raw(&profile);
        raw.one_card_runs[0].requests[0].wave_index = MAXIMUM_SCHEDULER_WAVE_COUNT;
        let rejection = summarize_throughput(&raw, &profile).expect_err("must reject wave");
        assert_eq!(rejection.code, "wave_index_out_of_range");
    }

    #[test]
    fn rejects_unbound_machine_facts() {
        let profile = profile();
        let mut raw = raw(&profile);
        raw.machine_facts.before_blake3 = "estimated".to_owned();
        let rejection = summarize_throughput(&raw, &profile).expect_err("must reject facts");
        assert_eq!(rejection.code, "invalid_throughput_machine_facts");
    }

    #[test]
    fn rejects_bf16_recurrent_state_metadata() {
        let profile = profile();
        let mut raw = raw(&profile);
        raw.execution.recurrent_state_residency = "device_bf16".to_owned();
        let rejection = summarize_throughput(&raw, &profile).expect_err("must reject BF16 state");
        assert_eq!(rejection.code, "execution_profile_mismatch");
    }

    #[test]
    fn rejects_stale_dynamic_bindings() {
        let profile = profile();
        let mut raw = raw(&profile);
        raw.execution.dynamic_token_and_state_bindings_validated = false;
        let rejection =
            summarize_throughput(&raw, &profile).expect_err("must reject stale binding");
        assert_eq!(rejection.code, "execution_profile_mismatch");
    }

    #[test]
    fn rejects_non_finite_elapsed_time() {
        let profile = profile();
        let mut raw = raw(&profile);
        raw.one_card_runs[0].elapsed_seconds = f64::NAN;
        let rejection = summarize_throughput(&raw, &profile).expect_err("must reject NaN");
        assert_eq!(rejection.code, "invalid_throughput_metric");
    }

    #[test]
    fn rejects_speedup_below_bound() {
        let profile = profile();
        let mut raw = raw(&profile);
        for run in &mut raw.two_card_runs {
            run.elapsed_seconds = SINGLE_ELAPSED_SECONDS;
        }
        let receipt = summarize_throughput(&raw, &profile).expect("valid slow summary");
        let rejection = admit_throughput_receipt(&receipt, &raw, &profile)
            .expect_err("must reject slow dual run");
        assert_eq!(rejection.code, "dual_card_speedup_below_bound");
    }

    #[test]
    fn rejects_p99_latency_regression_above_bound() {
        let profile = profile();
        let mut raw = raw(&profile);
        for run in &mut raw.two_card_runs {
            for request in &mut run.requests {
                request.latency_seconds = 0.120;
            }
        }
        let receipt = summarize_throughput(&raw, &profile).expect("valid high latency summary");
        let rejection = admit_throughput_receipt(&receipt, &raw, &profile)
            .expect_err("must reject high p99 regression");
        assert_eq!(rejection.code, "p99_latency_regression_above_bound");
    }

    #[test]
    fn rejects_tampered_receipt() {
        let profile = profile();
        let raw = raw(&profile);
        let mut receipt = summarize_throughput(&raw, &profile).expect("valid summary");
        receipt.dual_card_speedup = 9.0;
        let rejection = admit_throughput_receipt(&receipt, &raw, &profile)
            .expect_err("must reject receipt tampering");
        assert_eq!(rejection.code, "throughput_receipt_mismatch");
    }
}
