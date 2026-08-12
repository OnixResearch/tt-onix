//! Pure bounded canary planning, evidence validation, and interlock decisions.

use crate::persistent_decode_monitoring::{
    MONITORING_EVENT_KIND, MONITORING_SCHEMA_VERSION, MonitoringVerdict,
    PersistentAdmissionRecommendation, PersistentDecodeMonitoringReceipt, parse_monitoring_policy,
};
use crate::throughput::{ThroughputProfile, profile_identity};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

pub const CANARY_SCHEMA_VERSION: u32 = 1;
pub const CANARY_POLICY_ID: &str = "rwkv7-p150x2-persistent-decode-canary-v1";
const WINDOW_EIGHT_CANARY_POLICY_ID: &str = "rwkv7-p150x2-persistent-decode-canary-window8-v1";
pub const CANARY_RECEIPT_KIND: &str = "rwkv7_persistent_decode_canary";
pub const CANARY_INTERLOCK_KIND: &str = "rwkv7_persistent_decode_canary_interlock";
pub const PERSISTENT_PROFILE_ID: &str = "rwkv7-p150x2-persistent-decode-v2";
const WINDOW_EIGHT_PERSISTENT_PROFILE_ID: &str =
    "rwkv7-p150x2-persistent-decode-window8-candidate-v1";
pub const MONITORING_POLICY_ID: &str = "rwkv7-p150x2-persistent-decode-monitoring-v1";
const WINDOW_EIGHT_MONITORING_POLICY_ID: &str =
    "rwkv7-p150x2-persistent-decode-monitoring-window8-v1";
pub const HASH_ALGORITHM: &str = "blake3";
pub const CLEAN_ACTION: &str = "retain";
pub const WARNING_ACTION: &str = "hold";
pub const CRITICAL_ACTION: &str = "disable";
const CANARY_HASH_DOMAIN: &[u8] = b"rwkv7-p150x2-persistent-decode-canary-v1";
const BLAKE3_HEX_LENGTH: usize = 64;
const POLICY_MAXIMUM_BYTES: usize = 65_536;
const INPUT_MAXIMUM_BYTES: usize = 1_048_576;
const EXPECTED_MACHINE_COUNT: usize = 1;
const EXPECTED_PHYSICAL_DEVICE_IDS: [u32; 2] = [0, 1];
const EXPECTED_ADMITTED_WINDOWS: [u32; 2] = [2, 4];
const EXPECTED_WINDOW_EIGHT_ADMITTED_WINDOWS: [u32; 3] = [2, 4, 8];
const EXPECTED_SAMPLE_COUNT: usize = 4;
const EXPECTED_REQUESTS_PER_WORKER: usize = 4;
const EXPECTED_PROFILER_RUN_COUNT: usize = 2;
const EXPECTED_MAXIMUM_REQUEST_EXECUTIONS: usize = 104;
const EXPECTED_PLANNED_REQUEST_EXECUTIONS: usize = 72;
const EXPECTED_CANDIDATE_TELEMETRY_EVENTS: usize = 12;
const EXPECTED_MAXIMUM_DURATION_SECONDS: u64 = 600;
const EXPECTED_CANDIDATE_TRAFFIC_FRACTION: f64 = 0.5;
const EXPECTED_MINIMUM_PAIRED_SPEEDUP: f64 = 1.0;
const EXPECTED_PRIMARY_THREAD_CPU_AFFINITIES: [u32; 2] = [2, 6];
const EXPECTED_PRIMARY_THREAD_FIFO_PRIORITY: i32 = 1;
const ONE_CARD_WORKERS: usize = 1;
const TWO_CARD_WORKERS: usize = 2;
const COMPARISON_MODES: usize = 2;
const COMPARISON_TOPOLOGIES: usize = 2;
const CANDIDATE_WORKERS_PER_SAMPLE: usize = ONE_CARD_WORKERS + TWO_CARD_WORKERS;
const ALL_COMPARISON_WORKERS_PER_SAMPLE: usize = CANDIDATE_WORKERS_PER_SAMPLE * COMPARISON_MODES;
const PHYSICAL_AB_KIND: &str = "rwkv7_persistent_decode_window_physical_ab";
const PHYSICAL_AB_STATUS: &str = "stable_physical_gain";
const PHYSICAL_AB_BASELINE_MODE: &str = "normal_dispatch";
const PHYSICAL_AB_CANDIDATE_MODE: &str = "persistent_trace";
const ONE_CARD_PAIRING: &str = "concurrent_card_balanced";
const CARD_BALANCE_PERIOD: usize = 2;
const NON_CLAIMS: [&str; 5] = [
    "the canary does not route live service traffic",
    "the canary does not mutate the installed persistent profile",
    "the canary does not page an external provider",
    "the canary does not admit decode window eight",
    "the canary does not replace worker process supervision",
];

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CanaryPolicy {
    pub schema_version: u32,
    pub policy_id: String,
    pub persistent_profile_id: String,
    pub persistent_profile_blake3: String,
    pub monitoring_policy_id: String,
    pub admitted_windows: Vec<u32>,
    pub physical_device_ids: Vec<u32>,
    pub machine_count: usize,
    pub sample_count: usize,
    pub requests_per_worker: usize,
    pub profiler_run_count: usize,
    pub maximum_request_executions: usize,
    pub required_candidate_telemetry_events: usize,
    pub maximum_duration_seconds: u64,
    pub candidate_traffic_fraction: f64,
    pub minimum_paired_speedup: f64,
    pub primary_thread_cpu_affinities: Vec<u32>,
    pub primary_thread_fifo_priority: i32,
    pub clean_action: String,
    pub warning_action: String,
    pub critical_action: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CanaryError {
    pub code: &'static str,
    pub detail: String,
}

impl CanaryError {
    pub fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for CanaryError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for CanaryError {}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CanaryInterlockState {
    Armed,
    Retain,
    Hold,
    Disabled,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CanaryReason {
    CanaryRunning,
    CleanCanary,
    MonitoringWarning,
    MonitoringCritical,
    InvalidEvidence,
    BenchmarkFailed,
    CanaryTimeout,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CanaryLimits {
    pub maximum_request_executions: usize,
    pub maximum_duration_seconds: u64,
    pub candidate_traffic_fraction: f64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CanaryInterlock {
    pub schema_version: u32,
    pub event_kind: String,
    pub hash_algorithm: String,
    pub canary_binding_blake3: String,
    pub profile_id: String,
    pub profile_blake3: String,
    pub state: CanaryInterlockState,
    pub reason: CanaryReason,
    pub limits: CanaryLimits,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CanaryContext {
    pub policy: CanaryPolicy,
    pub policy_blake3: String,
    pub monitoring_policy_blake3: String,
    pub throughput_profile_blake3: String,
    pub workload_blake3: String,
    pub canary_binding_blake3: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpectedCanaryDispatch {
    Normal,
    Persistent,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OneCardDeviceAssignment {
    normal_dispatch_device_id: u32,
    persistent_trace_device_id: u32,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PhysicalAbReceipt {
    schema_version: u32,
    kind: String,
    status: String,
    physical_hardware: bool,
    production_physically_admitted: bool,
    batch_capacity: usize,
    repeat_count: usize,
    order_schedule: Vec<Vec<String>>,
    one_card_pairing: String,
    one_card_device_schedule: Vec<OneCardDeviceAssignment>,
    primary_thread_cpu_affinities: Vec<u32>,
    primary_thread_fifo_priority: i32,
    exact_generated_token_parity: bool,
    cpu_reference_profile_blake3: String,
    baseline_mode: String,
    candidate_mode: String,
    baseline_receipt_identity_blake3: String,
    candidate_receipt_identity_blake3: String,
    baseline_one_card_tokens_per_second: f64,
    candidate_one_card_tokens_per_second: f64,
    baseline_two_card_tokens_per_second: f64,
    candidate_two_card_tokens_per_second: f64,
    one_card_paired_speedups: Vec<f64>,
    two_card_paired_speedups: Vec<f64>,
    stable_gain: bool,
    profiler_attestation: String,
    machine_facts_before: String,
    machine_facts_after: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CanaryAbSummary {
    pub receipt_blake3: String,
    pub repeat_count: usize,
    pub exact_generated_token_parity: bool,
    pub minimum_one_card_paired_speedup: f64,
    pub minimum_two_card_paired_speedup: f64,
    pub stable_gain: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CanaryBudgetReceipt {
    pub maximum_request_executions: usize,
    pub planned_request_executions: usize,
    pub maximum_requests_per_worker: usize,
    pub maximum_duration_seconds: u64,
    pub observed_duration_seconds: f64,
    pub candidate_traffic_fraction: f64,
    pub planned_candidate_request_executions: usize,
    pub planned_control_request_executions: usize,
    pub planned_profiler_request_executions: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CanaryMonitoringSummary {
    pub batch_blake3: String,
    pub accepted_candidate_events: u64,
    pub rejected_candidate_events: u64,
    pub completed_candidate_requests: u64,
    pub parity_passed_events: u64,
    pub cleanup_complete_events: u64,
    pub alert_count: usize,
    pub verdict: MonitoringVerdict,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CanaryDecision {
    pub state: CanaryInterlockState,
    pub reason: CanaryReason,
    pub action: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CanaryEvidenceBindings {
    pub policy_blake3: String,
    pub monitoring_policy_blake3: String,
    pub throughput_profile_blake3: String,
    pub workload_blake3: String,
    pub canary_binding_blake3: String,
    pub physical_ab_receipt_blake3: Option<String>,
    pub monitoring_batch_blake3: Option<String>,
    pub final_interlock_blake3: String,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PersistentDecodeCanaryReceipt {
    pub schema_version: u32,
    pub event_kind: String,
    pub hash_algorithm: String,
    pub policy_id: String,
    pub profile_id: String,
    pub physical_device_ids: Vec<u32>,
    pub bindings: CanaryEvidenceBindings,
    pub budget: CanaryBudgetReceipt,
    pub monitoring: Option<CanaryMonitoringSummary>,
    pub physical_ab: Option<CanaryAbSummary>,
    pub decision: CanaryDecision,
    pub non_claims: Vec<String>,
}

fn valid_blake3(value: &str) -> bool {
    value.len() == BLAKE3_HEX_LENGTH
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_json(input: &str, label: &'static str) -> Result<Vec<u8>, CanaryError> {
    if input.len() > INPUT_MAXIMUM_BYTES {
        return Err(CanaryError::new(
            "canary_input_too_large",
            format!("{label} exceeds the byte limit"),
        ));
    }
    let value: JsonValue = serde_json::from_str(input)
        .map_err(|error| CanaryError::new("invalid_canary_input", error.to_string()))?;
    serde_json::to_vec(&value)
        .map_err(|error| CanaryError::new("invalid_canary_input", error.to_string()))
}

fn domain_hash(parts: &[&[u8]]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CANARY_HASH_DOMAIN);
    for part in parts {
        let length = u64::try_from(part.len()).expect("bounded canary hash part must fit u64");
        hasher.update(&length.to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().to_hex().to_string()
}

pub fn exact_file_blake3(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn checked_product(values: &[usize], label: &'static str) -> Result<usize, CanaryError> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or_else(|| CanaryError::new("canary_budget_overflow", label))
    })
}

fn valid_policy_identity(policy: &CanaryPolicy) -> bool {
    (policy.policy_id == CANARY_POLICY_ID
        && policy.persistent_profile_id == PERSISTENT_PROFILE_ID
        && policy.monitoring_policy_id == MONITORING_POLICY_ID
        && policy.admitted_windows == EXPECTED_ADMITTED_WINDOWS)
        || (policy.policy_id == WINDOW_EIGHT_CANARY_POLICY_ID
            && policy.persistent_profile_id == WINDOW_EIGHT_PERSISTENT_PROFILE_ID
            && policy.monitoring_policy_id == WINDOW_EIGHT_MONITORING_POLICY_ID
            && policy.admitted_windows == EXPECTED_WINDOW_EIGHT_ADMITTED_WINDOWS)
}

// r[impl tenstorrent.rwkv7_p150x2.persistent_decode_canary.policy]
pub fn parse_canary_policy(policy_json: &str) -> Result<CanaryPolicy, CanaryError> {
    if policy_json.len() > POLICY_MAXIMUM_BYTES {
        return Err(CanaryError::new(
            "canary_policy_too_large",
            "canary policy exceeds the byte limit",
        ));
    }
    let policy: CanaryPolicy = serde_json::from_str(policy_json)
        .map_err(|error| CanaryError::new("invalid_canary_policy", error.to_string()))?;
    let valid_numbers =
        policy.candidate_traffic_fraction.is_finite() && policy.minimum_paired_speedup.is_finite();
    let expected_candidate_events = checked_product(
        &[policy.sample_count, CANDIDATE_WORKERS_PER_SAMPLE],
        "candidate telemetry event count overflowed",
    )?;
    let comparison_requests = checked_product(
        &[
            policy.sample_count,
            ALL_COMPARISON_WORKERS_PER_SAMPLE,
            policy.requests_per_worker,
        ],
        "comparison request count overflowed",
    )?;
    let profiler_requests = checked_product(
        &[policy.profiler_run_count, policy.requests_per_worker],
        "profiler request count overflowed",
    )?;
    let maximum_request_capacity = comparison_requests
        .checked_add(profiler_requests)
        .ok_or_else(|| CanaryError::new("canary_budget_overflow", "request capacity overflowed"))?;
    let planned_comparison_requests = checked_product(
        &[
            policy.sample_count,
            COMPARISON_TOPOLOGIES,
            COMPARISON_MODES,
            policy.requests_per_worker,
        ],
        "planned comparison request count overflowed",
    )?;
    let planned_request_executions = planned_comparison_requests
        .checked_add(profiler_requests)
        .ok_or_else(|| {
            CanaryError::new("canary_budget_overflow", "planned request count overflowed")
        })?;
    let candidate_requests = checked_product(
        &[
            policy.sample_count,
            CANDIDATE_WORKERS_PER_SAMPLE,
            policy.requests_per_worker,
        ],
        "candidate request count overflowed",
    )?;
    let expected_fraction = candidate_requests as f64 / comparison_requests as f64;
    if policy.schema_version != CANARY_SCHEMA_VERSION
        || !valid_policy_identity(&policy)
        || !valid_blake3(&policy.persistent_profile_blake3)
        || policy.physical_device_ids != EXPECTED_PHYSICAL_DEVICE_IDS
        || policy.machine_count != EXPECTED_MACHINE_COUNT
        || policy.sample_count != EXPECTED_SAMPLE_COUNT
        || policy.requests_per_worker != EXPECTED_REQUESTS_PER_WORKER
        || policy.profiler_run_count != EXPECTED_PROFILER_RUN_COUNT
        || policy.maximum_request_executions != EXPECTED_MAXIMUM_REQUEST_EXECUTIONS
        || policy.required_candidate_telemetry_events != EXPECTED_CANDIDATE_TELEMETRY_EVENTS
        || policy.maximum_duration_seconds != EXPECTED_MAXIMUM_DURATION_SECONDS
        || !valid_numbers
        || policy.candidate_traffic_fraction != EXPECTED_CANDIDATE_TRAFFIC_FRACTION
        || expected_fraction != policy.candidate_traffic_fraction
        || policy.minimum_paired_speedup != EXPECTED_MINIMUM_PAIRED_SPEEDUP
        || policy.primary_thread_cpu_affinities != EXPECTED_PRIMARY_THREAD_CPU_AFFINITIES
        || policy.primary_thread_fifo_priority != EXPECTED_PRIMARY_THREAD_FIFO_PRIORITY
        || policy.clean_action != CLEAN_ACTION
        || policy.warning_action != WARNING_ACTION
        || policy.critical_action != CRITICAL_ACTION
        || expected_candidate_events != policy.required_candidate_telemetry_events
        || maximum_request_capacity != policy.maximum_request_executions
        || planned_request_executions != EXPECTED_PLANNED_REQUEST_EXECUTIONS
        || planned_request_executions > policy.maximum_request_executions
    {
        return Err(CanaryError::new(
            "invalid_canary_policy",
            "canary policy violates the bounded production contract",
        ));
    }
    Ok(policy)
}

pub fn build_canary_context(
    policy_json: &str,
    throughput_profile_json: &str,
    workload_json: &str,
    monitoring_policy_json: &str,
) -> Result<CanaryContext, CanaryError> {
    let policy = parse_canary_policy(policy_json)?;
    let monitoring_policy = parse_monitoring_policy(monitoring_policy_json)
        .map_err(|error| CanaryError::new("invalid_monitoring_policy", error.to_string()))?;
    if monitoring_policy.policy_id != policy.monitoring_policy_id {
        return Err(CanaryError::new(
            "canary_monitoring_policy_mismatch",
            "monitoring policy ID differs from the canary policy",
        ));
    }
    let throughput_profile: ThroughputProfile = serde_json::from_str(throughput_profile_json)
        .map_err(|error| CanaryError::new("invalid_throughput_profile", error.to_string()))?;
    profile_identity(&throughput_profile)
        .map_err(|error| CanaryError::new("invalid_throughput_profile", error.to_string()))?;
    let selected_window =
        usize::try_from(*policy.admitted_windows.last().expect("admitted windows"))
            .expect("admitted window must fit usize");
    if throughput_profile.repeat_count != policy.sample_count
        || throughput_profile.workload.request_count != policy.requests_per_worker
        || throughput_profile.scheduler.per_card_slot_count != policy.requests_per_worker
        || throughput_profile.workload.generated_token_count != selected_window
        || throughput_profile.timing.warmup_token_count != selected_window
    {
        return Err(CanaryError::new(
            "canary_throughput_profile_mismatch",
            "throughput profile differs from the canary budget",
        ));
    }
    let canonical_policy = canonical_json(policy_json, "canary policy")?;
    let canonical_monitoring = canonical_json(monitoring_policy_json, "monitoring policy")?;
    let canonical_throughput = canonical_json(throughput_profile_json, "throughput profile")?;
    let canonical_workload = canonical_json(workload_json, "canary workload")?;
    let policy_blake3 = domain_hash(&[b"policy", &canonical_policy]);
    let monitoring_policy_blake3 = domain_hash(&[b"monitoring", &canonical_monitoring]);
    let throughput_profile_blake3 = domain_hash(&[b"throughput", &canonical_throughput]);
    let workload_blake3 = domain_hash(&[b"workload", &canonical_workload]);
    let canary_binding_blake3 = domain_hash(&[
        b"binding",
        policy_blake3.as_bytes(),
        monitoring_policy_blake3.as_bytes(),
        throughput_profile_blake3.as_bytes(),
        workload_blake3.as_bytes(),
    ]);
    Ok(CanaryContext {
        policy,
        policy_blake3,
        monitoring_policy_blake3,
        throughput_profile_blake3,
        workload_blake3,
        canary_binding_blake3,
    })
}

pub fn validate_worker_interlock_receipts(
    context: &CanaryContext,
    expected_interlock_blake3: &str,
    expected_dispatch: ExpectedCanaryDispatch,
    receipt_json: &[String],
) -> Result<(), CanaryError> {
    let expected_count = match expected_dispatch {
        ExpectedCanaryDispatch::Normal | ExpectedCanaryDispatch::Persistent => {
            context.policy.required_candidate_telemetry_events
        }
    };
    if receipt_json.len() != expected_count || !valid_blake3(expected_interlock_blake3) {
        return Err(CanaryError::new(
            "invalid_canary_worker_count",
            "worker receipt count or interlock identity is invalid",
        ));
    }
    let expected_mode = match expected_dispatch {
        ExpectedCanaryDispatch::Normal => PHYSICAL_AB_BASELINE_MODE,
        ExpectedCanaryDispatch::Persistent => PHYSICAL_AB_CANDIDATE_MODE,
    };
    for receipt in receipt_json {
        let value: JsonValue = serde_json::from_str(receipt).map_err(|error| {
            CanaryError::new("invalid_canary_worker_receipt", error.to_string())
        })?;
        let persistent = value.get("persistent_decode").ok_or_else(|| {
            CanaryError::new(
                "invalid_canary_worker_receipt",
                "worker receipt omits persistent decode evidence",
            )
        })?;
        let valid = value.get("physical_hardware").and_then(JsonValue::as_bool) == Some(true)
            && persistent.get("simulated").and_then(JsonValue::as_bool) == Some(false)
            && persistent.get("profile_id").and_then(JsonValue::as_str)
                == Some(context.policy.persistent_profile_id.as_str())
            && persistent.get("profile_blake3").and_then(JsonValue::as_str)
                == Some(context.policy.persistent_profile_blake3.as_str())
            && persistent
                .get("canary_interlock_present")
                .and_then(JsonValue::as_bool)
                == Some(true)
            && persistent
                .get("canary_interlock_valid")
                .and_then(JsonValue::as_bool)
                == Some(true)
            && persistent
                .get("canary_interlock_state")
                .and_then(JsonValue::as_str)
                == Some("armed")
            && persistent
                .get("canary_interlock_allows_persistent")
                .and_then(JsonValue::as_bool)
                == Some(true)
            && persistent
                .get("canary_interlock_file_blake3")
                .and_then(JsonValue::as_str)
                == Some(expected_interlock_blake3)
            && persistent
                .get("canary_binding_blake3")
                .and_then(JsonValue::as_str)
                == Some(context.canary_binding_blake3.as_str())
            && persistent.get("selected_mode").and_then(JsonValue::as_str) == Some(expected_mode);
        if !valid {
            return Err(CanaryError::new(
                "invalid_canary_worker_interlock",
                "worker did not enforce the armed canary interlock",
            ));
        }
    }
    Ok(())
}

pub fn canary_interlock(
    context: &CanaryContext,
    state: CanaryInterlockState,
    reason: CanaryReason,
) -> Result<CanaryInterlock, CanaryError> {
    let valid_transition = matches!(
        (state, reason),
        (CanaryInterlockState::Armed, CanaryReason::CanaryRunning)
            | (CanaryInterlockState::Retain, CanaryReason::CleanCanary)
            | (CanaryInterlockState::Hold, CanaryReason::MonitoringWarning)
            | (
                CanaryInterlockState::Disabled,
                CanaryReason::MonitoringCritical
            )
            | (
                CanaryInterlockState::Disabled,
                CanaryReason::InvalidEvidence
            )
            | (
                CanaryInterlockState::Disabled,
                CanaryReason::BenchmarkFailed
            )
            | (CanaryInterlockState::Disabled, CanaryReason::CanaryTimeout)
    );
    if !valid_transition {
        return Err(CanaryError::new(
            "invalid_canary_transition",
            "canary state and reason do not form a valid transition",
        ));
    }
    Ok(CanaryInterlock {
        schema_version: CANARY_SCHEMA_VERSION,
        event_kind: CANARY_INTERLOCK_KIND.to_owned(),
        hash_algorithm: HASH_ALGORITHM.to_owned(),
        canary_binding_blake3: context.canary_binding_blake3.clone(),
        profile_id: context.policy.persistent_profile_id.clone(),
        profile_blake3: context.policy.persistent_profile_blake3.clone(),
        state,
        reason,
        limits: CanaryLimits {
            maximum_request_executions: context.policy.maximum_request_executions,
            maximum_duration_seconds: context.policy.maximum_duration_seconds,
            candidate_traffic_fraction: context.policy.candidate_traffic_fraction,
        },
    })
}

fn valid_one_card_device_schedule(
    schedule: &[OneCardDeviceAssignment],
    expected_samples: usize,
) -> bool {
    schedule.len() == expected_samples
        && schedule.iter().enumerate().all(|(index, assignment)| {
            let (normal, persistent) = if index.is_multiple_of(CARD_BALANCE_PERIOD) {
                (
                    EXPECTED_PHYSICAL_DEVICE_IDS[0],
                    EXPECTED_PHYSICAL_DEVICE_IDS[1],
                )
            } else {
                (
                    EXPECTED_PHYSICAL_DEVICE_IDS[1],
                    EXPECTED_PHYSICAL_DEVICE_IDS[0],
                )
            };
            assignment.normal_dispatch_device_id == normal
                && assignment.persistent_trace_device_id == persistent
        })
}

fn valid_order_schedule(schedule: &[Vec<String>], expected_samples: usize) -> bool {
    if schedule.len() != expected_samples {
        return false;
    }
    schedule.iter().enumerate().all(|(index, pair)| {
        let expected = if index.is_multiple_of(COMPARISON_MODES) {
            [PHYSICAL_AB_BASELINE_MODE, PHYSICAL_AB_CANDIDATE_MODE]
        } else {
            [PHYSICAL_AB_CANDIDATE_MODE, PHYSICAL_AB_BASELINE_MODE]
        };
        pair.len() == COMPARISON_MODES && pair[0] == expected[0] && pair[1] == expected[1]
    })
}

// r[impl tenstorrent.rwkv7_p150x2.persistent_decode_canary.evaluate]
pub fn validate_physical_ab_receipt(
    context: &CanaryContext,
    receipt_json: &str,
) -> Result<CanaryAbSummary, CanaryError> {
    let receipt: PhysicalAbReceipt = serde_json::from_str(receipt_json)
        .map_err(|error| CanaryError::new("invalid_canary_ab_receipt", error.to_string()))?;
    let all_rates = [
        receipt.baseline_one_card_tokens_per_second,
        receipt.candidate_one_card_tokens_per_second,
        receipt.baseline_two_card_tokens_per_second,
        receipt.candidate_two_card_tokens_per_second,
    ];
    let speedups_valid = receipt.one_card_paired_speedups.len() == context.policy.sample_count
        && receipt.two_card_paired_speedups.len() == context.policy.sample_count
        && receipt
            .one_card_paired_speedups
            .iter()
            .chain(&receipt.two_card_paired_speedups)
            .all(|speedup| speedup.is_finite() && *speedup > context.policy.minimum_paired_speedup);
    if receipt.schema_version != CANARY_SCHEMA_VERSION
        || receipt.kind != PHYSICAL_AB_KIND
        || receipt.status != PHYSICAL_AB_STATUS
        || !receipt.physical_hardware
        || receipt.production_physically_admitted
        || receipt.batch_capacity != context.policy.requests_per_worker
        || receipt.repeat_count != context.policy.sample_count
        || !valid_order_schedule(&receipt.order_schedule, context.policy.sample_count)
        || receipt.one_card_pairing != ONE_CARD_PAIRING
        || !valid_one_card_device_schedule(
            &receipt.one_card_device_schedule,
            context.policy.sample_count,
        )
        || receipt.primary_thread_cpu_affinities != context.policy.primary_thread_cpu_affinities
        || receipt.primary_thread_fifo_priority != context.policy.primary_thread_fifo_priority
        || !receipt.exact_generated_token_parity
        || !valid_blake3(&receipt.cpu_reference_profile_blake3)
        || receipt.baseline_mode != PHYSICAL_AB_BASELINE_MODE
        || receipt.candidate_mode != PHYSICAL_AB_CANDIDATE_MODE
        || !valid_blake3(&receipt.baseline_receipt_identity_blake3)
        || !valid_blake3(&receipt.candidate_receipt_identity_blake3)
        || all_rates
            .iter()
            .any(|rate| !rate.is_finite() || *rate <= 0.0)
        || !speedups_valid
        || !receipt.stable_gain
        || receipt.profiler_attestation != "profiler-attestation.json"
        || receipt.machine_facts_before != "machine-before.json"
        || receipt.machine_facts_after != "machine-after.json"
    {
        return Err(CanaryError::new(
            "invalid_canary_ab_receipt",
            "physical A/B receipt violates the canary contract",
        ));
    }
    let canonical = canonical_json(receipt_json, "physical A/B receipt")?;
    Ok(CanaryAbSummary {
        receipt_blake3: domain_hash(&[b"physical-ab", &canonical]),
        repeat_count: receipt.repeat_count,
        exact_generated_token_parity: receipt.exact_generated_token_parity,
        minimum_one_card_paired_speedup: receipt
            .one_card_paired_speedups
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        minimum_two_card_paired_speedup: receipt
            .two_card_paired_speedups
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        stable_gain: receipt.stable_gain,
    })
}

fn planned_candidate_request_executions(context: &CanaryContext) -> usize {
    context
        .policy
        .sample_count
        .checked_mul(COMPARISON_TOPOLOGIES)
        .and_then(|count| count.checked_mul(context.policy.requests_per_worker))
        .expect("validated canary request plan must fit usize")
}

fn budget_receipt(context: &CanaryContext, duration_seconds: f64) -> CanaryBudgetReceipt {
    let planned_candidate_request_executions = planned_candidate_request_executions(context);
    let planned_control_request_executions = planned_candidate_request_executions;
    let planned_profiler_request_executions = context
        .policy
        .profiler_run_count
        .checked_mul(context.policy.requests_per_worker)
        .expect("validated profiler request plan must fit usize");
    let planned_request_executions = planned_candidate_request_executions
        .checked_add(planned_control_request_executions)
        .and_then(|count| count.checked_add(planned_profiler_request_executions))
        .expect("validated canary request plan must fit usize");
    assert!(planned_request_executions <= context.policy.maximum_request_executions);
    CanaryBudgetReceipt {
        maximum_request_executions: context.policy.maximum_request_executions,
        planned_request_executions,
        maximum_requests_per_worker: context.policy.requests_per_worker,
        maximum_duration_seconds: context.policy.maximum_duration_seconds,
        observed_duration_seconds: duration_seconds,
        candidate_traffic_fraction: context.policy.candidate_traffic_fraction,
        planned_candidate_request_executions,
        planned_control_request_executions,
        planned_profiler_request_executions,
    }
}

fn monitoring_summary(receipt: &PersistentDecodeMonitoringReceipt) -> CanaryMonitoringSummary {
    CanaryMonitoringSummary {
        batch_blake3: receipt.source.batch_blake3.clone(),
        accepted_candidate_events: receipt.source.accepted_event_count,
        rejected_candidate_events: receipt.source.rejected_event_count,
        completed_candidate_requests: receipt.metrics.terminal.completed,
        parity_passed_events: receipt.metrics.parity.passed,
        cleanup_complete_events: receipt.metrics.cleanup.complete,
        alert_count: receipt.alerts.len(),
        verdict: receipt.verdict,
    }
}

fn completed_evidence_is_clean(
    context: &CanaryContext,
    duration_seconds: f64,
    monitoring: &PersistentDecodeMonitoringReceipt,
    physical_ab: &CanaryAbSummary,
) -> bool {
    let required_events = u64::try_from(context.policy.required_candidate_telemetry_events)
        .expect("candidate event count must fit u64");
    let required_requests = u64::try_from(planned_candidate_request_executions(context))
        .expect("candidate request count must fit u64");
    duration_seconds.is_finite()
        && duration_seconds > 0.0
        && duration_seconds <= context.policy.maximum_duration_seconds as f64
        && monitoring.schema_version == MONITORING_SCHEMA_VERSION
        && monitoring.event_kind == MONITORING_EVENT_KIND
        && monitoring.hash_algorithm == HASH_ALGORITHM
        && monitoring.policy_id == context.policy.monitoring_policy_id
        && valid_blake3(&monitoring.policy_blake3)
        && valid_blake3(&monitoring.source.batch_blake3)
        && monitoring.source.event_count == required_events
        && monitoring.source.event_blake3.len()
            == context.policy.required_candidate_telemetry_events
        && monitoring
            .source
            .event_blake3
            .iter()
            .all(|hash| valid_blake3(hash))
        && monitoring.verdict == MonitoringVerdict::Clean
        && monitoring.persistent_admission_recommendation
            == PersistentAdmissionRecommendation::Retain
        && monitoring.source.accepted_event_count == required_events
        && monitoring.source.rejected_event_count == 0
        && monitoring.metrics.dispatch.persistent_trace == required_events
        && monitoring.metrics.dispatch.normal_dispatch == 0
        && monitoring.metrics.fallback.count == 0
        && monitoring.metrics.terminal.completed == required_requests
        && monitoring.metrics.terminal.cancelled == 0
        && monitoring.metrics.terminal.timed_out == 0
        && monitoring.metrics.terminal.failed == 0
        && monitoring.metrics.parity.passed == required_events
        && monitoring.metrics.parity.not_applicable == 0
        && monitoring.metrics.cleanup.complete == required_events
        && monitoring.metrics.cleanup.not_applicable == 0
        && monitoring.alerts.is_empty()
        && physical_ab.exact_generated_token_parity
        && physical_ab.stable_gain
}

fn receipt_with_decision(
    context: &CanaryContext,
    duration_seconds: f64,
    monitoring: Option<&PersistentDecodeMonitoringReceipt>,
    physical_ab: Option<&CanaryAbSummary>,
    state: CanaryInterlockState,
    reason: CanaryReason,
) -> Result<(CanaryInterlock, PersistentDecodeCanaryReceipt), CanaryError> {
    let interlock = canary_interlock(context, state, reason)?;
    let interlock_bytes = serde_json::to_vec(&interlock)
        .map_err(|error| CanaryError::new("canary_serialization_failed", error.to_string()))?;
    let interlock_blake3 = exact_file_blake3(&interlock_bytes);
    let action = match state {
        CanaryInterlockState::Armed => {
            return Err(CanaryError::new(
                "invalid_canary_transition",
                "final canary receipt cannot remain armed",
            ));
        }
        CanaryInterlockState::Retain => context.policy.clean_action.clone(),
        CanaryInterlockState::Hold => context.policy.warning_action.clone(),
        CanaryInterlockState::Disabled => context.policy.critical_action.clone(),
    };
    let monitoring_summary = monitoring.map(monitoring_summary);
    let receipt = PersistentDecodeCanaryReceipt {
        schema_version: CANARY_SCHEMA_VERSION,
        event_kind: CANARY_RECEIPT_KIND.to_owned(),
        hash_algorithm: HASH_ALGORITHM.to_owned(),
        policy_id: context.policy.policy_id.clone(),
        profile_id: context.policy.persistent_profile_id.clone(),
        physical_device_ids: context.policy.physical_device_ids.clone(),
        bindings: CanaryEvidenceBindings {
            policy_blake3: context.policy_blake3.clone(),
            monitoring_policy_blake3: context.monitoring_policy_blake3.clone(),
            throughput_profile_blake3: context.throughput_profile_blake3.clone(),
            workload_blake3: context.workload_blake3.clone(),
            canary_binding_blake3: context.canary_binding_blake3.clone(),
            physical_ab_receipt_blake3: physical_ab.map(|summary| summary.receipt_blake3.clone()),
            monitoring_batch_blake3: monitoring.map(|receipt| receipt.source.batch_blake3.clone()),
            final_interlock_blake3: interlock_blake3,
        },
        budget: budget_receipt(context, duration_seconds),
        monitoring: monitoring_summary,
        physical_ab: physical_ab.cloned(),
        decision: CanaryDecision {
            state,
            reason,
            action,
        },
        non_claims: NON_CLAIMS.iter().map(|claim| (*claim).to_owned()).collect(),
    };
    Ok((interlock, receipt))
}

pub fn evaluate_completed_canary(
    context: &CanaryContext,
    duration_seconds: f64,
    monitoring: &PersistentDecodeMonitoringReceipt,
    physical_ab: &CanaryAbSummary,
) -> Result<(CanaryInterlock, PersistentDecodeCanaryReceipt), CanaryError> {
    let (state, reason) =
        if completed_evidence_is_clean(context, duration_seconds, monitoring, physical_ab) {
            (CanaryInterlockState::Retain, CanaryReason::CleanCanary)
        } else if monitoring.verdict == MonitoringVerdict::Warning {
            (CanaryInterlockState::Hold, CanaryReason::MonitoringWarning)
        } else {
            (
                CanaryInterlockState::Disabled,
                CanaryReason::MonitoringCritical,
            )
        };
    receipt_with_decision(
        context,
        duration_seconds,
        Some(monitoring),
        Some(physical_ab),
        state,
        reason,
    )
}

pub fn evaluate_failed_canary(
    context: &CanaryContext,
    duration_seconds: f64,
    reason: CanaryReason,
) -> Result<(CanaryInterlock, PersistentDecodeCanaryReceipt), CanaryError> {
    if !matches!(
        reason,
        CanaryReason::InvalidEvidence | CanaryReason::BenchmarkFailed | CanaryReason::CanaryTimeout
    ) {
        return Err(CanaryError::new(
            "invalid_canary_transition",
            "failed canary requires an invalid, benchmark, or timeout reason",
        ));
    }
    receipt_with_decision(
        context,
        duration_seconds,
        None,
        None,
        CanaryInterlockState::Disabled,
        reason,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistent_decode_monitoring::{
        AlertSeverity, CleanupMetrics, DispatchMetrics, EligibilityMetrics, FallbackMetrics,
        MonitoringAlert, MonitoringMetrics, MonitoringSourceBinding, MonitoringVerdict,
        ParityMetrics, PersistentAdmissionRecommendation, TerminalMetrics, TimingMetrics,
    };
    use std::collections::BTreeMap;

    const HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const TEST_SPEEDUP: f64 = 2.0;
    const POLICY: &str = r#"{
      "schema_version": 1,
      "policy_id": "rwkv7-p150x2-persistent-decode-canary-v1",
      "persistent_profile_id": "rwkv7-p150x2-persistent-decode-v2",
      "persistent_profile_blake3": "e41319204ddf36bda088afb8a4d55a367dc163951ea29c45738e3b11b815819a",
      "monitoring_policy_id": "rwkv7-p150x2-persistent-decode-monitoring-v1",
      "admitted_windows": [2, 4],
      "physical_device_ids": [0, 1],
      "machine_count": 1,
      "sample_count": 4,
      "requests_per_worker": 4,
      "profiler_run_count": 2,
      "maximum_request_executions": 104,
      "required_candidate_telemetry_events": 12,
      "maximum_duration_seconds": 600,
      "candidate_traffic_fraction": 0.5,
      "minimum_paired_speedup": 1.0,
      "primary_thread_cpu_affinities": [2, 6],
      "primary_thread_fifo_priority": 1,
      "clean_action": "retain",
      "warning_action": "hold",
      "critical_action": "disable"
    }"#;
    const MONITORING_POLICY: &str = r#"{
      "schema_version": 1,
      "policy_id": "rwkv7-p150x2-persistent-decode-monitoring-v1",
      "maximum_batch_events": 256,
      "maximum_event_bytes": 1048576,
      "maximum_timing_seconds": 60.0,
      "minimum_fallback_samples": 8,
      "maximum_fallback_rate": 0.25,
      "admitted_windows": [2, 4],
      "critical_action": "disable_persistent_admission",
      "warning_action": "investigate_fallback_rate"
    }"#;
    const THROUGHPUT_PROFILE: &str = r#"{
      "schema_version": 1,
      "profile_id": "rwkv7-p150x2-throughput-v1",
      "model_profile_blake3": "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88",
      "repeat_count": 4,
      "workload": {"request_count": 4, "short_prompt_token_count": 8, "long_prompt_token_count": 8, "generated_token_count": 4, "homogeneous_prompts": true},
      "scheduler": {"mode": "deterministic_continuous_batching", "tie_break": "least_loaded_card_then_lowest_card_then_lowest_slot", "per_card_slot_count": 4},
      "runtime": {"replica_mode": "independent_card_replicas", "recurrent_state_residency": "device_fp32"},
      "timing": {"timed_region": "decode_only", "warmup_token_count": 4, "exclude_state_readback": true, "exclude_profiler_flush": true, "exclude_control_replay": true, "exclude_artifact_writes": true},
      "admission": {"minimum_dual_card_speedup": 1.5, "maximum_p99_latency_regression_fraction": 0.1, "require_physical_hardware": true, "require_profiler_attribution": true, "machine_fact_source": "tt-smi -s --snapshot_no_tty", "one_card_physical_device_id": 0, "two_card_physical_device_ids": [0, 1]}
    }"#;
    const WORKLOAD: &str = r#"{"schema_version":1,"fixture":"bounded"}"#;

    fn context() -> CanaryContext {
        build_canary_context(POLICY, THROUGHPUT_PROFILE, WORKLOAD, MONITORING_POLICY)
            .expect("canary context")
    }

    fn clean_monitoring() -> PersistentDecodeMonitoringReceipt {
        let required_events = u64::try_from(EXPECTED_CANDIDATE_TELEMETRY_EVENTS).unwrap();
        let completed = u64::try_from(
            EXPECTED_SAMPLE_COUNT * COMPARISON_TOPOLOGIES * EXPECTED_REQUESTS_PER_WORKER,
        )
        .unwrap();
        PersistentDecodeMonitoringReceipt {
            schema_version: 1,
            event_kind: "rwkv7_persistent_decode_monitoring".to_owned(),
            hash_algorithm: HASH_ALGORITHM.to_owned(),
            policy_id: MONITORING_POLICY_ID.to_owned(),
            policy_blake3: HASH.to_owned(),
            source: MonitoringSourceBinding {
                event_count: required_events,
                accepted_event_count: required_events,
                rejected_event_count: 0,
                event_blake3: vec![HASH.to_owned(); EXPECTED_CANDIDATE_TELEMETRY_EVENTS],
                batch_blake3: HASH.to_owned(),
            },
            metrics: MonitoringMetrics {
                eligibility: EligibilityMetrics {
                    window_admitted: required_events,
                    workload_eligible: required_events,
                    bindings_validated: required_events,
                    profile_physically_admitted: required_events,
                    candidate_requested: 0,
                },
                dispatch: DispatchMetrics {
                    persistent_trace: required_events,
                    normal_dispatch: 0,
                    selected_windows: BTreeMap::from([
                        ("2".to_owned(), 0),
                        ("4".to_owned(), required_events),
                        ("none".to_owned(), 0),
                    ]),
                },
                fallback: FallbackMetrics {
                    count: 0,
                    rate: 0.0,
                    reasons: BTreeMap::new(),
                },
                terminal: TerminalMetrics {
                    completed,
                    cancelled: 0,
                    timed_out: 0,
                    failed: 0,
                },
                parity: ParityMetrics {
                    passed: required_events,
                    not_applicable: 0,
                },
                cleanup: CleanupMetrics {
                    complete: required_events,
                    not_applicable: 0,
                },
                timing: TimingMetrics::default(),
            },
            alerts: Vec::new(),
            verdict: MonitoringVerdict::Clean,
            persistent_admission_recommendation: PersistentAdmissionRecommendation::Retain,
            non_claims: Vec::new(),
        }
    }

    fn physical_ab() -> CanaryAbSummary {
        CanaryAbSummary {
            receipt_blake3: HASH.to_owned(),
            repeat_count: EXPECTED_SAMPLE_COUNT,
            exact_generated_token_parity: true,
            minimum_one_card_paired_speedup: TEST_SPEEDUP,
            minimum_two_card_paired_speedup: TEST_SPEEDUP,
            stable_gain: true,
        }
    }

    fn physical_ab_json() -> String {
        const BASELINE_RATE: f64 = 10.0;
        const CANDIDATE_RATE: f64 = 20.0;
        let order_schedule = (0..EXPECTED_SAMPLE_COUNT)
            .map(|index| {
                if index.is_multiple_of(CARD_BALANCE_PERIOD) {
                    vec![PHYSICAL_AB_BASELINE_MODE, PHYSICAL_AB_CANDIDATE_MODE]
                } else {
                    vec![PHYSICAL_AB_CANDIDATE_MODE, PHYSICAL_AB_BASELINE_MODE]
                }
            })
            .collect::<Vec<_>>();
        let one_card_device_schedule = (0..EXPECTED_SAMPLE_COUNT)
            .map(|index| {
                let (normal, persistent) = if index.is_multiple_of(CARD_BALANCE_PERIOD) {
                    (
                        EXPECTED_PHYSICAL_DEVICE_IDS[0],
                        EXPECTED_PHYSICAL_DEVICE_IDS[1],
                    )
                } else {
                    (
                        EXPECTED_PHYSICAL_DEVICE_IDS[1],
                        EXPECTED_PHYSICAL_DEVICE_IDS[0],
                    )
                };
                serde_json::json!({
                    "normal_dispatch_device_id": normal,
                    "persistent_trace_device_id": persistent,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "schema_version": CANARY_SCHEMA_VERSION,
            "kind": PHYSICAL_AB_KIND,
            "status": PHYSICAL_AB_STATUS,
            "physical_hardware": true,
            "production_physically_admitted": false,
            "batch_capacity": EXPECTED_REQUESTS_PER_WORKER,
            "repeat_count": EXPECTED_SAMPLE_COUNT,
            "order_schedule": order_schedule,
            "one_card_pairing": ONE_CARD_PAIRING,
            "one_card_device_schedule": one_card_device_schedule,
            "primary_thread_cpu_affinities": EXPECTED_PRIMARY_THREAD_CPU_AFFINITIES,
            "primary_thread_fifo_priority": EXPECTED_PRIMARY_THREAD_FIFO_PRIORITY,
            "exact_generated_token_parity": true,
            "cpu_reference_profile_blake3": HASH,
            "baseline_mode": PHYSICAL_AB_BASELINE_MODE,
            "candidate_mode": PHYSICAL_AB_CANDIDATE_MODE,
            "baseline_receipt_identity_blake3": HASH,
            "candidate_receipt_identity_blake3": HASH,
            "baseline_one_card_tokens_per_second": BASELINE_RATE,
            "candidate_one_card_tokens_per_second": CANDIDATE_RATE,
            "baseline_two_card_tokens_per_second": BASELINE_RATE,
            "candidate_two_card_tokens_per_second": CANDIDATE_RATE,
            "one_card_paired_speedups": vec![TEST_SPEEDUP; EXPECTED_SAMPLE_COUNT],
            "two_card_paired_speedups": vec![TEST_SPEEDUP; EXPECTED_SAMPLE_COUNT],
            "stable_gain": true,
            "profiler_attestation": "profiler-attestation.json",
            "machine_facts_before": "machine-before.json",
            "machine_facts_after": "machine-after.json",
        })
        .to_string()
    }

    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_canary.policy]
    #[test]
    fn validates_exact_cross_field_budget() {
        let parsed = parse_canary_policy(POLICY).expect("policy");
        assert_eq!(
            parsed.maximum_request_executions,
            EXPECTED_MAXIMUM_REQUEST_EXECUTIONS
        );
        let mut invalid: JsonValue = serde_json::from_str(POLICY).unwrap();
        invalid["maximum_request_executions"] =
            serde_json::json!(EXPECTED_MAXIMUM_REQUEST_EXECUTIONS - 1);
        assert!(parse_canary_policy(&invalid.to_string()).is_err());
        invalid = serde_json::from_str(POLICY).unwrap();
        invalid["unknown"] = serde_json::json!(true);
        assert!(parse_canary_policy(&invalid.to_string()).is_err());
    }

    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_canary.interlock]
    #[test]
    fn creates_deterministic_armed_interlock() {
        let context = context();
        let first = canary_interlock(
            &context,
            CanaryInterlockState::Armed,
            CanaryReason::CanaryRunning,
        )
        .unwrap();
        let second = canary_interlock(
            &context,
            CanaryInterlockState::Armed,
            CanaryReason::CanaryRunning,
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(
            first.limits.maximum_request_executions,
            EXPECTED_MAXIMUM_REQUEST_EXECUTIONS
        );
        assert!(
            canary_interlock(
                &context,
                CanaryInterlockState::Retain,
                CanaryReason::MonitoringCritical,
            )
            .is_err()
        );
    }

    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_canary.evaluate]
    #[test]
    fn validates_balanced_physical_ab_receipt() {
        let context = context();
        let valid = physical_ab_json();
        let summary =
            validate_physical_ab_receipt(&context, &valid).expect("balanced physical A/B receipt");
        assert_eq!(summary.repeat_count, EXPECTED_SAMPLE_COUNT);
        assert_eq!(summary.minimum_one_card_paired_speedup, TEST_SPEEDUP);

        let mut invalid: JsonValue = serde_json::from_str(&valid).unwrap();
        invalid["one_card_device_schedule"][0]["persistent_trace_device_id"] =
            serde_json::json!(EXPECTED_PHYSICAL_DEVICE_IDS[0]);
        assert!(validate_physical_ab_receipt(&context, &invalid.to_string()).is_err());
    }

    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_canary.evaluate]
    #[test]
    fn clean_evidence_retains_interlock() {
        let context = context();
        let monitoring = clean_monitoring();
        let (interlock, receipt) =
            evaluate_completed_canary(&context, 100.0, &monitoring, &physical_ab()).unwrap();
        assert_eq!(interlock.state, CanaryInterlockState::Retain);
        assert_eq!(receipt.decision.reason, CanaryReason::CleanCanary);
        assert_eq!(
            receipt.budget.maximum_request_executions,
            EXPECTED_MAXIMUM_REQUEST_EXECUTIONS
        );
        assert_eq!(
            receipt.budget.planned_request_executions,
            EXPECTED_PLANNED_REQUEST_EXECUTIONS
        );
    }

    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_canary.invalid]
    #[test]
    fn warning_holds_and_critical_disables() {
        let context = context();
        let mut warning = clean_monitoring();
        warning.verdict = MonitoringVerdict::Warning;
        warning.alerts.push(MonitoringAlert {
            code: "fallback_rate_high".to_owned(),
            severity: AlertSeverity::Warning,
            count: 1,
            action: "investigate_fallback_rate".to_owned(),
        });
        let (interlock, _) =
            evaluate_completed_canary(&context, 100.0, &warning, &physical_ab()).unwrap();
        assert_eq!(interlock.state, CanaryInterlockState::Hold);

        let mut critical = clean_monitoring();
        critical.verdict = MonitoringVerdict::Critical;
        critical.persistent_admission_recommendation = PersistentAdmissionRecommendation::Disable;
        critical.alerts.push(MonitoringAlert {
            code: "parity_failure".to_owned(),
            severity: AlertSeverity::Critical,
            count: 1,
            action: "disable_persistent_admission".to_owned(),
        });
        let (interlock, _) =
            evaluate_completed_canary(&context, 100.0, &critical, &physical_ab()).unwrap();
        assert_eq!(interlock.state, CanaryInterlockState::Disabled);

        let (interlock, receipt) =
            evaluate_failed_canary(&context, 600.0, CanaryReason::CanaryTimeout).unwrap();
        assert_eq!(interlock.state, CanaryInterlockState::Disabled);
        assert_eq!(receipt.decision.reason, CanaryReason::CanaryTimeout);
    }
}
