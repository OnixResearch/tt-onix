//! Static guard for bounded, physically admitted persistent decode.
//!
//! The compiled C++ self-test proves lifecycle behavior. This guard ensures
//! that source drift cannot remove admission, typed limits, BLAKE3 bindings,
//! cleanup handshakes, or normal-dispatch fallback without a Rust test failure.

use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentDecodeContractError {
    pub surface: &'static str,
    pub pattern: &'static str,
}

impl std::fmt::Display for PersistentDecodeContractError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "persistent decode {} omitted required pattern: {}",
            self.surface, self.pattern
        )
    }
}

impl std::error::Error for PersistentDecodeContractError {}

const HEADER_PATTERNS: [&str; 9] = [
    "enum class BindingKind",
    "CancellationPending",
    "TimeoutPending",
    "FailurePending",
    "StateZeroed",
    "maximum_elapsed_milliseconds",
    "cancellation_poll_interval_steps",
    "PersistentTelemetryInput",
    "validate_persistent_telemetry",
];
const CORE_PATTERNS: [&str; 12] = [
    "rwkv7-persistent-decode-binding-v1",
    "derive_binding_identity",
    "validate_bindings",
    "validate_active_slots",
    "output_overflow",
    "slot_releasable",
    "select_execution_mode",
    "select_window_limits",
    "ProfileNotAdmitted",
    "WindowNotAdmitted",
    "summarize_persistent_telemetry",
    "persistent telemetry durations must be finite and non-negative",
];
const SELF_TEST_PATTERNS: [&str; 11] = [
    "test_normal_completion",
    "test_maximum_length_completion",
    "test_cancel_timeout_and_failure",
    "test_invalid_transitions",
    "test_output_overflow",
    "test_slot_admission_and_recycling",
    "test_binding_drift_and_receipts",
    "test_fail_closed_mode_selection",
    "test_window_selection",
    "test_persistent_telemetry_summary",
    "test_invalid_persistent_telemetry",
];
const RUNTIME_PATTERNS: [&str; 26] = [
    "derive_fast_worker_bindings",
    "persistent decode profile targets a different model profile",
    "begin_trace_capture",
    "end_trace_capture",
    "execute_trace",
    "release_trace",
    "submit_persistent_decode_step",
    "if (step_index == kFirstIndex)",
    "op_reshape(context, selected_keepdim, {batch_size, 1U})",
    "retained_intermediates",
    "zero_persistent_model_state",
    "\"binding_identities\"",
    "\"selected_mode\"",
    "\"persistent_region_entered\"",
    "\"trace_capture_outside_timed_region\"",
    "\"prompt_prefill_outside_timed_region\"",
    "\"zero_per_token_host_submissions_proven\"",
    "\"selected_token_device_resident_inside_region\"",
    "\"selected_token_readback_inside_region\"",
    "persistent_workload_limits",
    "\"admitted_decode_windows\"",
    "\"selected_decode_window_steps\"",
    "\"persistent_decode_telemetry\"",
    "\"trace_setup_seconds\"",
    "\"trace_replay_seconds\"",
    "\"cleanup_seconds\"",
];
const PROFILE_PATTERNS: [&str; 7] = [
    "bounded_metal_trace_chunk",
    "physically_admitted =",
    "maximum_decode_steps",
    "maximum_elapsed_milliseconds",
    "cancellation_poll_interval_steps",
    "output_capacity_tokens",
    "admitted_decode_windows",
];

fn require_patterns(
    surface: &'static str,
    source: &str,
    patterns: &'static [&'static str],
) -> Result<(), PersistentDecodeContractError> {
    for pattern in patterns {
        if !source.contains(pattern) {
            return Err(PersistentDecodeContractError { surface, pattern });
        }
    }
    Ok(())
}

pub fn validate_persistent_decode_sources(
    header: &str,
    core: &str,
    self_test: &str,
    runtime: &str,
    profile: &str,
) -> Result<(), PersistentDecodeContractError> {
    require_patterns("header", header, &HEADER_PATTERNS)?;
    require_patterns("core", core, &CORE_PATTERNS)?;
    require_patterns("self-test", self_test, &SELF_TEST_PATTERNS)?;
    require_patterns("runtime", runtime, &RUNTIME_PATTERNS)?;
    require_patterns("profile", profile, &PROFILE_PATTERNS)?;
    Ok(())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PersistentReceiptPolicy {
    Candidate,
    Production,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PersistentReceiptError {
    pub code: &'static str,
    pub detail: String,
}

impl PersistentReceiptError {
    fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for PersistentReceiptError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for PersistentReceiptError {}

#[derive(Debug, Deserialize)]
struct WorkerReceipt {
    physical_hardware: bool,
    profiler_attributed: bool,
    #[serde(default)]
    profiler_trace_only: bool,
    persistent_decode: PersistentSection,
    #[serde(default)]
    persistent_decode_telemetry: Option<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct PersistentSection {
    profile_id: String,
    profile_blake3: String,
    #[serde(default)]
    admitted_decode_windows: Vec<u32>,
    #[serde(default)]
    selected_decode_window_steps: Option<u32>,
    model_profile_blake3: String,
    hash_algorithm: String,
    simulated: bool,
    physically_admitted: bool,
    candidate_requested: bool,
    #[serde(default)]
    terminal_probe: String,
    #[serde(default)]
    terminal_event_poll_step: Option<u32>,
    #[serde(default)]
    cancellation_poll_interval_steps: u32,
    #[serde(default)]
    pre_replay_terminal_event: bool,
    #[serde(default)]
    window_admitted: bool,
    workload_eligible: bool,
    prompt_prefill_outside_timed_region: bool,
    binding_identities: BTreeMap<String, String>,
    bindings_validated: bool,
    selected_mode: String,
    fallback_reason: String,
    persistent_region_entered: bool,
    trace_region_bytes: usize,
    trace_capture_outside_timed_region: bool,
    trace_execution_seconds: f64,
    timed_host_model_submissions: u64,
    zero_per_token_host_submissions_proven: bool,
    selected_token_device_resident_inside_region: bool,
    selected_token_readback_inside_region: bool,
    output_readback_after_region: bool,
    recurrent_state_dtype: String,
    final_state_blake3: Option<String>,
    reference_state_blake3: Option<String>,
    exact_generated_token_parity: bool,
    exact_fp32_recurrent_state_parity: bool,
    reference_state_zeroed: bool,
    state_zeroed_before_slot_release: bool,
    lifecycle_receipts: Vec<LifecycleReceipt>,
}

#[derive(Debug, Deserialize)]
struct LifecycleReceipt {
    slot: u32,
    phase: String,
    #[serde(default)]
    terminal_reason: String,
    #[serde(default)]
    completed_steps: u32,
    #[serde(default)]
    failure_code: String,
    slot_releasable: bool,
    receipt_blake3: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistentTelemetry {
    schema_version: u32,
    event_kind: String,
    eligibility: TelemetryEligibility,
    dispatch: TelemetryDispatch,
    terminal_outcomes: TelemetryTerminalOutcomes,
    timing: TelemetryTiming,
    parity: TelemetryParity,
    cleanup_status: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TelemetryEligibility {
    window_admitted: bool,
    workload_eligible: bool,
    bindings_validated: bool,
    profile_physically_admitted: bool,
    candidate_requested: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TelemetryDispatch {
    selected_mode: String,
    selected_window_steps: Option<u32>,
    fallback_reason: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TelemetryTerminalOutcomes {
    completed: u32,
    cancelled: u32,
    timed_out: u32,
    failed: u32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TelemetryTiming {
    clock: String,
    unit: String,
    trace_setup_seconds: f64,
    trace_replay_seconds: f64,
    cleanup_seconds: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TelemetryParity {
    generated_tokens: String,
    fp32_recurrent_state: String,
}

const BLAKE3_HEX_LENGTH: usize = 64;
const SINGLE_TRACE_SUBMISSION: u64 = 1;
const MAXIMUM_TRACE_REGION_BYTES: usize = 268_435_456;
const MAXIMUM_TRACE_EXECUTION_SECONDS: f64 = 60.0;
const MAXIMUM_RECEIPT_QUEUE_DEPTH: usize = 4;
const WINDOW_PROFILE_ID: &str = "rwkv7-p150x2-persistent-decode-v2";
const SHORT_DECODE_WINDOW: u32 = 2;
const LONG_DECODE_WINDOW: u32 = 4;
const CANCELLATION_POLL_STEP: u32 = 2;
const TELEMETRY_SCHEMA_VERSION: u32 = 1;
const TELEMETRY_EVENT_KIND: &str = "rwkv7_persistent_decode_worker";
const TELEMETRY_CLOCK: &str = "steady_monotonic";
const TELEMETRY_UNIT: &str = "seconds";
const TELEMETRY_PASSED: &str = "passed";
const TELEMETRY_NOT_APPLICABLE: &str = "not_applicable";
const TELEMETRY_CLEANUP_COMPLETE: &str = "complete";
const REQUIRED_BINDINGS: [&str; 9] = [
    "token",
    "state",
    "slot",
    "weight",
    "program",
    "runtime",
    "checkpoint",
    "output",
    "profile",
];

fn valid_blake3(value: &str) -> bool {
    value.len() == BLAKE3_HEX_LENGTH
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn terminal_outcomes_from_lifecycle(lifecycles: &[LifecycleReceipt]) -> TelemetryTerminalOutcomes {
    let mut outcomes = TelemetryTerminalOutcomes {
        completed: 0,
        cancelled: 0,
        timed_out: 0,
        failed: 0,
    };
    for lifecycle in lifecycles {
        match lifecycle.terminal_reason.as_str() {
            "completed" => outcomes.completed += 1,
            "cancelled" => outcomes.cancelled += 1,
            "timed_out" => outcomes.timed_out += 1,
            "failed" => outcomes.failed += 1,
            _ => {}
        }
    }
    outcomes
}

fn telemetry_outcome_count(outcomes: &TelemetryTerminalOutcomes) -> u64 {
    u64::from(outcomes.completed)
        + u64::from(outcomes.cancelled)
        + u64::from(outcomes.timed_out)
        + u64::from(outcomes.failed)
}

fn validate_persistent_telemetry_value(
    value: &serde_json::Value,
    persistent: &PersistentSection,
) -> Result<(), PersistentReceiptError> {
    let telemetry: PersistentTelemetry =
        serde_json::from_value(value.clone()).map_err(|error| {
            PersistentReceiptError::new("invalid_persistent_telemetry", error.to_string())
        })?;
    if telemetry.schema_version != TELEMETRY_SCHEMA_VERSION
        || telemetry.event_kind != TELEMETRY_EVENT_KIND
        || telemetry.timing.clock != TELEMETRY_CLOCK
        || telemetry.timing.unit != TELEMETRY_UNIT
    {
        return Err(PersistentReceiptError::new(
            "invalid_persistent_telemetry",
            "telemetry schema, event kind, clock, or unit is invalid",
        ));
    }
    if telemetry.eligibility.window_admitted != persistent.window_admitted
        || telemetry.eligibility.workload_eligible != persistent.workload_eligible
        || telemetry.eligibility.bindings_validated != persistent.bindings_validated
        || telemetry.eligibility.profile_physically_admitted != persistent.physically_admitted
        || telemetry.eligibility.candidate_requested != persistent.candidate_requested
        || telemetry.dispatch.selected_mode != persistent.selected_mode
        || telemetry.dispatch.selected_window_steps != persistent.selected_decode_window_steps
        || telemetry.dispatch.fallback_reason != persistent.fallback_reason
    {
        return Err(PersistentReceiptError::new(
            "persistent_telemetry_mismatch",
            "telemetry eligibility or dispatch differs from detailed evidence",
        ));
    }
    let durations = [
        telemetry.timing.trace_setup_seconds,
        telemetry.timing.trace_replay_seconds,
        telemetry.timing.cleanup_seconds,
    ];
    if durations
        .iter()
        .any(|duration| !duration.is_finite() || *duration < 0.0)
        || telemetry.timing.trace_replay_seconds != persistent.trace_execution_seconds
    {
        return Err(PersistentReceiptError::new(
            "invalid_persistent_telemetry_timing",
            "telemetry timing is invalid or differs from trace evidence",
        ));
    }
    let expected_outcomes = terminal_outcomes_from_lifecycle(&persistent.lifecycle_receipts);
    let mut lifecycle_slots = BTreeSet::new();
    let lifecycle_valid = persistent.lifecycle_receipts.iter().all(|lifecycle| {
        let phase_matches_reason = matches!(
            (lifecycle.phase.as_str(), lifecycle.terminal_reason.as_str()),
            ("completed", "completed")
                | ("cancelled", "cancelled")
                | ("timed_out", "timed_out")
                | ("failed", "failed")
        );
        phase_matches_reason && lifecycle.slot_releasable && lifecycle_slots.insert(lifecycle.slot)
    });
    if telemetry.terminal_outcomes.completed != expected_outcomes.completed
        || telemetry.terminal_outcomes.cancelled != expected_outcomes.cancelled
        || telemetry.terminal_outcomes.timed_out != expected_outcomes.timed_out
        || telemetry.terminal_outcomes.failed != expected_outcomes.failed
    {
        return Err(PersistentReceiptError::new(
            "persistent_telemetry_mismatch",
            "telemetry terminal counts differ from lifecycle evidence",
        ));
    }
    let parity_passed = persistent.persistent_region_entered
        && persistent.exact_generated_token_parity
        && persistent.exact_fp32_recurrent_state_parity;
    let expected_parity = if parity_passed {
        TELEMETRY_PASSED
    } else {
        TELEMETRY_NOT_APPLICABLE
    };
    if telemetry.parity.generated_tokens != expected_parity
        || telemetry.parity.fp32_recurrent_state != expected_parity
    {
        return Err(PersistentReceiptError::new(
            "persistent_telemetry_mismatch",
            "telemetry parity differs from detailed evidence",
        ));
    }
    let cleanup_complete = persistent.selected_mode == "persistent_trace"
        && persistent.state_zeroed_before_slot_release
        && !persistent.lifecycle_receipts.is_empty()
        && lifecycle_valid;
    let expected_cleanup = if cleanup_complete {
        TELEMETRY_CLEANUP_COMPLETE
    } else {
        TELEMETRY_NOT_APPLICABLE
    };
    if telemetry.cleanup_status != expected_cleanup {
        return Err(PersistentReceiptError::new(
            "persistent_telemetry_mismatch",
            "telemetry cleanup differs from lifecycle evidence",
        ));
    }
    let lifecycle_count = u64::try_from(persistent.lifecycle_receipts.len()).map_err(|error| {
        PersistentReceiptError::new("invalid_persistent_telemetry", error.to_string())
    })?;
    if persistent.selected_mode == "normal_dispatch" {
        if persistent.fallback_reason == "none"
            || !persistent.lifecycle_receipts.is_empty()
            || telemetry_outcome_count(&telemetry.terminal_outcomes) != 0
            || durations.iter().any(|duration| *duration != 0.0)
            || expected_parity != TELEMETRY_NOT_APPLICABLE
            || expected_cleanup != TELEMETRY_NOT_APPLICABLE
        {
            return Err(PersistentReceiptError::new(
                "invalid_persistent_telemetry",
                "normal-dispatch telemetry contains a persistent execution claim",
            ));
        }
    } else if persistent.selected_mode == "persistent_trace" {
        if persistent.fallback_reason != "none"
            || telemetry_outcome_count(&telemetry.terminal_outcomes) != lifecycle_count
            || !cleanup_complete
            || (persistent.persistent_region_entered
                && (telemetry.timing.trace_replay_seconds <= 0.0 || !parity_passed))
            || (!persistent.persistent_region_entered
                && (telemetry.timing.trace_replay_seconds != 0.0 || parity_passed))
        {
            return Err(PersistentReceiptError::new(
                "invalid_persistent_telemetry",
                "persistent telemetry lacks replay or cleanup evidence",
            ));
        }
    } else {
        return Err(PersistentReceiptError::new(
            "invalid_persistent_telemetry",
            "telemetry selected mode is unsupported",
        ));
    }
    Ok(())
}

pub fn validate_persistent_telemetry_receipt(
    receipt_json: &str,
) -> Result<(), PersistentReceiptError> {
    let receipt: WorkerReceipt = serde_json::from_str(receipt_json)
        .map_err(|error| PersistentReceiptError::new("malformed_receipt", error.to_string()))?;
    let telemetry = receipt
        .persistent_decode_telemetry
        .as_ref()
        .ok_or_else(|| {
            PersistentReceiptError::new(
                "missing_persistent_telemetry",
                "worker receipt omitted persistent decode telemetry",
            )
        })?;
    validate_persistent_telemetry_value(telemetry, &receipt.persistent_decode)
}

pub fn validate_persistent_receipt(
    receipt_json: &str,
    policy: PersistentReceiptPolicy,
) -> Result<(), PersistentReceiptError> {
    let receipt: WorkerReceipt = serde_json::from_str(receipt_json)
        .map_err(|error| PersistentReceiptError::new("malformed_receipt", error.to_string()))?;
    let persistent = &receipt.persistent_decode;
    if persistent.hash_algorithm != "blake3"
        || !valid_blake3(&persistent.profile_blake3)
        || !valid_blake3(&persistent.model_profile_blake3)
    {
        return Err(PersistentReceiptError::new(
            "invalid_profile_identity",
            "persistent profile identities must use BLAKE3",
        ));
    }
    if persistent.profile_id == WINDOW_PROFILE_ID
        && (persistent.admitted_decode_windows != [SHORT_DECODE_WINDOW, LONG_DECODE_WINDOW]
            || persistent
                .selected_decode_window_steps
                .is_none_or(|window| !persistent.admitted_decode_windows.contains(&window))
            || !persistent.window_admitted
            || persistent.terminal_probe != "none"
            || persistent.terminal_event_poll_step.is_some()
            || persistent.cancellation_poll_interval_steps != CANCELLATION_POLL_STEP
            || persistent.pre_replay_terminal_event)
    {
        return Err(PersistentReceiptError::new(
            "invalid_decode_window",
            "persistent decode window is not listed in the bound profile",
        ));
    }
    if persistent.binding_identities.len() != REQUIRED_BINDINGS.len()
        || REQUIRED_BINDINGS.iter().any(|binding| {
            persistent
                .binding_identities
                .get(*binding)
                .is_none_or(|identity| !valid_blake3(identity))
        })
        || !persistent.bindings_validated
    {
        return Err(PersistentReceiptError::new(
            "invalid_bindings",
            "persistent bindings are missing, malformed, or unvalidated",
        ));
    }
    if !persistent.workload_eligible
        || persistent.selected_mode != "persistent_trace"
        || persistent.fallback_reason != "none"
        || !persistent.persistent_region_entered
    {
        return Err(PersistentReceiptError::new(
            "persistent_mode_not_entered",
            "the worker did not enter an eligible persistent trace",
        ));
    }
    if persistent.trace_region_bytes == 0
        || persistent.trace_region_bytes > MAXIMUM_TRACE_REGION_BYTES
        || !persistent.trace_capture_outside_timed_region
        || !persistent.prompt_prefill_outside_timed_region
        || !persistent.trace_execution_seconds.is_finite()
        || persistent.trace_execution_seconds <= 0.0
        || persistent.trace_execution_seconds > MAXIMUM_TRACE_EXECUTION_SECONDS
        || persistent.timed_host_model_submissions != SINGLE_TRACE_SUBMISSION
        || !persistent.zero_per_token_host_submissions_proven
    {
        return Err(PersistentReceiptError::new(
            "invalid_trace_execution",
            "trace execution lacks a bounded one-command replay",
        ));
    }
    if !persistent.selected_token_device_resident_inside_region
        || persistent.selected_token_readback_inside_region
        || !persistent.output_readback_after_region
        || persistent.recurrent_state_dtype != "fp32"
    {
        return Err(PersistentReceiptError::new(
            "invalid_device_residency",
            "token or recurrent-state residency crossed the persistent boundary",
        ));
    }
    let Some(final_state) = persistent.final_state_blake3.as_deref() else {
        return Err(PersistentReceiptError::new(
            "missing_state_parity",
            "persistent final state identity is missing",
        ));
    };
    let Some(reference_state) = persistent.reference_state_blake3.as_deref() else {
        return Err(PersistentReceiptError::new(
            "missing_state_parity",
            "normal-dispatch state identity is missing",
        ));
    };
    if !valid_blake3(final_state)
        || final_state != reference_state
        || !persistent.exact_generated_token_parity
        || !persistent.exact_fp32_recurrent_state_parity
    {
        return Err(PersistentReceiptError::new(
            "state_or_token_mismatch",
            "persistent output differs from normal dispatch",
        ));
    }
    if !persistent.reference_state_zeroed || !persistent.state_zeroed_before_slot_release {
        return Err(PersistentReceiptError::new(
            "state_not_zeroed",
            "a reference or persistent state was released before zeroing",
        ));
    }
    if persistent.lifecycle_receipts.is_empty()
        || persistent.lifecycle_receipts.len() > MAXIMUM_RECEIPT_QUEUE_DEPTH
    {
        return Err(PersistentReceiptError::new(
            "invalid_lifecycle_receipts",
            "persistent lifecycle receipt count is outside the queue bound",
        ));
    }
    let mut slots = BTreeSet::new();
    for lifecycle in &persistent.lifecycle_receipts {
        if lifecycle.phase != "completed"
            || (persistent.profile_id == WINDOW_PROFILE_ID
                && (lifecycle.terminal_reason != "completed"
                    || lifecycle.completed_steps
                        != persistent.selected_decode_window_steps.unwrap_or_default()
                    || !lifecycle.failure_code.is_empty()))
            || !lifecycle.slot_releasable
            || !valid_blake3(&lifecycle.receipt_blake3)
            || !slots.insert(lifecycle.slot)
        {
            return Err(PersistentReceiptError::new(
                "invalid_lifecycle_receipts",
                "persistent lifecycle cleanup is incomplete or duplicated",
            ));
        }
    }
    match policy {
        PersistentReceiptPolicy::Candidate => {
            if !receipt.physical_hardware && !persistent.simulated {
                return Err(PersistentReceiptError::new(
                    "unattributed_candidate",
                    "candidate evidence is neither physical nor simulated",
                ));
            }
        }
        PersistentReceiptPolicy::Production => {
            if !receipt.physical_hardware
                || persistent.simulated
                || !receipt.profiler_attributed
                || !receipt.profiler_trace_only
                || !persistent.physically_admitted
                || persistent.candidate_requested
            {
                return Err(PersistentReceiptError::new(
                    "production_not_admitted",
                    "production evidence lacks physical, profiler, or profile admission",
                ));
            }
        }
    }
    if let Some(telemetry) = receipt.persistent_decode_telemetry.as_ref() {
        validate_persistent_telemetry_value(telemetry, persistent)?;
    }
    Ok(())
}

pub fn validate_persistent_terminal_probe_receipt(
    receipt_json: &str,
    expected_probe: &str,
) -> Result<(), PersistentReceiptError> {
    let receipt: WorkerReceipt = serde_json::from_str(receipt_json)
        .map_err(|error| PersistentReceiptError::new("malformed_receipt", error.to_string()))?;
    let persistent = &receipt.persistent_decode;
    let (expected_phase, expected_reason, expected_failure_code) = match expected_probe {
        "cancel" => ("cancelled", "cancelled", ""),
        "timeout" => ("timed_out", "timed_out", ""),
        "failure" => ("failed", "failed", "physical_terminal_probe"),
        _ => {
            return Err(PersistentReceiptError::new(
                "invalid_terminal_probe",
                "expected terminal probe must be cancel, timeout, or failure",
            ));
        }
    };
    let window_valid = persistent.profile_id == WINDOW_PROFILE_ID
        && persistent.admitted_decode_windows == [SHORT_DECODE_WINDOW, LONG_DECODE_WINDOW]
        && persistent.selected_decode_window_steps == Some(LONG_DECODE_WINDOW)
        && persistent.window_admitted
        && persistent.terminal_probe == expected_probe
        && persistent.terminal_event_poll_step == Some(0)
        && persistent.cancellation_poll_interval_steps == CANCELLATION_POLL_STEP
        && persistent.pre_replay_terminal_event;
    let execution_valid = receipt.physical_hardware
        && !persistent.simulated
        && persistent.candidate_requested
        && persistent.workload_eligible
        && persistent.selected_mode == "persistent_trace"
        && persistent.fallback_reason == "none"
        && !persistent.persistent_region_entered
        && persistent.trace_region_bytes > 0
        && persistent.trace_region_bytes <= MAXIMUM_TRACE_REGION_BYTES
        && persistent.trace_capture_outside_timed_region
        && persistent.prompt_prefill_outside_timed_region
        && persistent.trace_execution_seconds == 0.0
        && persistent.timed_host_model_submissions == 0
        && !persistent.zero_per_token_host_submissions_proven
        && !persistent.selected_token_device_resident_inside_region
        && !persistent.selected_token_readback_inside_region
        && !persistent.output_readback_after_region
        && persistent.recurrent_state_dtype == "fp32";
    let identity_valid = persistent.hash_algorithm == "blake3"
        && valid_blake3(&persistent.profile_blake3)
        && valid_blake3(&persistent.model_profile_blake3)
        && persistent.binding_identities.len() == REQUIRED_BINDINGS.len()
        && REQUIRED_BINDINGS.iter().all(|binding| {
            persistent
                .binding_identities
                .get(*binding)
                .is_some_and(|identity| valid_blake3(identity))
        })
        && persistent.bindings_validated;
    let parity_valid = persistent.final_state_blake3.is_none()
        && persistent
            .reference_state_blake3
            .as_deref()
            .is_some_and(valid_blake3)
        && !persistent.exact_generated_token_parity
        && !persistent.exact_fp32_recurrent_state_parity
        && persistent.reference_state_zeroed
        && persistent.state_zeroed_before_slot_release;
    if !window_valid || !execution_valid || !identity_valid || !parity_valid {
        return Err(PersistentReceiptError::new(
            "invalid_terminal_probe",
            "physical terminal probe identity, execution, parity, or cleanup is invalid",
        ));
    }
    if persistent.lifecycle_receipts.is_empty()
        || persistent.lifecycle_receipts.len() > MAXIMUM_RECEIPT_QUEUE_DEPTH
    {
        return Err(PersistentReceiptError::new(
            "invalid_terminal_probe_lifecycle",
            "terminal probe lifecycle count is outside the queue bound",
        ));
    }
    let mut slots = BTreeSet::new();
    for lifecycle in &persistent.lifecycle_receipts {
        if lifecycle.phase != expected_phase
            || lifecycle.terminal_reason != expected_reason
            || lifecycle.completed_steps != 0
            || lifecycle.failure_code != expected_failure_code
            || !lifecycle.slot_releasable
            || !valid_blake3(&lifecycle.receipt_blake3)
            || !slots.insert(lifecycle.slot)
        {
            return Err(PersistentReceiptError::new(
                "invalid_terminal_probe_lifecycle",
                "terminal probe did not acknowledge, zero, and release each slot",
            ));
        }
    }
    if let Some(telemetry) = receipt.persistent_decode_telemetry.as_ref() {
        validate_persistent_telemetry_value(telemetry, persistent)?;
    }
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistentAbReceipt {
    schema_version: u32,
    kind: String,
    status: String,
    physical_hardware: bool,
    physical_device_ids: Vec<u32>,
    profile_id: String,
    profile_blake3: String,
    model_profile_blake3: String,
    workload: AbWorkload,
    order_schedule: Vec<Vec<String>>,
    one_card: OneCardAb,
    two_card: TwoCardAb,
    correctness: AbCorrectness,
    dispatch: AbDispatch,
    profiler: AbProfiler,
    simulator: AbSimulator,
    terminal_probes: AbTerminalProbes,
    slot_recycling: AbSlotRecycling,
    fallback: AbFallback,
    machine_snapshot_blake3: String,
    stable_gain: bool,
    production_physically_admitted: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AbWorkload {
    capacity: String,
    request_count: usize,
    prompt_token_ids: Vec<u32>,
    generated_token_count: usize,
    pinned_fp32_cpu_reference_tokens: Vec<u32>,
    same_runtime: bool,
    same_workload: bool,
    prompt_prefill_outside_timed_region: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OneCardAb {
    repeat_count: usize,
    normal_elapsed_seconds: Vec<f64>,
    persistent_elapsed_seconds: Vec<f64>,
    paired_speedups: Vec<f64>,
    minimum_paired_speedup: f64,
    median_paired_speedup: f64,
    normal_median_seconds: f64,
    persistent_median_seconds: f64,
    median_elapsed_ratio: f64,
    normal_receipt_blake3: Vec<String>,
    persistent_receipt_blake3: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TwoCardAb {
    repeat_count: usize,
    normal_wave_seconds: Vec<f64>,
    persistent_wave_seconds: Vec<f64>,
    paired_speedups: Vec<f64>,
    minimum_paired_speedup: f64,
    median_paired_speedup: f64,
    normal_median_seconds: f64,
    persistent_median_seconds: f64,
    median_elapsed_ratio: f64,
    normal_receipt_blake3_by_sample_and_card: Vec<Vec<String>>,
    persistent_receipt_blake3_by_sample_and_card: Vec<Vec<String>>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AbCorrectness {
    exact_generated_token_parity: bool,
    exact_fp32_recurrent_state_parity: bool,
    reference_state_zeroed: bool,
    persistent_state_zeroed_before_slot_release: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AbDispatch {
    admitted_decode_windows: Vec<u32>,
    selected_decode_window_steps: u32,
    short_window_receipt_blake3: String,
    normal_host_model_submissions_for_admitted_window: u64,
    persistent_trace_replay_submissions_for_admitted_window: u64,
    per_token_host_submissions_inside_persistent_region: u64,
    trace_capture_outside_timed_region: bool,
    trace_region_bytes: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AbProfiler {
    trace_only: bool,
    trace_row_count: usize,
    dropped_marker_warning: bool,
    raw_csv_blake3: String,
    production_worker_receipt_blake3: String,
    retained_worker_receipt_blake3: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AbSimulator {
    performance_claim: bool,
    determinism_claim: bool,
    raw_receipt_blake3: String,
    retained_receipt_blake3: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AbTerminalProbes {
    pre_replay: bool,
    cancellation_poll_interval_steps: u32,
    cancel_receipt_blake3: String,
    timeout_receipt_blake3: String,
    failure_receipt_blake3: String,
    all_acknowledged_zeroed_and_released: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AbSlotRecycling {
    cycle_count: usize,
    slot_ids: Vec<u32>,
    persistent_receipt_blake3: Vec<String>,
    all_completed_zeroed_and_released: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AbFallback {
    unlisted_window_steps: u32,
    selected_mode: String,
    fallback_reason: String,
    receipt_blake3: String,
}

const PHYSICAL_ACCEPTANCE_KIND: &str = "rwkv7_p150x2_persistent_decode_physical_acceptance";
const PHYSICAL_ACCEPTANCE_STATUS: &str = "production_admitted";
const PHYSICAL_ACCEPTANCE_PROFILE_ID: &str = "rwkv7-p150x2-persistent-decode-v2";
const PHYSICAL_DEVICE_IDS: [u32; 2] = [0, 1];
const PHYSICAL_ACCEPTANCE_SAMPLE_COUNT: usize = 5;
const PHYSICAL_ACCEPTANCE_REQUEST_COUNT: usize = 4;
const PHYSICAL_ACCEPTANCE_TOKEN_COUNT: usize = 4;
const PHYSICAL_ACCEPTANCE_PROMPT: [u32; 8] = [0, 24281, 59, 36786, 65530, 5585, 41693, 59];
const PHYSICAL_ACCEPTANCE_REFERENCE: [u32; 4] = [36786, 34, 308, 459];
const PHYSICAL_ACCEPTANCE_CAPACITY: &str = "B4";
const NORMAL_SUBMISSION_COUNT: u64 = 4;
const PERSISTENT_SUBMISSION_COUNT: u64 = 1;
const ORDER_PERIOD: usize = 2;
const UNLISTED_WINDOW_STEPS: u32 = 1;
const FLOAT_COMPARISON_TOLERANCE: f64 = 1.0e-12;

fn approximately_equal(left: f64, right: f64) -> bool {
    left.is_finite() && right.is_finite() && (left - right).abs() <= FLOAT_COMPARISON_TOLERANCE
}

fn median(values: &[f64]) -> Option<f64> {
    if values.len() != PHYSICAL_ACCEPTANCE_SAMPLE_COUNT {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / ORDER_PERIOD).copied()
}

fn validate_ab_hashes(hashes: &[String], expected_count: usize) -> bool {
    hashes.len() == expected_count && hashes.iter().all(|hash| valid_blake3(hash))
}

struct AbTimingEvidence<'a> {
    normal_seconds: &'a [f64],
    persistent_seconds: &'a [f64],
    recorded_speedups: &'a [f64],
    recorded_minimum: f64,
    recorded_median: f64,
    normal_median: f64,
    persistent_median: f64,
    recorded_median_ratio: f64,
}

fn validate_ab_topology(timing: AbTimingEvidence<'_>) -> bool {
    if timing.normal_seconds.len() != PHYSICAL_ACCEPTANCE_SAMPLE_COUNT
        || timing.persistent_seconds.len() != PHYSICAL_ACCEPTANCE_SAMPLE_COUNT
        || timing.recorded_speedups.len() != PHYSICAL_ACCEPTANCE_SAMPLE_COUNT
        || timing
            .normal_seconds
            .iter()
            .chain(timing.persistent_seconds)
            .any(|seconds| !seconds.is_finite() || *seconds <= 0.0)
    {
        return false;
    }
    let computed_speedups = timing
        .normal_seconds
        .iter()
        .zip(timing.persistent_seconds)
        .map(|(normal, persistent)| normal / persistent)
        .collect::<Vec<_>>();
    if computed_speedups
        .iter()
        .zip(timing.recorded_speedups)
        .any(|(computed, recorded)| !approximately_equal(*computed, *recorded) || *computed <= 1.0)
    {
        return false;
    }
    let Some(computed_normal_median) = median(timing.normal_seconds) else {
        return false;
    };
    let Some(computed_persistent_median) = median(timing.persistent_seconds) else {
        return false;
    };
    let Some(computed_speedup_median) = median(&computed_speedups) else {
        return false;
    };
    let computed_minimum = computed_speedups
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    approximately_equal(computed_minimum, timing.recorded_minimum)
        && approximately_equal(computed_speedup_median, timing.recorded_median)
        && approximately_equal(computed_normal_median, timing.normal_median)
        && approximately_equal(computed_persistent_median, timing.persistent_median)
        && approximately_equal(
            computed_normal_median / computed_persistent_median,
            timing.recorded_median_ratio,
        )
}

pub fn validate_persistent_ab_receipt(receipt_json: &str) -> Result<(), PersistentReceiptError> {
    let receipt: PersistentAbReceipt = serde_json::from_str(receipt_json)
        .map_err(|error| PersistentReceiptError::new("malformed_ab_receipt", error.to_string()))?;
    if receipt.schema_version != 1
        || receipt.kind != PHYSICAL_ACCEPTANCE_KIND
        || receipt.status != PHYSICAL_ACCEPTANCE_STATUS
        || !receipt.physical_hardware
        || receipt.physical_device_ids != PHYSICAL_DEVICE_IDS
        || receipt.profile_id != PHYSICAL_ACCEPTANCE_PROFILE_ID
        || !valid_blake3(&receipt.profile_blake3)
        || !valid_blake3(&receipt.model_profile_blake3)
        || !valid_blake3(&receipt.machine_snapshot_blake3)
        || !receipt.stable_gain
        || !receipt.production_physically_admitted
    {
        return Err(PersistentReceiptError::new(
            "invalid_ab_identity",
            "physical A/B identity or admission fields are invalid",
        ));
    }
    let workload = &receipt.workload;
    if workload.capacity != PHYSICAL_ACCEPTANCE_CAPACITY
        || workload.request_count != PHYSICAL_ACCEPTANCE_REQUEST_COUNT
        || workload.prompt_token_ids != PHYSICAL_ACCEPTANCE_PROMPT
        || workload.generated_token_count != PHYSICAL_ACCEPTANCE_TOKEN_COUNT
        || workload.pinned_fp32_cpu_reference_tokens != PHYSICAL_ACCEPTANCE_REFERENCE
        || !workload.same_runtime
        || !workload.same_workload
        || !workload.prompt_prefill_outside_timed_region
    {
        return Err(PersistentReceiptError::new(
            "invalid_ab_workload",
            "physical A/B workload or timing boundary changed",
        ));
    }
    if receipt.order_schedule.len() != PHYSICAL_ACCEPTANCE_SAMPLE_COUNT
        || receipt
            .order_schedule
            .iter()
            .enumerate()
            .any(|(index, order)| {
                let expected = if index % ORDER_PERIOD == 0 {
                    ["normal_dispatch", "persistent_trace"]
                } else {
                    ["persistent_trace", "normal_dispatch"]
                };
                order.iter().map(String::as_str).collect::<Vec<_>>() != expected
            })
    {
        return Err(PersistentReceiptError::new(
            "invalid_ab_order",
            "physical A/B order is not balanced",
        ));
    }
    let one = &receipt.one_card;
    let one_valid = one.repeat_count == PHYSICAL_ACCEPTANCE_SAMPLE_COUNT
        && validate_ab_topology(AbTimingEvidence {
            normal_seconds: &one.normal_elapsed_seconds,
            persistent_seconds: &one.persistent_elapsed_seconds,
            recorded_speedups: &one.paired_speedups,
            recorded_minimum: one.minimum_paired_speedup,
            recorded_median: one.median_paired_speedup,
            normal_median: one.normal_median_seconds,
            persistent_median: one.persistent_median_seconds,
            recorded_median_ratio: one.median_elapsed_ratio,
        })
        && validate_ab_hashes(&one.normal_receipt_blake3, PHYSICAL_ACCEPTANCE_SAMPLE_COUNT)
        && validate_ab_hashes(
            &one.persistent_receipt_blake3,
            PHYSICAL_ACCEPTANCE_SAMPLE_COUNT,
        );
    let two = &receipt.two_card;
    let two_hashes_valid = two
        .normal_receipt_blake3_by_sample_and_card
        .iter()
        .chain(&two.persistent_receipt_blake3_by_sample_and_card)
        .all(|sample| validate_ab_hashes(sample, PHYSICAL_DEVICE_IDS.len()));
    let two_valid = two.repeat_count == PHYSICAL_ACCEPTANCE_SAMPLE_COUNT
        && two.normal_receipt_blake3_by_sample_and_card.len() == PHYSICAL_ACCEPTANCE_SAMPLE_COUNT
        && two.persistent_receipt_blake3_by_sample_and_card.len()
            == PHYSICAL_ACCEPTANCE_SAMPLE_COUNT
        && two_hashes_valid
        && validate_ab_topology(AbTimingEvidence {
            normal_seconds: &two.normal_wave_seconds,
            persistent_seconds: &two.persistent_wave_seconds,
            recorded_speedups: &two.paired_speedups,
            recorded_minimum: two.minimum_paired_speedup,
            recorded_median: two.median_paired_speedup,
            normal_median: two.normal_median_seconds,
            persistent_median: two.persistent_median_seconds,
            recorded_median_ratio: two.median_elapsed_ratio,
        });
    if !one_valid || !two_valid {
        return Err(PersistentReceiptError::new(
            "unstable_ab_gain",
            "one-card or two-card physical A/B evidence is malformed or has no stable gain",
        ));
    }
    let correctness = &receipt.correctness;
    let dispatch = &receipt.dispatch;
    if !correctness.exact_generated_token_parity
        || !correctness.exact_fp32_recurrent_state_parity
        || !correctness.reference_state_zeroed
        || !correctness.persistent_state_zeroed_before_slot_release
        || dispatch.admitted_decode_windows != [SHORT_DECODE_WINDOW, LONG_DECODE_WINDOW]
        || dispatch.selected_decode_window_steps != LONG_DECODE_WINDOW
        || !valid_blake3(&dispatch.short_window_receipt_blake3)
        || dispatch.normal_host_model_submissions_for_admitted_window != NORMAL_SUBMISSION_COUNT
        || dispatch.persistent_trace_replay_submissions_for_admitted_window
            != PERSISTENT_SUBMISSION_COUNT
        || dispatch.per_token_host_submissions_inside_persistent_region != 0
        || !dispatch.trace_capture_outside_timed_region
        || dispatch.trace_region_bytes == 0
        || dispatch.trace_region_bytes > MAXIMUM_TRACE_REGION_BYTES
    {
        return Err(PersistentReceiptError::new(
            "invalid_ab_correctness",
            "physical A/B correctness or dispatch reduction is invalid",
        ));
    }
    let profiler = &receipt.profiler;
    let simulator = &receipt.simulator;
    let terminal = &receipt.terminal_probes;
    let recycling = &receipt.slot_recycling;
    let fallback = &receipt.fallback;
    if !terminal.pre_replay
        || terminal.cancellation_poll_interval_steps != CANCELLATION_POLL_STEP
        || !valid_blake3(&terminal.cancel_receipt_blake3)
        || !valid_blake3(&terminal.timeout_receipt_blake3)
        || !valid_blake3(&terminal.failure_receipt_blake3)
        || !terminal.all_acknowledged_zeroed_and_released
        || recycling.cycle_count != PHYSICAL_ACCEPTANCE_SAMPLE_COUNT
        || recycling.slot_ids != [0, 1, 2, 3]
        || !validate_ab_hashes(
            &recycling.persistent_receipt_blake3,
            PHYSICAL_ACCEPTANCE_SAMPLE_COUNT,
        )
        || !recycling.all_completed_zeroed_and_released
        || fallback.unlisted_window_steps != UNLISTED_WINDOW_STEPS
        || fallback.selected_mode != "normal_dispatch"
        || fallback.fallback_reason != "window_not_admitted"
        || !valid_blake3(&fallback.receipt_blake3)
    {
        return Err(PersistentReceiptError::new(
            "invalid_ab_cleanup",
            "terminal probes, slot recycling, or fallback evidence is invalid",
        ));
    }
    if !profiler.trace_only
        || profiler.trace_row_count == 0
        || profiler.dropped_marker_warning
        || !valid_blake3(&profiler.raw_csv_blake3)
        || !valid_blake3(&profiler.production_worker_receipt_blake3)
        || !valid_blake3(&profiler.retained_worker_receipt_blake3)
        || simulator.performance_claim
        || simulator.determinism_claim
        || !valid_blake3(&simulator.raw_receipt_blake3)
        || !valid_blake3(&simulator.retained_receipt_blake3)
    {
        return Err(PersistentReceiptError::new(
            "invalid_ab_proof",
            "profiler or simulator proof is invalid",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEADER: &str = include_str!("../../rwkv7-p150x2/src/persistent_decode.hpp");
    const CORE: &str = include_str!("../../rwkv7-p150x2/src/persistent_decode.cpp");
    const SELF_TEST: &str = include_str!("../../rwkv7-p150x2/src/persistent_decode_self_test.cpp");
    const RUNTIME: &str = include_str!("../../rwkv7-p150x2/src/main.cpp");
    const PROFILE: &str = include_str!("../../rwkv7-p150x2/persistent-decode-profile.ncl");
    const VALID_HASH: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const VALID_TRACE_REGION_BYTES: usize = MAXIMUM_TRACE_REGION_BYTES;
    const VALID_TRACE_SECONDS: f64 = 0.5;
    const VALID_TRACE_SETUP_SECONDS: f64 = 0.25;
    const VALID_CLEANUP_SECONDS: f64 = 0.125;
    const INVALID_TRACE_SUBMISSIONS: u64 = 2;

    fn valid_candidate_receipt() -> serde_json::Value {
        let mut receipt = serde_json::json!({
            "physical_hardware": false,
            "profiler_attributed": false,
            "persistent_decode": {
                "profile_blake3": VALID_HASH,
                "model_profile_blake3": VALID_HASH,
                "hash_algorithm": "blake3",
                "simulated": true,
                "physically_admitted": false,
                "candidate_requested": true,
                "workload_eligible": true,
                "prompt_prefill_outside_timed_region": true,
                "binding_identities": {
                    "token": VALID_HASH,
                    "state": VALID_HASH,
                    "slot": VALID_HASH,
                    "weight": VALID_HASH,
                    "program": VALID_HASH,
                    "runtime": VALID_HASH,
                    "checkpoint": VALID_HASH,
                    "output": VALID_HASH,
                    "profile": VALID_HASH
                },
                "bindings_validated": true,
                "selected_mode": "persistent_trace",
                "fallback_reason": "none",
                "persistent_region_entered": true,
                "trace_region_bytes": VALID_TRACE_REGION_BYTES,
                "trace_capture_outside_timed_region": true,
                "trace_execution_seconds": VALID_TRACE_SECONDS,
                "timed_host_model_submissions": 1,
                "zero_per_token_host_submissions_proven": true,
                "selected_token_device_resident_inside_region": true,
                "selected_token_readback_inside_region": false,
                "output_readback_after_region": true,
                "recurrent_state_dtype": "fp32",
                "final_state_blake3": VALID_HASH,
                "reference_state_blake3": VALID_HASH,
                "exact_generated_token_parity": true,
                "exact_fp32_recurrent_state_parity": true,
                "reference_state_zeroed": true,
                "state_zeroed_before_slot_release": true,
                "lifecycle_receipts": [{
                    "slot": 0,
                    "phase": "completed",
                    "slot_releasable": true,
                    "receipt_blake3": VALID_HASH
                }]
            }
        });
        receipt["persistent_decode"]["profile_id"] = serde_json::json!(WINDOW_PROFILE_ID);
        receipt["persistent_decode"]["admitted_decode_windows"] =
            serde_json::json!([SHORT_DECODE_WINDOW, LONG_DECODE_WINDOW]);
        receipt["persistent_decode"]["selected_decode_window_steps"] =
            serde_json::json!(LONG_DECODE_WINDOW);
        receipt["persistent_decode"]["terminal_probe"] = serde_json::json!("none");
        receipt["persistent_decode"]["terminal_event_poll_step"] = serde_json::Value::Null;
        receipt["persistent_decode"]["cancellation_poll_interval_steps"] =
            serde_json::json!(CANCELLATION_POLL_STEP);
        receipt["persistent_decode"]["pre_replay_terminal_event"] = serde_json::json!(false);
        receipt["persistent_decode"]["window_admitted"] = serde_json::json!(true);
        receipt["persistent_decode"]["lifecycle_receipts"][0]["terminal_reason"] =
            serde_json::json!("completed");
        receipt["persistent_decode"]["lifecycle_receipts"][0]["completed_steps"] =
            serde_json::json!(LONG_DECODE_WINDOW);
        receipt["persistent_decode"]["lifecycle_receipts"][0]["failure_code"] =
            serde_json::json!("");
        receipt["persistent_decode_telemetry"] = serde_json::json!({
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "event_kind": TELEMETRY_EVENT_KIND,
            "eligibility": {
                "window_admitted": true,
                "workload_eligible": true,
                "bindings_validated": true,
                "profile_physically_admitted": false,
                "candidate_requested": true
            },
            "dispatch": {
                "selected_mode": "persistent_trace",
                "selected_window_steps": LONG_DECODE_WINDOW,
                "fallback_reason": "none"
            },
            "terminal_outcomes": {
                "completed": 1,
                "cancelled": 0,
                "timed_out": 0,
                "failed": 0
            },
            "timing": {
                "clock": TELEMETRY_CLOCK,
                "unit": TELEMETRY_UNIT,
                "trace_setup_seconds": VALID_TRACE_SETUP_SECONDS,
                "trace_replay_seconds": VALID_TRACE_SECONDS,
                "cleanup_seconds": VALID_CLEANUP_SECONDS
            },
            "parity": {
                "generated_tokens": TELEMETRY_PASSED,
                "fp32_recurrent_state": TELEMETRY_PASSED
            },
            "cleanup_status": TELEMETRY_CLEANUP_COMPLETE
        });
        receipt
    }

    fn valid_terminal_probe_receipt(probe: &str) -> serde_json::Value {
        let mut receipt = valid_candidate_receipt();
        let (phase, reason, failure_code) = match probe {
            "cancel" => ("cancelled", "cancelled", ""),
            "timeout" => ("timed_out", "timed_out", ""),
            "failure" => ("failed", "failed", "physical_terminal_probe"),
            _ => panic!("invalid test terminal probe"),
        };
        receipt["physical_hardware"] = serde_json::json!(true);
        receipt["persistent_decode"]["simulated"] = serde_json::json!(false);
        receipt["persistent_decode"]["terminal_probe"] = serde_json::json!(probe);
        receipt["persistent_decode"]["terminal_event_poll_step"] = serde_json::json!(0);
        receipt["persistent_decode"]["pre_replay_terminal_event"] = serde_json::json!(true);
        receipt["persistent_decode"]["persistent_region_entered"] = serde_json::json!(false);
        receipt["persistent_decode"]["trace_execution_seconds"] = serde_json::json!(0.0);
        receipt["persistent_decode"]["timed_host_model_submissions"] = serde_json::json!(0);
        receipt["persistent_decode"]["zero_per_token_host_submissions_proven"] =
            serde_json::json!(false);
        receipt["persistent_decode"]["selected_token_device_resident_inside_region"] =
            serde_json::json!(false);
        receipt["persistent_decode"]["output_readback_after_region"] = serde_json::json!(false);
        receipt["persistent_decode"]["final_state_blake3"] = serde_json::Value::Null;
        receipt["persistent_decode"]["exact_generated_token_parity"] = serde_json::json!(false);
        receipt["persistent_decode"]["exact_fp32_recurrent_state_parity"] =
            serde_json::json!(false);
        receipt["persistent_decode"]["lifecycle_receipts"][0]["phase"] = serde_json::json!(phase);
        receipt["persistent_decode"]["lifecycle_receipts"][0]["terminal_reason"] =
            serde_json::json!(reason);
        receipt["persistent_decode"]["lifecycle_receipts"][0]["completed_steps"] =
            serde_json::json!(0);
        receipt["persistent_decode"]["lifecycle_receipts"][0]["failure_code"] =
            serde_json::json!(failure_code);
        receipt["persistent_decode_telemetry"]["terminal_outcomes"] = serde_json::json!({
            "completed": 0,
            "cancelled": u32::from(probe == "cancel"),
            "timed_out": u32::from(probe == "timeout"),
            "failed": u32::from(probe == "failure")
        });
        receipt["persistent_decode_telemetry"]["timing"]["trace_replay_seconds"] =
            serde_json::json!(0.0);
        receipt["persistent_decode_telemetry"]["parity"]["generated_tokens"] =
            serde_json::json!(TELEMETRY_NOT_APPLICABLE);
        receipt["persistent_decode_telemetry"]["parity"]["fp32_recurrent_state"] =
            serde_json::json!(TELEMETRY_NOT_APPLICABLE);
        receipt
    }

    fn valid_fallback_telemetry_receipt() -> serde_json::Value {
        let mut receipt = valid_candidate_receipt();
        receipt["persistent_decode"]["physically_admitted"] = serde_json::json!(true);
        receipt["persistent_decode"]["candidate_requested"] = serde_json::json!(false);
        receipt["persistent_decode"]["selected_decode_window_steps"] = serde_json::Value::Null;
        receipt["persistent_decode"]["window_admitted"] = serde_json::json!(false);
        receipt["persistent_decode"]["workload_eligible"] = serde_json::json!(false);
        receipt["persistent_decode"]["selected_mode"] = serde_json::json!("normal_dispatch");
        receipt["persistent_decode"]["fallback_reason"] = serde_json::json!("window_not_admitted");
        receipt["persistent_decode"]["persistent_region_entered"] = serde_json::json!(false);
        receipt["persistent_decode"]["trace_execution_seconds"] = serde_json::json!(0.0);
        receipt["persistent_decode"]["timed_host_model_submissions"] = serde_json::json!(0);
        receipt["persistent_decode"]["zero_per_token_host_submissions_proven"] =
            serde_json::json!(false);
        receipt["persistent_decode"]["selected_token_device_resident_inside_region"] =
            serde_json::json!(false);
        receipt["persistent_decode"]["output_readback_after_region"] = serde_json::json!(false);
        receipt["persistent_decode"]["final_state_blake3"] = serde_json::Value::Null;
        receipt["persistent_decode"]["reference_state_blake3"] = serde_json::Value::Null;
        receipt["persistent_decode"]["exact_generated_token_parity"] = serde_json::json!(false);
        receipt["persistent_decode"]["exact_fp32_recurrent_state_parity"] =
            serde_json::json!(false);
        receipt["persistent_decode"]["reference_state_zeroed"] = serde_json::json!(false);
        receipt["persistent_decode"]["state_zeroed_before_slot_release"] = serde_json::json!(false);
        receipt["persistent_decode"]["lifecycle_receipts"] = serde_json::json!([]);
        receipt["persistent_decode_telemetry"] = serde_json::json!({
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "event_kind": TELEMETRY_EVENT_KIND,
            "eligibility": {
                "window_admitted": false,
                "workload_eligible": false,
                "bindings_validated": true,
                "profile_physically_admitted": true,
                "candidate_requested": false
            },
            "dispatch": {
                "selected_mode": "normal_dispatch",
                "selected_window_steps": null,
                "fallback_reason": "window_not_admitted"
            },
            "terminal_outcomes": {
                "completed": 0,
                "cancelled": 0,
                "timed_out": 0,
                "failed": 0
            },
            "timing": {
                "clock": TELEMETRY_CLOCK,
                "unit": TELEMETRY_UNIT,
                "trace_setup_seconds": 0.0,
                "trace_replay_seconds": 0.0,
                "cleanup_seconds": 0.0
            },
            "parity": {
                "generated_tokens": TELEMETRY_NOT_APPLICABLE,
                "fp32_recurrent_state": TELEMETRY_NOT_APPLICABLE
            },
            "cleanup_status": TELEMETRY_NOT_APPLICABLE
        });
        receipt
    }

    fn valid_ab_receipt() -> serde_json::Value {
        serde_json::json!({
            "schema_version": 1,
            "kind": PHYSICAL_ACCEPTANCE_KIND,
            "status": PHYSICAL_ACCEPTANCE_STATUS,
            "physical_hardware": true,
            "physical_device_ids": PHYSICAL_DEVICE_IDS,
            "profile_id": PHYSICAL_ACCEPTANCE_PROFILE_ID,
            "profile_blake3": VALID_HASH,
            "model_profile_blake3": VALID_HASH,
            "workload": {
                "capacity": PHYSICAL_ACCEPTANCE_CAPACITY,
                "request_count": PHYSICAL_ACCEPTANCE_REQUEST_COUNT,
                "prompt_token_ids": PHYSICAL_ACCEPTANCE_PROMPT,
                "generated_token_count": PHYSICAL_ACCEPTANCE_TOKEN_COUNT,
                "pinned_fp32_cpu_reference_tokens": PHYSICAL_ACCEPTANCE_REFERENCE,
                "same_runtime": true,
                "same_workload": true,
                "prompt_prefill_outside_timed_region": true
            },
            "order_schedule": [
                ["normal_dispatch", "persistent_trace"],
                ["persistent_trace", "normal_dispatch"],
                ["normal_dispatch", "persistent_trace"],
                ["persistent_trace", "normal_dispatch"],
                ["normal_dispatch", "persistent_trace"]
            ],
            "one_card": {
                "repeat_count": PHYSICAL_ACCEPTANCE_SAMPLE_COUNT,
                "normal_elapsed_seconds": [2.0, 4.0, 6.0, 8.0, 10.0],
                "persistent_elapsed_seconds": [1.0, 2.0, 3.0, 4.0, 5.0],
                "paired_speedups": [2.0, 2.0, 2.0, 2.0, 2.0],
                "minimum_paired_speedup": 2.0,
                "median_paired_speedup": 2.0,
                "normal_median_seconds": 6.0,
                "persistent_median_seconds": 3.0,
                "median_elapsed_ratio": 2.0,
                "normal_receipt_blake3": [VALID_HASH, VALID_HASH, VALID_HASH, VALID_HASH, VALID_HASH],
                "persistent_receipt_blake3": [VALID_HASH, VALID_HASH, VALID_HASH, VALID_HASH, VALID_HASH]
            },
            "two_card": {
                "repeat_count": PHYSICAL_ACCEPTANCE_SAMPLE_COUNT,
                "normal_wave_seconds": [2.0, 4.0, 6.0, 8.0, 10.0],
                "persistent_wave_seconds": [1.0, 2.0, 3.0, 4.0, 5.0],
                "paired_speedups": [2.0, 2.0, 2.0, 2.0, 2.0],
                "minimum_paired_speedup": 2.0,
                "median_paired_speedup": 2.0,
                "normal_median_seconds": 6.0,
                "persistent_median_seconds": 3.0,
                "median_elapsed_ratio": 2.0,
                "normal_receipt_blake3_by_sample_and_card": [
                    [VALID_HASH, VALID_HASH],
                    [VALID_HASH, VALID_HASH],
                    [VALID_HASH, VALID_HASH],
                    [VALID_HASH, VALID_HASH],
                    [VALID_HASH, VALID_HASH]
                ],
                "persistent_receipt_blake3_by_sample_and_card": [
                    [VALID_HASH, VALID_HASH],
                    [VALID_HASH, VALID_HASH],
                    [VALID_HASH, VALID_HASH],
                    [VALID_HASH, VALID_HASH],
                    [VALID_HASH, VALID_HASH]
                ]
            },
            "correctness": {
                "exact_generated_token_parity": true,
                "exact_fp32_recurrent_state_parity": true,
                "reference_state_zeroed": true,
                "persistent_state_zeroed_before_slot_release": true
            },
            "dispatch": {
                "admitted_decode_windows": [SHORT_DECODE_WINDOW, LONG_DECODE_WINDOW],
                "selected_decode_window_steps": LONG_DECODE_WINDOW,
                "short_window_receipt_blake3": VALID_HASH,
                "normal_host_model_submissions_for_admitted_window": NORMAL_SUBMISSION_COUNT,
                "persistent_trace_replay_submissions_for_admitted_window": PERSISTENT_SUBMISSION_COUNT,
                "per_token_host_submissions_inside_persistent_region": 0,
                "trace_capture_outside_timed_region": true,
                "trace_region_bytes": MAXIMUM_TRACE_REGION_BYTES
            },
            "profiler": {
                "trace_only": true,
                "trace_row_count": 1,
                "dropped_marker_warning": false,
                "raw_csv_blake3": VALID_HASH,
                "production_worker_receipt_blake3": VALID_HASH,
                "retained_worker_receipt_blake3": VALID_HASH
            },
            "simulator": {
                "performance_claim": false,
                "determinism_claim": false,
                "raw_receipt_blake3": VALID_HASH,
                "retained_receipt_blake3": VALID_HASH
            },
            "terminal_probes": {
                "pre_replay": true,
                "cancellation_poll_interval_steps": CANCELLATION_POLL_STEP,
                "cancel_receipt_blake3": VALID_HASH,
                "timeout_receipt_blake3": VALID_HASH,
                "failure_receipt_blake3": VALID_HASH,
                "all_acknowledged_zeroed_and_released": true
            },
            "slot_recycling": {
                "cycle_count": PHYSICAL_ACCEPTANCE_SAMPLE_COUNT,
                "slot_ids": [0, 1, 2, 3],
                "persistent_receipt_blake3": [VALID_HASH, VALID_HASH, VALID_HASH, VALID_HASH, VALID_HASH],
                "all_completed_zeroed_and_released": true
            },
            "fallback": {
                "unlisted_window_steps": UNLISTED_WINDOW_STEPS,
                "selected_mode": "normal_dispatch",
                "fallback_reason": "window_not_admitted",
                "receipt_blake3": VALID_HASH
            },
            "machine_snapshot_blake3": VALID_HASH,
            "stable_gain": true,
            "production_physically_admitted": true
        })
    }

    fn validate_value(
        value: &serde_json::Value,
        policy: PersistentReceiptPolicy,
    ) -> Result<(), PersistentReceiptError> {
        validate_persistent_receipt(&value.to_string(), policy)
    }

    #[test]
    fn accepts_the_window_profile_sources() {
        validate_persistent_decode_sources(HEADER, CORE, SELF_TEST, RUNTIME, PROFILE)
            .expect("reviewed persistent decode sources");
    }

    #[test]
    fn rejects_missing_lifecycle_cleanup() {
        let invalid = CORE.replace("slot_releasable", "removed_slot_release");
        let error =
            validate_persistent_decode_sources(HEADER, &invalid, SELF_TEST, RUNTIME, PROFILE)
                .expect_err("missing slot cleanup");
        assert_eq!(error.surface, "core");
        assert_eq!(error.pattern, "slot_releasable");
    }

    #[test]
    fn rejects_missing_binding_drift_coverage() {
        let invalid = SELF_TEST.replacen(
            "test_binding_drift_and_receipts",
            "removed_binding_drift_test",
            2,
        );
        let error = validate_persistent_decode_sources(HEADER, CORE, &invalid, RUNTIME, PROFILE)
            .expect_err("missing binding drift test");
        assert_eq!(error.surface, "self-test");
    }

    #[test]
    fn rejects_a_trace_without_device_resident_selection() {
        let invalid = RUNTIME.replacen(
            "\"selected_token_device_resident_inside_region\"",
            "\"removed_device_resident_selection\"",
            1,
        );
        let error = validate_persistent_decode_sources(HEADER, CORE, SELF_TEST, &invalid, PROFILE)
            .expect_err("missing device-resident selection claim");
        assert_eq!(error.surface, "runtime");
    }

    #[test]
    fn rejects_a_trace_without_one_command_replay() {
        let invalid = RUNTIME.replacen("execute_trace", "removed_trace_replay", 1);
        let error = validate_persistent_decode_sources(HEADER, CORE, SELF_TEST, &invalid, PROFILE)
            .expect_err("missing trace replay");
        assert_eq!(error.surface, "runtime");
    }

    #[test]
    fn rejects_a_profile_without_admitted_windows() {
        let invalid = PROFILE.replacen("admitted_decode_windows", "removed_decode_windows", 1);
        let error = validate_persistent_decode_sources(HEADER, CORE, SELF_TEST, RUNTIME, &invalid)
            .expect_err("removed admitted windows");
        assert_eq!(error.surface, "profile");
    }

    #[test]
    fn accepts_candidate_and_production_receipts_at_their_own_boundaries() {
        let candidate = valid_candidate_receipt();
        validate_value(&candidate, PersistentReceiptPolicy::Candidate)
            .expect("simulated candidate receipt");

        let mut production = candidate;
        production["physical_hardware"] = serde_json::json!(true);
        production["profiler_attributed"] = serde_json::json!(true);
        production["profiler_trace_only"] = serde_json::json!(true);
        production["persistent_decode"]["simulated"] = serde_json::json!(false);
        production["persistent_decode"]["physically_admitted"] = serde_json::json!(true);
        production["persistent_decode"]["candidate_requested"] = serde_json::json!(false);
        production["persistent_decode_telemetry"]["eligibility"]["profile_physically_admitted"] =
            serde_json::json!(true);
        production["persistent_decode_telemetry"]["eligibility"]["candidate_requested"] =
            serde_json::json!(false);
        validate_value(&production, PersistentReceiptPolicy::Production)
            .expect("physical production receipt");
    }

    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.valid]
    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.fallback]
    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.terminal]
    #[test]
    fn accepts_strict_persistent_terminal_and_fallback_telemetry() {
        validate_persistent_telemetry_receipt(&valid_candidate_receipt().to_string())
            .expect("persistent replay telemetry");
        validate_persistent_telemetry_receipt(&valid_terminal_probe_receipt("cancel").to_string())
            .expect("pre-replay terminal telemetry");
        validate_persistent_telemetry_receipt(&valid_fallback_telemetry_receipt().to_string())
            .expect("normal fallback telemetry");
    }

    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.invalid]
    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_telemetry.aggregate_only]
    #[test]
    fn rejects_missing_mismatched_and_high_cardinality_telemetry() {
        let mut invalid = valid_candidate_receipt();
        invalid
            .as_object_mut()
            .expect("worker receipt object")
            .remove("persistent_decode_telemetry");
        assert_eq!(
            validate_persistent_telemetry_receipt(&invalid.to_string())
                .expect_err("missing telemetry")
                .code,
            "missing_persistent_telemetry"
        );

        invalid = valid_candidate_receipt();
        invalid["persistent_decode_telemetry"]["dispatch"]["fallback_reason"] =
            serde_json::json!("trace_unavailable");
        assert_eq!(
            validate_persistent_telemetry_receipt(&invalid.to_string())
                .expect_err("fallback mismatch")
                .code,
            "persistent_telemetry_mismatch"
        );

        invalid = valid_candidate_receipt();
        invalid["persistent_decode_telemetry"]["terminal_outcomes"]["completed"] =
            serde_json::json!(0);
        assert_eq!(
            validate_persistent_telemetry_receipt(&invalid.to_string())
                .expect_err("terminal count mismatch")
                .code,
            "persistent_telemetry_mismatch"
        );

        invalid = valid_candidate_receipt();
        let duplicate_lifecycle = invalid["persistent_decode"]["lifecycle_receipts"][0].clone();
        invalid["persistent_decode"]["lifecycle_receipts"]
            .as_array_mut()
            .expect("lifecycle receipt array")
            .push(duplicate_lifecycle);
        invalid["persistent_decode_telemetry"]["terminal_outcomes"]["completed"] =
            serde_json::json!(2);
        assert_eq!(
            validate_persistent_telemetry_receipt(&invalid.to_string())
                .expect_err("duplicate lifecycle slot")
                .code,
            "persistent_telemetry_mismatch"
        );

        invalid = valid_candidate_receipt();
        invalid["persistent_decode_telemetry"]["timing"]["cleanup_seconds"] =
            serde_json::json!(-1.0);
        assert_eq!(
            validate_persistent_telemetry_receipt(&invalid.to_string())
                .expect_err("negative cleanup time")
                .code,
            "invalid_persistent_telemetry_timing"
        );

        invalid = valid_candidate_receipt();
        invalid["persistent_decode_telemetry"]["timing"]["cleanup_seconds"] =
            serde_json::json!("NaN");
        assert_eq!(
            validate_persistent_telemetry_receipt(&invalid.to_string())
                .expect_err("non-finite cleanup time")
                .code,
            "invalid_persistent_telemetry"
        );

        invalid = valid_candidate_receipt();
        invalid["persistent_decode_telemetry"]["request_id"] =
            serde_json::json!("high-cardinality-request");
        assert_eq!(
            validate_persistent_telemetry_receipt(&invalid.to_string())
                .expect_err("high-cardinality telemetry")
                .code,
            "invalid_persistent_telemetry"
        );
    }

    #[test]
    fn accepts_bounded_terminal_probe_cleanup_and_rejects_drift() {
        for probe in ["cancel", "timeout", "failure"] {
            let receipt = valid_terminal_probe_receipt(probe);
            validate_persistent_terminal_probe_receipt(&receipt.to_string(), probe)
                .expect("physical terminal probe receipt");
        }

        let mut invalid = valid_terminal_probe_receipt("cancel");
        invalid["persistent_decode"]["lifecycle_receipts"][0]["slot_releasable"] =
            serde_json::json!(false);
        assert_eq!(
            validate_persistent_terminal_probe_receipt(&invalid.to_string(), "cancel")
                .expect_err("terminal probe released state before cleanup")
                .code,
            "invalid_terminal_probe_lifecycle"
        );

        invalid = valid_terminal_probe_receipt("timeout");
        invalid["persistent_decode"]["terminal_event_poll_step"] =
            serde_json::json!(CANCELLATION_POLL_STEP);
        assert_eq!(
            validate_persistent_terminal_probe_receipt(&invalid.to_string(), "timeout")
                .expect_err("terminal probe moved beyond the polling boundary")
                .code,
            "invalid_terminal_probe"
        );
    }

    #[test]
    fn rejects_binding_submission_parity_and_zeroing_failures() {
        let mut invalid = valid_candidate_receipt();
        invalid["persistent_decode"]["binding_identities"]["token"] = serde_json::json!("stale");
        assert_eq!(
            validate_value(&invalid, PersistentReceiptPolicy::Candidate)
                .expect_err("stale binding")
                .code,
            "invalid_bindings"
        );

        invalid = valid_candidate_receipt();
        invalid["persistent_decode"]["timed_host_model_submissions"] =
            serde_json::json!(INVALID_TRACE_SUBMISSIONS);
        assert_eq!(
            validate_value(&invalid, PersistentReceiptPolicy::Candidate)
                .expect_err("per-token submissions")
                .code,
            "invalid_trace_execution"
        );

        invalid = valid_candidate_receipt();
        invalid["persistent_decode"]["trace_capture_outside_timed_region"] =
            serde_json::json!(false);
        assert_eq!(
            validate_value(&invalid, PersistentReceiptPolicy::Candidate)
                .expect_err("capture inside timed region")
                .code,
            "invalid_trace_execution"
        );

        invalid = valid_candidate_receipt();
        invalid["persistent_decode"]["prompt_prefill_outside_timed_region"] =
            serde_json::json!(false);
        assert_eq!(
            validate_value(&invalid, PersistentReceiptPolicy::Candidate)
                .expect_err("prefill inside timed region")
                .code,
            "invalid_trace_execution"
        );

        invalid = valid_candidate_receipt();
        invalid["persistent_decode"]["final_state_blake3"] =
            serde_json::json!("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
        assert_eq!(
            validate_value(&invalid, PersistentReceiptPolicy::Candidate)
                .expect_err("state mismatch")
                .code,
            "state_or_token_mismatch"
        );

        invalid = valid_candidate_receipt();
        invalid["persistent_decode"]["state_zeroed_before_slot_release"] = serde_json::json!(false);
        assert_eq!(
            validate_value(&invalid, PersistentReceiptPolicy::Candidate)
                .expect_err("state not zeroed")
                .code,
            "state_not_zeroed"
        );
    }

    #[test]
    fn rejects_simulator_evidence_for_production() {
        let invalid = valid_candidate_receipt();
        assert_eq!(
            validate_value(&invalid, PersistentReceiptPolicy::Production)
                .expect_err("simulator cannot admit production")
                .code,
            "production_not_admitted"
        );
    }

    #[test]
    fn accepts_stable_one_card_and_two_card_ab_evidence() {
        validate_persistent_ab_receipt(&valid_ab_receipt().to_string())
            .expect("stable physical A/B evidence");
    }

    #[test]
    fn rejects_unstable_or_unbalanced_ab_evidence() {
        let mut invalid = valid_ab_receipt();
        invalid["one_card"]["persistent_elapsed_seconds"][0] = serde_json::json!(4.0);
        assert_eq!(
            validate_persistent_ab_receipt(&invalid.to_string())
                .expect_err("one-card regression")
                .code,
            "unstable_ab_gain"
        );

        invalid = valid_ab_receipt();
        invalid["order_schedule"][1] = serde_json::json!(["normal_dispatch", "persistent_trace"]);
        assert_eq!(
            validate_persistent_ab_receipt(&invalid.to_string())
                .expect_err("unbalanced A/B order")
                .code,
            "invalid_ab_order"
        );

        invalid = valid_ab_receipt();
        invalid["terminal_probes"]["all_acknowledged_zeroed_and_released"] =
            serde_json::json!(false);
        assert_eq!(
            validate_persistent_ab_receipt(&invalid.to_string())
                .expect_err("terminal cleanup drift")
                .code,
            "invalid_ab_cleanup"
        );
    }
}
