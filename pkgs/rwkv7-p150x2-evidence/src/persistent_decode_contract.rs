//! Static guard for bounded, physically admitted persistent decode.
//!
//! The compiled C++ self-test proves lifecycle behavior. This guard ensures
//! that source drift cannot remove admission, typed limits, BLAKE3 bindings,
//! cleanup handshakes, or normal-dispatch fallback without a Rust test failure.

use serde::{Deserialize, Serialize};
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

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ValidatedPersistentTelemetry {
    pub window_admitted: bool,
    pub workload_eligible: bool,
    pub bindings_validated: bool,
    pub profile_physically_admitted: bool,
    pub candidate_requested: bool,
    pub selected_mode: String,
    pub selected_window_steps: Option<u32>,
    pub fallback_reason: String,
    pub terminal_completed: u32,
    pub terminal_cancelled: u32,
    pub terminal_timed_out: u32,
    pub terminal_failed: u32,
    pub trace_setup_seconds: f64,
    pub trace_replay_seconds: f64,
    pub cleanup_seconds: f64,
    pub generated_token_parity: String,
    pub fp32_recurrent_state_parity: String,
    pub cleanup_status: String,
}

const BLAKE3_HEX_LENGTH: usize = 64;
const SINGLE_TRACE_SUBMISSION: u64 = 1;
const MAXIMUM_TRACE_REGION_BYTES: usize = 268_435_456;
const MAXIMUM_TRACE_EXECUTION_SECONDS: f64 = 60.0;
const MAXIMUM_RECEIPT_QUEUE_DEPTH: usize = 4;
const WINDOW_PROFILE_ID: &str = "rwkv7-p150x2-persistent-decode-v2";
const WINDOW_EIGHT_CANDIDATE_PROFILE_ID: &str =
    "rwkv7-p150x2-persistent-decode-window8-candidate-v1";
const SHORT_DECODE_WINDOW: u32 = 2;
const LONG_DECODE_WINDOW: u32 = 4;
const WINDOW_EIGHT_DECODE_WINDOW: u32 = 8;
const PRODUCTION_DECODE_WINDOWS: [u32; 2] = [SHORT_DECODE_WINDOW, LONG_DECODE_WINDOW];
const WINDOW_EIGHT_CANDIDATE_DECODE_WINDOWS: [u32; 3] = [
    SHORT_DECODE_WINDOW,
    LONG_DECODE_WINDOW,
    WINDOW_EIGHT_DECODE_WINDOW,
];
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

fn expected_profile_windows(profile_id: &str) -> Option<(&'static [u32], u32)> {
    match profile_id {
        WINDOW_PROFILE_ID => Some((&PRODUCTION_DECODE_WINDOWS, LONG_DECODE_WINDOW)),
        WINDOW_EIGHT_CANDIDATE_PROFILE_ID => Some((
            &WINDOW_EIGHT_CANDIDATE_DECODE_WINDOWS,
            WINDOW_EIGHT_DECODE_WINDOW,
        )),
        _ => None,
    }
}

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

pub fn validated_persistent_telemetry(
    receipt_json: &str,
) -> Result<ValidatedPersistentTelemetry, PersistentReceiptError> {
    let receipt: WorkerReceipt = serde_json::from_str(receipt_json)
        .map_err(|error| PersistentReceiptError::new("malformed_receipt", error.to_string()))?;
    let telemetry_value = receipt
        .persistent_decode_telemetry
        .as_ref()
        .ok_or_else(|| {
            PersistentReceiptError::new(
                "missing_persistent_telemetry",
                "worker receipt omitted persistent decode telemetry",
            )
        })?;
    validate_persistent_telemetry_value(telemetry_value, &receipt.persistent_decode)?;
    let telemetry: PersistentTelemetry =
        serde_json::from_value(telemetry_value.clone()).map_err(|error| {
            PersistentReceiptError::new("invalid_persistent_telemetry", error.to_string())
        })?;
    Ok(ValidatedPersistentTelemetry {
        window_admitted: telemetry.eligibility.window_admitted,
        workload_eligible: telemetry.eligibility.workload_eligible,
        bindings_validated: telemetry.eligibility.bindings_validated,
        profile_physically_admitted: telemetry.eligibility.profile_physically_admitted,
        candidate_requested: telemetry.eligibility.candidate_requested,
        selected_mode: telemetry.dispatch.selected_mode,
        selected_window_steps: telemetry.dispatch.selected_window_steps,
        fallback_reason: telemetry.dispatch.fallback_reason,
        terminal_completed: telemetry.terminal_outcomes.completed,
        terminal_cancelled: telemetry.terminal_outcomes.cancelled,
        terminal_timed_out: telemetry.terminal_outcomes.timed_out,
        terminal_failed: telemetry.terminal_outcomes.failed,
        trace_setup_seconds: telemetry.timing.trace_setup_seconds,
        trace_replay_seconds: telemetry.timing.trace_replay_seconds,
        cleanup_seconds: telemetry.timing.cleanup_seconds,
        generated_token_parity: telemetry.parity.generated_tokens,
        fp32_recurrent_state_parity: telemetry.parity.fp32_recurrent_state,
        cleanup_status: telemetry.cleanup_status,
    })
}

pub fn validate_persistent_telemetry_receipt(
    receipt_json: &str,
) -> Result<(), PersistentReceiptError> {
    validated_persistent_telemetry(receipt_json).map(|_| ())
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
    let Some((expected_windows, _expected_maximum_window)) =
        expected_profile_windows(&persistent.profile_id)
    else {
        return Err(PersistentReceiptError::new(
            "invalid_decode_window",
            "persistent decode profile is not an accepted production or window-8 candidate profile",
        ));
    };
    if persistent.admitted_decode_windows != expected_windows
        || persistent
            .selected_decode_window_steps
            .is_none_or(|window| !persistent.admitted_decode_windows.contains(&window))
        || !persistent.window_admitted
        || persistent.terminal_probe != "none"
        || persistent.terminal_event_poll_step.is_some()
        || persistent.cancellation_poll_interval_steps != CANCELLATION_POLL_STEP
        || persistent.pre_replay_terminal_event
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
            || lifecycle.terminal_reason != "completed"
            || lifecycle.completed_steps
                != persistent.selected_decode_window_steps.unwrap_or_default()
            || !lifecycle.failure_code.is_empty()
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
    let expected_profile = expected_profile_windows(&persistent.profile_id);
    let window_valid = expected_profile.is_some_and(|(windows, maximum_window)| {
        persistent.admitted_decode_windows == windows
            && persistent.selected_decode_window_steps == Some(maximum_window)
    }) && persistent.window_admitted
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

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct OneCardDeviceAssignment {
    normal_dispatch_device_id: u32,
    persistent_trace_device_id: u32,
}

const PHYSICAL_AB_SCHEMA_VERSION: u32 = 1;
const PHYSICAL_AB_KIND: &str = "rwkv7_persistent_decode_window_physical_ab";
const PHYSICAL_AB_STATUS: &str = "stable_physical_gain";
const PHYSICAL_AB_BASELINE_MODE: &str = "normal_dispatch";
const PHYSICAL_AB_CANDIDATE_MODE: &str = "persistent_trace";
const PHYSICAL_AB_ONE_CARD_PAIRING: &str = "concurrent_card_balanced";
const PHYSICAL_AB_PROFILER_ATTESTATION: &str = "profiler-attestation.json";
const PHYSICAL_AB_MACHINE_FACTS_BEFORE: &str = "machine-before.json";
const PHYSICAL_AB_MACHINE_FACTS_AFTER: &str = "machine-after.json";
const PHYSICAL_AB_CPU_REFERENCE_PROFILE_BLAKE3: &str =
    "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88";
const PHYSICAL_DEVICE_IDS: [u32; 2] = [0, 1];
const PHYSICAL_ACCEPTANCE_SAMPLE_COUNT: usize = 5;
const PHYSICAL_ACCEPTANCE_REQUEST_COUNT: usize = 4;
const PHYSICAL_AB_PRIMARY_THREAD_CPU_AFFINITIES: [u32; 2] = [2, 6];
const PHYSICAL_AB_PRIMARY_THREAD_FIFO_PRIORITY: i32 = 1;
const PHYSICAL_AB_MINIMUM_PAIRED_SPEEDUP: f64 = 1.0;
const ORDER_PERIOD: usize = 2;

fn valid_ab_order_schedule(schedule: &[Vec<String>], expected_count: usize) -> bool {
    schedule.len() == expected_count
        && schedule.iter().enumerate().all(|(index, order)| {
            let expected = if index.is_multiple_of(ORDER_PERIOD) {
                [PHYSICAL_AB_BASELINE_MODE, PHYSICAL_AB_CANDIDATE_MODE]
            } else {
                [PHYSICAL_AB_CANDIDATE_MODE, PHYSICAL_AB_BASELINE_MODE]
            };
            order.len() == expected.len() && order[0] == expected[0] && order[1] == expected[1]
        })
}

fn valid_one_card_device_schedule(
    schedule: &[OneCardDeviceAssignment],
    expected_count: usize,
) -> bool {
    schedule.len() == expected_count
        && schedule.iter().enumerate().all(|(index, assignment)| {
            let (normal, persistent) = if index.is_multiple_of(ORDER_PERIOD) {
                (PHYSICAL_DEVICE_IDS[0], PHYSICAL_DEVICE_IDS[1])
            } else {
                (PHYSICAL_DEVICE_IDS[1], PHYSICAL_DEVICE_IDS[0])
            };
            assignment.normal_dispatch_device_id == normal
                && assignment.persistent_trace_device_id == persistent
        })
}

fn valid_ab_speedups(speedups: &[f64], expected_count: usize) -> bool {
    speedups.len() == expected_count
        && speedups
            .iter()
            .all(|speedup| speedup.is_finite() && *speedup >= PHYSICAL_AB_MINIMUM_PAIRED_SPEEDUP)
}

fn valid_ab_rates(receipt: &PersistentAbReceipt) -> bool {
    let rates = [
        receipt.baseline_one_card_tokens_per_second,
        receipt.candidate_one_card_tokens_per_second,
        receipt.baseline_two_card_tokens_per_second,
        receipt.candidate_two_card_tokens_per_second,
    ];
    rates.iter().all(|rate| rate.is_finite() && *rate > 0.0)
        && receipt.candidate_one_card_tokens_per_second
            >= receipt.baseline_one_card_tokens_per_second
        && receipt.candidate_two_card_tokens_per_second
            >= receipt.baseline_two_card_tokens_per_second
}

pub fn validate_persistent_ab_receipt(receipt_json: &str) -> Result<(), PersistentReceiptError> {
    let receipt: PersistentAbReceipt = serde_json::from_str(receipt_json)
        .map_err(|error| PersistentReceiptError::new("malformed_ab_receipt", error.to_string()))?;
    if receipt.schema_version != PHYSICAL_AB_SCHEMA_VERSION
        || receipt.kind != PHYSICAL_AB_KIND
        || receipt.status != PHYSICAL_AB_STATUS
        || !receipt.physical_hardware
        || receipt.production_physically_admitted
        || receipt.baseline_mode != PHYSICAL_AB_BASELINE_MODE
        || receipt.candidate_mode != PHYSICAL_AB_CANDIDATE_MODE
        || receipt.cpu_reference_profile_blake3 != PHYSICAL_AB_CPU_REFERENCE_PROFILE_BLAKE3
        || !valid_blake3(&receipt.baseline_receipt_identity_blake3)
        || !valid_blake3(&receipt.candidate_receipt_identity_blake3)
        || receipt.baseline_receipt_identity_blake3 == receipt.candidate_receipt_identity_blake3
    {
        return Err(PersistentReceiptError::new(
            "invalid_ab_identity",
            "physical A/B identity or admission fields are invalid",
        ));
    }
    if receipt.batch_capacity != PHYSICAL_ACCEPTANCE_REQUEST_COUNT
        || receipt.repeat_count != PHYSICAL_ACCEPTANCE_SAMPLE_COUNT
        || receipt.one_card_pairing != PHYSICAL_AB_ONE_CARD_PAIRING
        || receipt.primary_thread_cpu_affinities != PHYSICAL_AB_PRIMARY_THREAD_CPU_AFFINITIES
        || receipt.primary_thread_fifo_priority != PHYSICAL_AB_PRIMARY_THREAD_FIFO_PRIORITY
    {
        return Err(PersistentReceiptError::new(
            "invalid_ab_workload",
            "physical A/B workload, sample count, or host controls changed",
        ));
    }
    if !valid_ab_order_schedule(&receipt.order_schedule, PHYSICAL_ACCEPTANCE_SAMPLE_COUNT)
        || !valid_one_card_device_schedule(
            &receipt.one_card_device_schedule,
            PHYSICAL_ACCEPTANCE_SAMPLE_COUNT,
        )
    {
        return Err(PersistentReceiptError::new(
            "invalid_ab_order",
            "physical A/B execution or card order is not balanced",
        ));
    }
    if !receipt.stable_gain
        || !valid_ab_rates(&receipt)
        || !valid_ab_speedups(
            &receipt.one_card_paired_speedups,
            PHYSICAL_ACCEPTANCE_SAMPLE_COUNT,
        )
        || !valid_ab_speedups(
            &receipt.two_card_paired_speedups,
            PHYSICAL_ACCEPTANCE_SAMPLE_COUNT,
        )
    {
        return Err(PersistentReceiptError::new(
            "unstable_ab_gain",
            "one-card or two-card physical A/B evidence has no stable non-negative gain",
        ));
    }
    if !receipt.exact_generated_token_parity
        || receipt.profiler_attestation != PHYSICAL_AB_PROFILER_ATTESTATION
        || receipt.machine_facts_before != PHYSICAL_AB_MACHINE_FACTS_BEFORE
        || receipt.machine_facts_after != PHYSICAL_AB_MACHINE_FACTS_AFTER
    {
        return Err(PersistentReceiptError::new(
            "invalid_ab_proof",
            "physical A/B parity or retained proof references are invalid",
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
    const OTHER_VALID_HASH: &str =
        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
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
            "schema_version": PHYSICAL_AB_SCHEMA_VERSION,
            "kind": PHYSICAL_AB_KIND,
            "status": PHYSICAL_AB_STATUS,
            "physical_hardware": true,
            "production_physically_admitted": false,
            "batch_capacity": PHYSICAL_ACCEPTANCE_REQUEST_COUNT,
            "repeat_count": PHYSICAL_ACCEPTANCE_SAMPLE_COUNT,
            "order_schedule": [
                [PHYSICAL_AB_BASELINE_MODE, PHYSICAL_AB_CANDIDATE_MODE],
                [PHYSICAL_AB_CANDIDATE_MODE, PHYSICAL_AB_BASELINE_MODE],
                [PHYSICAL_AB_BASELINE_MODE, PHYSICAL_AB_CANDIDATE_MODE],
                [PHYSICAL_AB_CANDIDATE_MODE, PHYSICAL_AB_BASELINE_MODE],
                [PHYSICAL_AB_BASELINE_MODE, PHYSICAL_AB_CANDIDATE_MODE]
            ],
            "one_card_pairing": PHYSICAL_AB_ONE_CARD_PAIRING,
            "one_card_device_schedule": [
                {
                    "normal_dispatch_device_id": PHYSICAL_DEVICE_IDS[0],
                    "persistent_trace_device_id": PHYSICAL_DEVICE_IDS[1]
                },
                {
                    "normal_dispatch_device_id": PHYSICAL_DEVICE_IDS[1],
                    "persistent_trace_device_id": PHYSICAL_DEVICE_IDS[0]
                },
                {
                    "normal_dispatch_device_id": PHYSICAL_DEVICE_IDS[0],
                    "persistent_trace_device_id": PHYSICAL_DEVICE_IDS[1]
                },
                {
                    "normal_dispatch_device_id": PHYSICAL_DEVICE_IDS[1],
                    "persistent_trace_device_id": PHYSICAL_DEVICE_IDS[0]
                },
                {
                    "normal_dispatch_device_id": PHYSICAL_DEVICE_IDS[0],
                    "persistent_trace_device_id": PHYSICAL_DEVICE_IDS[1]
                }
            ],
            "primary_thread_cpu_affinities": PHYSICAL_AB_PRIMARY_THREAD_CPU_AFFINITIES,
            "primary_thread_fifo_priority": PHYSICAL_AB_PRIMARY_THREAD_FIFO_PRIORITY,
            "exact_generated_token_parity": true,
            "cpu_reference_profile_blake3": PHYSICAL_AB_CPU_REFERENCE_PROFILE_BLAKE3,
            "baseline_mode": PHYSICAL_AB_BASELINE_MODE,
            "candidate_mode": PHYSICAL_AB_CANDIDATE_MODE,
            "baseline_receipt_identity_blake3": VALID_HASH,
            "candidate_receipt_identity_blake3": OTHER_VALID_HASH,
            "baseline_one_card_tokens_per_second": 100.0,
            "candidate_one_card_tokens_per_second": 200.0,
            "baseline_two_card_tokens_per_second": 100.0,
            "candidate_two_card_tokens_per_second": 200.0,
            "one_card_paired_speedups": [2.0, 2.0, 2.0, 2.0, 2.0],
            "two_card_paired_speedups": [2.0, 2.0, 2.0, 2.0, 2.0],
            "stable_gain": true,
            "profiler_attestation": PHYSICAL_AB_PROFILER_ATTESTATION,
            "machine_facts_before": PHYSICAL_AB_MACHINE_FACTS_BEFORE,
            "machine_facts_after": PHYSICAL_AB_MACHINE_FACTS_AFTER
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
    fn accepts_compact_stable_one_card_and_two_card_ab_evidence() {
        validate_persistent_ab_receipt(&valid_ab_receipt().to_string())
            .expect("compact stable physical A/B evidence");
    }

    #[test]
    fn rejects_compact_ab_gain_order_and_device_drift() {
        let mut invalid = valid_ab_receipt();
        invalid["one_card_paired_speedups"][0] = serde_json::json!(0.5);
        assert_eq!(
            validate_persistent_ab_receipt(&invalid.to_string())
                .expect_err("one-card regression")
                .code,
            "unstable_ab_gain"
        );

        invalid = valid_ab_receipt();
        invalid["order_schedule"][1] =
            serde_json::json!([PHYSICAL_AB_BASELINE_MODE, PHYSICAL_AB_CANDIDATE_MODE]);
        assert_eq!(
            validate_persistent_ab_receipt(&invalid.to_string())
                .expect_err("unbalanced A/B order")
                .code,
            "invalid_ab_order"
        );

        invalid = valid_ab_receipt();
        invalid["one_card_device_schedule"][0]["persistent_trace_device_id"] =
            serde_json::json!(PHYSICAL_DEVICE_IDS[0]);
        assert_eq!(
            validate_persistent_ab_receipt(&invalid.to_string())
                .expect_err("same-card comparative pair")
                .code,
            "invalid_ab_order"
        );
    }

    #[test]
    fn rejects_compact_ab_identity_parity_and_schema_drift() {
        let mut invalid = valid_ab_receipt();
        invalid["production_physically_admitted"] = serde_json::json!(true);
        assert_eq!(
            validate_persistent_ab_receipt(&invalid.to_string())
                .expect_err("candidate A/B receipt claimed production admission")
                .code,
            "invalid_ab_identity"
        );

        invalid = valid_ab_receipt();
        invalid["cpu_reference_profile_blake3"] = serde_json::json!(VALID_HASH);
        assert_eq!(
            validate_persistent_ab_receipt(&invalid.to_string())
                .expect_err("CPU reference identity drift")
                .code,
            "invalid_ab_identity"
        );

        invalid = valid_ab_receipt();
        invalid["exact_generated_token_parity"] = serde_json::json!(false);
        assert_eq!(
            validate_persistent_ab_receipt(&invalid.to_string())
                .expect_err("generated-token parity failure")
                .code,
            "invalid_ab_proof"
        );

        invalid = valid_ab_receipt();
        invalid["unexpected_detail"] = serde_json::json!("not compact");
        assert_eq!(
            validate_persistent_ab_receipt(&invalid.to_string())
                .expect_err("unknown compact receipt field")
                .code,
            "malformed_ab_receipt"
        );
    }
}
