//! Bounded production aggregation and alert classification for persistent decode telemetry.

use crate::persistent_decode_contract::{
    ValidatedPersistentTelemetry, validated_persistent_telemetry,
};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::{BTreeMap, BTreeSet};

pub const MONITORING_SCHEMA_VERSION: u32 = 1;
pub const MONITORING_EVENT_KIND: &str = "rwkv7_persistent_decode_monitoring";
pub const MONITORING_POLICY_ID: &str = "rwkv7-p150x2-persistent-decode-monitoring-v1";
pub const CRITICAL_ACTION: &str = "disable_persistent_admission";
pub const WARNING_ACTION: &str = "investigate_fallback_rate";
const MONITORING_HASH_DOMAIN: &[u8] = b"rwkv7-p150x2-persistent-decode-monitoring-v1";
const MAXIMUM_POLICY_BYTES: usize = 65_536;
const HARD_MAXIMUM_BATCH_EVENTS: usize = 256;
const HARD_MAXIMUM_EVENT_BYTES: usize = 1_048_576;
const SHORT_DECODE_WINDOW: u32 = 2;
const LONG_DECODE_WINDOW: u32 = 4;
const WINDOW_EIGHT_DECODE_WINDOW: u32 = 8;
const WINDOW_EIGHT_MONITORING_POLICY_ID: &str =
    "rwkv7-p150x2-persistent-decode-monitoring-window8-v1";
const PRODUCTION_ADMITTED_WINDOWS: [u32; 2] = [SHORT_DECODE_WINDOW, LONG_DECODE_WINDOW];
const WINDOW_EIGHT_ADMITTED_WINDOWS: [u32; 3] = [
    SHORT_DECODE_WINDOW,
    LONG_DECODE_WINDOW,
    WINDOW_EIGHT_DECODE_WINDOW,
];
const SELECTED_MODE_PERSISTENT: &str = "persistent_trace";
const SELECTED_MODE_NORMAL: &str = "normal_dispatch";
const FALLBACK_NONE: &str = "none";
const TELEMETRY_PASSED: &str = "passed";
const TELEMETRY_NOT_APPLICABLE: &str = "not_applicable";
const TELEMETRY_CLEANUP_COMPLETE: &str = "complete";
const TELEMETRY_CLEANUP_NOT_APPLICABLE: &str = "not_applicable";
const ALERT_INVALID_TELEMETRY: &str = "invalid_telemetry";
const ALERT_DUPLICATE_EVENT: &str = "duplicate_event";
const ALERT_PARITY_FAILURE: &str = "parity_failure";
const ALERT_CLEANUP_FAILURE: &str = "cleanup_failure";
const ALERT_REPLAY_TIMEOUT: &str = "replay_timeout";
const ALERT_TERMINAL_FAILURE: &str = "terminal_failure";
const ALERT_UNSUPPORTED_ADMISSION: &str = "unsupported_admission";
const ALERT_FALLBACK_RATE: &str = "fallback_rate_high";
const NON_CLAIMS: [&str; 5] = [
    "the receipt does not deliver network metrics or pages",
    "the receipt does not mutate persistent admission",
    "the receipt does not control canary traffic",
    "the receipt does not prove telemetry producer correctness",
    "the receipt does not replace worker process supervision",
];
const FALLBACK_REASONS: [&str; 9] = [
    "binding_mismatch",
    "trace_unavailable",
    "parity_evidence_missing",
    "simulator_evidence_missing",
    "profiler_evidence_missing",
    "stable_gain_missing",
    "profile_not_admitted",
    "window_not_admitted",
    "canary_interlock_blocked",
];

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct MonitoringPolicy {
    pub schema_version: u32,
    pub policy_id: String,
    pub maximum_batch_events: usize,
    pub maximum_event_bytes: usize,
    pub maximum_timing_seconds: f64,
    pub minimum_fallback_samples: usize,
    pub maximum_fallback_rate: f64,
    pub admitted_windows: Vec<u32>,
    pub critical_action: String,
    pub warning_action: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MonitoringError {
    pub code: &'static str,
    pub detail: String,
}

impl MonitoringError {
    fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for MonitoringError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

impl std::error::Error for MonitoringError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertSeverity {
    Warning,
    Critical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MonitoringVerdict {
    Clean,
    Warning,
    Critical,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PersistentAdmissionRecommendation {
    Retain,
    Disable,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MonitoringAlert {
    pub code: String,
    pub severity: AlertSeverity,
    pub count: u64,
    pub action: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct MonitoringSourceBinding {
    pub event_count: u64,
    pub accepted_event_count: u64,
    pub rejected_event_count: u64,
    pub event_blake3: Vec<String>,
    pub batch_blake3: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct EligibilityMetrics {
    pub window_admitted: u64,
    pub workload_eligible: u64,
    pub bindings_validated: u64,
    pub profile_physically_admitted: u64,
    pub candidate_requested: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct DispatchMetrics {
    pub persistent_trace: u64,
    pub normal_dispatch: u64,
    pub selected_windows: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FallbackMetrics {
    pub count: u64,
    pub rate: f64,
    pub reasons: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct TerminalMetrics {
    pub completed: u64,
    pub cancelled: u64,
    pub timed_out: u64,
    pub failed: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct ParityMetrics {
    pub passed: u64,
    pub not_applicable: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct CleanupMetrics {
    pub complete: u64,
    pub not_applicable: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TimingMetric {
    pub sum_seconds: f64,
    pub maximum_seconds: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TimingMetrics {
    pub trace_setup: TimingMetric,
    pub trace_replay: TimingMetric,
    pub cleanup: TimingMetric,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MonitoringMetrics {
    pub eligibility: EligibilityMetrics,
    pub dispatch: DispatchMetrics,
    pub fallback: FallbackMetrics,
    pub terminal: TerminalMetrics,
    pub parity: ParityMetrics,
    pub cleanup: CleanupMetrics,
    pub timing: TimingMetrics,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PersistentDecodeMonitoringReceipt {
    pub schema_version: u32,
    pub event_kind: String,
    pub hash_algorithm: String,
    pub policy_id: String,
    pub policy_blake3: String,
    pub source: MonitoringSourceBinding,
    pub metrics: MonitoringMetrics,
    pub alerts: Vec<MonitoringAlert>,
    pub verdict: MonitoringVerdict,
    pub persistent_admission_recommendation: PersistentAdmissionRecommendation,
    pub non_claims: Vec<String>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct RawSafetySignals {
    parity_failure: u64,
    cleanup_failure: u64,
    replay_timeout: u64,
    terminal_failure: u64,
    unsupported_admission: u64,
}

impl RawSafetySignals {
    fn add(&mut self, other: Self) {
        self.parity_failure = self.parity_failure.saturating_add(other.parity_failure);
        self.cleanup_failure = self.cleanup_failure.saturating_add(other.cleanup_failure);
        self.replay_timeout = self.replay_timeout.saturating_add(other.replay_timeout);
        self.terminal_failure = self.terminal_failure.saturating_add(other.terminal_failure);
        self.unsupported_admission = self
            .unsupported_admission
            .saturating_add(other.unsupported_admission);
    }
}

fn valid_policy_window_set(policy_id: &str, admitted_windows: &[u32]) -> bool {
    (policy_id == MONITORING_POLICY_ID && admitted_windows == PRODUCTION_ADMITTED_WINDOWS)
        || (policy_id == WINDOW_EIGHT_MONITORING_POLICY_ID
            && admitted_windows == WINDOW_EIGHT_ADMITTED_WINDOWS)
}

// r[impl tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.policy]
pub fn parse_monitoring_policy(policy_json: &str) -> Result<MonitoringPolicy, MonitoringError> {
    if policy_json.len() > MAXIMUM_POLICY_BYTES {
        return Err(MonitoringError::new(
            "monitoring_policy_too_large",
            "monitoring policy exceeds the hard byte limit",
        ));
    }
    let policy: MonitoringPolicy = serde_json::from_str(policy_json)
        .map_err(|error| MonitoringError::new("invalid_monitoring_policy", error.to_string()))?;
    let valid_rate = policy.maximum_fallback_rate.is_finite()
        && (0.0..=1.0).contains(&policy.maximum_fallback_rate);
    let valid_timing =
        policy.maximum_timing_seconds.is_finite() && policy.maximum_timing_seconds > 0.0;
    if policy.schema_version != MONITORING_SCHEMA_VERSION
        || !valid_policy_window_set(&policy.policy_id, &policy.admitted_windows)
        || policy.maximum_batch_events == 0
        || policy.maximum_batch_events > HARD_MAXIMUM_BATCH_EVENTS
        || policy.maximum_event_bytes == 0
        || policy.maximum_event_bytes > HARD_MAXIMUM_EVENT_BYTES
        || policy.minimum_fallback_samples == 0
        || policy.minimum_fallback_samples > policy.maximum_batch_events
        || !valid_rate
        || !valid_timing
        || policy.critical_action != CRITICAL_ACTION
        || policy.warning_action != WARNING_ACTION
    {
        return Err(MonitoringError::new(
            "invalid_monitoring_policy",
            "monitoring policy violates its bounded production contract",
        ));
    }
    Ok(policy)
}

fn domain_hash(parts: &[&[u8]]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(MONITORING_HASH_DOMAIN);
    for part in parts {
        let part_length =
            u64::try_from(part.len()).expect("bounded monitoring hash part must fit u64");
        hasher.update(&part_length.to_le_bytes());
        hasher.update(part);
    }
    hasher.finalize().to_hex().to_string()
}

fn canonical_json_bytes(input: &str) -> Vec<u8> {
    serde_json::from_str::<JsonValue>(input)
        .and_then(|value| serde_json::to_vec(&value))
        .unwrap_or_else(|_| input.as_bytes().to_vec())
}

fn initial_metrics(policy: &MonitoringPolicy) -> MonitoringMetrics {
    let mut selected_windows = BTreeMap::new();
    for window in &policy.admitted_windows {
        selected_windows.insert(window.to_string(), 0);
    }
    selected_windows.insert("none".to_owned(), 0);
    let reasons = FALLBACK_REASONS
        .iter()
        .map(|reason| ((*reason).to_owned(), 0))
        .collect();
    MonitoringMetrics {
        dispatch: DispatchMetrics {
            selected_windows,
            ..DispatchMetrics::default()
        },
        fallback: FallbackMetrics {
            reasons,
            ..FallbackMetrics::default()
        },
        ..MonitoringMetrics::default()
    }
}

fn bounded_category(event: &ValidatedPersistentTelemetry, policy: &MonitoringPolicy) -> bool {
    let valid_mode = match event.selected_mode.as_str() {
        SELECTED_MODE_PERSISTENT => event.fallback_reason == FALLBACK_NONE,
        SELECTED_MODE_NORMAL => FALLBACK_REASONS.contains(&event.fallback_reason.as_str()),
        _ => false,
    };
    let valid_window = event
        .selected_window_steps
        .is_none_or(|window| policy.admitted_windows.contains(&window));
    let valid_parity = matches!(
        (
            event.generated_token_parity.as_str(),
            event.fp32_recurrent_state_parity.as_str()
        ),
        (TELEMETRY_PASSED, TELEMETRY_PASSED) | (TELEMETRY_NOT_APPLICABLE, TELEMETRY_NOT_APPLICABLE)
    );
    let valid_cleanup = matches!(
        event.cleanup_status.as_str(),
        TELEMETRY_CLEANUP_COMPLETE | TELEMETRY_CLEANUP_NOT_APPLICABLE
    );
    let timings = [
        event.trace_setup_seconds,
        event.trace_replay_seconds,
        event.cleanup_seconds,
    ];
    valid_mode
        && valid_window
        && valid_parity
        && valid_cleanup
        && timings.iter().all(|timing| {
            timing.is_finite() && *timing >= 0.0 && *timing <= policy.maximum_timing_seconds
        })
}

fn raw_safety_signals(input: &str, policy: &MonitoringPolicy) -> RawSafetySignals {
    let Ok(value) = serde_json::from_str::<JsonValue>(input) else {
        return RawSafetySignals::default();
    };
    let telemetry = value.get("persistent_decode_telemetry");
    let generated_parity = telemetry
        .and_then(|item| item.pointer("/parity/generated_tokens"))
        .and_then(JsonValue::as_str);
    let state_parity = telemetry
        .and_then(|item| item.pointer("/parity/fp32_recurrent_state"))
        .and_then(JsonValue::as_str);
    let cleanup = telemetry
        .and_then(|item| item.get("cleanup_status"))
        .and_then(JsonValue::as_str);
    let selected_mode = telemetry
        .and_then(|item| item.pointer("/dispatch/selected_mode"))
        .and_then(JsonValue::as_str);
    let selected_window = telemetry
        .and_then(|item| item.pointer("/dispatch/selected_window_steps"))
        .and_then(JsonValue::as_u64)
        .and_then(|window| u32::try_from(window).ok());
    let window_admitted = telemetry
        .and_then(|item| item.pointer("/eligibility/window_admitted"))
        .and_then(JsonValue::as_bool);
    let replay_seconds = telemetry
        .and_then(|item| item.pointer("/timing/trace_replay_seconds"))
        .and_then(JsonValue::as_f64);
    let timed_out = telemetry
        .and_then(|item| item.pointer("/terminal_outcomes/timed_out"))
        .and_then(JsonValue::as_u64)
        .unwrap_or(0);
    let failed = telemetry
        .and_then(|item| item.pointer("/terminal_outcomes/failed"))
        .and_then(JsonValue::as_u64)
        .unwrap_or(0);
    RawSafetySignals {
        parity_failure: u64::from(
            generated_parity == Some("failed") || state_parity == Some("failed"),
        ),
        cleanup_failure: u64::from(cleanup == Some("incomplete")),
        replay_timeout: timed_out.saturating_add(u64::from(replay_seconds.is_some_and(
            |seconds| !seconds.is_finite() || seconds > policy.maximum_timing_seconds,
        ))),
        terminal_failure: failed,
        unsupported_admission: u64::from(
            selected_mode == Some(SELECTED_MODE_PERSISTENT)
                && (window_admitted != Some(true)
                    || selected_window
                        .is_none_or(|window| !policy.admitted_windows.contains(&window))),
        ),
    }
}

fn increment_timing(metric: &mut TimingMetric, value: f64) {
    metric.sum_seconds += value;
    metric.maximum_seconds = metric.maximum_seconds.max(value);
}

fn increment_accepted_metrics(
    metrics: &mut MonitoringMetrics,
    event: &ValidatedPersistentTelemetry,
) {
    metrics.eligibility.window_admitted += u64::from(event.window_admitted);
    metrics.eligibility.workload_eligible += u64::from(event.workload_eligible);
    metrics.eligibility.bindings_validated += u64::from(event.bindings_validated);
    metrics.eligibility.profile_physically_admitted += u64::from(event.profile_physically_admitted);
    metrics.eligibility.candidate_requested += u64::from(event.candidate_requested);
    match event.selected_mode.as_str() {
        SELECTED_MODE_PERSISTENT => metrics.dispatch.persistent_trace += 1,
        SELECTED_MODE_NORMAL => metrics.dispatch.normal_dispatch += 1,
        _ => unreachable!("bounded category validation rejected the selected mode"),
    }
    let window_key = event
        .selected_window_steps
        .map_or_else(|| "none".to_owned(), |window| window.to_string());
    *metrics
        .dispatch
        .selected_windows
        .get_mut(&window_key)
        .expect("bounded category validation rejected the selected window") += 1;
    if event.fallback_reason != FALLBACK_NONE {
        metrics.fallback.count += 1;
        *metrics
            .fallback
            .reasons
            .get_mut(&event.fallback_reason)
            .expect("bounded category validation rejected the fallback reason") += 1;
    }
    metrics.terminal.completed += u64::from(event.terminal_completed);
    metrics.terminal.cancelled += u64::from(event.terminal_cancelled);
    metrics.terminal.timed_out += u64::from(event.terminal_timed_out);
    metrics.terminal.failed += u64::from(event.terminal_failed);
    if event.generated_token_parity == TELEMETRY_PASSED {
        metrics.parity.passed += 1;
    } else {
        metrics.parity.not_applicable += 1;
    }
    if event.cleanup_status == TELEMETRY_CLEANUP_COMPLETE {
        metrics.cleanup.complete += 1;
    } else {
        metrics.cleanup.not_applicable += 1;
    }
    increment_timing(&mut metrics.timing.trace_setup, event.trace_setup_seconds);
    increment_timing(&mut metrics.timing.trace_replay, event.trace_replay_seconds);
    increment_timing(&mut metrics.timing.cleanup, event.cleanup_seconds);
}

fn push_alert(
    alerts: &mut Vec<MonitoringAlert>,
    code: &str,
    severity: AlertSeverity,
    count: u64,
    action: &str,
) {
    if count > 0 {
        alerts.push(MonitoringAlert {
            code: code.to_owned(),
            severity,
            count,
            action: action.to_owned(),
        });
    }
}

// r[impl tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.collect]
// r[impl tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.critical]
// r[impl tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.fallback]
pub fn collect_persistent_decode_monitoring(
    policy_json: &str,
    event_json: &[String],
) -> Result<PersistentDecodeMonitoringReceipt, MonitoringError> {
    let policy = parse_monitoring_policy(policy_json)?;
    if event_json.is_empty() {
        return Err(MonitoringError::new(
            "empty_monitoring_batch",
            "monitoring requires at least one worker receipt",
        ));
    }
    if event_json.len() > policy.maximum_batch_events {
        return Err(MonitoringError::new(
            "monitoring_batch_too_large",
            "worker receipt count exceeds the monitoring policy",
        ));
    }
    if event_json
        .iter()
        .any(|event| event.len() > policy.maximum_event_bytes)
    {
        return Err(MonitoringError::new(
            "monitoring_event_too_large",
            "worker receipt exceeds the monitoring policy byte limit",
        ));
    }
    let canonical_policy = serde_json::to_vec(&policy)
        .map_err(|error| MonitoringError::new("invalid_monitoring_policy", error.to_string()))?;
    let policy_blake3 = domain_hash(&[b"policy", &canonical_policy]);
    let mut event_blake3 = Vec::with_capacity(event_json.len());
    let mut seen_events = BTreeSet::new();
    let mut metrics = initial_metrics(&policy);
    let mut accepted_count = 0_u64;
    let mut rejected_count = 0_u64;
    let mut invalid_count = 0_u64;
    let mut duplicate_count = 0_u64;
    let mut safety = RawSafetySignals::default();

    for event in event_json {
        let canonical_event = canonical_json_bytes(event);
        let event_hash = domain_hash(&[b"event", &canonical_event]);
        event_blake3.push(event_hash.clone());
        if !seen_events.insert(event_hash) {
            duplicate_count += 1;
            rejected_count += 1;
            continue;
        }
        let raw_signals = raw_safety_signals(event, &policy);
        safety.add(raw_signals);
        match validated_persistent_telemetry(event) {
            Ok(validated) if bounded_category(&validated, &policy) => {
                increment_accepted_metrics(&mut metrics, &validated);
                accepted_count += 1;
            }
            Ok(_) | Err(_) => {
                invalid_count += 1;
                rejected_count += 1;
            }
        }
    }

    if accepted_count > 0 {
        metrics.fallback.rate = metrics.fallback.count as f64 / accepted_count as f64;
    }
    let event_count = u64::try_from(event_json.len())
        .map_err(|error| MonitoringError::new("monitoring_batch_too_large", error.to_string()))?;
    if accepted_count + rejected_count != event_count {
        return Err(MonitoringError::new(
            "monitoring_count_mismatch",
            "accepted and rejected event counts do not cover the batch",
        ));
    }
    let event_hash_bytes = event_blake3
        .iter()
        .map(String::as_bytes)
        .collect::<Vec<_>>();
    let mut batch_parts = Vec::with_capacity(event_hash_bytes.len() + 2);
    batch_parts.push(b"batch".as_slice());
    batch_parts.push(policy_blake3.as_bytes());
    batch_parts.extend(event_hash_bytes);
    let batch_blake3 = domain_hash(&batch_parts);

    let mut alerts = Vec::new();
    push_alert(
        &mut alerts,
        ALERT_INVALID_TELEMETRY,
        AlertSeverity::Critical,
        invalid_count,
        &policy.critical_action,
    );
    push_alert(
        &mut alerts,
        ALERT_DUPLICATE_EVENT,
        AlertSeverity::Critical,
        duplicate_count,
        &policy.critical_action,
    );
    push_alert(
        &mut alerts,
        ALERT_PARITY_FAILURE,
        AlertSeverity::Critical,
        safety.parity_failure,
        &policy.critical_action,
    );
    push_alert(
        &mut alerts,
        ALERT_CLEANUP_FAILURE,
        AlertSeverity::Critical,
        safety.cleanup_failure,
        &policy.critical_action,
    );
    push_alert(
        &mut alerts,
        ALERT_REPLAY_TIMEOUT,
        AlertSeverity::Critical,
        safety.replay_timeout,
        &policy.critical_action,
    );
    push_alert(
        &mut alerts,
        ALERT_TERMINAL_FAILURE,
        AlertSeverity::Critical,
        safety.terminal_failure,
        &policy.critical_action,
    );
    push_alert(
        &mut alerts,
        ALERT_UNSUPPORTED_ADMISSION,
        AlertSeverity::Critical,
        safety.unsupported_admission,
        &policy.critical_action,
    );
    let minimum_samples = u64::try_from(policy.minimum_fallback_samples)
        .map_err(|error| MonitoringError::new("invalid_monitoring_policy", error.to_string()))?;
    if accepted_count >= minimum_samples && metrics.fallback.rate > policy.maximum_fallback_rate {
        push_alert(
            &mut alerts,
            ALERT_FALLBACK_RATE,
            AlertSeverity::Warning,
            metrics.fallback.count,
            &policy.warning_action,
        );
    }
    let has_critical = alerts
        .iter()
        .any(|alert| alert.severity == AlertSeverity::Critical);
    let has_warning = alerts
        .iter()
        .any(|alert| alert.severity == AlertSeverity::Warning);
    let verdict = if has_critical {
        MonitoringVerdict::Critical
    } else if has_warning {
        MonitoringVerdict::Warning
    } else {
        MonitoringVerdict::Clean
    };
    let persistent_admission_recommendation = if has_critical {
        PersistentAdmissionRecommendation::Disable
    } else {
        PersistentAdmissionRecommendation::Retain
    };
    Ok(PersistentDecodeMonitoringReceipt {
        schema_version: MONITORING_SCHEMA_VERSION,
        event_kind: MONITORING_EVENT_KIND.to_owned(),
        hash_algorithm: "blake3".to_owned(),
        policy_id: policy.policy_id,
        policy_blake3,
        source: MonitoringSourceBinding {
            event_count,
            accepted_event_count: accepted_count,
            rejected_event_count: rejected_count,
            event_blake3,
            batch_blake3,
        },
        metrics,
        alerts,
        verdict,
        persistent_admission_recommendation,
        non_claims: NON_CLAIMS.iter().map(|claim| (*claim).to_owned()).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const POLICY_JSON: &str = r#"{
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
    const PERSISTENT_RECEIPT: &str =
        include_str!("../fixtures/monitoring/physical-persistent-telemetry-receipt.json");
    const FALLBACK_RECEIPT: &str =
        include_str!("../fixtures/monitoring/physical-fallback-telemetry-receipt.json");
    const TERMINAL_RECEIPT: &str =
        include_str!("../fixtures/monitoring/physical-terminal-cancel-telemetry-receipt.json");
    const EXPECTED_MINIMUM_SAMPLES: usize = 8;
    const EXPECTED_MAXIMUM_BATCH_EVENTS: usize = 256;

    fn alert_codes(receipt: &PersistentDecodeMonitoringReceipt) -> Vec<&str> {
        receipt
            .alerts
            .iter()
            .map(|alert| alert.code.as_str())
            .collect()
    }

    fn unique_fallback_receipt(ordinal: usize) -> String {
        let mut value: JsonValue =
            serde_json::from_str(FALLBACK_RECEIPT).expect("fallback fixture");
        value["monitoring_fixture_ordinal"] = serde_json::json!(ordinal);
        serde_json::to_string(&value).expect("serialize fallback fixture")
    }

    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.collect]
    #[test]
    fn collects_clean_batch_deterministically() {
        let events = vec![PERSISTENT_RECEIPT.to_owned(), FALLBACK_RECEIPT.to_owned()];
        let first =
            collect_persistent_decode_monitoring(POLICY_JSON, &events).expect("first batch");
        let second =
            collect_persistent_decode_monitoring(POLICY_JSON, &events).expect("second batch");
        assert_eq!(first, second);
        assert_eq!(first.verdict, MonitoringVerdict::Clean);
        assert_eq!(first.source.accepted_event_count, 2);
        assert_eq!(first.metrics.dispatch.persistent_trace, 1);
        assert_eq!(first.metrics.dispatch.normal_dispatch, 1);
        assert_eq!(first.metrics.fallback.count, 1);
        assert!(first.alerts.is_empty());
    }

    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.fallback]
    #[test]
    fn warns_after_fallback_threshold() {
        let events = (0..EXPECTED_MINIMUM_SAMPLES)
            .map(unique_fallback_receipt)
            .collect::<Vec<_>>();
        let receipt = collect_persistent_decode_monitoring(POLICY_JSON, &events)
            .expect("fallback warning batch");
        assert_eq!(receipt.verdict, MonitoringVerdict::Warning);
        assert_eq!(alert_codes(&receipt), [ALERT_FALLBACK_RATE]);
        assert_eq!(
            receipt.persistent_admission_recommendation,
            PersistentAdmissionRecommendation::Retain
        );
    }

    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.critical]
    #[test]
    fn classifies_critical_safety_signals() {
        let mut parity: JsonValue = serde_json::from_str(PERSISTENT_RECEIPT).expect("fixture");
        parity["persistent_decode_telemetry"]["parity"]["generated_tokens"] =
            serde_json::json!("failed");
        let mut cleanup: JsonValue = serde_json::from_str(PERSISTENT_RECEIPT).expect("fixture");
        cleanup["persistent_decode_telemetry"]["cleanup_status"] = serde_json::json!("incomplete");
        let mut timeout: JsonValue = serde_json::from_str(TERMINAL_RECEIPT).expect("fixture");
        timeout["persistent_decode_telemetry"]["terminal_outcomes"]["cancelled"] =
            serde_json::json!(0);
        timeout["persistent_decode_telemetry"]["terminal_outcomes"]["timed_out"] =
            serde_json::json!(1);
        let mut failure: JsonValue = serde_json::from_str(TERMINAL_RECEIPT).expect("fixture");
        failure["persistent_decode_telemetry"]["terminal_outcomes"]["cancelled"] =
            serde_json::json!(0);
        failure["persistent_decode_telemetry"]["terminal_outcomes"]["failed"] =
            serde_json::json!(1);
        let mut unsupported: JsonValue = serde_json::from_str(PERSISTENT_RECEIPT).expect("fixture");
        unsupported["persistent_decode_telemetry"]["dispatch"]["selected_window_steps"] =
            serde_json::json!(8);
        let events = vec![
            serde_json::to_string(&parity).expect("parity"),
            serde_json::to_string(&cleanup).expect("cleanup"),
            serde_json::to_string(&timeout).expect("timeout"),
            serde_json::to_string(&failure).expect("failure"),
            serde_json::to_string(&unsupported).expect("unsupported"),
            "{".to_owned(),
        ];
        let receipt = collect_persistent_decode_monitoring(POLICY_JSON, &events)
            .expect("critical safety batch");
        assert_eq!(receipt.verdict, MonitoringVerdict::Critical);
        assert_eq!(
            receipt.persistent_admission_recommendation,
            PersistentAdmissionRecommendation::Disable
        );
        let codes = alert_codes(&receipt);
        assert!(codes.contains(&ALERT_INVALID_TELEMETRY));
        assert!(codes.contains(&ALERT_PARITY_FAILURE));
        assert!(codes.contains(&ALERT_CLEANUP_FAILURE));
        assert!(codes.contains(&ALERT_REPLAY_TIMEOUT));
        assert!(codes.contains(&ALERT_TERMINAL_FAILURE));
        assert!(codes.contains(&ALERT_UNSUPPORTED_ADMISSION));
    }

    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.invalid]
    #[test]
    fn rejects_unbounded_inputs_and_alerts_on_duplicates() {
        let empty = collect_persistent_decode_monitoring(POLICY_JSON, &[])
            .expect_err("empty batch must fail");
        assert_eq!(empty.code, "empty_monitoring_batch");

        let oversized = (0..=EXPECTED_MAXIMUM_BATCH_EVENTS)
            .map(unique_fallback_receipt)
            .collect::<Vec<_>>();
        let error = collect_persistent_decode_monitoring(POLICY_JSON, &oversized)
            .expect_err("oversized batch must fail");
        assert_eq!(error.code, "monitoring_batch_too_large");

        let duplicate = vec![PERSISTENT_RECEIPT.to_owned(), PERSISTENT_RECEIPT.to_owned()];
        let receipt = collect_persistent_decode_monitoring(POLICY_JSON, &duplicate)
            .expect("duplicate alert receipt");
        assert_eq!(receipt.verdict, MonitoringVerdict::Critical);
        assert!(alert_codes(&receipt).contains(&ALERT_DUPLICATE_EVENT));

        let mut high_cardinality: JsonValue =
            serde_json::from_str(PERSISTENT_RECEIPT).expect("fixture");
        high_cardinality["persistent_decode_telemetry"]["request_id"] =
            serde_json::json!("forbidden");
        let receipt = collect_persistent_decode_monitoring(
            POLICY_JSON,
            &[serde_json::to_string(&high_cardinality).expect("high-cardinality fixture")],
        )
        .expect("invalid telemetry alert receipt");
        assert!(alert_codes(&receipt).contains(&ALERT_INVALID_TELEMETRY));
    }

    // r[verify tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.policy]
    #[test]
    fn rejects_invalid_policy() {
        let mut invalid: JsonValue = serde_json::from_str(POLICY_JSON).expect("policy fixture");
        invalid["maximum_fallback_rate"] = serde_json::json!(1.25);
        let error = collect_persistent_decode_monitoring(
            &serde_json::to_string(&invalid).expect("invalid policy"),
            &[PERSISTENT_RECEIPT.to_owned()],
        )
        .expect_err("invalid rate must fail");
        assert_eq!(error.code, "invalid_monitoring_policy");

        invalid = serde_json::from_str(POLICY_JSON).expect("policy fixture");
        invalid["maximum_fallback_rate"] = serde_json::json!("NaN");
        let error = collect_persistent_decode_monitoring(
            &serde_json::to_string(&invalid).expect("non-finite policy"),
            &[PERSISTENT_RECEIPT.to_owned()],
        )
        .expect_err("non-finite rate must fail");
        assert_eq!(error.code, "invalid_monitoring_policy");

        invalid = serde_json::from_str(POLICY_JSON).expect("policy fixture");
        invalid["unknown_policy_field"] = serde_json::json!(true);
        let error = collect_persistent_decode_monitoring(
            &serde_json::to_string(&invalid).expect("unknown-field policy"),
            &[PERSISTENT_RECEIPT.to_owned()],
        )
        .expect_err("unknown policy field must fail");
        assert_eq!(error.code, "invalid_monitoring_policy");

        invalid = serde_json::from_str(POLICY_JSON).expect("policy fixture");
        invalid["maximum_event_bytes"] = serde_json::json!(1);
        let error = collect_persistent_decode_monitoring(
            &serde_json::to_string(&invalid).expect("small event policy"),
            &[PERSISTENT_RECEIPT.to_owned()],
        )
        .expect_err("oversized event must fail");
        assert_eq!(error.code, "monitoring_event_too_large");
    }
}
