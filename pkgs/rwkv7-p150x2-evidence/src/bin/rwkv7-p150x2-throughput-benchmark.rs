//! Physical one-card versus two-card throughput benchmark shell.
//!
//! Scheduling and admission stay in the pure throughput module. This binary
//! owns process launch, file retention, and profiler-source checks.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use rwkv7_p150x2_evidence::persistent_decode_contract::{
    PersistentReceiptPolicy, validate_persistent_receipt,
};
use rwkv7_p150x2_evidence::throughput::{
    ExecutionEvidence, MachineFactsEvidence, RawThroughputEvidence, RequestMeasurement,
    ThroughputProfile, ThroughputRun, admit_throughput_receipt, deterministic_slot_assignments,
    profile_identity, summarize_throughput,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

const DEVICE_RUNTIME_PATH: Option<&str> = option_env!("RWKV7_DEVICE_RUNTIME");
const DIAGNOSTIC_DEVICE_RUNTIME_PATH: Option<&str> = option_env!("RWKV7_DIAGNOSTIC_DEVICE_RUNTIME");
const TT_SMI_PATH: Option<&str> = option_env!("TT_SMI_PATH");
const MACHINE_FACT_SOURCE: &str = "tt-smi -s --snapshot_no_tty";
const PROGRAM_ARGUMENT_COUNT: usize = 1;
const EXPECTED_ARGUMENT_COUNT: usize = 3;
const FUSED_CANDIDATE_ARGUMENT_COUNT: usize = EXPECTED_ARGUMENT_COUNT + 1;
const FUSED_CANDIDATE_MODE: &str = "--fused-wkv-candidate";
const STALL_CAPTURE_ARGUMENT_COUNT: usize = EXPECTED_ARGUMENT_COUNT + 2;
const STALL_CAPTURE_MODE: &str = "--fused-wkv-stall-capture";
const STALL_CAPTURE_SAMPLE_COUNT_INDEX: usize = 1;
const STALL_CAPTURE_PROFILE_ARGUMENT_INDEX: usize = 2;
const STALL_CAPTURE_REQUESTS_ARGUMENT_INDEX: usize = 3;
const STALL_CAPTURE_OUTPUT_ARGUMENT_INDEX: usize = 4;
const MAXIMUM_STALL_CAPTURE_SAMPLE_COUNT: usize = 16;
const MAXIMUM_STALL_CAPTURE_ATTEMPTS_PER_SAMPLE: usize = 2;
const STALL_CAPTURE_TOPOLOGY: &str = "stall-profiler-dual";
const STALL_PROBE_ENVIRONMENT: &str = "RWKV7_STALL_PROBE_STEPS";
const PRIMARY_THREAD_CPUS_ENVIRONMENT: &str = "RWKV7_PRIMARY_THREAD_CPUS";
const PRIMARY_THREAD_CPU_ENVIRONMENT: &str = "RWKV7_PRIMARY_THREAD_CPU";
const PRIMARY_THREAD_FIFO_PRIORITY_ENVIRONMENT: &str = "RWKV7_PRIMARY_THREAD_FIFO_PRIORITY";
const FETCH_QUEUE_DIAGNOSTICS_ENVIRONMENT: &str = "TT_METAL_FETCH_QUEUE_DIAGNOSTICS";
const PROFILER_PROGRAM_SUPPORT_COUNT_ENVIRONMENT: &str = "TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT";
// TT-Metal defaults to 1,000 programs. Prompt setup exceeded that limit.
// This fortyfold bound fits within the admitted device-memory margin.
const STALL_PROBE_PROFILER_PROGRAM_SUPPORT_COUNT: usize = 40_000;
const PROFILER_GENERATED_TOKEN_COUNT: u32 = 1;
const PROFILER_WARMUP_TOKEN_COUNT: usize = 1;
const ORDER_BALANCE_PERIOD: usize = 2;
const PROFILE_ARGUMENT_INDEX: usize = 0;
const REQUESTS_ARGUMENT_INDEX: usize = 1;
const OUTPUT_ARGUMENT_INDEX: usize = 2;
const SCHEMA_VERSION: u32 = 1;
const ONE_CARD_IDS: [usize; 1] = [0];
const TWO_CARD_IDS: [usize; 2] = [0, 1];
const FILTERED_WORKER_LOGICAL_CARD_ID: usize = 0;
const PROFILER_RELATIVE_PATH: &str = "profiler/.logs/profile_log_device.csv";
const PROFILER_SUMMARY_FILENAME: &str = "profiler-attribution.csv";
const PROFILER_PROGRAM_SIGNATURE_FILENAME: &str = "profiler-program-signature.json";
const PROFILER_HEADER_LINE_COUNT: usize = 2;
const PROFILER_CORE_X_INDEX: usize = 1;
const PROFILER_CORE_Y_INDEX: usize = 2;
const PROFILER_PROCESSOR_INDEX: usize = 3;
const PROFILER_TIMER_INDEX: usize = 4;
const PROFILER_CYCLE_INDEX: usize = 5;
const PROFILER_ZONE_NAME_INDEX: usize = 10;
const PROFILER_EVENT_TYPE_INDEX: usize = 11;
const PROFILER_MINIMUM_FIELD_COUNT: usize = PROFILER_EVENT_TYPE_INDEX + 1;
const PROFILER_ZONE_START: &str = "ZONE_START";
const PROFILER_ZONE_END: &str = "ZONE_END";
const PROFILER_DROPPED_MARKER: &str = "Profiler DRAM buffers were full, markers were dropped!";
const WORKER_STDOUT_FILENAME: &str = "stdout.log";
const WORKER_STDERR_FILENAME: &str = "stderr.log";
const CPU_REFERENCE_METHOD: &str = "pinned_fp32_full_sequence_greedy_argmax";
const KIBIBYTE: usize = 1024;
const MAX_PROFILER_LINE_BYTES: usize = KIBIBYTE * KIBIBYTE;
const PROFILER_READER_BUFFER_MEBIBYTES: usize = 8;
const PROFILER_READER_BUFFER_BYTES: usize = PROFILER_READER_BUFFER_MEBIBYTES * KIBIBYTE * KIBIBYTE;
// r[impl tenstorrent.rwkv7_p150x2.scalable_card_local_batching.valid]
// These match the exported `throughput-capacity-profile.ncl` memory bounds.
const WORKER_CAPACITY_TEMPORARY_BYTES_PER_SLOT: u64 = 5 * 1024 * 1024;
const WORKER_CAPACITY_DEVICE_BUDGET_BYTES: u64 = 3 * 1024 * 1024 * 1024;
const WORKER_CAPACITY_SAFETY_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
const WORKER_INITIALIZATION_READY_BARRIER_NAME: &str = "initialization-ready.barrier";
const WORKER_WARMUP_START_BARRIER_NAME: &str = "warmup-start.barrier";
const WORKER_TIMED_READY_BARRIER_NAME: &str = "timed-ready.barrier";
const WORKER_TIMED_START_BARRIER_NAME: &str = "timed-start.barrier";
const NORMAL_WORKER_BARRIER_TIMEOUT_SECONDS: u64 = 120;
const PERSISTENT_WORKER_BARRIER_TIMEOUT_SECONDS: u64 = 600;
const PERSISTENT_DECODE_CANDIDATE_ENVIRONMENT: &str = "RWKV7_PERSISTENT_DECODE_CANDIDATE";
const PERSISTENT_DECODE_AB_ENVIRONMENT: &str = "RWKV7_PERSISTENT_DECODE_AB";
const WORKER_BARRIER_POLL_INTERVAL: Duration = Duration::from_millis(1);
const MAXIMUM_WORKER_START_SKEW_SECONDS: f64 = 0.010;
const WORKER_ELAPSED_TOLERANCE_SECONDS: f64 = 0.001;
const WALL_CLOCK_CORRELATION_TOLERANCE_SECONDS: f64 = 0.050;
const BOUNDARY_TIMING_TOLERANCE_SECONDS: f64 = 0.0001;
const MAXIMUM_FETCH_QUEUE_WAIT_CHECKS_PER_RESERVE: u64 = 2;
const RWKV7_LAYER_COUNT: usize = 12;
const NANOSECONDS_PER_SECOND: f64 = 1_000_000_000.0;
const MINIMUM_HARDWARE_COUNTER_RUNNING_FRACTION: f64 = 0.99;
const HARDWARE_COUNTER_SCOPE: &str = "primary_worker_thread_timed_decode";
const ADMITTED_PRIMARY_THREAD_CPUS: [usize; 2] = [2, 6];
const ADMITTED_PRIMARY_THREAD_FIFO_PRIORITY: u32 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WkvPolicy {
    Production,
    FusedCandidate,
    GenericReference,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PersistentDispatchControl {
    Inherit,
    Normal,
    Candidate,
}

impl PersistentDispatchControl {
    fn as_str(self) -> &'static str {
        match self {
            Self::Inherit => "inherit",
            Self::Normal => "normal_dispatch",
            Self::Candidate => "persistent_trace",
        }
    }
}

impl WkvPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Production => "production",
            Self::FusedCandidate => "fused-candidate",
            Self::GenericReference => "generic-reference",
        }
    }
}

fn production_worker_physically_admitted(
    policy: WkvPolicy,
    card_id: usize,
    primary_thread_cpu_affinity: Option<usize>,
    primary_thread_fifo_priority: Option<u32>,
) -> bool {
    policy == WkvPolicy::Production
        && ADMITTED_PRIMARY_THREAD_CPUS.get(card_id).copied() == primary_thread_cpu_affinity
        && primary_thread_fifo_priority == Some(ADMITTED_PRIMARY_THREAD_FIFO_PRIORITY)
}

fn production_run_physically_admitted(
    policy: WkvPolicy,
    primary_thread_cpu_affinities: Option<&[usize]>,
    primary_thread_fifo_priority: Option<u32>,
) -> bool {
    policy == WkvPolicy::Production
        && primary_thread_cpu_affinities == Some(ADMITTED_PRIMARY_THREAD_CPUS.as_slice())
        && primary_thread_fifo_priority == Some(ADMITTED_PRIMARY_THREAD_FIFO_PRIORITY)
}

fn profiler_program_support_count(
    collect_profiler: bool,
    stall_probe: bool,
) -> Result<Option<usize>, String> {
    if stall_probe && !collect_profiler {
        return Err("stall probe requires device-profiler collection".to_owned());
    }
    Ok(stall_probe.then_some(STALL_PROBE_PROFILER_PROGRAM_SUPPORT_COUNT))
}

fn parse_stall_capture_sample_count(arguments: &[String]) -> Result<Option<usize>, String> {
    if arguments.first().map(String::as_str) != Some(STALL_CAPTURE_MODE) {
        return Ok(None);
    }
    if arguments.len() != STALL_CAPTURE_ARGUMENT_COUNT {
        return Err("usage: rwkv7-p150x2-throughput-benchmark --fused-wkv-stall-capture SAMPLE_COUNT PROFILE_JSON REQUESTS_JSON NEW_OUTPUT_ROOT".to_owned());
    }
    let sample_count = arguments[STALL_CAPTURE_SAMPLE_COUNT_INDEX]
        .parse::<usize>()
        .map_err(|_| "stall capture sample count must be an integer".to_owned())?;
    if sample_count == 0 || sample_count > MAXIMUM_STALL_CAPTURE_SAMPLE_COUNT {
        return Err(format!(
            "stall capture sample count must be within 1..={MAXIMUM_STALL_CAPTURE_SAMPLE_COUNT}"
        ));
    }
    Ok(Some(sample_count))
}

fn parse_primary_thread_cpus(value: Option<&str>) -> Result<Option<Vec<usize>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let cpus = value
        .split(',')
        .map(|field| {
            field
                .parse::<usize>()
                .map_err(|_| "primary thread CPU must be an unsigned integer".to_owned())
        })
        .collect::<Result<Vec<_>, _>>()?;
    if cpus.len() != TWO_CARD_IDS.len() {
        return Err(format!(
            "primary thread CPU list must contain exactly {} entries",
            TWO_CARD_IDS.len()
        ));
    }
    if cpus[0] == cpus[1] {
        return Err("primary threads must use distinct CPUs".to_owned());
    }
    Ok(Some(cpus))
}

fn primary_thread_cpus() -> Result<Option<Vec<usize>>, String> {
    match std::env::var(PRIMARY_THREAD_CPUS_ENVIRONMENT) {
        Ok(value) => parse_primary_thread_cpus(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_primary_thread_cpus(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("primary thread CPU list is not UTF-8".to_owned())
        }
    }
}

fn parse_primary_thread_fifo_priority(value: Option<&str>) -> Result<Option<u32>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let priority = value
        .parse::<u32>()
        .map_err(|_| "primary thread FIFO priority must be an unsigned integer".to_owned())?;
    if priority == 0 {
        return Err("primary thread FIFO priority must be positive".to_owned());
    }
    Ok(Some(priority))
}

fn primary_thread_fifo_priority() -> Result<Option<u32>, String> {
    match std::env::var(PRIMARY_THREAD_FIFO_PRIORITY_ENVIRONMENT) {
        Ok(value) => parse_primary_thread_fifo_priority(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_primary_thread_fifo_priority(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err("primary thread FIFO priority is not UTF-8".to_owned())
        }
    }
}

fn validate_primary_thread_scheduling(
    primary_thread_cpu_affinity: Option<usize>,
    primary_thread_fifo_priority: Option<u32>,
) -> Result<(), String> {
    if primary_thread_fifo_priority.is_some() && primary_thread_cpu_affinity.is_none() {
        return Err("primary thread FIFO scheduling requires CPU affinity".to_owned());
    }
    Ok(())
}

fn parse_wkv_policy(arguments: &[String]) -> Result<(WkvPolicy, usize), String> {
    if arguments.len() == EXPECTED_ARGUMENT_COUNT {
        return Ok((WkvPolicy::Production, PROFILE_ARGUMENT_INDEX));
    }
    if arguments.len() == FUSED_CANDIDATE_ARGUMENT_COUNT
        && arguments[PROFILE_ARGUMENT_INDEX] == FUSED_CANDIDATE_MODE
    {
        return Ok((WkvPolicy::FusedCandidate, PROFILE_ARGUMENT_INDEX + 1));
    }
    Err("usage: rwkv7-p150x2-throughput-benchmark [--fused-wkv-candidate] PROFILE_JSON REQUESTS_JSON NEW_OUTPUT_ROOT".to_owned())
}

fn order_balanced_wkv_policies(sample_index: usize) -> [WkvPolicy; ORDER_BALANCE_PERIOD] {
    if sample_index.is_multiple_of(ORDER_BALANCE_PERIOD) {
        return [WkvPolicy::GenericReference, WkvPolicy::FusedCandidate];
    }
    [WkvPolicy::FusedCandidate, WkvPolicy::GenericReference]
}

fn order_balanced_persistent_dispatches(
    sample_index: usize,
) -> [PersistentDispatchControl; ORDER_BALANCE_PERIOD] {
    if sample_index.is_multiple_of(ORDER_BALANCE_PERIOD) {
        return [
            PersistentDispatchControl::Normal,
            PersistentDispatchControl::Candidate,
        ];
    }
    [
        PersistentDispatchControl::Candidate,
        PersistentDispatchControl::Normal,
    ]
}

fn paired_speedups(baseline_rates: &[f64], candidate_rates: &[f64]) -> Result<Vec<f64>, String> {
    if baseline_rates.is_empty() || baseline_rates.len() != candidate_rates.len() {
        return Err("A/B rates must be non-empty and have equal lengths".to_owned());
    }
    baseline_rates
        .iter()
        .zip(candidate_rates)
        .map(|(baseline, candidate)| {
            if !baseline.is_finite()
                || !candidate.is_finite()
                || *baseline <= 0.0
                || *candidate <= 0.0
            {
                return Err("A/B rates must be finite and positive".to_owned());
            }
            Ok(candidate / baseline)
        })
        .collect()
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkRequest {
    request_id: String,
    prompt_token_ids: Vec<u32>,
    max_new_tokens: u32,
    expected_generated_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct BenchmarkWorkload {
    schema_version: u32,
    cpu_reference_profile_blake3: String,
    cpu_reference_method: String,
    requests: Vec<BenchmarkRequest>,
}

#[derive(Clone, Debug, Serialize)]
struct WorkerRequest {
    schema_version: u32,
    run_id: String,
    capacity: WorkerCapacity,
    requests: Vec<WorkerRequestEntry>,
}

#[derive(Clone, Debug, Serialize)]
struct WorkerCapacity {
    id: String,
    per_card_slot_count: usize,
    temporary_tensor_bytes_per_slot: u64,
    device_budget_bytes: u64,
    safety_reserve_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
struct WorkerRequestEntry {
    request_id: String,
    prompt_token_ids: Vec<u32>,
    max_new_tokens: u32,
    state_slot: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FetchQueueDiagnostics {
    reserve_calls: u64,
    wait_calls: u64,
    poll_iterations: u64,
    total_wait_nanoseconds: u64,
    maximum_wait_nanoseconds: u64,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct HardwareCounterReading {
    value: u64,
    time_enabled_nanoseconds: u64,
    time_running_nanoseconds: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct HardwareCounterDiagnostics {
    scope: String,
    cycles: HardwareCounterReading,
    instructions: HardwareCounterReading,
    cache_references: HardwareCounterReading,
    cache_misses: HardwareCounterReading,
    aperf: HardwareCounterReading,
    mperf: HardwareCounterReading,
    task_clock_nanoseconds: HardwareCounterReading,
    context_switches: HardwareCounterReading,
    cpu_migrations: HardwareCounterReading,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct HardwareCounterMetrics {
    instructions_per_cycle: f64,
    cache_miss_fraction: f64,
    aperf_mperf_ratio: f64,
    average_core_ghz: f64,
    instruction_rate_giga_per_second: f64,
    scheduled_fraction: f64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct StallProbeStepBoundaryTiming {
    selection_requested: bool,
    model_submission_seconds: f64,
    layer_submission_seconds: Vec<f64>,
    model_submission_thread_cpu_seconds: f64,
    layer_submission_thread_cpu_seconds: Vec<f64>,
    model_submission_process_cpu_seconds: f64,
    layer_submission_process_cpu_seconds: Vec<f64>,
    blocking_readback_seconds: f64,
    host_materialization_seconds: f64,
    readback_validation_seconds: f64,
    total_seconds: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerReceipt {
    schema_version: u32,
    run_id: String,
    physical_device_ids: Vec<usize>,
    visible_physical_device_ids: Vec<usize>,
    logical_device_ids: Vec<usize>,
    physical_hardware: bool,
    profiler_attributed: bool,
    profiler_trace_only: bool,
    elapsed_seconds: f64,
    timed_start_monotonic_ns: u64,
    timed_end_monotonic_ns: u64,
    timed_start_unix_ns: u64,
    timed_end_unix_ns: u64,
    capacity: WorkerReceiptCapacity,
    effective_batch_size: usize,
    queue_occupancy_fraction: f64,
    batch_step_count: usize,
    host_gap_accumulated_seconds: f64,
    primary_thread_cpu_affinity: Option<usize>,
    primary_thread_fifo_priority: Option<u32>,
    stall_probe_step_wall_seconds: Option<Vec<f64>>,
    stall_probe_step_boundary_timings: Option<Vec<StallProbeStepBoundaryTiming>>,
    fetch_queue_diagnostics: Option<FetchQueueDiagnostics>,
    hardware_counter_diagnostics: Option<HardwareCounterDiagnostics>,
    requests: Vec<WorkerRequestMeasurement>,
    execution: WorkerExecution,
    persistent_decode: serde_json::Value,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerReceiptCapacity {
    id: String,
    per_card_slot_count: usize,
    memory_plan_bytes: WorkerReceiptMemoryPlan,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerReceiptMemoryPlan {
    recurrent_state_bytes: u64,
    token_shift_bytes: u64,
    temporary_tensor_bytes: u64,
    safety_reserve_bytes: u64,
    total_required_bytes: u64,
    device_budget_bytes: u64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerRequestMeasurement {
    #[serde(flatten)]
    measurement: RequestMeasurement,
    generated_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerExecution {
    scheduler_mode: String,
    scheduler_tie_break: String,
    replica_mode: String,
    recurrent_state_residency: String,
    timed_region: String,
    warmup_token_count: usize,
    program_cache_stable: bool,
    dynamic_token_and_state_bindings_validated: bool,
    state_readback_during_timing: bool,
    profiler_flush_during_timing: bool,
    control_replay_during_timing: bool,
    artifact_writes_during_timing: bool,
    inspector_during_timing: bool,
    fetch_queue_diagnostics_during_timing: bool,
    hardware_counters_during_timing: bool,
    primary_thread_only_affinity: bool,
    primary_thread_fifo_scheduling: bool,
    synchronized_timed_start: bool,
    wkv_execution_policy: String,
    production_physically_admitted: bool,
}

struct WorkerProcess {
    card_id: usize,
    wave_index: usize,
    root: PathBuf,
    output_path: PathBuf,
    initialization_ready_barrier_path: PathBuf,
    warmup_start_barrier_path: PathBuf,
    timed_ready_barrier_path: PathBuf,
    timed_start_barrier_path: PathBuf,
    primary_thread_cpu_affinity: Option<usize>,
    primary_thread_fifo_priority: Option<u32>,
    persistent_candidate: bool,
    child: std::process::Child,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
struct WorkerTiming {
    start_monotonic_ns: u64,
    end_monotonic_ns: u64,
    elapsed_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SynchronizedWaveTiming {
    elapsed_seconds: f64,
    start_skew_seconds: f64,
}

fn synchronized_wave_timing(timings: &[WorkerTiming]) -> Result<SynchronizedWaveTiming, String> {
    let first = timings
        .first()
        .ok_or_else(|| "synchronized wave has no worker timings".to_owned())?;
    let mut minimum_start = first.start_monotonic_ns;
    let mut maximum_start = first.start_monotonic_ns;
    let mut maximum_end = first.end_monotonic_ns;
    for timing in timings {
        if timing.end_monotonic_ns <= timing.start_monotonic_ns
            || !timing.elapsed_seconds.is_finite()
            || timing.elapsed_seconds <= 0.0
        {
            return Err("worker timing range must be finite and positive".to_owned());
        }
        let measured_elapsed =
            (timing.end_monotonic_ns - timing.start_monotonic_ns) as f64 / NANOSECONDS_PER_SECOND;
        if (measured_elapsed - timing.elapsed_seconds).abs() > WORKER_ELAPSED_TOLERANCE_SECONDS {
            return Err("worker elapsed time differs from its monotonic timing range".to_owned());
        }
        minimum_start = minimum_start.min(timing.start_monotonic_ns);
        maximum_start = maximum_start.max(timing.start_monotonic_ns);
        maximum_end = maximum_end.max(timing.end_monotonic_ns);
    }
    let start_skew_seconds = (maximum_start - minimum_start) as f64 / NANOSECONDS_PER_SECOND;
    if start_skew_seconds > MAXIMUM_WORKER_START_SKEW_SECONDS {
        return Err(format!(
            "worker start skew is {start_skew_seconds}, maximum is {MAXIMUM_WORKER_START_SKEW_SECONDS}"
        ));
    }
    let elapsed_seconds = (maximum_end - minimum_start) as f64 / NANOSECONDS_PER_SECOND;
    if !elapsed_seconds.is_finite() || elapsed_seconds <= 0.0 {
        return Err("synchronized wave elapsed time must be finite and positive".to_owned());
    }
    Ok(SynchronizedWaveTiming {
        elapsed_seconds,
        start_skew_seconds,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProfilerEventType {
    Start,
    End,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ProfilerEvent {
    logical_card_id: usize,
    key: String,
    zone_name: String,
    cycle: u64,
    event_type: ProfilerEventType,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
struct ProfilerZoneAggregate {
    start_count: usize,
    complete_count: usize,
    total_cycles: u64,
    maximum_cycles: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ProfilerProgramSignature {
    schema_version: u32,
    logical_card_id: usize,
    physical_card_id: usize,
    raw_blake3: String,
    data_row_count: usize,
    complete_zone_event_count: usize,
    unmatched_start_count: usize,
    unmatched_end_count: usize,
    total_zone_cycles: u64,
    zones: BTreeMap<String, ProfilerZoneAggregate>,
    program_signature_blake3: String,
}

#[derive(Clone, Debug, Serialize)]
struct StallCaptureWorker {
    physical_card_id: usize,
    elapsed_seconds: f64,
    timed_start_unix_ns: u64,
    timed_end_unix_ns: u64,
    host_gap_accumulated_seconds: f64,
    primary_thread_cpu_affinity: Option<usize>,
    batch_step_count: usize,
    maximum_step_index: usize,
    maximum_step_wall_seconds: f64,
    summed_step_wall_seconds: f64,
    step_wall_seconds: Vec<f64>,
    step_boundary_timings: Vec<StallProbeStepBoundaryTiming>,
    fetch_queue_diagnostics: FetchQueueDiagnostics,
    hardware_counter_diagnostics: HardwareCounterDiagnostics,
    hardware_counter_metrics: HardwareCounterMetrics,
    profiler: ProfilerProgramSignature,
}

#[derive(Clone, Debug, Serialize)]
struct StallCaptureSample {
    sample_index: usize,
    wave_elapsed_seconds: f64,
    start_skew_seconds: f64,
    workers: Vec<StallCaptureWorker>,
}

#[derive(Clone, Debug, Serialize)]
struct StallCaptureBoundaryComparison {
    summed_model_submission_ratio: f64,
    summed_model_submission_thread_cpu_ratio: f64,
    summed_model_submission_process_cpu_ratio: f64,
    summed_blocking_readback_ratio: f64,
    summed_total_ratio: f64,
    layer_submission_ratios: Vec<f64>,
    layer_submission_thread_cpu_ratios: Vec<f64>,
    layer_submission_process_cpu_ratios: Vec<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct StallCaptureFetchQueueComparison {
    normal_total_wait_seconds: f64,
    slow_total_wait_seconds: f64,
    extra_wait_seconds: f64,
    wall_time_extra_seconds: f64,
    extra_wait_fraction_of_wall_time_extra: Option<f64>,
    reserve_call_ratio: f64,
    wait_call_ratio: Option<f64>,
    total_wait_ratio: Option<f64>,
    poll_iteration_ratio: Option<f64>,
    maximum_wait_ratio: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
struct StallCaptureHardwareCounterComparison {
    cycles_ratio: f64,
    instructions_ratio: f64,
    cache_references_ratio: f64,
    cache_misses_ratio: Option<f64>,
    aperf_ratio: f64,
    mperf_ratio: f64,
    task_clock_ratio: f64,
    context_switch_ratio: Option<f64>,
    cpu_migration_ratio: Option<f64>,
    instructions_per_cycle_ratio: f64,
    cache_miss_fraction_ratio: Option<f64>,
    aperf_mperf_ratio_ratio: f64,
    average_core_ghz_ratio: f64,
    instruction_rate_ratio: f64,
    scheduled_fraction_ratio: f64,
}

#[derive(Clone, Debug, Serialize)]
struct StallCaptureCardComparison {
    physical_card_id: usize,
    total_zone_cycle_ratio: f64,
    maximum_step_wall_ratio: f64,
    summed_step_wall_ratio: f64,
    boundary: StallCaptureBoundaryComparison,
    fetch_queue: StallCaptureFetchQueueComparison,
    hardware_counters: StallCaptureHardwareCounterComparison,
    zone_total_cycle_ratios: BTreeMap<String, f64>,
}

#[derive(Clone, Debug, Serialize)]
struct StallCaptureComparison {
    normal_sample_index: usize,
    slow_sample_index: usize,
    wall_slowdown_ratio: f64,
    cards: Vec<StallCaptureCardComparison>,
}

fn wall_clock_elapsed_seconds(start_unix_ns: u64, end_unix_ns: u64) -> Result<f64, String> {
    let elapsed_ns = end_unix_ns
        .checked_sub(start_unix_ns)
        .ok_or_else(|| "worker wall-clock timing range is not positive".to_owned())?;
    if elapsed_ns == 0 {
        return Err("worker wall-clock timing range is not positive".to_owned());
    }
    Ok(elapsed_ns as f64 / NANOSECONDS_PER_SECOND)
}

fn profiler_output_has_dropped_markers(line: &str) -> bool {
    line.contains(PROFILER_DROPPED_MARKER)
}

fn require_complete_profiler_output(worker: &WorkerProcess) -> Result<(), String> {
    for filename in [WORKER_STDOUT_FILENAME, WORKER_STDERR_FILENAME] {
        let path = worker.root.join(filename);
        let file = File::open(&path)
            .map_err(|error| format!("cannot open {}: {error}", path.display()))?;
        for line in BufReader::new(file).lines() {
            let line = line.map_err(|error| format!("cannot read {}: {error}", path.display()))?;
            if profiler_output_has_dropped_markers(&line) {
                return Err(format!(
                    "card {} dropped device-profiler markers; complete zone attribution is unavailable",
                    worker.card_id
                ));
            }
        }
    }
    Ok(())
}

fn maximum_step(values: &[f64]) -> Result<(usize, f64), String> {
    let first = values
        .first()
        .copied()
        .ok_or_else(|| "stall capture has no step timings".to_owned())?;
    if !first.is_finite() || first <= 0.0 {
        return Err("stall capture step timings must be finite and positive".to_owned());
    }
    let mut maximum_index = 0_usize;
    let mut maximum_value = first;
    for (index, value) in values.iter().copied().enumerate().skip(1) {
        if !value.is_finite() || value <= 0.0 {
            return Err("stall capture step timings must be finite and positive".to_owned());
        }
        if value > maximum_value {
            maximum_index = index;
            maximum_value = value;
        }
    }
    Ok((maximum_index, maximum_value))
}

fn validate_fetch_queue_diagnostics(diagnostics: FetchQueueDiagnostics) -> Result<(), String> {
    let maximum_wait_calls = diagnostics
        .reserve_calls
        .checked_mul(MAXIMUM_FETCH_QUEUE_WAIT_CHECKS_PER_RESERVE)
        .ok_or_else(|| "fetch queue diagnostics wait bound overflowed".to_owned())?;
    if diagnostics.reserve_calls == 0 {
        return Err("fetch queue diagnostics contain no reserve calls".to_owned());
    }
    if diagnostics.wait_calls > maximum_wait_calls {
        return Err("fetch queue diagnostics contain too many wait calls".to_owned());
    }
    if diagnostics.poll_iterations < diagnostics.wait_calls {
        return Err("fetch queue diagnostics contain too few poll iterations".to_owned());
    }
    if diagnostics.maximum_wait_nanoseconds > diagnostics.total_wait_nanoseconds {
        return Err("fetch queue diagnostics maximum exceeds total wait".to_owned());
    }
    let wait_values_are_zero = diagnostics.poll_iterations == 0
        && diagnostics.total_wait_nanoseconds == 0
        && diagnostics.maximum_wait_nanoseconds == 0;
    if diagnostics.wait_calls == 0 && !wait_values_are_zero {
        return Err("fetch queue diagnostics contain data without a wait".to_owned());
    }
    if diagnostics.wait_calls > 0
        && (diagnostics.poll_iterations == 0
            || diagnostics.total_wait_nanoseconds == 0
            || diagnostics.maximum_wait_nanoseconds == 0)
    {
        return Err("fetch queue diagnostics omit data for a recorded wait".to_owned());
    }
    Ok(())
}

fn validate_hardware_counter_reading(
    reading: HardwareCounterReading,
    name: &str,
    require_nonzero_value: bool,
) -> Result<(), String> {
    if require_nonzero_value && reading.value == 0 {
        return Err(format!("{name} hardware counter is zero"));
    }
    if reading.time_enabled_nanoseconds == 0
        || reading.time_running_nanoseconds == 0
        || reading.time_running_nanoseconds > reading.time_enabled_nanoseconds
    {
        return Err(format!("{name} hardware counter has invalid running time"));
    }
    let running_fraction =
        reading.time_running_nanoseconds as f64 / reading.time_enabled_nanoseconds as f64;
    if running_fraction < MINIMUM_HARDWARE_COUNTER_RUNNING_FRACTION {
        return Err(format!(
            "{name} hardware counter was multiplexed below the bound"
        ));
    }
    Ok(())
}

fn validate_hardware_counter_diagnostics(
    diagnostics: &HardwareCounterDiagnostics,
) -> Result<(), String> {
    if diagnostics.scope != HARDWARE_COUNTER_SCOPE {
        return Err("hardware counter scope is invalid".to_owned());
    }
    for (name, reading, require_nonzero_value) in [
        ("cycles", diagnostics.cycles, true),
        ("instructions", diagnostics.instructions, true),
        ("cache references", diagnostics.cache_references, true),
        ("cache misses", diagnostics.cache_misses, false),
        ("APERF", diagnostics.aperf, true),
        ("MPERF", diagnostics.mperf, true),
        ("task clock", diagnostics.task_clock_nanoseconds, true),
        ("context switches", diagnostics.context_switches, false),
        ("CPU migrations", diagnostics.cpu_migrations, false),
    ] {
        validate_hardware_counter_reading(reading, name, require_nonzero_value)?;
    }
    if diagnostics.cache_misses.value > diagnostics.cache_references.value {
        return Err("hardware counter cache misses exceed references".to_owned());
    }
    Ok(())
}

fn hardware_counter_metrics(
    diagnostics: &HardwareCounterDiagnostics,
    elapsed_seconds: f64,
) -> Result<HardwareCounterMetrics, String> {
    validate_hardware_counter_diagnostics(diagnostics)?;
    if !elapsed_seconds.is_finite() || elapsed_seconds <= 0.0 {
        return Err("hardware counter elapsed time must be finite and positive".to_owned());
    }
    let instructions_per_cycle =
        diagnostics.instructions.value as f64 / diagnostics.cycles.value as f64;
    let cache_miss_fraction =
        diagnostics.cache_misses.value as f64 / diagnostics.cache_references.value as f64;
    let aperf_mperf_ratio = diagnostics.aperf.value as f64 / diagnostics.mperf.value as f64;
    let average_core_ghz =
        diagnostics.cycles.value as f64 / diagnostics.task_clock_nanoseconds.value as f64;
    let instruction_rate_giga_per_second =
        diagnostics.instructions.value as f64 / diagnostics.task_clock_nanoseconds.value as f64;
    let elapsed_nanoseconds = elapsed_seconds * NANOSECONDS_PER_SECOND;
    let scheduled_fraction = diagnostics.task_clock_nanoseconds.value as f64 / elapsed_nanoseconds;
    let metrics = HardwareCounterMetrics {
        instructions_per_cycle,
        cache_miss_fraction,
        aperf_mperf_ratio,
        average_core_ghz,
        instruction_rate_giga_per_second,
        scheduled_fraction,
    };
    let values = [
        metrics.instructions_per_cycle,
        metrics.cache_miss_fraction,
        metrics.aperf_mperf_ratio,
        metrics.average_core_ghz,
        metrics.instruction_rate_giga_per_second,
        metrics.scheduled_fraction,
    ];
    if values
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || metrics.instructions_per_cycle == 0.0
        || metrics.aperf_mperf_ratio == 0.0
        || metrics.average_core_ghz == 0.0
        || metrics.instruction_rate_giga_per_second == 0.0
        || metrics.scheduled_fraction == 0.0
    {
        return Err("derived hardware counter metrics are invalid".to_owned());
    }
    Ok(metrics)
}

fn optional_counter_ratio(numerator: u64, denominator: u64) -> Option<f64> {
    if denominator == 0 {
        None
    } else {
        Some(numerator as f64 / denominator as f64)
    }
}

fn optional_positive_f64_ratio(numerator: f64, denominator: f64) -> Option<f64> {
    if !numerator.is_finite() || numerator < 0.0 || !denominator.is_finite() || denominator <= 0.0 {
        None
    } else {
        Some(numerator / denominator)
    }
}

fn compare_hardware_counter_diagnostics(
    normal: &HardwareCounterDiagnostics,
    slow: &HardwareCounterDiagnostics,
    normal_elapsed_seconds: f64,
    slow_elapsed_seconds: f64,
) -> Result<StallCaptureHardwareCounterComparison, String> {
    let normal_metrics = hardware_counter_metrics(normal, normal_elapsed_seconds)?;
    let slow_metrics = hardware_counter_metrics(slow, slow_elapsed_seconds)?;
    let counter_ratio = |slow_value: u64, normal_value: u64, name: &str| {
        positive_ratio(slow_value as f64, normal_value as f64, name)
    };
    Ok(StallCaptureHardwareCounterComparison {
        cycles_ratio: counter_ratio(slow.cycles.value, normal.cycles.value, "cycles")?,
        instructions_ratio: counter_ratio(
            slow.instructions.value,
            normal.instructions.value,
            "instructions",
        )?,
        cache_references_ratio: counter_ratio(
            slow.cache_references.value,
            normal.cache_references.value,
            "cache references",
        )?,
        cache_misses_ratio: optional_counter_ratio(
            slow.cache_misses.value,
            normal.cache_misses.value,
        ),
        aperf_ratio: counter_ratio(slow.aperf.value, normal.aperf.value, "APERF")?,
        mperf_ratio: counter_ratio(slow.mperf.value, normal.mperf.value, "MPERF")?,
        task_clock_ratio: counter_ratio(
            slow.task_clock_nanoseconds.value,
            normal.task_clock_nanoseconds.value,
            "task clock",
        )?,
        context_switch_ratio: optional_counter_ratio(
            slow.context_switches.value,
            normal.context_switches.value,
        ),
        cpu_migration_ratio: optional_counter_ratio(
            slow.cpu_migrations.value,
            normal.cpu_migrations.value,
        ),
        instructions_per_cycle_ratio: positive_ratio(
            slow_metrics.instructions_per_cycle,
            normal_metrics.instructions_per_cycle,
            "instructions per cycle",
        )?,
        cache_miss_fraction_ratio: optional_positive_f64_ratio(
            slow_metrics.cache_miss_fraction,
            normal_metrics.cache_miss_fraction,
        ),
        aperf_mperf_ratio_ratio: positive_ratio(
            slow_metrics.aperf_mperf_ratio,
            normal_metrics.aperf_mperf_ratio,
            "APERF MPERF ratio",
        )?,
        average_core_ghz_ratio: positive_ratio(
            slow_metrics.average_core_ghz,
            normal_metrics.average_core_ghz,
            "average core GHz",
        )?,
        instruction_rate_ratio: positive_ratio(
            slow_metrics.instruction_rate_giga_per_second,
            normal_metrics.instruction_rate_giga_per_second,
            "instruction rate",
        )?,
        scheduled_fraction_ratio: positive_ratio(
            slow_metrics.scheduled_fraction,
            normal_metrics.scheduled_fraction,
            "scheduled fraction",
        )?,
    })
}

fn compare_fetch_queue_diagnostics(
    normal: FetchQueueDiagnostics,
    slow: FetchQueueDiagnostics,
    normal_wall_seconds: f64,
    slow_wall_seconds: f64,
) -> Result<StallCaptureFetchQueueComparison, String> {
    validate_fetch_queue_diagnostics(normal)?;
    validate_fetch_queue_diagnostics(slow)?;
    if !normal_wall_seconds.is_finite()
        || normal_wall_seconds <= 0.0
        || !slow_wall_seconds.is_finite()
        || slow_wall_seconds <= 0.0
    {
        return Err("fetch queue comparison wall times must be finite and positive".to_owned());
    }
    let normal_total_wait_seconds = normal.total_wait_nanoseconds as f64 / NANOSECONDS_PER_SECOND;
    let slow_total_wait_seconds = slow.total_wait_nanoseconds as f64 / NANOSECONDS_PER_SECOND;
    let extra_wait_seconds = slow_total_wait_seconds - normal_total_wait_seconds;
    let wall_time_extra_seconds = slow_wall_seconds - normal_wall_seconds;
    let extra_wait_fraction_of_wall_time_extra = if wall_time_extra_seconds > 0.0 {
        Some(extra_wait_seconds / wall_time_extra_seconds)
    } else {
        None
    };
    Ok(StallCaptureFetchQueueComparison {
        normal_total_wait_seconds,
        slow_total_wait_seconds,
        extra_wait_seconds,
        wall_time_extra_seconds,
        extra_wait_fraction_of_wall_time_extra,
        reserve_call_ratio: slow.reserve_calls as f64 / normal.reserve_calls as f64,
        wait_call_ratio: optional_counter_ratio(slow.wait_calls, normal.wait_calls),
        total_wait_ratio: optional_counter_ratio(
            slow.total_wait_nanoseconds,
            normal.total_wait_nanoseconds,
        ),
        poll_iteration_ratio: optional_counter_ratio(slow.poll_iterations, normal.poll_iterations),
        maximum_wait_ratio: optional_counter_ratio(
            slow.maximum_wait_nanoseconds,
            normal.maximum_wait_nanoseconds,
        ),
    })
}

fn validate_step_boundary_timing(timing: &StallProbeStepBoundaryTiming) -> Result<(), String> {
    if timing.layer_submission_seconds.len() != RWKV7_LAYER_COUNT
        || timing.layer_submission_thread_cpu_seconds.len() != RWKV7_LAYER_COUNT
        || timing.layer_submission_process_cpu_seconds.len() != RWKV7_LAYER_COUNT
    {
        return Err("stall boundary timing has the wrong layer count".to_owned());
    }
    let scalar_timings = [
        timing.model_submission_seconds,
        timing.model_submission_thread_cpu_seconds,
        timing.model_submission_process_cpu_seconds,
        timing.blocking_readback_seconds,
        timing.host_materialization_seconds,
        timing.readback_validation_seconds,
        timing.total_seconds,
    ];
    if scalar_timings
        .iter()
        .any(|seconds| !seconds.is_finite() || *seconds < 0.0)
        || timing
            .layer_submission_seconds
            .iter()
            .chain(&timing.layer_submission_thread_cpu_seconds)
            .chain(&timing.layer_submission_process_cpu_seconds)
            .any(|seconds| !seconds.is_finite() || *seconds < 0.0)
    {
        return Err("stall boundary timings must be finite and non-negative".to_owned());
    }
    let summed_layer_seconds = timing.layer_submission_seconds.iter().sum::<f64>();
    if summed_layer_seconds > timing.model_submission_seconds + BOUNDARY_TIMING_TOLERANCE_SECONDS {
        return Err("stall layer timings exceed model submission time".to_owned());
    }
    let classified_seconds = timing.model_submission_seconds
        + timing.blocking_readback_seconds
        + timing.host_materialization_seconds
        + timing.readback_validation_seconds;
    if (classified_seconds - timing.total_seconds).abs() > BOUNDARY_TIMING_TOLERANCE_SECONDS {
        return Err("stall boundary timings do not cover the full step".to_owned());
    }
    if !timing.selection_requested
        && (timing.blocking_readback_seconds != 0.0
            || timing.host_materialization_seconds != 0.0
            || timing.readback_validation_seconds != 0.0)
    {
        return Err("stall non-selection step contains readback timing".to_owned());
    }
    Ok(())
}

fn validate_step_boundary_timings(
    timings: &[StallProbeStepBoundaryTiming],
    expected_step_count: usize,
) -> Result<(), String> {
    if timings.len() != expected_step_count {
        return Err("stall boundary timing count differs from the step count".to_owned());
    }
    for timing in timings {
        validate_step_boundary_timing(timing)?;
    }
    if !timings.iter().any(|timing| timing.selection_requested) {
        return Err("stall boundary timings contain no selection step".to_owned());
    }
    Ok(())
}

fn sum_boundary_field(
    timings: &[StallProbeStepBoundaryTiming],
    field: fn(&StallProbeStepBoundaryTiming) -> f64,
    name: &str,
) -> Result<f64, String> {
    let mut total = 0.0_f64;
    for timing in timings {
        let value = field(timing);
        if !value.is_finite() || value < 0.0 {
            return Err(format!("{name} values must be finite and non-negative"));
        }
        total += value;
    }
    if !total.is_finite() || total <= 0.0 {
        return Err(format!("{name} sum must be finite and positive"));
    }
    Ok(total)
}

fn compare_boundary_timings(
    normal: &[StallProbeStepBoundaryTiming],
    slow: &[StallProbeStepBoundaryTiming],
) -> Result<StallCaptureBoundaryComparison, String> {
    if normal.len() != slow.len() || normal.is_empty() {
        return Err("stall boundary comparisons require equal non-empty steps".to_owned());
    }
    validate_step_boundary_timings(normal, normal.len())?;
    validate_step_boundary_timings(slow, slow.len())?;
    let ratio = |field: fn(&StallProbeStepBoundaryTiming) -> f64, name: &str| {
        positive_ratio(
            sum_boundary_field(slow, field, name)?,
            sum_boundary_field(normal, field, name)?,
            name,
        )
    };
    let mut layer_submission_ratios = Vec::with_capacity(RWKV7_LAYER_COUNT);
    let mut layer_submission_thread_cpu_ratios = Vec::with_capacity(RWKV7_LAYER_COUNT);
    let mut layer_submission_process_cpu_ratios = Vec::with_capacity(RWKV7_LAYER_COUNT);
    for layer_index in 0..RWKV7_LAYER_COUNT {
        let normal_sum = normal
            .iter()
            .map(|timing| timing.layer_submission_seconds[layer_index])
            .sum::<f64>();
        let slow_sum = slow
            .iter()
            .map(|timing| timing.layer_submission_seconds[layer_index])
            .sum::<f64>();
        layer_submission_ratios.push(positive_ratio(
            slow_sum,
            normal_sum,
            "stall layer submission",
        )?);
        let normal_thread_cpu_sum = normal
            .iter()
            .map(|timing| timing.layer_submission_thread_cpu_seconds[layer_index])
            .sum::<f64>();
        let slow_thread_cpu_sum = slow
            .iter()
            .map(|timing| timing.layer_submission_thread_cpu_seconds[layer_index])
            .sum::<f64>();
        layer_submission_thread_cpu_ratios.push(positive_ratio(
            slow_thread_cpu_sum,
            normal_thread_cpu_sum,
            "stall layer thread CPU",
        )?);
        let normal_process_cpu_sum = normal
            .iter()
            .map(|timing| timing.layer_submission_process_cpu_seconds[layer_index])
            .sum::<f64>();
        let slow_process_cpu_sum = slow
            .iter()
            .map(|timing| timing.layer_submission_process_cpu_seconds[layer_index])
            .sum::<f64>();
        layer_submission_process_cpu_ratios.push(positive_ratio(
            slow_process_cpu_sum,
            normal_process_cpu_sum,
            "stall layer process CPU",
        )?);
    }
    Ok(StallCaptureBoundaryComparison {
        summed_model_submission_ratio: ratio(
            |timing| timing.model_submission_seconds,
            "stall model submission",
        )?,
        summed_model_submission_thread_cpu_ratio: ratio(
            |timing| timing.model_submission_thread_cpu_seconds,
            "stall model submission thread CPU",
        )?,
        summed_model_submission_process_cpu_ratio: ratio(
            |timing| timing.model_submission_process_cpu_seconds,
            "stall model submission process CPU",
        )?,
        summed_blocking_readback_ratio: ratio(
            |timing| timing.blocking_readback_seconds,
            "stall blocking readback",
        )?,
        summed_total_ratio: ratio(|timing| timing.total_seconds, "stall boundary total")?,
        layer_submission_ratios,
        layer_submission_thread_cpu_ratios,
        layer_submission_process_cpu_ratios,
    })
}

fn stall_capture_extremes(elapsed_seconds: &[f64]) -> Result<(usize, usize), String> {
    let first = elapsed_seconds
        .first()
        .copied()
        .ok_or_else(|| "stall capture has no samples".to_owned())?;
    if !first.is_finite() || first <= 0.0 {
        return Err("stall capture elapsed times must be finite and positive".to_owned());
    }
    let mut normal_index = 0_usize;
    let mut slow_index = 0_usize;
    for (index, elapsed) in elapsed_seconds.iter().copied().enumerate().skip(1) {
        if !elapsed.is_finite() || elapsed <= 0.0 {
            return Err("stall capture elapsed times must be finite and positive".to_owned());
        }
        if elapsed < elapsed_seconds[normal_index] {
            normal_index = index;
        }
        if elapsed > elapsed_seconds[slow_index] {
            slow_index = index;
        }
    }
    Ok((normal_index, slow_index))
}

fn positive_ratio(numerator: f64, denominator: f64, name: &str) -> Result<f64, String> {
    if !numerator.is_finite() || numerator <= 0.0 || !denominator.is_finite() || denominator <= 0.0
    {
        return Err(format!("{name} values must be finite and positive"));
    }
    Ok(numerator / denominator)
}

fn compare_stall_capture_samples(
    normal: &StallCaptureSample,
    slow: &StallCaptureSample,
) -> Result<StallCaptureComparison, String> {
    if normal.workers.len() != slow.workers.len() || normal.workers.is_empty() {
        return Err("stall capture samples must have equal non-empty worker sets".to_owned());
    }
    let wall_slowdown_ratio = positive_ratio(
        slow.wave_elapsed_seconds,
        normal.wave_elapsed_seconds,
        "stall capture wall time",
    )?;
    let mut cards = Vec::with_capacity(normal.workers.len());
    for (normal_worker, slow_worker) in normal.workers.iter().zip(&slow.workers) {
        if normal_worker.physical_card_id != slow_worker.physical_card_id {
            return Err("stall capture worker ownership changed between samples".to_owned());
        }
        if normal_worker.profiler.zones.len() != slow_worker.profiler.zones.len() {
            return Err("stall capture profiler zone sets changed between samples".to_owned());
        }
        let mut zone_total_cycle_ratios = BTreeMap::new();
        for (zone_name, normal_zone) in &normal_worker.profiler.zones {
            let slow_zone = slow_worker
                .profiler
                .zones
                .get(zone_name)
                .ok_or_else(|| "stall capture profiler zone is missing".to_owned())?;
            zone_total_cycle_ratios.insert(
                zone_name.clone(),
                positive_ratio(
                    slow_zone.total_cycles as f64,
                    normal_zone.total_cycles as f64,
                    "stall capture named zone cycles",
                )?,
            );
        }
        cards.push(StallCaptureCardComparison {
            physical_card_id: normal_worker.physical_card_id,
            total_zone_cycle_ratio: positive_ratio(
                slow_worker.profiler.total_zone_cycles as f64,
                normal_worker.profiler.total_zone_cycles as f64,
                "stall capture zone cycles",
            )?,
            maximum_step_wall_ratio: positive_ratio(
                slow_worker.maximum_step_wall_seconds,
                normal_worker.maximum_step_wall_seconds,
                "stall capture maximum step",
            )?,
            summed_step_wall_ratio: positive_ratio(
                slow_worker.summed_step_wall_seconds,
                normal_worker.summed_step_wall_seconds,
                "stall capture summed steps",
            )?,
            boundary: compare_boundary_timings(
                &normal_worker.step_boundary_timings,
                &slow_worker.step_boundary_timings,
            )?,
            fetch_queue: compare_fetch_queue_diagnostics(
                normal_worker.fetch_queue_diagnostics,
                slow_worker.fetch_queue_diagnostics,
                normal_worker.summed_step_wall_seconds,
                slow_worker.summed_step_wall_seconds,
            )?,
            hardware_counters: compare_hardware_counter_diagnostics(
                &normal_worker.hardware_counter_diagnostics,
                &slow_worker.hardware_counter_diagnostics,
                normal_worker.elapsed_seconds,
                slow_worker.elapsed_seconds,
            )?,
            zone_total_cycle_ratios,
        });
    }
    Ok(StallCaptureComparison {
        normal_sample_index: normal.sample_index,
        slow_sample_index: slow.sample_index,
        wall_slowdown_ratio,
        cards,
    })
}

struct WorkerLaunchRequest<'a> {
    runtime: &'a Path,
    run_root: &'a Path,
    run_id: &'a str,
    card_id: usize,
    wave_index: usize,
    card_ids: &'a [usize],
    workload: &'a BenchmarkWorkload,
    profile: &'a ThroughputProfile,
    collect_profiler: bool,
    stall_probe: bool,
    wkv_policy: WkvPolicy,
    persistent_dispatch: PersistentDispatchControl,
}

struct SampleRequest<'a> {
    runtime: &'a Path,
    root: &'a Path,
    topology: &'a str,
    sample_index: usize,
    card_ids: &'a [usize],
    workload: &'a BenchmarkWorkload,
    profile: &'a ThroughputProfile,
    collect_profiler: bool,
    stall_probe: bool,
    profiler_attested: bool,
    wkv_policy: WkvPolicy,
    persistent_dispatch: PersistentDispatchControl,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse {}: {error}", path.display()))
}

fn write_new_json<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("cannot create {}: {error}", path.display()))?;
    let mut output = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut output, value)
        .map_err(|error| format!("cannot serialize {}: {error}", path.display()))?;
    output
        .write_all(b"\n")
        .and_then(|()| output.flush())
        .map_err(|error| format!("cannot finish {}: {error}", path.display()))
}

fn require_input_file(text: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(text);
    if !path.is_absolute() || !path.is_file() {
        return Err(format!("input must be an absolute regular file: {text}"));
    }
    fs::canonicalize(&path).map_err(|error| format!("cannot canonicalize {text}: {error}"))
}

fn require_new_output_root(text: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(text);
    if !path.is_absolute() || path.starts_with("/nix/store/") || path.exists() {
        return Err("output root must be a new absolute non-store path".to_owned());
    }
    fs::create_dir(&path).map_err(|error| format!("cannot create output root: {error}"))?;
    fs::canonicalize(&path).map_err(|error| format!("cannot canonicalize output root: {error}"))
}

fn tt_smi_executable() -> Result<PathBuf, String> {
    let package =
        TT_SMI_PATH.ok_or_else(|| "TT_SMI_PATH was not embedded at build time".to_owned())?;
    let executable = Path::new(package).join("bin/tt-smi");
    if !executable.starts_with("/nix/store/") || !executable.is_file() {
        return Err("embedded tt-smi is not a Nix store executable".to_owned());
    }
    Ok(executable)
}

fn capture_machine_snapshot(
    tt_smi: &Path,
    output_root: &Path,
    name: &str,
) -> Result<String, String> {
    let output = Command::new(tt_smi)
        .args(["-s", "--snapshot_no_tty"])
        .output()
        .map_err(|error| format!("cannot launch tt-smi snapshot: {error}"))?;
    if !output.status.success() || output.stdout.is_empty() {
        return Err(format!("tt-smi snapshot failed with {}", output.status));
    }
    let path = output_root.join(format!("machine-{name}.json"));
    fs::write(&path, &output.stdout)
        .map_err(|error| format!("cannot retain {}: {error}", path.display()))?;
    Ok(blake3::hash(&output.stdout).to_hex().to_string())
}

fn embedded_runtime_executable(
    package: Option<&str>,
    environment_name: &str,
) -> Result<PathBuf, String> {
    let package =
        package.ok_or_else(|| format!("{environment_name} was not embedded at build time"))?;
    let executable = Path::new(package).join("bin/rwkv7-p150x2-runtime");
    if !executable.starts_with("/nix/store/") || !executable.is_file() {
        return Err(format!(
            "{environment_name} does not name a Nix store executable"
        ));
    }
    Ok(executable)
}

fn runtime_executable() -> Result<PathBuf, String> {
    embedded_runtime_executable(DEVICE_RUNTIME_PATH, "RWKV7_DEVICE_RUNTIME")
}

fn diagnostic_runtime_executable() -> Result<PathBuf, String> {
    embedded_runtime_executable(
        DIAGNOSTIC_DEVICE_RUNTIME_PATH,
        "RWKV7_DIAGNOSTIC_DEVICE_RUNTIME",
    )
}

fn validate_workload(
    workload: &BenchmarkWorkload,
    profile: &ThroughputProfile,
) -> Result<(), String> {
    if workload.schema_version != SCHEMA_VERSION
        || workload.cpu_reference_profile_blake3 != profile.model_profile_blake3
        || workload.cpu_reference_method != CPU_REFERENCE_METHOD
        || workload.requests.is_empty()
    {
        return Err("benchmark workload has a wrong schema or no requests".to_owned());
    }
    if workload.requests.len() != profile.workload.request_count {
        return Err(format!(
            "workload has {} requests; profile requires {}",
            workload.requests.len(),
            profile.workload.request_count
        ));
    }
    deterministic_slot_assignments(
        workload.requests.len(),
        &ONE_CARD_IDS,
        profile.scheduler.per_card_slot_count,
    )
    .map_err(|error| format!("workload exceeds bounded one-card waves: {error}"))?;
    if workload.requests.len() < TWO_CARD_IDS.len() {
        return Err("workload needs at least one independent request per card".to_owned());
    }
    let mut identities = std::collections::BTreeSet::new();
    let mut saw_short_prompt = false;
    let mut saw_long_prompt = false;
    for request in &workload.requests {
        if request.request_id.is_empty()
            || !identities.insert(request.request_id.as_str())
            || request.prompt_token_ids.is_empty()
            || request.max_new_tokens as usize != profile.workload.generated_token_count
            || request.expected_generated_token_ids.len() != profile.workload.generated_token_count
        {
            return Err(
                "workload contains an empty, duplicate, or wrong-length request".to_owned(),
            );
        }
        if profile.workload.homogeneous_prompts
            && request.prompt_token_ids.len() != profile.workload.short_prompt_token_count
        {
            return Err("homogeneous workload contains a different prompt length".to_owned());
        }
        saw_short_prompt |=
            request.prompt_token_ids.len() == profile.workload.short_prompt_token_count;
        saw_long_prompt |=
            request.prompt_token_ids.len() == profile.workload.long_prompt_token_count;
    }
    if !profile.workload.homogeneous_prompts && (!saw_short_prompt || !saw_long_prompt) {
        return Err("workload does not contain both named short and long prompts".to_owned());
    }
    Ok(())
}

fn bounded_profiler_workload(workload: &BenchmarkWorkload) -> Result<BenchmarkWorkload, String> {
    let mut bounded = workload.clone();
    for request in &mut bounded.requests {
        let expected_token = request
            .expected_generated_token_ids
            .first()
            .copied()
            .ok_or_else(|| "profiler workload request has no expected token".to_owned())?;
        request.max_new_tokens = PROFILER_GENERATED_TOKEN_COUNT;
        request.expected_generated_token_ids = vec![expected_token];
    }
    Ok(bounded)
}

fn worker_request(
    run_id: &str,
    card_id: usize,
    wave_index: usize,
    card_ids: &[usize],
    workload: &BenchmarkWorkload,
    profile: &ThroughputProfile,
) -> Result<WorkerRequest, String> {
    let assignments = deterministic_slot_assignments(
        workload.requests.len(),
        card_ids,
        profile.scheduler.per_card_slot_count,
    )
    .map_err(|error| error.to_string())?;
    let requests = assignments
        .into_iter()
        .filter(|assignment| assignment.card_id == card_id && assignment.wave_index == wave_index)
        .map(|assignment| {
            let request = &workload.requests[assignment.request_index];
            WorkerRequestEntry {
                request_id: request.request_id.clone(),
                prompt_token_ids: request.prompt_token_ids.clone(),
                max_new_tokens: request.max_new_tokens,
                state_slot: assignment.state_slot,
            }
        })
        .collect::<Vec<_>>();
    if requests.is_empty() {
        return Err(format!(
            "scheduler assigned no requests to physical card {card_id}"
        ));
    }
    let slot_count = profile.scheduler.per_card_slot_count;
    let capacity_id = match slot_count {
        4 => "B4",
        16 => "B16",
        32 => "B32",
        64 => "B64",
        _ => {
            return Err(format!(
                "profile slot count {slot_count} is not a supported grid capacity"
            ));
        }
    };
    Ok(WorkerRequest {
        schema_version: SCHEMA_VERSION,
        run_id: format!("{run_id}-wave-{wave_index}-card-{card_id}"),
        capacity: WorkerCapacity {
            id: capacity_id.to_owned(),
            per_card_slot_count: slot_count,
            temporary_tensor_bytes_per_slot: WORKER_CAPACITY_TEMPORARY_BYTES_PER_SLOT,
            device_budget_bytes: WORKER_CAPACITY_DEVICE_BUDGET_BYTES,
            safety_reserve_bytes: WORKER_CAPACITY_SAFETY_RESERVE_BYTES,
        },
        requests,
    })
}

fn launch_worker(request: WorkerLaunchRequest<'_>) -> Result<WorkerProcess, String> {
    let WorkerLaunchRequest {
        runtime,
        run_root,
        run_id,
        card_id,
        wave_index,
        card_ids,
        workload,
        profile,
        collect_profiler,
        stall_probe,
        wkv_policy,
        persistent_dispatch,
    } = request;
    let primary_thread_cpu_affinity = primary_thread_cpus()?
        .map(|cpus| {
            cpus.get(card_id)
                .copied()
                .ok_or_else(|| "primary thread CPU has no card entry".to_owned())
        })
        .transpose()?;
    let primary_thread_fifo_priority = primary_thread_fifo_priority()?;
    validate_primary_thread_scheduling(primary_thread_cpu_affinity, primary_thread_fifo_priority)?;
    let wave_root = run_root.join(format!("wave-{wave_index}"));
    let worker_root = wave_root.join(format!("card-{card_id}"));
    fs::create_dir_all(worker_root.join("profiler"))
        .map_err(|error| format!("cannot create worker root: {error}"))?;
    fs::create_dir_all(worker_root.join("logs"))
        .map_err(|error| format!("cannot create worker logs: {error}"))?;
    let request_path = worker_root.join("request.json");
    let output_path = worker_root.join("receipt.json");
    let initialization_ready_barrier_path =
        worker_root.join(WORKER_INITIALIZATION_READY_BARRIER_NAME);
    let warmup_start_barrier_path = wave_root.join(WORKER_WARMUP_START_BARRIER_NAME);
    let timed_ready_barrier_path = worker_root.join(WORKER_TIMED_READY_BARRIER_NAME);
    let timed_start_barrier_path = wave_root.join(WORKER_TIMED_START_BARRIER_NAME);
    write_new_json(
        &request_path,
        &worker_request(run_id, card_id, wave_index, card_ids, workload, profile)?,
    )?;
    let stdout = File::create(worker_root.join(WORKER_STDOUT_FILENAME))
        .map_err(|error| format!("cannot create worker stdout: {error}"))?;
    let stderr = File::create(worker_root.join(WORKER_STDERR_FILENAME))
        .map_err(|error| format!("cannot create worker stderr: {error}"))?;
    let mut command = Command::new(runtime);
    command
        .arg("--fast-worker")
        .arg("--device")
        .arg(card_id.to_string())
        .arg("--requests")
        .arg(&request_path)
        .arg("--output")
        .arg(&output_path)
        .arg("--wkv-policy")
        .arg(wkv_policy.as_str())
        .arg("--initialization-ready-barrier")
        .arg(&initialization_ready_barrier_path)
        .arg("--warmup-start-barrier")
        .arg(&warmup_start_barrier_path)
        .arg("--timed-ready-barrier")
        .arg(&timed_ready_barrier_path)
        .arg("--timed-start-barrier")
        .arg(&timed_start_barrier_path)
        .env("TT_METAL_LOGS_PATH", worker_root.join("logs"))
        .env("TT_METAL_PROFILER_DIR", worker_root.join("profiler"))
        .env("TT_METAL_INSPECTOR", "0")
        .env("TT_METAL_INSPECTOR_INITIALIZATION_IS_IMPORTANT", "0")
        .env("TT_METAL_INSPECTOR_LOG_RUNTIME_ENTRIES", "0")
        .env("TT_METAL_INSPECTOR_RPC", "0")
        .env_remove("TT_METAL_WATCHER")
        .env_remove("TT_METAL_DPRINT_CORES");
    let persistent_candidate = match persistent_dispatch {
        PersistentDispatchControl::Inherit => persistent_candidate_requested()?,
        PersistentDispatchControl::Normal => {
            command.env(PERSISTENT_DECODE_CANDIDATE_ENVIRONMENT, "0");
            false
        }
        PersistentDispatchControl::Candidate => {
            command.env(PERSISTENT_DECODE_CANDIDATE_ENVIRONMENT, "1");
            true
        }
    };
    let profiler_program_support_count =
        profiler_program_support_count(collect_profiler, stall_probe)?;
    if collect_profiler {
        command
            .env("TT_METAL_DEVICE_PROFILER", "1")
            .env("TT_METAL_PROFILER_MID_RUN_DUMP", "1");
    } else {
        command
            .env_remove("TT_METAL_DEVICE_PROFILER")
            .env_remove("TT_METAL_PROFILER_MID_RUN_DUMP");
    }
    if stall_probe {
        command
            .env(STALL_PROBE_ENVIRONMENT, "1")
            .env(FETCH_QUEUE_DIAGNOSTICS_ENVIRONMENT, "1");
    } else {
        command
            .env_remove(STALL_PROBE_ENVIRONMENT)
            .env_remove(FETCH_QUEUE_DIAGNOSTICS_ENVIRONMENT);
    }
    match primary_thread_cpu_affinity {
        Some(cpu) => {
            command.env(PRIMARY_THREAD_CPU_ENVIRONMENT, cpu.to_string());
        }
        None => {
            command.env_remove(PRIMARY_THREAD_CPU_ENVIRONMENT);
        }
    }
    match profiler_program_support_count {
        Some(count) => {
            command.env(
                PROFILER_PROGRAM_SUPPORT_COUNT_ENVIRONMENT,
                count.to_string(),
            );
        }
        None => {
            command.env_remove(PROFILER_PROGRAM_SUPPORT_COUNT_ENVIRONMENT);
        }
    }
    let child = command
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr))
        .spawn()
        .map_err(|error| format!("cannot launch card {card_id} worker: {error}"))?;
    Ok(WorkerProcess {
        card_id,
        wave_index,
        root: worker_root,
        output_path,
        initialization_ready_barrier_path,
        warmup_start_barrier_path,
        timed_ready_barrier_path,
        timed_start_barrier_path,
        primary_thread_cpu_affinity,
        primary_thread_fifo_priority,
        persistent_candidate,
        child,
    })
}

fn persistent_candidate_flag(value: Option<&str>) -> Result<bool, String> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => Err("RWKV7_PERSISTENT_DECODE_CANDIDATE must be 0 or 1".to_owned()),
    }
}

fn worker_barrier_timeout(persistent_candidate: bool) -> Duration {
    Duration::from_secs(if persistent_candidate {
        PERSISTENT_WORKER_BARRIER_TIMEOUT_SECONDS
    } else {
        NORMAL_WORKER_BARRIER_TIMEOUT_SECONDS
    })
}

fn persistent_candidate_requested() -> Result<bool, String> {
    match std::env::var(PERSISTENT_DECODE_CANDIDATE_ENVIRONMENT) {
        Ok(value) => persistent_candidate_flag(Some(&value)),
        Err(std::env::VarError::NotPresent) => persistent_candidate_flag(None),
        Err(error) => Err(format!(
            "cannot read {PERSISTENT_DECODE_CANDIDATE_ENVIRONMENT}: {error}"
        )),
    }
}

fn persistent_ab_requested() -> Result<bool, String> {
    match std::env::var(PERSISTENT_DECODE_AB_ENVIRONMENT) {
        Ok(value) if value == "1" => Ok(true),
        Ok(value) if value == "0" => Ok(false),
        Ok(_) => Err(format!("{PERSISTENT_DECODE_AB_ENVIRONMENT} must be 0 or 1")),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(error) => Err(format!(
            "cannot read {PERSISTENT_DECODE_AB_ENVIRONMENT}: {error}"
        )),
    }
}

fn wait_for_worker_barriers(
    workers: &mut [WorkerProcess],
    barrier_name: &str,
    barrier_path: fn(&WorkerProcess) -> &Path,
) -> Result<(), String> {
    if workers.is_empty() {
        return Err("cannot synchronize an empty worker set".to_owned());
    }
    let barrier_timeout =
        worker_barrier_timeout(workers.iter().any(|worker| worker.persistent_candidate));
    let deadline = Instant::now() + barrier_timeout;
    loop {
        let mut ready_count = 0_usize;
        for worker in workers.iter_mut() {
            let path = barrier_path(worker);
            if path.is_dir() {
                ready_count = ready_count
                    .checked_add(1)
                    .ok_or_else(|| "ready worker count overflowed".to_owned())?;
                continue;
            }
            if path.exists() {
                return Err(format!(
                    "card {} {barrier_name} barrier is not a directory",
                    worker.card_id
                ));
            }
            if let Some(status) = worker
                .child
                .try_wait()
                .map_err(|error| format!("cannot poll card {} worker: {error}", worker.card_id))?
            {
                return Err(format!(
                    "card {} exited before its {barrier_name} barrier with {status}",
                    worker.card_id
                ));
            }
        }
        if ready_count == workers.len() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "workers did not reach the {barrier_name} barrier within {} seconds",
                barrier_timeout.as_secs()
            ));
        }
        thread::sleep(WORKER_BARRIER_POLL_INTERVAL);
    }
}

fn initialization_ready_barrier_path(worker: &WorkerProcess) -> &Path {
    &worker.initialization_ready_barrier_path
}

fn timed_ready_barrier_path(worker: &WorkerProcess) -> &Path {
    &worker.timed_ready_barrier_path
}

fn release_worker_barrier(
    workers: &[WorkerProcess],
    barrier_name: &str,
    barrier_path: fn(&WorkerProcess) -> &Path,
) -> Result<(), String> {
    let first = workers
        .first()
        .ok_or_else(|| "cannot release an empty worker set".to_owned())?;
    let first_path = barrier_path(first);
    if workers
        .iter()
        .any(|worker| barrier_path(worker) != first_path)
    {
        return Err(format!("workers do not share one {barrier_name} barrier"));
    }
    fs::create_dir(first_path).map_err(|error| {
        format!(
            "cannot release worker {barrier_name} barrier {}: {error}",
            first_path.display()
        )
    })
}

fn warmup_start_barrier_path(worker: &WorkerProcess) -> &Path {
    &worker.warmup_start_barrier_path
}

fn timed_start_barrier_path(worker: &WorkerProcess) -> &Path {
    &worker.timed_start_barrier_path
}

fn stop_workers(workers: &mut [WorkerProcess]) {
    for worker in workers.iter_mut() {
        if matches!(worker.child.try_wait(), Ok(None)) {
            let _ = worker.child.kill();
        }
    }
    for worker in workers.iter_mut() {
        let _ = worker.child.wait();
    }
}

fn synchronize_workers(workers: &mut [WorkerProcess]) -> Result<(), String> {
    let result = (|| {
        wait_for_worker_barriers(
            workers,
            "initialization ready",
            initialization_ready_barrier_path,
        )?;
        release_worker_barrier(workers, "warmup start", warmup_start_barrier_path)?;
        wait_for_worker_barriers(workers, "timed ready", timed_ready_barrier_path)?;
        release_worker_barrier(workers, "timed start", timed_start_barrier_path)
    })();
    if result.is_err() {
        stop_workers(workers);
    }
    result
}

fn wait_for_workers(workers: &mut [WorkerProcess]) -> Result<(), String> {
    for index in 0..workers.len() {
        let result = {
            let worker = &mut workers[index];
            match worker.child.wait() {
                Ok(status) if status.success() => Ok(()),
                Ok(status) => Err(format!(
                    "card {} wave {} worker failed with {status}; see {}",
                    worker.card_id,
                    worker.wave_index,
                    worker.root.display()
                )),
                Err(error) => Err(format!("cannot wait for card {}: {error}", worker.card_id)),
            }
        };
        if let Err(error) = result {
            stop_workers(workers);
            return Err(error);
        }
    }
    Ok(())
}

struct ProfilerHashingReader<R> {
    inner: R,
    hasher: blake3::Hasher,
    byte_count: u64,
}

impl<R> ProfilerHashingReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            hasher: blake3::Hasher::new(),
            byte_count: 0,
        }
    }

    fn finish(self) -> (u64, String) {
        (self.byte_count, self.hasher.finalize().to_hex().to_string())
    }
}

impl<R: Read> Read for ProfilerHashingReader<R> {
    fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
        let byte_count = self.inner.read(buffer)?;
        self.byte_count = self
            .byte_count
            .checked_add(byte_count as u64)
            .ok_or_else(|| io::Error::other("profiler byte count overflowed"))?;
        self.hasher.update(&buffer[..byte_count]);
        Ok(byte_count)
    }
}

fn profiler_card_from_record(
    record: &csv::StringRecord,
    line_index: usize,
) -> Result<Option<usize>, String> {
    if line_index < PROFILER_HEADER_LINE_COUNT || record.iter().all(str::is_empty) {
        return Ok(None);
    }
    let card_text = record
        .get(0)
        .ok_or_else(|| format!("profiler line {} has no card field", line_index + 1))?
        .trim();
    let card_id = card_text
        .parse::<usize>()
        .map_err(|error| format!("profiler line {} card is invalid: {error}", line_index + 1))?;
    Ok(Some(card_id))
}

fn profiler_event_from_record(
    record: &csv::StringRecord,
    line_index: usize,
    logical_card_id: usize,
) -> Result<Option<ProfilerEvent>, String> {
    if record.len() < PROFILER_MINIMUM_FIELD_COUNT {
        return Err(format!(
            "profiler line {} has {} fields; expected at least {}",
            line_index + 1,
            record.len(),
            PROFILER_MINIMUM_FIELD_COUNT
        ));
    }
    let event_type = match record[PROFILER_EVENT_TYPE_INDEX].trim() {
        PROFILER_ZONE_START => ProfilerEventType::Start,
        PROFILER_ZONE_END => ProfilerEventType::End,
        _ => return Ok(None),
    };
    let cycle = record[PROFILER_CYCLE_INDEX]
        .trim()
        .parse::<u64>()
        .map_err(|error| format!("profiler line {} cycle is invalid: {error}", line_index + 1))?;
    let zone_name = record[PROFILER_ZONE_NAME_INDEX].trim();
    if zone_name.is_empty() {
        return Err(format!(
            "profiler line {} has an empty zone name",
            line_index + 1
        ));
    }
    let key = format!(
        "{}|{}|{}|{}|{}",
        record[PROFILER_CORE_X_INDEX].trim(),
        record[PROFILER_CORE_Y_INDEX].trim(),
        record[PROFILER_PROCESSOR_INDEX].trim(),
        record[PROFILER_TIMER_INDEX].trim(),
        zone_name
    );
    Ok(Some(ProfilerEvent {
        logical_card_id,
        key,
        zone_name: zone_name.to_owned(),
        cycle,
        event_type,
    }))
}

#[cfg(test)]
fn profiler_record_from_line(line: &[u8], line_index: usize) -> Result<csv::StringRecord, String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(line);
    reader
        .records()
        .next()
        .ok_or_else(|| format!("profiler line {} has no CSV record", line_index + 1))?
        .map_err(|error| format!("profiler line {} is invalid CSV: {error}", line_index + 1))
}

#[cfg(test)]
fn profiler_card_from_line(line: &[u8], line_index: usize) -> Result<Option<usize>, String> {
    let record = profiler_record_from_line(line, line_index)?;
    profiler_card_from_record(&record, line_index)
}

#[cfg(test)]
fn profiler_event_from_line(
    line: &[u8],
    line_index: usize,
) -> Result<Option<ProfilerEvent>, String> {
    let record = profiler_record_from_line(line, line_index)?;
    let Some(logical_card_id) = profiler_card_from_record(&record, line_index)? else {
        return Ok(None);
    };
    profiler_event_from_record(&record, line_index, logical_card_id)
}

fn profiler_binding_matches(logical_card_id: usize, physical_card_id: usize) -> bool {
    logical_card_id == FILTERED_WORKER_LOGICAL_CARD_ID && TWO_CARD_IDS.contains(&physical_card_id)
}

fn summarize_profiler(worker: &WorkerProcess) -> Result<(), String> {
    let profiler_path = worker.root.join(PROFILER_RELATIVE_PATH);
    let profiler = File::open(&profiler_path)
        .map_err(|error| format!("cannot open {}: {error}", profiler_path.display()))?;
    let buffered = BufReader::with_capacity(PROFILER_READER_BUFFER_BYTES, profiler);
    let hashing_reader = ProfilerHashingReader::new(buffered);
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(hashing_reader);
    let mut record = csv::StringRecord::new();
    let mut line_index = 0_usize;
    let mut previous_byte_position = 0_u64;
    let mut data_row_count = 0_usize;
    let mut active_zones = BTreeMap::<String, Vec<u64>>::new();
    let mut zone_aggregates = BTreeMap::<String, ProfilerZoneAggregate>::new();
    let mut unmatched_end_count = 0_usize;
    loop {
        let has_record = reader
            .read_record(&mut record)
            .map_err(|error| format!("cannot stream {}: {error}", profiler_path.display()))?;
        if !has_record {
            break;
        }
        let next_byte_position = reader.position().byte();
        let record_byte_count = next_byte_position
            .checked_sub(previous_byte_position)
            .ok_or_else(|| "profiler record byte position moved backward".to_owned())?;
        if record_byte_count > MAX_PROFILER_LINE_BYTES as u64 {
            return Err(format!(
                "profiler line {} exceeds {} bytes",
                line_index + 1,
                MAX_PROFILER_LINE_BYTES
            ));
        }
        previous_byte_position = next_byte_position;
        if let Some(card_id) = profiler_card_from_record(&record, line_index)? {
            if !profiler_binding_matches(card_id, worker.card_id) {
                return Err(format!(
                    "profiler line {} attributes logical card {card_id}, expected filtered logical card {} for physical card {}",
                    line_index + 1,
                    FILTERED_WORKER_LOGICAL_CARD_ID,
                    worker.card_id
                ));
            }
            data_row_count = data_row_count
                .checked_add(1)
                .ok_or_else(|| "profiler row count overflowed".to_owned())?;
            if let Some(event) = profiler_event_from_record(&record, line_index, card_id)? {
                if event.logical_card_id != card_id {
                    return Err(format!(
                        "profiler line {} parser card mismatch",
                        line_index + 1
                    ));
                }
                let aggregate = zone_aggregates.entry(event.zone_name.clone()).or_default();
                match event.event_type {
                    ProfilerEventType::Start => {
                        aggregate.start_count = aggregate
                            .start_count
                            .checked_add(1)
                            .ok_or_else(|| "profiler zone start count overflowed".to_owned())?;
                        active_zones.entry(event.key).or_default().push(event.cycle);
                    }
                    ProfilerEventType::End => {
                        let start = active_zones.entry(event.key).or_default().pop();
                        if let Some(start_cycle) = start {
                            let duration = event
                                .cycle
                                .checked_sub(start_cycle)
                                .ok_or_else(|| "profiler zone end precedes its start".to_owned())?;
                            aggregate.complete_count =
                                aggregate.complete_count.checked_add(1).ok_or_else(|| {
                                    "profiler complete zone count overflowed".to_owned()
                                })?;
                            aggregate.total_cycles = aggregate
                                .total_cycles
                                .checked_add(duration)
                                .ok_or_else(|| {
                                "profiler zone cycle total overflowed".to_owned()
                            })?;
                            aggregate.maximum_cycles = aggregate.maximum_cycles.max(duration);
                        } else {
                            unmatched_end_count =
                                unmatched_end_count.checked_add(1).ok_or_else(|| {
                                    "profiler unmatched end count overflowed".to_owned()
                                })?;
                        }
                    }
                }
            }
        }
        line_index = line_index
            .checked_add(1)
            .ok_or_else(|| "profiler line count overflowed".to_owned())?;
    }
    if data_row_count == 0 {
        return Err(format!(
            "profiler has no logical-card rows for physical card {}",
            worker.card_id
        ));
    }
    let (raw_byte_count, raw_blake3) = reader.into_inner().finish();
    let summary_path = worker.root.join(PROFILER_SUMMARY_FILENAME);
    let summary = format!(
        "physical_card_id,logical_card_id,raw_blake3,raw_byte_count,data_row_count,raw_csv_retained\n{},{},{},{},{},false\n",
        worker.card_id, FILTERED_WORKER_LOGICAL_CARD_ID, raw_blake3, raw_byte_count, data_row_count
    );
    fs::write(&summary_path, summary)
        .map_err(|error| format!("cannot write {}: {error}", summary_path.display()))?;
    let unmatched_start_count = active_zones.values().try_fold(0_usize, |total, starts| {
        total
            .checked_add(starts.len())
            .ok_or_else(|| "profiler unmatched start count overflowed".to_owned())
    })?;
    let complete_zone_event_count =
        zone_aggregates
            .values()
            .try_fold(0_usize, |total, aggregate| {
                total
                    .checked_add(aggregate.complete_count)
                    .ok_or_else(|| "profiler complete event count overflowed".to_owned())
            })?;
    let total_zone_cycles = zone_aggregates
        .values()
        .try_fold(0_u64, |total, aggregate| {
            total
                .checked_add(aggregate.total_cycles)
                .ok_or_else(|| "profiler aggregate cycle count overflowed".to_owned())
        })?;
    let signature_bytes = serde_json::to_vec(&zone_aggregates)
        .map_err(|error| format!("cannot serialize profiler program signature: {error}"))?;
    let program_signature_path = worker.root.join(PROFILER_PROGRAM_SIGNATURE_FILENAME);
    write_new_json(
        &program_signature_path,
        &json!({
            "schema_version": SCHEMA_VERSION,
            "physical_card_id": worker.card_id,
            "logical_card_id": FILTERED_WORKER_LOGICAL_CARD_ID,
            "raw_blake3": raw_blake3,
            "data_row_count": data_row_count,
            "program_signature_blake3": blake3::hash(&signature_bytes).to_hex().to_string(),
            "complete_zone_event_count": complete_zone_event_count,
            "total_zone_cycles": total_zone_cycles,
            "unmatched_start_count": unmatched_start_count,
            "unmatched_end_count": unmatched_end_count,
            "zones": zone_aggregates,
        }),
    )?;
    fs::remove_file(&profiler_path).map_err(|error| {
        format!(
            "cannot remove summarized {}: {error}",
            profiler_path.display()
        )
    })?;
    Ok(())
}

fn expected_generated_tokens<'a>(
    workload: &'a BenchmarkWorkload,
    request_id: &str,
) -> Option<&'a [u32]> {
    workload
        .requests
        .iter()
        .find(|request| request.request_id == request_id)
        .map(|request| request.expected_generated_token_ids.as_slice())
}

fn validate_worker(
    worker: &WorkerProcess,
    receipt: &WorkerReceipt,
    workload: &BenchmarkWorkload,
    profile: &ThroughputProfile,
    collect_profiler: bool,
    stall_probe: bool,
    wkv_policy: WkvPolicy,
) -> Result<(), String> {
    let wall_clock_elapsed =
        wall_clock_elapsed_seconds(receipt.timed_start_unix_ns, receipt.timed_end_unix_ns)?;
    let expected_production_physically_admitted = production_worker_physically_admitted(
        wkv_policy,
        worker.card_id,
        worker.primary_thread_cpu_affinity,
        worker.primary_thread_fifo_priority,
    );
    if receipt.schema_version != SCHEMA_VERSION
        || receipt.run_id.is_empty()
        || receipt.physical_device_ids != vec![worker.card_id]
        || receipt.visible_physical_device_ids != vec![worker.card_id]
        || receipt.logical_device_ids != vec![FILTERED_WORKER_LOGICAL_CARD_ID]
        || !receipt.physical_hardware
        || receipt.profiler_attributed != collect_profiler
        || receipt.profiler_trace_only
        || receipt.primary_thread_cpu_affinity != worker.primary_thread_cpu_affinity
        || receipt.primary_thread_fifo_priority != worker.primary_thread_fifo_priority
        || !receipt.elapsed_seconds.is_finite()
        || receipt.elapsed_seconds <= 0.0
        || (wall_clock_elapsed - receipt.elapsed_seconds).abs()
            > WALL_CLOCK_CORRELATION_TOLERANCE_SECONDS
    {
        return Err(format!(
            "card {} returned malformed worker evidence",
            worker.card_id
        ));
    }
    let persistent_mode = receipt
        .persistent_decode
        .get("selected_mode")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| {
            format!(
                "card {} omitted its persistent-decode selection",
                worker.card_id
            )
        })?;
    if persistent_mode == "persistent_trace" {
        let receipt_json = fs::read_to_string(&worker.output_path).map_err(|error| {
            format!(
                "cannot read persistent worker receipt {}: {error}",
                worker.output_path.display()
            )
        })?;
        validate_persistent_receipt(&receipt_json, PersistentReceiptPolicy::Candidate)
            .map_err(|error| format!("card {} {error}", worker.card_id))?;
    } else if persistent_mode != "normal_dispatch" {
        return Err(format!(
            "card {} selected unknown persistent mode '{persistent_mode}'",
            worker.card_id
        ));
    }
    match (
        stall_probe,
        receipt.stall_probe_step_wall_seconds.as_deref(),
        receipt.stall_probe_step_boundary_timings.as_deref(),
        receipt.fetch_queue_diagnostics,
        receipt.hardware_counter_diagnostics.as_ref(),
    ) {
        (true, Some(steps), Some(boundaries), Some(diagnostics), Some(hardware_counters))
            if steps.len() == receipt.batch_step_count =>
        {
            maximum_step(steps)?;
            validate_step_boundary_timings(boundaries, receipt.batch_step_count)?;
            validate_fetch_queue_diagnostics(diagnostics)?;
            hardware_counter_metrics(hardware_counters, receipt.elapsed_seconds)?;
        }
        (false, None, None, None, None) => {}
        _ => {
            return Err(format!(
                "card {} returned malformed stall step evidence",
                worker.card_id
            ));
        }
    }
    if receipt.execution.production_physically_admitted != expected_production_physically_admitted {
        return Err(format!(
            "card {} production admission was {}, expected {}",
            worker.card_id,
            receipt.execution.production_physically_admitted,
            expected_production_physically_admitted,
        ));
    }
    let expected_warmup_token_count = if stall_probe {
        PROFILER_WARMUP_TOKEN_COUNT
    } else {
        profile.timing.warmup_token_count
    };
    let expected_execution = (
        "deterministic_continuous_batching",
        "least_loaded_card_then_lowest_card_then_lowest_slot",
        "independent_card_replica",
        "device_fp32",
        "decode_only",
    );
    let actual_execution = (
        receipt.execution.scheduler_mode.as_str(),
        receipt.execution.scheduler_tie_break.as_str(),
        receipt.execution.replica_mode.as_str(),
        receipt.execution.recurrent_state_residency.as_str(),
        receipt.execution.timed_region.as_str(),
    );
    let contamination_checks = [
        ("execution contract", actual_execution != expected_execution),
        (
            "warmup token count",
            receipt.execution.warmup_token_count != expected_warmup_token_count,
        ),
        ("program cache", !receipt.execution.program_cache_stable),
        (
            "dynamic bindings",
            !receipt.execution.dynamic_token_and_state_bindings_validated,
        ),
        (
            "state readback",
            receipt.execution.state_readback_during_timing,
        ),
        (
            "profiler flush",
            receipt.execution.profiler_flush_during_timing,
        ),
        (
            "control replay",
            receipt.execution.control_replay_during_timing,
        ),
        (
            "artifact writes",
            receipt.execution.artifact_writes_during_timing,
        ),
        ("Inspector", receipt.execution.inspector_during_timing),
        (
            "fetch queue diagnostics",
            receipt.execution.fetch_queue_diagnostics_during_timing != stall_probe,
        ),
        (
            "hardware counters",
            receipt.execution.hardware_counters_during_timing != stall_probe,
        ),
        (
            "primary thread affinity",
            receipt.execution.primary_thread_only_affinity
                != worker.primary_thread_cpu_affinity.is_some(),
        ),
        (
            "primary thread FIFO scheduling",
            receipt.execution.primary_thread_fifo_scheduling
                != worker.primary_thread_fifo_priority.is_some(),
        ),
        (
            "synchronized start",
            !receipt.execution.synchronized_timed_start,
        ),
        (
            "WKV execution policy",
            receipt.execution.wkv_execution_policy != wkv_policy.as_str(),
        ),
    ];
    if let Some((name, _)) = contamination_checks.iter().find(|(_, failed)| *failed) {
        return Err(format!(
            "card {} contaminated its timed region: {name}",
            worker.card_id
        ));
    }
    // r[verify tenstorrent.rwkv7_p150x2.scalable_card_local_batching.valid]
    let expected_capacity_id = match profile.scheduler.per_card_slot_count {
        4 => "B4",
        16 => "B16",
        32 => "B32",
        64 => "B64",
        _ => {
            return Err(format!(
                "profile slot count {} is not a supported grid capacity",
                profile.scheduler.per_card_slot_count
            ));
        }
    };
    if receipt.capacity.id != expected_capacity_id
        || receipt.capacity.per_card_slot_count != profile.scheduler.per_card_slot_count
        || receipt.effective_batch_size == 0
        || receipt.effective_batch_size > profile.scheduler.per_card_slot_count
        || receipt.effective_batch_size != receipt.requests.len()
        || !receipt.queue_occupancy_fraction.is_finite()
        || receipt.queue_occupancy_fraction <= 0.0
        || receipt.queue_occupancy_fraction > 1.0 + 1.0e-9
        || receipt.batch_step_count == 0
        || !receipt.host_gap_accumulated_seconds.is_finite()
        || receipt.host_gap_accumulated_seconds < 0.0
        || receipt.capacity.memory_plan_bytes.recurrent_state_bytes == 0
        || receipt.capacity.memory_plan_bytes.token_shift_bytes == 0
        || receipt.capacity.memory_plan_bytes.temporary_tensor_bytes == 0
        || receipt.capacity.memory_plan_bytes.safety_reserve_bytes == 0
        || receipt.capacity.memory_plan_bytes.total_required_bytes
            != receipt.capacity.memory_plan_bytes.recurrent_state_bytes
                + receipt.capacity.memory_plan_bytes.token_shift_bytes
                + receipt.capacity.memory_plan_bytes.temporary_tensor_bytes
                + receipt.capacity.memory_plan_bytes.safety_reserve_bytes
        || receipt.capacity.memory_plan_bytes.total_required_bytes
            > receipt.capacity.memory_plan_bytes.device_budget_bytes
    {
        return Err(format!(
            "card {} returned a capacity or batch metric outside its admitted profile",
            worker.card_id
        ));
    }
    let profiler_path = worker.root.join(PROFILER_RELATIVE_PATH);
    if collect_profiler {
        require_complete_profiler_output(worker)?;
        summarize_profiler(worker)?;
    } else if profiler_path.exists() {
        return Err(format!(
            "unprofiled worker created unexpected profiler data at {}",
            profiler_path.display()
        ));
    }
    if receipt.requests.is_empty()
        || receipt.requests.iter().any(|request| {
            request.measurement.card_id != worker.card_id
                || expected_generated_tokens(workload, &request.measurement.request_id)
                    != Some(request.generated_token_ids.as_slice())
        })
    {
        return Err(format!(
            "card {} has invalid request attribution",
            worker.card_id
        ));
    }
    Ok(())
}

fn throughput_run_rate(run: &ThroughputRun) -> Result<f64, String> {
    if !run.elapsed_seconds.is_finite() || run.elapsed_seconds <= 0.0 {
        return Err(format!(
            "run '{}' elapsed time must be finite and positive",
            run.run_id
        ));
    }
    let token_count = run.requests.iter().try_fold(0_usize, |total, request| {
        total
            .checked_add(request.token_count)
            .ok_or_else(|| format!("run '{}' token count overflowed", run.run_id))
    })?;
    if token_count == 0 {
        return Err(format!("run '{}' produced no tokens", run.run_id));
    }
    Ok(token_count as f64 / run.elapsed_seconds)
}

fn throughput_run_rates(runs: &[ThroughputRun]) -> Result<Vec<f64>, String> {
    runs.iter().map(throughput_run_rate).collect()
}

fn paired_elapsed_speedups(
    baseline_seconds: &[f64],
    candidate_seconds: &[f64],
) -> Result<Vec<f64>, String> {
    if baseline_seconds.is_empty() || baseline_seconds.len() != candidate_seconds.len() {
        return Err("A/B elapsed samples must be non-empty and have equal lengths".to_owned());
    }
    baseline_seconds
        .iter()
        .zip(candidate_seconds)
        .map(|(baseline, candidate)| {
            if !baseline.is_finite()
                || !candidate.is_finite()
                || *baseline <= 0.0
                || *candidate <= 0.0
            {
                return Err("A/B elapsed times must be finite and positive".to_owned());
            }
            Ok(baseline / candidate)
        })
        .collect()
}

fn run_sample(request: SampleRequest<'_>) -> Result<ThroughputRun, String> {
    let SampleRequest {
        runtime,
        root,
        topology,
        sample_index,
        card_ids,
        workload,
        profile,
        collect_profiler,
        stall_probe,
        profiler_attested,
        wkv_policy,
        persistent_dispatch,
    } = request;
    let run_id = format!("{topology}-{sample_index}");
    let run_root = root.join(&run_id);
    fs::create_dir(&run_root).map_err(|error| format!("cannot create run root: {error}"))?;
    let assignments = deterministic_slot_assignments(
        workload.requests.len(),
        card_ids,
        profile.scheduler.per_card_slot_count,
    )
    .map_err(|error| error.to_string())?;
    let maximum_wave_index = assignments
        .iter()
        .map(|assignment| assignment.wave_index)
        .max()
        .ok_or_else(|| "scheduler returned no wave assignments".to_owned())?;
    let mut elapsed_seconds = 0.0_f64;
    let mut requests = Vec::new();
    for wave_index in 0..=maximum_wave_index {
        let active_cards = assignments
            .iter()
            .filter(|assignment| assignment.wave_index == wave_index)
            .map(|assignment| assignment.card_id)
            .collect::<std::collections::BTreeSet<_>>();
        let mut workers = Vec::with_capacity(active_cards.len());
        for card_id in card_ids
            .iter()
            .filter(|card_id| active_cards.contains(card_id))
        {
            let result = launch_worker(WorkerLaunchRequest {
                runtime,
                run_root: &run_root,
                run_id: &run_id,
                card_id: *card_id,
                wave_index,
                card_ids,
                workload,
                profile,
                collect_profiler,
                stall_probe,
                wkv_policy,
                persistent_dispatch,
            });
            match result {
                Ok(worker) => workers.push(worker),
                Err(error) => {
                    stop_workers(&mut workers);
                    return Err(error);
                }
            }
        }
        synchronize_workers(&mut workers)?;
        wait_for_workers(&mut workers)?;
        let mut worker_timings = Vec::with_capacity(workers.len());
        for worker in &workers {
            let receipt: WorkerReceipt = read_json(&worker.output_path)?;
            validate_worker(
                worker,
                &receipt,
                workload,
                profile,
                collect_profiler,
                stall_probe,
                wkv_policy,
            )?;
            worker_timings.push(WorkerTiming {
                start_monotonic_ns: receipt.timed_start_monotonic_ns,
                end_monotonic_ns: receipt.timed_end_monotonic_ns,
                elapsed_seconds: receipt.elapsed_seconds,
            });
            requests.extend(receipt.requests.into_iter().map(|request| {
                let mut measurement = request.measurement;
                measurement.wave_index = worker.wave_index;
                measurement
            }));
        }
        let wave_timing = synchronized_wave_timing(&worker_timings)?;
        write_new_json(
            &run_root
                .join(format!("wave-{wave_index}"))
                .join("synchronized-timing.json"),
            &json!({
                "schema_version": SCHEMA_VERSION,
                "synchronized_timed_start": true,
                "elapsed_seconds": wave_timing.elapsed_seconds,
                "start_skew_seconds": wave_timing.start_skew_seconds,
                "workers": worker_timings,
            }),
        )?;
        elapsed_seconds += wave_timing.elapsed_seconds;
    }
    requests.sort_by(|left, right| left.request_id.cmp(&right.request_id));
    Ok(ThroughputRun {
        run_id,
        physical_device_ids: card_ids.to_vec(),
        physical_hardware: true,
        profiler_attributed: profiler_attested,
        elapsed_seconds,
        requests,
    })
}

fn load_stall_capture_sample(
    output_root: &Path,
    sample_index: usize,
    wave_elapsed_seconds: f64,
) -> Result<StallCaptureSample, String> {
    let run_root = output_root.join(format!("{STALL_CAPTURE_TOPOLOGY}-{sample_index}"));
    let mut workers = Vec::with_capacity(TWO_CARD_IDS.len());
    let mut worker_timings = Vec::with_capacity(TWO_CARD_IDS.len());
    for physical_card_id in TWO_CARD_IDS {
        let worker_root = run_root
            .join("wave-0")
            .join(format!("card-{physical_card_id}"));
        let receipt: WorkerReceipt = read_json(&worker_root.join("receipt.json"))?;
        worker_timings.push(WorkerTiming {
            start_monotonic_ns: receipt.timed_start_monotonic_ns,
            end_monotonic_ns: receipt.timed_end_monotonic_ns,
            elapsed_seconds: receipt.elapsed_seconds,
        });
        let step_wall_seconds = receipt
            .stall_probe_step_wall_seconds
            .ok_or_else(|| "profiled stall capture omitted step timings".to_owned())?;
        if step_wall_seconds.len() != receipt.batch_step_count {
            return Err("profiled stall capture step count is inconsistent".to_owned());
        }
        let step_boundary_timings = receipt
            .stall_probe_step_boundary_timings
            .ok_or_else(|| "profiled stall capture omitted boundary timings".to_owned())?;
        validate_step_boundary_timings(&step_boundary_timings, receipt.batch_step_count)?;
        let (maximum_step_index, maximum_step_wall_seconds) = maximum_step(&step_wall_seconds)?;
        let summed_step_wall_seconds = step_wall_seconds.iter().sum::<f64>();
        if !summed_step_wall_seconds.is_finite() || summed_step_wall_seconds <= 0.0 {
            return Err("profiled stall capture step sum is not finite and positive".to_owned());
        }
        let fetch_queue_diagnostics = receipt
            .fetch_queue_diagnostics
            .ok_or_else(|| "profiled stall capture omitted fetch queue diagnostics".to_owned())?;
        validate_fetch_queue_diagnostics(fetch_queue_diagnostics)?;
        let hardware_counter_diagnostics = receipt
            .hardware_counter_diagnostics
            .ok_or_else(|| "profiled stall capture omitted hardware counters".to_owned())?;
        let hardware_counter_metrics =
            hardware_counter_metrics(&hardware_counter_diagnostics, receipt.elapsed_seconds)?;
        let profiler: ProfilerProgramSignature =
            read_json(&worker_root.join(PROFILER_PROGRAM_SIGNATURE_FILENAME))?;
        if profiler.schema_version != SCHEMA_VERSION
            || profiler.logical_card_id != FILTERED_WORKER_LOGICAL_CARD_ID
            || profiler.physical_card_id != physical_card_id
            || profiler.unmatched_start_count != 0
            || profiler.unmatched_end_count != 0
            || profiler.total_zone_cycles == 0
        {
            return Err(format!(
                "card {physical_card_id} returned malformed stall profiler evidence"
            ));
        }
        workers.push(StallCaptureWorker {
            physical_card_id,
            elapsed_seconds: receipt.elapsed_seconds,
            timed_start_unix_ns: receipt.timed_start_unix_ns,
            timed_end_unix_ns: receipt.timed_end_unix_ns,
            host_gap_accumulated_seconds: receipt.host_gap_accumulated_seconds,
            primary_thread_cpu_affinity: receipt.primary_thread_cpu_affinity,
            batch_step_count: receipt.batch_step_count,
            maximum_step_index,
            maximum_step_wall_seconds,
            summed_step_wall_seconds,
            step_wall_seconds,
            step_boundary_timings,
            fetch_queue_diagnostics,
            hardware_counter_diagnostics,
            hardware_counter_metrics,
            profiler,
        });
    }
    let wave_timing = synchronized_wave_timing(&worker_timings)?;
    if (wave_timing.elapsed_seconds - wave_elapsed_seconds).abs() > WORKER_ELAPSED_TOLERANCE_SECONDS
    {
        return Err("stall capture wave timing changed during collection".to_owned());
    }
    Ok(StallCaptureSample {
        sample_index,
        wave_elapsed_seconds,
        start_skew_seconds: wave_timing.start_skew_seconds,
        workers,
    })
}

fn validate_stall_capture_attempt_counts(
    attempt_counts: &[usize],
    expected_sample_count: usize,
) -> Result<(), String> {
    if attempt_counts.len() != expected_sample_count {
        return Err("stall capture attempt count differs from sample count".to_owned());
    }
    if attempt_counts.iter().any(|attempt_count| {
        *attempt_count == 0 || *attempt_count > MAXIMUM_STALL_CAPTURE_ATTEMPTS_PER_SAMPLE
    }) {
        return Err("stall capture attempt count is outside its bound".to_owned());
    }
    Ok(())
}

fn run_stall_capture_sample(
    runtime: &Path,
    output_root: &Path,
    sample_index: usize,
    workload: &BenchmarkWorkload,
    profile: &ThroughputProfile,
) -> Result<(StallCaptureSample, usize), String> {
    let run_root = output_root.join(format!("{STALL_CAPTURE_TOPOLOGY}-{sample_index}"));
    let mut last_error = None;
    for attempt_index in 0..MAXIMUM_STALL_CAPTURE_ATTEMPTS_PER_SAMPLE {
        if attempt_index > 0 && run_root.exists() {
            fs::remove_dir_all(&run_root).map_err(|error| {
                format!(
                    "cannot remove rejected stall sample {}: {error}",
                    run_root.display()
                )
            })?;
        }
        match run_sample(SampleRequest {
            runtime,
            root: output_root,
            topology: STALL_CAPTURE_TOPOLOGY,
            sample_index,
            card_ids: &TWO_CARD_IDS,
            workload,
            profile,
            collect_profiler: true,
            stall_probe: true,
            profiler_attested: true,
            wkv_policy: WkvPolicy::FusedCandidate,
            persistent_dispatch: PersistentDispatchControl::Inherit,
        }) {
            Ok(run) => {
                match load_stall_capture_sample(output_root, sample_index, run.elapsed_seconds) {
                    Ok(sample) => {
                        let attempt_count = attempt_index
                            .checked_add(1)
                            .ok_or_else(|| "stall capture attempt count overflowed".to_owned())?;
                        return Ok((sample, attempt_count));
                    }
                    Err(error) => last_error = Some(error),
                }
            }
            Err(error) => last_error = Some(error),
        }
    }
    Err(format!(
        "stall sample {sample_index} failed after {MAXIMUM_STALL_CAPTURE_ATTEMPTS_PER_SAMPLE} attempts: {}",
        last_error.unwrap_or_else(|| "no attempt error was recorded".to_owned())
    ))
}

fn run_stall_capture(arguments: &[String], sample_count: usize) -> Result<(), String> {
    let profile_path = require_input_file(&arguments[STALL_CAPTURE_PROFILE_ARGUMENT_INDEX])?;
    let workload_path = require_input_file(&arguments[STALL_CAPTURE_REQUESTS_ARGUMENT_INDEX])?;
    let output_root = require_new_output_root(&arguments[STALL_CAPTURE_OUTPUT_ARGUMENT_INDEX])?;
    let profile: ThroughputProfile = read_json(&profile_path)?;
    let workload: BenchmarkWorkload = read_json(&workload_path)?;
    validate_workload(&workload, &profile)?;
    if profile.workload.generated_token_count
        != usize::try_from(PROFILER_GENERATED_TOKEN_COUNT)
            .map_err(|_| "profiler token count does not fit usize".to_owned())?
        || profile.timing.warmup_token_count != PROFILER_WARMUP_TOKEN_COUNT
    {
        return Err(format!(
            "stall capture requires exactly {PROFILER_GENERATED_TOKEN_COUNT} generated token and {PROFILER_WARMUP_TOKEN_COUNT} warmup token"
        ));
    }
    let expected_request_count = profile
        .scheduler
        .per_card_slot_count
        .checked_mul(TWO_CARD_IDS.len())
        .ok_or_else(|| "stall capture request count overflowed".to_owned())?;
    if workload.requests.len() != expected_request_count {
        return Err(format!(
            "stall capture requires exactly {expected_request_count} requests"
        ));
    }
    let profile_identity_blake3 = profile_identity(&profile).map_err(|error| error.to_string())?;
    let primary_thread_cpu_affinities = primary_thread_cpus()?;
    let primary_thread_fifo_priority = primary_thread_fifo_priority()?;
    let runtime = diagnostic_runtime_executable()?;
    let tt_smi = tt_smi_executable()?;
    let machine_facts_before = capture_machine_snapshot(&tt_smi, &output_root, "before")?;
    let mut samples = Vec::with_capacity(sample_count);
    let mut sample_attempt_counts = Vec::with_capacity(sample_count);
    for sample_index in 0..sample_count {
        let (sample, attempt_count) =
            run_stall_capture_sample(&runtime, &output_root, sample_index, &workload, &profile)?;
        samples.push(sample);
        sample_attempt_counts.push(attempt_count);
    }
    validate_stall_capture_attempt_counts(&sample_attempt_counts, sample_count)?;
    let machine_facts_after = capture_machine_snapshot(&tt_smi, &output_root, "after")?;
    let elapsed_seconds = samples
        .iter()
        .map(|sample| sample.wave_elapsed_seconds)
        .collect::<Vec<_>>();
    let (normal_index, slow_index) = stall_capture_extremes(&elapsed_seconds)?;
    let comparison = compare_stall_capture_samples(&samples[normal_index], &samples[slow_index])?;
    let receipt_path = output_root.join("stall-capture.json");
    write_new_json(
        &receipt_path,
        &json!({
            "schema_version": SCHEMA_VERSION,
            "kind": "rwkv7_fused_wkv_physical_stall_capture",
            "status": "diagnostic_only",
            "physical_hardware": true,
            "exact_generated_token_parity": true,
            "wkv_execution_policy": WkvPolicy::FusedCandidate.as_str(),
            "production_physically_admitted": false,
            "profile_identity_blake3": profile_identity_blake3,
            "primary_thread_cpu_affinities": primary_thread_cpu_affinities,
            "primary_thread_fifo_priority": primary_thread_fifo_priority,
            "sample_count": sample_count,
            "maximum_attempts_per_sample": MAXIMUM_STALL_CAPTURE_ATTEMPTS_PER_SAMPLE,
            "sample_attempt_counts": sample_attempt_counts,
            "failed_attempt_artifacts_retained": false,
            "timed_samples_profiled": true,
            "warmup_token_count": PROFILER_WARMUP_TOKEN_COUNT,
            "generated_token_count": PROFILER_GENERATED_TOKEN_COUNT,
            "profiler_program_support_count": STALL_PROBE_PROFILER_PROGRAM_SUPPORT_COUNT,
            "profiler_flush_during_timing": false,
            "dropped_profiler_markers": false,
            "raw_profiler_csv_retained": false,
            "machine_facts_before_blake3": machine_facts_before,
            "machine_facts_after_blake3": machine_facts_after,
            "comparison": comparison,
            "samples": samples,
        }),
    )?;
    write_new_json(
        &output_root.join("manifest.json"),
        &BTreeMap::from([("stall_capture", receipt_path)]),
    )?;
    println!("{}", output_root.display());
    Ok(())
}

fn run() -> Result<(), String> {
    let arguments = std::env::args()
        .skip(PROGRAM_ARGUMENT_COUNT)
        .collect::<Vec<_>>();
    if let Some(sample_count) = parse_stall_capture_sample_count(&arguments)? {
        return run_stall_capture(&arguments, sample_count);
    }
    let (wkv_policy, argument_offset) = parse_wkv_policy(&arguments)?;
    let persistent_ab = persistent_ab_requested()?;
    let comparative_ab = persistent_ab || wkv_policy == WkvPolicy::FusedCandidate;
    if persistent_ab && (wkv_policy != WkvPolicy::Production || persistent_candidate_requested()?) {
        return Err(
            "persistent A/B requires production WKV policy and no candidate override".to_owned(),
        );
    }
    let profile_path = require_input_file(&arguments[argument_offset + PROFILE_ARGUMENT_INDEX])?;
    let workload_path = require_input_file(&arguments[argument_offset + REQUESTS_ARGUMENT_INDEX])?;
    let output_root = require_new_output_root(&arguments[argument_offset + OUTPUT_ARGUMENT_INDEX])?;
    let profile: ThroughputProfile = read_json(&profile_path)?;
    let workload: BenchmarkWorkload = read_json(&workload_path)?;
    validate_workload(&workload, &profile)?;
    let identity = profile_identity(&profile).map_err(|error| error.to_string())?;
    let primary_thread_cpu_affinities = primary_thread_cpus()?;
    let primary_thread_fifo_priority = primary_thread_fifo_priority()?;
    let production_physically_admitted = production_run_physically_admitted(
        wkv_policy,
        primary_thread_cpu_affinities.as_deref(),
        primary_thread_fifo_priority,
    );
    let profiler_workload = bounded_profiler_workload(&workload)?;
    let runtime = runtime_executable()?;
    let diagnostic_runtime = diagnostic_runtime_executable()?;
    let tt_smi = tt_smi_executable()?;
    let machine_facts_before = capture_machine_snapshot(&tt_smi, &output_root, "before")?;

    let profiler_single = run_sample(SampleRequest {
        runtime: &diagnostic_runtime,
        root: &output_root,
        topology: "profiler-single",
        sample_index: 0,
        card_ids: &ONE_CARD_IDS,
        workload: &profiler_workload,
        profile: &profile,
        collect_profiler: true,
        stall_probe: true,
        profiler_attested: true,
        wkv_policy,
        persistent_dispatch: PersistentDispatchControl::Inherit,
    })?;
    let profiler_dual = run_sample(SampleRequest {
        runtime: &diagnostic_runtime,
        root: &output_root,
        topology: "profiler-dual",
        sample_index: 0,
        card_ids: &TWO_CARD_IDS,
        workload: &profiler_workload,
        profile: &profile,
        collect_profiler: true,
        stall_probe: true,
        profiler_attested: true,
        wkv_policy,
        persistent_dispatch: PersistentDispatchControl::Inherit,
    })?;
    let baseline_profiler_single = if wkv_policy == WkvPolicy::FusedCandidate {
        Some(run_sample(SampleRequest {
            runtime: &diagnostic_runtime,
            root: &output_root,
            topology: "profiler-generic-single",
            sample_index: 0,
            card_ids: &ONE_CARD_IDS,
            workload: &profiler_workload,
            profile: &profile,
            collect_profiler: true,
            stall_probe: true,
            profiler_attested: true,
            wkv_policy: WkvPolicy::GenericReference,
            persistent_dispatch: PersistentDispatchControl::Inherit,
        })?)
    } else {
        None
    };
    let baseline_profiler_dual = if wkv_policy == WkvPolicy::FusedCandidate {
        Some(run_sample(SampleRequest {
            runtime: &diagnostic_runtime,
            root: &output_root,
            topology: "profiler-generic-dual",
            sample_index: 0,
            card_ids: &TWO_CARD_IDS,
            workload: &profiler_workload,
            profile: &profile,
            collect_profiler: true,
            stall_probe: true,
            profiler_attested: true,
            wkv_policy: WkvPolicy::GenericReference,
            persistent_dispatch: PersistentDispatchControl::Inherit,
        })?)
    } else {
        None
    };
    let profiler_attestation_path = output_root.join("profiler-attestation.json");
    write_new_json(
        &profiler_attestation_path,
        &json!({
            "schema_version": SCHEMA_VERSION,
            "profile_identity_blake3": identity.clone(),
            "single_run_id": profiler_single.run_id,
            "single_physical_device_ids": profiler_single.physical_device_ids,
            "dual_run_id": profiler_dual.run_id,
            "dual_physical_device_ids": profiler_dual.physical_device_ids,
            "baseline_single_run_id": baseline_profiler_single.as_ref().map(|run| &run.run_id),
            "baseline_dual_run_id": baseline_profiler_dual.as_ref().map(|run| &run.run_id),
            "timed_samples_profiled": false,
            "profiled_warmup_token_count": PROFILER_WARMUP_TOKEN_COUNT,
            "profiled_generated_token_count": PROFILER_GENERATED_TOKEN_COUNT,
            "dropped_profiler_markers": false,
            "program_signature_filename": PROFILER_PROGRAM_SIGNATURE_FILENAME,
            "wkv_execution_policy": wkv_policy.as_str(),
            "primary_thread_cpu_affinities": primary_thread_cpu_affinities.clone(),
            "primary_thread_fifo_priority": primary_thread_fifo_priority,
            "production_physically_admitted": production_physically_admitted,
        }),
    )?;

    let mut one_card_runs = Vec::with_capacity(profile.repeat_count);
    let mut two_card_runs = Vec::with_capacity(profile.repeat_count);
    let mut baseline_one_card_runs = Vec::with_capacity(profile.repeat_count);
    let mut baseline_two_card_runs = Vec::with_capacity(profile.repeat_count);
    for sample_index in 0..profile.repeat_count {
        let variants = if persistent_ab {
            order_balanced_persistent_dispatches(sample_index)
                .into_iter()
                .map(|dispatch| (WkvPolicy::Production, dispatch))
                .collect::<Vec<_>>()
        } else if wkv_policy == WkvPolicy::FusedCandidate {
            order_balanced_wkv_policies(sample_index)
                .into_iter()
                .map(|policy| (policy, PersistentDispatchControl::Inherit))
                .collect::<Vec<_>>()
        } else {
            vec![(wkv_policy, PersistentDispatchControl::Inherit)]
        };
        for (sample_policy, persistent_dispatch) in variants {
            let topology = match persistent_dispatch {
                PersistentDispatchControl::Normal => "single-normal",
                PersistentDispatchControl::Candidate => "single-persistent",
                PersistentDispatchControl::Inherit => match sample_policy {
                    WkvPolicy::Production => "single",
                    WkvPolicy::FusedCandidate => "single-fused",
                    WkvPolicy::GenericReference => "single-generic",
                },
            };
            let run = run_sample(SampleRequest {
                runtime: &runtime,
                root: &output_root,
                topology,
                sample_index,
                card_ids: &ONE_CARD_IDS,
                workload: &workload,
                profile: &profile,
                collect_profiler: false,
                stall_probe: false,
                profiler_attested: true,
                wkv_policy: sample_policy,
                persistent_dispatch,
            })?;
            if persistent_ab {
                match persistent_dispatch {
                    PersistentDispatchControl::Normal => baseline_one_card_runs.push(run),
                    PersistentDispatchControl::Candidate => one_card_runs.push(run),
                    PersistentDispatchControl::Inherit => {
                        return Err("persistent A/B inherited a dispatch mode".to_owned());
                    }
                }
            } else {
                match sample_policy {
                    WkvPolicy::GenericReference => baseline_one_card_runs.push(run),
                    WkvPolicy::Production | WkvPolicy::FusedCandidate => one_card_runs.push(run),
                }
            }
        }
    }
    for sample_index in 0..profile.repeat_count {
        let variants = if persistent_ab {
            order_balanced_persistent_dispatches(sample_index)
                .into_iter()
                .map(|dispatch| (WkvPolicy::Production, dispatch))
                .collect::<Vec<_>>()
        } else if wkv_policy == WkvPolicy::FusedCandidate {
            order_balanced_wkv_policies(sample_index)
                .into_iter()
                .map(|policy| (policy, PersistentDispatchControl::Inherit))
                .collect::<Vec<_>>()
        } else {
            vec![(wkv_policy, PersistentDispatchControl::Inherit)]
        };
        for (sample_policy, persistent_dispatch) in variants {
            let topology = match persistent_dispatch {
                PersistentDispatchControl::Normal => "dual-normal",
                PersistentDispatchControl::Candidate => "dual-persistent",
                PersistentDispatchControl::Inherit => match sample_policy {
                    WkvPolicy::Production => "dual",
                    WkvPolicy::FusedCandidate => "dual-fused",
                    WkvPolicy::GenericReference => "dual-generic",
                },
            };
            let run = run_sample(SampleRequest {
                runtime: &runtime,
                root: &output_root,
                topology,
                sample_index,
                card_ids: &TWO_CARD_IDS,
                workload: &workload,
                profile: &profile,
                collect_profiler: false,
                stall_probe: false,
                profiler_attested: true,
                wkv_policy: sample_policy,
                persistent_dispatch,
            })?;
            if persistent_ab {
                match persistent_dispatch {
                    PersistentDispatchControl::Normal => baseline_two_card_runs.push(run),
                    PersistentDispatchControl::Candidate => two_card_runs.push(run),
                    PersistentDispatchControl::Inherit => {
                        return Err("persistent A/B inherited a dispatch mode".to_owned());
                    }
                }
            } else {
                match sample_policy {
                    WkvPolicy::GenericReference => baseline_two_card_runs.push(run),
                    WkvPolicy::Production | WkvPolicy::FusedCandidate => two_card_runs.push(run),
                }
            }
        }
    }
    let machine_facts_after = capture_machine_snapshot(&tt_smi, &output_root, "after")?;
    let machine_facts = MachineFactsEvidence {
        source: MACHINE_FACT_SOURCE.to_owned(),
        before_blake3: machine_facts_before,
        after_blake3: machine_facts_after,
        physical_device_ids: TWO_CARD_IDS,
    };
    let execution = ExecutionEvidence {
        scheduler_mode: profile.scheduler.mode.clone(),
        scheduler_tie_break: profile.scheduler.tie_break.clone(),
        replica_mode: profile.runtime.replica_mode.clone(),
        recurrent_state_residency: profile.runtime.recurrent_state_residency.clone(),
        timed_region: profile.timing.timed_region.clone(),
        warmup_token_count: profile.timing.warmup_token_count,
        program_cache_stable: true,
        dynamic_token_and_state_bindings_validated: true,
        state_readback_during_timing: false,
        profiler_flush_during_timing: false,
        control_replay_during_timing: false,
        artifact_writes_during_timing: false,
    };
    let baseline_raw = if comparative_ab {
        Some(RawThroughputEvidence {
            schema_version: SCHEMA_VERSION,
            profile_identity_blake3: identity.clone(),
            machine_facts: machine_facts.clone(),
            execution: execution.clone(),
            one_card_runs: baseline_one_card_runs,
            two_card_runs: baseline_two_card_runs,
        })
    } else {
        None
    };
    let raw = RawThroughputEvidence {
        schema_version: SCHEMA_VERSION,
        profile_identity_blake3: identity,
        machine_facts,
        execution,
        one_card_runs,
        two_card_runs,
    };
    let raw_path = output_root.join("raw-throughput-evidence.json");
    write_new_json(&raw_path, &raw)?;
    let receipt = summarize_throughput(&raw, &profile).map_err(|error| error.to_string())?;
    let receipt_path = output_root.join("throughput-receipt.json");
    write_new_json(&receipt_path, &receipt)?;
    let admission_path = output_root.join("throughput-admission.json");
    match admit_throughput_receipt(&receipt, &raw, &profile) {
        Ok(admission) => write_new_json(
            &admission_path,
            &json!({
                "status": if wkv_policy == WkvPolicy::Production {
                    "production_scaling_admitted"
                } else {
                    "fused_candidate_scaling_pass"
                },
                "wkv_execution_policy": wkv_policy.as_str(),
                "production_physically_admitted": production_physically_admitted,
                "identity_blake3": admission.identity_blake3,
                "dual_card_speedup": admission.dual_card_speedup,
                "scaling_efficiency": admission.scaling_efficiency,
                "p99_latency_regression_fraction": admission.p99_latency_regression_fraction,
                "aggregate_tokens_per_second": admission.aggregate_tokens_per_second,
            }),
        )?,
        Err(rejection) if comparative_ab => write_new_json(
            &admission_path,
            &json!({
                "status": if persistent_ab {
                    "persistent_candidate_scaling_rejected"
                } else {
                    "fused_candidate_scaling_rejected"
                },
                "wkv_execution_policy": wkv_policy.as_str(),
                "production_physically_admitted": false,
                "rejection": rejection.to_string(),
            }),
        )?,
        Err(rejection) => return Err(rejection.to_string()),
    }
    let mut manifest = BTreeMap::from([
        ("raw", raw_path),
        ("receipt", receipt_path),
        ("admission", admission_path),
        ("profiler_attestation", profiler_attestation_path),
    ]);
    if let Some(baseline) = baseline_raw {
        let baseline_raw_path = output_root.join("baseline-raw-throughput-evidence.json");
        write_new_json(&baseline_raw_path, &baseline)?;
        let baseline_receipt =
            summarize_throughput(&baseline, &profile).map_err(|error| error.to_string())?;
        let baseline_receipt_path = output_root.join("baseline-throughput-receipt.json");
        write_new_json(&baseline_receipt_path, &baseline_receipt)?;
        let baseline_admission_path = output_root.join("baseline-throughput-admission.json");
        match admit_throughput_receipt(&baseline_receipt, &baseline, &profile) {
            Ok(baseline_admission) => write_new_json(
                &baseline_admission_path,
                &json!({
                    "status": "generic_baseline_scaling_admitted",
                    "identity_blake3": baseline_admission.identity_blake3,
                    "dual_card_speedup": baseline_admission.dual_card_speedup,
                    "scaling_efficiency": baseline_admission.scaling_efficiency,
                    "p99_latency_regression_fraction": baseline_admission.p99_latency_regression_fraction,
                    "aggregate_tokens_per_second": baseline_admission.aggregate_tokens_per_second,
                }),
            )?,
            Err(rejection) => write_new_json(
                &baseline_admission_path,
                &json!({
                    "status": "generic_baseline_scaling_rejected",
                    "rejection": rejection.to_string(),
                }),
            )?,
        }

        let one_card_speedups = if persistent_ab {
            paired_elapsed_speedups(
                &baseline
                    .one_card_runs
                    .iter()
                    .map(|run| run.elapsed_seconds)
                    .collect::<Vec<_>>(),
                &raw.one_card_runs
                    .iter()
                    .map(|run| run.elapsed_seconds)
                    .collect::<Vec<_>>(),
            )?
        } else {
            paired_speedups(
                &throughput_run_rates(&baseline.one_card_runs)?,
                &throughput_run_rates(&raw.one_card_runs)?,
            )?
        };
        let two_card_speedups = if persistent_ab {
            paired_elapsed_speedups(
                &baseline
                    .two_card_runs
                    .iter()
                    .map(|run| run.elapsed_seconds)
                    .collect::<Vec<_>>(),
                &raw.two_card_runs
                    .iter()
                    .map(|run| run.elapsed_seconds)
                    .collect::<Vec<_>>(),
            )?
        } else {
            paired_speedups(
                &throughput_run_rates(&baseline.two_card_runs)?,
                &throughput_run_rates(&raw.two_card_runs)?,
            )?
        };
        let stable_gain = one_card_speedups.iter().all(|speedup| *speedup > 1.0)
            && two_card_speedups.iter().all(|speedup| *speedup > 1.0);
        let order_schedule = (0..profile.repeat_count)
            .map(|sample_index| {
                if persistent_ab {
                    order_balanced_persistent_dispatches(sample_index)
                        .map(PersistentDispatchControl::as_str)
                        .to_vec()
                } else {
                    order_balanced_wkv_policies(sample_index)
                        .map(WkvPolicy::as_str)
                        .to_vec()
                }
            })
            .collect::<Vec<_>>();
        let ab_receipt_path = output_root.join("physical-ab-receipt.json");
        write_new_json(
            &ab_receipt_path,
            &json!({
                "schema_version": SCHEMA_VERSION,
                "kind": if persistent_ab {
                    "rwkv7_persistent_decode_window_physical_ab"
                } else {
                    "rwkv7_fused_wkv_physical_ab"
                },
                "status": if stable_gain {
                    "stable_physical_gain"
                } else {
                    "rejected_unstable_or_no_gain"
                },
                "physical_hardware": true,
                "production_physically_admitted": false,
                "batch_capacity": profile.scheduler.per_card_slot_count,
                "repeat_count": profile.repeat_count,
                "order_schedule": order_schedule,
                "primary_thread_cpu_affinities": primary_thread_cpu_affinities,
                "primary_thread_fifo_priority": primary_thread_fifo_priority,
                "exact_generated_token_parity": true,
                "cpu_reference_profile_blake3": workload.cpu_reference_profile_blake3,
                "baseline_mode": if persistent_ab { "normal_dispatch" } else { "generic_reference" },
                "candidate_mode": if persistent_ab { "persistent_trace" } else { "fused_candidate" },
                "baseline_receipt_identity_blake3": baseline_receipt.identity_blake3,
                "candidate_receipt_identity_blake3": receipt.identity_blake3,
                "baseline_one_card_tokens_per_second": baseline_receipt.one_card.aggregate_tokens_per_second,
                "candidate_one_card_tokens_per_second": receipt.one_card.aggregate_tokens_per_second,
                "baseline_two_card_tokens_per_second": baseline_receipt.two_card.aggregate_tokens_per_second,
                "candidate_two_card_tokens_per_second": receipt.two_card.aggregate_tokens_per_second,
                "one_card_paired_speedups": one_card_speedups,
                "two_card_paired_speedups": two_card_speedups,
                "stable_gain": stable_gain,
                "profiler_attestation": "profiler-attestation.json",
                "machine_facts_before": "machine-before.json",
                "machine_facts_after": "machine-after.json",
            }),
        )?;
        manifest.insert("baseline_raw", baseline_raw_path);
        manifest.insert("baseline_receipt", baseline_receipt_path);
        manifest.insert("baseline_admission", baseline_admission_path);
        manifest.insert("physical_ab_receipt", ab_receipt_path);
    }
    write_new_json(&output_root.join("manifest.json"), &manifest)?;
    println!("{}", output_root.display());
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rwkv7-p150x2-throughput-benchmark: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLAKE3_HEX_LENGTH: usize = 64;
    const TEST_RATIO_ULP_MARGIN: f64 = 2.0;
    const TEST_RATIO_TOLERANCE: f64 = TEST_RATIO_ULP_MARGIN * f64::EPSILON;

    fn test_hardware_counter_diagnostics(elapsed_nanoseconds: u64) -> HardwareCounterDiagnostics {
        const TEST_CYCLE_FACTOR: u64 = 4;
        const TEST_INSTRUCTION_FACTOR: u64 = 2;
        const TEST_CACHE_REFERENCE_DIVISOR: u64 = 10;
        const TEST_CACHE_MISS_DIVISOR: u64 = 4;
        const TEST_MPERF_FACTOR: u64 = 3;
        const TEST_CONTEXT_SWITCHES: u64 = 10;
        let reading = |value| HardwareCounterReading {
            value,
            time_enabled_nanoseconds: elapsed_nanoseconds,
            time_running_nanoseconds: elapsed_nanoseconds,
        };
        let cache_references = elapsed_nanoseconds / TEST_CACHE_REFERENCE_DIVISOR;
        HardwareCounterDiagnostics {
            scope: HARDWARE_COUNTER_SCOPE.to_owned(),
            cycles: reading(elapsed_nanoseconds * TEST_CYCLE_FACTOR),
            instructions: reading(elapsed_nanoseconds * TEST_INSTRUCTION_FACTOR),
            cache_references: reading(cache_references),
            cache_misses: reading(cache_references / TEST_CACHE_MISS_DIVISOR),
            aperf: reading(elapsed_nanoseconds * TEST_CYCLE_FACTOR),
            mperf: reading(elapsed_nanoseconds * TEST_MPERF_FACTOR),
            task_clock_nanoseconds: reading(elapsed_nanoseconds),
            context_switches: reading(TEST_CONTEXT_SWITCHES),
            cpu_migrations: reading(1),
        }
    }

    fn stall_capture_worker(
        physical_card_id: usize,
        total_zone_cycles: u64,
        maximum_step_wall_seconds: f64,
        summed_step_wall_seconds: f64,
    ) -> StallCaptureWorker {
        const TEST_HOST_GAP_SECONDS: f64 = 0.001;
        const TEST_BATCH_STEP_COUNT: usize = 2;
        const TEST_PROFILER_DATA_ROWS: usize = 2;
        const TEST_START_UNIX_NS: u64 = 1_000_000_000;
        const TEST_READBACK_COMPONENT_FRACTION: f64 = 0.05;
        const TEST_READBACK_COMPONENT_COUNT: f64 = 3.0;
        const TEST_FETCH_QUEUE_RESERVE_CALLS: u64 = 100;
        const TEST_FETCH_QUEUE_WAIT_CALLS: u64 = 4;
        const TEST_FETCH_QUEUE_POLL_ITERATIONS: u64 = 8;
        const TEST_FETCH_QUEUE_WAIT_DIVISOR: u64 = 2;
        const TEST_THREAD_CPU_FRACTION: f64 = 0.5;
        const TEST_PROCESS_CPU_FRACTION: f64 = 0.75;
        let elapsed_ns = Duration::from_secs_f64(summed_step_wall_seconds).as_nanos() as u64;
        let prompt_step_seconds = summed_step_wall_seconds - maximum_step_wall_seconds;
        let readback_component_seconds =
            maximum_step_wall_seconds * TEST_READBACK_COMPONENT_FRACTION;
        let selection_model_seconds =
            maximum_step_wall_seconds - readback_component_seconds * TEST_READBACK_COMPONENT_COUNT;
        let hardware_counter_diagnostics = test_hardware_counter_diagnostics(elapsed_ns);
        let hardware_counter_metrics =
            hardware_counter_metrics(&hardware_counter_diagnostics, summed_step_wall_seconds)
                .expect("test hardware counters");
        StallCaptureWorker {
            physical_card_id,
            elapsed_seconds: summed_step_wall_seconds,
            timed_start_unix_ns: TEST_START_UNIX_NS,
            timed_end_unix_ns: TEST_START_UNIX_NS + elapsed_ns,
            host_gap_accumulated_seconds: TEST_HOST_GAP_SECONDS,
            primary_thread_cpu_affinity: None,
            batch_step_count: TEST_BATCH_STEP_COUNT,
            maximum_step_index: 1,
            maximum_step_wall_seconds,
            summed_step_wall_seconds,
            step_wall_seconds: vec![prompt_step_seconds, maximum_step_wall_seconds],
            step_boundary_timings: vec![
                StallProbeStepBoundaryTiming {
                    selection_requested: false,
                    model_submission_seconds: prompt_step_seconds,
                    layer_submission_seconds: vec![
                        prompt_step_seconds / RWKV7_LAYER_COUNT as f64;
                        RWKV7_LAYER_COUNT
                    ],
                    model_submission_thread_cpu_seconds: prompt_step_seconds
                        * TEST_THREAD_CPU_FRACTION,
                    layer_submission_thread_cpu_seconds: vec![
                        prompt_step_seconds
                            * TEST_THREAD_CPU_FRACTION
                            / RWKV7_LAYER_COUNT as f64;
                        RWKV7_LAYER_COUNT
                    ],
                    model_submission_process_cpu_seconds: prompt_step_seconds
                        * TEST_PROCESS_CPU_FRACTION,
                    layer_submission_process_cpu_seconds: vec![
                        prompt_step_seconds
                            * TEST_PROCESS_CPU_FRACTION
                            / RWKV7_LAYER_COUNT as f64;
                        RWKV7_LAYER_COUNT
                    ],
                    blocking_readback_seconds: 0.0,
                    host_materialization_seconds: 0.0,
                    readback_validation_seconds: 0.0,
                    total_seconds: prompt_step_seconds,
                },
                StallProbeStepBoundaryTiming {
                    selection_requested: true,
                    model_submission_seconds: selection_model_seconds,
                    layer_submission_seconds: vec![
                        selection_model_seconds
                            / RWKV7_LAYER_COUNT as f64;
                        RWKV7_LAYER_COUNT
                    ],
                    model_submission_thread_cpu_seconds: selection_model_seconds
                        * TEST_THREAD_CPU_FRACTION,
                    layer_submission_thread_cpu_seconds: vec![
                        selection_model_seconds
                            * TEST_THREAD_CPU_FRACTION
                            / RWKV7_LAYER_COUNT as f64;
                        RWKV7_LAYER_COUNT
                    ],
                    model_submission_process_cpu_seconds: selection_model_seconds
                        * TEST_PROCESS_CPU_FRACTION,
                    layer_submission_process_cpu_seconds: vec![
                        selection_model_seconds
                            * TEST_PROCESS_CPU_FRACTION
                            / RWKV7_LAYER_COUNT as f64;
                        RWKV7_LAYER_COUNT
                    ],
                    blocking_readback_seconds: readback_component_seconds,
                    host_materialization_seconds: readback_component_seconds,
                    readback_validation_seconds: readback_component_seconds,
                    total_seconds: maximum_step_wall_seconds,
                },
            ],
            fetch_queue_diagnostics: FetchQueueDiagnostics {
                reserve_calls: TEST_FETCH_QUEUE_RESERVE_CALLS,
                wait_calls: TEST_FETCH_QUEUE_WAIT_CALLS,
                poll_iterations: TEST_FETCH_QUEUE_POLL_ITERATIONS,
                total_wait_nanoseconds: elapsed_ns / TEST_FETCH_QUEUE_WAIT_DIVISOR,
                maximum_wait_nanoseconds: elapsed_ns
                    / TEST_FETCH_QUEUE_WAIT_DIVISOR
                    / TEST_FETCH_QUEUE_WAIT_DIVISOR,
            },
            hardware_counter_diagnostics,
            hardware_counter_metrics,
            profiler: ProfilerProgramSignature {
                schema_version: SCHEMA_VERSION,
                logical_card_id: FILTERED_WORKER_LOGICAL_CARD_ID,
                physical_card_id,
                raw_blake3: "a".repeat(BLAKE3_HEX_LENGTH),
                data_row_count: TEST_PROFILER_DATA_ROWS,
                complete_zone_event_count: 1,
                unmatched_start_count: 0,
                unmatched_end_count: 0,
                total_zone_cycles,
                zones: BTreeMap::from([(
                    "TRISC-KERNEL".to_owned(),
                    ProfilerZoneAggregate {
                        start_count: 1,
                        complete_count: 1,
                        total_cycles: total_zone_cycles,
                        maximum_cycles: total_zone_cycles,
                    },
                )]),
                program_signature_blake3: "b".repeat(BLAKE3_HEX_LENGTH),
            },
        }
    }

    #[test]
    fn hardware_counter_metrics_accept_complete_nonmultiplexed_counts() {
        const TEST_ELAPSED_NANOSECONDS: u64 = 1_000_000;
        let diagnostics = test_hardware_counter_diagnostics(TEST_ELAPSED_NANOSECONDS);
        let elapsed_seconds = TEST_ELAPSED_NANOSECONDS as f64 / NANOSECONDS_PER_SECOND;
        let metrics = hardware_counter_metrics(&diagnostics, elapsed_seconds)
            .expect("valid hardware counters");
        assert_eq!(metrics.instructions_per_cycle, 0.5);
        assert_eq!(metrics.cache_miss_fraction, 0.25);
        assert_eq!(metrics.average_core_ghz, 4.0);
        assert_eq!(metrics.instruction_rate_giga_per_second, 2.0);
        assert_eq!(metrics.scheduled_fraction, 1.0);
    }

    #[test]
    fn hardware_counter_metrics_reject_zero_and_multiplexed_counts() {
        const TEST_ELAPSED_NANOSECONDS: u64 = 1_000_000;
        const TEST_MULTIPLEXED_RUNNING_NANOSECONDS: u64 = 1;
        let valid = test_hardware_counter_diagnostics(TEST_ELAPSED_NANOSECONDS);
        let elapsed_seconds = TEST_ELAPSED_NANOSECONDS as f64 / NANOSECONDS_PER_SECOND;
        let mut zero_instructions = valid.clone();
        zero_instructions.instructions.value = 0;
        assert!(hardware_counter_metrics(&zero_instructions, elapsed_seconds).is_err());
        let mut multiplexed = valid;
        multiplexed.cycles.time_running_nanoseconds = TEST_MULTIPLEXED_RUNNING_NANOSECONDS;
        assert!(hardware_counter_metrics(&multiplexed, elapsed_seconds).is_err());
        assert!(
            hardware_counter_metrics(
                &test_hardware_counter_diagnostics(TEST_ELAPSED_NANOSECONDS),
                0.0,
            )
            .is_err()
        );
    }

    #[test]
    fn stall_probe_uses_bounded_profiler_program_capacity() {
        assert_eq!(
            profiler_program_support_count(true, true).expect("profiled stall probe"),
            Some(STALL_PROBE_PROFILER_PROGRAM_SUPPORT_COUNT)
        );
        assert_eq!(
            profiler_program_support_count(true, false).expect("ordinary profiler sample"),
            None
        );
    }

    #[test]
    fn stall_probe_rejects_missing_profiler_collection() {
        assert!(
            profiler_program_support_count(false, true)
                .expect_err("unprofiled stall probe must fail")
                .contains("requires device-profiler collection")
        );
    }

    #[test]
    fn fetch_queue_diagnostics_accept_consistent_waits() {
        const RESERVE_CALLS: u64 = 10;
        const WAIT_CALLS: u64 = 2;
        const POLL_ITERATIONS: u64 = 8;
        const TOTAL_WAIT_NANOSECONDS: u64 = 100;
        const MAXIMUM_WAIT_NANOSECONDS: u64 = 75;
        assert_eq!(
            validate_fetch_queue_diagnostics(FetchQueueDiagnostics {
                reserve_calls: RESERVE_CALLS,
                wait_calls: WAIT_CALLS,
                poll_iterations: POLL_ITERATIONS,
                total_wait_nanoseconds: TOTAL_WAIT_NANOSECONDS,
                maximum_wait_nanoseconds: MAXIMUM_WAIT_NANOSECONDS,
            }),
            Ok(())
        );
    }

    #[test]
    fn fetch_queue_diagnostics_reject_missing_poll_data() {
        let error = validate_fetch_queue_diagnostics(FetchQueueDiagnostics {
            reserve_calls: 1,
            wait_calls: 1,
            poll_iterations: 0,
            total_wait_nanoseconds: 1,
            maximum_wait_nanoseconds: 1,
        })
        .expect_err("wait without a poll must fail");
        assert!(error.contains("too few poll iterations"));
    }

    #[test]
    fn primary_thread_cpu_parser_accepts_distinct_card_cpus() {
        assert_eq!(parse_primary_thread_cpus(Some("2,6")), Ok(Some(vec![2, 6])));
        assert_eq!(parse_primary_thread_cpus(None), Ok(None));
    }

    #[test]
    fn primary_thread_cpu_parser_rejects_bad_layouts() {
        for value in ["2", "2,2", "2,6,10", "2,not-a-cpu"] {
            assert!(parse_primary_thread_cpus(Some(value)).is_err());
        }
    }

    #[test]
    fn primary_thread_fifo_parser_accepts_a_positive_priority() {
        assert_eq!(parse_primary_thread_fifo_priority(Some("1")), Ok(Some(1)));
        assert_eq!(parse_primary_thread_fifo_priority(None), Ok(None));
    }

    #[test]
    fn primary_thread_fifo_parser_rejects_bad_priorities() {
        for value in ["", "0", "not-a-priority"] {
            assert!(parse_primary_thread_fifo_priority(Some(value)).is_err());
        }
    }

    #[test]
    fn primary_thread_scheduling_requires_affinity_for_fifo() {
        assert_eq!(validate_primary_thread_scheduling(Some(2), Some(1)), Ok(()));
        assert_eq!(validate_primary_thread_scheduling(None, None), Ok(()));
        assert!(validate_primary_thread_scheduling(None, Some(1)).is_err());
    }

    #[test]
    fn production_admission_accepts_only_the_measured_scheduler_contract() {
        assert!(production_worker_physically_admitted(
            WkvPolicy::Production,
            0,
            Some(ADMITTED_PRIMARY_THREAD_CPUS[0]),
            Some(ADMITTED_PRIMARY_THREAD_FIFO_PRIORITY),
        ));
        assert!(production_run_physically_admitted(
            WkvPolicy::Production,
            Some(ADMITTED_PRIMARY_THREAD_CPUS.as_slice()),
            Some(ADMITTED_PRIMARY_THREAD_FIFO_PRIORITY),
        ));
        assert!(!production_worker_physically_admitted(
            WkvPolicy::FusedCandidate,
            0,
            Some(ADMITTED_PRIMARY_THREAD_CPUS[0]),
            Some(ADMITTED_PRIMARY_THREAD_FIFO_PRIORITY),
        ));
        assert!(!production_run_physically_admitted(
            WkvPolicy::Production,
            Some(ADMITTED_PRIMARY_THREAD_CPUS.as_slice()),
            None,
        ));
    }

    #[test]
    fn stall_capture_attempt_counts_accept_the_bounded_retry() {
        assert_eq!(validate_stall_capture_attempt_counts(&[1, 2], 2), Ok(()));
    }

    #[test]
    fn stall_capture_attempt_counts_reject_bad_counts() {
        let above_maximum = MAXIMUM_STALL_CAPTURE_ATTEMPTS_PER_SAMPLE
            .checked_add(1)
            .expect("attempt test bound");
        for counts in [vec![1], vec![0, 1], vec![above_maximum, 1]] {
            assert!(validate_stall_capture_attempt_counts(&counts, 2).is_err());
        }
    }

    #[test]
    fn stall_capture_parser_accepts_bounded_samples() {
        const TEST_SAMPLE_COUNT: usize = 6;
        let arguments = vec![
            STALL_CAPTURE_MODE.to_owned(),
            TEST_SAMPLE_COUNT.to_string(),
            "profile".to_owned(),
            "workload".to_owned(),
            "output".to_owned(),
        ];
        assert_eq!(
            parse_stall_capture_sample_count(&arguments),
            Ok(Some(TEST_SAMPLE_COUNT))
        );
        assert_eq!(
            parse_stall_capture_sample_count(&["profile".to_owned()]),
            Ok(None)
        );
    }

    #[test]
    fn stall_capture_parser_rejects_bad_counts() {
        let above_maximum = MAXIMUM_STALL_CAPTURE_SAMPLE_COUNT
            .checked_add(1)
            .expect("stall capture test bound");
        for count in [
            "0".to_owned(),
            above_maximum.to_string(),
            "not-a-count".to_owned(),
        ] {
            let arguments = vec![
                STALL_CAPTURE_MODE.to_owned(),
                count,
                "profile".to_owned(),
                "workload".to_owned(),
                "output".to_owned(),
            ];
            assert!(parse_stall_capture_sample_count(&arguments).is_err());
        }
    }

    #[test]
    fn wall_clock_correlation_accepts_a_positive_range() {
        const START_UNIX_NS: u64 = 1_000_000_000;
        const END_UNIX_NS: u64 = 1_500_000_000;
        const EXPECTED_SECONDS: f64 = 0.5;
        assert_eq!(
            wall_clock_elapsed_seconds(START_UNIX_NS, END_UNIX_NS),
            Ok(EXPECTED_SECONDS)
        );
    }

    #[test]
    fn wall_clock_correlation_rejects_nonpositive_ranges() {
        const START_UNIX_NS: u64 = 1_000_000_000;
        assert!(wall_clock_elapsed_seconds(START_UNIX_NS, START_UNIX_NS).is_err());
        assert!(wall_clock_elapsed_seconds(START_UNIX_NS, START_UNIX_NS - 1).is_err());
    }

    #[test]
    fn bounded_profiler_workload_keeps_one_expected_token() {
        const SECOND_PROMPT_TOKEN: u32 = 2;
        const FULL_GENERATED_TOKEN_COUNT: u32 = 2;
        const FIRST_EXPECTED_TOKEN: u32 = 39;
        const SECOND_EXPECTED_TOKEN: u32 = 40;
        let workload = BenchmarkWorkload {
            schema_version: SCHEMA_VERSION,
            cpu_reference_profile_blake3: "profile".to_owned(),
            cpu_reference_method: CPU_REFERENCE_METHOD.to_owned(),
            requests: vec![BenchmarkRequest {
                request_id: "request-0".to_owned(),
                prompt_token_ids: vec![0, SECOND_PROMPT_TOKEN],
                max_new_tokens: FULL_GENERATED_TOKEN_COUNT,
                expected_generated_token_ids: vec![FIRST_EXPECTED_TOKEN, SECOND_EXPECTED_TOKEN],
            }],
        };
        let bounded = bounded_profiler_workload(&workload).expect("bounded profiler workload");
        assert_eq!(
            bounded.requests[0].max_new_tokens,
            PROFILER_GENERATED_TOKEN_COUNT
        );
        assert_eq!(
            bounded.requests[0].expected_generated_token_ids,
            vec![FIRST_EXPECTED_TOKEN]
        );
    }

    #[test]
    fn bounded_profiler_workload_rejects_missing_expected_tokens() {
        const SECOND_PROMPT_TOKEN: u32 = 2;
        let workload = BenchmarkWorkload {
            schema_version: SCHEMA_VERSION,
            cpu_reference_profile_blake3: "profile".to_owned(),
            cpu_reference_method: CPU_REFERENCE_METHOD.to_owned(),
            requests: vec![BenchmarkRequest {
                request_id: "request-0".to_owned(),
                prompt_token_ids: vec![0, SECOND_PROMPT_TOKEN],
                max_new_tokens: PROFILER_GENERATED_TOKEN_COUNT,
                expected_generated_token_ids: Vec::new(),
            }],
        };
        assert!(bounded_profiler_workload(&workload).is_err());
    }

    #[test]
    fn profiler_marker_guard_accepts_complete_output() {
        assert!(!profiler_output_has_dropped_markers(
            "device profiler completed without warnings"
        ));
    }

    #[test]
    fn profiler_marker_guard_rejects_dropped_markers() {
        assert!(profiler_output_has_dropped_markers(&format!(
            "warning: {PROFILER_DROPPED_MARKER}"
        )));
    }

    #[test]
    fn stall_capture_cores_select_and_compare_extremes() {
        const STEP_A: f64 = 0.1;
        const STEP_B: f64 = 0.3;
        const STEP_C: f64 = 0.2;
        const NORMAL_WALL_SECONDS: f64 = 0.8;
        const MIDDLE_WALL_SECONDS: f64 = 0.9;
        const SLOW_WALL_SECONDS: f64 = 1.2;
        const NORMAL_ZONE_CYCLES: u64 = 100;
        const SLOW_ZONE_CYCLES: u64 = 110;
        const NORMAL_STEP_SUM_SECONDS: f64 = 0.7;
        const SLOW_MAXIMUM_STEP_SECONDS: f64 = 0.6;
        const SLOW_STEP_SUM_SECONDS: f64 = 1.1;
        const EXPECTED_WALL_RATIO: f64 = 1.5;
        const EXPECTED_CYCLE_RATIO: f64 = 1.1;
        const EXPECTED_MAXIMUM_STEP_RATIO: f64 = 2.0;
        const EXPECTED_SUMMED_STEP_RATIO: f64 = SLOW_STEP_SUM_SECONDS / NORMAL_STEP_SUM_SECONDS;
        const EXPECTED_FETCH_QUEUE_EXTRA_FRACTION: f64 = 0.5;
        assert_eq!(maximum_step(&[STEP_A, STEP_B, STEP_C]), Ok((1, STEP_B)));
        assert_eq!(
            stall_capture_extremes(&[NORMAL_WALL_SECONDS, SLOW_WALL_SECONDS, MIDDLE_WALL_SECONDS,]),
            Ok((0, 1))
        );
        let normal = StallCaptureSample {
            sample_index: 0,
            wave_elapsed_seconds: NORMAL_WALL_SECONDS,
            start_skew_seconds: 0.0,
            workers: vec![stall_capture_worker(
                0,
                NORMAL_ZONE_CYCLES,
                STEP_B,
                NORMAL_STEP_SUM_SECONDS,
            )],
        };
        let slow = StallCaptureSample {
            sample_index: 1,
            wave_elapsed_seconds: SLOW_WALL_SECONDS,
            start_skew_seconds: 0.0,
            workers: vec![stall_capture_worker(
                0,
                SLOW_ZONE_CYCLES,
                SLOW_MAXIMUM_STEP_SECONDS,
                SLOW_STEP_SUM_SECONDS,
            )],
        };
        let comparison = compare_stall_capture_samples(&normal, &slow).expect("stall comparison");
        assert_eq!(comparison.normal_sample_index, 0);
        assert_eq!(comparison.slow_sample_index, 1);
        assert!(
            (comparison.wall_slowdown_ratio - EXPECTED_WALL_RATIO).abs() <= TEST_RATIO_TOLERANCE
        );
        assert!(
            (comparison.cards[0].total_zone_cycle_ratio - EXPECTED_CYCLE_RATIO).abs()
                <= TEST_RATIO_TOLERANCE
        );
        assert!(
            (comparison.cards[0].maximum_step_wall_ratio - EXPECTED_MAXIMUM_STEP_RATIO).abs()
                <= TEST_RATIO_TOLERANCE
        );
        assert!(
            (comparison.cards[0].zone_total_cycle_ratios["TRISC-KERNEL"] - EXPECTED_CYCLE_RATIO)
                .abs()
                <= TEST_RATIO_TOLERANCE
        );
        assert!(
            (comparison.cards[0].boundary.summed_total_ratio - EXPECTED_SUMMED_STEP_RATIO).abs()
                <= TEST_RATIO_TOLERANCE
        );
        assert!(
            (comparison.cards[0].hardware_counters.cycles_ratio - EXPECTED_SUMMED_STEP_RATIO).abs()
                <= TEST_RATIO_TOLERANCE
        );
        assert_eq!(
            comparison.cards[0].boundary.layer_submission_ratios.len(),
            RWKV7_LAYER_COUNT
        );
        assert!(
            (comparison.cards[0]
                .fetch_queue
                .extra_wait_fraction_of_wall_time_extra
                .expect("positive wall-time delta")
                - EXPECTED_FETCH_QUEUE_EXTRA_FRACTION)
                .abs()
                <= TEST_RATIO_TOLERANCE
        );
    }

    #[test]
    fn stall_capture_cores_reject_malformed_evidence() {
        const TEST_WALL_SECONDS: f64 = 0.8;
        const TEST_ZONE_CYCLES: u64 = 100;
        const TEST_MAXIMUM_STEP_SECONDS: f64 = 0.3;
        const TEST_STEP_SUM_SECONDS: f64 = 0.7;
        assert!(maximum_step(&[]).is_err());
        assert!(maximum_step(&[f64::NAN]).is_err());
        assert!(stall_capture_extremes(&[]).is_err());
        let normal = StallCaptureSample {
            sample_index: 0,
            wave_elapsed_seconds: TEST_WALL_SECONDS,
            start_skew_seconds: 0.0,
            workers: vec![stall_capture_worker(
                0,
                TEST_ZONE_CYCLES,
                TEST_MAXIMUM_STEP_SECONDS,
                TEST_STEP_SUM_SECONDS,
            )],
        };
        let changed_owner = StallCaptureSample {
            sample_index: 1,
            wave_elapsed_seconds: 1.0,
            start_skew_seconds: 0.0,
            workers: vec![stall_capture_worker(
                1,
                TEST_ZONE_CYCLES,
                TEST_MAXIMUM_STEP_SECONDS,
                TEST_STEP_SUM_SECONDS,
            )],
        };
        assert!(compare_stall_capture_samples(&normal, &changed_owner).is_err());
        let mut missing_zone = normal.clone();
        missing_zone.workers[0].profiler.zones.clear();
        assert!(compare_stall_capture_samples(&normal, &missing_zone).is_err());
        let mut malformed_boundary = normal.clone();
        malformed_boundary.workers[0].step_boundary_timings[0]
            .layer_submission_seconds
            .clear();
        assert!(
            compare_boundary_timings(
                &normal.workers[0].step_boundary_timings,
                &malformed_boundary.workers[0].step_boundary_timings,
            )
            .is_err()
        );
        let mut malformed_fetch_queue = normal.clone();
        malformed_fetch_queue.workers[0]
            .fetch_queue_diagnostics
            .poll_iterations = 0;
        assert!(compare_stall_capture_samples(&normal, &malformed_fetch_queue).is_err());
        let mut malformed_hardware_counters = normal.clone();
        malformed_hardware_counters.workers[0]
            .hardware_counter_diagnostics
            .instructions
            .value = 0;
        assert!(compare_stall_capture_samples(&normal, &malformed_hardware_counters).is_err());
    }

    #[test]
    fn wkv_policy_parser_accepts_production_and_candidate_modes() {
        let production = vec![
            "profile".to_owned(),
            "workload".to_owned(),
            "output".to_owned(),
        ];
        assert_eq!(
            parse_wkv_policy(&production),
            Ok((WkvPolicy::Production, PROFILE_ARGUMENT_INDEX))
        );
        let candidate = vec![
            FUSED_CANDIDATE_MODE.to_owned(),
            "profile".to_owned(),
            "workload".to_owned(),
            "output".to_owned(),
        ];
        assert_eq!(
            parse_wkv_policy(&candidate),
            Ok((WkvPolicy::FusedCandidate, PROFILE_ARGUMENT_INDEX + 1))
        );
    }

    #[test]
    fn wkv_policy_parser_rejects_an_unknown_mode() {
        let invalid = vec![
            "--unknown".to_owned(),
            "profile".to_owned(),
            "workload".to_owned(),
            "output".to_owned(),
        ];
        assert!(parse_wkv_policy(&invalid).is_err());
    }

    #[test]
    fn ab_order_and_speedup_core_accepts_a_stable_gain() {
        assert_eq!(
            order_balanced_wkv_policies(0),
            [WkvPolicy::GenericReference, WkvPolicy::FusedCandidate]
        );
        assert_eq!(
            order_balanced_wkv_policies(1),
            [WkvPolicy::FusedCandidate, WkvPolicy::GenericReference]
        );
        assert_eq!(
            order_balanced_persistent_dispatches(0),
            [
                PersistentDispatchControl::Normal,
                PersistentDispatchControl::Candidate,
            ]
        );
        assert_eq!(
            order_balanced_persistent_dispatches(1),
            [
                PersistentDispatchControl::Candidate,
                PersistentDispatchControl::Normal,
            ]
        );
        assert_eq!(
            paired_speedups(&[10.0, 20.0], &[15.0, 30.0]),
            Ok(vec![1.5, 1.5])
        );
        assert_eq!(
            paired_elapsed_speedups(&[3.0, 6.0], &[1.0, 2.0]),
            Ok(vec![3.0, 3.0])
        );
    }

    #[test]
    fn ab_speedup_core_rejects_malformed_rates() {
        assert!(paired_speedups(&[], &[]).is_err());
        assert!(paired_speedups(&[10.0], &[10.0, 20.0]).is_err());
        assert!(paired_speedups(&[0.0], &[10.0]).is_err());
        assert!(paired_speedups(&[10.0], &[f64::NAN]).is_err());
        assert!(paired_elapsed_speedups(&[], &[]).is_err());
        assert!(paired_elapsed_speedups(&[1.0], &[1.0, 2.0]).is_err());
        assert!(paired_elapsed_speedups(&[1.0], &[0.0]).is_err());
        assert!(paired_elapsed_speedups(&[f64::NAN], &[1.0]).is_err());
    }

    #[test]
    fn synchronized_wave_timing_accepts_overlapping_workers() {
        const FIRST_START_NS: u64 = 1_000_000_000;
        const SECOND_START_NS: u64 = FIRST_START_NS + 1_000_000;
        const FIRST_ELAPSED_NS: u64 = 800_000_000;
        const SECOND_ELAPSED_NS: u64 = 810_000_000;
        const FIRST_ELAPSED_SECONDS: f64 = 0.8;
        const SECOND_ELAPSED_SECONDS: f64 = 0.81;
        const EXPECTED_WAVE_SECONDS: f64 = 0.811;
        const EXPECTED_SKEW_SECONDS: f64 = 0.001;
        let timing = synchronized_wave_timing(&[
            WorkerTiming {
                start_monotonic_ns: FIRST_START_NS,
                end_monotonic_ns: FIRST_START_NS + FIRST_ELAPSED_NS,
                elapsed_seconds: FIRST_ELAPSED_SECONDS,
            },
            WorkerTiming {
                start_monotonic_ns: SECOND_START_NS,
                end_monotonic_ns: SECOND_START_NS + SECOND_ELAPSED_NS,
                elapsed_seconds: SECOND_ELAPSED_SECONDS,
            },
        ])
        .expect("synchronized worker timing");
        assert!((timing.elapsed_seconds - EXPECTED_WAVE_SECONDS).abs() < f64::EPSILON);
        assert!((timing.start_skew_seconds - EXPECTED_SKEW_SECONDS).abs() < f64::EPSILON);
    }

    #[test]
    fn synchronized_wave_timing_rejects_excessive_start_skew() {
        const FIRST_START_NS: u64 = 1_000_000_000;
        const SECOND_START_NS: u64 = FIRST_START_NS + 20_000_000;
        const ELAPSED_NS: u64 = 800_000_000;
        const ELAPSED_SECONDS: f64 = 0.8;
        assert!(
            synchronized_wave_timing(&[
                WorkerTiming {
                    start_monotonic_ns: FIRST_START_NS,
                    end_monotonic_ns: FIRST_START_NS + ELAPSED_NS,
                    elapsed_seconds: ELAPSED_SECONDS,
                },
                WorkerTiming {
                    start_monotonic_ns: SECOND_START_NS,
                    end_monotonic_ns: SECOND_START_NS + ELAPSED_NS,
                    elapsed_seconds: ELAPSED_SECONDS,
                },
            ])
            .is_err()
        );
    }

    #[test]
    fn synchronized_wave_timing_rejects_malformed_ranges() {
        const START_NS: u64 = 1_000_000_000;
        assert!(
            synchronized_wave_timing(&[WorkerTiming {
                start_monotonic_ns: START_NS,
                end_monotonic_ns: START_NS,
                elapsed_seconds: 0.0,
            }])
            .is_err()
        );
    }

    #[test]
    fn profiler_card_parser_accepts_a_logical_data_row() {
        let row = b"1,1,2,BRISC,52564,14597231684249\n";
        assert_eq!(
            profiler_card_from_line(row, PROFILER_HEADER_LINE_COUNT),
            Ok(Some(1))
        );
    }

    #[test]
    fn profiler_card_parser_rejects_a_missing_or_malformed_card() {
        assert!(profiler_card_from_line(b"not-a-card,row\n", PROFILER_HEADER_LINE_COUNT).is_err());
        assert!(
            profiler_card_from_line(b"missing separator\n", PROFILER_HEADER_LINE_COUNT).is_err()
        );
    }

    #[test]
    fn profiler_event_parser_accepts_a_zone_start() {
        let event = profiler_event_from_line(
            b"0,1,2,BRISC,3,100,0,42,,,fused_wkv_compute,ZONE_START,1,k.cpp,\n",
            PROFILER_HEADER_LINE_COUNT,
        )
        .expect("valid profiler row")
        .expect("zone event");
        assert_eq!(event.logical_card_id, FILTERED_WORKER_LOGICAL_CARD_ID);
        assert_eq!(event.key, "1|2|BRISC|3|fused_wkv_compute");
        assert_eq!(event.zone_name, "fused_wkv_compute");
        assert_eq!(event.cycle, 100);
        assert_eq!(event.event_type, ProfilerEventType::Start);
    }

    #[test]
    fn profiler_event_parser_rejects_an_invalid_cycle() {
        assert!(
            profiler_event_from_line(
                b"0,1,2,BRISC,3,bad,0,42,,,fused_wkv_compute,ZONE_END,1,k.cpp,\n",
                PROFILER_HEADER_LINE_COUNT,
            )
            .is_err()
        );
    }

    #[test]
    fn profiler_binding_accepts_filtered_workers_and_rejects_other_ids() {
        assert!(profiler_binding_matches(
            FILTERED_WORKER_LOGICAL_CARD_ID,
            ONE_CARD_IDS[0]
        ));
        assert!(profiler_binding_matches(
            FILTERED_WORKER_LOGICAL_CARD_ID,
            TWO_CARD_IDS[1]
        ));
        assert!(!profiler_binding_matches(1, ONE_CARD_IDS[0]));
        assert!(!profiler_binding_matches(
            FILTERED_WORKER_LOGICAL_CARD_ID,
            TWO_CARD_IDS.len()
        ));
    }

    #[test]
    fn persistent_candidate_uses_a_bounded_setup_timeout() {
        assert!(!persistent_candidate_flag(None).expect("absent flag"));
        assert!(!persistent_candidate_flag(Some("0")).expect("disabled flag"));
        assert!(persistent_candidate_flag(Some("1")).expect("enabled flag"));
        assert_eq!(
            worker_barrier_timeout(true).as_secs(),
            PERSISTENT_WORKER_BARRIER_TIMEOUT_SECONDS
        );
    }

    #[test]
    fn persistent_candidate_rejects_an_invalid_flag() {
        assert!(persistent_candidate_flag(Some("enabled")).is_err());
        assert_eq!(
            worker_barrier_timeout(false).as_secs(),
            NORMAL_WORKER_BARRIER_TIMEOUT_SECONDS
        );
    }

    #[test]
    fn cpu_reference_lookup_accepts_known_and_rejects_unknown_requests() {
        let workload = BenchmarkWorkload {
            schema_version: SCHEMA_VERSION,
            cpu_reference_profile_blake3: "profile".to_owned(),
            cpu_reference_method: CPU_REFERENCE_METHOD.to_owned(),
            requests: vec![BenchmarkRequest {
                request_id: "known".to_owned(),
                prompt_token_ids: vec![0, 2],
                max_new_tokens: 1,
                expected_generated_token_ids: vec![2],
            }],
        };
        assert_eq!(
            expected_generated_tokens(&workload, "known"),
            Some(&[2][..])
        );
        assert_eq!(expected_generated_tokens(&workload, "unknown"), None);
    }
}
