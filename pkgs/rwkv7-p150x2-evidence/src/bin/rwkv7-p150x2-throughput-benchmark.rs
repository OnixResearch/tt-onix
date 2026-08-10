//! Physical one-card versus two-card throughput benchmark shell.
//!
//! Scheduling and admission stay in the pure throughput module. This binary
//! owns process launch, file retention, and profiler-source checks.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use rwkv7_p150x2_evidence::throughput::{
    ExecutionEvidence, MachineFactsEvidence, RawThroughputEvidence, RequestMeasurement,
    ThroughputProfile, ThroughputRun, admit_throughput_receipt, deterministic_slot_assignments,
    profile_identity, summarize_throughput,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

const DEVICE_RUNTIME_PATH: Option<&str> = option_env!("RWKV7_DEVICE_RUNTIME");
const TT_SMI_PATH: Option<&str> = option_env!("TT_SMI_PATH");
const MACHINE_FACT_SOURCE: &str = "tt-smi -s --snapshot_no_tty";
const PROGRAM_ARGUMENT_COUNT: usize = 1;
const EXPECTED_ARGUMENT_COUNT: usize = 3;
const FUSED_CANDIDATE_ARGUMENT_COUNT: usize = EXPECTED_ARGUMENT_COUNT + 1;
const FUSED_CANDIDATE_MODE: &str = "--fused-wkv-candidate";
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
const CPU_REFERENCE_METHOD: &str = "pinned_fp32_full_sequence_greedy_argmax";
const KIBIBYTE: usize = 1024;
const MAX_PROFILER_LINE_BYTES: usize = KIBIBYTE * KIBIBYTE;
// r[impl tenstorrent.rwkv7_p150x2.scalable_card_local_batching.valid]
// These match the exported `throughput-capacity-profile.ncl` memory bounds.
const WORKER_CAPACITY_TEMPORARY_BYTES_PER_SLOT: u64 = 5 * 1024 * 1024;
const WORKER_CAPACITY_DEVICE_BUDGET_BYTES: u64 = 3 * 1024 * 1024 * 1024;
const WORKER_CAPACITY_SAFETY_RESERVE_BYTES: u64 = 256 * 1024 * 1024;
const WORKER_INITIALIZATION_READY_BARRIER_NAME: &str = "initialization-ready.barrier";
const WORKER_WARMUP_START_BARRIER_NAME: &str = "warmup-start.barrier";
const WORKER_TIMED_READY_BARRIER_NAME: &str = "timed-ready.barrier";
const WORKER_TIMED_START_BARRIER_NAME: &str = "timed-start.barrier";
const WORKER_BARRIER_TIMEOUT: Duration = Duration::from_secs(120);
const WORKER_BARRIER_POLL_INTERVAL: Duration = Duration::from_millis(1);
const MAXIMUM_WORKER_START_SKEW_SECONDS: f64 = 0.010;
const WORKER_ELAPSED_TOLERANCE_SECONDS: f64 = 0.001;
const NANOSECONDS_PER_SECOND: f64 = 1_000_000_000.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WkvPolicy {
    Production,
    FusedCandidate,
}

impl WkvPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Production => "production",
            Self::FusedCandidate => "fused-candidate",
        }
    }
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
        return [WkvPolicy::Production, WkvPolicy::FusedCandidate];
    }
    [WkvPolicy::FusedCandidate, WkvPolicy::Production]
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
    elapsed_seconds: f64,
    timed_start_monotonic_ns: u64,
    timed_end_monotonic_ns: u64,
    capacity: WorkerReceiptCapacity,
    effective_batch_size: usize,
    queue_occupancy_fraction: f64,
    batch_step_count: usize,
    host_gap_accumulated_seconds: f64,
    requests: Vec<WorkerRequestMeasurement>,
    execution: WorkerExecution,
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

#[derive(Clone, Debug, Default, Serialize)]
struct ProfilerZoneAggregate {
    start_count: usize,
    complete_count: usize,
    total_cycles: u64,
    maximum_cycles: u64,
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
    wkv_policy: WkvPolicy,
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
    profiler_attested: bool,
    wkv_policy: WkvPolicy,
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

fn runtime_executable() -> Result<PathBuf, String> {
    let package = DEVICE_RUNTIME_PATH
        .ok_or_else(|| "RWKV7_DEVICE_RUNTIME was not embedded at build time".to_owned())?;
    let executable = Path::new(package).join("bin/rwkv7-p150x2-runtime");
    if !executable.starts_with("/nix/store/") || !executable.is_file() {
        return Err("embedded device runtime is not a Nix store executable".to_owned());
    }
    Ok(executable)
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
        saw_short_prompt |=
            request.prompt_token_ids.len() == profile.workload.short_prompt_token_count;
        saw_long_prompt |=
            request.prompt_token_ids.len() == profile.workload.long_prompt_token_count;
    }
    if !saw_short_prompt || !saw_long_prompt {
        return Err("workload does not contain both named short and long prompts".to_owned());
    }
    Ok(())
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
        wkv_policy,
    } = request;
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
    let stdout = File::create(worker_root.join("stdout.log"))
        .map_err(|error| format!("cannot create worker stdout: {error}"))?;
    let stderr = File::create(worker_root.join("stderr.log"))
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
    if collect_profiler {
        command
            .env("TT_METAL_DEVICE_PROFILER", "1")
            .env("TT_METAL_PROFILER_MID_RUN_DUMP", "1");
    } else {
        command
            .env_remove("TT_METAL_DEVICE_PROFILER")
            .env_remove("TT_METAL_PROFILER_MID_RUN_DUMP");
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
        child,
    })
}

fn wait_for_worker_barriers(
    workers: &mut [WorkerProcess],
    barrier_name: &str,
    barrier_path: fn(&WorkerProcess) -> &Path,
) -> Result<(), String> {
    if workers.is_empty() {
        return Err("cannot synchronize an empty worker set".to_owned());
    }
    let deadline = Instant::now() + WORKER_BARRIER_TIMEOUT;
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
                WORKER_BARRIER_TIMEOUT.as_secs()
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

fn profiler_card_from_line(line: &[u8], line_index: usize) -> Result<Option<usize>, String> {
    if line_index < PROFILER_HEADER_LINE_COUNT || line.iter().all(u8::is_ascii_whitespace) {
        return Ok(None);
    }
    let separator = line
        .iter()
        .position(|byte| *byte == b',')
        .ok_or_else(|| format!("profiler line {} has no first separator", line_index + 1))?;
    let card_text = std::str::from_utf8(&line[..separator])
        .map_err(|error| {
            format!(
                "profiler line {} card is not UTF-8: {error}",
                line_index + 1
            )
        })?
        .trim();
    let card_id = card_text
        .parse::<usize>()
        .map_err(|error| format!("profiler line {} card is invalid: {error}", line_index + 1))?;
    Ok(Some(card_id))
}

fn profiler_event_from_line(
    line: &[u8],
    line_index: usize,
) -> Result<Option<ProfilerEvent>, String> {
    let Some(logical_card_id) = profiler_card_from_line(line, line_index)? else {
        return Ok(None);
    };
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(line);
    let record = reader
        .records()
        .next()
        .ok_or_else(|| format!("profiler line {} has no CSV record", line_index + 1))?
        .map_err(|error| format!("profiler line {} is invalid CSV: {error}", line_index + 1))?;
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

fn profiler_binding_matches(logical_card_id: usize, physical_card_id: usize) -> bool {
    logical_card_id == FILTERED_WORKER_LOGICAL_CARD_ID && TWO_CARD_IDS.contains(&physical_card_id)
}

fn summarize_profiler(worker: &WorkerProcess) -> Result<(), String> {
    let profiler_path = worker.root.join(PROFILER_RELATIVE_PATH);
    let profiler = File::open(&profiler_path)
        .map_err(|error| format!("cannot open {}: {error}", profiler_path.display()))?;
    let mut reader = BufReader::new(profiler);
    let mut line = Vec::new();
    let mut line_index = 0_usize;
    let mut data_row_count = 0_usize;
    let mut raw_byte_count = 0_u64;
    let mut hasher = blake3::Hasher::new();
    let mut active_zones = BTreeMap::<String, Vec<u64>>::new();
    let mut zone_aggregates = BTreeMap::<String, ProfilerZoneAggregate>::new();
    let mut unmatched_end_count = 0_usize;
    loop {
        line.clear();
        let byte_count = reader
            .read_until(b'\n', &mut line)
            .map_err(|error| format!("cannot stream {}: {error}", profiler_path.display()))?;
        if byte_count == 0 {
            break;
        }
        if byte_count > MAX_PROFILER_LINE_BYTES {
            return Err(format!(
                "profiler line {} exceeds {} bytes",
                line_index + 1,
                MAX_PROFILER_LINE_BYTES
            ));
        }
        raw_byte_count = raw_byte_count
            .checked_add(byte_count as u64)
            .ok_or_else(|| "profiler byte count overflowed".to_owned())?;
        hasher.update(&line);
        if let Some(card_id) = profiler_card_from_line(&line, line_index)? {
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
            if let Some(event) = profiler_event_from_line(&line, line_index)? {
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
    let raw_blake3 = hasher.finalize().to_hex().to_string();
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
    wkv_policy: WkvPolicy,
) -> Result<(), String> {
    if receipt.schema_version != SCHEMA_VERSION
        || receipt.run_id.is_empty()
        || receipt.physical_device_ids != vec![worker.card_id]
        || receipt.visible_physical_device_ids != vec![worker.card_id]
        || receipt.logical_device_ids != vec![FILTERED_WORKER_LOGICAL_CARD_ID]
        || !receipt.physical_hardware
        || receipt.profiler_attributed != collect_profiler
        || !receipt.elapsed_seconds.is_finite()
        || receipt.elapsed_seconds <= 0.0
    {
        return Err(format!(
            "card {} returned malformed worker evidence",
            worker.card_id
        ));
    }
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
    if actual_execution != expected_execution
        || receipt.execution.warmup_token_count != profile.timing.warmup_token_count
        || !receipt.execution.program_cache_stable
        || !receipt.execution.dynamic_token_and_state_bindings_validated
        || receipt.execution.state_readback_during_timing
        || receipt.execution.profiler_flush_during_timing
        || receipt.execution.control_replay_during_timing
        || receipt.execution.artifact_writes_during_timing
        || receipt.execution.inspector_during_timing
        || !receipt.execution.synchronized_timed_start
        || receipt.execution.wkv_execution_policy != wkv_policy.as_str()
        || receipt.execution.production_physically_admitted
    {
        return Err(format!(
            "card {} contaminated its timed region",
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
        profiler_attested,
        wkv_policy,
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
                wkv_policy,
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

fn run() -> Result<(), String> {
    let arguments = std::env::args()
        .skip(PROGRAM_ARGUMENT_COUNT)
        .collect::<Vec<_>>();
    let (wkv_policy, argument_offset) = parse_wkv_policy(&arguments)?;
    let profile_path = require_input_file(&arguments[argument_offset + PROFILE_ARGUMENT_INDEX])?;
    let workload_path = require_input_file(&arguments[argument_offset + REQUESTS_ARGUMENT_INDEX])?;
    let output_root = require_new_output_root(&arguments[argument_offset + OUTPUT_ARGUMENT_INDEX])?;
    let profile: ThroughputProfile = read_json(&profile_path)?;
    let workload: BenchmarkWorkload = read_json(&workload_path)?;
    validate_workload(&workload, &profile)?;
    let identity = profile_identity(&profile).map_err(|error| error.to_string())?;
    let runtime = runtime_executable()?;
    let tt_smi = tt_smi_executable()?;
    let machine_facts_before = capture_machine_snapshot(&tt_smi, &output_root, "before")?;

    let profiler_single = run_sample(SampleRequest {
        runtime: &runtime,
        root: &output_root,
        topology: "profiler-single",
        sample_index: 0,
        card_ids: &ONE_CARD_IDS,
        workload: &workload,
        profile: &profile,
        collect_profiler: true,
        profiler_attested: true,
        wkv_policy,
    })?;
    let profiler_dual = run_sample(SampleRequest {
        runtime: &runtime,
        root: &output_root,
        topology: "profiler-dual",
        sample_index: 0,
        card_ids: &TWO_CARD_IDS,
        workload: &workload,
        profile: &profile,
        collect_profiler: true,
        profiler_attested: true,
        wkv_policy,
    })?;
    let baseline_profiler_single = if wkv_policy == WkvPolicy::FusedCandidate {
        Some(run_sample(SampleRequest {
            runtime: &runtime,
            root: &output_root,
            topology: "profiler-generic-single",
            sample_index: 0,
            card_ids: &ONE_CARD_IDS,
            workload: &workload,
            profile: &profile,
            collect_profiler: true,
            profiler_attested: true,
            wkv_policy: WkvPolicy::Production,
        })?)
    } else {
        None
    };
    let baseline_profiler_dual = if wkv_policy == WkvPolicy::FusedCandidate {
        Some(run_sample(SampleRequest {
            runtime: &runtime,
            root: &output_root,
            topology: "profiler-generic-dual",
            sample_index: 0,
            card_ids: &TWO_CARD_IDS,
            workload: &workload,
            profile: &profile,
            collect_profiler: true,
            profiler_attested: true,
            wkv_policy: WkvPolicy::Production,
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
            "program_signature_filename": PROFILER_PROGRAM_SIGNATURE_FILENAME,
            "wkv_execution_policy": wkv_policy.as_str(),
            "production_physically_admitted": false,
        }),
    )?;

    let mut one_card_runs = Vec::with_capacity(profile.repeat_count);
    let mut two_card_runs = Vec::with_capacity(profile.repeat_count);
    let mut baseline_one_card_runs = Vec::with_capacity(profile.repeat_count);
    let mut baseline_two_card_runs = Vec::with_capacity(profile.repeat_count);
    for sample_index in 0..profile.repeat_count {
        let policies = if wkv_policy == WkvPolicy::FusedCandidate {
            order_balanced_wkv_policies(sample_index).to_vec()
        } else {
            vec![WkvPolicy::Production]
        };
        for sample_policy in policies {
            let topology = match sample_policy {
                WkvPolicy::Production if wkv_policy == WkvPolicy::FusedCandidate => {
                    "single-generic"
                }
                WkvPolicy::Production => "single",
                WkvPolicy::FusedCandidate => "single-fused",
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
                profiler_attested: true,
                wkv_policy: sample_policy,
            })?;
            match sample_policy {
                WkvPolicy::Production if wkv_policy == WkvPolicy::FusedCandidate => {
                    baseline_one_card_runs.push(run);
                }
                WkvPolicy::Production | WkvPolicy::FusedCandidate => one_card_runs.push(run),
            }
        }
    }
    for sample_index in 0..profile.repeat_count {
        let policies = if wkv_policy == WkvPolicy::FusedCandidate {
            order_balanced_wkv_policies(sample_index).to_vec()
        } else {
            vec![WkvPolicy::Production]
        };
        for sample_policy in policies {
            let topology = match sample_policy {
                WkvPolicy::Production if wkv_policy == WkvPolicy::FusedCandidate => "dual-generic",
                WkvPolicy::Production => "dual",
                WkvPolicy::FusedCandidate => "dual-fused",
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
                profiler_attested: true,
                wkv_policy: sample_policy,
            })?;
            match sample_policy {
                WkvPolicy::Production if wkv_policy == WkvPolicy::FusedCandidate => {
                    baseline_two_card_runs.push(run);
                }
                WkvPolicy::Production | WkvPolicy::FusedCandidate => two_card_runs.push(run),
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
    let baseline_raw = if wkv_policy == WkvPolicy::FusedCandidate {
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
                "production_physically_admitted": false,
                "identity_blake3": admission.identity_blake3,
                "dual_card_speedup": admission.dual_card_speedup,
                "scaling_efficiency": admission.scaling_efficiency,
                "p99_latency_regression_fraction": admission.p99_latency_regression_fraction,
                "aggregate_tokens_per_second": admission.aggregate_tokens_per_second,
            }),
        )?,
        Err(rejection) if wkv_policy == WkvPolicy::FusedCandidate => write_new_json(
            &admission_path,
            &json!({
                "status": "fused_candidate_scaling_rejected",
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
        let baseline_raw_path = output_root.join("generic-raw-throughput-evidence.json");
        write_new_json(&baseline_raw_path, &baseline)?;
        let baseline_receipt =
            summarize_throughput(&baseline, &profile).map_err(|error| error.to_string())?;
        let baseline_receipt_path = output_root.join("generic-throughput-receipt.json");
        write_new_json(&baseline_receipt_path, &baseline_receipt)?;
        let baseline_admission_path = output_root.join("generic-throughput-admission.json");
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

        let one_card_speedups = paired_speedups(
            &throughput_run_rates(&baseline.one_card_runs)?,
            &throughput_run_rates(&raw.one_card_runs)?,
        )?;
        let two_card_speedups = paired_speedups(
            &throughput_run_rates(&baseline.two_card_runs)?,
            &throughput_run_rates(&raw.two_card_runs)?,
        )?;
        let stable_gain = one_card_speedups.iter().all(|speedup| *speedup > 1.0)
            && two_card_speedups.iter().all(|speedup| *speedup > 1.0);
        let order_schedule = (0..profile.repeat_count)
            .map(|sample_index| {
                order_balanced_wkv_policies(sample_index)
                    .map(WkvPolicy::as_str)
                    .to_vec()
            })
            .collect::<Vec<_>>();
        let ab_receipt_path = output_root.join("fused-wkv-ab-receipt.json");
        write_new_json(
            &ab_receipt_path,
            &json!({
                "schema_version": SCHEMA_VERSION,
                "kind": "rwkv7_fused_wkv_physical_ab",
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
                "exact_generated_token_parity": true,
                "cpu_reference_profile_blake3": workload.cpu_reference_profile_blake3,
                "generic_receipt_identity_blake3": baseline_receipt.identity_blake3,
                "fused_receipt_identity_blake3": receipt.identity_blake3,
                "generic_one_card_tokens_per_second": baseline_receipt.one_card.aggregate_tokens_per_second,
                "fused_one_card_tokens_per_second": receipt.one_card.aggregate_tokens_per_second,
                "generic_two_card_tokens_per_second": baseline_receipt.two_card.aggregate_tokens_per_second,
                "fused_two_card_tokens_per_second": receipt.two_card.aggregate_tokens_per_second,
                "one_card_paired_speedups": one_card_speedups,
                "two_card_paired_speedups": two_card_speedups,
                "stable_gain": stable_gain,
                "profiler_attestation": "profiler-attestation.json",
                "machine_facts_before": "machine-before.json",
                "machine_facts_after": "machine-after.json",
            }),
        )?;
        manifest.insert("generic_raw", baseline_raw_path);
        manifest.insert("generic_receipt", baseline_receipt_path);
        manifest.insert("generic_admission", baseline_admission_path);
        manifest.insert("fused_wkv_ab_receipt", ab_receipt_path);
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
            [WkvPolicy::Production, WkvPolicy::FusedCandidate]
        );
        assert_eq!(
            order_balanced_wkv_policies(1),
            [WkvPolicy::FusedCandidate, WkvPolicy::Production]
        );
        assert_eq!(
            paired_speedups(&[10.0, 20.0], &[15.0, 30.0]),
            Ok(vec![1.5, 1.5])
        );
    }

    #[test]
    fn ab_speedup_core_rejects_malformed_rates() {
        assert!(paired_speedups(&[], &[]).is_err());
        assert!(paired_speedups(&[10.0], &[10.0, 20.0]).is_err());
        assert!(paired_speedups(&[0.0], &[10.0]).is_err());
        assert!(paired_speedups(&[10.0], &[f64::NAN]).is_err());
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
