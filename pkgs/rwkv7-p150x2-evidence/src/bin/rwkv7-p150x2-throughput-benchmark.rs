//! Physical one-card versus two-card throughput benchmark shell.
//!
//! Scheduling and admission stay in the pure throughput module. This binary
//! owns process launch, file retention, and profiler-source checks.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};

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
const PROFILE_ARGUMENT_INDEX: usize = 0;
const REQUESTS_ARGUMENT_INDEX: usize = 1;
const OUTPUT_ARGUMENT_INDEX: usize = 2;
const SCHEMA_VERSION: u32 = 1;
const ONE_CARD_IDS: [usize; 1] = [0];
const TWO_CARD_IDS: [usize; 2] = [0, 1];
const PROFILER_RELATIVE_PATH: &str = "profiler/.logs/profile_log_device.csv";
const PROFILER_SUMMARY_FILENAME: &str = "profiler-attribution.csv";
const PROFILER_HEADER_LINE_COUNT: usize = 2;
const CPU_REFERENCE_METHOD: &str = "pinned_fp32_full_sequence_greedy_argmax";
const KIBIBYTE: usize = 1024;
const MAX_PROFILER_LINE_BYTES: usize = KIBIBYTE * KIBIBYTE;

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
    requests: Vec<WorkerRequestEntry>,
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
    physical_hardware: bool,
    profiler_attributed: bool,
    elapsed_seconds: f64,
    requests: Vec<WorkerRequestMeasurement>,
    execution: WorkerExecution,
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
}

struct WorkerProcess {
    card_id: usize,
    root: PathBuf,
    output_path: PathBuf,
    child: std::process::Child,
}

struct WorkerLaunchRequest<'a> {
    runtime: &'a Path,
    run_root: &'a Path,
    run_id: &'a str,
    card_id: usize,
    card_ids: &'a [usize],
    workload: &'a BenchmarkWorkload,
    profile: &'a ThroughputProfile,
    collect_profiler: bool,
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
    let path = output_root.join(format!("machine-facts-{name}.json"));
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
    if workload.requests.len() > profile.scheduler.per_card_slot_count {
        return Err(format!(
            "workload has {} requests; one-card active capacity is {}",
            workload.requests.len(),
            profile.scheduler.per_card_slot_count
        ));
    }
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
        .filter(|assignment| assignment.card_id == card_id)
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
    Ok(WorkerRequest {
        schema_version: SCHEMA_VERSION,
        run_id: format!("{run_id}-card-{card_id}"),
        requests,
    })
}

fn launch_worker(request: WorkerLaunchRequest<'_>) -> Result<WorkerProcess, String> {
    let WorkerLaunchRequest {
        runtime,
        run_root,
        run_id,
        card_id,
        card_ids,
        workload,
        profile,
        collect_profiler,
    } = request;
    let worker_root = run_root.join(format!("card-{card_id}"));
    fs::create_dir_all(worker_root.join("profiler"))
        .map_err(|error| format!("cannot create worker root: {error}"))?;
    fs::create_dir_all(worker_root.join("logs"))
        .map_err(|error| format!("cannot create worker logs: {error}"))?;
    let request_path = worker_root.join("request.json");
    let output_path = worker_root.join("receipt.json");
    write_new_json(
        &request_path,
        &worker_request(run_id, card_id, card_ids, workload, profile)?,
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
        .env("TT_METAL_LOGS_PATH", worker_root.join("logs"))
        .env("TT_METAL_PROFILER_DIR", worker_root.join("profiler"))
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
        root: worker_root,
        output_path,
        child,
    })
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
            if card_id != worker.card_id {
                return Err(format!(
                    "profiler line {} attributes card {card_id}, expected {}",
                    line_index + 1,
                    worker.card_id
                ));
            }
            data_row_count = data_row_count
                .checked_add(1)
                .ok_or_else(|| "profiler row count overflowed".to_owned())?;
        }
        line_index = line_index
            .checked_add(1)
            .ok_or_else(|| "profiler line count overflowed".to_owned())?;
    }
    if data_row_count == 0 {
        return Err(format!(
            "profiler has no physical-card rows for card {}",
            worker.card_id
        ));
    }
    let summary_path = worker.root.join(PROFILER_SUMMARY_FILENAME);
    let summary = format!(
        "card_id,raw_blake3,raw_byte_count,data_row_count,raw_csv_retained\n{},{},{},{},false\n",
        worker.card_id,
        hasher.finalize().to_hex(),
        raw_byte_count,
        data_row_count
    );
    fs::write(&summary_path, summary)
        .map_err(|error| format!("cannot write {}: {error}", summary_path.display()))?;
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
) -> Result<(), String> {
    if receipt.schema_version != SCHEMA_VERSION
        || receipt.run_id.is_empty()
        || receipt.physical_device_ids != vec![worker.card_id]
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
    {
        return Err(format!(
            "card {} contaminated its timed region",
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
    } = request;
    let run_id = format!("{topology}-{sample_index}");
    let run_root = root.join(&run_id);
    fs::create_dir(&run_root).map_err(|error| format!("cannot create run root: {error}"))?;
    let mut workers = card_ids
        .iter()
        .map(|card_id| {
            launch_worker(WorkerLaunchRequest {
                runtime,
                run_root: &run_root,
                run_id: &run_id,
                card_id: *card_id,
                card_ids,
                workload,
                profile,
                collect_profiler,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    for worker in &mut workers {
        let status = worker
            .child
            .wait()
            .map_err(|error| format!("cannot wait for card {}: {error}", worker.card_id))?;
        if !status.success() {
            return Err(format!(
                "card {} worker failed with {status}; see {}",
                worker.card_id,
                worker.root.display()
            ));
        }
    }
    let mut elapsed_seconds = 0.0_f64;
    let mut requests = Vec::new();
    for worker in &workers {
        let receipt: WorkerReceipt = read_json(&worker.output_path)?;
        validate_worker(worker, &receipt, workload, profile, collect_profiler)?;
        elapsed_seconds = elapsed_seconds.max(receipt.elapsed_seconds);
        requests.extend(
            receipt
                .requests
                .into_iter()
                .map(|request| request.measurement),
        );
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
    if arguments.len() != EXPECTED_ARGUMENT_COUNT {
        return Err(
            "usage: rwkv7-p150x2-throughput-benchmark PROFILE_JSON REQUESTS_JSON NEW_OUTPUT_ROOT"
                .to_owned(),
        );
    }
    let profile_path = require_input_file(&arguments[PROFILE_ARGUMENT_INDEX])?;
    let workload_path = require_input_file(&arguments[REQUESTS_ARGUMENT_INDEX])?;
    let output_root = require_new_output_root(&arguments[OUTPUT_ARGUMENT_INDEX])?;
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
    })?;
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
            "timed_samples_profiled": false,
        }),
    )?;

    let mut one_card_runs = Vec::with_capacity(profile.repeat_count);
    let mut two_card_runs = Vec::with_capacity(profile.repeat_count);
    for sample_index in 0..profile.repeat_count {
        one_card_runs.push(run_sample(SampleRequest {
            runtime: &runtime,
            root: &output_root,
            topology: "single",
            sample_index,
            card_ids: &ONE_CARD_IDS,
            workload: &workload,
            profile: &profile,
            collect_profiler: false,
            profiler_attested: true,
        })?);
    }
    for sample_index in 0..profile.repeat_count {
        two_card_runs.push(run_sample(SampleRequest {
            runtime: &runtime,
            root: &output_root,
            topology: "dual",
            sample_index,
            card_ids: &TWO_CARD_IDS,
            workload: &workload,
            profile: &profile,
            collect_profiler: false,
            profiler_attested: true,
        })?);
    }
    let machine_facts_after = capture_machine_snapshot(&tt_smi, &output_root, "after")?;
    let raw = RawThroughputEvidence {
        schema_version: SCHEMA_VERSION,
        profile_identity_blake3: identity,
        machine_facts: MachineFactsEvidence {
            source: MACHINE_FACT_SOURCE.to_owned(),
            before_blake3: machine_facts_before,
            after_blake3: machine_facts_after,
            physical_device_ids: TWO_CARD_IDS,
        },
        execution: ExecutionEvidence {
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
        },
        one_card_runs,
        two_card_runs,
    };
    let raw_path = output_root.join("raw-throughput-evidence.json");
    write_new_json(&raw_path, &raw)?;
    let receipt = summarize_throughput(&raw, &profile).map_err(|error| error.to_string())?;
    let receipt_path = output_root.join("throughput-receipt.json");
    write_new_json(&receipt_path, &receipt)?;
    let admission =
        admit_throughput_receipt(&receipt, &raw, &profile).map_err(|error| error.to_string())?;
    let admission_path = output_root.join("throughput-admission.json");
    write_new_json(
        &admission_path,
        &json!({
            "status": "admitted",
            "identity_blake3": admission.identity_blake3,
            "dual_card_speedup": admission.dual_card_speedup,
            "scaling_efficiency": admission.scaling_efficiency,
            "p99_latency_regression_fraction": admission.p99_latency_regression_fraction,
            "aggregate_tokens_per_second": admission.aggregate_tokens_per_second,
        }),
    )?;
    let manifest = BTreeMap::from([
        ("raw", raw_path),
        ("receipt", receipt_path),
        ("admission", admission_path),
        ("profiler_attestation", profiler_attestation_path),
    ]);
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
    fn profiler_card_parser_accepts_a_physical_data_row() {
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
