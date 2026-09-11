use rwkv7_p150x2_evidence::persistent_decode_canary::{
    CanaryContext, CanaryError, CanaryInterlock, CanaryInterlockState, CanaryReason,
    ExpectedCanaryDispatch, PersistentDecodeCanaryReceipt, build_canary_context, canary_interlock,
    evaluate_completed_canary, evaluate_failed_canary, exact_file_blake3,
    validate_physical_ab_receipt, validate_worker_interlock_receipts,
};
use rwkv7_p150x2_evidence::persistent_decode_monitoring::{
    MonitoringVerdict, collect_persistent_decode_monitoring,
};
use serde::Serialize;
use std::ffi::OsString;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode};
use std::thread;
use std::time::{Duration, Instant};

const ARGUMENT_COUNT: usize = 5;
const POLICY_ARGUMENT_INDEX: usize = 0;
const THROUGHPUT_ARGUMENT_INDEX: usize = 1;
const WORKLOAD_ARGUMENT_INDEX: usize = 2;
const MONITORING_ARGUMENT_INDEX: usize = 3;
const OUTPUT_ARGUMENT_INDEX: usize = 4;
const POLICY_READ_LIMIT_BYTES: usize = 65_536;
const INPUT_READ_LIMIT_BYTES: usize = 1_048_576;
const POLL_INTERVAL_MILLISECONDS: u64 = 100;
const CARD_BALANCE_PERIOD: usize = 2;
const CLEAN_EXIT_STATUS: u8 = 0;
const WARNING_EXIT_STATUS: u8 = 2;
const CRITICAL_EXIT_STATUS: u8 = 3;
const INPUT_ERROR_EXIT_STATUS: u8 = 64;
const PERSISTENT_AB_ENVIRONMENT: &str = "RWKV7_PERSISTENT_DECODE_AB";
const PERSISTENT_CANDIDATE_ENVIRONMENT: &str = "RWKV7_PERSISTENT_DECODE_CANDIDATE";
const CANARY_INTERLOCK_ENVIRONMENT: &str = "RWKV7_PERSISTENT_DECODE_CANARY_INTERLOCK";
const CANARY_INTERLOCK_BLAKE3_ENVIRONMENT: &str = "RWKV7_PERSISTENT_DECODE_CANARY_INTERLOCK_BLAKE3";
const CANARY_BINDING_BLAKE3_ENVIRONMENT: &str = "RWKV7_PERSISTENT_DECODE_CANARY_BINDING_BLAKE3";
const PRIMARY_THREAD_CPUS_ENVIRONMENT: &str = "RWKV7_PRIMARY_THREAD_CPUS";
const PRIMARY_THREAD_FIFO_PRIORITY_ENVIRONMENT: &str = "RWKV7_PRIMARY_THREAD_FIFO_PRIORITY";
const BENCHMARK_BINARY_NAME: &str = "rwkv7-p150x2-throughput-benchmark";
const INTERLOCK_FILENAME: &str = "canary-interlock.json";
const CANARY_RECEIPT_FILENAME: &str = "canary-receipt.json";
const MONITORING_RECEIPT_FILENAME: &str = "candidate-monitoring-receipt.json";
const PHYSICAL_AB_RECEIPT_FILENAME: &str = "physical-ab-receipt.json";
const WORKER_KERNEL_CACHE_DIRECTORY: &str = ".tt-metal-kernel-cache";
const USAGE: &str = "usage: rwkv7-p150x2-persistent-decode-canary CANARY_POLICY_JSON THROUGHPUT_PROFILE_JSON WORKLOAD_JSON MONITORING_POLICY_JSON NEW_OUTPUT_ROOT";

#[derive(Serialize)]
struct ErrorEnvelope<'a> {
    schema_version: u32,
    error: ErrorBody<'a>,
}

#[derive(Serialize)]
struct ErrorBody<'a> {
    code: &'a str,
    detail: &'a str,
}

enum BenchmarkOutcome {
    Completed { duration_seconds: f64 },
    Failed { duration_seconds: f64 },
    TimedOut { duration_seconds: f64 },
}

fn read_bounded(path: &Path, limit: usize, label: &'static str) -> Result<String, CanaryError> {
    let metadata = fs::symlink_metadata(path).map_err(|_| {
        CanaryError::new(
            "canary_input_metadata_failed",
            format!("cannot inspect {label}"),
        )
    })?;
    if !metadata.file_type().is_file() {
        return Err(CanaryError::new(
            "canary_input_not_regular_file",
            format!("{label} is not a regular file"),
        ));
    }
    let declared_limit = u64::try_from(limit)
        .map_err(|error| CanaryError::new("canary_input_limit_overflow", error.to_string()))?;
    if metadata.len() > declared_limit {
        return Err(CanaryError::new(
            "canary_input_too_large",
            format!("{label} exceeds its byte limit"),
        ));
    }
    let mut file = File::open(path).map_err(|_| {
        CanaryError::new("canary_input_open_failed", format!("cannot open {label}"))
    })?;
    let read_limit = limit
        .checked_add(1)
        .ok_or_else(|| CanaryError::new("canary_input_limit_overflow", label))?;
    let mut bytes = Vec::with_capacity(read_limit);
    Read::by_ref(&mut file)
        .take(
            u64::try_from(read_limit).map_err(|error| {
                CanaryError::new("canary_input_limit_overflow", error.to_string())
            })?,
        )
        .read_to_end(&mut bytes)
        .map_err(|_| {
            CanaryError::new("canary_input_read_failed", format!("cannot read {label}"))
        })?;
    if bytes.len() > limit {
        return Err(CanaryError::new(
            "canary_input_too_large",
            format!("{label} exceeds its byte limit"),
        ));
    }
    String::from_utf8(bytes)
        .map_err(|_| CanaryError::new("canary_input_not_utf8", format!("{label} is not UTF-8")))
}

fn compact_json<T: Serialize>(value: &T) -> Result<Vec<u8>, CanaryError> {
    serde_json::to_vec(value)
        .map_err(|error| CanaryError::new("canary_serialization_failed", error.to_string()))
}

fn pretty_json<T: Serialize>(value: &T) -> Result<Vec<u8>, CanaryError> {
    serde_json::to_vec_pretty(value)
        .map_err(|error| CanaryError::new("canary_serialization_failed", error.to_string()))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), CanaryError> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|_| CanaryError::new("canary_output_exists", "canary output already exists"))?;
    file.write_all(bytes)
        .and_then(|_| file.sync_all())
        .map_err(|_| CanaryError::new("canary_output_write_failed", "cannot write canary output"))
}

fn temporary_path(path: &Path) -> PathBuf {
    let mut value = OsString::from(path.as_os_str());
    value.push(".tmp");
    PathBuf::from(value)
}

fn atomic_replace(path: &Path, bytes: &[u8]) -> Result<(), CanaryError> {
    let temporary = temporary_path(path);
    if temporary.exists() {
        return Err(CanaryError::new(
            "canary_temporary_output_exists",
            "canary temporary output already exists",
        ));
    }
    write_new(&temporary, bytes)?;
    fs::rename(&temporary, path)
        .map_err(|_| CanaryError::new("canary_output_replace_failed", "cannot replace interlock"))
}

fn sidecar_path(output_root: &Path) -> PathBuf {
    let mut value = OsString::from(output_root.as_os_str());
    value.push(".canary-interlock.json");
    PathBuf::from(value)
}

fn sibling_benchmark() -> Result<PathBuf, CanaryError> {
    let executable = std::env::current_exe().map_err(|_| {
        CanaryError::new(
            "canary_executable_lookup_failed",
            "cannot locate canary executable",
        )
    })?;
    let directory = executable.parent().ok_or_else(|| {
        CanaryError::new(
            "canary_executable_lookup_failed",
            "canary executable has no parent",
        )
    })?;
    let benchmark = directory.join(BENCHMARK_BINARY_NAME);
    if !benchmark.is_file() {
        return Err(CanaryError::new(
            "canary_benchmark_missing",
            "packaged throughput benchmark is missing",
        ));
    }
    Ok(benchmark)
}

fn cpu_list(context: &CanaryContext) -> String {
    context
        .policy
        .primary_thread_cpu_affinities
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn launch_benchmark(
    context: &CanaryContext,
    throughput_path: &Path,
    workload_path: &Path,
    output_root: &Path,
    interlock_path: &Path,
    interlock_blake3: &str,
) -> Result<Child, CanaryError> {
    let benchmark = sibling_benchmark()?;
    Command::new(benchmark)
        .arg(throughput_path)
        .arg(workload_path)
        .arg(output_root)
        .env(PERSISTENT_AB_ENVIRONMENT, "1")
        .env_remove(PERSISTENT_CANDIDATE_ENVIRONMENT)
        .env(CANARY_INTERLOCK_ENVIRONMENT, interlock_path)
        .env(CANARY_INTERLOCK_BLAKE3_ENVIRONMENT, interlock_blake3)
        .env(
            CANARY_BINDING_BLAKE3_ENVIRONMENT,
            &context.canary_binding_blake3,
        )
        .env(PRIMARY_THREAD_CPUS_ENVIRONMENT, cpu_list(context))
        .env(
            PRIMARY_THREAD_FIFO_PRIORITY_ENVIRONMENT,
            context.policy.primary_thread_fifo_priority.to_string(),
        )
        .spawn()
        .map_err(|_| CanaryError::new("canary_benchmark_start_failed", "cannot start benchmark"))
}

fn stop_child(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

fn wait_bounded(child: &mut Child, maximum_seconds: u64) -> Result<BenchmarkOutcome, CanaryError> {
    let start = Instant::now();
    let maximum = Duration::from_secs(maximum_seconds);
    let poll = Duration::from_millis(POLL_INTERVAL_MILLISECONDS);
    loop {
        if let Some(status) = child.try_wait().map_err(|_| {
            CanaryError::new("canary_benchmark_wait_failed", "cannot poll benchmark")
        })? {
            let duration_seconds = start.elapsed().as_secs_f64();
            return Ok(if status.success() {
                BenchmarkOutcome::Completed { duration_seconds }
            } else {
                BenchmarkOutcome::Failed { duration_seconds }
            });
        }
        if start.elapsed() >= maximum {
            child.kill().map_err(|_| {
                CanaryError::new(
                    "canary_benchmark_kill_failed",
                    "cannot stop timed-out benchmark",
                )
            })?;
            child.wait().map_err(|_| {
                CanaryError::new(
                    "canary_benchmark_wait_failed",
                    "cannot reap timed-out benchmark",
                )
            })?;
            return Ok(BenchmarkOutcome::TimedOut {
                duration_seconds: start.elapsed().as_secs_f64(),
            });
        }
        thread::sleep(poll);
    }
}

fn worker_receipt_path(output_root: &Path, topology: &str, sample: usize, card: u32) -> PathBuf {
    output_root
        .join(format!("{topology}-{sample}"))
        .join("wave-0")
        .join(format!("card-{card}"))
        .join("receipt.json")
}

fn load_worker_receipts(
    context: &CanaryContext,
    output_root: &Path,
    mode: ExpectedCanaryDispatch,
) -> Result<Vec<String>, CanaryError> {
    let topology_suffix = match mode {
        ExpectedCanaryDispatch::Normal => "normal",
        ExpectedCanaryDispatch::Persistent => "persistent",
    };
    let mut receipts = Vec::with_capacity(context.policy.required_candidate_telemetry_events);
    for sample in 0..context.policy.sample_count {
        let even_sample = sample.is_multiple_of(CARD_BALANCE_PERIOD);
        let single_card_index = match (mode, even_sample) {
            (ExpectedCanaryDispatch::Normal, true)
            | (ExpectedCanaryDispatch::Persistent, false) => 0,
            (ExpectedCanaryDispatch::Normal, false)
            | (ExpectedCanaryDispatch::Persistent, true) => 1,
        };
        receipts.push(read_bounded(
            &worker_receipt_path(
                output_root,
                &format!("single-{topology_suffix}"),
                sample,
                context.policy.physical_device_ids[single_card_index],
            ),
            INPUT_READ_LIMIT_BYTES,
            "one-card worker receipt",
        )?);
        for card in &context.policy.physical_device_ids {
            receipts.push(read_bounded(
                &worker_receipt_path(
                    output_root,
                    &format!("dual-{topology_suffix}"),
                    sample,
                    *card,
                ),
                INPUT_READ_LIMIT_BYTES,
                "two-card worker receipt",
            )?);
        }
    }
    Ok(receipts)
}

fn clean_worker_kernel_cache(output_root: &Path) -> Result<(), CanaryError> {
    let cache = output_root.join(WORKER_KERNEL_CACHE_DIRECTORY);
    if !cache.exists() {
        return Ok(());
    }
    fs::remove_dir_all(cache).map_err(|_| {
        CanaryError::new(
            "canary_kernel_cache_cleanup_failed",
            "cannot remove the bounded worker kernel cache",
        )
    })
}

fn write_final_outputs(
    output_root: &Path,
    sidecar: &Path,
    interlock: &CanaryInterlock,
    receipt: &PersistentDecodeCanaryReceipt,
) -> Result<(), CanaryError> {
    fs::create_dir_all(output_root).map_err(|_| {
        CanaryError::new(
            "canary_output_create_failed",
            "cannot create canary output root",
        )
    })?;
    let interlock_bytes = compact_json(interlock)?;
    if exact_file_blake3(&interlock_bytes) != receipt.bindings.final_interlock_blake3 {
        return Err(CanaryError::new(
            "canary_interlock_hash_mismatch",
            "final interlock differs from the canary receipt",
        ));
    }
    atomic_replace(sidecar, &interlock_bytes)?;
    atomic_replace(&output_root.join(INTERLOCK_FILENAME), &interlock_bytes)?;
    atomic_replace(
        &output_root.join(CANARY_RECEIPT_FILENAME),
        &pretty_json(receipt)?,
    )
}

fn finalize_failure(
    context: &CanaryContext,
    duration_seconds: f64,
    reason: CanaryReason,
    output_root: &Path,
    sidecar: &Path,
) -> Result<ExitCode, CanaryError> {
    let _ = clean_worker_kernel_cache(output_root);
    let (interlock, receipt) = evaluate_failed_canary(context, duration_seconds, reason)?;
    write_final_outputs(output_root, sidecar, &interlock, &receipt)?;
    Ok(ExitCode::from(CRITICAL_EXIT_STATUS))
}

fn evaluate_success(
    context: &CanaryContext,
    duration_seconds: f64,
    monitoring_policy_json: &str,
    output_root: &Path,
    sidecar: &Path,
    armed_interlock_blake3: &str,
) -> Result<ExitCode, CanaryError> {
    let candidate_receipts =
        load_worker_receipts(context, output_root, ExpectedCanaryDispatch::Persistent)?;
    let control_receipts =
        load_worker_receipts(context, output_root, ExpectedCanaryDispatch::Normal)?;
    validate_worker_interlock_receipts(
        context,
        armed_interlock_blake3,
        ExpectedCanaryDispatch::Persistent,
        &candidate_receipts,
    )?;
    validate_worker_interlock_receipts(
        context,
        armed_interlock_blake3,
        ExpectedCanaryDispatch::Normal,
        &control_receipts,
    )?;
    clean_worker_kernel_cache(output_root)?;
    let monitoring =
        collect_persistent_decode_monitoring(monitoring_policy_json, &candidate_receipts)
            .map_err(|error| CanaryError::new("canary_monitoring_failed", error.to_string()))?;
    write_new(
        &output_root.join(MONITORING_RECEIPT_FILENAME),
        &pretty_json(&monitoring)?,
    )?;
    let ab_json = read_bounded(
        &output_root.join(PHYSICAL_AB_RECEIPT_FILENAME),
        INPUT_READ_LIMIT_BYTES,
        "physical A/B receipt",
    )?;
    let physical_ab = validate_physical_ab_receipt(context, &ab_json)?;
    let (interlock, receipt) =
        evaluate_completed_canary(context, duration_seconds, &monitoring, &physical_ab)?;
    write_final_outputs(output_root, sidecar, &interlock, &receipt)?;
    Ok(match monitoring.verdict {
        MonitoringVerdict::Clean if interlock.state == CanaryInterlockState::Retain => {
            ExitCode::from(CLEAN_EXIT_STATUS)
        }
        MonitoringVerdict::Warning => ExitCode::from(WARNING_EXIT_STATUS),
        MonitoringVerdict::Clean | MonitoringVerdict::Critical => {
            ExitCode::from(CRITICAL_EXIT_STATUS)
        }
    })
}

// r[impl tenstorrent.rwkv7_p150x2.persistent_decode_canary.command]
fn run(arguments: &[String]) -> Result<ExitCode, CanaryError> {
    if arguments.len() != ARGUMENT_COUNT {
        return Err(CanaryError::new("canary_usage", USAGE));
    }
    let policy_path = Path::new(&arguments[POLICY_ARGUMENT_INDEX]);
    let throughput_path = Path::new(&arguments[THROUGHPUT_ARGUMENT_INDEX]);
    let workload_path = Path::new(&arguments[WORKLOAD_ARGUMENT_INDEX]);
    let monitoring_path = Path::new(&arguments[MONITORING_ARGUMENT_INDEX]);
    let output_root = Path::new(&arguments[OUTPUT_ARGUMENT_INDEX]);
    if output_root.exists() {
        return Err(CanaryError::new(
            "canary_output_exists",
            "canary output root must not exist",
        ));
    }
    let sidecar = sidecar_path(output_root);
    if sidecar.exists() || temporary_path(&sidecar).exists() {
        return Err(CanaryError::new(
            "canary_output_exists",
            "canary interlock sidecar must not exist",
        ));
    }
    let policy_json = read_bounded(policy_path, POLICY_READ_LIMIT_BYTES, "canary policy")?;
    let throughput_json = read_bounded(
        throughput_path,
        POLICY_READ_LIMIT_BYTES,
        "canary throughput profile",
    )?;
    let workload_json = read_bounded(workload_path, INPUT_READ_LIMIT_BYTES, "canary workload")?;
    let monitoring_json = read_bounded(
        monitoring_path,
        POLICY_READ_LIMIT_BYTES,
        "monitoring policy",
    )?;
    let context = build_canary_context(
        &policy_json,
        &throughput_json,
        &workload_json,
        &monitoring_json,
    )?;
    let armed = canary_interlock(
        &context,
        CanaryInterlockState::Armed,
        CanaryReason::CanaryRunning,
    )?;
    let armed_bytes = compact_json(&armed)?;
    let armed_blake3 = exact_file_blake3(&armed_bytes);
    write_new(&sidecar, &armed_bytes)?;
    let mut child = match launch_benchmark(
        &context,
        throughput_path,
        workload_path,
        output_root,
        &sidecar,
        &armed_blake3,
    ) {
        Ok(child) => child,
        Err(_) => {
            return finalize_failure(
                &context,
                0.0,
                CanaryReason::BenchmarkFailed,
                output_root,
                &sidecar,
            );
        }
    };
    let outcome = match wait_bounded(&mut child, context.policy.maximum_duration_seconds) {
        Ok(outcome) => outcome,
        Err(_) => {
            stop_child(&mut child);
            return finalize_failure(
                &context,
                0.0,
                CanaryReason::BenchmarkFailed,
                output_root,
                &sidecar,
            );
        }
    };
    match outcome {
        BenchmarkOutcome::Completed { duration_seconds } => {
            match evaluate_success(
                &context,
                duration_seconds,
                &monitoring_json,
                output_root,
                &sidecar,
                &armed_blake3,
            ) {
                Ok(status) => Ok(status),
                Err(_) => finalize_failure(
                    &context,
                    duration_seconds,
                    CanaryReason::InvalidEvidence,
                    output_root,
                    &sidecar,
                ),
            }
        }
        BenchmarkOutcome::Failed { duration_seconds } => finalize_failure(
            &context,
            duration_seconds,
            CanaryReason::BenchmarkFailed,
            output_root,
            &sidecar,
        ),
        BenchmarkOutcome::TimedOut { duration_seconds } => finalize_failure(
            &context,
            duration_seconds,
            CanaryReason::CanaryTimeout,
            output_root,
            &sidecar,
        ),
    }
}

fn emit_error(error: &CanaryError) {
    let envelope = ErrorEnvelope {
        schema_version: 1,
        error: ErrorBody {
            code: error.code,
            detail: &error.detail,
        },
    };
    match serde_json::to_string_pretty(&envelope) {
        Ok(json) => eprintln!("{json}"),
        Err(_) => eprintln!("canary error: {}", error.code),
    }
}

fn main() -> ExitCode {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match run(&arguments) {
        Ok(status) => status,
        Err(error) => {
            emit_error(&error);
            ExitCode::from(INPUT_ERROR_EXIT_STATUS)
        }
    }
}
