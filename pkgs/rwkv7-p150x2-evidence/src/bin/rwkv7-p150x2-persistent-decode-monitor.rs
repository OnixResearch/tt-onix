use rwkv7_p150x2_evidence::persistent_decode_monitoring::{
    MonitoringError, MonitoringVerdict, collect_persistent_decode_monitoring,
    parse_monitoring_policy,
};
use serde::Serialize;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::process::ExitCode;

const POLICY_READ_LIMIT_BYTES: usize = 65_536;
const CLEAN_EXIT_STATUS: u8 = 0;
const WARNING_EXIT_STATUS: u8 = 2;
const CRITICAL_EXIT_STATUS: u8 = 3;
const INPUT_ERROR_EXIT_STATUS: u8 = 64;
const USAGE: &str =
    "usage: rwkv7-p150x2-persistent-decode-monitor <policy.json> <worker-receipt.json>...";

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

fn read_bounded(path: &Path, limit: usize, label: &'static str) -> Result<String, MonitoringError> {
    let metadata = std::fs::symlink_metadata(path).map_err(|_| MonitoringError {
        code: "monitoring_input_metadata_failed",
        detail: format!("cannot inspect {label}"),
    })?;
    if !metadata.file_type().is_file() {
        return Err(MonitoringError {
            code: "monitoring_input_not_regular_file",
            detail: format!("{label} is not a regular file"),
        });
    }
    let declared_limit = u64::try_from(limit).map_err(|error| MonitoringError {
        code: "monitoring_input_limit_overflow",
        detail: error.to_string(),
    })?;
    if metadata.len() > declared_limit {
        return Err(MonitoringError {
            code: "monitoring_input_too_large",
            detail: format!("{label} exceeds its byte limit"),
        });
    }
    let mut file = File::open(path).map_err(|_| MonitoringError {
        code: "monitoring_input_open_failed",
        detail: format!("cannot open {label}"),
    })?;
    let read_limit = limit.checked_add(1).ok_or_else(|| MonitoringError {
        code: "monitoring_input_limit_overflow",
        detail: format!("{label} byte limit overflows"),
    })?;
    let mut bytes = Vec::with_capacity(read_limit);
    file.by_ref()
        .take(u64::try_from(read_limit).map_err(|error| MonitoringError {
            code: "monitoring_input_limit_overflow",
            detail: error.to_string(),
        })?)
        .read_to_end(&mut bytes)
        .map_err(|_| MonitoringError {
            code: "monitoring_input_read_failed",
            detail: format!("cannot read {label}"),
        })?;
    if bytes.len() > limit {
        return Err(MonitoringError {
            code: "monitoring_input_too_large",
            detail: format!("{label} exceeds its byte limit"),
        });
    }
    String::from_utf8(bytes).map_err(|_| MonitoringError {
        code: "monitoring_input_not_utf8",
        detail: format!("{label} is not UTF-8"),
    })
}

fn emit_error(error: &MonitoringError) {
    let envelope = ErrorEnvelope {
        schema_version: 1,
        error: ErrorBody {
            code: error.code,
            detail: &error.detail,
        },
    };
    match serde_json::to_string_pretty(&envelope) {
        Ok(json) => eprintln!("{json}"),
        Err(_) => eprintln!("monitoring error: {}", error.code),
    }
}

// r[impl tenstorrent.rwkv7_p150x2.persistent_decode_monitoring.command]
fn run(arguments: &[String]) -> Result<ExitCode, MonitoringError> {
    let policy_path = arguments.first().ok_or_else(|| MonitoringError {
        code: "monitoring_usage",
        detail: USAGE.to_owned(),
    })?;
    if arguments.len() < 2 {
        return Err(MonitoringError {
            code: "monitoring_usage",
            detail: USAGE.to_owned(),
        });
    }
    let policy_json = read_bounded(
        Path::new(policy_path),
        POLICY_READ_LIMIT_BYTES,
        "monitoring policy",
    )?;
    let policy = parse_monitoring_policy(&policy_json)?;
    let event_count = arguments.len() - 1;
    if event_count > policy.maximum_batch_events {
        return Err(MonitoringError {
            code: "monitoring_batch_too_large",
            detail: "worker receipt count exceeds the monitoring policy".to_owned(),
        });
    }
    let mut event_json = Vec::with_capacity(event_count);
    for path in &arguments[1..] {
        event_json.push(read_bounded(
            Path::new(path),
            policy.maximum_event_bytes,
            "worker receipt",
        )?);
    }
    let receipt = collect_persistent_decode_monitoring(&policy_json, &event_json)?;
    let output = serde_json::to_string_pretty(&receipt).map_err(|error| MonitoringError {
        code: "monitoring_output_serialization_failed",
        detail: error.to_string(),
    })?;
    println!("{output}");
    Ok(match receipt.verdict {
        MonitoringVerdict::Clean => ExitCode::from(CLEAN_EXIT_STATUS),
        MonitoringVerdict::Warning => ExitCode::from(WARNING_EXIT_STATUS),
        MonitoringVerdict::Critical => ExitCode::from(CRITICAL_EXIT_STATUS),
    })
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
