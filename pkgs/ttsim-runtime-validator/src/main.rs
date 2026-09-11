use std::{env, fs, path::Path, process::ExitCode};
use ttsim_runtime_validator::{
    BoundIdentity, DEFAULT_TIMEOUT_SECONDS, ProcessObservation, RECEIPT_SCHEMA_VERSION,
    ReceiptIdentities, SUPPORTED_SIMULATOR_VERSION, SimulatorReceipt, SmokeStatus, classify_smoke,
    required_non_claims, validate_receipt,
};

const BLOCKED_EXIT_CODE: u8 = 2;
const VALIDATE_ARGUMENT_COUNT: usize = 3;
const EMIT_ARGUMENT_COUNT: usize = 12;
const COMMAND_INDEX: usize = 1;
const RECEIPT_PATH_INDEX: usize = 2;
const STDOUT_PATH_INDEX: usize = 2;
const STDERR_PATH_INDEX: usize = 3;
const EXIT_CODE_INDEX: usize = 4;
const TIMED_OUT_INDEX: usize = 5;
const LIBRARY_PATH_INDEX: usize = 6;
const DESCRIPTOR_PATH_INDEX: usize = 7;
const HOST_SYSTEM_INDEX: usize = 8;
const TT_METAL_IDENTITY_INDEX: usize = 9;
const TIMEOUT_SECONDS_INDEX: usize = 10;
const OUTPUT_PATH_INDEX: usize = 11;
const TOPOLOGY: &str = "blackhole-x1";
const DISPATCH_MODE: &str = "slow";
const WORKLOAD: &str = "ttnn-ones-add-1x32x32";

fn main() -> ExitCode {
    match run(env::args().collect()) {
        Ok(status) => status,
        Err(error) => {
            eprintln!("{error}");
            ExitCode::FAILURE
        }
    }
}

fn run(arguments: Vec<String>) -> Result<ExitCode, String> {
    let command = arguments
        .get(COMMAND_INDEX)
        .ok_or_else(|| usage("a command is required"))?;
    match command.as_str() {
        "validate" => validate_command(&arguments),
        "emit" => emit_command(&arguments),
        other => Err(usage(&format!("unsupported command `{other}`"))),
    }
}

fn validate_command(arguments: &[String]) -> Result<ExitCode, String> {
    if arguments.len() != VALIDATE_ARGUMENT_COUNT {
        return Err(usage("validate needs one receipt path"));
    }
    let receipt_path = argument(arguments, RECEIPT_PATH_INDEX, "receipt path")?;
    let receipt_bytes = fs::read(receipt_path)
        .map_err(|error| format!("failed to read simulator receipt `{receipt_path}`: {error}"))?;
    validate_receipt(&receipt_bytes, None)?;
    println!("simulator receipt admitted: {receipt_path}");
    Ok(ExitCode::SUCCESS)
}

fn emit_command(arguments: &[String]) -> Result<ExitCode, String> {
    if arguments.len() != EMIT_ARGUMENT_COUNT {
        return Err(usage("emit received the wrong argument count"));
    }

    let stdout_path = argument(arguments, STDOUT_PATH_INDEX, "stdout path")?;
    let stderr_path = argument(arguments, STDERR_PATH_INDEX, "stderr path")?;
    let exit_code = parse_i32(
        argument(arguments, EXIT_CODE_INDEX, "exit code")?,
        "exit code",
    )?;
    let timed_out = parse_bool(argument(arguments, TIMED_OUT_INDEX, "timed-out flag")?)?;
    let library_path = argument(arguments, LIBRARY_PATH_INDEX, "library path")?;
    let descriptor_path = argument(arguments, DESCRIPTOR_PATH_INDEX, "descriptor path")?;
    let host_system = argument(arguments, HOST_SYSTEM_INDEX, "host system")?;
    let tt_metal_identity = argument(arguments, TT_METAL_IDENTITY_INDEX, "TT-Metal identity")?;
    let timeout_seconds = parse_u64(
        argument(arguments, TIMEOUT_SECONDS_INDEX, "timeout seconds")?,
        "timeout seconds",
    )?;
    let output_path = argument(arguments, OUTPUT_PATH_INDEX, "output path")?;

    if timeout_seconds == 0 || timeout_seconds > DEFAULT_TIMEOUT_SECONDS {
        return Err(format!(
            "timeout seconds must be in the range 1..={DEFAULT_TIMEOUT_SECONDS}"
        ));
    }

    let stdout = read_log(stdout_path, "stdout")?;
    let stderr = read_log(stderr_path, "stderr")?;
    let outcome = classify_smoke(&ProcessObservation {
        exit_code,
        timed_out,
        stdout: &stdout,
        stderr: &stderr,
    });
    let receipt = SimulatorReceipt {
        schema_version: RECEIPT_SCHEMA_VERSION,
        simulated: true,
        simulator_version: SUPPORTED_SIMULATOR_VERSION.to_owned(),
        host_system: host_system.to_owned(),
        topology: TOPOLOGY.to_owned(),
        dispatch_mode: DISPATCH_MODE.to_owned(),
        workload: WORKLOAD.to_owned(),
        timeout_seconds,
        status: outcome.status.clone(),
        classification: outcome.classification,
        blocker: outcome.blocker,
        observations: outcome.observations,
        identities: ReceiptIdentities {
            library: file_identity(library_path)?,
            descriptor: file_identity(descriptor_path)?,
            tt_metal: text_identity(tt_metal_identity),
        },
        claims: Vec::new(),
        non_claims: required_non_claims(),
    };
    let encoded = serde_json::to_vec_pretty(&receipt)
        .map_err(|error| format!("failed to encode simulator receipt: {error}"))?;
    validate_receipt(&encoded, None)?;
    fs::write(output_path, encoded)
        .map_err(|error| format!("failed to write simulator receipt `{output_path}`: {error}"))?;
    println!("simulator receipt written: {output_path}");

    if receipt.status == SmokeStatus::Passed {
        Ok(ExitCode::SUCCESS)
    } else {
        Ok(ExitCode::from(BLOCKED_EXIT_CODE))
    }
}

fn file_identity(path: &str) -> Result<BoundIdentity, String> {
    let bytes = fs::read(path)
        .map_err(|error| format!("failed to read identity file `{path}`: {error}"))?;
    Ok(BoundIdentity {
        value: path.to_owned(),
        blake3: blake3::hash(&bytes).to_hex().to_string(),
    })
}

fn text_identity(value: &str) -> BoundIdentity {
    BoundIdentity {
        value: value.to_owned(),
        blake3: blake3::hash(value.as_bytes()).to_hex().to_string(),
    }
}

fn read_log(path: &str, stream: &str) -> Result<String, String> {
    fs::read_to_string(Path::new(path))
        .map_err(|error| format!("failed to read simulator {stream} log `{path}`: {error}"))
}

fn argument<'a>(arguments: &'a [String], index: usize, name: &str) -> Result<&'a str, String> {
    arguments
        .get(index)
        .map(String::as_str)
        .ok_or_else(|| usage(&format!("{name} is required")))
}

fn parse_i32(value: &str, name: &str) -> Result<i32, String> {
    value
        .parse::<i32>()
        .map_err(|error| format!("{name} is invalid: {error}"))
}

fn parse_u64(value: &str, name: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|error| format!("{name} is invalid: {error}"))
}

fn parse_bool(value: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err("timed-out flag must be `true` or `false`".to_owned()),
    }
}

fn usage(message: &str) -> String {
    format!(
        "{message}\nusage:\n  ttsim-runtime-validator validate <receipt>\n  \
         ttsim-runtime-validator emit <stdout> <stderr> <exit-code> <timed-out> \
         <library> <descriptor> <host-system> <tt-metal-identity> <timeout-seconds> <output>"
    )
}
