use rwkv_ttwkv7_persistent_receipt_validator::{
    MAXIMUM_PERSISTENT_RECEIPT_BYTES, PersistentReceiptExpectations,
    validate_persistent_receipt,
};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

const RECEIPT_FLAG: &str = "--receipt";
const EXPECTED_CALL_COUNT_FLAG: &str = "--expected-call-count";
const EXPECTED_CONTINUITY_COUNT_FLAG: &str = "--expected-continuity-count";
const USAGE: &str = "usage: rwkv-ttwkv7-persistent-receipt-validator --receipt PATH --expected-call-count COUNT --expected-continuity-count COUNT";
const MAXIMUM_U32_DECIMAL_DIGITS: usize = 10;

struct CommandArguments {
    receipt_path: PathBuf,
    expectations: PersistentReceiptExpectations,
}

fn parse_arguments() -> Result<CommandArguments, String> {
    let mut arguments = env::args().skip(1);
    require_flag(arguments.next(), RECEIPT_FLAG)?;
    let receipt_path = arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| USAGE.to_owned())?;
    require_flag(arguments.next(), EXPECTED_CALL_COUNT_FLAG)?;
    let call_count = parse_count(arguments.next(), "expected call count")?;
    require_flag(arguments.next(), EXPECTED_CONTINUITY_COUNT_FLAG)?;
    let continuity_count = parse_count(arguments.next(), "expected continuity count")?;
    if arguments.next().is_some() {
        return Err(USAGE.to_owned());
    }
    Ok(CommandArguments {
        receipt_path,
        expectations: PersistentReceiptExpectations {
            call_count,
            continuity_count,
        },
    })
}

fn require_flag(argument: Option<String>, expected: &str) -> Result<(), String> {
    let actual = argument.ok_or_else(|| USAGE.to_owned())?;
    if actual != expected {
        return Err(USAGE.to_owned());
    }
    Ok(())
}

fn parse_count(argument: Option<String>, name: &str) -> Result<u32, String> {
    let value = argument.ok_or_else(|| USAGE.to_owned())?;
    if value.is_empty() || value.len() > MAXIMUM_U32_DECIMAL_DIGITS {
        return Err(format!("{name} is not an exact nonnegative decimal"));
    }
    if !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!("{name} is not an exact nonnegative decimal"));
    }
    value
        .parse::<u32>()
        .map_err(|error| format!("{name} is not an exact nonnegative decimal: {error}"))
}

fn run() -> Result<(), String> {
    let arguments = parse_arguments()?;
    let metadata = fs::metadata(&arguments.receipt_path).map_err(|error| {
        format!(
            "receipt metadata could not be read at {}: {error}",
            arguments.receipt_path.display()
        )
    })?;
    if !metadata.is_file() {
        return Err("receipt path is not a regular file".to_owned());
    }
    if metadata.len() > MAXIMUM_PERSISTENT_RECEIPT_BYTES {
        return Err(format!(
            "persistent receipt exceeds {MAXIMUM_PERSISTENT_RECEIPT_BYTES} bytes"
        ));
    }
    let receipt_bytes = fs::read(&arguments.receipt_path).map_err(|error| {
        format!(
            "receipt could not be read at {}: {error}",
            arguments.receipt_path.display()
        )
    })?;
    let summary = validate_persistent_receipt(&receipt_bytes, arguments.expectations)?;
    println!(
        "rwkv persistent receipt validation: PASS calls={} continuity={}",
        summary.call_count, summary.continuity_count
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rwkv persistent receipt validation: {error}");
            ExitCode::FAILURE
        }
    }
}
