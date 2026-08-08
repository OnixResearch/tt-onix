//! Filesystem shell for the pure throughput admission core.

use std::fs;
use std::path::Path;
use std::process::ExitCode;

use rwkv7_p150x2_evidence::throughput::{
    RawThroughputEvidence, ThroughputProfile, ThroughputReceipt, admit_throughput_receipt,
    profile_identity, summarize_throughput,
};
use serde::de::DeserializeOwned;
use serde_json::json;

const SUCCESS: u8 = 0;
const REJECTED: u8 = 1;
const USAGE_ERROR: u8 = 2;
const SUMMARIZE_ARGUMENT_COUNT: usize = 3;
const ADMIT_ARGUMENT_COUNT: usize = 4;
const PROFILE_IDENTITY_ARGUMENT_COUNT: usize = 2;

fn usage() -> &'static str {
    "usage: rwkv7-p150x2-throughput profile-identity PROFILE_JSON\n\
     usage: rwkv7-p150x2-throughput summarize PROFILE_JSON RAW_JSON\n\
     usage: rwkv7-p150x2-throughput admit PROFILE_JSON RAW_JSON RECEIPT_JSON"
}

fn read_json<T: DeserializeOwned>(path: &Path) -> Result<T, String> {
    let bytes =
        fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| format!("cannot parse {}: {error}", path.display()))
}

fn print_rejection(rejection: &rwkv7_p150x2_evidence::throughput::Rejection) {
    println!(
        "{}",
        json!({
            "status": "rejected",
            "code": rejection.code,
            "detail": rejection.detail,
        })
    );
    eprintln!("rwkv7-p150x2-throughput: {rejection}");
}

fn run(arguments: &[String]) -> Result<u8, String> {
    let Some(command) = arguments.first().map(String::as_str) else {
        return Err(usage().to_owned());
    };
    match command {
        "profile-identity" if arguments.len() == PROFILE_IDENTITY_ARGUMENT_COUNT => {
            let profile: ThroughputProfile = read_json(Path::new(&arguments[1]))?;
            match profile_identity(&profile) {
                Ok(identity) => {
                    println!("{identity}");
                    Ok(SUCCESS)
                }
                Err(rejection) => {
                    print_rejection(&rejection);
                    Ok(REJECTED)
                }
            }
        }
        "summarize" if arguments.len() == SUMMARIZE_ARGUMENT_COUNT => {
            let profile: ThroughputProfile = read_json(Path::new(&arguments[1]))?;
            let raw: RawThroughputEvidence = read_json(Path::new(&arguments[2]))?;
            match summarize_throughput(&raw, &profile) {
                Ok(receipt) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&receipt)
                            .map_err(|error| format!("cannot serialize receipt: {error}"))?
                    );
                    Ok(SUCCESS)
                }
                Err(rejection) => {
                    print_rejection(&rejection);
                    Ok(REJECTED)
                }
            }
        }
        "admit" if arguments.len() == ADMIT_ARGUMENT_COUNT => {
            let profile: ThroughputProfile = read_json(Path::new(&arguments[1]))?;
            let raw: RawThroughputEvidence = read_json(Path::new(&arguments[2]))?;
            let receipt: ThroughputReceipt = read_json(Path::new(&arguments[3]))?;
            match admit_throughput_receipt(&receipt, &raw, &profile) {
                Ok(admission) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&json!({
                            "status": "admitted",
                            "identity_blake3": admission.identity_blake3,
                            "dual_card_speedup": admission.dual_card_speedup,
                            "scaling_efficiency": admission.scaling_efficiency,
                            "p99_latency_regression_fraction": admission.p99_latency_regression_fraction,
                            "aggregate_tokens_per_second": admission.aggregate_tokens_per_second,
                        }))
                        .map_err(|error| format!("cannot serialize admission: {error}"))?
                    );
                    Ok(SUCCESS)
                }
                Err(rejection) => {
                    print_rejection(&rejection);
                    Ok(REJECTED)
                }
            }
        }
        _ => Err(usage().to_owned()),
    }
}

fn main() -> ExitCode {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    match run(&arguments) {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("{error}");
            ExitCode::from(USAGE_ERROR)
        }
    }
}
