//! Filesystem and process shell for the single-P150A bring-up core.
//!
//! All admission decisions live in the pure core (`lib.rs`). This binary
//! only reads argument paths, parses JSON, prints results, and sets exit
//! codes.

use serde_json::{Value, json};
use std::process::ExitCode;

const SUCCESS: u8 = 0;
const REJECTED: u8 = 1;
const USAGE_ERROR: u8 = 2;

fn usage() -> String {
    "usage: rwkv7-lora-p150a-core self-test\n\
     usage: rwkv7-lora-p150a-core admit-profile PROFILE_JSON\n\
     usage: rwkv7-lora-p150a-core admit-prerequisites --profile PATH --platform-admission PATH \
     --platform-receipt PATH --cpu-receipt PATH --cpu-receipt-blake3 HEX"
        .to_string()
}

fn read_json(path: &str) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_str(&text).map_err(|error| format!("cannot parse {path}: {error}"))
}

fn print_rejection(rejection: &rwkv7_lora_p150a_core::Rejection) {
    let body = json!({
        "status": "rejected",
        "code": rejection.code,
        "detail": rejection.detail,
    });
    println!("{body}");
    eprintln!("rwkv7-lora-p150a-core: {rejection}");
}

struct PrerequisiteArgs {
    profile: String,
    platform_admission: String,
    platform_receipt: String,
    cpu_receipt: String,
    cpu_receipt_blake3: String,
}

fn parse_prerequisite_args(args: &[String]) -> Result<PrerequisiteArgs, String> {
    let mut parsed = PrerequisiteArgs {
        profile: String::new(),
        platform_admission: String::new(),
        platform_receipt: String::new(),
        cpu_receipt: String::new(),
        cpu_receipt_blake3: String::new(),
    };
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let value = args
            .get(index + 1)
            .ok_or_else(|| format!("flag {flag} needs a value"))?;
        match flag {
            "--profile" => parsed.profile = value.clone(),
            "--platform-admission" => parsed.platform_admission = value.clone(),
            "--platform-receipt" => parsed.platform_receipt = value.clone(),
            "--cpu-receipt" => parsed.cpu_receipt = value.clone(),
            "--cpu-receipt-blake3" => parsed.cpu_receipt_blake3 = value.clone(),
            other => return Err(format!("unknown flag '{other}'")),
        }
        index += 2;
    }
    for (name, value) in [
        ("--profile", &parsed.profile),
        ("--platform-admission", &parsed.platform_admission),
        ("--platform-receipt", &parsed.platform_receipt),
        ("--cpu-receipt", &parsed.cpu_receipt),
        ("--cpu-receipt-blake3", &parsed.cpu_receipt_blake3),
    ] {
        if value.is_empty() {
            return Err(format!("missing required flag {name}"));
        }
    }
    Ok(parsed)
}

fn run() -> Result<u8, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err(usage());
    }
    match args[0].as_str() {
        "self-test" => {
            if args.len() != 1 {
                return Err(usage());
            }
            match rwkv7_lora_p150a_core::self_test() {
                Ok(line) => {
                    println!("{line}");
                    Ok(SUCCESS)
                }
                Err(rejection) => {
                    print_rejection(&rejection);
                    Ok(REJECTED)
                }
            }
        }
        "admit-profile" => {
            if args.len() != 2 {
                return Err(usage());
            }
            let profile = read_json(&args[1])?;
            match rwkv7_lora_p150a_core::admit_profile(&profile) {
                Ok(admitted) => {
                    let body = json!({
                        "status": "admitted",
                        "identity_blake3": admitted.identity_blake3,
                        "profile_id": admitted.profile_id,
                        "sequence_length": admitted.sequence_length,
                        "micro_batch": admitted.micro_batch,
                        "probe_compile_timeout_seconds": admitted.probe_compile_timeout_seconds,
                        "full_compile_timeout_seconds": admitted.full_compile_timeout_seconds,
                        "max_stablehlo_operations": admitted.max_stablehlo_operations,
                        "hardware_attempt_seconds": admitted.hardware_attempt_seconds,
                    });
                    println!("{body}");
                    Ok(SUCCESS)
                }
                Err(rejection) => {
                    print_rejection(&rejection);
                    Ok(REJECTED)
                }
            }
        }
        "admit-prerequisites" => {
            let parsed = parse_prerequisite_args(&args[1..]).map_err(|error| format!("{error}\n{}", usage()))?;
            let profile = read_json(&parsed.profile)?;
            let platform_admission = read_json(&parsed.platform_admission)?;
            let platform_receipt = read_json(&parsed.platform_receipt)?;
            let cpu_receipt = read_json(&parsed.cpu_receipt)?;
            match rwkv7_lora_p150a_core::admit_prerequisites(
                &profile,
                &platform_admission,
                &platform_receipt,
                &cpu_receipt,
                &parsed.cpu_receipt_blake3,
            ) {
                Ok(admission) => {
                    let body = json!({
                        "status": "admitted",
                        "prerequisite_identity": admission.prerequisite_identity,
                        "bound": {
                            "profile_identity": admission.profile_identity,
                            "platform_bundle_identity": admission.platform_bundle_identity,
                            "platform_config_identity": admission.platform_config_identity,
                            "platform_plan_identity": admission.platform_plan_identity,
                            "cpu_authority_blake3": admission.cpu_authority_blake3,
                            "cpu_receipt_blake3": admission.cpu_receipt_blake3,
                        },
                    });
                    println!("{body}");
                    Ok(SUCCESS)
                }
                Err(rejection) => {
                    print_rejection(&rejection);
                    Ok(REJECTED)
                }
            }
        }
        _ => Err(usage()),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => ExitCode::from(code),
        Err(message) => {
            eprintln!("rwkv7-lora-p150a-core: {message}");
            ExitCode::from(USAGE_ERROR)
        }
    }
}
