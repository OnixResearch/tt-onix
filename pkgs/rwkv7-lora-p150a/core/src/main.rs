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
     usage: rwkv7-lora-p150a-core admit-measurement-profile PROFILE_JSON\n\
     usage: rwkv7-lora-p150a-core bind-machine-facts VALUES_JSON\n\
     usage: rwkv7-lora-p150a-core summarize-measurement RAW_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-p150a-core admit-measurement RECEIPT_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-p150a-core summarize-utilization RAW_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-p150a-core admit-utilization RECEIPT_JSON RAW_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-p150a-core create-baseline RECEIPT_JSON PROFILE_JSON POLICY_JSON\n\
     usage: rwkv7-lora-p150a-core admit-baseline STORE_JSON PROFILE_JSON RECEIPT_JSON\n\
     usage: rwkv7-lora-p150a-core gate-regression RECEIPT_JSON PROFILE_JSON STORE_JSON POLICY_JSON\n\
     usage: rwkv7-lora-p150a-core compare-determinism LEFT_JSON RIGHT_JSON PROFILE_JSON POLICY_JSON\n\
     usage: rwkv7-lora-p150a-core admit-probe RECEIPT_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-p150a-core admit-step RECEIPT_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-p150a-core admit-run RECEIPT_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-p150a-core admit-checkpoint RECEIPT_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-p150a-core admit-prerequisites --profile PATH --platform-admission PATH \
     --platform-receipt PATH --cpu-receipt PATH --cpu-receipt-blake3 HEX"
        .to_string()
}

fn read_json(path: &str) -> Result<Value, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("cannot read {path}: {error}"))?;
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
        "admit-measurement-profile" => {
            if args.len() != 2 {
                return Err(usage());
            }
            let profile = read_json(&args[1])?;
            match rwkv7_lora_p150a_core::admit_measurement_profile(&profile) {
                Ok(admission) => {
                    let body = json!({
                        "status": "admitted",
                        "profile_id": admission.profile_id,
                        "repeat_count": admission.repeat_count,
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
        "bind-machine-facts" => {
            if args.len() != 2 {
                return Err(usage());
            }
            let values = read_json(&args[1])?;
            match rwkv7_lora_p150a_core::bind_machine_facts(&values) {
                Ok(facts) => {
                    println!("{facts}");
                    Ok(SUCCESS)
                }
                Err(rejection) => {
                    print_rejection(&rejection);
                    Ok(REJECTED)
                }
            }
        }
        "summarize-measurement" => {
            if args.len() != 3 {
                return Err(usage());
            }
            let raw = read_json(&args[1])?;
            let profile = read_json(&args[2])?;
            match rwkv7_lora_p150a_core::summarize_measurement(&raw, &profile) {
                Ok(receipt) => {
                    println!("{receipt}");
                    Ok(SUCCESS)
                }
                Err(rejection) => {
                    print_rejection(&rejection);
                    Ok(REJECTED)
                }
            }
        }
        "admit-measurement" => {
            if args.len() != 3 {
                return Err(usage());
            }
            let receipt = read_json(&args[1])?;
            let profile = read_json(&args[2])?;
            match rwkv7_lora_p150a_core::admit_measurement_receipt(&receipt, &profile) {
                Ok(admission) => {
                    let body = json!({
                        "status": "admitted",
                        "identity_blake3": admission.identity_blake3,
                        "measurement_profile_id": admission.measurement_profile_id,
                        "repeats": admission.repeats,
                        "speedup_ratio": admission.speedup_ratio,
                        "time_seconds": {
                            "mean": admission.time_seconds.mean,
                            "p50": admission.time_seconds.p50,
                            "p99": admission.time_seconds.p99,
                            "standard_deviation": admission.time_seconds.standard_deviation,
                        },
                        "loss": {
                            "mean": admission.loss.mean,
                            "p50": admission.loss.p50,
                            "p99": admission.loss.p99,
                            "standard_deviation": admission.loss.standard_deviation,
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
        "summarize-utilization" => {
            if args.len() != 3 {
                return Err(usage());
            }
            let raw = read_json(&args[1])?;
            let profile = read_json(&args[2])?;
            match rwkv7_lora_p150a_core::summarize_utilization(&raw, &profile) {
                Ok(receipt) => {
                    println!("{receipt}");
                    Ok(SUCCESS)
                }
                Err(rejection) => {
                    print_rejection(&rejection);
                    Ok(REJECTED)
                }
            }
        }
        "admit-utilization" => {
            if args.len() != 4 {
                return Err(usage());
            }
            let receipt = read_json(&args[1])?;
            let raw = read_json(&args[2])?;
            let profile = read_json(&args[3])?;
            match rwkv7_lora_p150a_core::admit_utilization_receipt(&receipt, &raw, &profile) {
                Ok(admission) => {
                    let body = json!({
                        "status": "admitted",
                        "identity_blake3": admission.identity_blake3,
                        "device_count": admission.device_count,
                        "operation_count": admission.operation_count,
                        "math_busy_fraction": admission.math_busy_fraction,
                        "noc_idle_fraction": admission.noc_idle_fraction,
                        "scaling_efficiency": admission.scaling_efficiency,
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
        "create-baseline" => {
            if args.len() != 4 {
                return Err(usage());
            }
            let receipt = read_json(&args[1])?;
            let profile = read_json(&args[2])?;
            let policy = read_json(&args[3])?;
            match rwkv7_lora_p150a_core::create_baseline(&receipt, &profile, &policy) {
                Ok(baseline) => {
                    println!("{baseline}");
                    Ok(SUCCESS)
                }
                Err(rejection) => {
                    print_rejection(&rejection);
                    Ok(REJECTED)
                }
            }
        }
        "admit-baseline" => {
            if args.len() != 4 {
                return Err(usage());
            }
            let store = read_json(&args[1])?;
            let profile = read_json(&args[2])?;
            let receipt = read_json(&args[3])?;
            match rwkv7_lora_p150a_core::admit_baseline_store(&store, &profile, &receipt) {
                Ok(baseline) => {
                    println!("{baseline}");
                    Ok(SUCCESS)
                }
                Err(rejection) => {
                    print_rejection(&rejection);
                    Ok(REJECTED)
                }
            }
        }
        "gate-regression" => {
            if args.len() != 5 {
                return Err(usage());
            }
            let receipt = read_json(&args[1])?;
            let profile = read_json(&args[2])?;
            let store = read_json(&args[3])?;
            let policy = read_json(&args[4])?;
            match rwkv7_lora_p150a_core::gate_regression(&receipt, &profile, &store, &policy) {
                Ok(gate_receipt) => {
                    println!("{gate_receipt}");
                    Ok(SUCCESS)
                }
                Err(rejection) => {
                    print_rejection(&rejection);
                    Ok(REJECTED)
                }
            }
        }
        "compare-determinism" => {
            if args.len() != 5 {
                return Err(usage());
            }
            let left = read_json(&args[1])?;
            let right = read_json(&args[2])?;
            let profile = read_json(&args[3])?;
            let policy = read_json(&args[4])?;
            match rwkv7_lora_p150a_core::compare_determinism(&left, &right, &profile, &policy) {
                Ok(determinism_receipt) => {
                    println!("{determinism_receipt}");
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
        "admit-probe" => {
            if args.len() != 3 {
                return Err(usage());
            }
            let receipt = read_json(&args[1])?;
            let profile = read_json(&args[2])?;
            match rwkv7_lora_p150a_core::admit_probe(&receipt, &profile) {
                Ok(admission) => {
                    let body = json!({
                        "status": "admitted",
                        "identity_blake3": admission.identity_blake3,
                        "probe": admission.probe,
                        "shape_kind": admission.shape_kind,
                        "stablehlo_op_count": admission.stablehlo_op_count,
                        "compile_seconds": admission.compile_seconds,
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
        "admit-step" => {
            if args.len() != 3 {
                return Err(usage());
            }
            let receipt = read_json(&args[1])?;
            let profile = read_json(&args[2])?;
            match rwkv7_lora_p150a_core::admit_step_receipt(&receipt, &profile) {
                Ok(admission) => {
                    let body = json!({
                        "status": "admitted",
                        "identity_blake3": admission.identity_blake3,
                        "stablehlo_op_count": admission.stablehlo_op_count,
                        "loss_dev": admission.loss_dev,
                        "grad_pass_fraction": admission.grad_pass_fraction,
                        "delta_pass_fraction": admission.delta_pass_fraction,
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
        "admit-run" => {
            if args.len() != 3 {
                return Err(usage());
            }
            let receipt = read_json(&args[1])?;
            let profile = read_json(&args[2])?;
            match rwkv7_lora_p150a_core::admit_run_receipt(&receipt, &profile) {
                Ok(admission) => {
                    let body = json!({
                        "status": "admitted",
                        "identity_blake3": admission.identity_blake3,
                        "steps": admission.steps,
                        "initial_loss": admission.initial_loss,
                        "final_loss": admission.final_loss,
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
        "admit-checkpoint" => {
            if args.len() != 3 {
                return Err(usage());
            }
            let receipt = read_json(&args[1])?;
            let profile = read_json(&args[2])?;
            match rwkv7_lora_p150a_core::admit_checkpoint_receipt(&receipt, &profile) {
                Ok(admission) => {
                    let body = json!({
                        "status": "admitted",
                        "identity_blake3": admission.identity_blake3,
                        "adapter_load_max_diff": admission.adapter_load_max_diff,
                        "resume_loss_dev": admission.resume_loss_dev,
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
            let parsed = parse_prerequisite_args(&args[1..])
                .map_err(|error| format!("{error}\n{}", usage()))?;
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
