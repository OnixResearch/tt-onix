//! Filesystem and process shell for the dual-P150A admission core.
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
    "usage: rwkv7-lora-dual-p150a-core self-test\n\
     usage: rwkv7-lora-dual-p150a-core admit-profile PROFILE_JSON\n\
     usage: rwkv7-lora-dual-p150a-core admit-measurement-profile PROFILE_JSON\n\
     usage: rwkv7-lora-dual-p150a-core bind-machine-facts VALUES_JSON\n\
     usage: rwkv7-lora-dual-p150a-core summarize-measurement RAW_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-dual-p150a-core admit-measurement RECEIPT_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-dual-p150a-core plan-shards PROFILE_JSON BATCH_JSON\n\
     usage: rwkv7-lora-dual-p150a-core admit-collective RECEIPT_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-dual-p150a-core admit-step RECEIPT_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-dual-p150a-core admit-run RECEIPT_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-dual-p150a-core admit-checkpoint RECEIPT_JSON PROFILE_JSON\n\
     usage: rwkv7-lora-dual-p150a-core admit-prerequisites --profile PATH --platform-admission PATH \
     --platform-receipt PATH --cpu-receipt PATH --cpu-receipt-blake3 HEX \
     --single-profile PATH --single-step-receipt PATH --single-run-receipt PATH \
     --single-checkpoint-receipt PATH"
        .to_string()
}

fn read_json(path: &str) -> Result<Value, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("cannot read {path}: {error}"))?;
    serde_json::from_str(&text).map_err(|error| format!("cannot parse {path}: {error}"))
}

fn print_rejection(rejection: &rwkv7_lora_dual_p150a_core::Rejection) {
    let body = json!({
        "status": "rejected",
        "code": rejection.code,
        "detail": rejection.detail,
    });
    println!("{body}");
    eprintln!("rwkv7-lora-dual-p150a-core: {rejection}");
}

struct PrerequisiteArgs {
    profile: String,
    platform_admission: String,
    platform_receipt: String,
    cpu_receipt: String,
    cpu_receipt_blake3: String,
    single_profile: String,
    single_step_receipt: String,
    single_run_receipt: String,
    single_checkpoint_receipt: String,
}

fn parse_prerequisite_args(args: &[String]) -> Result<PrerequisiteArgs, String> {
    let mut parsed = PrerequisiteArgs {
        profile: String::new(),
        platform_admission: String::new(),
        platform_receipt: String::new(),
        cpu_receipt: String::new(),
        cpu_receipt_blake3: String::new(),
        single_profile: String::new(),
        single_step_receipt: String::new(),
        single_run_receipt: String::new(),
        single_checkpoint_receipt: String::new(),
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
            "--single-profile" => parsed.single_profile = value.clone(),
            "--single-step-receipt" => parsed.single_step_receipt = value.clone(),
            "--single-run-receipt" => parsed.single_run_receipt = value.clone(),
            "--single-checkpoint-receipt" => parsed.single_checkpoint_receipt = value.clone(),
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
        ("--single-profile", &parsed.single_profile),
        ("--single-step-receipt", &parsed.single_step_receipt),
        ("--single-run-receipt", &parsed.single_run_receipt),
        (
            "--single-checkpoint-receipt",
            &parsed.single_checkpoint_receipt,
        ),
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
            match rwkv7_lora_dual_p150a_core::self_test() {
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
            match rwkv7_lora_dual_p150a_core::admit_measurement_profile(&profile) {
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
            match rwkv7_lora_dual_p150a_core::bind_machine_facts(&values) {
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
            match rwkv7_lora_dual_p150a_core::summarize_measurement(&raw, &profile) {
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
            match rwkv7_lora_dual_p150a_core::admit_measurement_receipt(&receipt, &profile) {
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
        "admit-profile" => {
            if args.len() != 2 {
                return Err(usage());
            }
            let profile = read_json(&args[1])?;
            match rwkv7_lora_dual_p150a_core::admit_profile(&profile) {
                Ok(admitted) => {
                    let body = json!({
                        "status": "admitted",
                        "identity_blake3": admitted.identity_blake3,
                        "profile_id": admitted.profile_id,
                        "replicas": admitted.replicas,
                        "sequence_length": admitted.sequence_length,
                        "micro_batch": admitted.micro_batch,
                        "global_batch": admitted.global_batch,
                        "probe_compile_timeout_seconds": admitted.probe_compile_timeout_seconds,
                        "full_compile_timeout_seconds": admitted.full_compile_timeout_seconds,
                        "max_stablehlo_operations": admitted.max_stablehlo_operations,
                        "hardware_attempt_seconds": admitted.hardware_attempt_seconds,
                        "collective_timeout_seconds": admitted.collective_timeout_seconds,
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
        "plan-shards" => {
            if args.len() != 3 {
                return Err(usage());
            }
            let profile = read_json(&args[1])?;
            let batch = read_json(&args[2])?;
            match rwkv7_lora_dual_p150a_core::plan_shards(&profile, &batch) {
                Ok(plan) => {
                    let body = json!({
                        "status": "admitted",
                        "identity_blake3": plan.identity_blake3,
                        "global_batch_index": plan.global_batch_index,
                        "replicas": plan
                            .replicas
                            .iter()
                            .map(|shard| json!({
                                "ordinal": shard.ordinal,
                                "sample_ids": shard.sample_ids,
                                "active_tokens": shard.active_tokens,
                            }))
                            .collect::<Vec<_>>(),
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
        "admit-collective" => {
            if args.len() != 3 {
                return Err(usage());
            }
            let receipt = read_json(&args[1])?;
            let profile = read_json(&args[2])?;
            match rwkv7_lora_dual_p150a_core::admit_collective(&receipt, &profile) {
                Ok(admission) => {
                    let body = json!({
                        "status": "admitted",
                        "identity_blake3": admission.identity_blake3,
                        "all_reduce_count": admission.all_reduce_count,
                        "collective_seconds": admission.collective_seconds,
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
            match rwkv7_lora_dual_p150a_core::admit_step_receipt(&receipt, &profile) {
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
            match rwkv7_lora_dual_p150a_core::admit_run_receipt(&receipt, &profile) {
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
            match rwkv7_lora_dual_p150a_core::admit_checkpoint_receipt(&receipt, &profile) {
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
            let single_profile = read_json(&parsed.single_profile)?;
            let single_step_receipt = read_json(&parsed.single_step_receipt)?;
            let single_run_receipt = read_json(&parsed.single_run_receipt)?;
            let single_checkpoint_receipt = read_json(&parsed.single_checkpoint_receipt)?;
            match rwkv7_lora_dual_p150a_core::admit_prerequisites(
                &rwkv7_lora_dual_p150a_core::PrerequisiteInputs {
                    profile: &profile,
                    platform_admission: &platform_admission,
                    platform_receipt: &platform_receipt,
                    cpu_receipt: &cpu_receipt,
                    cpu_receipt_blake3: &parsed.cpu_receipt_blake3,
                    single_profile: &single_profile,
                    single_step_receipt: &single_step_receipt,
                    single_run_receipt: &single_run_receipt,
                    single_checkpoint_receipt: &single_checkpoint_receipt,
                },
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
                            "single_profile_identity": admission.single_profile_identity,
                            "single_step_identity": admission.single_step_identity,
                            "single_run_identity": admission.single_run_identity,
                            "single_checkpoint_identity": admission.single_checkpoint_identity,
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
            eprintln!("rwkv7-lora-dual-p150a-core: {message}");
            ExitCode::from(USAGE_ERROR)
        }
    }
}
