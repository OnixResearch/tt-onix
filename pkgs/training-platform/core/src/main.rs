//! Thin CLI shell over the pure training-platform admission core.
//! Owns argument parsing, file reads, stdout, and process exit codes.

use serde_json::{Value, json};
use std::process::ExitCode;
use training_platform_core::{
    Rejection, admit_bundle, admit_preflight_snapshot, build_launch_plan, canonical_json,
    validate_config, validate_hardware_receipt,
};

const USAGE: &str = "training-platform-core <command> [args]

commands:
  admit-bundle <manifest.json>
  validate-config <config.json> <manifest.json>
  launch-plan <manifest.json> <config.json> <descriptor_host_path>
  admit-preflight <snapshot.json> <manifest.json> <config.json>
  validate-receipt <receipt.json> <manifest.json> <config.json> <descriptor_host_path>
  self-test";

fn read_json(path: &str) -> Result<Value, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|source| format!("failed to read {path}: {source}"))?;
    serde_json::from_str(&text).map_err(|source| format!("failed to parse {path}: {source}"))
}

fn print_json(value: &Value) {
    println!("{}", canonical_json(value));
}

fn reject(command: &str, rejection: Rejection) -> ExitCode {
    eprintln!("{command}: {}: {}", rejection.code, rejection.detail);
    ExitCode::from(1)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(command) = args.first() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let rest = &args[1..];
    match command.as_str() {
        "admit-bundle" => {
            let [manifest_path] = rest else {
                eprintln!("usage: admit-bundle <manifest.json>");
                return ExitCode::from(2);
            };
            let manifest = match read_json(manifest_path) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            match admit_bundle(&manifest) {
                Ok(bundle) => {
                    print_json(&json!({
                        "bundle_id": bundle.bundle_id,
                        "identity_blake3": bundle.identity_blake3,
                        "image_reference": bundle.image_reference,
                        "pjrt_wheel_version": bundle.pjrt_wheel_version,
                        "tt_blacksmith_revision": bundle.tt_blacksmith_revision,
                        "driver_version": bundle.driver_version,
                        "firmware_floor": bundle.firmware_floor,
                        "status": "admitted"
                    }));
                    ExitCode::SUCCESS
                }
                Err(rejection) => reject("admit-bundle", rejection),
            }
        }
        "validate-config" => {
            let [config_path, manifest_path] = rest else {
                eprintln!("usage: validate-config <config.json> <manifest.json>");
                return ExitCode::from(2);
            };
            let (config, manifest) = match (read_json(config_path), read_json(manifest_path)) {
                (Ok(config), Ok(manifest)) => (config, manifest),
                (Err(error), _) | (_, Err(error)) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            let bundle = match admit_bundle(&manifest) {
                Ok(bundle) => bundle,
                Err(rejection) => return reject("validate-config", rejection),
            };
            match validate_config(&config, &bundle.bundle_id) {
                Ok(validated) => {
                    print_json(&json!({
                        "identity_blake3": validated.identity_blake3,
                        "bundle_id": validated.bundle_id,
                        "status": "admitted"
                    }));
                    ExitCode::SUCCESS
                }
                Err(rejection) => reject("validate-config", rejection),
            }
        }
        "launch-plan" => {
            let [manifest_path, config_path, descriptor_host_path] = rest else {
                eprintln!(
                    "usage: launch-plan <manifest.json> <config.json> <descriptor_host_path>"
                );
                return ExitCode::from(2);
            };
            match plan_from_files(manifest_path, config_path, descriptor_host_path) {
                Ok(plan) => {
                    print_json(&json!({
                        "identity_blake3": plan.launch_plan.identity_blake3,
                        "plan": plan.launch_plan.plan
                    }));
                    ExitCode::SUCCESS
                }
                Err(rejection) => reject("launch-plan", rejection),
            }
        }
        "admit-preflight" => {
            let [snapshot_path, manifest_path, config_path] = rest else {
                eprintln!("usage: admit-preflight <snapshot.json> <manifest.json> <config.json>");
                return ExitCode::from(2);
            };
            let (manifest, config) = match admitted_from_files(manifest_path, config_path) {
                Ok(pair) => pair,
                Err(rejection) => return reject("admit-preflight", rejection),
            };
            let snapshot = match read_json(snapshot_path) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            match admit_preflight_snapshot(&snapshot, &manifest, &config) {
                Ok(facts) => {
                    print_json(&json!({
                        "device_board_ids": facts.device_board_ids,
                        "firmware_bundle": facts.firmware_bundle,
                        "driver_version": facts.driver_version,
                        "status": "admitted"
                    }));
                    ExitCode::SUCCESS
                }
                Err(rejection) => reject("admit-preflight", rejection),
            }
        }
        "validate-receipt" => {
            let [
                receipt_path,
                manifest_path,
                config_path,
                descriptor_host_path,
            ] = rest
            else {
                eprintln!(
                    "usage: validate-receipt <receipt.json> <manifest.json> <config.json> <descriptor_host_path>"
                );
                return ExitCode::from(2);
            };
            let plan = match plan_from_files(manifest_path, config_path, descriptor_host_path) {
                Ok(plan) => plan,
                Err(rejection) => return reject("validate-receipt", rejection),
            };
            let receipt = match read_json(receipt_path) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            match validate_hardware_receipt(
                &receipt,
                &plan.bundle_identity,
                &plan.config_identity,
                &plan.launch_plan.identity_blake3,
            ) {
                Ok(()) => {
                    print_json(&json!({"status": "admitted"}));
                    ExitCode::SUCCESS
                }
                Err(rejection) => reject("validate-receipt", rejection),
            }
        }
        "self-test" => self_test(),
        _ => {
            eprintln!("unknown command {command}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

struct PlanIdentities {
    launch_plan: training_platform_core::LaunchPlan,
    bundle_identity: String,
    config_identity: String,
}

fn admitted_from_files(
    manifest_path: &str,
    config_path: &str,
) -> Result<
    (
        training_platform_core::AdmittedBundle,
        training_platform_core::ValidatedConfig,
    ),
    Rejection,
> {
    let manifest_text = std::fs::read_to_string(manifest_path).map_err(|source| {
        Rejection::new("io", format!("failed to read {manifest_path}: {source}"))
    })?;
    let manifest: Value = serde_json::from_str(&manifest_text).map_err(|source| {
        Rejection::new(
            "parse",
            format!("failed to parse {manifest_path}: {source}"),
        )
    })?;
    let bundle = admit_bundle(&manifest)?;
    let config_text = std::fs::read_to_string(config_path).map_err(|source| {
        Rejection::new("io", format!("failed to read {config_path}: {source}"))
    })?;
    let config: Value = serde_json::from_str(&config_text).map_err(|source| {
        Rejection::new("parse", format!("failed to parse {config_path}: {source}"))
    })?;
    let validated = validate_config(&config, &bundle.bundle_id)?;
    Ok((bundle, validated))
}

fn plan_from_files(
    manifest_path: &str,
    config_path: &str,
    descriptor_host_path: &str,
) -> Result<PlanIdentities, Rejection> {
    let (bundle, config) = admitted_from_files(manifest_path, config_path)?;
    let plan = build_launch_plan(&bundle, &config, descriptor_host_path)?;
    Ok(PlanIdentities {
        bundle_identity: bundle.identity_blake3,
        config_identity: config.identity_blake3,
        launch_plan: plan,
    })
}

fn self_test() -> ExitCode {
    let manifest = json!({
        "schema_version": 1,
        "bundle_id": "self-test",
        "recorded_date": "2026-08-04",
        "runtime_image": {
            "repository": "example.invalid/image",
            "digest": format!("sha256:{}", "1".repeat(64)),
            "created": "2026-08-03",
            "base_os": "ubuntu-24.04"
        },
        "pjrt_plugin": {
            "package": "pjrt-plugin-tt",
            "version": "1.0.0",
            "wheel_sha256": "2".repeat(64),
            "python_abi": "cp312",
            "index": "https://example.invalid"
        },
        "tt_xla_source": {
            "repository": "https://example.invalid/tt-xla",
            "revision": "3".repeat(40),
            "revision_date": "2026-08-04"
        },
        "tt_blacksmith_source": {
            "repository": "https://example.invalid/tt-blacksmith",
            "revision": "4".repeat(40),
            "revision_date": "2026-08-04"
        },
        "pytorch": {"package": "torch", "version": "1.0.0", "wheel_sha256": "5".repeat(64), "python_abi": "cp312"},
        "pytorch_xla": {"package": "torch-xla", "version": "1.0.0", "wheel_sha256": "6".repeat(64), "python_abi": "cp312"},
        "jax": {"package": "jax", "version": "1.0.0"},
        "jaxlib": {"package": "jaxlib", "version": "1.0.0"},
        "transformers": {"package": "transformers", "version": "1.0.0"},
        "driver": {"name": "TT-KMD", "version": "1.0.0"},
        "firmware": {"bundle_version": "1.0.0.0", "floor_version": "1.0.0.0"}
    });
    if admit_bundle(&manifest).is_err() {
        eprintln!("self-test: positive bundle admission failed");
        return ExitCode::FAILURE;
    }
    let mut mutable = manifest.clone();
    mutable["runtime_image"]["digest"] = json!("latest");
    match admit_bundle(&mutable) {
        Err(rejection) if rejection.code == "mutable_identity" => {}
        _ => {
            eprintln!("self-test: mutable image tag was not rejected");
            return ExitCode::FAILURE;
        }
    }
    println!("training-platform-core self-test: PASS");
    ExitCode::SUCCESS
}
