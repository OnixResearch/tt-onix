//! Thin CLI shell over the pure training-platform admission core.
//! Owns argument parsing, file reads, stdout, and process exit codes.

use serde_json::{Value, json};
use std::process::ExitCode;
use training_platform_core::{
    Rejection, admit_bundle, admit_preflight_snapshot, admit_runtime_authority,
    build_authorized_launch_plan, build_launch_plan, canonical_json, classify_smoke_failure,
    terminal_stage_states, validate_config, validate_hardware_receipt, validate_portable_artifact,
    validate_terminal_receipt,
};

const USAGE: &str = "training-platform-core <command> [args]

commands:
  admit-bundle <manifest.json>
  validate-config <config.json> <manifest.json>
  launch-plan <manifest.json> <config.json> <descriptor_host_path>
  authorized-launch-plan <authority.json>
  admit-authorized-preflight <snapshot.json> <authority.json>
  admit-preflight <snapshot.json> <manifest.json> <config.json>
  validate-receipt <receipt.json> <manifest.json> <config.json> <descriptor_host_path>
  validate-terminal-receipt <receipt.json> <authority.json>
  validate-portable-artifact <artifact.json>
  classify-smoke-failure <stdout.log>
  stage-states <success|failed-stage>
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
        "authorized-launch-plan" => {
            let [authority_path] = rest else {
                eprintln!("usage: authorized-launch-plan <authority.json>");
                return ExitCode::from(2);
            };
            match authorized_plan_from_file(authority_path) {
                Ok(plan) => {
                    print_json(&json!({
                        "identity_blake3": plan.launch_plan.identity_blake3,
                        "plan": plan.launch_plan.plan,
                        "status": "admitted"
                    }));
                    ExitCode::SUCCESS
                }
                Err(rejection) => reject("authorized-launch-plan", rejection),
            }
        }
        "admit-authorized-preflight" => {
            let [snapshot_path, authority_path] = rest else {
                eprintln!("usage: admit-authorized-preflight <snapshot.json> <authority.json>");
                return ExitCode::from(2);
            };
            let authority = match runtime_authority_from_file(authority_path) {
                Ok(authority) => authority,
                Err(rejection) => return reject("admit-authorized-preflight", rejection),
            };
            let snapshot = match read_json(snapshot_path) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            match admit_preflight_snapshot(&snapshot, &authority.bundle, &authority.config) {
                Ok(facts) => {
                    print_json(&json!({
                        "device_count": facts.device_board_ids.len(),
                        "firmware_bundle": facts.firmware_bundle,
                        "driver_version": facts.driver_version,
                        "runtime_authority_identity": authority.identity_blake3,
                        "status": "admitted"
                    }));
                    ExitCode::SUCCESS
                }
                Err(rejection) => reject("admit-authorized-preflight", rejection),
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
                        "device_count": facts.device_board_ids.len(),
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
        "validate-portable-artifact" => {
            let [artifact_path] = rest else {
                eprintln!("usage: validate-portable-artifact <artifact.json>");
                return ExitCode::from(2);
            };
            let artifact = match read_json(artifact_path) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            match validate_portable_artifact(&artifact) {
                Ok(()) => {
                    print_json(&json!({"status": "validated"}));
                    ExitCode::SUCCESS
                }
                Err(rejection) => reject("validate-portable-artifact", rejection),
            }
        }
        "classify-smoke-failure" => {
            let [output_path] = rest else {
                eprintln!("usage: classify-smoke-failure <stdout.log>");
                return ExitCode::from(2);
            };
            let output = match std::fs::read_to_string(output_path) {
                Ok(output) => output,
                Err(source) => {
                    eprintln!("failed to read {output_path}: {source}");
                    return ExitCode::from(2);
                }
            };
            println!("{}", classify_smoke_failure(&output));
            ExitCode::SUCCESS
        }
        "stage-states" => {
            let [outcome] = rest else {
                eprintln!("usage: stage-states <success|failed-stage>");
                return ExitCode::from(2);
            };
            let failed_stage = if outcome == "success" {
                None
            } else {
                Some(outcome.as_str())
            };
            match terminal_stage_states(failed_stage) {
                Ok(stages) => {
                    print_json(&stages);
                    ExitCode::SUCCESS
                }
                Err(rejection) => reject("stage-states", rejection),
            }
        }
        "validate-terminal-receipt" => {
            let [receipt_path, authority_path] = rest else {
                eprintln!("usage: validate-terminal-receipt <receipt.json> <authority.json>");
                return ExitCode::from(2);
            };
            let plan = match authorized_plan_from_file(authority_path) {
                Ok(plan) => plan,
                Err(rejection) => return reject("validate-terminal-receipt", rejection),
            };
            let receipt = match read_json(receipt_path) {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("{error}");
                    return ExitCode::from(2);
                }
            };
            match validate_terminal_receipt(
                &receipt,
                &plan.bundle_identity,
                &plan.config_identity,
                &plan.launch_plan.identity_blake3,
            ) {
                Ok(success) => {
                    print_json(&json!({
                        "outcome": if success { "success" } else { "failure" },
                        "status": "validated"
                    }));
                    ExitCode::SUCCESS
                }
                Err(rejection) => reject("validate-terminal-receipt", rejection),
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

fn required_authority_path<'a>(authority: &'a Value, pointer: &str) -> Result<&'a str, Rejection> {
    authority
        .pointer(pointer)
        .and_then(Value::as_str)
        .ok_or_else(|| {
            Rejection::new(
                "missing_field",
                format!("authority field {pointer} missing"),
            )
        })
}

fn file_blake3(path: &str) -> Result<String, Rejection> {
    let bytes = std::fs::read(path)
        .map_err(|source| Rejection::new("io", format!("failed to read {path}: {source}")))?;
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

fn runtime_authority_from_file(
    authority_path: &str,
) -> Result<training_platform_core::RuntimeAuthority, Rejection> {
    let text = std::fs::read_to_string(authority_path).map_err(|source| {
        Rejection::new("io", format!("failed to read {authority_path}: {source}"))
    })?;
    let authority_json: Value = serde_json::from_str(&text).map_err(|source| {
        Rejection::new(
            "parse",
            format!("failed to parse {authority_path}: {source}"),
        )
    })?;
    let descriptor_path = required_authority_path(&authority_json, "/descriptor/path")?;
    let requirements_path =
        required_authority_path(&authority_json, "/wheelhouse/requirements_path")?;
    let descriptor_blake3 = file_blake3(descriptor_path)?;
    let requirements_blake3 = file_blake3(requirements_path)?;
    admit_runtime_authority(&authority_json, &descriptor_blake3, &requirements_blake3)
}

fn authorized_plan_from_file(authority_path: &str) -> Result<PlanIdentities, Rejection> {
    let authority = runtime_authority_from_file(authority_path)?;
    let bundle_identity = authority.bundle.identity_blake3.clone();
    let config_identity = authority.config.identity_blake3.clone();
    let launch_plan = build_authorized_launch_plan(&authority)?;
    Ok(PlanIdentities {
        bundle_identity,
        config_identity,
        launch_plan,
    })
}

fn self_test() -> ExitCode {
    let manifest = json!({
        "schema_version": 1,
        "bundle_id": "tt-xla-training-2026-08-04",
        "recorded_date": "2026-08-04",
        "runtime_image": {
            "repository": "ghcr.io/tenstorrent/tt-xla/tt-xla-base-ubuntu-24-04",
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
            "repository": "https://github.com/tenstorrent/tt-xla",
            "revision": "3".repeat(40),
            "revision_date": "2026-08-04"
        },
        "tt_blacksmith_source": {
            "repository": "https://github.com/tenstorrent/tt-blacksmith",
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
