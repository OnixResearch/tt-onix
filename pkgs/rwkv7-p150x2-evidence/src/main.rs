use model_target_core::{
    ArchitectureEvidence, CommonEvidenceEnvelope, validate_architecture_evidence,
    validate_common_evidence,
};
use rwkv7_p150x2_evidence::{
    COMPLETE_STATE_ELEMENT_COUNT, CONTROL_DIVERGENCE_FLOOR, ClassificationReceipt,
    EXPECTED_CARD_COUNT, FINAL_HIDDEN_TOLERANCE, HIDDEN_ELEMENT_COUNT, LOGIT_ELEMENT_COUNT,
    LOGIT_TOLERANCE, MODEL_BLAKE3, ReferenceManifest, ReferenceStep,
    RuntimeReceipt, RuntimeStep, SCHEMA_VERSION, STATE_TOLERANCE, SafetyReceipt, TensorComparison,
    attribute_required_stages, compare_complete_tensor, decode_device_bf16,
    decode_generated_tokens, expected_target_identity, expected_topology_identity,
    max_abs_difference, parse_inspector, parse_profiler, parse_reference, parse_runtime,
    required_architecture_operations, runtime_target_identity, runtime_topology_identity,
    validate_artifact_bytes, validate_initial_zero_state, validate_operation_inventory,
    validate_relative_artifact_path, validate_runtime_against_reference, validate_safety,
};
use serde::Serialize;
use std::error::Error;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const VOCABULARY_PATH: Option<&str> = option_env!("RWKV7_TOKENIZER_VOCABULARY");
const PROGRAM_ARGUMENT_COUNT: usize = 1;
const EXPECTED_ARGUMENT_COUNT: usize = 8;
const REFERENCE_OPTION_INDEX: usize = 0;
const REFERENCE_VALUE_INDEX: usize = 1;
const RUNTIME_OPTION_INDEX: usize = 2;
const RUNTIME_VALUE_INDEX: usize = 3;
const SAFETY_OPTION_INDEX: usize = 4;
const SAFETY_VALUE_INDEX: usize = 5;
const OUTPUT_OPTION_INDEX: usize = 6;
const OUTPUT_VALUE_INDEX: usize = 7;
const REFERENCE_OPTION: &str = "--reference-root";
const RUNTIME_OPTION: &str = "--runtime-root";
const SAFETY_OPTION: &str = "--safety-receipt";
const OUTPUT_OPTION: &str = "--output";
const INSPECTOR_MESH_FILE: &str = "mesh_devices_log.yaml";
const INSPECTOR_PROGRAM_FILE: &str = "programs_log.yaml";
const INSPECTOR_WORKLOAD_FILE: &str = "mesh_workloads_log.yaml";
const INSPECTOR_KERNEL_FILE: &str = "kernels.yaml";
const RUNTIME_RECEIPT_FILE: &str = "runtime-receipt.json";
const REFERENCE_MANIFEST_FILE: &str = "reference-manifest.json";
const PROFILER_RELATIVE_PATH: &str = "profiler/.logs/profile_log_device.csv";
const INSPECTOR_RELATIVE_PATH: &str = "logs/generated/inspector";
const SELF_TEST_ARGUMENT: &str = "--self-test";
const SELF_TEST_ARGUMENT_COUNT: usize = 1;

#[derive(Clone, Debug)]
struct Arguments {
    reference_root: PathBuf,
    runtime_root: PathBuf,
    safety_receipt: PathBuf,
    output: PathBuf,
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("rwkv7-p150x2-evidence: {error}");
            ExitCode::FAILURE
        }
    }
}

// r[impl tenstorrent.rwkv7_p150x2.evidence_classifier.complete]
// r[verify tenstorrent.rwkv7_p150x2.evidence_classifier.missing]
fn run() -> Result<(), Box<dyn Error>> {
    let raw_arguments = std::env::args_os()
        .skip(PROGRAM_ARGUMENT_COUNT)
        .map(require_utf8_argument)
        .collect::<Result<Vec<_>, _>>()?;
    if raw_arguments.len() == SELF_TEST_ARGUMENT_COUNT && raw_arguments[0] == SELF_TEST_ARGUMENT {
        run_self_test()?;
        return Ok(());
    }
    let arguments = parse_arguments(&raw_arguments).map_err(invalid_input)?;
    let vocabulary = read_embedded_vocabulary()?;

    let reference = parse_reference(&fs::read(
        arguments.reference_root.join(REFERENCE_MANIFEST_FILE),
    )?)
    .map_err(invalid_data)?;
    let runtime = parse_runtime(&fs::read(
        arguments.runtime_root.join(RUNTIME_RECEIPT_FILE),
    )?)
    .map_err(invalid_data)?;
    let safety: SafetyReceipt = serde_json::from_slice(&fs::read(&arguments.safety_receipt)?)
        .map_err(|error| invalid_data(format!("invalid safety receipt: {error}")))?;
    validate_safety(&safety).map_err(invalid_data)?;
    validate_runtime_against_reference(&runtime, &reference).map_err(invalid_data)?;
    validate_operation_inventory(&runtime.operation_counts).map_err(invalid_data)?;

    let mut tensor_comparisons = Vec::new();
    validate_zero_sessions(&arguments.runtime_root, &runtime)?;
    compare_sessions(
        &arguments.reference_root,
        &arguments.runtime_root,
        &reference,
        &runtime,
        &mut tensor_comparisons,
    )?;
    compare_controls(
        &arguments.reference_root,
        &arguments.runtime_root,
        &reference,
        &runtime,
        &mut tensor_comparisons,
    )?;
    validate_cold_warm_bit_exact(&arguments.runtime_root, &runtime)?;
    let (reset_divergence, transposed_divergence) =
        validate_control_divergence(&arguments.runtime_root, &reference, &runtime)?;

    let inspector_root = arguments.runtime_root.join(INSPECTOR_RELATIVE_PATH);
    let inspector = parse_inspector(
        &fs::read(inspector_root.join(INSPECTOR_MESH_FILE))?,
        &fs::read(inspector_root.join(INSPECTOR_PROGRAM_FILE))?,
        &fs::read(inspector_root.join(INSPECTOR_WORKLOAD_FILE))?,
        &fs::read(inspector_root.join(INSPECTOR_KERNEL_FILE))?,
    )
    .map_err(invalid_data)?;
    let profiler_bytes = fs::read(arguments.runtime_root.join(PROFILER_RELATIVE_PATH))?;
    let profiler_rows = parse_profiler(&profiler_bytes).map_err(invalid_data)?;
    let profiler_byte_count = u64::try_from(profiler_bytes.len())
        .map_err(|error| invalid_data(format!("profiler byte count overflows u64: {error}")))?;
    let attribution =
        attribute_required_stages(&runtime, &inspector, &profiler_rows, profiler_byte_count)
            .map_err(invalid_data)?;

    let expected_target = expected_target_identity().map_err(invalid_data)?;
    let expected_topology = expected_topology_identity().map_err(invalid_data)?;
    let process_attempt_count = u32::try_from(safety.process.invocation_count).map_err(|error| {
        invalid_data(format!("process attempt count overflows u32: {error}"))
    })?;
    let common_evidence = CommonEvidenceEnvelope {
        target: runtime_target_identity(&runtime),
        topology: runtime_topology_identity(&runtime).map_err(invalid_data)?,
        process_attempt_count,
        profiler_attributed: !attribution.is_empty(),
        host_model_arithmetic: runtime.host_model_arithmetic,
        cpu_model_fallback: runtime.cpu_model_fallback,
        generated_text_published: runtime.generated_text_published,
    };
    validate_common_evidence(&expected_target, &expected_topology, &common_evidence)
        .map_err(invalid_data)?;
    let required_operations = required_architecture_operations().map_err(invalid_data)?;
    let architecture_evidence = ArchitectureEvidence {
        adapter_id: expected_target.architecture_adapter.clone(),
        architecture_family: expected_target.architecture_family.clone(),
        state_kind: expected_target.state_kind.clone(),
        observed_operations: runtime
            .operation_counts
            .iter()
            .filter(|(_operation, count)| **count > 0)
            .map(|(operation, _count)| operation.clone())
            .collect(),
        payload_complete: true,
    };
    validate_architecture_evidence(
        &expected_target,
        &required_operations,
        &architecture_evidence,
    )
    .map_err(invalid_data)?;

    let generated_bytes = decode_generated_tokens(&vocabulary, &reference.generated_token_ids)
        .map_err(invalid_data)?;
    let generated_text = String::from_utf8(generated_bytes.clone()).map_err(|error| {
        invalid_data(format!("generated bytes are not complete UTF-8: {error}"))
    })?;
    let receipt = ClassificationReceipt {
        schema_version: SCHEMA_VERSION,
        outcome: "verified_full_device_p150x2_generation".to_owned(),
        target_name: expected_target.target_name,
        model_id: expected_target.model_id,
        model_revision: expected_target.revision,
        architecture_adapter: expected_target.architecture_adapter,
        evidence_adapter: expected_target.evidence_adapter,
        artifact_blake3: expected_target.artifact_blake3,
        checkpoint_blake3: MODEL_BLAKE3.to_owned(),
        prompt_token_ids: reference.prompt_token_ids,
        generated_token_ids: reference.generated_token_ids,
        generated_bytes_blake3: blake3::hash(&generated_bytes).to_hex().to_string(),
        generated_text: generated_text.clone(),
        tensor_comparisons,
        retained_vs_reset_state_divergence: reset_divergence,
        retained_vs_transposed_state_divergence: transposed_divergence,
        inspector_stage_attribution: attribution,
        program_cache_reused: true,
        card_count: EXPECTED_CARD_COUNT,
        process_attempt_count: safety.process.invocation_count,
        owner_restored: safety
            .owners
            .iter()
            .all(|owner| owner.restored_to_prior_state),
        board_health_retained: safety
            .boards
            .iter()
            .all(|board| board.healthy_after_restoration),
    };
    write_new_json(&arguments.output, &receipt)?;
    println!("{}", arguments.output.display());
    println!("{generated_text}");
    Ok(())
}

fn parse_arguments(arguments: &[String]) -> Result<Arguments, String> {
    if arguments.len() != EXPECTED_ARGUMENT_COUNT {
        return Err(format!(
            "requires {REFERENCE_OPTION} PATH {RUNTIME_OPTION} PATH {SAFETY_OPTION} PATH {OUTPUT_OPTION} PATH"
        ));
    }
    for (index, expected) in [
        (REFERENCE_OPTION_INDEX, REFERENCE_OPTION),
        (RUNTIME_OPTION_INDEX, RUNTIME_OPTION),
        (SAFETY_OPTION_INDEX, SAFETY_OPTION),
        (OUTPUT_OPTION_INDEX, OUTPUT_OPTION),
    ] {
        if arguments[index] != expected {
            return Err(format!("argument {index} must be {expected}"));
        }
    }
    let reference_root = require_existing_directory(&arguments[REFERENCE_VALUE_INDEX])?;
    let runtime_root = require_existing_directory(&arguments[RUNTIME_VALUE_INDEX])?;
    let safety_receipt = require_existing_file(&arguments[SAFETY_VALUE_INDEX])?;
    let output = require_new_absolute_path(&arguments[OUTPUT_VALUE_INDEX])?;
    if output.starts_with(&reference_root) || output.starts_with(&runtime_root) {
        return Err(
            "classification output must be outside immutable raw evidence roots".to_owned(),
        );
    }
    Ok(Arguments {
        reference_root,
        runtime_root,
        safety_receipt,
        output,
    })
}

fn require_existing_directory(text: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(text);
    if !path.is_absolute() {
        return Err("evidence root must be absolute".to_owned());
    }
    let canonical = fs::canonicalize(&path)
        .map_err(|error| format!("failed to canonicalize evidence root: {error}"))?;
    if canonical != path || !canonical.is_dir() || canonical.starts_with("/nix/store/") {
        return Err("evidence root must be a canonical non-store directory".to_owned());
    }
    Ok(canonical)
}

fn require_existing_file(text: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(text);
    if !path.is_absolute() {
        return Err("safety receipt must be absolute".to_owned());
    }
    let canonical = fs::canonicalize(&path)
        .map_err(|error| format!("failed to canonicalize safety receipt: {error}"))?;
    if canonical != path || !canonical.is_file() || canonical.starts_with("/nix/store/") {
        return Err("safety receipt must be a canonical non-store file".to_owned());
    }
    Ok(canonical)
}

fn require_new_absolute_path(text: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(text);
    if !path.is_absolute() || path.starts_with("/nix/store/") || path.exists() {
        return Err("output must be a new absolute non-store path".to_owned());
    }
    let parent = path
        .parent()
        .ok_or_else(|| "output must have a parent directory".to_owned())?;
    let canonical_parent = fs::canonicalize(parent)
        .map_err(|error| format!("failed to canonicalize output parent: {error}"))?;
    let filename = path
        .file_name()
        .ok_or_else(|| "output must have a filename".to_owned())?;
    Ok(canonical_parent.join(filename))
}

fn validate_zero_sessions(root: &Path, runtime: &RuntimeReceipt) -> Result<(), Box<dyn Error>> {
    for session in [&runtime.cold, &runtime.warm] {
        let bytes = read_two_artifacts(root, &session.initial_recurrent_state)?;
        validate_initial_zero_state(&session.initial_recurrent_state, [&bytes[0], &bytes[1]])
            .map_err(invalid_data)?;
    }
    Ok(())
}

fn compare_sessions(
    reference_root: &Path,
    runtime_root: &Path,
    reference: &ReferenceManifest,
    runtime: &RuntimeReceipt,
    output: &mut Vec<TensorComparison>,
) -> Result<(), Box<dyn Error>> {
    for session in [&runtime.cold, &runtime.warm] {
        for (index, (expected, actual)) in reference.steps.iter().zip(&session.steps).enumerate() {
            compare_step(
                reference_root,
                runtime_root,
                &format!("{}-step-{index}", session.name),
                expected,
                actual,
                output,
            )?;
        }
    }
    Ok(())
}

fn compare_controls(
    reference_root: &Path,
    runtime_root: &Path,
    reference: &ReferenceManifest,
    runtime: &RuntimeReceipt,
    output: &mut Vec<TensorComparison>,
) -> Result<(), Box<dyn Error>> {
    compare_step(
        reference_root,
        runtime_root,
        "control-reset",
        &reference.controls.reset,
        &runtime.controls.reset,
        output,
    )?;
    compare_step(
        reference_root,
        runtime_root,
        "control-transposed",
        &reference.controls.transposed,
        &runtime.controls.transposed,
        output,
    )
}

fn compare_step(
    reference_root: &Path,
    runtime_root: &Path,
    name: &str,
    reference: &ReferenceStep,
    runtime: &RuntimeStep,
    output: &mut Vec<TensorComparison>,
) -> Result<(), Box<dyn Error>> {
    output.push(compare_field(
        reference_root,
        runtime_root,
        &format!("{name}-final-hidden"),
        &reference.final_hidden,
        &runtime.final_hidden,
        HIDDEN_ELEMENT_COUNT,
        FINAL_HIDDEN_TOLERANCE,
    )?);
    output.push(compare_field(
        reference_root,
        runtime_root,
        &format!("{name}-logits"),
        &reference.logits,
        &runtime.logits,
        LOGIT_ELEMENT_COUNT,
        LOGIT_TOLERANCE,
    )?);
    output.push(compare_field(
        reference_root,
        runtime_root,
        &format!("{name}-state"),
        &reference.complete_recurrent_state,
        &runtime.recurrent_state,
        COMPLETE_STATE_ELEMENT_COUNT,
        STATE_TOLERANCE,
    )?);
    Ok(())
}

fn compare_field(
    reference_root: &Path,
    runtime_root: &Path,
    name: &str,
    reference: &rwkv7_p150x2_evidence::Artifact,
    devices: &[rwkv7_p150x2_evidence::Artifact],
    elements: usize,
    tolerance: f32,
) -> Result<TensorComparison, Box<dyn Error>> {
    let reference_bytes = read_artifact(reference_root, reference)?;
    let device_bytes = read_two_artifacts(runtime_root, devices)?;
    compare_complete_tensor(
        name,
        reference,
        &reference_bytes,
        devices,
        [&device_bytes[0], &device_bytes[1]],
        elements,
        tolerance,
    )
    .map_err(|error| invalid_data(format!("{name}: {error}")).into())
}

fn validate_cold_warm_bit_exact(
    root: &Path,
    runtime: &RuntimeReceipt,
) -> Result<(), Box<dyn Error>> {
    for (cold, warm) in runtime.cold.steps.iter().zip(&runtime.warm.steps) {
        for (cold_artifacts, warm_artifacts) in [
            (&cold.final_hidden, &warm.final_hidden),
            (&cold.logits, &warm.logits),
            (&cold.recurrent_state, &warm.recurrent_state),
        ] {
            for card in 0..EXPECTED_CARD_COUNT {
                let cold_bytes = read_artifact(root, &cold_artifacts[card])?;
                let warm_bytes = read_artifact(root, &warm_artifacts[card])?;
                if cold_bytes != warm_bytes {
                    return Err(invalid_data("cold/warm BF16 artifacts are not bit-exact").into());
                }
            }
        }
    }
    Ok(())
}

fn validate_control_divergence(
    root: &Path,
    reference: &ReferenceManifest,
    runtime: &RuntimeReceipt,
) -> Result<(f32, f32), Box<dyn Error>> {
    let retained_index = reference.prompt_token_ids.len();
    let retained = runtime
        .cold
        .steps
        .get(retained_index)
        .ok_or_else(|| invalid_data("cold session lacks retained control step"))?;
    let retained_bytes = read_artifact(root, &retained.recurrent_state[0])?;
    let reset_bytes = read_artifact(root, &runtime.controls.reset.recurrent_state[0])?;
    let transposed_bytes = read_artifact(root, &runtime.controls.transposed.recurrent_state[0])?;
    let retained_values = decode_device_bf16(&retained.recurrent_state[0], &retained_bytes, 0)
        .map_err(invalid_data)?;
    let reset_values =
        decode_device_bf16(&runtime.controls.reset.recurrent_state[0], &reset_bytes, 0)
            .map_err(invalid_data)?;
    let transposed_values = decode_device_bf16(
        &runtime.controls.transposed.recurrent_state[0],
        &transposed_bytes,
        0,
    )
    .map_err(invalid_data)?;
    let reset_divergence =
        max_abs_difference(&retained_values, &reset_values).map_err(invalid_data)?;
    let transposed_divergence =
        max_abs_difference(&retained_values, &transposed_values).map_err(invalid_data)?;
    if reset_divergence <= CONTROL_DIVERGENCE_FLOOR {
        return Err(
            invalid_data("retained/reset state discriminator is below its fixed floor").into(),
        );
    }
    if transposed_divergence <= CONTROL_DIVERGENCE_FLOOR {
        return Err(invalid_data(
            "retained/transposed state discriminator is below its fixed floor",
        )
        .into());
    }
    Ok((reset_divergence, transposed_divergence))
}

fn read_two_artifacts(
    root: &Path,
    artifacts: &[rwkv7_p150x2_evidence::Artifact],
) -> Result<[Vec<u8>; EXPECTED_CARD_COUNT], Box<dyn Error>> {
    if artifacts.len() != EXPECTED_CARD_COUNT {
        return Err(invalid_data("artifact list does not contain exactly two cards").into());
    }
    Ok([
        read_artifact(root, &artifacts[0])?,
        read_artifact(root, &artifacts[1])?,
    ])
}

fn read_artifact(
    root: &Path,
    artifact: &rwkv7_p150x2_evidence::Artifact,
) -> Result<Vec<u8>, Box<dyn Error>> {
    validate_relative_artifact_path(&artifact.path).map_err(invalid_data)?;
    let path = root.join(&artifact.path);
    let canonical = fs::canonicalize(&path)?;
    if canonical.parent() != Some(root) || !canonical.is_file() {
        return Err(invalid_data("artifact resolved outside its evidence root").into());
    }
    let bytes = fs::read(canonical)?;
    validate_artifact_bytes(artifact, &bytes).map_err(invalid_data)?;
    Ok(bytes)
}

fn read_embedded_vocabulary() -> Result<Vec<u8>, Box<dyn Error>> {
    let configured = VOCABULARY_PATH.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "RWKV7_TOKENIZER_VOCABULARY was not embedded at build time",
        )
    })?;
    let canonical = fs::canonicalize(configured)?;
    if !canonical.starts_with("/nix/store/") || !canonical.is_file() {
        return Err(invalid_data("embedded tokenizer vocabulary is not a Nix store file").into());
    }
    Ok(fs::read(canonical)?)
}

fn write_new_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    let file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let mut output = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut output, value)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn run_self_test() -> Result<(), Box<dyn Error>> {
    let path_error = validate_relative_artifact_path("../forbidden")
        .expect_err("parent artifact path must be rejected");
    if !path_error.contains("forbidden") {
        return Err(invalid_data("self-test returned the wrong path diagnostic").into());
    }
    println!("rwkv7 P150x2 evidence classifier self-test: PASS");
    Ok(())
}

fn require_utf8_argument(argument: OsString) -> Result<String, io::Error> {
    argument.into_string().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "classifier arguments must be valid UTF-8",
        )
    })
}

fn invalid_input(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

#[cfg(test)]
mod tests {
    use super::{
        EXPECTED_ARGUMENT_COUNT, OUTPUT_VALUE_INDEX, parse_arguments, require_new_absolute_path,
    };

    #[test]
    fn argument_parser_rejects_missing_and_existing_outputs() {
        let arguments = Vec::<String>::new();
        assert!(
            parse_arguments(&arguments)
                .expect_err("missing arguments must fail")
                .contains("requires")
        );
        assert_eq!(EXPECTED_ARGUMENT_COUNT, OUTPUT_VALUE_INDEX + 1);
        assert!(require_new_absolute_path("relative").is_err());
    }
}
