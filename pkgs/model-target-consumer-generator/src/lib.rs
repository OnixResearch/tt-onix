use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

const REVISION_HEX_LENGTH: usize = 40;
const BLAKE3_HEX_LENGTH: usize = 64;
const CANONICAL_TARGET_NAME: &str = "rwkv7-goose-world2.8-0.1b-p150x2";
const GENERATED_NOTICE_RUST: &str = "// Generated from model-targets/profiles.ncl. Do not edit.\n";
const GENERATED_NOTICE_PYTHON: &str = "# Generated from model-targets/profiles.ncl. Do not edit.\n";
const GENERATED_NOTICE_CPP: &str = "// Generated from model-targets/profiles.ncl. Do not edit.\n";
const RUST_OUTPUTS: &[&str] = &[
    "model-targets/generated/consumer-identities.rs",
    "pkgs/rwkv-layer-harness/src/generated_model_target_identity.rs",
    "pkgs/rwkv7-p150x2-evidence/src/generated_model_target_identity.rs",
    "pkgs/rwkv7-lora-p150a/core/src/generated_model_target_identity.rs",
    "pkgs/rwkv7-lora-dual-p150a/core/src/generated_model_target_identity.rs",
];
const PYTHON_OUTPUTS: &[&str] = &[
    "model-targets/generated/consumer_identities.py",
    "pkgs/rwkv-layer-harness/reference/generated_model_target_identity.py",
    "pkgs/rwkv7-lora-reference/generated_model_target_identity.py",
];
const CPP_OUTPUTS: &[&str] = &[
    "model-targets/generated/consumer-identities.h",
    "pkgs/ttwkv7/generated-model-target-identity.h",
];
const NIX_OUTPUT: &str = "model-targets/generated/consumer-identities.nix";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RenderedFile {
    pub path: PathBuf,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Debug)]
struct Projection {
    target_name: String,
    profile_blake3: String,
    value: Value,
}

pub fn render_consumer_files(input: &[u8]) -> Result<Vec<RenderedFile>, String> {
    let root: Value = serde_json::from_slice(input)
        .map_err(|error| format!("invalid model profile JSON: {error}"))?;
    let projections = validate_and_project(&root)?;
    let canonical = projections
        .iter()
        .find(|projection| projection.target_name == CANONICAL_TARGET_NAME)
        .ok_or_else(|| format!("missing canonical profile '{CANONICAL_TARGET_NAME}'"))?;

    let rust = render_rust(&projections, canonical)?;
    let python = render_python(&projections, canonical)?;
    let cpp = render_cpp(&projections, canonical)?;
    let nix = render_nix(&projections)?;

    let mut files = Vec::new();
    for path in RUST_OUTPUTS {
        files.push(RenderedFile {
            path: PathBuf::from(path),
            bytes: rust.as_bytes().to_vec(),
        });
    }
    for path in PYTHON_OUTPUTS {
        files.push(RenderedFile {
            path: PathBuf::from(path),
            bytes: python.as_bytes().to_vec(),
        });
    }
    for path in CPP_OUTPUTS {
        files.push(RenderedFile {
            path: PathBuf::from(path),
            bytes: cpp.as_bytes().to_vec(),
        });
    }
    files.push(RenderedFile {
        path: PathBuf::from(NIX_OUTPUT),
        bytes: nix.as_bytes().to_vec(),
    });
    Ok(files)
}

pub fn find_drift(expected: &[RenderedFile], root: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut drifted = Vec::new();
    for rendered in expected {
        let path = root.join(&rendered.path);
        match std::fs::read(&path) {
            Ok(actual) if actual == rendered.bytes => {}
            Ok(_) | Err(_) => drifted.push(rendered.path.clone()),
        }
    }
    Ok(drifted)
}

fn validate_and_project(root: &Value) -> Result<Vec<Projection>, String> {
    let profiles = required_array(root, "/profiles")?;
    if profiles.is_empty() {
        return Err("field /profiles must not be empty".to_owned());
    }
    let mut names = BTreeSet::new();
    let mut projections = Vec::with_capacity(profiles.len());
    for (index, profile) in profiles.iter().enumerate() {
        let prefix = format!("/profiles/{index}");
        let target_name = required_string(profile, "/target_name", &prefix)?;
        if !names.insert(target_name.to_owned()) {
            return Err(format!(
                "field {prefix}/target_name is duplicated: {target_name}"
            ));
        }
        validate_profile(profile, &prefix)?;
        let canonical_bytes = serde_json::to_vec(profile)
            .map_err(|error| format!("failed to serialize profile {target_name}: {error}"))?;
        let profile_blake3 = blake3::hash(&canonical_bytes).to_hex().to_string();
        let mut projected = required_object(profile, "", &prefix)?.clone();
        projected.insert(
            "profile_blake3".to_owned(),
            Value::String(profile_blake3.clone()),
        );
        projections.push(Projection {
            target_name: target_name.to_owned(),
            profile_blake3,
            value: Value::Object(projected),
        });
    }
    projections.sort_by(|left, right| left.target_name.cmp(&right.target_name));
    Ok(projections)
}

fn validate_profile(profile: &Value, prefix: &str) -> Result<(), String> {
    let revision = required_string(profile, "/model/revision", prefix)?;
    validate_lower_hex(
        revision,
        REVISION_HEX_LENGTH,
        &format!("{prefix}/model/revision"),
    )?;

    let artifacts = required_array(profile, "/artifacts")?;
    if artifacts.is_empty() {
        return Err(format!("field {prefix}/artifacts must not be empty"));
    }
    let mut roles = BTreeSet::new();
    for (artifact_index, artifact) in artifacts.iter().enumerate() {
        let artifact_prefix = format!("{prefix}/artifacts/{artifact_index}");
        let role = required_string(artifact, "/role", &artifact_prefix)?;
        if !roles.insert(role.to_owned()) {
            return Err(format!(
                "field {artifact_prefix}/role is duplicated: {role}"
            ));
        }
        let hash = required_string(artifact, "/blake3", &artifact_prefix)?;
        validate_lower_hex(
            hash,
            BLAKE3_HEX_LENGTH,
            &format!("{artifact_prefix}/blake3"),
        )?;
    }
    let hidden_size = required_u64(profile, "/resources/hidden_size", prefix)?;
    let layer_count = required_u64(profile, "/resources/layer_count", prefix)?;
    let per_layer_state = required_u64(profile, "/state/per_layer_element_count", prefix)?;
    let head_count = required_u64(profile, "/consumer/dimensions/head_count", prefix)?;
    let head_size = required_u64(profile, "/consumer/dimensions/head_size", prefix)?;
    let complete_state = required_u64(
        profile,
        "/consumer/dimensions/complete_state_element_count",
        prefix,
    )?;
    let expected_hidden = head_count.checked_mul(head_size).ok_or_else(|| {
        format!("field {prefix}/consumer/dimensions overflows hidden-size validation")
    })?;
    if expected_hidden != hidden_size {
        return Err(format!(
            "field {prefix}/consumer/dimensions/head_count is inconsistent: head_count * head_size is {expected_hidden}, expected hidden_size {hidden_size}"
        ));
    }
    let expected_complete_state = layer_count.checked_mul(per_layer_state).ok_or_else(|| {
        format!("field {prefix}/consumer/dimensions overflows state-size validation")
    })?;
    if expected_complete_state != complete_state {
        return Err(format!(
            "field {prefix}/consumer/dimensions/complete_state_element_count is inconsistent: got {complete_state}, expected {expected_complete_state}"
        ));
    }

    let mesh_shape = required_array(profile, "/topology/mesh_shape")?;
    let physical_device_ids = required_array(profile, "/topology/physical_device_ids")?;
    if mesh_shape.is_empty() {
        return Err(format!(
            "field {prefix}/topology/mesh_shape must not be empty"
        ));
    }
    if physical_device_ids.is_empty() {
        return Err(format!(
            "field {prefix}/topology/physical_device_ids must not be empty"
        ));
    }

    for path in [
        "/consumer/precisions/cpu_reference",
        "/consumer/precisions/runtime_arithmetic",
        "/consumer/precisions/ttwkv7_boundary",
        "/consumer/adapter_format",
        "/adapters/architecture",
        "/adapters/evidence",
        "/topology/id",
    ] {
        let value = required_string(profile, path, prefix)?;
        if value.is_empty() {
            return Err(format!("field {prefix}{path} must not be empty"));
        }
    }
    for path in [
        "/consumer/tolerances/harness_oracle",
        "/consumer/tolerances/harness_replay",
        "/consumer/tolerances/state_carry_divergence_floor",
        "/consumer/tolerances/final_hidden",
        "/consumer/tolerances/logits",
        "/consumer/tolerances/recurrent_state",
        "/consumer/tolerances/control_divergence_floor",
        "/consumer/tolerances/lora_gradient_agreement",
        "/consumer/tolerances/lora_finite_difference",
    ] {
        let value = required_f64(profile, path, prefix)?;
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "field {prefix}{path} must be a positive finite number"
            ));
        }
    }
    Ok(())
}

fn render_rust(projections: &[Projection], canonical: &Projection) -> Result<String, String> {
    let value = &canonical.value;
    let artifacts = artifact_map(value)?;
    let mut output = String::from(GENERATED_NOTICE_RUST);
    output.push_str("#![allow(dead_code)]\n\n");
    render_rust_string(&mut output, "PROFILE_BLAKE3", &canonical.profile_blake3);
    for (name, path) in [
        ("TARGET_NAME", "/target_name"),
        ("MODEL_ID", "/model/id"),
        ("MODEL_REVISION", "/model/revision"),
        ("ARCHITECTURE_FAMILY", "/model/architecture_family"),
        ("STATE_KIND", "/state/kind"),
        ("MODEL_PRECISION", "/model/precision"),
        ("ARCHITECTURE_ADAPTER_ID", "/adapters/architecture"),
        ("EVIDENCE_ADAPTER_ID", "/adapters/evidence"),
        ("TOPOLOGY_ID", "/topology/id"),
        ("REPLICATION", "/topology/replication"),
        (
            "CPU_REFERENCE_PRECISION",
            "/consumer/precisions/cpu_reference",
        ),
        (
            "RUNTIME_ARITHMETIC_PRECISION",
            "/consumer/precisions/runtime_arithmetic",
        ),
        (
            "TTWKV7_BOUNDARY_PRECISION",
            "/consumer/precisions/ttwkv7_boundary",
        ),
        ("ADAPTER_FORMAT", "/consumer/adapter_format"),
    ] {
        render_rust_string(&mut output, name, required_string(value, path, "")?);
    }
    for (name, role, field) in [
        ("MODEL_BLAKE3", "checkpoint", "blake3"),
        ("MODEL_SHA256_SRI", "checkpoint", "nix_hash"),
        ("TOKENIZER_VOCAB_BLAKE3", "tokenizer_vocabulary", "blake3"),
        (
            "TOKENIZER_VOCAB_SHA256_SRI",
            "tokenizer_vocabulary",
            "nix_hash",
        ),
        ("TOKENIZER_CONFIG_BLAKE3", "tokenizer_config", "blake3"),
        (
            "TOKENIZER_CONFIG_SHA256_SRI",
            "tokenizer_config",
            "nix_hash",
        ),
        ("ADDED_TOKENS_BLAKE3", "tokenizer_added_tokens", "blake3"),
        (
            "ADDED_TOKENS_SHA256_SRI",
            "tokenizer_added_tokens",
            "nix_hash",
        ),
        (
            "TOKENIZER_IMPLEMENTATION_BLAKE3",
            "tokenizer_implementation",
            "blake3",
        ),
        (
            "TOKENIZER_IMPLEMENTATION_SHA256_SRI",
            "tokenizer_implementation",
            "nix_hash",
        ),
        (
            "SPECIAL_TOKENS_MAP_BLAKE3",
            "tokenizer_special_tokens",
            "blake3",
        ),
        (
            "SPECIAL_TOKENS_MAP_SHA256_SRI",
            "tokenizer_special_tokens",
            "nix_hash",
        ),
        ("MODEL_CONFIG_BLAKE3", "model_config", "blake3"),
        ("MODEL_CONFIG_SHA256_SRI", "model_config", "nix_hash"),
        ("GENERATION_CONFIG_BLAKE3", "generation_config", "blake3"),
        (
            "GENERATION_CONFIG_SHA256_SRI",
            "generation_config",
            "nix_hash",
        ),
    ] {
        let artifact = artifacts.get(role).ok_or_else(|| {
            format!(
                "profile {} is missing artifact {role}",
                canonical.target_name
            )
        })?;
        render_rust_string(
            &mut output,
            name,
            required_string(artifact, &format!("/{field}"), "")?,
        );
    }
    for (name, path, rust_type) in [
        ("MODEL_BYTE_COUNT", "/resources/checkpoint_bytes", "u64"),
        ("MODEL_LAYER_COUNT", "/resources/layer_count", "usize"),
        ("HIDDEN_SIZE", "/resources/hidden_size", "usize"),
        ("VOCABULARY_SIZE", "/resources/vocabulary_size", "usize"),
        (
            "MAXIMUM_PROMPT_TOKENS",
            "/resources/maximum_prompt_tokens",
            "usize",
        ),
        (
            "PER_LAYER_STATE_ELEMENT_COUNT",
            "/state/per_layer_element_count",
            "usize",
        ),
        ("HEAD_COUNT", "/consumer/dimensions/head_count", "usize"),
        ("HEAD_SIZE", "/consumer/dimensions/head_size", "usize"),
        (
            "INTERMEDIATE_SIZE",
            "/consumer/dimensions/intermediate_size",
            "usize",
        ),
        (
            "COMPLETE_STATE_ELEMENT_COUNT",
            "/consumer/dimensions/complete_state_element_count",
            "usize",
        ),
    ] {
        let number = required_u64(value, path, "")?;
        writeln!(output, "pub const {name}: {rust_type} = {number};")
            .map_err(|error| error.to_string())?;
    }
    for (name, path) in tolerance_fields() {
        let number = required_f64(value, path, "")?;
        let literal = format!("{number:?}");
        writeln!(output, "pub const {name}: f32 = {literal};")
            .map_err(|error| error.to_string())?;
    }
    let artifact_hashes = artifact_blake3_object(value)?;
    let mesh_shape = compact_json(required_value(value, "/topology/mesh_shape", "")?)?;
    let physical_ids = compact_json(required_value(value, "/topology/physical_device_ids", "")?)?;
    let all_json = projection_json(projections)?;
    render_rust_raw_string(
        &mut output,
        "ARTIFACT_BLAKE3_JSON",
        &compact_json(&artifact_hashes)?,
    );
    render_rust_raw_string(&mut output, "MESH_SHAPE_JSON", &mesh_shape);
    render_rust_raw_string(&mut output, "PHYSICAL_DEVICE_IDS_JSON", &physical_ids);
    render_rust_raw_string(&mut output, "PROFILES_JSON", &all_json);
    Ok(output)
}

fn render_python(projections: &[Projection], canonical: &Projection) -> Result<String, String> {
    let value = &canonical.value;
    let artifacts = artifact_map(value)?;
    let mut output = String::from(GENERATED_NOTICE_PYTHON);
    output.push_str("# ruff: noqa\n# fmt: off\nimport json\n\n");
    render_python_string(&mut output, "PROFILE_BLAKE3", &canonical.profile_blake3);
    for (name, path) in [
        ("TARGET_NAME", "/target_name"),
        ("MODEL_ID", "/model/id"),
        ("MODEL_REVISION", "/model/revision"),
        ("ARCHITECTURE_FAMILY", "/model/architecture_family"),
        ("STATE_KIND", "/state/kind"),
        ("MODEL_PRECISION", "/model/precision"),
        ("ARCHITECTURE_ADAPTER_ID", "/adapters/architecture"),
        ("EVIDENCE_ADAPTER_ID", "/adapters/evidence"),
        ("TOPOLOGY_ID", "/topology/id"),
        ("REPLICATION", "/topology/replication"),
        (
            "CPU_REFERENCE_PRECISION",
            "/consumer/precisions/cpu_reference",
        ),
        (
            "RUNTIME_ARITHMETIC_PRECISION",
            "/consumer/precisions/runtime_arithmetic",
        ),
        (
            "TTWKV7_BOUNDARY_PRECISION",
            "/consumer/precisions/ttwkv7_boundary",
        ),
        ("ADAPTER_FORMAT", "/consumer/adapter_format"),
    ] {
        render_python_string(&mut output, name, required_string(value, path, "")?);
    }
    for (name, role, field) in [
        ("MODEL_BLAKE3", "checkpoint", "blake3"),
        ("MODEL_SHA256_SRI", "checkpoint", "nix_hash"),
        ("TOKENIZER_VOCAB_BLAKE3", "tokenizer_vocabulary", "blake3"),
        (
            "TOKENIZER_VOCAB_SHA256_SRI",
            "tokenizer_vocabulary",
            "nix_hash",
        ),
    ] {
        let artifact = artifacts.get(role).ok_or_else(|| {
            format!(
                "profile {} is missing artifact {role}",
                canonical.target_name
            )
        })?;
        render_python_string(
            &mut output,
            name,
            required_string(artifact, &format!("/{field}"), "")?,
        );
    }
    for (name, path) in numeric_fields() {
        let number = required_value(value, path, "")?;
        writeln!(output, "{name} = {number}").map_err(|error| error.to_string())?;
    }
    for (name, path) in tolerance_fields() {
        let number = required_value(value, path, "")?;
        writeln!(output, "{name} = {number}").map_err(|error| error.to_string())?;
    }
    let all_json = projection_json(projections)?;
    writeln!(output, "PROFILES = json.loads(r'''{all_json}''')")
        .map_err(|error| error.to_string())?;
    writeln!(
        output,
        "ARTIFACT_BLAKE3 = json.loads(r'''{}''')",
        compact_json(&artifact_blake3_object(value)?)?
    )
    .map_err(|error| error.to_string())?;
    output.push_str("# fmt: on\n");
    Ok(output)
}

fn render_cpp(projections: &[Projection], canonical: &Projection) -> Result<String, String> {
    let value = &canonical.value;
    let artifacts = artifact_map(value)?;
    let mut output = String::from(GENERATED_NOTICE_CPP);
    output.push_str("#pragma once\n\n#include <cstddef>\n#include <cstdint>\n#include <string_view>\n\nnamespace model_target_identity {\n");
    render_cpp_string(&mut output, "kProfileBlake3", &canonical.profile_blake3);
    for (name, path) in [
        ("kTargetName", "/target_name"),
        ("kModelId", "/model/id"),
        ("kModelRevision", "/model/revision"),
        ("kArchitectureFamily", "/model/architecture_family"),
        ("kStateKind", "/state/kind"),
        ("kModelPrecision", "/model/precision"),
        ("kArchitectureAdapterId", "/adapters/architecture"),
        ("kEvidenceAdapterId", "/adapters/evidence"),
        ("kTopologyId", "/topology/id"),
        ("kReplication", "/topology/replication"),
        (
            "kCpuReferencePrecision",
            "/consumer/precisions/cpu_reference",
        ),
        (
            "kRuntimeArithmeticPrecision",
            "/consumer/precisions/runtime_arithmetic",
        ),
        (
            "kTtwkv7BoundaryPrecision",
            "/consumer/precisions/ttwkv7_boundary",
        ),
        ("kAdapterFormat", "/consumer/adapter_format"),
    ] {
        render_cpp_string(&mut output, name, required_string(value, path, "")?);
    }
    for (name, role, field) in [
        ("kModelBlake3", "checkpoint", "blake3"),
        ("kModelSha256Sri", "checkpoint", "nix_hash"),
        ("kTokenizerVocabBlake3", "tokenizer_vocabulary", "blake3"),
        (
            "kTokenizerVocabSha256Sri",
            "tokenizer_vocabulary",
            "nix_hash",
        ),
    ] {
        let artifact = artifacts.get(role).ok_or_else(|| {
            format!(
                "profile {} is missing artifact {role}",
                canonical.target_name
            )
        })?;
        render_cpp_string(
            &mut output,
            name,
            required_string(artifact, &format!("/{field}"), "")?,
        );
    }
    for (name, path) in [
        ("kModelByteCount", "/resources/checkpoint_bytes"),
        ("kModelLayerCount", "/resources/layer_count"),
        ("kHiddenSize", "/resources/hidden_size"),
        ("kVocabularySize", "/resources/vocabulary_size"),
        ("kMaximumPromptTokens", "/resources/maximum_prompt_tokens"),
        (
            "kPerLayerStateElementCount",
            "/state/per_layer_element_count",
        ),
        ("kHeadCount", "/consumer/dimensions/head_count"),
        ("kHeadSize", "/consumer/dimensions/head_size"),
        (
            "kIntermediateSize",
            "/consumer/dimensions/intermediate_size",
        ),
        (
            "kCompleteStateElementCount",
            "/consumer/dimensions/complete_state_element_count",
        ),
    ] {
        let number = required_u64(value, path, "")?;
        writeln!(output, "inline constexpr std::uint64_t {name} = {number};")
            .map_err(|error| error.to_string())?;
    }
    for (rust_name, path) in tolerance_fields() {
        let cpp_name = rust_constant_to_cpp(rust_name);
        let number = required_f64(value, path, "")?;
        writeln!(output, "inline constexpr double {cpp_name} = {number};")
            .map_err(|error| error.to_string())?;
    }
    let all_json = projection_json(projections)?;
    writeln!(
        output,
        "inline constexpr std::string_view kProfilesJson = R\"json({all_json})json\";"
    )
    .map_err(|error| error.to_string())?;
    output.push_str("}  // namespace model_target_identity\n");
    Ok(output)
}

fn render_nix(projections: &[Projection]) -> Result<String, String> {
    let all_json = projection_json(projections)?;
    Ok(format!(
        "# Generated from model-targets/profiles.ncl. Do not edit.\nbuiltins.fromJSON ''\n  {all_json}\n''\n"
    ))
}

fn projection_json(projections: &[Projection]) -> Result<String, String> {
    let values = projections
        .iter()
        .map(|projection| projection.value.clone())
        .collect::<Vec<_>>();
    compact_json(&Value::Array(values))
}

fn artifact_map(value: &Value) -> Result<BTreeMap<String, Value>, String> {
    let mut result = BTreeMap::new();
    for artifact in required_array(value, "/artifacts")? {
        let role = required_string(artifact, "/role", "")?.to_owned();
        result.insert(role, artifact.clone());
    }
    Ok(result)
}

fn artifact_blake3_object(value: &Value) -> Result<Value, String> {
    let mut result = Map::new();
    for (role, artifact) in artifact_map(value)? {
        result.insert(
            role,
            Value::String(required_string(&artifact, "/blake3", "")?.to_owned()),
        );
    }
    Ok(Value::Object(result))
}

fn numeric_fields() -> &'static [(&'static str, &'static str)] {
    &[
        ("MODEL_BYTE_COUNT", "/resources/checkpoint_bytes"),
        ("MODEL_LAYER_COUNT", "/resources/layer_count"),
        ("HIDDEN_SIZE", "/resources/hidden_size"),
        ("VOCABULARY_SIZE", "/resources/vocabulary_size"),
        ("MAXIMUM_PROMPT_TOKENS", "/resources/maximum_prompt_tokens"),
        (
            "PER_LAYER_STATE_ELEMENT_COUNT",
            "/state/per_layer_element_count",
        ),
        ("HEAD_COUNT", "/consumer/dimensions/head_count"),
        ("HEAD_SIZE", "/consumer/dimensions/head_size"),
        (
            "INTERMEDIATE_SIZE",
            "/consumer/dimensions/intermediate_size",
        ),
        (
            "COMPLETE_STATE_ELEMENT_COUNT",
            "/consumer/dimensions/complete_state_element_count",
        ),
    ]
}

fn tolerance_fields() -> &'static [(&'static str, &'static str)] {
    &[
        (
            "HARNESS_ORACLE_TOLERANCE",
            "/consumer/tolerances/harness_oracle",
        ),
        (
            "HARNESS_REPLAY_TOLERANCE",
            "/consumer/tolerances/harness_replay",
        ),
        (
            "STATE_CARRY_DIVERGENCE_FLOOR",
            "/consumer/tolerances/state_carry_divergence_floor",
        ),
        (
            "FINAL_HIDDEN_TOLERANCE",
            "/consumer/tolerances/final_hidden",
        ),
        ("LOGIT_TOLERANCE", "/consumer/tolerances/logits"),
        (
            "RECURRENT_STATE_TOLERANCE",
            "/consumer/tolerances/recurrent_state",
        ),
        (
            "CONTROL_DIVERGENCE_FLOOR",
            "/consumer/tolerances/control_divergence_floor",
        ),
        (
            "LORA_GRADIENT_AGREEMENT_TOLERANCE",
            "/consumer/tolerances/lora_gradient_agreement",
        ),
        (
            "LORA_FINITE_DIFFERENCE_TOLERANCE",
            "/consumer/tolerances/lora_finite_difference",
        ),
    ]
}

fn render_rust_string(output: &mut String, name: &str, value: &str) {
    let literal = serde_json::to_string(value).expect("a string must serialize");
    writeln!(
        output,
        "#[rustfmt::skip]\npub const {name}: &str = {literal};"
    )
    .expect("writing to a string cannot fail");
}

fn render_rust_raw_string(output: &mut String, name: &str, value: &str) {
    writeln!(
        output,
        "#[rustfmt::skip]\npub const {name}: &str = r#\"{value}\"#;"
    )
    .expect("writing to a string cannot fail");
}

fn render_python_string(output: &mut String, name: &str, value: &str) {
    let literal = serde_json::to_string(value).expect("a string must serialize");
    writeln!(output, "{name} = {literal}").expect("writing to a string cannot fail");
}

fn render_cpp_string(output: &mut String, name: &str, value: &str) {
    let literal = serde_json::to_string(value).expect("a string must serialize");
    writeln!(
        output,
        "inline constexpr std::string_view {name} = {literal};"
    )
    .expect("writing to a string cannot fail");
}

fn rust_constant_to_cpp(value: &str) -> String {
    let mut output = String::from("k");
    for part in value.split('_') {
        let mut characters = part.chars();
        if let Some(first) = characters.next() {
            output.extend(first.to_uppercase());
            output.extend(characters.map(|character| character.to_ascii_lowercase()));
        }
    }
    output
}

fn compact_json(value: &Value) -> Result<String, String> {
    serde_json::to_string(value)
        .map_err(|error| format!("failed to render canonical JSON: {error}"))
}

fn validate_lower_hex(value: &str, length: usize, field: &str) -> Result<(), String> {
    if value.len() != length {
        return Err(format!(
            "field {field} must contain exactly {length} lowercase hexadecimal characters; got {}",
            value.len()
        ));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!(
            "field {field} must contain lowercase hexadecimal characters only"
        ));
    }
    Ok(())
}

fn required_value<'a>(value: &'a Value, path: &str, prefix: &str) -> Result<&'a Value, String> {
    value
        .pointer(path)
        .ok_or_else(|| format!("missing field {prefix}{path}"))
}

fn required_string<'a>(value: &'a Value, path: &str, prefix: &str) -> Result<&'a str, String> {
    required_value(value, path, prefix)?
        .as_str()
        .ok_or_else(|| format!("field {prefix}{path} must be a string"))
}

fn required_u64(value: &Value, path: &str, prefix: &str) -> Result<u64, String> {
    required_value(value, path, prefix)?
        .as_u64()
        .ok_or_else(|| format!("field {prefix}{path} must be a non-negative integer"))
}

fn required_f64(value: &Value, path: &str, prefix: &str) -> Result<f64, String> {
    required_value(value, path, prefix)?
        .as_f64()
        .ok_or_else(|| format!("field {prefix}{path} must be a number"))
}

fn required_array<'a>(value: &'a Value, path: &str) -> Result<&'a Vec<Value>, String> {
    value
        .pointer(path)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("field {path} must be an array"))
}

fn required_object<'a>(
    value: &'a Value,
    path: &str,
    prefix: &str,
) -> Result<&'a Map<String, Value>, String> {
    let selected = if path.is_empty() {
        value
    } else {
        required_value(value, path, prefix)?
    };
    selected
        .as_object()
        .ok_or_else(|| format!("field {prefix}{path} must be an object"))
}

#[cfg(test)]
mod tests {
    use super::{BLAKE3_HEX_LENGTH, find_drift, render_consumer_files};
    use serde_json::{Value, json};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    const VALID_REVISION: &str = "1111111111111111111111111111111111111111";
    const VALID_BLAKE3: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const TEST_DIRECTORY_PREFIX: &str = "model-target-consumer-generator-test";

    fn valid_input() -> Value {
        json!({
            "schema_version": 1,
            "profiles": [{
                "target_name": "rwkv7-goose-world2.8-0.1b-p150x2",
                "runnable": true,
                "model": {
                    "id": "fixture/model",
                    "revision": VALID_REVISION,
                    "license": "fixture",
                    "architecture_family": "rwkv7",
                    "checkpoint_layout": "fixture",
                    "precision": "bf16"
                },
                "state": {
                    "kind": "recurrent_matrix",
                    "per_layer_element_count": 24,
                    "carry_semantics": "fixture"
                },
                "resources": {
                    "checkpoint_bytes": 16,
                    "layer_count": 2,
                    "hidden_size": 8,
                    "vocabulary_size": 32,
                    "maximum_prompt_tokens": 4
                },
                "adapters": {"architecture": "arch-v1", "evidence": "evidence-v1"},
                "topology": {
                    "id": "fixture-topology",
                    "mesh_shape": [1, 1],
                    "physical_device_ids": [0],
                    "replication": "none"
                },
                "consumer": {
                    "dimensions": {
                        "head_count": 2,
                        "head_size": 4,
                        "intermediate_size": 16,
                        "complete_state_element_count": 48
                    },
                    "precisions": {
                        "cpu_reference": "fp32",
                        "runtime_arithmetic": "fp32",
                        "ttwkv7_boundary": "bf16"
                    },
                    "tolerances": {
                        "harness_oracle": 0.001,
                        "harness_replay": 0.001,
                        "state_carry_divergence_floor": 0.001,
                        "final_hidden": 0.1,
                        "logits": 0.1,
                        "recurrent_state": 0.1,
                        "control_divergence_floor": 0.1,
                        "lora_gradient_agreement": 0.01,
                        "lora_finite_difference": 0.02
                    },
                    "adapter_format": "fixture"
                },
                "artifacts": [{
                    "role": "checkpoint",
                    "name": "fixture.bin",
                    "url": "fixture://checkpoint",
                    "nix_hash": "sha256-uWqL3CHhX3HgyVZT3MO+ieVkthmtUHPJ7b+9B/eElFM=",
                    "blake3": VALID_BLAKE3
                }, {
                    "role": "tokenizer_vocabulary",
                    "name": "vocab.txt",
                    "url": "fixture://vocab",
                    "nix_hash": "sha256-5t7j1OMbTVxArJlQisbHAc7vS+1oG/IWfOmpCFUryok=",
                    "blake3": VALID_BLAKE3
                }, {
                    "role": "tokenizer_config",
                    "name": "tokenizer.json",
                    "url": "fixture://tokenizer-config",
                    "nix_hash": "sha256-5t7j1OMbTVxArJlQisbHAc7vS+1oG/IWfOmpCFUryok=",
                    "blake3": VALID_BLAKE3
                }, {
                    "role": "tokenizer_added_tokens",
                    "name": "added.json",
                    "url": "fixture://added",
                    "nix_hash": "sha256-5t7j1OMbTVxArJlQisbHAc7vS+1oG/IWfOmpCFUryok=",
                    "blake3": VALID_BLAKE3
                }, {
                    "role": "tokenizer_implementation",
                    "name": "tokenizer.py",
                    "url": "fixture://tokenizer-implementation",
                    "nix_hash": "sha256-5t7j1OMbTVxArJlQisbHAc7vS+1oG/IWfOmpCFUryok=",
                    "blake3": VALID_BLAKE3
                }, {
                    "role": "tokenizer_special_tokens",
                    "name": "special.json",
                    "url": "fixture://special",
                    "nix_hash": "sha256-5t7j1OMbTVxArJlQisbHAc7vS+1oG/IWfOmpCFUryok=",
                    "blake3": VALID_BLAKE3
                }, {
                    "role": "model_config",
                    "name": "config.json",
                    "url": "fixture://model-config",
                    "nix_hash": "sha256-5t7j1OMbTVxArJlQisbHAc7vS+1oG/IWfOmpCFUryok=",
                    "blake3": VALID_BLAKE3
                }, {
                    "role": "generation_config",
                    "name": "generation.json",
                    "url": "fixture://generation-config",
                    "nix_hash": "sha256-5t7j1OMbTVxArJlQisbHAc7vS+1oG/IWfOmpCFUryok=",
                    "blake3": VALID_BLAKE3
                }]
            }]
        })
    }

    fn render(value: &Value) -> Result<Vec<super::RenderedFile>, String> {
        render_consumer_files(&serde_json::to_vec(value).expect("fixture must serialize"))
    }

    #[test]
    fn valid_profile_renders_all_languages_with_one_identity() {
        let files = render(&valid_input()).expect("valid profile must render");
        assert!(
            files
                .iter()
                .any(|file| file.path.extension().is_some_and(|value| value == "rs"))
        );
        assert!(
            files
                .iter()
                .any(|file| file.path.extension().is_some_and(|value| value == "py"))
        );
        assert!(
            files
                .iter()
                .any(|file| file.path.extension().is_some_and(|value| value == "h"))
        );
        assert!(
            files
                .iter()
                .any(|file| file.path.extension().is_some_and(|value| value == "nix"))
        );
        let rust = String::from_utf8(files[0].bytes.clone()).expect("Rust output must be UTF-8");
        let marker = "pub const PROFILE_BLAKE3: &str = \"";
        let start = rust.find(marker).expect("profile identity must exist") + marker.len();
        let identity = &rust[start..start + BLAKE3_HEX_LENGTH];
        assert!(
            files
                .iter()
                .all(|file| String::from_utf8_lossy(&file.bytes).contains(identity))
        );
    }

    #[test]
    fn invalid_hash_text_is_rejected_without_normalization() {
        for invalid in [
            "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaAaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "gaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ] {
            let mut value = valid_input();
            value["profiles"][0]["artifacts"][0]["blake3"] = json!(invalid);
            let error = render(&value).expect_err("invalid hash must fail");
            assert!(error.contains("/artifacts/0/blake3"));
        }
    }

    #[test]
    fn missing_field_and_inconsistent_dimension_are_rejected() {
        let mut missing = valid_input();
        missing["profiles"][0]
            .as_object_mut()
            .expect("profile must be an object")
            .remove("consumer");
        assert!(
            render(&missing)
                .expect_err("missing consumer must fail")
                .contains("/consumer")
        );

        let mut inconsistent = valid_input();
        inconsistent["profiles"][0]["consumer"]["dimensions"]["head_count"] = json!(3);
        assert!(
            render(&inconsistent)
                .expect_err("wrong dimension must fail")
                .contains("head_count * head_size")
        );
    }

    #[test]
    fn changed_tolerance_and_topology_change_generated_bytes() {
        let original = render(&valid_input()).expect("valid profile must render");
        let mut changed = valid_input();
        changed["profiles"][0]["consumer"]["tolerances"]["logits"] = json!(0.2);
        changed["profiles"][0]["topology"]["id"] = json!("other-topology");
        let changed_files = render(&changed).expect("changed valid profile must render");
        assert_ne!(original, changed_files);
    }

    #[test]
    fn stale_and_missing_outputs_are_reported() {
        let files = render(&valid_input()).expect("valid profile must render");
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock must be after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!("{TEST_DIRECTORY_PREFIX}-{unique}"));
        fs::create_dir_all(&root).expect("temporary directory must be created");
        for file in &files {
            let path = root.join(&file.path);
            fs::create_dir_all(path.parent().expect("output must have a parent"))
                .expect("parent directory must be created");
            fs::write(path, &file.bytes).expect("fixture output must be written");
        }
        assert!(
            find_drift(&files, &root)
                .expect("freshness check must run")
                .is_empty()
        );
        fs::write(root.join(&files[0].path), b"stale\n").expect("stale fixture must be written");
        fs::remove_file(root.join(&files[1].path)).expect("fixture must be removed");
        let drift = find_drift(&files, &root).expect("drift check must run");
        assert_eq!(drift, vec![files[0].path.clone(), files[1].path.clone()]);
        fs::remove_dir_all(root).expect("temporary directory must be removed");
    }
}
