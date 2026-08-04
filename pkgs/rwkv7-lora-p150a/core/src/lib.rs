//! Pure admission core for the single-P150A RWKV-7 LoRA bring-up.
//!
//! r[impl tenstorrent.rwkv7_p150a_finetuning.authority]
//! r[impl tenstorrent.rwkv7_p150a_finetuning.configuration]
//!
//! Every function is deterministic and free of I/O, clocks, and process
//! state. The CLI shell in `main.rs` owns all filesystem and process work.
//!
//! Later phases extend this crate with probe, placement, parity, optimizer,
//! run, and evidence admission. Each phase keeps the same core/shell split.

use serde_json::{Value, json};

const SCHEMA_VERSION: u64 = 1;
const HEX_HASH_LEN: usize = 64;

// Admitted runtime bundle from the training platform manifest.
const REQUIRED_BUNDLE_ID: &str = "tt-xla-training-2026-08-04";

// CPU authority identities from pkgs/rwkv7-lora-reference.
const REQUIRED_MODEL_ID: &str = "RWKV/RWKV7-Goose-World2.8-0.1B-HF";
const REQUIRED_MODEL_REVISION: &str = "d81965cb4e1a9f96696b4f70b84212b8f2e43216";
const REQUIRED_CHECKPOINT_BLAKE3: &str =
    "905f82048a64b881f9267117a398feb8a8a92bcc5233666bf67904e0d899d0e5";
const REQUIRED_TOKENIZER_VOCAB_BLAKE3: &str =
    "3997a74891dd68ced8daadae0d7475274b08988c9263ca042896c8106967aef2";
const REQUIRED_CPU_PRECISION: &str = "fp32_from_bf16";
const REQUIRED_LAYER_COUNT: u64 = 12;
const REQUIRED_HIDDEN_SIZE: u64 = 768;
const REQUIRED_HEAD_COUNT: u64 = 12;
const REQUIRED_HEAD_SIZE: u64 = 64;
const REQUIRED_VOCABULARY_SIZE: u64 = 65536;
const REQUIRED_VOCABULARY_ENTRY_COUNT: u64 = 65529;
const CPU_REFERENCE_OUTCOME: &str = "bounded_cpu_reference_pass";

// Device facts for one logical single-device profile over the dual-card
// platform. Container device isolation is not supported upstream.
const REQUIRED_PHYSICAL_COUNT: u64 = 2;
const REQUIRED_ARCHITECTURE: &str = "blackhole";
const REQUIRED_BOARD_CLASS: &str = "p150a";
const REQUIRED_DEVICE_ROOT: &str = "/dev/tenstorrent";
const REQUIRED_LOGICAL_ROWS: u64 = 1;
const REQUIRED_LOGICAL_COLUMNS: u64 = 1;
const REQUIRED_ISOLATION_STATE: &str = "unsupported";

const REQUIRED_PRECISION_CLASS: &str = "bf16_from_fp32_cpu";
const ADMITTED_DTYPES: [&str; 2] = ["bf16", "fp32"];
const REQUIRED_ADAPTER_FORMAT: &str = "safetensors-adapter-only";
const REQUIRED_OPTIMIZER_KIND: &str = "adamw";
const REQUIRED_HASH_ALGORITHM: &str = "blake3";

const ADMITTED_LORA_ROLES: [&str; 6] = [
    "attention.key",
    "attention.output",
    "attention.receptance",
    "attention.value",
    "feed_forward.key",
    "feed_forward.value",
];

const REQUIRED_TOLERANCE_ROLES: [&str; 8] = [
    "adapter_delta",
    "adapter_gradient",
    "clip_coefficient",
    "gradient_norm",
    "logits",
    "loss",
    "wkv7_output",
    "wkv7_state",
];

const REQUIRED_RECEIPT_KINDS: [&str; 9] = [
    "artifact",
    "compile_probe",
    "health",
    "optimizer",
    "parity",
    "placement",
    "prerequisite",
    "restoration",
    "run",
];

const BOUNDARY_NON_CLAIMS: [&str; 4] = [
    "no dual-card scaling claim",
    "no speedup claim",
    "no support claim for other RWKV models",
    "no useful generalization claim",
];

const PLATFORM_RECEIPT_KIND: &str = "dual_p150a_platform_smoke";
const PLATFORM_PASS_STATUS: &str = "pass";
const PLATFORM_ADMITTED_STATUS: &str = "admitted";

const PLATFORM_RECEIPT_STAGES: [&str; 6] = [
    "health_before",
    "mesh_open",
    "smoke_execute",
    "mesh_close",
    "restore",
    "health_after",
];

const FORBIDDEN_PROFILE_KEYS: [&str; 6] = [
    "password",
    "secret",
    "token",
    "credential",
    "serial_number",
    "raw_log",
];

/// One stable admission failure. The `code` is machine-stable; `detail`
/// names the offending component and value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rejection {
    pub code: &'static str,
    pub detail: String,
}

impl Rejection {
    pub fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Rejection {
            code,
            detail: detail.into(),
        }
    }
}

impl std::fmt::Display for Rejection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}

/// Profile admitted for probe compilation and device work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedProfile {
    pub identity_blake3: String,
    pub profile_id: String,
    pub sequence_length: u64,
    pub micro_batch: u64,
    pub probe_compile_timeout_seconds: u64,
    pub full_compile_timeout_seconds: u64,
    pub max_stablehlo_operations: u64,
    pub hardware_attempt_seconds: u64,
}

/// Deterministic prerequisite admission binding platform and CPU receipts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrerequisiteAdmission {
    pub prerequisite_identity: String,
    pub profile_identity: String,
    pub platform_bundle_identity: String,
    pub platform_config_identity: String,
    pub platform_plan_identity: String,
    pub cpu_authority_blake3: String,
    pub cpu_receipt_blake3: String,
}

/// Deterministic BLAKE3 hex digest over canonical JSON.
pub fn blake3_identity(value: &Value) -> String {
    blake3::hash(&serde_json::to_vec(value).expect("canonical JSON serialization")).to_hex()
        [..HEX_HASH_LEN]
        .to_string()
}

fn is_hash_hex(value: &str) -> bool {
    value.len() == HEX_HASH_LEN && value.chars().all(|c| c.is_ascii_hexdigit())
}

fn field<'a>(value: &'a Value, key: &str) -> Result<&'a Value, Rejection> {
    value
        .get(key)
        .ok_or_else(|| Rejection::new("missing_field", format!("missing field '{key}'")))
}

fn string_field<'a>(value: &'a Value, key: &str) -> Result<&'a str, Rejection> {
    field(value, key)?
        .as_str()
        .ok_or_else(|| Rejection::new("malformed_field", format!("field '{key}' is not a string")))
}

fn positive_u64(value: &Value, key: &str) -> Result<u64, Rejection> {
    let number = field(value, key)?
        .as_u64()
        .ok_or_else(|| Rejection::new("dynamic_shape", format!("field '{key}' is not a positive integer")))?;
    if number == 0 {
        return Err(Rejection::new(
            "dynamic_shape",
            format!("field '{key}' must be positive"),
        ));
    }
    Ok(number)
}

fn nonnegative_u64(value: &Value, key: &str) -> Result<u64, Rejection> {
    field(value, key)?
        .as_u64()
        .ok_or_else(|| Rejection::new("malformed_field", format!("field '{key}' is not a non-negative integer")))
}

fn positive_number(value: &Value, key: &str) -> Result<f64, Rejection> {
    let number = field(value, key)?
        .as_f64()
        .ok_or_else(|| Rejection::new("bad_limit", format!("field '{key}' is not a number")))?;
    if !(number > 0.0) {
        return Err(Rejection::new(
            "bad_limit",
            format!("field '{key}' must be positive"),
        ));
    }
    Ok(number)
}

fn bool_field(value: &Value, key: &str) -> Result<bool, Rejection> {
    field(value, key)?
        .as_bool()
        .ok_or_else(|| Rejection::new("malformed_field", format!("field '{key}' is not a boolean")))
}

fn require_equal_str(
    value: &Value,
    key: &str,
    expected: &str,
    code: &'static str,
) -> Result<(), Rejection> {
    let observed = string_field(value, key)?;
    if observed != expected {
        return Err(Rejection::new(
            code,
            format!("field '{key}' is '{observed}', expected '{expected}'"),
        ));
    }
    Ok(())
}

fn require_equal_u64(
    value: &Value,
    key: &str,
    expected: u64,
    code: &'static str,
) -> Result<(), Rejection> {
    let observed = field(value, key)?
        .as_u64()
        .ok_or_else(|| Rejection::new(code, format!("field '{key}' is not an integer")))?;
    if observed != expected {
        return Err(Rejection::new(
            code,
            format!("field '{key}' is {observed}, expected {expected}"),
        ));
    }
    Ok(())
}

fn string_array<'a>(value: &'a Value, key: &str) -> Result<Vec<&'a str>, Rejection> {
    let array = field(value, key)?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_field", format!("field '{key}' is not an array")))?;
    array
        .iter()
        .map(|item| {
            item.as_str().ok_or_else(|| {
                Rejection::new("malformed_field", format!("field '{key}' has a non-string entry"))
            })
        })
        .collect()
}

fn contains_all(entries: &[&str], required: &[&str], key: &str) -> Result<(), Rejection> {
    for needed in required {
        if !entries.contains(needed) {
            return Err(Rejection::new(
                "bad_evidence_policy",
                format!("field '{key}' is missing required entry '{needed}'"),
            ));
        }
    }
    Ok(())
}

/// Boundary-aware forbidden-word match. `token` matches `oauth_token` but
/// not `tokenizer`, so admitted vocabulary cannot collide with the safety
/// list.
fn key_hits_forbidden(key: &str, forbidden: &str) -> bool {
    if key == forbidden {
        return true;
    }
    let separators = ['_', '-', '.'];
    for (left, right) in key.match_indices(forbidden) {
        let before_ok = left == 0
            || key[..left]
                .chars()
                .next_back()
                .is_some_and(|c| separators.contains(&c));
        let end = left + right.len();
        let after_ok = end == key.len()
            || key[end..]
                .chars()
                .next()
                .is_some_and(|c| separators.contains(&c));
        if before_ok && after_ok {
            return true;
        }
    }
    false
}

fn reject_forbidden_keys(value: &Value) -> Result<(), Rejection> {
    match value {
        Value::Object(map) => {
            for (key, nested) in map {
                if FORBIDDEN_PROFILE_KEYS
                    .iter()
                    .any(|forbidden| key_hits_forbidden(key, forbidden))
                {
                    return Err(Rejection::new(
                        "unsafe_field",
                        format!("profile contains forbidden field '{key}'"),
                    ));
                }
                reject_forbidden_keys(nested)?;
            }
            Ok(())
        }
        Value::Array(items) => {
            for item in items {
                reject_forbidden_keys(item)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn check_model_authority(model: &Value) -> Result<(), Rejection> {
    require_equal_str(model, "model_id", REQUIRED_MODEL_ID, "model_mismatch")?;
    require_equal_str(model, "revision", REQUIRED_MODEL_REVISION, "model_mismatch")?;
    require_equal_str(
        model,
        "checkpoint_blake3",
        REQUIRED_CHECKPOINT_BLAKE3,
        "model_mismatch",
    )?;
    require_equal_u64(model, "layer_count", REQUIRED_LAYER_COUNT, "model_mismatch")?;
    require_equal_u64(model, "hidden_size", REQUIRED_HIDDEN_SIZE, "model_mismatch")?;
    require_equal_u64(model, "head_count", REQUIRED_HEAD_COUNT, "model_mismatch")?;
    require_equal_u64(model, "head_size", REQUIRED_HEAD_SIZE, "model_mismatch")?;
    require_equal_u64(
        model,
        "vocabulary_size",
        REQUIRED_VOCABULARY_SIZE,
        "model_mismatch",
    )?;
    require_equal_str(model, "precision", REQUIRED_CPU_PRECISION, "model_mismatch")?;
    Ok(())
}

fn check_tokenizer(tokenizer: &Value) -> Result<(), Rejection> {
    require_equal_str(
        tokenizer,
        "vocabulary_blake3",
        REQUIRED_TOKENIZER_VOCAB_BLAKE3,
        "tokenizer_mismatch",
    )?;
    require_equal_u64(
        tokenizer,
        "vocabulary_entry_count",
        REQUIRED_VOCABULARY_ENTRY_COUNT,
        "tokenizer_mismatch",
    )?;
    Ok(())
}

fn check_device(device: &Value) -> Result<u64, Rejection> {
    require_equal_u64(device, "physical_count", REQUIRED_PHYSICAL_COUNT, "device_mismatch")?;
    require_equal_str(device, "architecture", REQUIRED_ARCHITECTURE, "device_mismatch")?;
    require_equal_str(device, "board_class", REQUIRED_BOARD_CLASS, "device_mismatch")?;
    require_equal_str(device, "device_root", REQUIRED_DEVICE_ROOT, "device_mismatch")?;
    require_equal_str(
        device,
        "container_device_isolation",
        REQUIRED_ISOLATION_STATE,
        "device_mismatch",
    )?;
    let logical = field(device, "logical_mesh")?;
    require_equal_u64(logical, "rows", REQUIRED_LOGICAL_ROWS, "device_mismatch")?;
    require_equal_u64(logical, "columns", REQUIRED_LOGICAL_COLUMNS, "device_mismatch")?;
    let ordinal = nonnegative_u64(device, "selection_ordinal")?;
    if ordinal >= REQUIRED_PHYSICAL_COUNT {
        return Err(Rejection::new(
            "device_mismatch",
            format!("selection_ordinal {ordinal} is outside the physical device set"),
        ));
    }
    Ok(ordinal)
}

fn check_probe(probe: &Value, static_shape: &Value, model: &Value) -> Result<(), Rejection> {
    let focused = field(probe, "focused")?;
    for key in ["batch", "time", "heads", "width"] {
        positive_u64(focused, key)?;
    }
    let representative = field(probe, "representative")?;
    // The representative probe must exercise the admitted full static shape
    // so a focused-only pass cannot hide a full-shape failure.
    require_equal_u64(
        representative,
        "time",
        positive_u64(static_shape, "sequence_length")?,
        "bad_probe_shape",
    )?;
    require_equal_u64(
        representative,
        "batch",
        positive_u64(static_shape, "micro_batch")?,
        "bad_probe_shape",
    )?;
    require_equal_u64(
        representative,
        "heads",
        field(model, "head_count")?.as_u64().unwrap_or_default(),
        "bad_probe_shape",
    )?;
    require_equal_u64(
        representative,
        "width",
        field(model, "head_size")?.as_u64().unwrap_or_default(),
        "bad_probe_shape",
    )?;
    Ok(())
}

fn check_precision(precision: &Value) -> Result<(), Rejection> {
    require_equal_str(
        precision,
        "precision_class",
        REQUIRED_PRECISION_CLASS,
        "unknown_precision",
    )?;
    for key in ["model_dtype", "optimizer_dtype", "comparison_dtype"] {
        let observed = string_field(precision, key)?;
        if !ADMITTED_DTYPES.contains(&observed) {
            return Err(Rejection::new(
                "unknown_precision",
                format!("dtype '{observed}' for '{key}' is not admitted"),
            ));
        }
    }
    Ok(())
}

fn check_tolerances(tolerances: &Value) -> Result<(), Rejection> {
    for role in REQUIRED_TOLERANCE_ROLES {
        let entry = tolerances
            .get(role)
            .ok_or_else(|| Rejection::new("missing_tolerance", format!("tolerance role '{role}' is missing")))?;
        positive_number(entry, "rtol").map_err(|_| {
            Rejection::new("missing_tolerance", format!("tolerance role '{role}' has a bad rtol"))
        })?;
        positive_number(entry, "atol").map_err(|_| {
            Rejection::new("missing_tolerance", format!("tolerance role '{role}' has a bad atol"))
        })?;
    }
    Ok(())
}

fn check_run_criteria(criteria: &Value) -> Result<(), Rejection> {
    let min_steps = positive_u64(criteria, "min_steps")?;
    let max_steps = positive_u64(criteria, "max_steps")?;
    if min_steps > max_steps {
        return Err(Rejection::new(
            "bad_run_criteria",
            format!("min_steps {min_steps} exceeds max_steps {max_steps}"),
        ));
    }
    let ratio = positive_number(criteria, "loss_improvement_ratio")?;
    if ratio >= 1.0 {
        return Err(Rejection::new(
            "bad_run_criteria",
            "loss_improvement_ratio must be below one to require improvement",
        ));
    }
    positive_u64(criteria, "min_active_labels_per_sample")?;
    bool_field(criteria, "require_finite")?;
    Ok(())
}

fn check_lora(lora: &Value) -> Result<(), Rejection> {
    positive_u64(lora, "rank")?;
    positive_number(lora, "scale")?;
    let mut roles = string_array(lora, "target_roles")?;
    if roles.is_empty() {
        return Err(Rejection::new("bad_lora_plan", "target_roles is empty"));
    }
    roles.sort_unstable();
    let admitted: Vec<&str> = ADMITTED_LORA_ROLES.iter().copied().collect();
    if roles != admitted {
        return Err(Rejection::new(
            "bad_lora_plan",
            format!("target_roles {roles:?} must equal the CPU authority allowlist"),
        ));
    }
    Ok(())
}

fn check_optimizer(optimizer: &Value, static_shape: &Value) -> Result<(), Rejection> {
    require_equal_str(optimizer, "kind", REQUIRED_OPTIMIZER_KIND, "optimizer_mismatch")?;
    positive_number(optimizer, "learning_rate")?;
    positive_number(optimizer, "clip_norm")?;
    require_equal_u64(
        optimizer,
        "gradient_accumulation_steps",
        positive_u64(static_shape, "gradient_accumulation_steps")?,
        "optimizer_mismatch",
    )?;
    Ok(())
}

fn check_artifact_policy(policy: &Value) -> Result<(), Rejection> {
    require_equal_str(policy, "format", REQUIRED_ADAPTER_FORMAT, "bad_artifact_policy")?;
    let root = string_field(policy, "artifact_root")?;
    if !root.starts_with('/') || root.contains("..") {
        return Err(Rejection::new(
            "bad_artifact_policy",
            format!("artifact_root '{root}' is not an absolute clean path"),
        ));
    }
    Ok(())
}

fn check_evidence_policy(policy: &Value) -> Result<(), Rejection> {
    require_equal_str(policy, "hash_algorithm", REQUIRED_HASH_ALGORITHM, "bad_evidence_policy")?;
    let exclude = string_array(policy, "exclude")?;
    for entry in &exclude {
        let known = ["credentials", "raw_logs", "serial_numbers", "absolute_private_paths"];
        if !known.contains(entry) {
            return Err(Rejection::new(
                "bad_evidence_policy",
                format!("unknown exclusion '{entry}'"),
            ));
        }
    }
    let receipts = string_array(policy, "required_receipts")?;
    contains_all(&receipts, &REQUIRED_RECEIPT_KINDS, "required_receipts")?;
    let non_claims = string_array(policy, "non_claims")?;
    contains_all(&non_claims, &BOUNDARY_NON_CLAIMS, "non_claims")?;
    Ok(())
}

/// Admit one exported Nickel profile. Re-checks every Nickel contract in
/// Rust so the launcher never trusts configuration text alone.
pub fn admit_profile(profile: &Value) -> Result<AdmittedProfile, Rejection> {
    reject_forbidden_keys(profile)?;
    require_equal_u64(profile, "schema_version", SCHEMA_VERSION, "bad_schema_version")?;
    let profile_id = string_field(profile, "profile_id")?.to_string();
    if profile_id.is_empty() {
        return Err(Rejection::new("bad_profile_id", "profile_id is empty"));
    }
    require_equal_str(profile, "bundle_id", REQUIRED_BUNDLE_ID, "bad_bundle_id")?;

    let model = field(profile, "model")?;
    check_model_authority(model)?;
    check_tokenizer(field(profile, "tokenizer")?)?;
    check_device(field(profile, "device")?)?;

    let static_shape = field(profile, "static_shape")?;
    let micro_batch = positive_u64(static_shape, "micro_batch")?;
    let sequence_length = positive_u64(static_shape, "sequence_length")?;
    positive_u64(static_shape, "gradient_accumulation_steps")?;

    check_probe(field(profile, "probe")?, static_shape, model)?;
    check_precision(field(profile, "precision")?)?;

    let compile_limits = field(profile, "compile_limits")?;
    let probe_compile_timeout_seconds = positive_u64(compile_limits, "probe_compile_timeout_seconds")?;
    let full_compile_timeout_seconds = positive_u64(compile_limits, "full_compile_timeout_seconds")?;
    let max_stablehlo_operations = positive_u64(compile_limits, "max_stablehlo_operations")?;

    let memory_limits = field(profile, "memory_limits")?;
    positive_number(memory_limits, "max_device_bytes")?;
    positive_number(memory_limits, "max_host_bytes")?;

    check_tolerances(field(profile, "tolerances")?)?;
    check_run_criteria(field(profile, "run_criteria")?)?;

    let health_limits = field(profile, "health_limits")?;
    positive_number(health_limits, "max_board_temperature_c")?;
    positive_number(health_limits, "max_power_watts")?;
    bool_field(health_limits, "require_dram_ok")?;

    let timeouts = field(profile, "timeouts")?;
    let hardware_attempt_seconds = positive_u64(timeouts, "hardware_attempt_seconds")?;
    positive_u64(timeouts, "multi_step_run_seconds")?;

    check_lora(field(profile, "lora")?)?;
    check_optimizer(field(profile, "optimizer")?, static_shape)?;
    nonnegative_u64(profile, "seed")?;
    check_artifact_policy(field(profile, "artifact_policy")?)?;
    check_evidence_policy(field(profile, "evidence_policy")?)?;

    Ok(AdmittedProfile {
        identity_blake3: blake3_identity(profile),
        profile_id,
        sequence_length,
        micro_batch,
        probe_compile_timeout_seconds,
        full_compile_timeout_seconds,
        max_stablehlo_operations,
        hardware_attempt_seconds,
    })
}

/// Rebuild the CPU authority identity from profile facts. The byte form must
/// match `lora_core.authority_identity` in the CPU reference package:
/// sorted keys, compact separators, integer fields as integers.
pub fn cpu_authority_identity(profile: &Value) -> Result<String, Rejection> {
    let model = field(profile, "model")?;
    let tokenizer = field(profile, "tokenizer")?;
    let fields = json!({
        "checkpoint_blake3": string_field(model, "checkpoint_blake3")?,
        "head_count": field(model, "head_count")?,
        "head_size": field(model, "head_size")?,
        "hidden_size": field(model, "hidden_size")?,
        "layer_count": field(model, "layer_count")?,
        "model_id": string_field(model, "model_id")?,
        "precision": string_field(model, "precision")?,
        "revision": string_field(model, "revision")?,
        "tokenizer_vocab_blake3": string_field(tokenizer, "vocabulary_blake3")?,
    });
    Ok(blake3_identity(&fields))
}

fn platform_stage_status(receipt: &Value, stage: &str) -> Result<String, Rejection> {
    let stages = field(receipt, "stages")?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_receipt", "platform receipt stages is not an array"))?;
    for entry in stages {
        if entry.get("name").and_then(Value::as_str) == Some(stage) {
            return Ok(entry
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string());
        }
    }
    Err(Rejection::new(
        "missing_stage",
        format!("platform receipt is missing stage '{stage}'"),
    ))
}

/// Admit prerequisite receipts before any accelerator work.
///
/// The platform receipt must be admitted by the platform core and carry all
/// passing stages. The CPU receipt must carry the bounded pass outcome and
/// an authority identity that matches the profile exactly.
pub fn admit_prerequisites(
    profile: &Value,
    platform_admission: &Value,
    platform_receipt: &Value,
    cpu_receipt: &Value,
    cpu_receipt_blake3: &str,
) -> Result<PrerequisiteAdmission, Rejection> {
    let admitted = admit_profile(profile)?;

    let admission_status = string_field(platform_admission, "status")?;
    if admission_status != PLATFORM_ADMITTED_STATUS {
        return Err(Rejection::new(
            "stale_prerequisite",
            format!("platform receipt admission status is '{admission_status}'"),
        ));
    }

    require_equal_str(platform_receipt, "kind", PLATFORM_RECEIPT_KIND, "stale_prerequisite")?;
    for stage in PLATFORM_RECEIPT_STAGES {
        let status = platform_stage_status(platform_receipt, stage)?;
        if status != PLATFORM_PASS_STATUS {
            return Err(Rejection::new(
                "failing_stage",
                format!("platform receipt stage '{stage}' has status '{status}'"),
            ));
        }
    }
    let platform_bundle_identity = string_field(platform_receipt, "bundle_identity")?.to_string();
    let platform_config_identity = string_field(platform_receipt, "config_identity")?.to_string();
    let platform_plan_identity =
        string_field(platform_receipt, "launch_plan_identity")?.to_string();
    for (name, identity) in [
        ("bundle_identity", &platform_bundle_identity),
        ("config_identity", &platform_config_identity),
        ("launch_plan_identity", &platform_plan_identity),
    ] {
        if !is_hash_hex(identity) {
            return Err(Rejection::new(
                "malformed_receipt",
                format!("platform receipt {name} is not a hex hash"),
            ));
        }
    }

    let outcome = string_field(cpu_receipt, "outcome")?;
    if outcome != CPU_REFERENCE_OUTCOME {
        return Err(Rejection::new(
            "outcome_mismatch",
            format!("CPU reference outcome is '{outcome}'"),
        ));
    }

    let expected_authority = cpu_authority_identity(profile)?;
    let observed_authority = string_field(field(cpu_receipt, "model")?, "authority_blake3")?;
    if observed_authority != expected_authority {
        return Err(Rejection::new(
            "authority_mismatch",
            format!(
                "CPU authority identity {observed_authority} differs from profile authority {expected_authority}"
            ),
        ));
    }

    let observed_checkpoint =
        string_field(field(cpu_receipt, "model_load")?, "checkpoint_blake3")?;
    if observed_checkpoint != REQUIRED_CHECKPOINT_BLAKE3 {
        return Err(Rejection::new(
            "checkpoint_mismatch",
            format!("CPU receipt checkpoint identity {observed_checkpoint} differs"),
        ));
    }
    if !is_hash_hex(cpu_receipt_blake3) {
        return Err(Rejection::new(
            "malformed_receipt",
            "CPU receipt BLAKE3 is not a hex hash",
        ));
    }

    let binding = json!({
        "cpu_authority_blake3": expected_authority,
        "cpu_receipt_blake3": cpu_receipt_blake3,
        "platform_bundle_identity": platform_bundle_identity,
        "platform_config_identity": platform_config_identity,
        "platform_plan_identity": platform_plan_identity,
        "profile_identity": admitted.identity_blake3,
        "schema_version": SCHEMA_VERSION,
    });

    Ok(PrerequisiteAdmission {
        prerequisite_identity: blake3_identity(&binding),
        profile_identity: admitted.identity_blake3,
        platform_bundle_identity,
        platform_config_identity,
        platform_plan_identity,
        cpu_authority_blake3: expected_authority,
        cpu_receipt_blake3: cpu_receipt_blake3.to_string(),
    })
}

fn minimal_valid_profile() -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "profile_id": "fixture-profile",
        "bundle_id": REQUIRED_BUNDLE_ID,
        "model": {
            "model_id": REQUIRED_MODEL_ID,
            "revision": REQUIRED_MODEL_REVISION,
            "checkpoint_blake3": REQUIRED_CHECKPOINT_BLAKE3,
            "layer_count": REQUIRED_LAYER_COUNT,
            "hidden_size": REQUIRED_HIDDEN_SIZE,
            "head_count": REQUIRED_HEAD_COUNT,
            "head_size": REQUIRED_HEAD_SIZE,
            "vocabulary_size": REQUIRED_VOCABULARY_SIZE,
            "precision": REQUIRED_CPU_PRECISION,
        },
        "tokenizer": {
            "vocabulary_blake3": REQUIRED_TOKENIZER_VOCAB_BLAKE3,
            "vocabulary_entry_count": REQUIRED_VOCABULARY_ENTRY_COUNT,
        },
        "device": {
            "physical_count": REQUIRED_PHYSICAL_COUNT,
            "architecture": REQUIRED_ARCHITECTURE,
            "board_class": REQUIRED_BOARD_CLASS,
            "device_root": REQUIRED_DEVICE_ROOT,
            "logical_mesh": { "rows": REQUIRED_LOGICAL_ROWS, "columns": REQUIRED_LOGICAL_COLUMNS },
            "selection_ordinal": 0,
            "container_device_isolation": REQUIRED_ISOLATION_STATE,
        },
        "static_shape": {
            "micro_batch": 1,
            "sequence_length": 64,
            "gradient_accumulation_steps": 1,
        },
        "probe": {
            "focused": { "batch": 1, "time": 4, "heads": 1, "width": 16 },
            "representative": { "batch": 1, "time": 64, "heads": 12, "width": 64 },
        },
        "precision": {
            "precision_class": REQUIRED_PRECISION_CLASS,
            "model_dtype": "bf16",
            "optimizer_dtype": "bf16",
            "comparison_dtype": "fp32",
        },
        "compile_limits": {
            "probe_compile_timeout_seconds": 1200,
            "full_compile_timeout_seconds": 3600,
            "max_stablehlo_operations": 200000,
        },
        "memory_limits": { "max_device_bytes": 6000000000_u64, "max_host_bytes": 8000000000_u64 },
        "tolerances": {
            "adapter_delta": { "rtol": 0.05, "atol": 0.001 },
            "adapter_gradient": { "rtol": 0.05, "atol": 0.2 },
            "clip_coefficient": { "rtol": 0.02, "atol": 0.01 },
            "gradient_norm": { "rtol": 0.02, "atol": 0.01 },
            "logits": { "rtol": 0.02, "atol": 0.1 },
            "loss": { "rtol": 0.01, "atol": 0.05 },
            "wkv7_output": { "rtol": 0.02, "atol": 0.1 },
            "wkv7_state": { "rtol": 0.02, "atol": 0.1 },
        },
        "run_criteria": {
            "min_steps": 8,
            "max_steps": 32,
            "loss_improvement_ratio": 0.9,
            "min_active_labels_per_sample": 1,
            "require_finite": true,
        },
        "health_limits": {
            "max_board_temperature_c": 90,
            "max_power_watts": 180,
            "require_dram_ok": true,
        },
        "timeouts": { "hardware_attempt_seconds": 3600, "multi_step_run_seconds": 3600 },
        "lora": {
            "rank": 2,
            "scale": 1.0,
            "target_roles": ADMITTED_LORA_ROLES,
        },
        "optimizer": {
            "kind": REQUIRED_OPTIMIZER_KIND,
            "learning_rate": 0.01,
            "gradient_accumulation_steps": 1,
            "clip_norm": 1.0,
        },
        "seed": 0,
        "artifact_policy": {
            "format": REQUIRED_ADAPTER_FORMAT,
            "artifact_root": "/var/lib/tenstorrent-training/artifacts/rwkv7-lora-p150a",
        },
        "evidence_policy": {
            "hash_algorithm": REQUIRED_HASH_ALGORITHM,
            "exclude": ["credentials", "raw_logs", "serial_numbers", "absolute_private_paths"],
            "required_receipts": REQUIRED_RECEIPT_KINDS,
            "non_claims": BOUNDARY_NON_CLAIMS,
        },
    })
}

fn minimal_platform_receipt() -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "kind": PLATFORM_RECEIPT_KIND,
        "bundle_identity": "a".repeat(HEX_HASH_LEN),
        "config_identity": "b".repeat(HEX_HASH_LEN),
        "launch_plan_identity": "c".repeat(HEX_HASH_LEN),
        "stages": PLATFORM_RECEIPT_STAGES
            .iter()
            .map(|stage| json!({ "name": stage, "status": PLATFORM_PASS_STATUS }))
            .collect::<Vec<_>>(),
        "artifacts": {},
        "non_claims": [],
    })
}

fn minimal_cpu_receipt() -> Value {
    let profile = minimal_valid_profile();
    let authority = cpu_authority_identity(&profile).expect("fixture authority identity");
    json!({
        "schema_version": SCHEMA_VERSION,
        "outcome": CPU_REFERENCE_OUTCOME,
        "model": { "authority_blake3": authority },
        "model_load": { "checkpoint_blake3": REQUIRED_CHECKPOINT_BLAKE3 },
    })
}

/// Built-in positive and negative admission fixtures.
pub fn self_test() -> Result<String, Rejection> {
    // Positive: profile admission is deterministic.
    let profile = minimal_valid_profile();
    let first = admit_profile(&profile)?;
    let second = admit_profile(&profile)?;
    if first != second {
        return Err(Rejection::new("self_test", "profile admission is not deterministic"));
    }

    // Negative: zero sequence length fails as a dynamic shape.
    let mut bad_shape = profile.clone();
    bad_shape["static_shape"]["sequence_length"] = json!(0);
    match admit_profile(&bad_shape) {
        Err(rejection) if rejection.code == "dynamic_shape" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("zero sequence length produced {other:?}"),
            ));
        }
    }

    // Negative: a forbidden credential field fails, while admitted
    // vocabulary such as `tokenizer` stays legal.
    let mut unsafe_profile = profile.clone();
    unsafe_profile["oauth_token"] = json!("deadbeef");
    match admit_profile(&unsafe_profile) {
        Err(rejection) if rejection.code == "unsafe_field" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("forbidden field produced {other:?}"),
            ));
        }
    }

    // Negative: unknown precision class fails.
    let mut bad_precision = profile.clone();
    bad_precision["precision"]["precision_class"] = json!("fp8_mixed");
    match admit_profile(&bad_precision) {
        Err(rejection) if rejection.code == "unknown_precision" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("unknown precision produced {other:?}"),
            ));
        }
    }

    // Negative: missing tolerance role fails.
    let mut bad_tolerances = profile.clone();
    if let Some(map) = bad_tolerances["tolerances"].as_object_mut() {
        map.remove("adapter_delta");
    }
    match admit_profile(&bad_tolerances) {
        Err(rejection) if rejection.code == "missing_tolerance" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("missing tolerance produced {other:?}"),
            ));
        }
    }

    // Negative: a non-representative probe shape fails.
    let mut bad_probe = profile.clone();
    bad_probe["probe"]["representative"]["time"] = json!(4);
    match admit_profile(&bad_probe) {
        Err(rejection) if rejection.code == "bad_probe_shape" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("non-representative probe produced {other:?}"),
            ));
        }
    }

    // Positive: prerequisite admission binds platform and CPU receipts.
    let platform_receipt = minimal_platform_receipt();
    let platform_admission = json!({ "status": PLATFORM_ADMITTED_STATUS });
    let cpu_receipt = minimal_cpu_receipt();
    let cpu_receipt_blake3 = "d".repeat(HEX_HASH_LEN);
    let first_admission = admit_prerequisites(
        &profile,
        &platform_admission,
        &platform_receipt,
        &cpu_receipt,
        &cpu_receipt_blake3,
    )?;
    let second_admission = admit_prerequisites(
        &profile,
        &platform_admission,
        &platform_receipt,
        &cpu_receipt,
        &cpu_receipt_blake3,
    )?;
    if first_admission != second_admission {
        return Err(Rejection::new(
            "self_test",
            "prerequisite admission is not deterministic",
        ));
    }

    // Negative: stale platform admission blocks.
    let stale_admission = json!({ "status": "rejected" });
    match admit_prerequisites(
        &profile,
        &stale_admission,
        &platform_receipt,
        &cpu_receipt,
        &cpu_receipt_blake3,
    ) {
        Err(rejection) if rejection.code == "stale_prerequisite" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("stale platform admission produced {other:?}"),
            ));
        }
    }

    // Negative: a failing platform stage blocks.
    let mut failing_receipt = platform_receipt.clone();
    failing_receipt["stages"][2]["status"] = json!("fail");
    match admit_prerequisites(
        &profile,
        &platform_admission,
        &failing_receipt,
        &cpu_receipt,
        &cpu_receipt_blake3,
    ) {
        Err(rejection) if rejection.code == "failing_stage" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("failing stage produced {other:?}"),
            ));
        }
    }

    // Negative: a missing platform stage blocks.
    let mut missing_stage_receipt = platform_receipt.clone();
    if let Some(stages) = missing_stage_receipt["stages"].as_array_mut() {
        stages.pop();
    }
    match admit_prerequisites(
        &profile,
        &platform_admission,
        &missing_stage_receipt,
        &cpu_receipt,
        &cpu_receipt_blake3,
    ) {
        Err(rejection) if rejection.code == "missing_stage" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("missing stage produced {other:?}"),
            ));
        }
    }

    // Negative: a drifted model revision breaks authority binding.
    let mut drifted_profile = profile.clone();
    drifted_profile["model"]["revision"] = json!("f".repeat(HEX_HASH_LEN - 24));
    match admit_prerequisites(
        &drifted_profile,
        &platform_admission,
        &platform_receipt,
        &cpu_receipt,
        &cpu_receipt_blake3,
    ) {
        Err(rejection) if rejection.code == "model_mismatch" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("drifted revision produced {other:?}"),
            ));
        }
    }

    // Negative: a non-pass CPU outcome blocks.
    let mut failed_cpu = cpu_receipt.clone();
    failed_cpu["outcome"] = json!("bounded_cpu_reference_fail");
    match admit_prerequisites(
        &profile,
        &platform_admission,
        &platform_receipt,
        &failed_cpu,
        &cpu_receipt_blake3,
    ) {
        Err(rejection) if rejection.code == "outcome_mismatch" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("failed CPU outcome produced {other:?}"),
            ));
        }
    }

    // Negative: a malformed CPU receipt BLAKE3 blocks.
    match admit_prerequisites(
        &profile,
        &platform_admission,
        &platform_receipt,
        &cpu_receipt,
        "not-a-hash",
    ) {
        Err(rejection) if rejection.code == "malformed_receipt" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("malformed CPU receipt hash produced {other:?}"),
            ));
        }
    }

    Ok("self-test: PASS".to_string())
}
