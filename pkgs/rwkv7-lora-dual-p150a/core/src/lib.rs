//! Pure admission core for dual-P150A RWKV-7 LoRA data-parallel fine-tuning.
//!
//! r[impl tenstorrent.rwkv7_dual_p150a.authority]
//! r[impl tenstorrent.rwkv7_dual_p150a.configuration]
//! r[impl tenstorrent.rwkv7_dual_p150a.sharding]
//! r[impl tenstorrent.rwkv7_dual_p150a.collectives]
//! r[impl tenstorrent.rwkv7_dual_p150a.replicas]
//! r[impl tenstorrent.rwkv7_dual_p150a.parity]
//!
//! Every function is deterministic and free of I/O, clocks, and process
//! state. The CLI shell in `main.rs` owns all filesystem and process work.

mod generated_model_target_identity;

pub use rwkv7_lora_p150a_core::{
    DeterminismAdmission, Distribution, MeasurementAdmission, MeasurementProfile,
    RegressionAdmission, UtilizationAdmission,
};

use serde_json::{Value, json};

const SCHEMA_VERSION: u64 = 1;
const HEX_HASH_LEN: usize = 64;

// Admitted runtime bundle from the training platform manifest.
const REQUIRED_BUNDLE_ID: &str = "tt-xla-training-2026-08-04";

// Generated CPU authority identities from the canonical model target profile.
const REQUIRED_MODEL_ID: &str = generated_model_target_identity::MODEL_ID;
const REQUIRED_MODEL_REVISION: &str = generated_model_target_identity::MODEL_REVISION;
const REQUIRED_CHECKPOINT_BLAKE3: &str = generated_model_target_identity::MODEL_BLAKE3;
const REQUIRED_TOKENIZER_VOCAB_BLAKE3: &str =
    generated_model_target_identity::TOKENIZER_VOCAB_BLAKE3;
const REQUIRED_CPU_PRECISION: &str = generated_model_target_identity::CPU_REFERENCE_PRECISION;
const REQUIRED_LAYER_COUNT: u64 = generated_model_target_identity::MODEL_LAYER_COUNT as u64;
const REQUIRED_HIDDEN_SIZE: u64 = generated_model_target_identity::HIDDEN_SIZE as u64;
const REQUIRED_HEAD_COUNT: u64 = generated_model_target_identity::HEAD_COUNT as u64;
const REQUIRED_HEAD_SIZE: u64 = generated_model_target_identity::HEAD_SIZE as u64;
const REQUIRED_VOCABULARY_SIZE: u64 = generated_model_target_identity::VOCABULARY_SIZE as u64;
const REQUIRED_VOCABULARY_ENTRY_COUNT: u64 = 65529;
const CPU_REFERENCE_OUTCOME: &str = "bounded_cpu_reference_pass";

// Device facts for the dual-card 1x2 mesh profile. Container device
// isolation is not supported upstream. The runner is fail-closed on
// physical device IDs 0 and 1.
const REQUIRED_PHYSICAL_COUNT: u64 = 2;
const REQUIRED_ARCHITECTURE: &str = "blackhole";
const REQUIRED_BOARD_CLASS: &str = "p150a";
const REQUIRED_DEVICE_ROOT: &str = "/dev/tenstorrent";
const REQUIRED_MESH_DESCRIPTOR: &str = "p150_x2";
const REQUIRED_PHYSICAL_IDS: [u64; 2] = [0, 1];
const REQUIRED_LOGICAL_ROWS: u64 = 1;
const REQUIRED_LOGICAL_COLUMNS: u64 = 2;
const REQUIRED_ISOLATION_STATE: &str = "unsupported";

// Logical data axis, shard policy, reduction policy, and replica policy.
const REQUIRED_AXIS_NAME: &str = "data";
const REQUIRED_REPLICAS: u64 = 2;
const REQUIRED_AXIS_ORIENTATION: &str = "columns";
const REQUIRED_SHARD_KIND: &str = "even_contiguous";
const REQUIRED_GRADIENT_OP: &str = "mean";
const REQUIRED_GRADIENT_SCOPE: &str = "adapter_gradients";
const REQUIRED_LOSS_REDUCTION: &str = "weighted_active_labels";
const REQUIRED_COLLECTIVE_PLACEMENT: &str = "tt";
const REQUIRED_REPLICA_EQUALITY: &str = "exact";
const REQUIRED_REPLICA_MECHANISM: &str = "spmd_replicated";
const REQUIRED_REPLICA_BOUNDARIES: [&str; 3] = ["canonical_save", "optimizer_update", "resume"];

const REQUIRED_PRECISION_CLASS: &str = "bf16_from_fp32_cpu";
const ADMITTED_DTYPES: [&str; 2] = ["bf16", "fp32"];
const REQUIRED_ADAPTER_FORMAT: &str = "safetensors-adapter-only";
const REQUIRED_ARTIFACT_CANONICAL: &str = "single_adapter_after_replica_equality";
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

const REQUIRED_RECEIPT_KINDS: [&str; 12] = [
    "artifact",
    "cleanup",
    "collective",
    "health",
    "optimizer",
    "parity",
    "placement",
    "prerequisite",
    "replica",
    "restoration",
    "run",
    "shard",
];

const BOUNDARY_NON_CLAIMS: [&str; 5] = [
    "no long-run stability claim",
    "no scaling efficiency claim",
    "no speedup claim",
    "no support claim for other RWKV models",
    "no useful generalization claim",
];

const PLATFORM_RECEIPT_KIND: &str = "dual_p150a_platform_smoke";
const COLLECTIVE_RECEIPT_KIND: &str = "rwkv7_lora_dual_p150a_collective";
const STEP_RECEIPT_KIND: &str = "rwkv7_lora_dual_p150a_step";
const RUN_RECEIPT_KIND: &str = "rwkv7_lora_dual_p150a_run";
const CHECKPOINT_RECEIPT_KIND: &str = "rwkv7_lora_dual_p150a_checkpoint";
const PASS_STATUS: &str = "pass";
const DEVICE_TYPE_TT: &str = "TT";
const PLATFORM_ADMITTED_STATUS: &str = "admitted";

const COLLECTIVE_RECEIPT_STAGES: [&str; 6] = [
    "health_before",
    "ownership",
    "probe_execute",
    "cleanup",
    "restore",
    "health_after",
];

const DUAL_RUN_STAGES: [&str; 6] = [
    "health_before",
    "ownership",
    "step_execute",
    "cleanup",
    "restore",
    "health_after",
];

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

/// Profile admitted for collective probes and dual device work.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AdmittedProfile {
    pub identity_blake3: String,
    pub profile_id: String,
    pub replicas: u64,
    pub sequence_length: u64,
    pub micro_batch: u64,
    pub global_batch: u64,
    pub probe_compile_timeout_seconds: u64,
    pub full_compile_timeout_seconds: u64,
    pub max_stablehlo_operations: u64,
    pub hardware_attempt_seconds: u64,
    pub collective_timeout_seconds: u64,
}

/// Deterministic shard plan for one admitted global batch.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShardPlan {
    pub identity_blake3: String,
    pub global_batch_index: u64,
    pub replicas: Vec<ReplicaShard>,
}

/// One replica's contiguous even shard.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplicaShard {
    pub ordinal: u64,
    pub sample_ids: Vec<String>,
    pub active_tokens: u64,
}

/// Deterministic prerequisite admission binding platform, CPU, and
/// single-P150A receipts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PrerequisiteAdmission {
    pub prerequisite_identity: String,
    pub profile_identity: String,
    pub platform_bundle_identity: String,
    pub platform_config_identity: String,
    pub platform_plan_identity: String,
    pub cpu_authority_blake3: String,
    pub cpu_receipt_blake3: String,
    pub single_profile_identity: String,
    pub single_step_identity: String,
    pub single_run_identity: String,
    pub single_checkpoint_identity: String,
}

/// Admitted collective probe receipt.
// PartialEq only: float fields cannot satisfy Eq.
#[derive(Clone, Debug, PartialEq)]
pub struct CollectiveAdmission {
    pub identity_blake3: String,
    pub all_reduce_count: u64,
    pub collective_seconds: f64,
}

/// Admitted dual-replica parity step receipt.
// PartialEq only: float fields cannot satisfy Eq.
#[derive(Clone, Debug, PartialEq)]
pub struct StepAdmission {
    pub identity_blake3: String,
    pub stablehlo_op_count: u64,
    pub loss_dev: f64,
    pub grad_pass_fraction: f64,
    pub delta_pass_fraction: f64,
}

/// Admitted bounded dual-replica run receipt.
// PartialEq only: float fields cannot satisfy Eq.
#[derive(Clone, Debug, PartialEq)]
pub struct RunAdmission {
    pub identity_blake3: String,
    pub steps: u64,
    pub initial_loss: f64,
    pub final_loss: f64,
}

/// Admitted canonical checkpoint receipt.
// PartialEq only: float fields cannot satisfy Eq.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckpointAdmission {
    pub identity_blake3: String,
    pub adapter_load_max_diff: f64,
    pub resume_loss_dev: f64,
}

/// Deterministic BLAKE3 hex digest over canonical JSON.
pub fn blake3_identity(value: &Value) -> String {
    blake3::hash(&serde_json::to_vec(value).expect("canonical JSON serialization")).to_hex()
        [..HEX_HASH_LEN]
        .to_string()
}

fn shared_measurement_rejection(rejection: rwkv7_lora_p150a_core::Rejection) -> Rejection {
    Rejection::new(rejection.code, rejection.detail)
}

pub fn bind_machine_facts(values: &Value) -> Result<Value, Rejection> {
    rwkv7_lora_p150a_core::bind_machine_facts(values).map_err(shared_measurement_rejection)
}

pub fn summarize_measurement(raw: &Value, profile: &Value) -> Result<Value, Rejection> {
    rwkv7_lora_p150a_core::summarize_measurement(raw, profile).map_err(shared_measurement_rejection)
}

pub fn admit_measurement_receipt(
    receipt: &Value,
    profile: &Value,
) -> Result<MeasurementAdmission, Rejection> {
    rwkv7_lora_p150a_core::admit_measurement_receipt(receipt, profile)
        .map_err(shared_measurement_rejection)
}

pub fn admit_measurement_profile(profile: &Value) -> Result<MeasurementProfile, Rejection> {
    rwkv7_lora_p150a_core::admit_measurement_profile(profile).map_err(shared_measurement_rejection)
}

pub fn summarize_utilization(raw: &Value, profile: &Value) -> Result<Value, Rejection> {
    rwkv7_lora_p150a_core::summarize_utilization(raw, profile).map_err(shared_measurement_rejection)
}

pub fn admit_utilization_receipt(
    receipt: &Value,
    raw: &Value,
    profile: &Value,
) -> Result<UtilizationAdmission, Rejection> {
    rwkv7_lora_p150a_core::admit_utilization_receipt(receipt, raw, profile)
        .map_err(shared_measurement_rejection)
}

pub fn create_baseline(
    receipt: &Value,
    profile: &Value,
    policy: &Value,
) -> Result<Value, Rejection> {
    rwkv7_lora_p150a_core::create_baseline(receipt, profile, policy)
        .map_err(shared_measurement_rejection)
}

pub fn admit_baseline_store(
    store: &Value,
    profile: &Value,
    receipt: &Value,
) -> Result<Value, Rejection> {
    rwkv7_lora_p150a_core::admit_baseline_store(store, profile, receipt)
        .map_err(shared_measurement_rejection)
}

pub fn gate_regression(
    receipt: &Value,
    profile: &Value,
    store: &Value,
    policy: &Value,
) -> Result<Value, Rejection> {
    rwkv7_lora_p150a_core::gate_regression(receipt, profile, store, policy)
        .map_err(shared_measurement_rejection)
}

pub fn admit_regression_receipt(receipt: &Value) -> Result<RegressionAdmission, Rejection> {
    rwkv7_lora_p150a_core::admit_regression_receipt(receipt).map_err(shared_measurement_rejection)
}

pub fn compare_determinism(
    left: &Value,
    right: &Value,
    profile: &Value,
    policy: &Value,
) -> Result<Value, Rejection> {
    rwkv7_lora_p150a_core::compare_determinism(left, right, profile, policy)
        .map_err(shared_measurement_rejection)
}

pub fn admit_determinism_receipt(receipt: &Value) -> Result<DeterminismAdmission, Rejection> {
    rwkv7_lora_p150a_core::admit_determinism_receipt(receipt).map_err(shared_measurement_rejection)
}

fn is_hash_hex(value: &str) -> bool {
    value.len() == HEX_HASH_LEN
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
    let number = field(value, key)?.as_u64().ok_or_else(|| {
        Rejection::new(
            "dynamic_shape",
            format!("field '{key}' is not a positive integer"),
        )
    })?;
    if number == 0 {
        return Err(Rejection::new(
            "dynamic_shape",
            format!("field '{key}' must be positive"),
        ));
    }
    Ok(number)
}

fn nonnegative_u64(value: &Value, key: &str) -> Result<u64, Rejection> {
    field(value, key)?.as_u64().ok_or_else(|| {
        Rejection::new(
            "malformed_field",
            format!("field '{key}' is not a non-negative integer"),
        )
    })
}

fn positive_number(value: &Value, key: &str) -> Result<f64, Rejection> {
    let number = field(value, key)?
        .as_f64()
        .ok_or_else(|| Rejection::new("bad_limit", format!("field '{key}' is not a number")))?;
    if number <= 0.0 || number.is_nan() {
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
    let array = field(value, key)?.as_array().ok_or_else(|| {
        Rejection::new("malformed_field", format!("field '{key}' is not an array"))
    })?;
    array
        .iter()
        .map(|item| {
            item.as_str().ok_or_else(|| {
                Rejection::new(
                    "malformed_field",
                    format!("field '{key}' has a non-string entry"),
                )
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

fn check_device(device: &Value) -> Result<(), Rejection> {
    require_equal_u64(
        device,
        "physical_count",
        REQUIRED_PHYSICAL_COUNT,
        "device_mismatch",
    )?;
    require_equal_str(
        device,
        "architecture",
        REQUIRED_ARCHITECTURE,
        "device_mismatch",
    )?;
    require_equal_str(
        device,
        "board_class",
        REQUIRED_BOARD_CLASS,
        "device_mismatch",
    )?;
    require_equal_str(
        device,
        "device_root",
        REQUIRED_DEVICE_ROOT,
        "device_mismatch",
    )?;
    require_equal_str(
        device,
        "mesh_descriptor",
        REQUIRED_MESH_DESCRIPTOR,
        "device_mismatch",
    )?;
    require_equal_str(
        device,
        "container_device_isolation",
        REQUIRED_ISOLATION_STATE,
        "device_mismatch",
    )?;
    let ids = field(device, "physical_device_ids")?
        .as_array()
        .ok_or_else(|| Rejection::new("device_mismatch", "physical_device_ids is not an array"))?;
    let observed: Vec<u64> = ids
        .iter()
        .map(|entry| {
            entry.as_u64().ok_or_else(|| {
                Rejection::new(
                    "device_mismatch",
                    "physical_device_ids has a non-integer entry",
                )
            })
        })
        .collect::<Result<_, _>>()?;
    if observed != REQUIRED_PHYSICAL_IDS {
        return Err(Rejection::new(
            "device_mismatch",
            format!(
                "physical_device_ids {observed:?} must be {:?}",
                REQUIRED_PHYSICAL_IDS
            ),
        ));
    }
    let logical = field(device, "logical_mesh")?;
    require_equal_u64(logical, "rows", REQUIRED_LOGICAL_ROWS, "device_mismatch")?;
    require_equal_u64(
        logical,
        "columns",
        REQUIRED_LOGICAL_COLUMNS,
        "device_mismatch",
    )?;
    Ok(())
}

fn check_data_axis(axis: &Value) -> Result<(), Rejection> {
    require_equal_str(axis, "name", REQUIRED_AXIS_NAME, "invalid_axis")?;
    require_equal_u64(axis, "replicas", REQUIRED_REPLICAS, "invalid_axis")?;
    require_equal_str(
        axis,
        "orientation",
        REQUIRED_AXIS_ORIENTATION,
        "invalid_axis",
    )?;
    Ok(())
}

fn check_shard_policy(policy: &Value) -> Result<(), Rejection> {
    require_equal_str(policy, "kind", REQUIRED_SHARD_KIND, "invalid_shard_policy")?;
    let allow_uneven = bool_field(policy, "allow_uneven")?;
    if allow_uneven {
        return Err(Rejection::new(
            "invalid_shard_policy",
            "uneven shards are not admitted for this profile",
        ));
    }
    Ok(())
}

fn check_reduction_policy(policy: &Value) -> Result<(), Rejection> {
    require_equal_str(
        policy,
        "gradient_op",
        REQUIRED_GRADIENT_OP,
        "wrong_reduction",
    )?;
    require_equal_str(
        policy,
        "gradient_scope",
        REQUIRED_GRADIENT_SCOPE,
        "wrong_reduction",
    )?;
    require_equal_str(
        policy,
        "loss_reduction",
        REQUIRED_LOSS_REDUCTION,
        "wrong_reduction",
    )?;
    require_equal_str(
        policy,
        "placement",
        REQUIRED_COLLECTIVE_PLACEMENT,
        "host_fallback",
    )?;
    Ok(())
}

fn check_replica_policy(policy: &Value) -> Result<(), Rejection> {
    require_equal_str(
        policy,
        "equality",
        REQUIRED_REPLICA_EQUALITY,
        "invalid_replica_policy",
    )?;
    require_equal_str(
        policy,
        "mechanism",
        REQUIRED_REPLICA_MECHANISM,
        "invalid_replica_policy",
    )?;
    let boundaries = string_array(policy, "boundaries")?;
    for required in REQUIRED_REPLICA_BOUNDARIES {
        if !boundaries.contains(&required) {
            return Err(Rejection::new(
                "invalid_replica_policy",
                format!("replica boundaries are missing '{required}'"),
            ));
        }
    }
    Ok(())
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
        let entry = tolerances.get(role).ok_or_else(|| {
            Rejection::new(
                "missing_tolerance",
                format!("tolerance role '{role}' is missing"),
            )
        })?;
        positive_number(entry, "rtol").map_err(|_| {
            Rejection::new(
                "missing_tolerance",
                format!("tolerance role '{role}' has a bad rtol"),
            )
        })?;
        positive_number(entry, "atol").map_err(|_| {
            Rejection::new(
                "missing_tolerance",
                format!("tolerance role '{role}' has a bad atol"),
            )
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
    let pass_fraction = positive_number(criteria, "parity_pass_fraction")?;
    if pass_fraction > 1.0 {
        return Err(Rejection::new(
            "bad_run_criteria",
            format!("parity_pass_fraction {pass_fraction} exceeds one"),
        ));
    }
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
    let admitted: Vec<&str> = ADMITTED_LORA_ROLES.to_vec();
    if roles != admitted {
        return Err(Rejection::new(
            "bad_lora_plan",
            format!("target_roles {roles:?} must equal the CPU authority allowlist"),
        ));
    }
    Ok(())
}

fn check_optimizer(optimizer: &Value, static_shape: &Value) -> Result<(), Rejection> {
    require_equal_str(
        optimizer,
        "kind",
        REQUIRED_OPTIMIZER_KIND,
        "optimizer_mismatch",
    )?;
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
    require_equal_str(
        policy,
        "format",
        REQUIRED_ADAPTER_FORMAT,
        "bad_artifact_policy",
    )?;
    require_equal_str(
        policy,
        "canonical",
        REQUIRED_ARTIFACT_CANONICAL,
        "bad_artifact_policy",
    )?;
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
    require_equal_str(
        policy,
        "hash_algorithm",
        REQUIRED_HASH_ALGORITHM,
        "bad_evidence_policy",
    )?;
    let exclude = string_array(policy, "exclude")?;
    for entry in &exclude {
        let known = [
            "credentials",
            "raw_logs",
            "serial_numbers",
            "absolute_private_paths",
        ];
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
    require_equal_u64(
        profile,
        "schema_version",
        SCHEMA_VERSION,
        "bad_schema_version",
    )?;
    let profile_id = string_field(profile, "profile_id")?.to_string();
    if profile_id.is_empty() {
        return Err(Rejection::new("bad_profile_id", "profile_id is empty"));
    }
    require_equal_str(profile, "bundle_id", REQUIRED_BUNDLE_ID, "bad_bundle_id")?;

    let model = field(profile, "model")?;
    check_model_authority(model)?;
    check_tokenizer(field(profile, "tokenizer")?)?;
    check_device(field(profile, "device")?)?;
    check_data_axis(field(profile, "data_axis")?)?;
    check_shard_policy(field(profile, "shard_policy")?)?;
    check_reduction_policy(field(profile, "reduction_policy")?)?;
    check_replica_policy(field(profile, "replica_policy")?)?;

    let static_shape = field(profile, "static_shape")?;
    let micro_batch = positive_u64(static_shape, "micro_batch")?;
    let sequence_length = positive_u64(static_shape, "sequence_length")?;
    positive_u64(static_shape, "gradient_accumulation_steps")?;

    check_probe(field(profile, "probe")?, static_shape, model)?;
    check_precision(field(profile, "precision")?)?;

    let compile_limits = field(profile, "compile_limits")?;
    let probe_compile_timeout_seconds =
        positive_u64(compile_limits, "probe_compile_timeout_seconds")?;
    let full_compile_timeout_seconds =
        positive_u64(compile_limits, "full_compile_timeout_seconds")?;
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
    let collective_timeout_seconds = positive_u64(timeouts, "collective_timeout_seconds")?;

    check_lora(field(profile, "lora")?)?;
    check_optimizer(field(profile, "optimizer")?, static_shape)?;
    nonnegative_u64(profile, "seed")?;
    check_artifact_policy(field(profile, "artifact_policy")?)?;
    check_evidence_policy(field(profile, "evidence_policy")?)?;

    Ok(AdmittedProfile {
        identity_blake3: blake3_identity(profile),
        profile_id,
        replicas: REQUIRED_REPLICAS,
        sequence_length,
        micro_batch,
        global_batch: micro_batch * REQUIRED_REPLICAS,
        probe_compile_timeout_seconds,
        full_compile_timeout_seconds,
        max_stablehlo_operations,
        hardware_attempt_seconds,
        collective_timeout_seconds,
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

/// Derive the deterministic shard plan for one admitted global batch.
///
/// The plan assigns every sample to exactly one logical replica under the
/// even contiguous policy: replica `r` owns samples
/// `[r * per_replica, (r + 1) * per_replica)` in the global order. The plan
/// rejects a batch that cannot shard evenly, a duplicate or missing sample,
/// and a sample below the active-label floor.
pub fn plan_shards(profile: &Value, batch: &Value) -> Result<ShardPlan, Rejection> {
    let admitted = admit_profile(profile)?;
    let global_batch_index = nonnegative_u64(batch, "global_batch_index")?;
    let samples = field(batch, "samples")?
        .as_array()
        .ok_or_else(|| Rejection::new("invalid_shard_plan", "samples is not an array"))?;

    let expected = admitted.global_batch as usize;
    if samples.len() != expected {
        return Err(Rejection::new(
            "uneven_batch",
            format!(
                "global batch has {} samples, the even policy requires exactly {expected}",
                samples.len()
            ),
        ));
    }

    let min_active = positive_u64(
        field(profile, "run_criteria")?,
        "min_active_labels_per_sample",
    )?;
    let mut seen: Vec<&str> = Vec::with_capacity(samples.len());
    let mut replicas: Vec<ReplicaShard> = (0..admitted.replicas)
        .map(|ordinal| ReplicaShard {
            ordinal,
            sample_ids: Vec::new(),
            active_tokens: 0,
        })
        .collect();
    let per_replica = expected / admitted.replicas as usize;

    for (index, sample) in samples.iter().enumerate() {
        let sample_id = string_field(sample, "sample_id")?;
        if seen.contains(&sample_id) {
            return Err(Rejection::new(
                "duplicate_sample",
                format!("sample '{sample_id}' appears twice in the global batch"),
            ));
        }
        seen.push(sample_id);
        let active_labels = positive_u64(sample, "active_labels")?;
        if active_labels < min_active {
            return Err(Rejection::new(
                "inactive_sample",
                format!(
                    "sample '{sample_id}' has {active_labels} active labels below the floor {min_active}"
                ),
            ));
        }
        let replica = index / per_replica;
        let shard = replicas
            .get_mut(replica)
            .ok_or_else(|| Rejection::new("invalid_shard_plan", "replica index out of range"))?;
        shard.sample_ids.push(sample_id.to_string());
        shard.active_tokens += active_labels;
    }

    let plan_doc = json!({
        "schema_version": SCHEMA_VERSION,
        "profile_identity": admitted.identity_blake3,
        "global_batch_index": global_batch_index,
        "shard_kind": REQUIRED_SHARD_KIND,
        "replicas": replicas
            .iter()
            .map(|shard| json!({
                "ordinal": shard.ordinal,
                "sample_ids": shard.sample_ids,
                "active_tokens": shard.active_tokens,
            }))
            .collect::<Vec<_>>(),
    });

    Ok(ShardPlan {
        identity_blake3: blake3_identity(&plan_doc),
        global_batch_index,
        replicas,
    })
}

fn platform_stage_status(receipt: &Value, stage: &str) -> Result<String, Rejection> {
    let stages = field(receipt, "stages")?.as_array().ok_or_else(|| {
        Rejection::new(
            "malformed_receipt",
            "platform receipt stages is not an array",
        )
    })?;
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

/// Inputs for dual prerequisite admission. Grouped so the admission
/// signature stays explicit about the full authority chain.
pub struct PrerequisiteInputs<'a> {
    pub profile: &'a Value,
    pub platform_admission: &'a Value,
    pub platform_receipt: &'a Value,
    pub cpu_receipt: &'a Value,
    pub cpu_receipt_blake3: &'a str,
    pub single_profile: &'a Value,
    pub single_step_receipt: &'a Value,
    pub single_run_receipt: &'a Value,
    pub single_checkpoint_receipt: &'a Value,
}

/// Admit prerequisite receipts before any accelerator work.
///
/// The platform receipt must be admitted by the platform core and carry all
/// passing stages. The CPU receipt must carry the bounded pass outcome and
/// an authority identity that matches the profile exactly. The single-P150A
/// step, run, and checkpoint receipts must re-validate against the admitted
/// single-device profile through the single-P150A admission core.
pub fn admit_prerequisites(
    inputs: &PrerequisiteInputs,
) -> Result<PrerequisiteAdmission, Rejection> {
    let profile = inputs.profile;
    let platform_admission = inputs.platform_admission;
    let platform_receipt = inputs.platform_receipt;
    let cpu_receipt = inputs.cpu_receipt;
    let cpu_receipt_blake3 = inputs.cpu_receipt_blake3;
    let single_profile = inputs.single_profile;
    let single_step_receipt = inputs.single_step_receipt;
    let single_run_receipt = inputs.single_run_receipt;
    let single_checkpoint_receipt = inputs.single_checkpoint_receipt;
    let admitted = admit_profile(profile)?;

    let admission_status = string_field(platform_admission, "status")?;
    if admission_status != PLATFORM_ADMITTED_STATUS {
        return Err(Rejection::new(
            "stale_prerequisite",
            format!("platform receipt admission status is '{admission_status}'"),
        ));
    }

    require_equal_str(
        platform_receipt,
        "kind",
        PLATFORM_RECEIPT_KIND,
        "stale_prerequisite",
    )?;
    for stage in PLATFORM_RECEIPT_STAGES {
        let status = platform_stage_status(platform_receipt, stage)?;
        if status != PASS_STATUS {
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

    let observed_checkpoint = string_field(field(cpu_receipt, "model_load")?, "checkpoint_blake3")?;
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

    // The single-P150A authority chain re-validates through its own core.
    // A stale, drifted, or failing single-device receipt blocks dual launch.
    let map_single = |rejection: rwkv7_lora_p150a_core::Rejection| {
        Rejection::new(
            "stale_prerequisite",
            format!("single-P150A chain: {rejection}"),
        )
    };
    let single_admitted =
        rwkv7_lora_p150a_core::admit_profile(single_profile).map_err(map_single)?;
    let single_step =
        rwkv7_lora_p150a_core::admit_step_receipt(single_step_receipt, single_profile)
            .map_err(map_single)?;
    let single_run = rwkv7_lora_p150a_core::admit_run_receipt(single_run_receipt, single_profile)
        .map_err(map_single)?;
    let single_checkpoint =
        rwkv7_lora_p150a_core::admit_checkpoint_receipt(single_checkpoint_receipt, single_profile)
            .map_err(map_single)?;

    // The single-device authority must name the same model checkpoint as the
    // dual profile. Both cores re-check the authority constants, so an
    // admitted single profile already binds the same model; the identity
    // binding below makes the pair explicit in the prerequisite receipt.
    let binding = json!({
        "cpu_authority_blake3": expected_authority,
        "cpu_receipt_blake3": cpu_receipt_blake3,
        "platform_bundle_identity": platform_bundle_identity,
        "platform_config_identity": platform_config_identity,
        "platform_plan_identity": platform_plan_identity,
        "profile_identity": admitted.identity_blake3,
        "schema_version": SCHEMA_VERSION,
        "single_profile_identity": single_admitted.identity_blake3,
        "single_step_identity": single_step.identity_blake3,
        "single_run_identity": single_run.identity_blake3,
        "single_checkpoint_identity": single_checkpoint.identity_blake3,
    });

    Ok(PrerequisiteAdmission {
        prerequisite_identity: blake3_identity(&binding),
        profile_identity: admitted.identity_blake3,
        platform_bundle_identity,
        platform_config_identity,
        platform_plan_identity,
        cpu_authority_blake3: expected_authority,
        cpu_receipt_blake3: cpu_receipt_blake3.to_string(),
        single_profile_identity: single_admitted.identity_blake3,
        single_step_identity: single_step.identity_blake3,
        single_run_identity: single_run.identity_blake3,
        single_checkpoint_identity: single_checkpoint.identity_blake3,
    })
}

fn receipt_stage_status(receipt: &Value, stage: &str) -> Result<String, Rejection> {
    let stages = field(receipt, "stages")?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_receipt", "receipt stages is not an array"))?;
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
        format!("receipt is missing stage '{stage}'"),
    ))
}

fn check_receipt_identity(
    receipt: &Value,
    kind: &str,
    stale_code: &'static str,
    admitted: &AdmittedProfile,
) -> Result<(), Rejection> {
    require_equal_u64(
        receipt,
        "schema_version",
        SCHEMA_VERSION,
        "bad_schema_version",
    )?;
    require_equal_str(receipt, "kind", kind, stale_code)?;
    let profile_identity = string_field(receipt, "profile_identity")?;
    if profile_identity != admitted.identity_blake3 {
        return Err(Rejection::new(
            stale_code,
            format!(
                "receipt profile identity {profile_identity} differs from admitted profile {}",
                admitted.identity_blake3
            ),
        ));
    }
    Ok(())
}

fn check_stages(receipt: &Value, stages: &[&str]) -> Result<(), Rejection> {
    for stage in stages {
        let status = receipt_stage_status(receipt, stage)?;
        if status != PASS_STATUS {
            return Err(Rejection::new(
                "failing_stage",
                format!("receipt stage '{stage}' has status '{status}'"),
            ));
        }
    }
    Ok(())
}

/// Admit one collective probe receipt against the profile.
///
/// The receipt must carry all passing host stages, a two-replica logical
/// mesh over both physical devices, exact mean all-reduce evidence with both
/// devices contributing, TT placement, and collective timing inside the
/// named timeout.
pub fn admit_collective(
    receipt: &Value,
    profile: &Value,
) -> Result<CollectiveAdmission, Rejection> {
    reject_forbidden_keys(receipt)?;
    let admitted_profile = admit_profile(profile)?;
    check_receipt_identity(
        receipt,
        COLLECTIVE_RECEIPT_KIND,
        "stale_collective_receipt",
        &admitted_profile,
    )?;
    check_stages(receipt, &COLLECTIVE_RECEIPT_STAGES)?;

    let result = field(receipt, "probe_result")?;
    require_equal_str(result, "status", PASS_STATUS, "probe_failed")?;

    let mesh = field(result, "mesh")?;
    require_equal_u64(mesh, "rows", REQUIRED_LOGICAL_ROWS, "invalid_axis")?;
    require_equal_u64(mesh, "columns", REQUIRED_LOGICAL_COLUMNS, "invalid_axis")?;
    require_equal_str(mesh, "axis_name", REQUIRED_AXIS_NAME, "invalid_axis")?;
    require_equal_u64(
        mesh,
        "device_count",
        REQUIRED_PHYSICAL_COUNT,
        "device_mismatch",
    )?;

    let all_reduce = field(result, "all_reduce")?;
    require_equal_str(all_reduce, "op", REQUIRED_GRADIENT_OP, "wrong_reduction")?;
    require_equal_str(all_reduce, "placement", "TT", "host_fallback")?;
    let both_contributed = bool_field(all_reduce, "both_contributed")?;
    if !both_contributed {
        return Err(Rejection::new(
            "collective_contribution",
            "the all-reduce did not observe an admitted input from both devices",
        ));
    }
    let values_exact = bool_field(all_reduce, "values_exact")?;
    if !values_exact {
        return Err(Rejection::new(
            "collective_values",
            "the all-reduce returned wrong values",
        ));
    }
    // The collective lowers inside the TT backend after SPMD partitioning,
    // so the pre-partition graph cannot carry an explicit all-reduce op.
    // Placement evidence is the data-axis sharding annotation on the input
    // plus the replicated result, which only fabric execution can satisfy
    // with exact values from distinct per-device contributions.
    let contribution_spec = string_field(all_reduce, "contribution_spec")?;
    if !contribution_spec.contains("devices=[2") {
        return Err(Rejection::new(
            "host_fallback",
            format!("collective input is not sharded over both devices: '{contribution_spec}'"),
        ));
    }
    let result_spec = string_field(all_reduce, "result_spec")?;
    if result_spec != "{replicated}" {
        return Err(Rejection::new(
            "host_fallback",
            format!("collective result is not replicated: '{result_spec}'"),
        ));
    }
    let all_reduce_count = field(all_reduce, "count")?
        .as_u64()
        .ok_or_else(|| Rejection::new("malformed_receipt", "all_reduce count is not an integer"))?;
    if all_reduce_count == 0 {
        return Err(Rejection::new(
            "collective_values",
            "the collective probe executed no all-reduce iterations",
        ));
    }

    let barrier = field(result, "barrier")?;
    let synchronized = bool_field(barrier, "synchronized")?;
    if !synchronized {
        return Err(Rejection::new(
            "collective_synchronization",
            "the barrier did not synchronize both replicas",
        ));
    }

    let finite = bool_field(result, "finite")?;
    if !finite {
        return Err(Rejection::new(
            "nonfinite_values",
            "collective probe reports nonfinite values",
        ));
    }

    let collective_seconds = field(result, "collective_seconds")?
        .as_f64()
        .ok_or_else(|| Rejection::new("malformed_receipt", "collective_seconds is not a number"))?;
    if collective_seconds < 0.0 {
        return Err(Rejection::new(
            "malformed_receipt",
            "collective_seconds is negative",
        ));
    }
    if collective_seconds > admitted_profile.collective_timeout_seconds as f64 {
        return Err(Rejection::new(
            "collective_timeout",
            format!(
                "collectives took {collective_seconds}s above the limit {}s",
                admitted_profile.collective_timeout_seconds
            ),
        ));
    }

    Ok(CollectiveAdmission {
        identity_blake3: blake3_identity(receipt),
        all_reduce_count,
        collective_seconds,
    })
}

/// Check the shard plan block carried by a dual receipt.
fn check_shard_plan(receipt: &Value) -> Result<(), Rejection> {
    let shard_plan = field(receipt, "shard_plan")?;
    let shard_identity = string_field(shard_plan, "identity_blake3")?;
    if !is_hash_hex(shard_identity) {
        return Err(Rejection::new(
            "malformed_receipt",
            "shard plan identity is not a hex hash",
        ));
    }
    require_equal_str(
        shard_plan,
        "shard_kind",
        REQUIRED_SHARD_KIND,
        "invalid_shard_plan",
    )?;
    let shards = field(shard_plan, "replicas")?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_receipt", "shard replicas is not an array"))?;
    if shards.len() != REQUIRED_REPLICAS as usize {
        return Err(Rejection::new(
            "invalid_shard_plan",
            format!(
                "shard plan has {} replicas, expected {REQUIRED_REPLICAS}",
                shards.len()
            ),
        ));
    }
    for shard in shards {
        let active = field(shard, "active_tokens")?.as_u64().ok_or_else(|| {
            Rejection::new("malformed_receipt", "active_tokens is not an integer")
        })?;
        if active == 0 {
            return Err(Rejection::new(
                "missing_work",
                "a replica shard carries no active tokens",
            ));
        }
    }
    Ok(())
}

/// Check the shared dual-run evidence block: mesh placement, shard plan,
/// per-device work, reduction order, and replica equality.
fn check_dual_evidence(
    receipt: &Value,
    admitted_profile: &AdmittedProfile,
) -> Result<u64, Rejection> {
    let placement = field(receipt, "placement")?;
    require_equal_str(placement, "device", DEVICE_TYPE_TT, "host_fallback")?;
    let mesh = field(placement, "mesh")?;
    require_equal_u64(mesh, "rows", REQUIRED_LOGICAL_ROWS, "invalid_axis")?;
    require_equal_u64(mesh, "columns", REQUIRED_LOGICAL_COLUMNS, "invalid_axis")?;
    let visible_count = field(placement, "visible_count")?
        .as_u64()
        .ok_or_else(|| Rejection::new("malformed_receipt", "visible_count is not an integer"))?;
    if visible_count != REQUIRED_PHYSICAL_COUNT {
        return Err(Rejection::new(
            "device_mismatch",
            format!(
                "visible device count {visible_count} differs from physical count {REQUIRED_PHYSICAL_COUNT}"
            ),
        ));
    }
    let op_count = field(placement, "stablehlo_op_count")?
        .as_u64()
        .ok_or_else(|| {
            Rejection::new("malformed_receipt", "stablehlo_op_count is not an integer")
        })?;
    if op_count == 0 {
        return Err(Rejection::new(
            "host_fallback",
            "placement graph contains no StableHLO operations",
        ));
    }
    if op_count > admitted_profile.max_stablehlo_operations {
        return Err(Rejection::new(
            "op_limit_exceeded",
            format!(
                "placement graph has {op_count} operations above the limit {}",
                admitted_profile.max_stablehlo_operations
            ),
        ));
    }
    let input_spec = string_field(placement, "input_spec")?;
    if !input_spec.contains("devices=[2") {
        return Err(Rejection::new(
            "host_fallback",
            format!("batch input is not sharded over both devices: '{input_spec}'"),
        ));
    }

    let shard_plan = field(receipt, "shard_plan")?;
    let shard_identity = string_field(shard_plan, "identity_blake3")?;
    if !is_hash_hex(shard_identity) {
        return Err(Rejection::new(
            "malformed_receipt",
            "shard plan identity is not a hex hash",
        ));
    }
    require_equal_str(
        shard_plan,
        "shard_kind",
        REQUIRED_SHARD_KIND,
        "invalid_shard_plan",
    )?;
    let shards = field(shard_plan, "replicas")?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_receipt", "shard replicas is not an array"))?;
    if shards.len() != REQUIRED_REPLICAS as usize {
        return Err(Rejection::new(
            "invalid_shard_plan",
            format!(
                "shard plan has {} replicas, expected {REQUIRED_REPLICAS}",
                shards.len()
            ),
        ));
    }
    for shard in shards {
        let active = field(shard, "active_tokens")?.as_u64().ok_or_else(|| {
            Rejection::new("malformed_receipt", "active_tokens is not an integer")
        })?;
        if active == 0 {
            return Err(Rejection::new(
                "missing_work",
                "a replica shard carries no active tokens",
            ));
        }
    }

    let optimizer = field(receipt, "optimizer")?;
    require_equal_str(
        optimizer,
        "gradient_op",
        REQUIRED_GRADIENT_OP,
        "wrong_reduction",
    )?;
    require_equal_str(optimizer, "placement", "TT", "host_fallback")?;
    let reduce_before_clip = bool_field(optimizer, "reduce_before_clip")?;
    if !reduce_before_clip {
        return Err(Rejection::new(
            "wrong_order",
            "clipping happened before the gradient reduction",
        ));
    }
    let grads_replicated = bool_field(optimizer, "grads_replicated")?;
    if !grads_replicated {
        return Err(Rejection::new(
            "replica_divergence",
            "reduced adapter gradients are not replicated across devices",
        ));
    }
    let base_unchanged = bool_field(optimizer, "base_unchanged")?;
    if !base_unchanged {
        return Err(Rejection::new(
            "base_mutated",
            "a base tensor changed during the step",
        ));
    }

    let replicas = field(receipt, "replicas")?;
    require_equal_str(
        replicas,
        "equality",
        REQUIRED_REPLICA_EQUALITY,
        "replica_divergence",
    )?;
    require_equal_str(
        replicas,
        "mechanism",
        REQUIRED_REPLICA_MECHANISM,
        "replica_divergence",
    )?;
    let verified = bool_field(replicas, "verified")?;
    if !verified {
        return Err(Rejection::new(
            "replica_divergence",
            "replica equality was not verified at the optimizer boundary",
        ));
    }

    Ok(op_count)
}

/// Admit one dual-replica parity step receipt against the profile.
///
/// The receipt must carry all passing host stages, current mesh placement
/// evidence, a complete shard plan, synchronized adapter gradients before
/// clipping, exact replica equality, and parity results that satisfy the
/// profile's named tolerances and pass fractions against the single-device
/// global-batch reference.
pub fn admit_step_receipt(receipt: &Value, profile: &Value) -> Result<StepAdmission, Rejection> {
    reject_forbidden_keys(receipt)?;
    let admitted_profile = admit_profile(profile)?;
    check_receipt_identity(
        receipt,
        STEP_RECEIPT_KIND,
        "stale_step_receipt",
        &admitted_profile,
    )?;
    check_stages(receipt, &DUAL_RUN_STAGES)?;
    let op_count = check_dual_evidence(receipt, &admitted_profile)?;

    let step = field(receipt, "step")?;
    let finite = bool_field(step, "finite")?;
    if !finite {
        return Err(Rejection::new(
            "nonfinite_values",
            "step outputs are nonfinite",
        ));
    }
    let loss_dev = field(step, "loss_dev")?
        .as_f64()
        .ok_or_else(|| Rejection::new("malformed_receipt", "loss_dev is not a number"))?;
    let reference_loss = field(step, "reference_loss")?
        .as_f64()
        .ok_or_else(|| Rejection::new("malformed_receipt", "reference_loss is not a number"))?;

    let tolerances = field(profile, "tolerances")?;
    let loss_tolerance = field(tolerances, "loss")?;
    let loss_bound = positive_number(loss_tolerance, "atol")?
        + positive_number(loss_tolerance, "rtol")? * reference_loss.abs();
    if loss_dev > loss_bound {
        return Err(Rejection::new(
            "tolerance_failure",
            format!("loss_dev {loss_dev} exceeds the loss bound {loss_bound}"),
        ));
    }
    let clip_dev = field(step, "clip_dev")?
        .as_f64()
        .ok_or_else(|| Rejection::new("malformed_receipt", "clip_dev is not a number"))?;
    let reference_clip = field(step, "reference_clip_norm")?
        .as_f64()
        .ok_or_else(|| {
            Rejection::new("malformed_receipt", "reference_clip_norm is not a number")
        })?;
    let clip_tolerance = field(tolerances, "clip_coefficient")?;
    let clip_bound = positive_number(clip_tolerance, "atol")?
        + positive_number(clip_tolerance, "rtol")? * reference_clip.abs();
    if clip_dev > clip_bound {
        return Err(Rejection::new(
            "tolerance_failure",
            format!("clip_dev {clip_dev} exceeds the clip bound {clip_bound}"),
        ));
    }

    let pass_fraction = positive_number(field(profile, "run_criteria")?, "parity_pass_fraction")?;
    let grad_pass = field(step, "grad_pass_fraction")?
        .as_f64()
        .ok_or_else(|| Rejection::new("malformed_receipt", "grad_pass_fraction is not a number"))?;
    if grad_pass < pass_fraction {
        return Err(Rejection::new(
            "tolerance_failure",
            format!("gradient pass fraction {grad_pass} below {pass_fraction}"),
        ));
    }
    let delta_pass = field(step, "delta_pass_fraction")?
        .as_f64()
        .ok_or_else(|| {
            Rejection::new("malformed_receipt", "delta_pass_fraction is not a number")
        })?;
    if delta_pass < pass_fraction {
        return Err(Rejection::new(
            "tolerance_failure",
            format!("adapter delta pass fraction {delta_pass} below {pass_fraction}"),
        ));
    }

    Ok(StepAdmission {
        identity_blake3: blake3_identity(receipt),
        stablehlo_op_count: op_count,
        loss_dev,
        grad_pass_fraction: grad_pass,
        delta_pass_fraction: delta_pass,
    })
}

/// Admit one bounded dual-replica run receipt against the profile.
///
/// The run binds the admitted parity step receipt, carries the admitted
/// shard plan, stays finite, mutates only adapters, keeps the step count
/// inside the named bounds, does model work on both devices, synchronizes
/// one gradient reduction per step, verifies replica equality at every
/// admitted boundary, passes cleanup and restoration, and satisfies the
/// configured loss-improvement criterion.
pub fn admit_run_receipt(receipt: &Value, profile: &Value) -> Result<RunAdmission, Rejection> {
    reject_forbidden_keys(receipt)?;
    let admitted_profile = admit_profile(profile)?;
    check_receipt_identity(
        receipt,
        RUN_RECEIPT_KIND,
        "stale_run_receipt",
        &admitted_profile,
    )?;
    check_stages(receipt, &DUAL_RUN_STAGES)?;
    check_shard_plan(receipt)?;

    // The bounded run is admitted only after the parity step gate passes.
    let step_identity = string_field(receipt, "step_identity")?;
    if !is_hash_hex(step_identity) {
        return Err(Rejection::new(
            "malformed_receipt",
            "step identity is not a hex hash",
        ));
    }
    if step_identity == "0".repeat(HEX_HASH_LEN) {
        return Err(Rejection::new(
            "stale_run_receipt",
            "step identity is the zero hash",
        ));
    }

    let run = field(receipt, "run")?;
    let finite = bool_field(run, "finite")?;
    if !finite {
        return Err(Rejection::new(
            "nonfinite_values",
            "run produced a nonfinite value",
        ));
    }
    let base_unchanged = bool_field(run, "base_unchanged")?;
    if !base_unchanged {
        return Err(Rejection::new(
            "base_mutated",
            "a base tensor changed during the bounded run",
        ));
    }

    let steps = field(run, "steps")?
        .as_u64()
        .ok_or_else(|| Rejection::new("malformed_receipt", "steps is not an integer"))?;
    let run_criteria = field(profile, "run_criteria")?;
    let min_steps = positive_u64(run_criteria, "min_steps")?;
    let max_steps = positive_u64(run_criteria, "max_steps")?;
    if steps < min_steps || steps > max_steps {
        return Err(Rejection::new(
            "bad_run_criteria",
            format!("steps {steps} is outside [{min_steps}, {max_steps}]"),
        ));
    }

    // Per-device work: every replica must execute every step with active
    // tokens, and every step must run exactly one gradient reduction.
    let per_device = field(run, "per_device")?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_receipt", "per_device is not an array"))?;
    if per_device.len() != REQUIRED_REPLICAS as usize {
        return Err(Rejection::new(
            "missing_work",
            format!(
                "per-device work covers {} replicas, expected {REQUIRED_REPLICAS}",
                per_device.len()
            ),
        ));
    }
    for device in per_device {
        let device_steps = field(device, "steps")?
            .as_u64()
            .ok_or_else(|| Rejection::new("malformed_receipt", "device steps is not an integer"))?;
        if device_steps != steps {
            return Err(Rejection::new(
                "missing_work",
                format!("a replica executed {device_steps} steps, expected {steps}"),
            ));
        }
        let tokens = field(device, "active_tokens")?.as_u64().ok_or_else(|| {
            Rejection::new(
                "malformed_receipt",
                "device active_tokens is not an integer",
            )
        })?;
        if tokens == 0 {
            return Err(Rejection::new(
                "missing_work",
                "a replica carried no active tokens",
            ));
        }
    }
    let gradient_reductions = field(run, "gradient_reductions")?.as_u64().ok_or_else(|| {
        Rejection::new("malformed_receipt", "gradient_reductions is not an integer")
    })?;
    if gradient_reductions != steps {
        return Err(Rejection::new(
            "wrong_reduction",
            format!("run recorded {gradient_reductions} gradient reductions over {steps} steps"),
        ));
    }

    // Replica equality at every admitted boundary.
    let boundaries = field(run, "replica_boundaries")?
        .as_array()
        .ok_or_else(|| Rejection::new("malformed_receipt", "replica_boundaries is not an array"))?;
    if boundaries.len() != steps as usize {
        return Err(Rejection::new(
            "replica_divergence",
            format!(
                "run recorded {} replica boundaries over {steps} steps",
                boundaries.len()
            ),
        ));
    }
    for boundary in boundaries {
        let equal = bool_field(boundary, "equal")?;
        if !equal {
            return Err(Rejection::new(
                "replica_divergence",
                "replicas diverged at an optimizer boundary",
            ));
        }
    }

    let initial_loss = field(run, "initial_loss")?
        .as_f64()
        .ok_or_else(|| Rejection::new("malformed_receipt", "initial_loss is not a number"))?;
    let final_loss = field(run, "final_loss")?
        .as_f64()
        .ok_or_else(|| Rejection::new("malformed_receipt", "final_loss is not a number"))?;
    let improvement_ratio = positive_number(run_criteria, "loss_improvement_ratio")?;
    if final_loss > improvement_ratio * initial_loss {
        return Err(Rejection::new(
            "loss_criterion",
            format!(
                "final loss {final_loss} did not satisfy the improvement criterion (ratio {improvement_ratio} of initial {initial_loss})"
            ),
        ));
    }

    Ok(RunAdmission {
        identity_blake3: blake3_identity(receipt),
        steps,
        initial_loss,
        final_loss,
    })
}

const MAX_ADAPTER_LOAD_DIFF: f64 = 0.0;

/// Admit one canonical checkpoint receipt against the profile.
///
/// Replica equality must be verified before the canonical save. The adapter
/// save/load round trip must be byte-identical, both replicas must reload
/// equal state, and the resumed result must match the uninterrupted run
/// within the named loss tolerance.
pub fn admit_checkpoint_receipt(
    receipt: &Value,
    profile: &Value,
) -> Result<CheckpointAdmission, Rejection> {
    reject_forbidden_keys(receipt)?;
    let admitted_profile = admit_profile(profile)?;
    check_receipt_identity(
        receipt,
        CHECKPOINT_RECEIPT_KIND,
        "stale_checkpoint_receipt",
        &admitted_profile,
    )?;
    check_stages(receipt, &DUAL_RUN_STAGES)?;

    let checkpoint = field(receipt, "checkpoint")?;
    let equality_before_save = bool_field(checkpoint, "replica_equality_before_save")?;
    if !equality_before_save {
        return Err(Rejection::new(
            "replica_divergence",
            "the canonical save ran without verified replica equality",
        ));
    }
    let canonical = bool_field(checkpoint, "canonical_single_artifact")?;
    if !canonical {
        return Err(Rejection::new(
            "bad_artifact_policy",
            "the checkpoint did not save one canonical adapter",
        ));
    }
    let adapter_blake3 = string_field(receipt, "adapter_blake3")?;
    if !is_hash_hex(adapter_blake3) {
        return Err(Rejection::new(
            "malformed_receipt",
            "adapter BLAKE3 is not a hex hash",
        ));
    }
    let adapter_load_max_diff = field(checkpoint, "adapter_load_max_diff")?
        .as_f64()
        .ok_or_else(|| {
            Rejection::new("malformed_receipt", "adapter_load_max_diff is not a number")
        })?;
    if adapter_load_max_diff > MAX_ADAPTER_LOAD_DIFF {
        return Err(Rejection::new(
            "adapter_load_drift",
            format!("adapter load max diff {adapter_load_max_diff} is not byte-identical"),
        ));
    }
    let replicas_loaded = field(checkpoint, "replicas_loaded")?
        .as_u64()
        .ok_or_else(|| Rejection::new("malformed_receipt", "replicas_loaded is not an integer"))?;
    if replicas_loaded != REQUIRED_REPLICAS {
        return Err(Rejection::new(
            "replica_divergence",
            format!("resume loaded {replicas_loaded} replicas, expected {REQUIRED_REPLICAS}"),
        ));
    }
    let post_load_equality = bool_field(checkpoint, "post_load_replica_equality")?;
    if !post_load_equality {
        return Err(Rejection::new(
            "replica_divergence",
            "loaded replicas were not equal before the resumed step",
        ));
    }

    let resume_loss_dev = field(checkpoint, "resume_loss_dev")?
        .as_f64()
        .ok_or_else(|| Rejection::new("malformed_receipt", "resume_loss_dev is not a number"))?;
    let tolerances = field(profile, "tolerances")?;
    let loss_tolerance = field(tolerances, "loss")?;
    let resumed = field(checkpoint, "final_loss_resumed")?
        .as_f64()
        .ok_or_else(|| Rejection::new("malformed_receipt", "final_loss_resumed is not a number"))?;
    let bound = positive_number(loss_tolerance, "atol")?
        + positive_number(loss_tolerance, "rtol")? * resumed.abs();
    if resume_loss_dev > bound {
        return Err(Rejection::new(
            "resume_drift",
            format!("resume loss deviation {resume_loss_dev} exceeds the bound {bound}"),
        ));
    }

    Ok(CheckpointAdmission {
        identity_blake3: blake3_identity(receipt),
        adapter_load_max_diff,
        resume_loss_dev,
    })
}

fn minimal_valid_profile() -> Value {
    json!({
        "schema_version": SCHEMA_VERSION,
        "profile_id": "fixture-dual-profile",
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
            "physical_device_ids": REQUIRED_PHYSICAL_IDS,
            "mesh_descriptor": REQUIRED_MESH_DESCRIPTOR,
            "logical_mesh": { "rows": REQUIRED_LOGICAL_ROWS, "columns": REQUIRED_LOGICAL_COLUMNS },
            "container_device_isolation": REQUIRED_ISOLATION_STATE,
        },
        "data_axis": {
            "name": REQUIRED_AXIS_NAME,
            "replicas": REQUIRED_REPLICAS,
            "orientation": REQUIRED_AXIS_ORIENTATION,
        },
        "shard_policy": { "kind": REQUIRED_SHARD_KIND, "allow_uneven": false },
        "reduction_policy": {
            "gradient_op": REQUIRED_GRADIENT_OP,
            "gradient_scope": REQUIRED_GRADIENT_SCOPE,
            "loss_reduction": REQUIRED_LOSS_REDUCTION,
            "placement": REQUIRED_COLLECTIVE_PLACEMENT,
        },
        "replica_policy": {
            "equality": REQUIRED_REPLICA_EQUALITY,
            "mechanism": REQUIRED_REPLICA_MECHANISM,
            "boundaries": REQUIRED_REPLICA_BOUNDARIES,
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
            "clip_coefficient": { "rtol": 0.05, "atol": 0.01 },
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
            "parity_pass_fraction": 0.9,
            "require_finite": true,
        },
        "health_limits": {
            "max_board_temperature_c": 90,
            "max_power_watts": 180,
            "require_dram_ok": true,
        },
        "timeouts": {
            "hardware_attempt_seconds": 3600,
            "multi_step_run_seconds": 3600,
            "collective_timeout_seconds": 600,
        },
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
            "canonical": REQUIRED_ARTIFACT_CANONICAL,
            "artifact_root": "/var/lib/tenstorrent-training/artifacts/rwkv7-lora-dual-p150a",
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
            .map(|stage| json!({ "name": stage, "status": PASS_STATUS }))
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

fn minimal_batch() -> Value {
    json!({
        "global_batch_index": 0,
        "samples": [
            { "sample_id": "sample-0", "active_labels": 64 },
            { "sample_id": "sample-1", "active_labels": 64 },
        ],
    })
}

fn minimal_shard_plan(profile: &Value) -> Value {
    let admitted = admit_profile(profile).expect("fixture profile admission");
    json!({
        "identity_blake3": "f".repeat(HEX_HASH_LEN),
        "shard_kind": REQUIRED_SHARD_KIND,
        "profile_identity": admitted.identity_blake3,
        "replicas": [
            { "ordinal": 0, "sample_ids": ["sample-0"], "active_tokens": 64 },
            { "ordinal": 1, "sample_ids": ["sample-1"], "active_tokens": 64 },
        ],
    })
}

fn minimal_dual_evidence(profile: &Value) -> Value {
    json!({
        "placement": {
            "device": DEVICE_TYPE_TT,
            "mesh": { "rows": 1, "columns": 2 },
            "visible_count": 2,
            "stablehlo_op_count": 25594,
            "input_spec": "{devices=[2,1]0,1}",
        },
        "shard_plan": minimal_shard_plan(profile),
        "optimizer": {
            "gradient_op": REQUIRED_GRADIENT_OP,
            "placement": "TT",
            "reduce_before_clip": true,
            "grads_replicated": true,
            "base_unchanged": true,
        },
        "replicas": {
            "equality": REQUIRED_REPLICA_EQUALITY,
            "mechanism": REQUIRED_REPLICA_MECHANISM,
            "verified": true,
        },
    })
}

fn dual_stages() -> Vec<Value> {
    DUAL_RUN_STAGES
        .iter()
        .map(|stage| json!({ "name": stage, "status": PASS_STATUS }))
        .collect()
}

fn minimal_step_receipt(profile: &Value) -> Value {
    let admitted = admit_profile(profile).expect("fixture profile admission");
    let mut receipt = json!({
        "schema_version": SCHEMA_VERSION,
        "kind": STEP_RECEIPT_KIND,
        "profile_identity": admitted.identity_blake3,
        "stages": dual_stages(),
        "step": {
            "finite": true,
            "loss": 9.125, "reference_loss": 9.1263, "loss_dev": 0.0013,
            "clip_norm": 3.78, "reference_clip_norm": 3.68, "clip_dev": 0.098,
            "grad_pass_fraction": 1.0, "delta_pass_fraction": 0.99,
        },
    });
    let evidence = minimal_dual_evidence(profile);
    receipt
        .as_object_mut()
        .expect("receipt object")
        .extend(evidence.as_object().expect("evidence object").clone());
    receipt
}

fn minimal_run_receipt(profile: &Value) -> Value {
    let admitted = admit_profile(profile).expect("fixture profile admission");
    json!({
        "schema_version": SCHEMA_VERSION,
        "kind": RUN_RECEIPT_KIND,
        "profile_identity": admitted.identity_blake3,
        "stages": dual_stages(),
        "shard_plan": minimal_shard_plan(profile),
        "step_identity": "8".repeat(HEX_HASH_LEN),
        "run": {
            "steps": 8,
            "finite": true,
            "base_unchanged": true,
            "improved": true,
            "initial_loss": 13.19,
            "final_loss": 0.0,
            "improvement_ratio": 0.9,
            "gradient_reductions": 8,
            "per_device": [
                { "ordinal": 0, "steps": 8, "active_tokens": 512 },
                { "ordinal": 1, "steps": 8, "active_tokens": 512 },
            ],
            "replica_boundaries": (0..8)
                .map(|step| json!({ "step": step, "equal": true }))
                .collect::<Vec<_>>(),
        },
    })
}

fn minimal_checkpoint_receipt(profile: &Value) -> Value {
    let admitted = admit_profile(profile).expect("fixture profile admission");
    json!({
        "schema_version": SCHEMA_VERSION,
        "kind": CHECKPOINT_RECEIPT_KIND,
        "profile_identity": admitted.identity_blake3,
        "stages": dual_stages(),
        "adapter_blake3": "9".repeat(HEX_HASH_LEN),
        "checkpoint": {
            "replica_equality_before_save": true,
            "canonical_single_artifact": true,
            "adapter_load_max_diff": 0.0,
            "replicas_loaded": 2,
            "post_load_replica_equality": true,
            "final_loss_interrupted": 0.039,
            "final_loss_resumed": 0.039,
            "resume_loss_dev": 0.0,
        },
    })
}

fn minimal_collective_receipt(profile: &Value) -> Value {
    let admitted = admit_profile(profile).expect("fixture profile admission");
    json!({
        "schema_version": SCHEMA_VERSION,
        "kind": COLLECTIVE_RECEIPT_KIND,
        "profile_identity": admitted.identity_blake3,
        "stages": COLLECTIVE_RECEIPT_STAGES
            .iter()
            .map(|stage| json!({ "name": stage, "status": PASS_STATUS }))
            .collect::<Vec<_>>(),
        "probe_result": {
            "status": PASS_STATUS,
            "mesh": {
                "rows": 1,
                "columns": 2,
                "axis_name": REQUIRED_AXIS_NAME,
                "device_count": 2,
            },
            "all_reduce": {
                "op": REQUIRED_GRADIENT_OP,
                "placement": "TT",
                "count": 4,
                "both_contributed": true,
                "values_exact": true,
                "contribution_spec": "{devices=[2]<=[2]}",
                "result_spec": "{replicated}",
            },
            "barrier": { "synchronized": true },
            "finite": true,
            "collective_seconds": 12.5,
        },
    })
}

/// Built-in positive and negative admission fixtures.
pub fn self_test() -> Result<String, Rejection> {
    // Positive: profile admission is deterministic.
    let profile = minimal_valid_profile();
    let first = admit_profile(&profile)?;
    let second = admit_profile(&profile)?;
    if first != second {
        return Err(Rejection::new(
            "self_test",
            "profile admission is not deterministic",
        ));
    }
    if first.global_batch != first.micro_batch * REQUIRED_REPLICAS {
        return Err(Rejection::new(
            "self_test",
            "global batch derivation is wrong",
        ));
    }

    // Negative: a 1x1 logical mesh fails as a device mismatch.
    let mut wrong_mesh = profile.clone();
    wrong_mesh["device"]["logical_mesh"]["columns"] = json!(1);
    match admit_profile(&wrong_mesh) {
        Err(rejection) if rejection.code == "device_mismatch" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("1x1 mesh produced {other:?}"),
            ));
        }
    }

    // Negative: a data axis with four replicas fails.
    let mut bad_axis = profile.clone();
    bad_axis["data_axis"]["replicas"] = json!(4);
    match admit_profile(&bad_axis) {
        Err(rejection) if rejection.code == "invalid_axis" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("four replicas produced {other:?}"),
            ));
        }
    }

    // Negative: uneven shards are rejected.
    let mut uneven = profile.clone();
    uneven["shard_policy"]["allow_uneven"] = json!(true);
    match admit_profile(&uneven) {
        Err(rejection) if rejection.code == "invalid_shard_policy" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("uneven shards produced {other:?}"),
            ));
        }
    }

    // Negative: a sum gradient reduction is rejected.
    let mut sum_reduction = profile.clone();
    sum_reduction["reduction_policy"]["gradient_op"] = json!("sum");
    match admit_profile(&sum_reduction) {
        Err(rejection) if rejection.code == "wrong_reduction" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("sum reduction produced {other:?}"),
            ));
        }
    }

    // Negative: host-placed collectives are rejected as forbidden fallback.
    let mut host_collective = profile.clone();
    host_collective["reduction_policy"]["placement"] = json!("host");
    match admit_profile(&host_collective) {
        Err(rejection) if rejection.code == "host_fallback" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("host collective produced {other:?}"),
            ));
        }
    }

    // Negative: tolerance-based replica equality is rejected.
    let mut tolerance_replicas = profile.clone();
    tolerance_replicas["replica_policy"]["equality"] = json!("tolerance");
    match admit_profile(&tolerance_replicas) {
        Err(rejection) if rejection.code == "invalid_replica_policy" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("tolerance replicas produced {other:?}"),
            ));
        }
    }

    // Negative: wrong physical device IDs are rejected.
    let mut wrong_ids = profile.clone();
    wrong_ids["device"]["physical_device_ids"] = json!([0, 2]);
    match admit_profile(&wrong_ids) {
        Err(rejection) if rejection.code == "device_mismatch" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("device IDs produced {other:?}"),
            ));
        }
    }

    // Positive: shard planning covers every sample exactly once.
    let batch = minimal_batch();
    let plan = plan_shards(&profile, &batch)?;
    if plan.replicas.len() != REQUIRED_REPLICAS as usize {
        return Err(Rejection::new(
            "self_test",
            "shard plan has the wrong replica count",
        ));
    }
    let total: u64 = plan.replicas.iter().map(|shard| shard.active_tokens).sum();
    if total != 128 {
        return Err(Rejection::new("self_test", "shard plan lost active tokens"));
    }
    let again = plan_shards(&profile, &batch)?;
    if plan != again {
        return Err(Rejection::new(
            "self_test",
            "shard planning is not deterministic",
        ));
    }

    // Negative: a three-sample batch cannot shard evenly over two replicas.
    let mut odd_batch = batch.clone();
    odd_batch["samples"]
        .as_array_mut()
        .expect("samples array")
        .push(json!({ "sample_id": "sample-2", "active_labels": 64 }));
    match plan_shards(&profile, &odd_batch) {
        Err(rejection) if rejection.code == "uneven_batch" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("odd batch produced {other:?}"),
            ));
        }
    }

    // Negative: a duplicate sample is rejected.
    let mut duplicate_batch = batch.clone();
    duplicate_batch["samples"][1]["sample_id"] = json!("sample-0");
    match plan_shards(&profile, &duplicate_batch) {
        Err(rejection) if rejection.code == "duplicate_sample" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("duplicate sample produced {other:?}"),
            ));
        }
    }

    // Negative: a sample below the active-label floor is rejected.
    let mut inactive_batch = batch.clone();
    inactive_batch["samples"][0]["active_labels"] = json!(0);
    match plan_shards(&profile, &inactive_batch) {
        Err(rejection)
            if rejection.code == "dynamic_shape" || rejection.code == "inactive_sample" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("inactive sample produced {other:?}"),
            ));
        }
    }

    // Positive: prerequisite admission binds the full authority chain. The
    // single-device receipts reuse the single-profile admission through the
    // single core, so the fixture needs a valid single profile and receipts.
    let platform_receipt = minimal_platform_receipt();
    let platform_admission = json!({ "status": PLATFORM_ADMITTED_STATUS });
    let cpu_receipt = minimal_cpu_receipt();
    let cpu_receipt_blake3 = "d".repeat(HEX_HASH_LEN);
    let single_profile = single_fixture_profile();
    let single_step = single_step_receipt(&single_profile);
    let single_run = single_run_receipt(&single_profile);
    let single_checkpoint = single_checkpoint_receipt(&single_profile);
    let first_admission = admit_prerequisites(&PrerequisiteInputs {
        profile: &profile,
        platform_admission: &platform_admission,
        platform_receipt: &platform_receipt,
        cpu_receipt: &cpu_receipt,
        cpu_receipt_blake3: &cpu_receipt_blake3,
        single_profile: &single_profile,
        single_step_receipt: &single_step,
        single_run_receipt: &single_run,
        single_checkpoint_receipt: &single_checkpoint,
    })?;
    let second_admission = admit_prerequisites(&PrerequisiteInputs {
        profile: &profile,
        platform_admission: &platform_admission,
        platform_receipt: &platform_receipt,
        cpu_receipt: &cpu_receipt,
        cpu_receipt_blake3: &cpu_receipt_blake3,
        single_profile: &single_profile,
        single_step_receipt: &single_step,
        single_run_receipt: &single_run,
        single_checkpoint_receipt: &single_checkpoint,
    })?;
    if first_admission != second_admission {
        return Err(Rejection::new(
            "self_test",
            "prerequisite admission is not deterministic",
        ));
    }

    // Negative: a stale platform admission blocks.
    let stale_admission = json!({ "status": "rejected" });
    match admit_prerequisites(&PrerequisiteInputs {
        profile: &profile,
        platform_admission: &stale_admission,
        platform_receipt: &platform_receipt,
        cpu_receipt: &cpu_receipt,
        cpu_receipt_blake3: &cpu_receipt_blake3,
        single_profile: &single_profile,
        single_step_receipt: &single_step,
        single_run_receipt: &single_run,
        single_checkpoint_receipt: &single_checkpoint,
    }) {
        Err(rejection) if rejection.code == "stale_prerequisite" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("stale platform admission produced {other:?}"),
            ));
        }
    }

    // Negative: a failing single-device step receipt blocks dual launch.
    let mut failing_single_step = single_step.clone();
    failing_single_step["step"]["finite"] = json!(false);
    match admit_prerequisites(&PrerequisiteInputs {
        profile: &profile,
        platform_admission: &platform_admission,
        platform_receipt: &platform_receipt,
        cpu_receipt: &cpu_receipt,
        cpu_receipt_blake3: &cpu_receipt_blake3,
        single_profile: &single_profile,
        single_step_receipt: &failing_single_step,
        single_run_receipt: &single_run,
        single_checkpoint_receipt: &single_checkpoint,
    }) {
        Err(rejection) if rejection.code == "stale_prerequisite" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("failing single-device receipt produced {other:?}"),
            ));
        }
    }

    // Positive: an admitted collective receipt validates deterministically.
    let collective_receipt = minimal_collective_receipt(&profile);
    let first_collective = admit_collective(&collective_receipt, &profile)?;
    let second_collective = admit_collective(&collective_receipt, &profile)?;
    if first_collective != second_collective {
        return Err(Rejection::new(
            "self_test",
            "collective admission is not deterministic",
        ));
    }

    // Negative: a collective without both device contributions is rejected.
    let mut one_sided = collective_receipt.clone();
    one_sided["probe_result"]["all_reduce"]["both_contributed"] = json!(false);
    match admit_collective(&one_sided, &profile) {
        Err(rejection) if rejection.code == "collective_contribution" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("one-sided collective produced {other:?}"),
            ));
        }
    }

    // Negative: a host-placed collective is rejected.
    let mut host_reduce = collective_receipt.clone();
    host_reduce["probe_result"]["all_reduce"]["placement"] = json!("CPU");
    match admit_collective(&host_reduce, &profile) {
        Err(rejection) if rejection.code == "host_fallback" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("host reduce produced {other:?}"),
            ));
        }
    }

    // Negative: a collective with a replicated-only input is host fallback.
    let mut unsharded = collective_receipt.clone();
    unsharded["probe_result"]["all_reduce"]["contribution_spec"] = json!("{replicated}");
    match admit_collective(&unsharded, &profile) {
        Err(rejection) if rejection.code == "host_fallback" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("unsharded collective produced {other:?}"),
            ));
        }
    }

    // Negative: a timed-out collective is rejected.
    let mut slow_collective = collective_receipt.clone();
    slow_collective["probe_result"]["collective_seconds"] = json!(99999.0);
    match admit_collective(&slow_collective, &profile) {
        Err(rejection) if rejection.code == "collective_timeout" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("slow collective produced {other:?}"),
            ));
        }
    }

    // Positive: an admitted dual step receipt validates deterministically.
    let step_receipt = minimal_step_receipt(&profile);
    let first_step = admit_step_receipt(&step_receipt, &profile)?;
    let second_step = admit_step_receipt(&step_receipt, &profile)?;
    if first_step != second_step {
        return Err(Rejection::new(
            "self_test",
            "step admission is not deterministic",
        ));
    }

    // Negative: a stale step profile identity blocks admission.
    let mut stale_step = step_receipt.clone();
    stale_step["profile_identity"] = json!("0".repeat(HEX_HASH_LEN));
    match admit_step_receipt(&stale_step, &profile) {
        Err(rejection) if rejection.code == "stale_step_receipt" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("stale step produced {other:?}"),
            ));
        }
    }

    // Negative: a step whose batch input is not sharded over both devices
    // is rejected as host fallback.
    let mut no_reduce = step_receipt.clone();
    no_reduce["placement"]["input_spec"] = json!("{replicated}");
    match admit_step_receipt(&no_reduce, &profile) {
        Err(rejection) if rejection.code == "host_fallback" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("reduce-free step produced {other:?}"),
            ));
        }
    }

    // Negative: a step with an empty shard is rejected.
    let mut empty_shard = step_receipt.clone();
    empty_shard["shard_plan"]["replicas"][1]["active_tokens"] = json!(0);
    match admit_step_receipt(&empty_shard, &profile) {
        Err(rejection) if rejection.code == "missing_work" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("empty shard produced {other:?}"),
            ));
        }
    }

    // Negative: a step that clipped before reducing is rejected.
    let mut wrong_order = step_receipt.clone();
    wrong_order["optimizer"]["reduce_before_clip"] = json!(false);
    match admit_step_receipt(&wrong_order, &profile) {
        Err(rejection) if rejection.code == "wrong_order" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("clip-first step produced {other:?}"),
            ));
        }
    }

    // Negative: a step with divergent replicas is rejected.
    let mut diverged = step_receipt.clone();
    diverged["replicas"]["verified"] = json!(false);
    match admit_step_receipt(&diverged, &profile) {
        Err(rejection) if rejection.code == "replica_divergence" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("diverged step produced {other:?}"),
            ));
        }
    }

    // Negative: a step with a low delta pass fraction is rejected.
    let mut low_delta = step_receipt.clone();
    low_delta["step"]["delta_pass_fraction"] = json!(0.4);
    match admit_step_receipt(&low_delta, &profile) {
        Err(rejection) if rejection.code == "tolerance_failure" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("low delta produced {other:?}"),
            ));
        }
    }

    // Positive: an admitted dual run receipt validates deterministically.
    let run_receipt = minimal_run_receipt(&profile);
    let first_run = admit_run_receipt(&run_receipt, &profile)?;
    let second_run = admit_run_receipt(&run_receipt, &profile)?;
    if first_run != second_run {
        return Err(Rejection::new(
            "self_test",
            "run admission is not deterministic",
        ));
    }

    // Negative: a run where one replica did no steps is rejected.
    let mut idle_replica = run_receipt.clone();
    idle_replica["run"]["per_device"][1]["steps"] = json!(0);
    match admit_run_receipt(&idle_replica, &profile) {
        Err(rejection) if rejection.code == "missing_work" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("idle replica produced {other:?}"),
            ));
        }
    }

    // Negative: a run with unsynchronized gradients is rejected.
    let mut unsynced = run_receipt.clone();
    unsynced["run"]["gradient_reductions"] = json!(4);
    match admit_run_receipt(&unsynced, &profile) {
        Err(rejection) if rejection.code == "wrong_reduction" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("unsynced run produced {other:?}"),
            ));
        }
    }

    // Negative: a run with a diverged boundary is rejected.
    let mut diverged_run = run_receipt.clone();
    diverged_run["run"]["replica_boundaries"][3]["equal"] = json!(false);
    match admit_run_receipt(&diverged_run, &profile) {
        Err(rejection) if rejection.code == "replica_divergence" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("diverged run produced {other:?}"),
            ));
        }
    }

    // Negative: a run with a failed restore stage is rejected.
    let mut restore_failed = run_receipt.clone();
    restore_failed["stages"][4]["status"] = json!("fail");
    match admit_run_receipt(&restore_failed, &profile) {
        Err(rejection) if rejection.code == "failing_stage" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("restore failure produced {other:?}"),
            ));
        }
    }

    // Positive: an admitted checkpoint receipt validates deterministically.
    let checkpoint_receipt = minimal_checkpoint_receipt(&profile);
    let first_ckpt = admit_checkpoint_receipt(&checkpoint_receipt, &profile)?;
    let second_ckpt = admit_checkpoint_receipt(&checkpoint_receipt, &profile)?;
    if first_ckpt != second_ckpt {
        return Err(Rejection::new(
            "self_test",
            "checkpoint admission is not deterministic",
        ));
    }

    // Negative: a canonical save without replica equality is rejected.
    let mut unequal_save = checkpoint_receipt.clone();
    unequal_save["checkpoint"]["replica_equality_before_save"] = json!(false);
    match admit_checkpoint_receipt(&unequal_save, &profile) {
        Err(rejection) if rejection.code == "replica_divergence" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("unequal save produced {other:?}"),
            ));
        }
    }

    // Negative: a resume that loaded only one replica is rejected.
    let mut half_loaded = checkpoint_receipt.clone();
    half_loaded["checkpoint"]["replicas_loaded"] = json!(1);
    match admit_checkpoint_receipt(&half_loaded, &profile) {
        Err(rejection) if rejection.code == "replica_divergence" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("half-loaded resume produced {other:?}"),
            ));
        }
    }

    // Negative: adapter load drift is rejected.
    let mut drifted = checkpoint_receipt.clone();
    drifted["checkpoint"]["adapter_load_max_diff"] = json!(0.5);
    match admit_checkpoint_receipt(&drifted, &profile) {
        Err(rejection) if rejection.code == "adapter_load_drift" => {}
        other => {
            return Err(Rejection::new(
                "self_test",
                format!("adapter drift produced {other:?}"),
            ));
        }
    }

    Ok("self-test: PASS".to_string())
}

// Single-P150A fixtures mirror the single core's self-test receipts so the
// prerequisite chain re-validates through the single admission core.
fn single_fixture_profile() -> Value {
    let mut single = minimal_valid_profile();
    let object = single.as_object_mut().expect("profile object");
    object.insert("profile_id".to_string(), json!("fixture-single-profile"));
    let device = object
        .get_mut("device")
        .and_then(Value::as_object_mut)
        .expect("device object");
    device.insert(
        "logical_mesh".to_string(),
        json!({ "rows": 1, "columns": 1 }),
    );
    device.insert("selection_ordinal".to_string(), json!(0));
    device.remove("physical_device_ids");
    device.remove("mesh_descriptor");
    object.remove("data_axis");
    object.remove("shard_policy");
    object.remove("reduction_policy");
    object.remove("replica_policy");
    let timeouts = object
        .get_mut("timeouts")
        .and_then(Value::as_object_mut)
        .expect("timeouts object");
    timeouts.remove("collective_timeout_seconds");
    let artifact = object
        .get_mut("artifact_policy")
        .and_then(Value::as_object_mut)
        .expect("artifact object");
    artifact.remove("canonical");
    artifact.insert(
        "artifact_root".to_string(),
        json!("/var/lib/tenstorrent-training/artifacts/rwkv7-lora-p150a"),
    );
    let evidence = object
        .get_mut("evidence_policy")
        .and_then(Value::as_object_mut)
        .expect("evidence object");
    evidence.insert(
        "required_receipts".to_string(),
        json!([
            "prerequisite",
            "compile_probe",
            "placement",
            "parity",
            "optimizer",
            "run",
            "artifact",
            "health",
            "restoration",
        ]),
    );
    evidence.insert(
        "non_claims".to_string(),
        json!([
            "no dual-card scaling claim",
            "no useful generalization claim",
            "no speedup claim",
            "no support claim for other RWKV models",
        ]),
    );
    single
}

fn single_stages() -> Vec<Value> {
    [
        "health_before",
        "ownership",
        "step_execute",
        "restore",
        "health_after",
    ]
    .iter()
    .map(|stage| json!({ "name": stage, "status": PASS_STATUS }))
    .collect()
}

fn single_step_receipt(single_profile: &Value) -> Value {
    let admitted =
        rwkv7_lora_p150a_core::admit_profile(single_profile).expect("single fixture profile");
    json!({
        "schema_version": SCHEMA_VERSION,
        "kind": "rwkv7_lora_p150a_step",
        "profile_identity": admitted.identity_blake3,
        "stages": single_stages(),
        "placement": {
            "device": "TT", "ordinal": 0, "visible_count": 2,
            "stablehlo_op_count": 25594,
        },
        "step": {
            "finite": true,
            "loss": 9.125, "cpu_loss": 9.1263, "loss_dev": 0.0013,
            "clip_norm": 3.78, "cpu_clip_norm": 3.68, "clip_dev": 0.098,
            "grad_pass_fraction": 1.0, "delta_pass_fraction": 0.99,
        },
    })
}

fn single_run_receipt(single_profile: &Value) -> Value {
    let admitted =
        rwkv7_lora_p150a_core::admit_profile(single_profile).expect("single fixture profile");
    json!({
        "schema_version": SCHEMA_VERSION,
        "kind": "rwkv7_lora_p150a_run",
        "profile_identity": admitted.identity_blake3,
        "stages": single_stages(),
        "run": {
            "steps": 8,
            "finite": true,
            "base_unchanged": true,
            "improved": true,
            "initial_loss": 13.19,
            "final_loss": 0.0,
            "improvement_ratio": 0.9,
        },
    })
}

fn single_checkpoint_receipt(single_profile: &Value) -> Value {
    let admitted =
        rwkv7_lora_p150a_core::admit_profile(single_profile).expect("single fixture profile");
    json!({
        "schema_version": SCHEMA_VERSION,
        "kind": "rwkv7_lora_p150a_checkpoint",
        "profile_identity": admitted.identity_blake3,
        "stages": single_stages(),
        "checkpoint": {
            "pre_steps": 2,
            "post_steps": 2,
            "adapter_load_max_diff": 0.0,
            "final_loss_interrupted": 0.039,
            "final_loss_resumed": 0.039,
            "resume_loss_dev": 0.0,
        },
    })
}

#[cfg(test)]
mod hash_tests {
    use super::{HEX_HASH_LEN, is_hash_hex};

    const LOWER_HEX_CHARACTER: &str = "a";
    const UPPER_HEX_CHARACTER: &str = "A";

    #[test]
    fn hash_grammar_accepts_lowercase_and_rejects_uppercase() {
        let lowercase = LOWER_HEX_CHARACTER.repeat(HEX_HASH_LEN);
        let uppercase = UPPER_HEX_CHARACTER.repeat(HEX_HASH_LEN);
        assert!(is_hash_hex(&lowercase));
        assert!(!is_hash_hex(&uppercase));
    }
}
