pub mod capacity;
pub mod fused_wkv;
pub mod fused_wkv_kernel_contract;
mod generated_model_target_identity;
pub mod throughput;

use half::bf16;
use model_target_core::{TargetIdentity, TopologyIdentity};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use serde_yaml::Value as YamlValue;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Component, Path};

pub const SCHEMA_VERSION: u32 = 1;
pub const SAFETY_SCHEMA_VERSION: u32 = 3;
pub use generated_model_target_identity::{
    ARCHITECTURE_ADAPTER_ID, ARCHITECTURE_FAMILY, CONTROL_DIVERGENCE_FLOOR, EVIDENCE_ADAPTER_ID,
    FINAL_HIDDEN_TOLERANCE, LOGIT_TOLERANCE, MESH_SHAPE_JSON, MODEL_BLAKE3, MODEL_ID,
    MODEL_PRECISION, MODEL_REVISION, PHYSICAL_DEVICE_IDS_JSON, PROFILE_BLAKE3, REPLICATION,
    STATE_KIND, TARGET_NAME, TOKENIZER_VOCAB_BLAKE3, TOPOLOGY_ID,
};
pub const MODEL_ARTIFACT_BLAKE3_JSON: &str = generated_model_target_identity::ARTIFACT_BLAKE3_JSON;
pub const BUILT_PROFILE_BLAKE3: &str = env!("MODEL_TARGET_PROFILE_BLAKE3");
pub const REQUIRED_OPERATIONS_JSON: &str = env!("MODEL_TARGET_REQUIRED_OPERATIONS_JSON");
pub const EXPECTED_BOARD_SERIALS: [&str; EXPECTED_CARD_COUNT] =
    ["0000040331921011", "000004033192100e"];
pub const EXPECTED_BOARD_BDFS: [&str; EXPECTED_CARD_COUNT] = ["0000:01:00.0", "0000:03:00.0"];
pub const EXPECTED_PRIMARY_OWNER_UNIT: &str =
    "docker-tt-inference-server-llama-3-1-8b-instruct-p150.service";
pub const EXPECTED_PRIMARY_OWNER_CONTAINER: &str = "tt-inference-server-llama-3-1-8b-instruct-p150";
pub const EXPECTED_SECONDARY_OWNER_UNIT: &str =
    "llamacpp-server-vibethinker-britton-desktop.service";
pub const EXPECTED_OWNER_COUNT: usize = 2;
pub const EXPECTED_CARD_COUNT: usize = 2;
pub const EXPECTED_LAYER_COUNT: usize = generated_model_target_identity::MODEL_LAYER_COUNT;
pub const HIDDEN_ELEMENT_COUNT: usize = generated_model_target_identity::HIDDEN_SIZE;
pub const LOGIT_ELEMENT_COUNT: usize = generated_model_target_identity::VOCABULARY_SIZE;
pub const COMPLETE_STATE_ELEMENT_COUNT: usize =
    generated_model_target_identity::COMPLETE_STATE_ELEMENT_COUNT;
pub const MINIMUM_GENERATED_TOKEN_COUNT: usize = 2;
pub const MAXIMUM_GENERATED_TOKEN_COUNT: usize = 4;
pub const MAXIMUM_PROMPT_TOKEN_COUNT: usize =
    generated_model_target_identity::MAXIMUM_PROMPT_TOKENS;
pub const MAXIMUM_MESSAGE_BYTE_COUNT: usize = 256;
pub const GENERATION_EOS_TOKEN_ID: usize = 0;
pub const TOKENIZER_WRAPPER_EOS_TOKEN_ID: usize = 65_530;
pub const TOKENIZER_LAST_VOCAB_ID: usize = 65_529;
pub const TOKENIZER_VOCAB_ENTRY_COUNT: usize = 65_529;
pub const STATE_TOLERANCE: f32 = generated_model_target_identity::RECURRENT_STATE_TOLERANCE;
const F32_BYTE_WIDTH: usize = 4;
const BF16_BYTE_WIDTH: usize = 2;
const PYTHON_BYTE_HEX_DIGITS: usize = 2;
const PYTHON_SHORT_UNICODE_HEX_DIGITS: usize = 4;
const PYTHON_LONG_UNICODE_HEX_DIGITS: usize = 8;
const PYTHON_MAX_OCTAL_DIGITS: usize = 3;
const UTF8_MAX_BYTES_PER_SCALAR: usize = 4;
const HEX_RADIX: u32 = 16;
const HEX_ALPHA_DIGIT_OFFSET: u32 = 10;
const OCTAL_RADIX: u32 = 8;
const ASCII_ALERT_BYTE: u8 = 0x07;
const ASCII_BACKSPACE_BYTE: u8 = 0x08;
const ASCII_FORM_FEED_BYTE: u8 = 0x0c;
const ASCII_VERTICAL_TAB_BYTE: u8 = 0x0b;
// Generated from quality/policy.ncl. r[impl tenstorrent.quality.parser_robustness]
const JSON_MAX_BYTES: usize = 1_048_576;
const YAML_MAX_BYTES: usize = 8_388_608;
const CSV_MAX_BYTES: usize = 8_388_608;
const CSV_MAX_ROWS: usize = 100_000;
const CSV_MAX_FIELDS: usize = 256;
const VOCABULARY_MAX_BYTES: usize = 16_777_216;
const PROFILER_MINIMUM_COLUMN_COUNT: usize = 12;
const PROFILER_CARD_COLUMN: usize = 0;
const PROFILER_CYCLE_COLUMN: usize = 5;
const PROFILER_RUNTIME_ID_COLUMN: usize = 7;
const PROFILER_ZONE_COLUMN: usize = 10;
const PROFILER_PHASE_COLUMN: usize = 11;
const MESH_CARD_COORDINATE_INDEX: usize = 1;
const PRIMARY_OWNER_INDEX: usize = 0;
const SECONDARY_OWNER_INDEX: usize = 1;
const ATTEMPT3_CLASSIFICATION_SCHEMA_VERSION: u32 = 1;
const ATTEMPT3_INVOCATION_COUNT: usize = 1;
const ATTEMPT3_FAILURE_EXIT_STATUS: i32 = 1;
const ATTEMPT3_OPENED_DEVICES_DIAGNOSTIC: &str = "Opening local chip ids/PCIe ids: {0, 1}/[0, 1]";
const ATTEMPT3_PROFILER_DEFINE: &str = "-DPROFILE_KERNEL=1";
const ATTEMPT3_WATCHER_DEFINE: &str = "-DWATCHER_ENABLED";
const ATTEMPT3_OVERFLOW_DIAGNOSTIC: &str = "overflows region:0 limit of 0x2200 bytes";
const ATTEMPT3_TERMINAL_OUTCOME: &str = "restoration_failed";
const ATTEMPT3_TERMINAL_DIAGNOSTIC: &str =
    "multi-owner restoration health window expired; rollback timer remains armed";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Artifact {
    pub path: String,
    pub element_count: usize,
    pub byte_count: usize,
    pub blake3: String,
    #[serde(default)]
    pub card: Option<usize>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReferenceStep {
    pub input_token_id: usize,
    pub selected_token_id: usize,
    pub final_hidden: Artifact,
    pub logits: Artifact,
    pub complete_recurrent_state: Artifact,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReferenceControls {
    pub reset: ReferenceStep,
    pub transposed: ReferenceStep,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReferenceRequest {
    pub user_message: String,
    pub max_prompt_tokens: usize,
    pub max_new_tokens: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ReferenceManifest {
    pub schema_version: u32,
    pub arithmetic_precision: String,
    pub request: ReferenceRequest,
    pub prompt_token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub steps: Vec<ReferenceStep>,
    pub controls: ReferenceControls,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RuntimeStep {
    pub step_index: usize,
    pub input_token_id: usize,
    pub selected_token_ids_by_card: Vec<usize>,
    pub selected_token_id: usize,
    pub final_hidden: Vec<Artifact>,
    pub logits: Vec<Artifact>,
    pub recurrent_state: Vec<Artifact>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RuntimeSession {
    pub name: String,
    pub initial_recurrent_state: Vec<Artifact>,
    pub generated_token_ids: Vec<usize>,
    pub steps: Vec<RuntimeStep>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RuntimeControls {
    pub input_token_id: usize,
    pub reset: RuntimeStep,
    pub transposed: RuntimeStep,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeModel {
    pub target_name: String,
    pub id: String,
    pub revision: String,
    pub architecture_family: String,
    pub state_kind: String,
    pub precision: String,
    pub architecture_adapter: String,
    pub evidence_adapter: String,
    pub artifact_blake3: BTreeMap<String, String>,
    pub checkpoint_path: String,
    pub checkpoint_byte_count: u64,
    pub checkpoint_blake3: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeMesh {
    pub id: String,
    pub descriptor: String,
    pub shape: Vec<usize>,
    pub physical_device_ids: Vec<usize>,
    pub opened_device_ids: Vec<usize>,
    pub replication: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RuntimeRequest {
    pub prompt_token_ids: Vec<usize>,
    pub max_new_tokens: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ProgramCacheReceipt {
    pub before_cold: usize,
    pub after_cold: usize,
    pub after_warm: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct OperationEvent {
    pub ordinal: usize,
    pub stage: String,
    pub operation: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct ProfilerRange {
    pub stage: String,
    pub begin_byte: u64,
    pub end_byte: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub struct RuntimeReceipt {
    pub schema_version: u32,
    pub model: RuntimeModel,
    pub mesh: RuntimeMesh,
    pub request: RuntimeRequest,
    pub program_cache: ProgramCacheReceipt,
    pub cold: RuntimeSession,
    pub warm: RuntimeSession,
    pub controls: RuntimeControls,
    pub operation_counts: BTreeMap<String, u64>,
    pub operation_events: Vec<OperationEvent>,
    pub profiler_ranges: Vec<ProfilerRange>,
    pub host_model_arithmetic: bool,
    pub cpu_model_fallback: bool,
    pub generated_text_published: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SafetyBoard {
    pub device_id: usize,
    pub serial: String,
    pub bdf: String,
    pub healthy_before: bool,
    pub healthy_after_isolation: bool,
    pub healthy_after_restoration: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SafetyOwner {
    pub unit: String,
    pub container: Option<String>,
    pub active_before: bool,
    pub stop_requested: bool,
    pub stop_request_exit_status: i32,
    pub inactive_after_stop: bool,
    pub main_pid_zero_after_stop: bool,
    pub isolated: bool,
    pub restored_to_prior_state: bool,
    pub endpoint_healthy_after: bool,
    pub container_healthy_after: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SafetyProcess {
    pub invocation_count: usize,
    pub retry_count: usize,
    pub reconnect_count: usize,
    pub fallback_count: usize,
    pub exit_status: i32,
    pub timed_out: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SafetyRestoration {
    pub independent_timer_armed: bool,
    pub exit_restoration_attempted: bool,
    pub independent_timer_disarmed_after_health: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SafetyReceipt {
    pub schema_version: u32,
    pub host: String,
    pub attempt_lock_created: bool,
    pub boards: Vec<SafetyBoard>,
    pub owners: Vec<SafetyOwner>,
    pub device_zero_unowned: bool,
    pub device_one_unowned: bool,
    pub process: SafetyProcess,
    pub restoration: SafetyRestoration,
}

#[derive(Clone, Debug, Deserialize)]
struct Attempt3TerminalOutcome {
    diagnostic: String,
    outcome: String,
    schema_version: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct Attempt3ClassificationReceipt {
    pub schema_version: u32,
    pub outcome: String,
    pub opened_device_ids: [usize; EXPECTED_CARD_COUNT],
    pub profiler_enabled: bool,
    pub watcher_enabled: bool,
    pub brisc_firmware_overflow: bool,
    pub process_invocation_count: usize,
    pub process_exit_status: i32,
    pub model_operation_verified: bool,
    pub fresh_hardware_success: bool,
    pub restoration_outcome: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct TensorComparison {
    pub name: String,
    pub element_count: usize,
    pub tolerance: f32,
    pub card_zero_max_abs_deviation: f32,
    pub card_one_max_abs_deviation: f32,
    pub cards_bit_exact: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct StageAttribution {
    pub stage: String,
    pub operation: String,
    pub inspector_name: String,
    pub runtime_id: u64,
    pub workload_id: u64,
    pub program_ids: Vec<u64>,
    pub kernel_names: Vec<String>,
    pub profiler_rows_by_card: [usize; EXPECTED_CARD_COUNT],
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TokenizedRequest {
    pub schema_version: u32,
    pub user_message: String,
    pub rendered_chat_prompt: String,
    pub prompt_token_ids: Vec<usize>,
    pub max_prompt_tokens: usize,
    pub max_new_tokens: usize,
    pub prompt_token_ids_blake3: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ClassificationReceipt {
    pub schema_version: u32,
    pub outcome: String,
    pub target_name: String,
    pub model_id: String,
    pub model_revision: String,
    pub architecture_adapter: String,
    pub evidence_adapter: String,
    pub artifact_blake3: BTreeMap<String, String>,
    pub checkpoint_blake3: String,
    pub prompt_token_ids: Vec<usize>,
    pub generated_token_ids: Vec<usize>,
    pub generated_bytes_blake3: String,
    pub generated_text: String,
    pub tensor_comparisons: Vec<TensorComparison>,
    pub retained_vs_reset_state_divergence: f32,
    pub retained_vs_transposed_state_divergence: f32,
    pub inspector_stage_attribution: Vec<StageAttribution>,
    pub program_cache_reused: bool,
    pub card_count: usize,
    pub process_attempt_count: usize,
    pub owner_restored: bool,
    pub board_health_retained: bool,
}

#[derive(Clone, Debug)]
pub struct InspectorRuntimeEntry {
    pub workload_id: u64,
    pub runtime_id: u64,
    pub name: String,
}

#[derive(Clone, Debug, Default)]
pub struct WorkloadEvidence {
    pub program_coordinates: HashMap<u64, BTreeSet<Vec<usize>>>,
    pub statuses: BTreeSet<String>,
}

#[derive(Clone, Debug, Default)]
pub struct ProgramEvidence {
    pub statuses_by_device: HashMap<usize, BTreeSet<String>>,
    pub kernels: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct InspectorEvidence {
    pub runtime_entries: Vec<InspectorRuntimeEntry>,
    pub workloads: HashMap<u64, WorkloadEvidence>,
    pub programs: HashMap<u64, ProgramEvidence>,
    pub mesh_valid: bool,
}

#[derive(Clone, Debug)]
pub struct ProfilerRow {
    pub byte: u64,
    pub card: usize,
    pub runtime_id: u64,
    pub cycle: u64,
    pub zone: String,
    pub phase: String,
}

// r[impl tenstorrent.rwkv7_p150x2.attempt3_retention]
pub fn classify_attempt3(
    hardware_stdout: &str,
    hardware_stderr: &str,
    safety_bytes: &[u8],
    terminal_bytes: &[u8],
) -> Result<Attempt3ClassificationReceipt, String> {
    require_bounded_bytes(safety_bytes, JSON_MAX_BYTES, "attempt-3 safety receipt")?;
    require_bounded_bytes(terminal_bytes, JSON_MAX_BYTES, "attempt-3 terminal outcome")?;
    let safety: SafetyReceipt = serde_json::from_slice(safety_bytes)
        .map_err(|error| format!("invalid attempt-3 safety receipt: {error}"))?;
    let terminal: Attempt3TerminalOutcome = serde_json::from_slice(terminal_bytes)
        .map_err(|error| format!("invalid attempt-3 terminal outcome: {error}"))?;

    require(
        safety.schema_version == SAFETY_SCHEMA_VERSION,
        "attempt-3 safety schema mismatch",
    )?;
    require(
        safety.attempt_lock_created,
        "attempt-3 process budget was not consumed",
    )?;
    require(
        safety.boards.len() == EXPECTED_CARD_COUNT,
        "attempt-3 board count mismatch",
    )?;
    for (expected_device, board) in safety.boards.iter().enumerate() {
        require(
            board.device_id == expected_device
                && board.serial == EXPECTED_BOARD_SERIALS[expected_device]
                && board.bdf == EXPECTED_BOARD_BDFS[expected_device]
                && board.healthy_before
                && board.healthy_after_isolation
                && !board.healthy_after_restoration,
            "attempt-3 board identity or health boundary mismatch",
        )?;
    }
    require(
        safety.owners.len() == EXPECTED_OWNER_COUNT,
        "attempt-3 owner count mismatch",
    )?;
    require(
        safety.owners[PRIMARY_OWNER_INDEX].unit == EXPECTED_PRIMARY_OWNER_UNIT
            && safety.owners[SECONDARY_OWNER_INDEX].unit == EXPECTED_SECONDARY_OWNER_UNIT
            && safety.owners.iter().all(|owner| {
                owner.active_before
                    && owner.stop_requested
                    && owner.stop_request_exit_status == 0
                    && owner.inactive_after_stop
                    && owner.main_pid_zero_after_stop
                    && owner.isolated
                    && !owner.restored_to_prior_state
                    && !owner.endpoint_healthy_after
            }),
        "attempt-3 owner isolation or restoration evidence mismatch",
    )?;
    require(
        safety.device_zero_unowned && safety.device_one_unowned,
        "attempt-3 device ownership evidence mismatch",
    )?;
    require(
        safety.process.invocation_count == ATTEMPT3_INVOCATION_COUNT
            && safety.process.retry_count == 0
            && safety.process.reconnect_count == 0
            && safety.process.fallback_count == 0
            && safety.process.exit_status == ATTEMPT3_FAILURE_EXIT_STATUS
            && !safety.process.timed_out,
        "attempt-3 process boundary mismatch",
    )?;
    require(
        safety.restoration.independent_timer_armed
            && safety.restoration.exit_restoration_attempted
            && !safety.restoration.independent_timer_disarmed_after_health,
        "attempt-3 restoration boundary mismatch",
    )?;
    require(
        terminal.schema_version == SCHEMA_VERSION
            && terminal.outcome == ATTEMPT3_TERMINAL_OUTCOME
            && terminal.diagnostic == ATTEMPT3_TERMINAL_DIAGNOSTIC,
        "attempt-3 terminal outcome mismatch",
    )?;
    require(
        hardware_stdout.contains(ATTEMPT3_OPENED_DEVICES_DIAGNOSTIC),
        "attempt-3 output does not prove devices 0 and 1 opened",
    )?;
    let conflicting_firmware_defines = hardware_stderr.lines().any(|line| {
        line.contains(ATTEMPT3_PROFILER_DEFINE) && line.contains(ATTEMPT3_WATCHER_DEFINE)
    });
    require(
        conflicting_firmware_defines,
        "attempt-3 compile output lacks the profiler and Watcher conflict",
    )?;
    require(
        hardware_stdout.contains(ATTEMPT3_OVERFLOW_DIAGNOSTIC)
            || hardware_stderr.contains(ATTEMPT3_OVERFLOW_DIAGNOSTIC),
        "attempt-3 output lacks the BRISC firmware overflow",
    )?;

    Ok(Attempt3ClassificationReceipt {
        schema_version: ATTEMPT3_CLASSIFICATION_SCHEMA_VERSION,
        outcome: "terminal_firmware_configuration_failure".to_owned(),
        opened_device_ids: [0, 1],
        profiler_enabled: true,
        watcher_enabled: true,
        brisc_firmware_overflow: true,
        process_invocation_count: safety.process.invocation_count,
        process_exit_status: safety.process.exit_status,
        model_operation_verified: false,
        fresh_hardware_success: false,
        restoration_outcome: terminal.outcome,
    })
}

pub fn parse_reference(bytes: &[u8]) -> Result<ReferenceManifest, String> {
    require_bounded_bytes(bytes, JSON_MAX_BYTES, "reference manifest")?;
    let manifest: ReferenceManifest = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid reference manifest: {error}"))?;
    require(
        manifest.schema_version == SCHEMA_VERSION,
        "reference schema version mismatch",
    )?;
    require(
        manifest.arithmetic_precision == "CPU FP32 from BF16 checkpoint values",
        "reference arithmetic authority mismatch",
    )?;
    require(
        manifest.generated_token_ids.len() >= MINIMUM_GENERATED_TOKEN_COUNT
            && manifest.generated_token_ids.len() <= MAXIMUM_GENERATED_TOKEN_COUNT,
        "reference generated-token count is outside package bounds",
    )?;
    require(
        manifest.request.max_new_tokens == manifest.generated_token_ids.len(),
        "reference request and generated-token counts differ",
    )?;
    let expected_steps = manifest
        .prompt_token_ids
        .len()
        .checked_add(manifest.generated_token_ids.len())
        .and_then(|value| value.checked_sub(1))
        .ok_or_else(|| "reference step count overflows".to_owned())?;
    require(
        manifest.steps.len() == expected_steps,
        "reference step count mismatch",
    )?;
    require(
        manifest
            .generated_token_ids
            .iter()
            .all(|token| *token != GENERATION_EOS_TOKEN_ID),
        "reference selected generation EOS before the fixed bound",
    )?;
    Ok(manifest)
}

// r[impl tenstorrent.model_targets.single_authority]
pub fn expected_target_identity() -> Result<TargetIdentity, String> {
    if BUILT_PROFILE_BLAKE3 != PROFILE_BLAKE3 {
        return Err(format!(
            "built profile identity mismatch: generated {PROFILE_BLAKE3}, package {BUILT_PROFILE_BLAKE3}"
        ));
    }
    let artifact_blake3 = serde_json::from_str(MODEL_ARTIFACT_BLAKE3_JSON)
        .map_err(|error| format!("invalid embedded artifact identity map: {error}"))?;
    Ok(TargetIdentity {
        target_name: TARGET_NAME.to_owned(),
        model_id: MODEL_ID.to_owned(),
        revision: MODEL_REVISION.to_owned(),
        architecture_family: ARCHITECTURE_FAMILY.to_owned(),
        state_kind: STATE_KIND.to_owned(),
        precision: MODEL_PRECISION.to_owned(),
        architecture_adapter: ARCHITECTURE_ADAPTER_ID.to_owned(),
        evidence_adapter: EVIDENCE_ADAPTER_ID.to_owned(),
        artifact_blake3,
    })
}

pub fn runtime_target_identity(receipt: &RuntimeReceipt) -> TargetIdentity {
    TargetIdentity {
        target_name: receipt.model.target_name.clone(),
        model_id: receipt.model.id.clone(),
        revision: receipt.model.revision.clone(),
        architecture_family: receipt.model.architecture_family.clone(),
        state_kind: receipt.model.state_kind.clone(),
        precision: receipt.model.precision.clone(),
        architecture_adapter: receipt.model.architecture_adapter.clone(),
        evidence_adapter: receipt.model.evidence_adapter.clone(),
        artifact_blake3: receipt.model.artifact_blake3.clone(),
    }
}

pub fn expected_topology_identity() -> Result<TopologyIdentity, String> {
    let mesh_shape = serde_json::from_str(MESH_SHAPE_JSON)
        .map_err(|error| format!("invalid embedded mesh shape: {error}"))?;
    let physical_device_ids = serde_json::from_str(PHYSICAL_DEVICE_IDS_JSON)
        .map_err(|error| format!("invalid embedded physical device IDs: {error}"))?;
    Ok(TopologyIdentity {
        id: TOPOLOGY_ID.to_owned(),
        mesh_shape,
        physical_device_ids,
        replication: REPLICATION.to_owned(),
    })
}

pub fn runtime_topology_identity(receipt: &RuntimeReceipt) -> Result<TopologyIdentity, String> {
    let mesh_shape = convert_u32_values(&receipt.mesh.shape, "runtime mesh shape")?;
    let physical_device_ids = convert_u32_values(
        &receipt.mesh.physical_device_ids,
        "runtime physical device IDs",
    )?;
    Ok(TopologyIdentity {
        id: receipt.mesh.id.clone(),
        mesh_shape,
        physical_device_ids,
        replication: receipt.mesh.replication.clone(),
    })
}

pub fn required_architecture_operations() -> Result<BTreeSet<String>, String> {
    let values: Vec<String> = serde_json::from_str(REQUIRED_OPERATIONS_JSON)
        .map_err(|error| format!("invalid embedded required operation list: {error}"))?;
    let operations = values.iter().cloned().collect::<BTreeSet<_>>();
    require(
        operations.len() == values.len(),
        "embedded required operation list contains duplicates",
    )?;
    Ok(operations)
}

fn convert_u32_values(values: &[usize], name: &str) -> Result<Vec<u32>, String> {
    values
        .iter()
        .map(|value| {
            u32::try_from(*value).map_err(|error| format!("{name} value overflows u32: {error}"))
        })
        .collect()
}

// r[verify tenstorrent.rwkv7_p150x2.full_device_runtime.invalid_mesh]
pub fn parse_runtime(bytes: &[u8]) -> Result<RuntimeReceipt, String> {
    require_bounded_bytes(bytes, JSON_MAX_BYTES, "runtime receipt")?;
    let receipt: RuntimeReceipt = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid runtime receipt: {error}"))?;
    require(
        receipt.schema_version == SCHEMA_VERSION,
        "runtime schema version mismatch",
    )?;
    require(
        runtime_target_identity(&receipt) == expected_target_identity()?,
        "runtime target identity mismatch",
    )?;
    require(
        receipt.model.checkpoint_blake3 == MODEL_BLAKE3,
        "runtime checkpoint BLAKE3 mismatch",
    )?;
    require(
        runtime_topology_identity(&receipt)? == expected_topology_identity()?,
        "runtime topology identity mismatch",
    )?;
    require(
        receipt.mesh.opened_device_ids == receipt.mesh.physical_device_ids,
        "runtime opened physical device IDs mismatch",
    )?;
    require(
        !receipt.host_model_arithmetic,
        "runtime reports host model arithmetic",
    )?;
    require(
        !receipt.cpu_model_fallback,
        "runtime reports a CPU model fallback",
    )?;
    require(
        !receipt.generated_text_published,
        "runtime published text before classification",
    )?;
    require(
        receipt.program_cache.after_cold > receipt.program_cache.before_cold,
        "cold session did not populate the program cache",
    )?;
    require(
        receipt.program_cache.after_warm == receipt.program_cache.after_cold,
        "warm session did not reuse the complete cold program cache",
    )?;
    require(
        !receipt.operation_events.is_empty(),
        "runtime operation timeline is empty",
    )?;
    for (index, event) in receipt.operation_events.iter().enumerate() {
        require(
            event.ordinal == index,
            "runtime operation ordinals are not contiguous",
        )?;
    }
    Ok(receipt)
}

pub fn validate_safety(receipt: &SafetyReceipt) -> Result<(), String> {
    require(
        receipt.schema_version == SAFETY_SCHEMA_VERSION,
        "safety schema version mismatch",
    )?;
    require(
        receipt.host == "britton-desktop",
        "safety receipt host mismatch",
    )?;
    require(
        receipt.attempt_lock_created,
        "one-shot attempt lock was not created",
    )?;
    require(
        receipt.boards.len() == EXPECTED_CARD_COUNT,
        "safety receipt board count mismatch",
    )?;
    for card in 0..EXPECTED_CARD_COUNT {
        let board = &receipt.boards[card];
        require(board.device_id == card, "safety board device ID mismatch")?;
        require(
            board.serial == EXPECTED_BOARD_SERIALS[card],
            "safety board serial mismatch",
        )?;
        require(
            board.bdf == EXPECTED_BOARD_BDFS[card],
            "safety board BDF mismatch",
        )?;
        require(
            board.healthy_before
                && board.healthy_after_isolation
                && board.healthy_after_restoration,
            "board health was not retained across isolation and restoration",
        )?;
    }
    require(
        receipt.owners.len() == EXPECTED_OWNER_COUNT,
        "safety receipt owner count mismatch",
    )?;
    validate_safety_owner(
        &receipt.owners[PRIMARY_OWNER_INDEX],
        EXPECTED_PRIMARY_OWNER_UNIT,
        Some(EXPECTED_PRIMARY_OWNER_CONTAINER),
    )?;
    validate_safety_owner(
        &receipt.owners[SECONDARY_OWNER_INDEX],
        EXPECTED_SECONDARY_OWNER_UNIT,
        None,
    )?;
    require(
        receipt.device_zero_unowned && receipt.device_one_unowned,
        "both device nodes were not proven unowned",
    )?;
    require(
        receipt.process.invocation_count == 1,
        "hardware process count is not exactly one",
    )?;
    require(
        receipt.process.retry_count == 0
            && receipt.process.reconnect_count == 0
            && receipt.process.fallback_count == 0,
        "retry, reconnect, or fallback behavior was observed",
    )?;
    require(
        receipt.process.exit_status == 0,
        "hardware process returned nonzero",
    )?;
    require(!receipt.process.timed_out, "hardware process timed out")?;
    require(
        receipt.restoration.independent_timer_armed
            && receipt.restoration.exit_restoration_attempted
            && receipt.restoration.independent_timer_disarmed_after_health,
        "independent or EXIT restoration evidence is incomplete",
    )?;
    Ok(())
}

fn validate_safety_owner(
    owner: &SafetyOwner,
    expected_unit: &str,
    expected_container: Option<&str>,
) -> Result<(), String> {
    require(owner.unit == expected_unit, "owner service mismatch")?;
    require(
        owner.container.as_deref() == expected_container,
        "owner container mismatch",
    )?;
    require(
        owner.active_before,
        "reviewed owner was not active before isolation",
    )?;
    require(
        owner.stop_requested,
        "reviewed owner stop was not requested",
    )?;
    require(
        owner.stop_request_exit_status == 0,
        "reviewed owner stop request returned nonzero",
    )?;
    require(
        owner.inactive_after_stop && owner.main_pid_zero_after_stop,
        "reviewed owner did not reach terminal zero-PID quiescence",
    )?;
    require(owner.isolated, "reviewed owner was not isolated")?;
    require(
        owner.restored_to_prior_state,
        "owner prior state was not restored",
    )?;
    require(
        owner.endpoint_healthy_after,
        "restored owner did not pass endpoint health",
    )?;
    require(
        owner.container_healthy_after == expected_container.map(|_| true),
        "restored owner container-health evidence mismatch",
    )
}

pub fn validate_relative_artifact_path(path: &str) -> Result<(), String> {
    let parsed = Path::new(path);
    require(!parsed.is_absolute(), "artifact path must be relative")?;
    let mut count = 0_usize;
    for component in parsed.components() {
        match component {
            Component::Normal(_) => count += 1,
            _ => return Err("artifact path contains a forbidden component".to_owned()),
        }
    }
    require(count == 1, "artifact path must be one filename")
}

pub fn validate_artifact_bytes(artifact: &Artifact, bytes: &[u8]) -> Result<(), String> {
    validate_relative_artifact_path(&artifact.path)?;
    require(
        bytes.len() == artifact.byte_count,
        "artifact byte count mismatch",
    )?;
    require(
        blake3::hash(bytes).to_hex().as_str() == artifact.blake3,
        "artifact BLAKE3 mismatch",
    )
}

pub fn decode_reference_f32(artifact: &Artifact, bytes: &[u8]) -> Result<Vec<f32>, String> {
    validate_artifact_bytes(artifact, bytes)?;
    require(
        bytes.len() == artifact.element_count * F32_BYTE_WIDTH,
        "reference artifact shape/byte count mismatch",
    )?;
    let mut values = Vec::with_capacity(artifact.element_count);
    for chunk in bytes.chunks_exact(F32_BYTE_WIDTH) {
        let value = f32::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| "invalid FP32 artifact chunk".to_owned())?,
        );
        require(
            value.is_finite(),
            "reference artifact contains a non-finite value",
        )?;
        values.push(value);
    }
    Ok(values)
}

pub fn decode_device_bf16(
    artifact: &Artifact,
    bytes: &[u8],
    card: usize,
) -> Result<Vec<f32>, String> {
    validate_artifact_bytes(artifact, bytes)?;
    require(
        artifact.card == Some(card),
        "device artifact card field mismatch",
    )?;
    require(
        bytes.len() == artifact.element_count * BF16_BYTE_WIDTH,
        "device artifact shape/byte count mismatch",
    )?;
    let mut values = Vec::with_capacity(artifact.element_count);
    for chunk in bytes.chunks_exact(BF16_BYTE_WIDTH) {
        let bits = u16::from_le_bytes(
            chunk
                .try_into()
                .map_err(|_| "invalid BF16 artifact chunk".to_owned())?,
        );
        let value = bf16::from_bits(bits).to_f32();
        require(
            value.is_finite(),
            "device artifact contains a non-finite value",
        )?;
        values.push(value);
    }
    Ok(values)
}

pub fn max_abs_difference(left: &[f32], right: &[f32]) -> Result<f32, String> {
    require(
        left.len() == right.len(),
        "tensor comparison length mismatch",
    )?;
    require(
        !left.is_empty(),
        "tensor comparison requires non-empty inputs",
    )?;
    let mut maximum = 0.0_f32;
    for (left_value, right_value) in left.iter().zip(right) {
        require(
            left_value.is_finite() && right_value.is_finite(),
            "tensor comparison encountered a non-finite value",
        )?;
        maximum = maximum.max((left_value - right_value).abs());
    }
    Ok(maximum)
}

pub fn compare_complete_tensor(
    name: &str,
    reference_artifact: &Artifact,
    reference_bytes: &[u8],
    device_artifacts: &[Artifact],
    device_bytes: [&[u8]; EXPECTED_CARD_COUNT],
    expected_elements: usize,
    tolerance: f32,
) -> Result<TensorComparison, String> {
    require(
        reference_artifact.element_count == expected_elements,
        "reference tensor element count mismatch",
    )?;
    require(
        device_artifacts.len() == EXPECTED_CARD_COUNT,
        "device tensor does not contain exactly two artifacts",
    )?;
    require(
        device_bytes[0] == device_bytes[1],
        "replicated card tensors are not bit-exact",
    )?;
    let reference = decode_reference_f32(reference_artifact, reference_bytes)?;
    let first = decode_device_bf16(&device_artifacts[0], device_bytes[0], 0)?;
    let second = decode_device_bf16(&device_artifacts[1], device_bytes[1], 1)?;
    require(
        first.len() == expected_elements,
        "device tensor element count mismatch",
    )?;
    let first_deviation = max_abs_difference(&reference, &first)?;
    let second_deviation = max_abs_difference(&reference, &second)?;
    require(
        first_deviation <= tolerance,
        "card zero exceeds complete-tensor tolerance",
    )?;
    require(
        second_deviation <= tolerance,
        "card one exceeds complete-tensor tolerance",
    )?;
    Ok(TensorComparison {
        name: name.to_owned(),
        element_count: expected_elements,
        tolerance,
        card_zero_max_abs_deviation: first_deviation,
        card_one_max_abs_deviation: second_deviation,
        cards_bit_exact: true,
    })
}

pub fn validate_initial_zero_state(
    artifacts: &[Artifact],
    bytes: [&[u8]; EXPECTED_CARD_COUNT],
) -> Result<(), String> {
    require(
        artifacts.len() == EXPECTED_CARD_COUNT,
        "initial state does not have two artifacts",
    )?;
    require(bytes[0] == bytes[1], "initial state differs between cards")?;
    for card in 0..EXPECTED_CARD_COUNT {
        require(
            artifacts[card].element_count == COMPLETE_STATE_ELEMENT_COUNT,
            "initial state element count mismatch",
        )?;
        validate_artifact_bytes(&artifacts[card], bytes[card])?;
        require(
            artifacts[card].card == Some(card),
            "initial state card field mismatch",
        )?;
        require(
            bytes[card]
                .chunks_exact(BF16_BYTE_WIDTH)
                .all(|chunk| chunk == [0, 0]),
            "initial recurrent state is not all-zero BF16",
        )?;
    }
    Ok(())
}

fn yaml_sequence(bytes: &[u8], label: &str) -> Result<Vec<YamlValue>, String> {
    require_bounded_bytes(bytes, YAML_MAX_BYTES, label)?;
    let value: YamlValue =
        serde_yaml::from_slice(bytes).map_err(|error| format!("invalid {label} YAML: {error}"))?;
    value
        .as_sequence()
        .cloned()
        .ok_or_else(|| format!("{label} YAML must be a sequence"))
}

fn yaml_event<'a>(value: &'a YamlValue, name: &str) -> Option<&'a serde_yaml::Mapping> {
    value
        .as_mapping()?
        .get(YamlValue::String(name.to_owned()))?
        .as_mapping()
}

fn yaml_u64(map: &serde_yaml::Mapping, key: &str) -> Option<u64> {
    map.get(YamlValue::String(key.to_owned()))?.as_u64()
}

fn yaml_string(map: &serde_yaml::Mapping, key: &str) -> Option<String> {
    map.get(YamlValue::String(key.to_owned()))?
        .as_str()
        .map(str::to_owned)
}

pub fn parse_inspector(
    mesh_bytes: &[u8],
    program_bytes: &[u8],
    workload_bytes: &[u8],
    kernel_bytes: &[u8],
) -> Result<InspectorEvidence, String> {
    let mesh_events = yaml_sequence(mesh_bytes, "Inspector mesh")?;
    let mut mesh_valid = false;
    for event in mesh_events {
        if let Some(map) = yaml_event(&event, "mesh_device_created") {
            let devices = map
                .get(YamlValue::String("devices".to_owned()))
                .and_then(YamlValue::as_sequence)
                .ok_or_else(|| "Inspector mesh creation lacks devices".to_owned())?;
            let shape = map
                .get(YamlValue::String("shape".to_owned()))
                .and_then(YamlValue::as_sequence)
                .ok_or_else(|| "Inspector mesh creation lacks shape".to_owned())?;
            let device_values = devices
                .iter()
                .filter_map(YamlValue::as_u64)
                .collect::<Vec<_>>();
            let shape_values = shape
                .iter()
                .filter_map(YamlValue::as_u64)
                .collect::<Vec<_>>();
            if device_values == [0, 1] && shape_values == [1, EXPECTED_CARD_COUNT as u64] {
                mesh_valid = true;
            }
        }
    }

    let mut programs = HashMap::<u64, ProgramEvidence>::new();
    for event in yaml_sequence(program_bytes, "Inspector programs")? {
        if let Some(map) = yaml_event(&event, "program_binary_status_change") {
            let program_id =
                yaml_u64(map, "id").ok_or_else(|| "program status lacks ID".to_owned())?;
            let device_id = yaml_u64(map, "device_id")
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| "program status lacks valid device ID".to_owned())?;
            let status = yaml_string(map, "status")
                .ok_or_else(|| "program status lacks status".to_owned())?;
            programs
                .entry(program_id)
                .or_default()
                .statuses_by_device
                .entry(device_id)
                .or_default()
                .insert(status);
        }
    }
    for event in yaml_sequence(kernel_bytes, "Inspector kernels")? {
        if let Some(map) = yaml_event(&event, "kernel") {
            let program_id =
                yaml_u64(map, "program_id").ok_or_else(|| "kernel lacks program ID".to_owned())?;
            let name = yaml_string(map, "name").ok_or_else(|| "kernel lacks name".to_owned())?;
            let path = yaml_string(map, "path").ok_or_else(|| "kernel lacks path".to_owned())?;
            require(
                !name.is_empty() && !path.is_empty(),
                "Inspector kernel name/path is empty",
            )?;
            programs
                .entry(program_id)
                .or_default()
                .kernels
                .push(format!("{name}:{path}"));
        }
    }

    let mut runtime_entries = Vec::new();
    let mut workloads = HashMap::<u64, WorkloadEvidence>::new();
    for event in yaml_sequence(workload_bytes, "Inspector mesh workloads")? {
        if let Some(map) = yaml_event(&event, "runtime_entry") {
            runtime_entries.push(InspectorRuntimeEntry {
                workload_id: yaml_u64(map, "mesh_workload_id")
                    .ok_or_else(|| "runtime entry lacks workload ID".to_owned())?,
                runtime_id: yaml_u64(map, "runtime_id")
                    .ok_or_else(|| "runtime entry lacks runtime ID".to_owned())?,
                name: yaml_string(map, "name")
                    .ok_or_else(|| "runtime entry lacks operation name".to_owned())?,
            });
        }
        if let Some(map) = yaml_event(&event, "mesh_workload_add_program") {
            let workload_id = yaml_u64(map, "mesh_workload_id")
                .ok_or_else(|| "workload program mapping lacks workload ID".to_owned())?;
            let program_id = yaml_u64(map, "program_id")
                .ok_or_else(|| "workload program mapping lacks program ID".to_owned())?;
            let coordinates = map
                .get(YamlValue::String("coordinates".to_owned()))
                .and_then(YamlValue::as_sequence)
                .ok_or_else(|| "workload program mapping lacks coordinates".to_owned())?
                .iter()
                .map(|coordinate| {
                    coordinate
                        .as_sequence()
                        .ok_or_else(|| "Inspector coordinate must be a sequence".to_owned())?
                        .iter()
                        .map(|value| {
                            value
                                .as_u64()
                                .and_then(|item| usize::try_from(item).ok())
                                .ok_or_else(|| "Inspector coordinate is invalid".to_owned())
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<BTreeSet<_>, _>>()?;
            workloads
                .entry(workload_id)
                .or_default()
                .program_coordinates
                .insert(program_id, coordinates);
        }
        if let Some(map) = yaml_event(&event, "mesh_workload_set_program_binary_status") {
            let workload_id = yaml_u64(map, "mesh_workload_id")
                .ok_or_else(|| "workload status lacks workload ID".to_owned())?;
            let status = yaml_string(map, "status")
                .ok_or_else(|| "workload status lacks status".to_owned())?;
            workloads
                .entry(workload_id)
                .or_default()
                .statuses
                .insert(status);
        }
    }
    require(
        mesh_valid,
        "Inspector did not record exact 1x2 devices 0 and 1",
    )?;
    require(
        !runtime_entries.is_empty(),
        "Inspector runtime entry log is empty",
    )?;
    Ok(InspectorEvidence {
        runtime_entries,
        workloads,
        programs,
        mesh_valid,
    })
}

pub fn parse_profiler(bytes: &[u8]) -> Result<Vec<ProfilerRow>, String> {
    require_bounded_bytes(bytes, CSV_MAX_BYTES, "profiler CSV")?;
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(true)
        .from_reader(bytes);
    let mut rows = Vec::new();
    for (row_index, record) in reader.records().enumerate() {
        if row_index >= CSV_MAX_ROWS {
            return Err(format!("profiler CSV row count exceeds {CSV_MAX_ROWS}"));
        }
        let record = record.map_err(|error| format!("invalid profiler CSV: {error}"))?;
        if record.len() > CSV_MAX_FIELDS {
            return Err(format!("profiler CSV field count exceeds {CSV_MAX_FIELDS}"));
        }
        if record.len() < PROFILER_MINIMUM_COLUMN_COUNT {
            continue;
        }
        let Some(card) = record
            .get(PROFILER_CARD_COLUMN)
            .and_then(|value| value.trim().parse::<usize>().ok())
        else {
            continue;
        };
        let Some(cycle) = record
            .get(PROFILER_CYCLE_COLUMN)
            .and_then(|value| value.trim().parse::<u64>().ok())
        else {
            continue;
        };
        let Some(runtime_id) = record
            .get(PROFILER_RUNTIME_ID_COLUMN)
            .and_then(|value| value.trim().parse::<u64>().ok())
        else {
            continue;
        };
        let byte = record
            .position()
            .map(csv::Position::byte)
            .ok_or_else(|| "profiler CSV record lacks byte position".to_owned())?;
        rows.push(ProfilerRow {
            byte,
            card,
            runtime_id,
            cycle,
            zone: record
                .get(PROFILER_ZONE_COLUMN)
                .unwrap_or_default()
                .trim()
                .to_owned(),
            phase: record
                .get(PROFILER_PHASE_COLUMN)
                .unwrap_or_default()
                .trim()
                .to_owned(),
        });
    }
    require(
        !rows.is_empty(),
        "device profiler CSV contains no data rows",
    )?;
    Ok(rows)
}

fn normalized_operation(operation: &str) -> Option<&'static str> {
    match operation {
        "embedding" => Some("embedding"),
        "layer-norm" | "layer-norm-unaffined" => Some("layernorm"),
        "matmul" => Some("matmul"),
        "argmax" => Some("argmax"),
        _ => None,
    }
}

fn normalize_name(name: &str) -> String {
    name.chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn representative_events(receipt: &RuntimeReceipt) -> Result<Vec<&OperationEvent>, String> {
    let mut representatives = Vec::new();
    for session in ["cold", "warm"] {
        let runtime_session = if session == "cold" {
            &receipt.cold
        } else {
            &receipt.warm
        };
        for step in 0..runtime_session.steps.len() {
            let step_prefix = format!("{session}-step-{step}");
            let required = [
                (format!("{step_prefix}/embedding"), "embedding"),
                (format!("{step_prefix}/final-normalization"), "layer-norm"),
                (format!("{step_prefix}/language-model-head"), "matmul"),
                (format!("{step_prefix}/greedy-selection"), "argmax"),
            ];
            for (stage, operation) in required {
                let event = receipt
                    .operation_events
                    .iter()
                    .find(|event| event.stage == stage && event.operation == operation)
                    .ok_or_else(|| format!("runtime timeline lacks {operation} in {stage}"))?;
                representatives.push(event);
            }
            for layer in 0..EXPECTED_LAYER_COUNT {
                let layer_prefix = format!("{step_prefix}/layer-{layer}");
                let required = [
                    ("pre-norm", "layer-norm"),
                    ("time-mix-preparation", "matmul"),
                    ("wkv", "matmul"),
                    ("attention-finish", "layer-norm-unaffined"),
                    ("channel-mix", "layer-norm"),
                ];
                for (suffix, operation) in required {
                    let stage = format!("{layer_prefix}/{suffix}");
                    let event = receipt
                        .operation_events
                        .iter()
                        .find(|event| event.stage == stage && event.operation == operation)
                        .ok_or_else(|| format!("runtime timeline lacks {operation} in {stage}"))?;
                    representatives.push(event);
                }
            }
        }
    }
    representatives.sort_by_key(|event| event.ordinal);
    Ok(representatives)
}

fn profiler_range_for_event<'a>(
    event: &OperationEvent,
    ranges: &'a [ProfilerRange],
) -> Option<&'a ProfilerRange> {
    if event.stage.ends_with("/embedding") {
        return ranges.iter().find(|range| range.stage == event.stage);
    }
    if let Some(layer_position) = event.stage.find("/layer-") {
        let after_layer = &event.stage[layer_position + 1..];
        let layer_segment = after_layer.split('/').next()?;
        let prefix = &event.stage[..layer_position];
        let range_stage = format!("{prefix}/{layer_segment}/complete");
        return ranges.iter().find(|range| range.stage == range_stage);
    }
    if event.stage.ends_with("/final-normalization")
        || event.stage.ends_with("/language-model-head")
        || event.stage.ends_with("/greedy-selection")
    {
        let prefix = event.stage.rsplit_once('/')?.0;
        let range_stage = format!("{prefix}/selection");
        return ranges.iter().find(|range| range.stage == range_stage);
    }
    None
}

pub fn attribute_required_stages(
    receipt: &RuntimeReceipt,
    inspector: &InspectorEvidence,
    profiler_rows: &[ProfilerRow],
    profiler_byte_count: u64,
) -> Result<Vec<StageAttribution>, String> {
    require(
        !receipt.profiler_ranges.is_empty(),
        "runtime profiler ranges are empty",
    )?;
    let mut previous_end = 0_u64;
    for range in &receipt.profiler_ranges {
        require(
            range.begin_byte == previous_end,
            "profiler byte ranges are not contiguous",
        )?;
        require(
            range.end_byte > range.begin_byte,
            "profiler byte range is empty",
        )?;
        require(
            range.end_byte <= profiler_byte_count,
            "profiler byte range exceeds raw CSV",
        )?;
        previous_end = range.end_byte;
    }

    let representatives = representative_events(receipt)?;
    let mut inspector_cursor = 0_usize;
    let mut output = Vec::with_capacity(representatives.len());
    for event in representatives {
        let needle = normalized_operation(&event.operation)
            .ok_or_else(|| format!("no Inspector mapping for {}", event.operation))?;
        let found = inspector.runtime_entries[inspector_cursor..]
            .iter()
            .position(|entry| normalize_name(&entry.name).contains(needle))
            .map(|position| inspector_cursor + position)
            .ok_or_else(|| {
                format!(
                    "Inspector lacks ordered operation {needle} for {}",
                    event.stage
                )
            })?;
        inspector_cursor = found + 1;
        let entry = &inspector.runtime_entries[found];
        let workload = inspector.workloads.get(&entry.workload_id).ok_or_else(|| {
            format!(
                "Inspector runtime {} lacks workload mapping",
                entry.runtime_id
            )
        })?;
        require(
            workload.statuses.contains("InFlight") && workload.statuses.contains("Committed"),
            "Inspector workload lacks InFlight/Committed lifecycle",
        )?;
        let required_coordinates = BTreeSet::from([vec![0, 0], vec![0, 1]]);
        let mut observed_coordinates = BTreeSet::new();
        let mut program_ids = Vec::new();
        let mut kernel_names = Vec::new();
        for (program_id, coordinates) in &workload.program_coordinates {
            require(
                !coordinates.is_empty(),
                "Inspector workload program has no mesh coordinates",
            )?;
            require(
                coordinates
                    .iter()
                    .all(|coordinate| required_coordinates.contains(coordinate)),
                "Inspector workload program contains an unexpected mesh coordinate",
            )?;
            let program = inspector
                .programs
                .get(program_id)
                .ok_or_else(|| format!("Inspector lacks program {program_id}"))?;
            for coordinate in coordinates {
                let card = *coordinate
                    .get(MESH_CARD_COORDINATE_INDEX)
                    .ok_or_else(|| "Inspector mesh coordinate lacks a card column".to_owned())?;
                let statuses = program
                    .statuses_by_device
                    .get(&card)
                    .ok_or_else(|| format!("program {program_id} lacks card {card} lifecycle"))?;
                require(
                    statuses.contains("InFlight") && statuses.contains("Committed"),
                    "Inspector program lacks per-card InFlight/Committed lifecycle",
                )?;
                observed_coordinates.insert(coordinate.clone());
            }
            require(
                !program.kernels.is_empty(),
                "Inspector program contains no named kernels",
            )?;
            program_ids.push(*program_id);
            kernel_names.extend(program.kernels.iter().cloned());
        }
        require(
            !program_ids.is_empty(),
            "Inspector workload contains no programs",
        )?;
        require(
            observed_coordinates == required_coordinates,
            "Inspector workload programs do not collectively cover both cards",
        )?;

        let range = profiler_range_for_event(event, &receipt.profiler_ranges)
            .ok_or_else(|| format!("runtime profiler ranges do not cover {}", event.stage))?;
        let mut rows_by_card = [0_usize; EXPECTED_CARD_COUNT];
        for (card, row_count) in rows_by_card.iter_mut().enumerate() {
            let rows = profiler_rows
                .iter()
                .filter(|row| {
                    row.byte >= range.begin_byte
                        && row.byte < range.end_byte
                        && row.card == card
                        && row.runtime_id == entry.runtime_id
                        && !row.zone.is_empty()
                })
                .collect::<Vec<_>>();
            require(
                !rows.is_empty(),
                "profiler lacks stage/runtime rows for one card",
            )?;
            let begins = rows
                .iter()
                .filter(|row| row.phase.eq_ignore_ascii_case("begin"))
                .map(|row| row.cycle)
                .collect::<Vec<_>>();
            let ends = rows
                .iter()
                .filter(|row| row.phase.eq_ignore_ascii_case("end"))
                .map(|row| row.cycle)
                .collect::<Vec<_>>();
            require(
                !begins.is_empty() && !ends.is_empty(),
                "profiler stage lacks begin/end samples",
            )?;
            require(
                ends.iter().max() > begins.iter().min(),
                "profiler stage timing is not positive",
            )?;
            *row_count = rows.len();
        }
        output.push(StageAttribution {
            stage: event.stage.clone(),
            operation: event.operation.clone(),
            inspector_name: entry.name.clone(),
            runtime_id: entry.runtime_id,
            workload_id: entry.workload_id,
            program_ids,
            kernel_names,
            profiler_rows_by_card: rows_by_card,
        });
    }
    Ok(output)
}

pub fn validate_runtime_against_reference(
    runtime: &RuntimeReceipt,
    reference: &ReferenceManifest,
) -> Result<(), String> {
    require(
        runtime.request.prompt_token_ids == reference.prompt_token_ids,
        "runtime prompt IDs mismatch",
    )?;
    require(
        runtime.request.max_new_tokens == reference.request.max_new_tokens,
        "runtime generation bound mismatch",
    )?;
    require(
        runtime.cold.name == "cold" && runtime.warm.name == "warm",
        "session names mismatch",
    )?;
    require(
        runtime.cold.generated_token_ids == reference.generated_token_ids
            && runtime.warm.generated_token_ids == reference.generated_token_ids,
        "cold/warm greedy IDs do not match the CPU reference",
    )?;
    require(
        runtime.cold.steps.len() == reference.steps.len(),
        "cold step count mismatch",
    )?;
    require(
        runtime.warm.steps.len() == reference.steps.len(),
        "warm step count mismatch",
    )?;
    for session in [&runtime.cold, &runtime.warm] {
        for (index, (actual, expected)) in session.steps.iter().zip(&reference.steps).enumerate() {
            require(actual.step_index == index, "runtime step index mismatch")?;
            require(
                actual.input_token_id == expected.input_token_id,
                "runtime input token mismatch",
            )?;
            require(
                actual.selected_token_id == expected.selected_token_id,
                "runtime selected token mismatch",
            )?;
            require(
                actual.selected_token_ids_by_card
                    == [expected.selected_token_id; EXPECTED_CARD_COUNT],
                "cards did not independently select the reference token",
            )?;
        }
    }
    require(
        runtime.controls.input_token_id == reference.controls.reset.input_token_id,
        "control input mismatch",
    )?;
    require(
        runtime.controls.reset.selected_token_id == reference.controls.reset.selected_token_id,
        "reset control selected token mismatch",
    )?;
    require(
        runtime.controls.transposed.selected_token_id
            == reference.controls.transposed.selected_token_id,
        "transposed control selected token mismatch",
    )?;
    Ok(())
}

pub fn validate_operation_inventory(counts: &BTreeMap<String, u64>) -> Result<(), String> {
    for operation in [
        "embedding",
        "layer-norm",
        "layer-norm-unaffined",
        "matmul",
        "add",
        "multiply",
        "subtract",
        "divide",
        "sigmoid",
        "tanh",
        "exp",
        "relu",
        "square",
        "sqrt",
        "sum-last",
        "maximum-scalar",
        "transpose",
        "argmax",
    ] {
        require(
            counts.get(operation).copied().unwrap_or_default() > 0,
            &format!("runtime operation inventory lacks {operation}"),
        )?;
    }
    Ok(())
}

#[derive(Clone, Debug, Default)]
struct TokenTrieNode {
    children: BTreeMap<u8, usize>,
    token_id: Option<usize>,
}

#[derive(Clone, Debug)]
struct BoundedTokenizer {
    trie: Vec<TokenTrieNode>,
}

impl BoundedTokenizer {
    fn parse(vocabulary: &[u8]) -> Result<Self, String> {
        require_bounded_bytes(vocabulary, VOCABULARY_MAX_BYTES, "tokenizer vocabulary")?;
        require(
            blake3::hash(vocabulary).to_hex().as_str() == TOKENIZER_VOCAB_BLAKE3,
            "tokenizer vocabulary BLAKE3 mismatch",
        )?;
        let source = std::str::from_utf8(vocabulary)
            .map_err(|error| format!("tokenizer vocabulary is not UTF-8: {error}"))?;
        let lines = source.lines().collect::<Vec<_>>();
        require(
            lines.len() == TOKENIZER_VOCAB_ENTRY_COUNT,
            "tokenizer vocabulary row count mismatch",
        )?;
        let mut trie = vec![TokenTrieNode::default()];
        let mut unique = BTreeSet::new();
        for (line_index, line) in lines.iter().enumerate() {
            let expected_id = line_index + 1;
            let (token_id, token) = parse_vocabulary_line(line)?;
            require(
                token_id == expected_id,
                "tokenizer row ID sequence mismatch",
            )?;
            require(!token.is_empty(), "tokenizer contains an empty byte token")?;
            require(
                unique.insert(token.clone()),
                "tokenizer contains duplicate byte tokens",
            )?;
            let mut node_index = 0_usize;
            for byte in token {
                let next = if let Some(existing) = trie[node_index].children.get(&byte) {
                    *existing
                } else {
                    let next = trie.len();
                    trie.push(TokenTrieNode::default());
                    trie[node_index].children.insert(byte, next);
                    next
                };
                node_index = next;
            }
            require(
                trie[node_index].token_id.replace(token_id).is_none(),
                "tokenizer trie contains a duplicate terminal",
            )?;
        }
        Ok(Self { trie })
    }

    fn encode_bytes(&self, source: &[u8]) -> Result<Vec<usize>, String> {
        let mut token_ids = Vec::new();
        let mut offset = 0_usize;
        while offset < source.len() {
            let mut node_index = 0_usize;
            let mut cursor = offset;
            let mut longest = None;
            while cursor < source.len() {
                let Some(next) = self.trie[node_index].children.get(&source[cursor]) else {
                    break;
                };
                node_index = *next;
                cursor += 1;
                if let Some(token_id) = self.trie[node_index].token_id {
                    longest = Some((cursor, token_id));
                }
            }
            let (next_offset, token_id) =
                longest.ok_or_else(|| format!("tokenizer has no byte token at offset {offset}"))?;
            require(
                next_offset > offset,
                "tokenizer longest-prefix match did not advance",
            )?;
            token_ids.push(token_id);
            offset = next_offset;
        }
        Ok(token_ids)
    }

    fn encode_wrapper_text(&self, source: &str) -> Result<Vec<usize>, String> {
        const SPECIAL_TEXT: &[u8] = b"<|rwkv_tokenizer_end_of_text|>";
        const EOS_TEXT: &[u8] = b"\n\n";
        let source = source.as_bytes();
        let mut token_ids = Vec::new();
        let mut offset = 0_usize;
        while offset < source.len() {
            if source[offset..].starts_with(SPECIAL_TEXT) {
                token_ids.push(GENERATION_EOS_TOKEN_ID);
                offset += SPECIAL_TEXT.len();
                continue;
            }
            if source[offset..].starts_with(EOS_TEXT) {
                token_ids.push(TOKENIZER_WRAPPER_EOS_TOKEN_ID);
                offset += EOS_TEXT.len();
                continue;
            }
            let mut end = offset + 1;
            while end < source.len()
                && !source[end..].starts_with(SPECIAL_TEXT)
                && !source[end..].starts_with(EOS_TEXT)
            {
                end += 1;
            }
            token_ids.extend(self.encode_bytes(&source[offset..end])?);
            offset = end;
        }
        Ok(token_ids)
    }
}

pub fn tokenize_request(
    vocabulary: &[u8],
    user_message: String,
    max_prompt_tokens: usize,
    max_new_tokens: usize,
) -> Result<TokenizedRequest, String> {
    require(
        user_message.len() <= MAXIMUM_MESSAGE_BYTE_COUNT,
        "user message exceeds the package byte cap",
    )?;
    require(
        (1..=MAXIMUM_PROMPT_TOKEN_COUNT).contains(&max_prompt_tokens),
        "prompt token bound is outside the package cap",
    )?;
    require(
        (MINIMUM_GENERATED_TOKEN_COUNT..=MAXIMUM_GENERATED_TOKEN_COUNT).contains(&max_new_tokens),
        "generation bound is outside the full-device proof range",
    )?;
    let rendered_chat_prompt =
        format!("<|rwkv_tokenizer_end_of_text|>User: {user_message}\n\nAssistant:");
    let tokenizer = BoundedTokenizer::parse(vocabulary)?;
    let prompt_token_ids = tokenizer.encode_wrapper_text(&rendered_chat_prompt)?;
    require(
        prompt_token_ids.len() <= max_prompt_tokens,
        "encoded prompt exceeds the caller token bound",
    )?;
    let mut token_bytes = Vec::with_capacity(prompt_token_ids.len() * size_of::<u32>());
    for token in &prompt_token_ids {
        let token = u32::try_from(*token).map_err(|_| "prompt token ID overflows u32")?;
        token_bytes.extend_from_slice(&token.to_le_bytes());
    }
    Ok(TokenizedRequest {
        schema_version: SCHEMA_VERSION,
        user_message,
        rendered_chat_prompt,
        prompt_token_ids,
        max_prompt_tokens,
        max_new_tokens,
        prompt_token_ids_blake3: blake3::hash(&token_bytes).to_hex().to_string(),
    })
}

pub fn decode_generated_tokens(vocabulary: &[u8], token_ids: &[usize]) -> Result<Vec<u8>, String> {
    require(
        blake3::hash(vocabulary).to_hex().as_str() == TOKENIZER_VOCAB_BLAKE3,
        "tokenizer vocabulary BLAKE3 mismatch",
    )?;
    let source = std::str::from_utf8(vocabulary)
        .map_err(|error| format!("tokenizer vocabulary is not UTF-8: {error}"))?;
    let lines = source.lines().collect::<Vec<_>>();
    require(
        lines.len() == TOKENIZER_VOCAB_ENTRY_COUNT,
        "tokenizer vocabulary row count mismatch",
    )?;
    let mut output = Vec::new();
    for token_id in token_ids {
        if *token_id == GENERATION_EOS_TOKEN_ID {
            break;
        }
        if *token_id == TOKENIZER_WRAPPER_EOS_TOKEN_ID {
            output.extend_from_slice(b"<|rwkv_tokenizer_end_of_text|>");
            continue;
        }
        require(
            (1..=TOKENIZER_LAST_VOCAB_ID).contains(token_id),
            "generated token ID is outside the decodable vocabulary",
        )?;
        let (parsed_id, bytes) = parse_vocabulary_line(lines[*token_id - 1])?;
        require(parsed_id == *token_id, "tokenizer row ID mismatch")?;
        output.extend_from_slice(&bytes);
    }
    require(!output.is_empty(), "generated token bytes are empty")?;
    Ok(output)
}

fn parse_vocabulary_line(line: &str) -> Result<(usize, Vec<u8>), String> {
    let (id_text, remainder) = line
        .split_once(' ')
        .ok_or_else(|| "tokenizer row has no ID separator".to_owned())?;
    let (literal, length_text) = remainder
        .rsplit_once(' ')
        .ok_or_else(|| "tokenizer row has no length separator".to_owned())?;
    let token_id = id_text
        .parse::<usize>()
        .map_err(|error| format!("invalid tokenizer ID: {error}"))?;
    let declared_length = length_text
        .parse::<usize>()
        .map_err(|error| format!("invalid tokenizer byte length: {error}"))?;
    let token_bytes = parse_python_bytes_literal(literal)?;
    require(
        token_bytes.len() == declared_length,
        "tokenizer declared byte length mismatch",
    )?;
    Ok((token_id, token_bytes))
}

fn parse_python_bytes_literal(literal: &str) -> Result<Vec<u8>, String> {
    let source = literal.as_bytes();
    let (bytes_mode, quote_index) = match source {
        [b'b', b'\'', ..] | [b'b', b'"', ..] => (true, 1_usize),
        [b'\'', ..] | [b'"', ..] => (false, 0_usize),
        _ => return Err("unsupported tokenizer literal".to_owned()),
    };
    let quote = source[quote_index];
    require(
        source.len() > quote_index + 1 && source.last().copied() == Some(quote),
        "unterminated tokenizer literal",
    )?;
    let end = source.len() - 1;
    let mut output = Vec::new();
    let mut index = quote_index + 1;
    while index < end {
        let byte = source[index];
        if byte != b'\\' {
            require(byte != quote, "unescaped tokenizer quote")?;
            require(
                !bytes_mode || byte.is_ascii(),
                "non-ASCII byte literal source",
            )?;
            output.push(byte);
            index += 1;
            continue;
        }
        index += 1;
        require(index < end, "trailing tokenizer escape")?;
        let escaped = source[index];
        index += 1;
        match escaped {
            b'\\' => output.push(b'\\'),
            b'\'' => output.push(b'\''),
            b'"' => output.push(b'"'),
            b'a' => output.push(ASCII_ALERT_BYTE),
            b'b' => output.push(ASCII_BACKSPACE_BYTE),
            b'f' => output.push(ASCII_FORM_FEED_BYTE),
            b'n' => output.push(b'\n'),
            b'r' => output.push(b'\r'),
            b't' => output.push(b'\t'),
            b'v' => output.push(ASCII_VERTICAL_TAB_BYTE),
            b'x' => {
                let value = parse_fixed_hex(source, &mut index, end, PYTHON_BYTE_HEX_DIGITS)?;
                if bytes_mode {
                    output
                        .push(u8::try_from(value).map_err(|_| "hex byte escape is out of range")?);
                } else {
                    push_scalar(&mut output, value)?;
                }
            }
            b'u' | b'U' => {
                require(!bytes_mode, "Unicode escape in tokenizer bytes literal")?;
                let digits = if escaped == b'u' {
                    PYTHON_SHORT_UNICODE_HEX_DIGITS
                } else {
                    PYTHON_LONG_UNICODE_HEX_DIGITS
                };
                let value = parse_fixed_hex(source, &mut index, end, digits)?;
                push_scalar(&mut output, value)?;
            }
            b'0'..=b'7' => {
                let mut value = u32::from(escaped - b'0');
                let mut digits = 1_usize;
                while digits < PYTHON_MAX_OCTAL_DIGITS
                    && index < end
                    && matches!(source[index], b'0'..=b'7')
                {
                    value = value * OCTAL_RADIX + u32::from(source[index] - b'0');
                    index += 1;
                    digits += 1;
                }
                if bytes_mode {
                    output.push(
                        u8::try_from(value).map_err(|_| "octal byte escape is out of range")?,
                    );
                } else {
                    push_scalar(&mut output, value)?;
                }
            }
            _ => return Err("unsupported tokenizer escape".to_owned()),
        }
    }
    Ok(output)
}

fn push_scalar(output: &mut Vec<u8>, value: u32) -> Result<(), String> {
    let character = char::from_u32(value).ok_or_else(|| "invalid Unicode scalar".to_owned())?;
    let mut encoded = [0_u8; UTF8_MAX_BYTES_PER_SCALAR];
    output.extend_from_slice(character.encode_utf8(&mut encoded).as_bytes());
    Ok(())
}

fn parse_fixed_hex(
    source: &[u8],
    index: &mut usize,
    end: usize,
    digit_count: usize,
) -> Result<u32, String> {
    let next = index
        .checked_add(digit_count)
        .ok_or_else(|| "hex escape index overflows".to_owned())?;
    require(next <= end, "short tokenizer hex escape")?;
    let mut value = 0_u32;
    for digit in &source[*index..next] {
        let digit_value = match digit {
            b'0'..=b'9' => u32::from(*digit - b'0'),
            b'a'..=b'f' => u32::from(*digit - b'a') + HEX_ALPHA_DIGIT_OFFSET,
            b'A'..=b'F' => u32::from(*digit - b'A') + HEX_ALPHA_DIGIT_OFFSET,
            _ => return Err("invalid tokenizer hex escape".to_owned()),
        };
        value = value
            .checked_mul(HEX_RADIX)
            .and_then(|partial| partial.checked_add(digit_value))
            .ok_or_else(|| "tokenizer hex escape overflows".to_owned())?;
    }
    *index = next;
    Ok(value)
}

pub fn parse_json_value(bytes: &[u8], label: &str) -> Result<JsonValue, String> {
    require_bounded_bytes(bytes, JSON_MAX_BYTES, label)?;
    serde_json::from_slice(bytes).map_err(|error| format!("invalid {label} JSON: {error}"))
}

fn require_bounded_bytes(bytes: &[u8], maximum: usize, label: &str) -> Result<(), String> {
    if bytes.len() > maximum {
        return Err(format!(
            "{label} byte count {} exceeds {maximum}",
            bytes.len()
        ));
    }
    Ok(())
}

fn require(condition: bool, message: &str) -> Result<(), String> {
    if condition {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_ELEMENT_COUNT: usize = 2;
    const TEST_TOLERANCE: f32 = 0.01;
    const TEST_PROFILER_RUNTIME_ID: u64 = 7;
    const TEST_CHECKPOINT_BYTE_COUNT: u64 = 1;

    fn artifact(path: &str, bytes: &[u8], elements: usize, card: Option<usize>) -> Artifact {
        Artifact {
            path: path.to_owned(),
            element_count: elements,
            byte_count: bytes.len(),
            blake3: blake3::hash(bytes).to_hex().to_string(),
            card,
        }
    }

    fn valid_safety_receipt() -> SafetyReceipt {
        let boards = (0..EXPECTED_CARD_COUNT)
            .map(|card| SafetyBoard {
                device_id: card,
                serial: EXPECTED_BOARD_SERIALS[card].to_owned(),
                bdf: EXPECTED_BOARD_BDFS[card].to_owned(),
                healthy_before: true,
                healthy_after_isolation: true,
                healthy_after_restoration: true,
            })
            .collect();
        SafetyReceipt {
            schema_version: SAFETY_SCHEMA_VERSION,
            host: "britton-desktop".to_owned(),
            attempt_lock_created: true,
            boards,
            owners: vec![
                SafetyOwner {
                    unit: EXPECTED_PRIMARY_OWNER_UNIT.to_owned(),
                    container: Some(EXPECTED_PRIMARY_OWNER_CONTAINER.to_owned()),
                    active_before: true,
                    stop_requested: true,
                    stop_request_exit_status: 0,
                    inactive_after_stop: true,
                    main_pid_zero_after_stop: true,
                    isolated: true,
                    restored_to_prior_state: true,
                    endpoint_healthy_after: true,
                    container_healthy_after: Some(true),
                },
                SafetyOwner {
                    unit: EXPECTED_SECONDARY_OWNER_UNIT.to_owned(),
                    container: None,
                    active_before: true,
                    stop_requested: true,
                    stop_request_exit_status: 0,
                    inactive_after_stop: true,
                    main_pid_zero_after_stop: true,
                    isolated: true,
                    restored_to_prior_state: true,
                    endpoint_healthy_after: true,
                    container_healthy_after: None,
                },
            ],
            device_zero_unowned: true,
            device_one_unowned: true,
            process: SafetyProcess {
                invocation_count: 1,
                retry_count: 0,
                reconnect_count: 0,
                fallback_count: 0,
                exit_status: 0,
                timed_out: false,
            },
            restoration: SafetyRestoration {
                independent_timer_armed: true,
                exit_restoration_attempted: true,
                independent_timer_disarmed_after_health: true,
            },
        }
    }

    fn attempt3_safety_receipt() -> SafetyReceipt {
        let mut receipt = valid_safety_receipt();
        for board in &mut receipt.boards {
            board.healthy_after_restoration = false;
        }
        for owner in &mut receipt.owners {
            owner.restored_to_prior_state = false;
            owner.endpoint_healthy_after = false;
            owner.container_healthy_after = owner.container.as_ref().map(|_| false);
        }
        receipt.process.exit_status = ATTEMPT3_FAILURE_EXIT_STATUS;
        receipt.restoration.independent_timer_disarmed_after_health = false;
        receipt
    }

    fn attempt3_terminal_bytes(outcome: &str) -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "diagnostic": ATTEMPT3_TERMINAL_DIAGNOSTIC,
            "outcome": outcome,
            "schema_version": SCHEMA_VERSION,
        }))
        .expect("terminal fixture must serialize")
    }

    // r[verify tenstorrent.rwkv7_p150x2.full_device_runtime.invalid_mesh]
    #[test]
    fn runtime_parser_rejects_an_unexpected_physical_device() {
        let target = expected_target_identity().expect("embedded target identity must parse");
        let topology = expected_topology_identity().expect("embedded topology identity must parse");
        let unexpected_device_ids = vec![0, EXPECTED_CARD_COUNT];
        let receipt = serde_json::json!({
            "schema_version": SCHEMA_VERSION,
            "model": {
                "target_name": target.target_name,
                "id": target.model_id,
                "revision": target.revision,
                "architecture_family": target.architecture_family,
                "state_kind": target.state_kind,
                "precision": target.precision,
                "architecture_adapter": target.architecture_adapter,
                "evidence_adapter": target.evidence_adapter,
                "artifact_blake3": target.artifact_blake3,
                "checkpoint_path": "/nix/store/fixture-checkpoint",
                "checkpoint_byte_count": TEST_CHECKPOINT_BYTE_COUNT,
                "checkpoint_blake3": MODEL_BLAKE3,
            },
            "mesh": {
                "id": topology.id,
                "descriptor": "/nix/store/fixture-mesh-descriptor",
                "shape": topology.mesh_shape,
                "physical_device_ids": unexpected_device_ids,
                "opened_device_ids": [0, EXPECTED_CARD_COUNT],
                "replication": topology.replication,
            },
            "request": { "prompt_token_ids": [], "max_new_tokens": MINIMUM_GENERATED_TOKEN_COUNT },
            "program_cache": { "before_cold": 0, "after_cold": 1, "after_warm": 1 },
            "cold": { "name": "cold", "initial_recurrent_state": [], "generated_token_ids": [], "steps": [] },
            "warm": { "name": "warm", "initial_recurrent_state": [], "generated_token_ids": [], "steps": [] },
            "controls": {
                "input_token_id": 0,
                "reset": {
                    "step_index": 0,
                    "input_token_id": 0,
                    "selected_token_ids_by_card": [0, 0],
                    "selected_token_id": 0,
                    "final_hidden": [],
                    "logits": [],
                    "recurrent_state": [],
                },
                "transposed": {
                    "step_index": 0,
                    "input_token_id": 0,
                    "selected_token_ids_by_card": [0, 0],
                    "selected_token_id": 0,
                    "final_hidden": [],
                    "logits": [],
                    "recurrent_state": [],
                },
            },
            "operation_counts": {},
            "operation_events": [],
            "profiler_ranges": [],
            "host_model_arithmetic": false,
            "cpu_model_fallback": false,
            "generated_text_published": false,
        });
        assert!(
            parse_runtime(&serde_json::to_vec(&receipt).expect("runtime fixture must serialize"))
                .expect_err("unexpected physical device must fail")
                .contains("runtime topology identity mismatch")
        );
    }

    // r[verify tenstorrent.rwkv7_p150x2.attempt3_retention.recorded]
    #[test]
    fn attempt3_classifier_accepts_the_recorded_firmware_failure() {
        let safety =
            serde_json::to_vec(&attempt3_safety_receipt()).expect("safety fixture must serialize");
        let stdout =
            format!("{ATTEMPT3_OPENED_DEVICES_DIAGNOSTIC}\n{ATTEMPT3_OVERFLOW_DIAGNOSTIC}\n");
        let stderr = format!("compiler {ATTEMPT3_PROFILER_DEFINE} {ATTEMPT3_WATCHER_DEFINE}\n");
        let classification = classify_attempt3(
            &stdout,
            &stderr,
            &safety,
            &attempt3_terminal_bytes(ATTEMPT3_TERMINAL_OUTCOME),
        )
        .expect("recorded attempt-3 boundary must pass");
        assert_eq!(
            classification.outcome,
            "terminal_firmware_configuration_failure"
        );
        assert_eq!(classification.opened_device_ids, [0, 1]);
        assert!(!classification.model_operation_verified);
        assert!(!classification.fresh_hardware_success);
    }

    // r[verify tenstorrent.rwkv7_p150x2.attempt3_retention.false_success]
    #[test]
    fn attempt3_classifier_rejects_false_success_and_missing_conflict() {
        let safety =
            serde_json::to_vec(&attempt3_safety_receipt()).expect("safety fixture must serialize");
        let stdout =
            format!("{ATTEMPT3_OPENED_DEVICES_DIAGNOSTIC}\n{ATTEMPT3_OVERFLOW_DIAGNOSTIC}\n");
        let stderr = format!("compiler {ATTEMPT3_PROFILER_DEFINE} {ATTEMPT3_WATCHER_DEFINE}\n");
        assert!(
            classify_attempt3(
                &stdout,
                &stderr,
                &safety,
                &attempt3_terminal_bytes("verified_success"),
            )
            .expect_err("false success must fail")
            .contains("terminal outcome mismatch")
        );
        assert!(
            classify_attempt3(
                &stdout,
                ATTEMPT3_PROFILER_DEFINE,
                &safety,
                &attempt3_terminal_bytes(ATTEMPT3_TERMINAL_OUTCOME),
            )
            .expect_err("missing Watcher conflict must fail")
            .contains("lacks the profiler and Watcher conflict")
        );
    }

    #[test]
    fn safety_accepts_exact_ordered_multi_owner_restoration() {
        validate_safety(&valid_safety_receipt()).expect("exact multi-owner safety must pass");
    }

    #[test]
    fn safety_rejects_missing_reordered_duplicate_or_unhealthy_owners() {
        let mut missing = valid_safety_receipt();
        missing.owners.pop();
        assert!(validate_safety(&missing).is_err());

        let mut reordered = valid_safety_receipt();
        reordered
            .owners
            .swap(PRIMARY_OWNER_INDEX, SECONDARY_OWNER_INDEX);
        assert!(validate_safety(&reordered).is_err());

        let mut duplicated = valid_safety_receipt();
        duplicated.owners[SECONDARY_OWNER_INDEX] = duplicated.owners[PRIMARY_OWNER_INDEX].clone();
        assert!(validate_safety(&duplicated).is_err());

        let mut incomplete_stop = valid_safety_receipt();
        incomplete_stop.owners[SECONDARY_OWNER_INDEX].main_pid_zero_after_stop = false;
        assert!(validate_safety(&incomplete_stop).is_err());

        let mut failed_request = valid_safety_receipt();
        failed_request.owners[PRIMARY_OWNER_INDEX].stop_request_exit_status = 1;
        assert!(validate_safety(&failed_request).is_err());

        let mut unhealthy = valid_safety_receipt();
        unhealthy.owners[SECONDARY_OWNER_INDEX].endpoint_healthy_after = false;
        assert!(validate_safety(&unhealthy).is_err());

        let mut owned = valid_safety_receipt();
        owned.device_one_unowned = false;
        assert!(validate_safety(&owned).is_err());
    }

    #[test]
    fn complete_tensor_comparison_accepts_bounded_two_card_parity() {
        let reference_bytes = [1.0_f32.to_le_bytes(), (-2.0_f32).to_le_bytes()].concat();
        let device_bytes = [
            bf16::from_f32(1.0).to_bits().to_le_bytes(),
            bf16::from_f32(-2.0).to_bits().to_le_bytes(),
        ]
        .concat();
        let reference = artifact("reference.f32", &reference_bytes, TEST_ELEMENT_COUNT, None);
        let devices = [
            artifact("card-0.bf16", &device_bytes, TEST_ELEMENT_COUNT, Some(0)),
            artifact("card-1.bf16", &device_bytes, TEST_ELEMENT_COUNT, Some(1)),
        ];
        let comparison = compare_complete_tensor(
            "positive",
            &reference,
            &reference_bytes,
            &devices,
            [&device_bytes, &device_bytes],
            TEST_ELEMENT_COUNT,
            TEST_TOLERANCE,
        )
        .expect("bounded two-card parity must pass");
        assert!(comparison.cards_bit_exact);
        assert_eq!(comparison.card_zero_max_abs_deviation, 0.0);
    }

    #[test]
    fn complete_tensor_comparison_rejects_asymmetry_and_non_finite_data() {
        let reference_bytes = [1.0_f32.to_le_bytes(), 2.0_f32.to_le_bytes()].concat();
        let first = [
            bf16::from_f32(1.0).to_bits().to_le_bytes(),
            bf16::from_f32(2.0).to_bits().to_le_bytes(),
        ]
        .concat();
        let second = [
            bf16::from_f32(1.0).to_bits().to_le_bytes(),
            bf16::from_f32(3.0).to_bits().to_le_bytes(),
        ]
        .concat();
        let reference = artifact("reference.f32", &reference_bytes, TEST_ELEMENT_COUNT, None);
        let devices = [
            artifact("card-0.bf16", &first, TEST_ELEMENT_COUNT, Some(0)),
            artifact("card-1.bf16", &second, TEST_ELEMENT_COUNT, Some(1)),
        ];
        assert!(
            compare_complete_tensor(
                "asymmetric",
                &reference,
                &reference_bytes,
                &devices,
                [&first, &second],
                TEST_ELEMENT_COUNT,
                TEST_TOLERANCE,
            )
            .expect_err("asymmetric cards must fail")
            .contains("bit-exact")
        );

        let non_finite = f32::NAN.to_le_bytes();
        let non_finite_artifact = artifact("nan.f32", &non_finite, 1, None);
        assert!(
            decode_reference_f32(&non_finite_artifact, &non_finite)
                .expect_err("non-finite authority must fail")
                .contains("non-finite")
        );
    }

    #[test]
    fn artifact_paths_reject_absolute_and_parent_components() {
        validate_relative_artifact_path("evidence.bf16").expect("single filename must pass");
        assert!(validate_relative_artifact_path("/tmp/evidence").is_err());
        assert!(validate_relative_artifact_path("../evidence").is_err());
        assert!(validate_relative_artifact_path("nested/evidence").is_err());
    }

    #[test]
    fn profiler_parser_accepts_complete_rows_and_rejects_empty_input() {
        let csv = b"ARCH: blackhole, CHIP_FREQ[MHz]: 1000\nPCIe slot,core_x,core_y,RISC processor type,timer_id,time[cycles since reset],data,run host ID,trace id,trace id counter,zone name,type,source line,source file,meta data\n0,1,1,BRISC,1,100,0,7,0,0,KERNEL,begin,1,kernel.cpp,\n0,1,1,BRISC,1,200,0,7,0,0,KERNEL,end,1,kernel.cpp,\n";
        let rows = parse_profiler(csv).expect("complete profiler rows must parse");
        assert_eq!(rows.len(), TEST_ELEMENT_COUNT);
        assert_eq!(rows[0].runtime_id, TEST_PROFILER_RUNTIME_ID);
        assert!(parse_profiler(b"header only\n").is_err());
    }

    #[test]
    fn parser_boundaries_reject_oversized_and_over_field_inputs() {
        let oversized_json = vec![b' '; JSON_MAX_BYTES + 1];
        assert!(parse_reference(&oversized_json).is_err());
        assert!(parse_runtime(&oversized_json).is_err());
        assert!(parse_json_value(&oversized_json, "fixture").is_err());

        let oversized_yaml = vec![b' '; YAML_MAX_BYTES + 1];
        assert!(yaml_sequence(&oversized_yaml, "fixture").is_err());

        let oversized_vocabulary = vec![b' '; VOCABULARY_MAX_BYTES + 1];
        assert!(BoundedTokenizer::parse(&oversized_vocabulary).is_err());

        let over_field_row = vec!["field"; CSV_MAX_FIELDS + 1].join(",");
        assert!(parse_profiler(over_field_row.as_bytes()).is_err());
    }

    #[test]
    fn python_literal_parser_accepts_bytes_and_rejects_bad_escape() {
        assert_eq!(
            parse_vocabulary_line(r"1 b'A\n' 2").unwrap(),
            (1, b"A\n".to_vec())
        );
        assert!(parse_vocabulary_line(r"1 b'\q' 1").is_err());
    }
}
