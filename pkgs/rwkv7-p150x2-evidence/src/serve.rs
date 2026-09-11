use crate::{
    GENERATION_EOS_TOKEN_ID, MAXIMUM_MESSAGE_BYTE_COUNT, SCHEMA_VERSION, TokenizationLimits,
    TokenizedRequest, decode_generated_tokens, tokenize_request_with_limits,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const NORMAL_DISPATCH_MODE: &str = "normal_dispatch";
pub const PERSISTENT_TRACE_MODE: &str = "persistent_trace";
pub const TEXT_COMPLETION_OBJECT: &str = "text_completion";
const SERVE_DEVICE_COUNT: usize = 2;
const PHYSICAL_DEVICE_ZERO: u32 = 0;
const PHYSICAL_DEVICE_ONE: u32 = 1;
const STRICT_DECODE_WINDOWS: &[u32] = &[2, 4, 8];
const STRICT_MAXIMUM_GENERATED_TOKENS: u32 = 8;
const STRICT_CAPACITY_SLOTS: u32 = 4;
const EXPERIMENTAL_CAPACITY_SLOTS: u32 = 16;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServeMode {
    Strict,
    Experimental,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServeProfile {
    pub schema_version: u32,
    pub profile_id: String,
    pub model: String,
    pub server: ServerProfile,
    pub limits: ServeLimits,
    pub devices: Vec<DeviceProfile>,
    pub strict_capacity: CapacityProfile,
    pub experimental_capacity: CapacityProfile,
    pub primary_thread_fifo_priority: u32,
    pub default_mode: ServeMode,
    pub retain_request_artifacts: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServerProfile {
    pub listen_address: String,
    pub openai_completions_path: String,
    pub health_path: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ServeLimits {
    pub maximum_prompt_tokens: u32,
    pub strict_decode_windows: Vec<u32>,
    pub experimental_maximum_generated_tokens: u32,
    pub maximum_message_bytes: u32,
    pub maximum_http_header_bytes: u32,
    pub maximum_http_body_bytes: u32,
    pub maximum_active_connections: u32,
    pub maximum_lifetime_connections: u64,
    pub connection_io_timeout_seconds: u64,
    pub worker_timeout_seconds: u64,
    pub barrier_poll_milliseconds: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DeviceProfile {
    pub physical_device_id: u32,
    pub primary_thread_cpu: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CapacityProfile {
    pub id: String,
    pub per_card_slot_count: u32,
    pub temporary_tensor_bytes_per_slot: u64,
    pub device_budget_bytes: u64,
    pub safety_reserve_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(untagged)]
pub enum PromptInput {
    One(String),
    Many(Vec<String>),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: PromptInput,
    pub max_tokens: u32,
    pub mode: Option<ServeMode>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreparedBatch {
    pub mode: ServeMode,
    pub max_tokens: u32,
    pub requests: Vec<TokenizedRequest>,
    pub capacity: CapacityProfile,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FastWorkerRequest {
    pub schema_version: u32,
    pub run_id: String,
    pub capacity: CapacityProfile,
    pub requests: Vec<FastRequest>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct FastRequest {
    pub request_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub max_new_tokens: u32,
    pub state_slot: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompletionResult {
    pub mode: ServeMode,
    pub physical_device_id: u32,
    pub prompt_token_counts: Vec<u32>,
    pub generated_token_ids: Vec<Vec<u32>>,
    pub generated_texts: Vec<String>,
    pub production_fused_wkv_admitted: bool,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<CompletionChoice>,
    pub usage: CompletionUsage,
    pub rwkv7: CompletionMetadata,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompletionChoice {
    pub text: String,
    pub index: u32,
    pub finish_reason: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompletionUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CompletionMetadata {
    pub mode: ServeMode,
    pub physical_device_id: u32,
    pub production_fused_wkv_admitted: bool,
    pub persistent_trace_parity_validated: bool,
    pub exact_cpu_parity_claimed: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct WorkerReceipt {
    physical_device_ids: Vec<u32>,
    physical_hardware: bool,
    requests: Vec<WorkerRequestMetric>,
    persistent_decode: WorkerPersistentDecode,
    execution: WorkerExecution,
}

#[derive(Clone, Debug, Deserialize)]
struct WorkerRequestMetric {
    request_id: String,
    state_slot: u32,
    generated_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, Deserialize)]
struct WorkerPersistentDecode {
    selected_mode: String,
    persistent_region_entered: bool,
    exact_generated_token_parity: bool,
    exact_fp32_recurrent_state_parity: bool,
}

#[derive(Clone, Debug, Deserialize)]
struct WorkerExecution {
    production_physically_admitted: bool,
}

pub fn parse_serve_profile(bytes: &[u8]) -> Result<ServeProfile, String> {
    let profile: ServeProfile =
        serde_json::from_slice(bytes).map_err(|error| format!("invalid serve profile: {error}"))?;
    validate_serve_profile(&profile)?;
    Ok(profile)
}

pub fn validate_serve_profile(profile: &ServeProfile) -> Result<(), String> {
    require(
        profile.schema_version == SCHEMA_VERSION,
        "serve profile schema mismatch",
    )?;
    require(!profile.profile_id.is_empty(), "serve profile ID is empty")?;
    require(!profile.model.is_empty(), "serve model ID is empty")?;
    validate_server_profile(&profile.server)?;
    validate_serve_limits(&profile.limits)?;
    validate_devices(&profile.devices)?;
    validate_capacity(&profile.strict_capacity, STRICT_CAPACITY_SLOTS, "strict")?;
    validate_capacity(
        &profile.experimental_capacity,
        EXPERIMENTAL_CAPACITY_SLOTS,
        "experimental",
    )?;
    require(
        profile.primary_thread_fifo_priority > 0,
        "FIFO priority must be positive",
    )?;
    require(
        profile.default_mode == ServeMode::Strict,
        "default serve mode is not strict",
    )?;
    require(
        !profile.retain_request_artifacts,
        "serve profile retains request artifacts",
    )?;
    Ok(())
}

fn validate_server_profile(server: &ServerProfile) -> Result<(), String> {
    require(
        server.listen_address.starts_with("127.0.0.1:"),
        "serve listener must use IPv4 loopback",
    )?;
    require(
        server.openai_completions_path == "/v1/completions",
        "serve completions path changed",
    )?;
    require(server.health_path == "/health", "serve health path changed")?;
    Ok(())
}

fn validate_serve_limits(limits: &ServeLimits) -> Result<(), String> {
    require(
        limits.maximum_prompt_tokens > 0,
        "prompt bound must be positive",
    )?;
    require(
        limits.maximum_message_bytes > 0,
        "message byte bound must be positive",
    )?;
    let maximum_message_bytes = usize::try_from(limits.maximum_message_bytes)
        .map_err(|_| "message byte bound overflows usize")?;
    require(
        maximum_message_bytes <= MAXIMUM_MESSAGE_BYTE_COUNT,
        "message byte bound exceeds the tokenizer cap",
    )?;
    require(
        limits.strict_decode_windows.as_slice() == STRICT_DECODE_WINDOWS,
        "strict decode windows changed",
    )?;
    require(
        limits.experimental_maximum_generated_tokens > STRICT_MAXIMUM_GENERATED_TOKENS,
        "experimental generation bound must exceed strict mode",
    )?;
    require(
        limits.maximum_http_header_bytes > 0,
        "HTTP header bound must be positive",
    )?;
    require(
        limits.maximum_http_body_bytes >= limits.maximum_http_header_bytes,
        "HTTP body bound is smaller than the header bound",
    )?;
    require(
        limits.maximum_active_connections > 0,
        "active connection bound must be positive",
    )?;
    require(
        limits.maximum_lifetime_connections >= u64::from(limits.maximum_active_connections),
        "lifetime connection bound is smaller than the active bound",
    )?;
    require(
        limits.connection_io_timeout_seconds > 0,
        "connection I/O timeout must be positive",
    )?;
    require(
        limits.worker_timeout_seconds > 0,
        "worker timeout must be positive",
    )?;
    require(
        limits.barrier_poll_milliseconds > 0,
        "barrier poll interval must be positive",
    )?;
    Ok(())
}

fn validate_devices(devices: &[DeviceProfile]) -> Result<(), String> {
    require(
        devices.len() == SERVE_DEVICE_COUNT,
        "serve profile must bind two devices",
    )?;
    let device_ids = devices
        .iter()
        .map(|device| device.physical_device_id)
        .collect::<BTreeSet<_>>();
    require(
        device_ids.len() == devices.len(),
        "serve device IDs are not unique",
    )?;
    require(
        device_ids == BTreeSet::from([PHYSICAL_DEVICE_ZERO, PHYSICAL_DEVICE_ONE]),
        "serve device IDs changed",
    )?;
    Ok(())
}

fn validate_capacity(
    capacity: &CapacityProfile,
    expected_slots: u32,
    name: &str,
) -> Result<(), String> {
    require(
        !capacity.id.is_empty(),
        &format!("{name} capacity ID is empty"),
    )?;
    require(
        capacity.per_card_slot_count == expected_slots,
        &format!("{name} capacity slot count changed"),
    )?;
    require(
        capacity.temporary_tensor_bytes_per_slot > 0,
        &format!("{name} temporary tensor bound must be positive"),
    )?;
    require(
        capacity.device_budget_bytes > capacity.safety_reserve_bytes,
        &format!("{name} device budget does not exceed its reserve"),
    )?;
    Ok(())
}

pub fn parse_completion_request(bytes: &[u8]) -> Result<CompletionRequest, String> {
    serde_json::from_slice(bytes).map_err(|error| format!("invalid completion request: {error}"))
}

pub fn prepare_completion_batch(
    profile: &ServeProfile,
    vocabulary: &[u8],
    request: CompletionRequest,
) -> Result<PreparedBatch, String> {
    let mode = request.mode.unwrap_or(profile.default_mode);
    let prompts = prompt_values(request.prompt)?;
    validate_completion_shape(profile, &request.model, &prompts, request.max_tokens, mode)?;
    let maximum_generated_tokens = match mode {
        ServeMode::Strict => *profile
            .limits
            .strict_decode_windows
            .last()
            .ok_or_else(|| "strict decode windows are empty".to_owned())?,
        ServeMode::Experimental => profile.limits.experimental_maximum_generated_tokens,
    };
    let maximum_prompt_tokens = usize::try_from(profile.limits.maximum_prompt_tokens)
        .map_err(|_| "maximum prompt token count overflows usize")?;
    let maximum_generated_tokens = usize::try_from(maximum_generated_tokens)
        .map_err(|_| "maximum generated token count overflows usize")?;
    let max_tokens =
        usize::try_from(request.max_tokens).map_err(|_| "requested token count overflows usize")?;
    let mut requests = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        requests.push(tokenize_request_with_limits(
            vocabulary,
            prompt,
            maximum_prompt_tokens,
            max_tokens,
            TokenizationLimits {
                maximum_prompt_tokens,
                minimum_generated_tokens: 1,
                maximum_generated_tokens,
            },
        )?);
    }
    if mode == ServeMode::Strict {
        let first_length = requests
            .first()
            .ok_or_else(|| "strict batch has no request".to_owned())?
            .prompt_token_ids
            .len();
        require(
            requests
                .iter()
                .all(|current| current.prompt_token_ids.len() == first_length),
            "strict batch prompts must have equal token lengths",
        )?;
    }
    let capacity = match mode {
        ServeMode::Strict => profile.strict_capacity.clone(),
        ServeMode::Experimental => profile.experimental_capacity.clone(),
    };
    Ok(PreparedBatch {
        mode,
        max_tokens: request.max_tokens,
        requests,
        capacity,
    })
}

fn prompt_values(input: PromptInput) -> Result<Vec<String>, String> {
    let prompts = match input {
        PromptInput::One(prompt) => vec![prompt],
        PromptInput::Many(prompts) => prompts,
    };
    require(!prompts.is_empty(), "completion prompt list is empty")?;
    require(
        prompts.iter().all(|prompt| !prompt.is_empty()),
        "completion prompt is empty",
    )?;
    Ok(prompts)
}

fn validate_completion_shape(
    profile: &ServeProfile,
    model: &str,
    prompts: &[String],
    max_tokens: u32,
    mode: ServeMode,
) -> Result<(), String> {
    require(
        model == profile.model,
        "completion model does not match the pinned model",
    )?;
    require(!prompts.is_empty(), "completion prompt list is empty")?;
    let capacity = match mode {
        ServeMode::Strict => &profile.strict_capacity,
        ServeMode::Experimental => &profile.experimental_capacity,
    };
    let prompt_count = u32::try_from(prompts.len()).map_err(|_| "prompt count overflows u32")?;
    require(
        prompt_count <= capacity.per_card_slot_count,
        "completion batch exceeds card capacity",
    )?;
    let maximum_message_bytes = usize::try_from(profile.limits.maximum_message_bytes)
        .map_err(|_| "message byte bound overflows usize")?;
    require(
        prompts
            .iter()
            .all(|prompt| prompt.len() <= maximum_message_bytes),
        "completion prompt exceeds the message byte bound",
    )?;
    match mode {
        ServeMode::Strict => require(
            profile.limits.strict_decode_windows.contains(&max_tokens),
            "strict max_tokens must be an admitted decode window",
        )?,
        ServeMode::Experimental => require(
            (1..=profile.limits.experimental_maximum_generated_tokens).contains(&max_tokens),
            "experimental max_tokens is outside the configured range",
        )?,
    }
    Ok(())
}

pub fn build_fast_worker_request(
    prepared: &PreparedBatch,
    run_id: &str,
) -> Result<FastWorkerRequest, String> {
    require(!run_id.is_empty(), "worker run ID is empty")?;
    require(
        !prepared.requests.is_empty(),
        "prepared batch has no requests",
    )?;
    let request_count = u32::try_from(prepared.requests.len())
        .map_err(|_| "prepared request count overflows u32")?;
    require(
        request_count <= prepared.capacity.per_card_slot_count,
        "prepared request count exceeds worker capacity",
    )?;
    let mut requests = Vec::with_capacity(prepared.requests.len());
    for (index, request) in prepared.requests.iter().enumerate() {
        let state_slot = u32::try_from(index).map_err(|_| "state slot overflows u32")?;
        let mut prompt_token_ids = Vec::with_capacity(request.prompt_token_ids.len());
        for token in &request.prompt_token_ids {
            prompt_token_ids
                .push(u32::try_from(*token).map_err(|_| "prompt token ID overflows u32")?);
        }
        requests.push(FastRequest {
            request_id: format!("{run_id}-{state_slot}"),
            prompt_token_ids,
            max_new_tokens: prepared.max_tokens,
            state_slot,
        });
    }
    Ok(FastWorkerRequest {
        schema_version: SCHEMA_VERSION,
        run_id: run_id.to_owned(),
        capacity: prepared.capacity.clone(),
        requests,
    })
}

pub fn interpret_worker_receipt(
    bytes: &[u8],
    vocabulary: &[u8],
    prepared: &PreparedBatch,
    expected: &FastWorkerRequest,
    physical_device_id: u32,
) -> Result<CompletionResult, String> {
    let receipt: WorkerReceipt = serde_json::from_slice(bytes)
        .map_err(|error| format!("invalid worker receipt: {error}"))?;
    require(
        receipt.physical_hardware,
        "worker receipt is not physical hardware",
    )?;
    require(
        receipt.physical_device_ids == [physical_device_id],
        "worker receipt has the wrong physical device",
    )?;
    validate_worker_mode(prepared.mode, &receipt)?;
    require(
        receipt.requests.len() == expected.requests.len(),
        "worker result count differs from the request count",
    )?;
    let mut metrics = receipt.requests;
    metrics.sort_by_key(|metric| metric.state_slot);
    let expected_generated_tokens = usize::try_from(prepared.max_tokens)
        .map_err(|_| "requested generated-token count overflows usize")?;
    let mut prompt_token_counts = Vec::with_capacity(metrics.len());
    let mut generated_token_ids = Vec::with_capacity(metrics.len());
    let mut generated_texts = Vec::with_capacity(metrics.len());
    for (index, metric) in metrics.into_iter().enumerate() {
        let expected_request = expected
            .requests
            .get(index)
            .ok_or_else(|| "worker result index exceeds the request list".to_owned())?;
        require(
            metric.request_id == expected_request.request_id,
            "worker request ID changed",
        )?;
        require(
            metric.state_slot == expected_request.state_slot,
            "worker state slot changed",
        )?;
        require(
            metric.generated_token_ids.len() == expected_generated_tokens,
            "worker generated-token count changed",
        )?;
        let token_ids = metric
            .generated_token_ids
            .iter()
            .map(|token| usize::try_from(*token).map_err(|_| "generated token ID overflows usize"))
            .collect::<Result<Vec<_>, _>>()?;
        let generated_text = if token_ids.first() == Some(&GENERATION_EOS_TOKEN_ID) {
            String::new()
        } else {
            String::from_utf8(decode_generated_tokens(vocabulary, &token_ids)?)
                .map_err(|error| format!("generated text is not UTF-8: {error}"))?
        };
        prompt_token_counts.push(
            u32::try_from(expected_request.prompt_token_ids.len())
                .map_err(|_| "prompt token count overflows u32")?,
        );
        generated_token_ids.push(metric.generated_token_ids);
        generated_texts.push(generated_text);
    }
    Ok(CompletionResult {
        mode: prepared.mode,
        physical_device_id,
        prompt_token_counts,
        generated_token_ids,
        generated_texts,
        production_fused_wkv_admitted: receipt.execution.production_physically_admitted,
    })
}

fn validate_worker_mode(mode: ServeMode, receipt: &WorkerReceipt) -> Result<(), String> {
    match mode {
        ServeMode::Strict => {
            require(
                receipt.persistent_decode.selected_mode == PERSISTENT_TRACE_MODE,
                "strict worker did not select persistent trace",
            )?;
            require(
                receipt.persistent_decode.persistent_region_entered,
                "strict worker did not enter the persistent region",
            )?;
            require(
                receipt.persistent_decode.exact_generated_token_parity,
                "strict worker lacks generated-token parity",
            )?;
            require(
                receipt.persistent_decode.exact_fp32_recurrent_state_parity,
                "strict worker lacks recurrent-state parity",
            )?;
        }
        ServeMode::Experimental => {
            require(
                receipt.persistent_decode.selected_mode == NORMAL_DISPATCH_MODE,
                "experimental worker did not select normal dispatch",
            )?;
            require(
                !receipt.persistent_decode.persistent_region_entered,
                "experimental worker entered the persistent region",
            )?;
        }
    }
    Ok(())
}

pub fn build_completion_response(
    profile: &ServeProfile,
    request_id: String,
    created_unix_seconds: u64,
    result: CompletionResult,
) -> Result<CompletionResponse, String> {
    require(!request_id.is_empty(), "completion response ID is empty")?;
    require(
        result.generated_texts.len() == result.generated_token_ids.len(),
        "completion text and token counts differ",
    )?;
    require(
        result.prompt_token_counts.len() == result.generated_token_ids.len(),
        "completion prompt and generated counts differ",
    )?;
    let generation_eos_token_id = u32::try_from(GENERATION_EOS_TOKEN_ID)
        .map_err(|_| "generation EOS token ID overflows u32")?;
    let mut choices = Vec::with_capacity(result.generated_texts.len());
    let mut prompt_tokens = 0_u32;
    let mut completion_tokens = 0_u32;
    for (index, ((text, tokens), prompt_count)) in result
        .generated_texts
        .into_iter()
        .zip(result.generated_token_ids)
        .zip(result.prompt_token_counts)
        .enumerate()
    {
        prompt_tokens = prompt_tokens
            .checked_add(prompt_count)
            .ok_or_else(|| "prompt usage overflows u32".to_owned())?;
        let generated_count =
            u32::try_from(tokens.len()).map_err(|_| "generated usage count overflows u32")?;
        completion_tokens = completion_tokens
            .checked_add(generated_count)
            .ok_or_else(|| "completion usage overflows u32".to_owned())?;
        let finish_reason = if tokens.contains(&generation_eos_token_id) {
            "stop"
        } else {
            "length"
        };
        choices.push(CompletionChoice {
            text,
            index: u32::try_from(index).map_err(|_| "choice index overflows u32")?,
            finish_reason,
        });
    }
    let total_tokens = prompt_tokens
        .checked_add(completion_tokens)
        .ok_or_else(|| "total usage overflows u32".to_owned())?;
    Ok(CompletionResponse {
        id: request_id,
        object: TEXT_COMPLETION_OBJECT,
        created: created_unix_seconds,
        model: profile.model.clone(),
        choices,
        usage: CompletionUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens,
        },
        rwkv7: CompletionMetadata {
            mode: result.mode,
            physical_device_id: result.physical_device_id,
            production_fused_wkv_admitted: result.production_fused_wkv_admitted,
            persistent_trace_parity_validated: result.mode == ServeMode::Strict,
            exact_cpu_parity_claimed: false,
        },
    })
}

pub fn select_least_loaded_device(loads: &[u32]) -> Result<usize, String> {
    require(!loads.is_empty(), "device load list is empty")?;
    let mut selected_index = 0_usize;
    let mut selected_load = loads[0];
    for (index, load) in loads.iter().enumerate().skip(1) {
        if *load < selected_load {
            selected_index = index;
            selected_load = *load;
        }
    }
    require(
        selected_index < loads.len(),
        "selected device index is outside the load list",
    )?;
    Ok(selected_index)
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

    const MODEL: &str = "RWKV/RWKV7-Goose-World2.8-0.1B-HF";
    const STRICT_SLOTS: u32 = 4;
    const EXPERIMENTAL_SLOTS: u32 = 16;
    const TEMPORARY_BYTES: u64 = 5_242_880;
    const DEVICE_BYTES: u64 = 3_221_225_472;
    const RESERVE_BYTES: u64 = 268_435_456;

    fn capacity(id: &str, slots: u32) -> CapacityProfile {
        CapacityProfile {
            id: id.to_owned(),
            per_card_slot_count: slots,
            temporary_tensor_bytes_per_slot: TEMPORARY_BYTES,
            device_budget_bytes: DEVICE_BYTES,
            safety_reserve_bytes: RESERVE_BYTES,
        }
    }

    fn profile() -> ServeProfile {
        ServeProfile {
            schema_version: SCHEMA_VERSION,
            profile_id: "serve-test".to_owned(),
            model: MODEL.to_owned(),
            server: ServerProfile {
                listen_address: "127.0.0.1:8080".to_owned(),
                openai_completions_path: "/v1/completions".to_owned(),
                health_path: "/health".to_owned(),
            },
            limits: ServeLimits {
                maximum_prompt_tokens: 32,
                strict_decode_windows: vec![2, 4, 8],
                experimental_maximum_generated_tokens: 64,
                maximum_message_bytes: 256,
                maximum_http_header_bytes: 16_384,
                maximum_http_body_bytes: 262_144,
                maximum_active_connections: 32,
                maximum_lifetime_connections: 1_000_000,
                connection_io_timeout_seconds: 30,
                worker_timeout_seconds: 900,
                barrier_poll_milliseconds: 10,
            },
            devices: vec![
                DeviceProfile {
                    physical_device_id: 0,
                    primary_thread_cpu: 2,
                },
                DeviceProfile {
                    physical_device_id: 1,
                    primary_thread_cpu: 6,
                },
            ],
            strict_capacity: capacity("B4", STRICT_SLOTS),
            experimental_capacity: capacity("B16", EXPERIMENTAL_SLOTS),
            primary_thread_fifo_priority: 1,
            default_mode: ServeMode::Strict,
            retain_request_artifacts: false,
        }
    }

    #[test]
    fn accepts_the_bounded_profile_and_rejects_external_binding() {
        let valid = profile();
        assert!(validate_serve_profile(&valid).is_ok());
        let mut invalid = valid;
        invalid.server.listen_address = "0.0.0.0:8080".to_owned();
        assert_eq!(
            validate_serve_profile(&invalid),
            Err("serve listener must use IPv4 loopback".to_owned())
        );
    }

    #[test]
    fn parses_positive_request_and_rejects_unknown_fields() {
        let valid = br#"{"model":"RWKV/RWKV7-Goose-World2.8-0.1B-HF","prompt":"Hello","max_tokens":8,"mode":"strict"}"#;
        let parsed = parse_completion_request(valid).expect("valid completion request must parse");
        assert_eq!(parsed.max_tokens, 8);
        assert_eq!(parsed.mode, Some(ServeMode::Strict));
        let invalid = br#"{"model":"RWKV/RWKV7-Goose-World2.8-0.1B-HF","prompt":"Hello","max_tokens":8,"mode":"strict","extra":true}"#;
        assert!(parse_completion_request(invalid).is_err());
    }

    #[test]
    fn enforces_strict_and_experimental_generation_bounds() {
        let profile = profile();
        let prompts = vec!["Hello".to_owned()];
        assert!(validate_completion_shape(&profile, MODEL, &prompts, 8, ServeMode::Strict).is_ok());
        assert!(
            validate_completion_shape(&profile, MODEL, &prompts, 7, ServeMode::Strict).is_err()
        );
        assert!(
            validate_completion_shape(&profile, MODEL, &prompts, 64, ServeMode::Experimental)
                .is_ok()
        );
        assert!(
            validate_completion_shape(&profile, MODEL, &prompts, 65, ServeMode::Experimental)
                .is_err()
        );
    }

    #[test]
    fn builds_bounded_worker_request_and_rejects_empty_run_id() {
        let tokenized = TokenizedRequest {
            schema_version: SCHEMA_VERSION,
            user_message: "Hello".to_owned(),
            rendered_chat_prompt: "prompt".to_owned(),
            prompt_token_ids: vec![0, 1],
            max_prompt_tokens: 32,
            max_new_tokens: 8,
            prompt_token_ids_blake3: "hash".to_owned(),
        };
        let prepared = PreparedBatch {
            mode: ServeMode::Strict,
            max_tokens: 8,
            requests: vec![tokenized],
            capacity: capacity("B4", STRICT_SLOTS),
        };
        let request = build_fast_worker_request(&prepared, "request-1")
            .expect("valid prepared request must build");
        assert_eq!(request.requests.len(), 1);
        assert_eq!(request.requests[0].state_slot, 0);
        assert!(build_fast_worker_request(&prepared, "").is_err());
    }

    #[test]
    fn selects_least_loaded_device_with_lowest_index_tie_break() {
        assert_eq!(select_least_loaded_device(&[1, 0]), Ok(1));
        assert_eq!(select_least_loaded_device(&[0, 0]), Ok(0));
        assert!(select_least_loaded_device(&[]).is_err());
    }

    #[test]
    fn builds_openai_response_and_marks_experimental_non_parity() {
        let result = CompletionResult {
            mode: ServeMode::Experimental,
            physical_device_id: 1,
            prompt_token_counts: vec![3],
            generated_token_ids: vec![vec![4, 5]],
            generated_texts: vec!["ok".to_owned()],
            production_fused_wkv_admitted: true,
        };
        let response = build_completion_response(&profile(), "cmpl-1".to_owned(), 1, result)
            .expect("valid completion result must build");
        assert_eq!(response.usage.total_tokens, 5);
        assert!(!response.rwkv7.exact_cpu_parity_claimed);
        assert_eq!(response.choices[0].finish_reason, "length");
    }
}
