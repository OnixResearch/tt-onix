# Candidate profile. Production stays eager until matched physical admission passes.
let contracts = import "decode-contracts.ncl" in
let profile_schema_version = 1 in
let candidate_slot_count = 1 in
let configured_trace_command_queue_id = 0 in
let configured_trace_capture_timeout_seconds = 300 in
let configured_request_timeout_seconds = 1800 in
let configured_maximum_sequence_length = 2048 in
let configured_kv_block_size = 64 in
let configured_trace_region_size_bytes = 1073741824 in
let configured_warmup_runs = 1 in
let configured_measured_runs = 3 in
let configured_generated_tokens = 64 in
let configured_minimum_median_decode_speedup_ratio = 1.10 in
let configured_maximum_ttft_regression_ratio = 1.05 in
let configured_maximum_tail_latency_regression_ratio = 1.10 in
{
  schema_version = profile_schema_version,
  profile_id = "qwen38-p150x2-traced-decode-v1",
  physically_admitted = false,
  identities = {
    authority_blake3 = "256ce1b98d864469413fe11eb6c2409d560bfcfb851d9ddf891e72b6d42324c7",
    baseline_receipt_blake3 = "060eb57990b663b4548423057eac68ab8d4108aac86b6a548b0a8e2ffb7a3fe5",
    candidate_receipt_blake3 = "not-measured",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
    tokenizer_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  trace = {
    capture_timeout_seconds = configured_trace_capture_timeout_seconds,
    request_timeout_seconds = configured_request_timeout_seconds,
    enabled = true,
    slot_count = candidate_slot_count,
    command_queue_id = configured_trace_command_queue_id,
  },
  buffers = {
    kv_block_size = configured_kv_block_size,
    maximum_sequence_length = configured_maximum_sequence_length,
    trace_region_size_bytes = configured_trace_region_size_bytes,
  },
  workload = {
    prompts = [
      "The capital of France is",
      "Empty prompt is a rejected request fixture.",
      "Repeat the token boundary exactly:",
      "Explain why deterministic cleanup matters for recurrent state.",
    ],
    warmup_runs = configured_warmup_runs,
    measured_runs = configured_measured_runs,
    generated_tokens = configured_generated_tokens,
  },
  admission = {
    maximum_tail_latency_regression_ratio = configured_maximum_tail_latency_regression_ratio,
    maximum_ttft_regression_ratio = configured_maximum_ttft_regression_ratio,
    minimum_median_decode_speedup_ratio = configured_minimum_median_decode_speedup_ratio,
    require_clean_reset = true,
    require_exact_token_parity = true,
    require_physical_evidence = true,
  },
} | contracts.ProfileClosed | contracts.ProfileSemantic | contracts.Profile
