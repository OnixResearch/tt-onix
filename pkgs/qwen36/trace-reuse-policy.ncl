# Immutable same-width trace-reuse execution policy.
let configured_width = 4 in
let configured_generated_tokens = 64 in
let configured_warmup_runs = 1 in
let configured_measured_runs = 3 in
let configured_maximum_resident_trace_count = 1 in
let configured_maximum_reused_trace_prepare_seconds = 0.05 in
let configured_maximum_duplicate_root_ratio = 1.25 in
let configured_maximum_ttft_ratio = 1.05 in
let configured_maximum_tail_latency_ratio = 1.10 in
let configured_minimum_decode_rate_retention_ratio = 0.95 in
let configured_minimum_request_wall_speedup_ratio = 1.05 in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-same-width-trace-reuse-b4-v1",
  identities = {
    authority_blake3 = "c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
    tokenizer_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  workload = {
    width = configured_width,
    execution_order = ["recapture", "reuse", "reuse", "recapture"],
    generated_tokens = configured_generated_tokens,
    measured_runs = configured_measured_runs,
    prompts = [
      "The capital of France is",
      "Empty prompt is a rejected request fixture.",
      "Repeat the token boundary exactly:",
      "Explain why deterministic cleanup matters for recurrent state.",
    ],
    warmup_runs = configured_warmup_runs,
  },
  admission = {
    maximum_duplicate_root_ratio = configured_maximum_duplicate_root_ratio,
    maximum_resident_trace_count = configured_maximum_resident_trace_count,
    maximum_reused_trace_prepare_seconds = configured_maximum_reused_trace_prepare_seconds,
    maximum_tail_latency_ratio = configured_maximum_tail_latency_ratio,
    maximum_ttft_ratio = configured_maximum_ttft_ratio,
    minimum_decode_rate_retention_ratio = configured_minimum_decode_rate_retention_ratio,
    minimum_request_wall_speedup_ratio = configured_minimum_request_wall_speedup_ratio,
    require_clean_reset = true,
    require_exact_token_parity = true,
    require_physical_evidence = true,
    require_stable_binding = true,
    require_zero_measured_recaptures = true,
  },
}
