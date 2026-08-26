# Bounded B=1 versus B=4 physical admission profile.
let contracts = import "batch-admission-contracts.ncl" in
let configured_baseline_width = 1 in
let configured_candidate_width = 4 in
let configured_generated_tokens = 64 in
let configured_warmup_runs = 1 in
let configured_measured_runs = 3 in
let configured_minimum_aggregate_decode_speedup_ratio = 3.0 in
let configured_minimum_per_user_decode_retention_ratio = 0.75 in
let configured_maximum_ttft_ratio = 4.5 in
let configured_maximum_tail_latency_ratio = 1.5 in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-batched-throughput-b4-v1",
  physically_admitted = true,
  identities = {
    authority_blake3 = "c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
    receipt_blake3 = "176d912457f4c0048be90c9b477709e4360dae0681212cf275d108f3ca48465f",
    tokenizer_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  workload = {
    baseline_width = configured_baseline_width,
    candidate_width = configured_candidate_width,
    execution_order = ["single", "batch", "batch", "single"],
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
    maximum_tail_latency_ratio = configured_maximum_tail_latency_ratio,
    maximum_ttft_ratio = configured_maximum_ttft_ratio,
    minimum_aggregate_decode_speedup_ratio = configured_minimum_aggregate_decode_speedup_ratio,
    minimum_per_user_decode_retention_ratio = configured_minimum_per_user_decode_retention_ratio,
    require_active_prefix = true,
    require_clean_reset = true,
    require_distinct_state_slices = true,
    require_exact_token_parity = true,
    require_physical_evidence = true,
  },
} | contracts.ProfileClosed | contracts.ProfileSemantic | contracts.Profile
