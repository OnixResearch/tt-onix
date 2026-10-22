# Immutable B=1 versus B=8 execution policy, v3.
#
# v3 changes the protocol's root order, not a bound: the roots interleave the two
# routes and there are four per route, so a drift in the host's load reaches both
# halves of a ratio at the same point in time, and the median over four roots
# settles where two did not. The per-user time to first token from v2 stays; the
# v1 and v2 profiles and their receipts remain the historical measurements.
let configured_baseline_width = 1 in
let configured_candidate_width = 8 in
let configured_generated_tokens = 64 in
let configured_warmup_runs = 1 in
let configured_measured_runs = 3 in
let configured_minimum_aggregate_decode_speedup_ratio = 6.0 in
let configured_minimum_per_user_decode_retention_ratio = 0.75 in
let configured_maximum_ttft_ratio = 9.0 in
let configured_maximum_tail_latency_ratio = 1.5 in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-batched-throughput-b8-v3",
  identities = {
    authority_blake3 = "c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
    tokenizer_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  workload = {
    baseline_width = configured_baseline_width,
    candidate_width = configured_candidate_width,
    execution_order = [
      "single",
      "batch",
      "single",
      "batch",
      "single",
      "batch",
      "single",
      "batch",
    ],
    generated_tokens = configured_generated_tokens,
    measured_runs = configured_measured_runs,
    prompts = [
      "The capital of France is",
      "Empty prompt is a rejected request fixture.",
      "Repeat the token boundary exactly:",
      "Explain why deterministic cleanup matters for recurrent state.",
      "Water boils at a temperature of",
      "The largest planet in the solar system is",
      "Summarize why batch isolation matters for recurrent state.",
      "Count from one to five in words:",
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
}
