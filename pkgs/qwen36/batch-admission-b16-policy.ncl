# Immutable B=1 versus B=16 execution policy.
#
# Same bounds as the B=4 and B=8 policies; the width is the only axis this
# policy adds. The corpus is sixteen prompts because a width-sixteen run needs
# sixteen concurrent requests, one per prompt.
let configured_baseline_width = 1 in
let configured_candidate_width = 16 in
let configured_generated_tokens = 64 in
let configured_warmup_runs = 1 in
let configured_measured_runs = 3 in
let configured_minimum_aggregate_decode_speedup_ratio = 6.0 in
let configured_minimum_per_user_decode_retention_ratio = 0.75 in
let configured_maximum_ttft_ratio = 9.0 in
let configured_maximum_tail_latency_ratio = 1.5 in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-batched-throughput-b16-v1",
  identities = {
    authority_blake3 = "c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
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
      "Water boils at a temperature of",
      "The largest planet in the solar system is",
      "Summarize why batch isolation matters for recurrent state.",
      "Count from one to five in words:",
      "The first president of the United States was",
      "Name the three primary colours:",
      "A triangle with equal sides is called",
      "Describe what a semaphore protects in one sentence.",
      "The chemical symbol for gold is",
      "State the speed of light in kilometres per second:",
      "List two properties of a recurrent state buffer.",
      "Repeat the word isolation once:",
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
