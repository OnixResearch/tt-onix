# Diagnostic-only matched MTP draft-trace profile variant for the k=4 width
# sweep. It cannot admit production behavior. Only the draft width and profile
# id differ from the default draft-trace profile.
let contracts = import "mtp-trace-contracts.ncl" in
let profile_schema_version = 1 in
let configured_expected_pcie_device_ids = [0, 1] in
let configured_draft_token_count = 4 in
let configured_minimum_prompt_count = 5 in
let configured_route_block_count = 3 in
let configured_minimum_samples_per_route = 6 in
let configured_route_schedule = [
  "eager", "traced", "traced", "eager",
  "eager", "traced", "traced", "eager",
  "eager", "traced", "traced", "eager",
] in
let configured_maximum_duplicate_root_ratio = 1.25 in
let configured_maximum_traced_draft_ratio = 0.95 in
let configured_minimum_emitted_token_speedup = 1.0 in
let configured_maximum_trace_capture_seconds = 300.0 in
{
  schema_version = profile_schema_version,
  profile_id = "qwen38-p150x2-mtp-draft-trace-v1-k4",
  diagnostic_only = true,
  identities = {
    authority_blake3 = "c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
    tokenizer_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  expected_pcie_device_ids = configured_expected_pcie_device_ids,
  draft_token_count = configured_draft_token_count,
  minimum_prompt_count = configured_minimum_prompt_count,
  route_block_count = configured_route_block_count,
  minimum_samples_per_route = configured_minimum_samples_per_route,
  route_schedule = configured_route_schedule,
  maximum_duplicate_root_ratio = configured_maximum_duplicate_root_ratio,
  maximum_traced_draft_ratio = configured_maximum_traced_draft_ratio,
  minimum_emitted_token_speedup = configured_minimum_emitted_token_speedup,
  maximum_trace_capture_seconds = configured_maximum_trace_capture_seconds,
  prompts = [
    "Explain why deterministic rollback matters.",
    "Write one sentence about compiler correctness.",
    "What is the capital of France?",
    "List three properties of a reliable cache.",
    "Continue this sequence: one, two, three,",
  ],
} | contracts.ProfileClosed | contracts.ProfileSemantic | contracts.Profile
