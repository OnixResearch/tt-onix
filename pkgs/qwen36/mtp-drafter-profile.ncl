# Diagnostic-only MTP drafter profile. It cannot admit production behavior.
let contracts = import "mtp-drafter-contracts.ncl" in
let profile_schema_version = 1 in
let configured_expected_pcie_device_ids = [0, 1] in
let configured_draft_token_count = 3 in
let configured_minimum_prompt_count = 5 in
let configured_maximum_sample_spread_ratio = 1.25 in
let configured_minimum_emitted_token_speedup = 1.0 in
{
  schema_version = profile_schema_version,
  profile_id = "qwen38-p150x2-mtp-drafter-probe-v1",
  diagnostic_only = true,
  identities = {
    authority_blake3 = "c6146a5b5ea290afad2f5abbd99e73e987ed0d85e7d8df9ffb3b4ec0c56ee444",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
    tokenizer_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  expected_pcie_device_ids = configured_expected_pcie_device_ids,
  draft_token_count = configured_draft_token_count,
  minimum_prompt_count = configured_minimum_prompt_count,
  maximum_sample_spread_ratio = configured_maximum_sample_spread_ratio,
  minimum_emitted_token_speedup = configured_minimum_emitted_token_speedup,
  prompts = [
    "Explain why deterministic rollback matters.",
    "Write one sentence about compiler correctness.",
    "What is the capital of France?",
    "List three properties of a reliable cache.",
    "Continue this sequence: one, two, three,",
  ],
} | contracts.ProfileClosed | contracts.ProfileSemantic | contracts.Profile
