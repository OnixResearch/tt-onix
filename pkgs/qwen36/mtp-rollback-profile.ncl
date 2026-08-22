# Diagnostic-only matched MTP K/V rollback profile.
let contracts = import "mtp-rollback-contracts.ncl" in
let configured_route_schedule = [
  "reprefill", "snapshot", "snapshot", "reprefill",
  "reprefill", "snapshot", "snapshot", "reprefill",
  "reprefill", "snapshot", "snapshot", "reprefill",
] in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-mtp-kv-rollback-v1",
  diagnostic_only = true,
  identities = {
    authority_blake3 = "c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
    tokenizer_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  expected_pcie_device_ids = [0, 1],
  draft_token_count = 3,
  minimum_prompt_count = 5,
  route_block_count = 3,
  minimum_samples_per_route = 6,
  route_schedule = configured_route_schedule,
  snapshot_row_count = 2,
  committed_advance_count = 1,
  required_acceptance_lengths = [0, 3],
  maximum_duplicate_root_ratio = 1.25,
  maximum_snapshot_recovery_ratio = 0.8,
  prompts = [
    "Explain why deterministic rollback matters.",
    "Write one sentence about compiler correctness.",
    "What is the capital of France?",
    "List three properties of a reliable cache.",
    "Continue this sequence: one, two, three,",
  ],
} | contracts.ProfileClosed | contracts.ProfileSemantic | contracts.Profile
