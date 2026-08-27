let contracts = import "gdn-decode-fusion-contracts.ncl" in

let configured_schema_version = 1 in
let configured_samples_per_root = 5 in
let configured_warmup_runs = 1 in
let configured_maximum_duplicate_spread_ratio = 1.25 in
let configured_maximum_gdn_median_ratio = 0.95 in
let configured_maximum_full_step_median_ratio = 0.97 in

{
  schema_version = configured_schema_version,
  profile_id = "qwen38-p150x2-gdn-decode-fusion-v1",
  diagnostic_only = true,
  identities = {
    authority_blake3 = "c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
    tokenizer_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  topology = {
    expected_pcie_device_ids = [ 0, 1 ],
    mesh_rows = 1,
    mesh_columns = 2,
  },
  workload = {
    modes = [ "off", "norm", "state", "full" ],
    execution_order = [ "control", "candidate", "candidate", "control" ],
    samples_per_root = configured_samples_per_root,
    warmup_runs = configured_warmup_runs,
  },
  gates = {
    require_exact_token_parity = true,
    require_exact_text_parity = true,
    require_clean_reset = true,
    require_distinct_state_slices = true,
    maximum_duplicate_spread_ratio = configured_maximum_duplicate_spread_ratio,
    maximum_gdn_median_ratio = configured_maximum_gdn_median_ratio,
    maximum_full_step_median_ratio = configured_maximum_full_step_median_ratio,
  },
} | contracts.Profile
