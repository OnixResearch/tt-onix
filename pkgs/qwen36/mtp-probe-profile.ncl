# Diagnostic-only target-verification profile. It cannot admit MTP.
let contracts = import "mtp-probe-contracts.ncl" in
let profile_schema_version = 2 in
let configured_expected_pcie_device_ids = [0, 1] in
let configured_verification_widths = [1, 2, 4] in
let configured_execution_path = "traced-hybrid-packed-decode" in
let configured_native_width_limit = 4 in
let configured_minimum_samples_per_width = 5 in
let configured_minimum_state_pcc = 0.999 in
let configured_maximum_state_nmse = 0.001 in
let configured_minimum_logits_pcc = 0.999 in
let configured_maximum_logits_nmse = 0.001 in
let configured_maximum_sample_spread_ratio = 1.25 in
let configured_minimum_end_to_end_amortization_ratio = 1.0 in
{
  schema_version = profile_schema_version,
  profile_id = "qwen38-p150x2-mtp-target-probe-v3",
  diagnostic_only = true,
  identities = {
    authority_blake3 = "c319f7eef1b58911457c128ca5397710f4c9af3b98d1ccc83914005f9d3726cd",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
    tokenizer_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  expected_pcie_device_ids = configured_expected_pcie_device_ids,
  verification_widths = configured_verification_widths,
  execution_path = configured_execution_path,
  native_width_limit = configured_native_width_limit,
  minimum_samples_per_width = configured_minimum_samples_per_width,
  minimum_state_pcc = configured_minimum_state_pcc,
  maximum_state_nmse = configured_maximum_state_nmse,
  minimum_logits_pcc = configured_minimum_logits_pcc,
  maximum_logits_nmse = configured_maximum_logits_nmse,
  maximum_sample_spread_ratio = configured_maximum_sample_spread_ratio,
  minimum_end_to_end_amortization_ratio = configured_minimum_end_to_end_amortization_ratio,
  prompt = "Explain why deterministic state rollback is required.",
} | contracts.ProfileClosed | contracts.ProfileSemantic | contracts.Profile
