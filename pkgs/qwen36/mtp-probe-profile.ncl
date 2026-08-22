# Diagnostic-only target-verification profile. It cannot admit MTP.
let contracts = import "mtp-probe-contracts.ncl" in
let profile_schema_version = 1 in
let configured_expected_pcie_device_ids = [0, 1] in
let configured_verification_widths = [1, 2, 4] in
let configured_masked_bucket_width = 128 in
let configured_minimum_samples_per_width = 5 in
let configured_minimum_state_pcc = 0.999 in
let configured_maximum_state_nmse = 0.001 in
let configured_maximum_sample_spread_ratio = 1.25 in
{
  schema_version = profile_schema_version,
  profile_id = "qwen38-p150x2-mtp-target-probe-v1",
  diagnostic_only = true,
  identities = {
    authority_blake3 = "c6146a5b5ea290afad2f5abbd99e73e987ed0d85e7d8df9ffb3b4ec0c56ee444",
    checkpoint_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
    model_source_revision = "d102e7f4512060e09e580e40cf26a870bfa88c55",
    tokenizer_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
  },
  expected_pcie_device_ids = configured_expected_pcie_device_ids,
  verification_widths = configured_verification_widths,
  masked_bucket_width = configured_masked_bucket_width,
  minimum_samples_per_width = configured_minimum_samples_per_width,
  minimum_state_pcc = configured_minimum_state_pcc,
  maximum_state_nmse = configured_maximum_state_nmse,
  maximum_sample_spread_ratio = configured_maximum_sample_spread_ratio,
  prompt = "Explain why deterministic state rollback is required.",
} | contracts.ProfileClosed | contracts.ProfileSemantic | contracts.Profile
