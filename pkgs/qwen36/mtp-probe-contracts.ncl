# Typed, fail-closed policy for the diagnostic Qwen3.8 MTP target probe.
let expected_schema_version = 1 in
let expected_pcie_device_ids = [0, 1] in
let expected_verification_widths = [1, 2, 4] in
let expected_masked_bucket_width = 128 in
let expected_minimum_samples_per_width = 5 in
let PositiveIntC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && std.number.is_integer value
) in
let PositiveNumberC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0
) in
let NonNegativeNumberC = std.contract.from_predicate (fun value =>
  std.is_number value && value >= 0
) in
let UnitIntervalC = std.contract.from_predicate (fun value =>
  std.is_number value && value >= 0 && value <= 1
) in
let NonEmptyStringC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.length value > 0
) in
let Blake3C = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
) in
let RevisionC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{40}$" value
) in
let ExactC = fun expected => std.contract.from_predicate (fun value => value == expected) in
let ClosedRecordC = fun allowed => std.contract.from_predicate (fun value =>
  std.is_record value
  && std.array.length (std.record.fields value) == std.array.length allowed
  && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
) in
let PositiveIntArrayC = std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.length value > 0
  && std.array.all (fun item => std.is_number item && item > 0 && std.number.is_integer item) value
) in
let NonNegativeIntArrayC = std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.length value > 0
  && std.array.all (fun item => std.is_number item && item >= 0 && std.number.is_integer item) value
) in

let IdentityC = {
  authority_blake3 | Blake3C,
  checkpoint_revision | RevisionC,
  model_source_revision | RevisionC,
  tokenizer_revision | RevisionC,
} in
let IdentityClosedC = ClosedRecordC [
  "authority_blake3",
  "checkpoint_revision",
  "model_source_revision",
  "tokenizer_revision",
] in

let ProfileC = {
  schema_version | ExactC expected_schema_version,
  profile_id | NonEmptyStringC,
  diagnostic_only | ExactC true,
  identities | IdentityClosedC | IdentityC,
  expected_pcie_device_ids | NonNegativeIntArrayC,
  verification_widths | PositiveIntArrayC,
  masked_bucket_width | PositiveIntC,
  minimum_samples_per_width | PositiveIntC,
  minimum_state_pcc | UnitIntervalC,
  maximum_state_nmse | NonNegativeNumberC,
  maximum_sample_spread_ratio | PositiveNumberC,
  prompt | NonEmptyStringC,
} in
let ProfileClosedC = ClosedRecordC [
  "schema_version",
  "profile_id",
  "diagnostic_only",
  "identities",
  "expected_pcie_device_ids",
  "verification_widths",
  "masked_bucket_width",
  "minimum_samples_per_width",
  "minimum_state_pcc",
  "maximum_state_nmse",
  "maximum_sample_spread_ratio",
  "prompt",
] in
let ProfileSemanticC = std.contract.from_predicate (fun value =>
  value.expected_pcie_device_ids == expected_pcie_device_ids
  && value.verification_widths == expected_verification_widths
  && value.masked_bucket_width == expected_masked_bucket_width
  && value.minimum_samples_per_width == expected_minimum_samples_per_width
  && value.minimum_state_pcc > 0
  && value.maximum_sample_spread_ratio >= 1
) in
{
  Profile = ProfileC,
  ProfileClosed = ProfileClosedC,
  ProfileSemantic = ProfileSemanticC,
}
