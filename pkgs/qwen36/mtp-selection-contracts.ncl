# Typed fail-closed policy for paired MTP draft selection.
let expected_schema_version = 1 in
let expected_pcie_device_ids = [0, 1] in
let expected_draft_token_count = 3 in
let expected_minimum_prompt_count = 5 in
let expected_route_schedule = ["full-logit-readback", "shard-local-top1", "shard-local-top1", "full-logit-readback"] in
let PositiveIntC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && std.number.is_integer value
) in
let PositiveNumberC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0
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
let DeviceArrayC = std.contract.from_predicate (fun value =>
  value == expected_pcie_device_ids
) in
let PromptArrayC = std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.length value == expected_minimum_prompt_count
  && std.array.all (fun item => std.is_string item && std.string.length item > 0) value
) in
let RouteScheduleC = std.contract.from_predicate (fun value =>
  value == expected_route_schedule
) in
let ProfileC = {
  schema_version | ExactC expected_schema_version,
  profile_id | NonEmptyStringC,
  diagnostic_only | ExactC true,
  identities | IdentityClosedC | IdentityC,
  expected_pcie_device_ids | DeviceArrayC,
  draft_token_count | ExactC expected_draft_token_count,
  minimum_prompt_count | ExactC expected_minimum_prompt_count,
  route_schedule | RouteScheduleC,
  maximum_duplicate_root_ratio | PositiveNumberC,
  maximum_selector_ratio | PositiveNumberC,
  minimum_emitted_token_speedup | PositiveNumberC,
  prompts | PromptArrayC,
} in
let ProfileClosedC = ClosedRecordC [
  "schema_version",
  "profile_id",
  "diagnostic_only",
  "identities",
  "expected_pcie_device_ids",
  "draft_token_count",
  "minimum_prompt_count",
  "route_schedule",
  "maximum_duplicate_root_ratio",
  "maximum_selector_ratio",
  "minimum_emitted_token_speedup",
  "prompts",
] in
let ProfileSemanticC = std.contract.from_predicate (fun value =>
  value.maximum_duplicate_root_ratio >= 1
  && value.maximum_selector_ratio < 1
  && value.minimum_emitted_token_speedup >= 1
) in
{
  Profile = ProfileC,
  ProfileClosed = ProfileClosedC,
  ProfileSemantic = ProfileSemanticC,
}
