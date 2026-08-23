# Typed fail-closed policy for matched persistent MTP recovery.
let configured_schema_version = 2 in
let configured_pcie_device_ids = [0, 1] in
let configured_recovery_execution_mode = "persistent-traced" in
let configured_process_trace_capture_count = 1 in
let configured_draft_token_count = 3 in
let configured_minimum_prompt_count = 5 in
let configured_route_block_count = 3 in
let configured_minimum_samples_per_route = 6 in
let configured_snapshot_row_count = 2 in
let configured_committed_advance_count = 1 in
let configured_acceptance_lengths = [0, 3] in
let configured_route_schedule = [
  "reprefill", "snapshot", "snapshot", "reprefill",
  "reprefill", "snapshot", "snapshot", "reprefill",
  "reprefill", "snapshot", "snapshot", "reprefill",
] in
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
let PromptArrayC = std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.length value == configured_minimum_prompt_count
  && std.array.all (fun item => std.is_string item && std.string.length item > 0) value
) in
let ProfileC = {
  schema_version | ExactC configured_schema_version,
  profile_id | NonEmptyStringC,
  diagnostic_only | ExactC true,
  identities | IdentityClosedC | IdentityC,
  expected_pcie_device_ids | ExactC configured_pcie_device_ids,
  recovery_execution_mode | ExactC configured_recovery_execution_mode,
  required_process_trace_capture_count | ExactC configured_process_trace_capture_count,
  draft_token_count | ExactC configured_draft_token_count,
  minimum_prompt_count | ExactC configured_minimum_prompt_count,
  route_block_count | ExactC configured_route_block_count,
  minimum_samples_per_route | ExactC configured_minimum_samples_per_route,
  route_schedule | ExactC configured_route_schedule,
  snapshot_row_count | ExactC configured_snapshot_row_count,
  committed_advance_count | ExactC configured_committed_advance_count,
  required_acceptance_lengths | ExactC configured_acceptance_lengths,
  maximum_duplicate_root_ratio | PositiveNumberC,
  maximum_snapshot_recovery_ratio | PositiveNumberC,
  prompts | PromptArrayC,
} in
let ProfileClosedC = ClosedRecordC [
  "schema_version",
  "profile_id",
  "diagnostic_only",
  "identities",
  "expected_pcie_device_ids",
  "recovery_execution_mode",
  "required_process_trace_capture_count",
  "draft_token_count",
  "minimum_prompt_count",
  "route_block_count",
  "minimum_samples_per_route",
  "route_schedule",
  "snapshot_row_count",
  "committed_advance_count",
  "required_acceptance_lengths",
  "maximum_duplicate_root_ratio",
  "maximum_snapshot_recovery_ratio",
  "prompts",
] in
let ProfileSemanticC = std.contract.from_predicate (fun value =>
  value.maximum_duplicate_root_ratio >= 1
  && value.maximum_snapshot_recovery_ratio < 1
) in
{
  Profile = ProfileC,
  ProfileClosed = ProfileClosedC,
  ProfileSemantic = ProfileSemanticC,
}
