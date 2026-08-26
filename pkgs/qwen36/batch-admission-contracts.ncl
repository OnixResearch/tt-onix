# Typed, fail-closed contracts for Qwen3.8-27B P150x2 batch admission.
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
let ReceiptC = std.contract.from_predicate (fun value =>
  std.is_string value
  && (value == "not-measured" || std.string.is_match "^[0-9a-f]{64}$" value)
) in
let ExactC = fun expected => std.contract.from_predicate (fun value => value == expected) in
let ClosedRecordC = fun allowed => std.contract.from_predicate (fun value =>
  std.is_record value
  && std.array.length (std.record.fields value) == std.array.length allowed
  && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
) in
let StringArrayC = std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.length value > 0
  && std.array.all std.is_string value
  && std.array.all (fun item => std.string.length item > 0) value
) in
let RootOrderC = std.contract.from_predicate (fun value =>
  value == ["single", "batch", "batch", "single"]
) in

let IdentityC = {
  authority_blake3 | Blake3C,
  checkpoint_revision | RevisionC,
  model_source_revision | RevisionC,
  receipt_blake3 | ReceiptC,
  tokenizer_revision | RevisionC,
} in
let IdentityClosedC = ClosedRecordC [
  "authority_blake3",
  "checkpoint_revision",
  "model_source_revision",
  "receipt_blake3",
  "tokenizer_revision",
] in

let WorkloadC = {
  baseline_width | ExactC 1,
  candidate_width | ExactC 4,
  execution_order | RootOrderC,
  generated_tokens | PositiveIntC,
  measured_runs | PositiveIntC,
  prompts | StringArrayC,
  warmup_runs | PositiveIntC,
} in
let WorkloadClosedC = ClosedRecordC [
  "baseline_width",
  "candidate_width",
  "execution_order",
  "generated_tokens",
  "measured_runs",
  "prompts",
  "warmup_runs",
] in

let AdmissionC = {
  maximum_tail_latency_ratio | PositiveNumberC,
  maximum_ttft_ratio | PositiveNumberC,
  minimum_aggregate_decode_speedup_ratio | PositiveNumberC,
  minimum_per_user_decode_retention_ratio | PositiveNumberC,
  require_active_prefix | ExactC true,
  require_clean_reset | ExactC true,
  require_distinct_state_slices | ExactC true,
  require_exact_token_parity | ExactC true,
  require_physical_evidence | ExactC true,
} in
let AdmissionClosedC = ClosedRecordC [
  "maximum_tail_latency_ratio",
  "maximum_ttft_ratio",
  "minimum_aggregate_decode_speedup_ratio",
  "minimum_per_user_decode_retention_ratio",
  "require_active_prefix",
  "require_clean_reset",
  "require_distinct_state_slices",
  "require_exact_token_parity",
  "require_physical_evidence",
] in

let ProfileC = {
  schema_version | ExactC 1,
  profile_id | NonEmptyStringC,
  physically_admitted | Bool,
  identities | IdentityClosedC | IdentityC,
  workload | WorkloadClosedC | WorkloadC,
  admission | AdmissionClosedC | AdmissionC,
} in
let ProfileClosedC = ClosedRecordC [
  "schema_version",
  "profile_id",
  "physically_admitted",
  "identities",
  "workload",
  "admission",
] in
let ProfileSemanticC = std.contract.from_predicate (fun value =>
  (!value.physically_admitted || value.identities.receipt_blake3 != "not-measured")
  && value.workload.candidate_width > value.workload.baseline_width
  && value.admission.minimum_aggregate_decode_speedup_ratio > 1
  && value.admission.minimum_per_user_decode_retention_ratio <= 1
  && value.admission.maximum_ttft_ratio >= 1
  && value.admission.maximum_tail_latency_ratio >= 1
) in
{
  Profile = ProfileC,
  ProfileClosed = ProfileClosedC,
  ProfileSemantic = ProfileSemanticC,
}
