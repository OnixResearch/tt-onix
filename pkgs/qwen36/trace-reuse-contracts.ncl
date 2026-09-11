# Typed, fail-closed contracts for Qwen3.8 P150x2 same-width trace reuse.
let PositiveIntC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && std.number.is_integer value
) in
let PositiveNumberC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0
) in
let UnitFractionC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && value <= 1
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
  value == ["recapture", "reuse", "reuse", "recapture"]
) in
let PolicyIdentityC = {
  authority_blake3 | Blake3C,
  checkpoint_revision | RevisionC,
  model_source_revision | RevisionC,
  tokenizer_revision | RevisionC,
} in
let PolicyIdentityClosedC = ClosedRecordC [
  "authority_blake3",
  "checkpoint_revision",
  "model_source_revision",
  "tokenizer_revision",
] in
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
  width | ExactC 4,
  execution_order | RootOrderC,
  generated_tokens | PositiveIntC,
  measured_runs | PositiveIntC,
  prompts | StringArrayC,
  warmup_runs | PositiveIntC,
} in
let WorkloadClosedC = ClosedRecordC [
  "width",
  "execution_order",
  "generated_tokens",
  "measured_runs",
  "prompts",
  "warmup_runs",
] in
let AdmissionC = {
  maximum_duplicate_root_ratio | PositiveNumberC,
  maximum_resident_trace_count | ExactC 1,
  maximum_reused_trace_prepare_seconds | PositiveNumberC,
  maximum_tail_latency_ratio | PositiveNumberC,
  maximum_ttft_ratio | PositiveNumberC,
  minimum_decode_rate_retention_ratio | UnitFractionC,
  minimum_request_wall_speedup_ratio | PositiveNumberC,
  require_clean_reset | ExactC true,
  require_exact_token_parity | ExactC true,
  require_physical_evidence | ExactC true,
  require_stable_binding | ExactC true,
  require_zero_measured_recaptures | ExactC true,
} in
let AdmissionClosedC = ClosedRecordC [
  "maximum_duplicate_root_ratio",
  "maximum_resident_trace_count",
  "maximum_reused_trace_prepare_seconds",
  "maximum_tail_latency_ratio",
  "maximum_ttft_ratio",
  "minimum_decode_rate_retention_ratio",
  "minimum_request_wall_speedup_ratio",
  "require_clean_reset",
  "require_exact_token_parity",
  "require_physical_evidence",
  "require_stable_binding",
  "require_zero_measured_recaptures",
] in
let PolicyC = {
  schema_version | ExactC 1,
  profile_id | NonEmptyStringC,
  identities | PolicyIdentityClosedC | PolicyIdentityC,
  workload | WorkloadClosedC | WorkloadC,
  admission | AdmissionClosedC | AdmissionC,
} in
let PolicyClosedC = ClosedRecordC [
  "schema_version",
  "profile_id",
  "identities",
  "workload",
  "admission",
] in
let PolicySemanticC = std.contract.from_predicate (fun value =>
  value.admission.maximum_duplicate_root_ratio >= 1
  && value.admission.maximum_tail_latency_ratio >= 1
  && value.admission.maximum_ttft_ratio >= 1
  && value.admission.minimum_request_wall_speedup_ratio > 1
) in
let BindingC = {
  physically_admitted | Bool,
  receipt_blake3 | ReceiptC,
} in
let BindingClosedC = ClosedRecordC [
  "physically_admitted",
  "receipt_blake3",
] in
let BindingSemanticC = std.contract.from_predicate (fun value =>
  !value.physically_admitted || value.receipt_blake3 != "not-measured"
) in
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
  && value.admission.maximum_duplicate_root_ratio >= 1
  && value.admission.maximum_tail_latency_ratio >= 1
  && value.admission.maximum_ttft_ratio >= 1
  && value.admission.minimum_request_wall_speedup_ratio > 1
) in
{
  Binding = BindingC,
  BindingClosed = BindingClosedC,
  BindingSemantic = BindingSemanticC,
  Policy = PolicyC,
  PolicyClosed = PolicyClosedC,
  PolicySemantic = PolicySemanticC,
  Profile = ProfileC,
  ProfileClosed = ProfileClosedC,
  ProfileSemantic = ProfileSemanticC,
}
