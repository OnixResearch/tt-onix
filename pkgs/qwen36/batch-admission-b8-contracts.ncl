# Typed, fail-closed contracts for Qwen3.8-27B P150x2 B=8 batch admission.
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
# Frozen predeclared protocol values. They mirror the named constants in
# batch-admission-b8-policy.ncl. Pinning them here is what makes an edited
# threshold or a shortened corpus fail to export instead of silently
# measuring against a weaker gate.
let frozen_baseline_width = 1 in
let frozen_candidate_width = 8 in
let frozen_generated_tokens = 64 in
let frozen_measured_runs = 3 in
let frozen_prompt_count = 8 in
let frozen_warmup_runs = 1 in
let frozen_minimum_aggregate_decode_speedup_ratio = 6.0 in
let frozen_minimum_per_user_decode_retention_ratio = 0.75 in
let frozen_maximum_ttft_ratio = 9.0 in
let frozen_maximum_tail_latency_ratio = 1.5 in
# The frozen corpus size. A shortened corpus cannot export: the predeclared
# rejection protocol is eight prompts, one warmup, and three measured runs.
let ExactlyNStringsC = fun count => std.contract.from_predicate (fun value =>
  std.is_array value
  && std.array.length value == count
  && std.array.all std.is_string value
  && std.array.all (fun item => std.string.length item > 0) value
) in
let RootOrderC = std.contract.from_predicate (fun value =>
  value == ["single", "batch", "batch", "single"]
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
  baseline_width | ExactC frozen_baseline_width,
  candidate_width | ExactC frozen_candidate_width,
  execution_order | RootOrderC,
  generated_tokens | ExactC frozen_generated_tokens,
  measured_runs | ExactC frozen_measured_runs,
  prompts | ExactlyNStringsC frozen_prompt_count,
  warmup_runs | ExactC frozen_warmup_runs,
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
  maximum_tail_latency_ratio | ExactC frozen_maximum_tail_latency_ratio,
  maximum_ttft_ratio | ExactC frozen_maximum_ttft_ratio,
  minimum_aggregate_decode_speedup_ratio | ExactC frozen_minimum_aggregate_decode_speedup_ratio,
  minimum_per_user_decode_retention_ratio | ExactC frozen_minimum_per_user_decode_retention_ratio,
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
  value.workload.candidate_width > value.workload.baseline_width
  && value.admission.minimum_aggregate_decode_speedup_ratio > 1
  && value.admission.minimum_per_user_decode_retention_ratio <= 1
  && value.admission.maximum_ttft_ratio >= 1
  && value.admission.maximum_tail_latency_ratio >= 1
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
  && value.workload.candidate_width > value.workload.baseline_width
  && value.admission.minimum_aggregate_decode_speedup_ratio > 1
  && value.admission.minimum_per_user_decode_retention_ratio <= 1
  && value.admission.maximum_ttft_ratio >= 1
  && value.admission.maximum_tail_latency_ratio >= 1
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
