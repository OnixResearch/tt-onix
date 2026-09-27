# Typed, fail-closed contracts for Qwen3.8-27B P150x2 traced decode.
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
let CandidateReceiptC = std.contract.from_predicate (fun value =>
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

let IdentityC = {
  authority_blake3 | Blake3C,
  baseline_receipt_blake3 | Blake3C,
  candidate_receipt_blake3 | CandidateReceiptC,
  checkpoint_revision | RevisionC,
  model_source_revision | RevisionC,
  tokenizer_revision | RevisionC,
} in
let IdentityClosedC = ClosedRecordC [
  "authority_blake3",
  "baseline_receipt_blake3",
  "candidate_receipt_blake3",
  "checkpoint_revision",
  "model_source_revision",
  "tokenizer_revision",
] in

let TraceC = {
  capture_timeout_seconds | PositiveIntC,
  request_timeout_seconds | PositiveIntC,
  enabled | Bool,
  slot_count | PositiveIntC,
  command_queue_id | Number,
} in
let TraceClosedC = ClosedRecordC [
  "capture_timeout_seconds",
  "request_timeout_seconds",
  "enabled",
  "slot_count",
  "command_queue_id",
] in

let BufferC = {
  kv_block_size | PositiveIntC,
  kv_pool_blocks | PositiveIntC,
  maximum_sequence_length | PositiveIntC,
  trace_region_size_bytes | PositiveIntC,
} in
let BufferClosedC = ClosedRecordC [
  "kv_block_size",
  "kv_pool_blocks",
  "maximum_sequence_length",
  "trace_region_size_bytes",
] in

let WorkloadC = {
  prompts | StringArrayC,
  warmup_runs | PositiveIntC,
  measured_runs | PositiveIntC,
  generated_tokens | PositiveIntC,
} in
let WorkloadClosedC = ClosedRecordC [
  "prompts",
  "warmup_runs",
  "measured_runs",
  "generated_tokens",
] in

let AdmissionC = {
  maximum_tail_latency_regression_ratio | PositiveNumberC,
  maximum_ttft_regression_ratio | PositiveNumberC,
  minimum_median_decode_speedup_ratio | PositiveNumberC,
  require_clean_reset | ExactC true,
  require_exact_token_parity | ExactC true,
  require_physical_evidence | ExactC true,
} in
let AdmissionClosedC = ClosedRecordC [
  "maximum_tail_latency_regression_ratio",
  "maximum_ttft_regression_ratio",
  "minimum_median_decode_speedup_ratio",
  "require_clean_reset",
  "require_exact_token_parity",
  "require_physical_evidence",
] in

let ProfileC = {
  schema_version | ExactC 1,
  profile_id | NonEmptyStringC,
  physically_admitted | Bool,
  identities | IdentityClosedC | IdentityC,
  trace | TraceClosedC | TraceC,
  buffers | BufferClosedC | BufferC,
  workload | WorkloadClosedC | WorkloadC,
  admission | AdmissionClosedC | AdmissionC,
} in
let ProfileClosedC = ClosedRecordC [
  "schema_version",
  "profile_id",
  "physically_admitted",
  "identities",
  "trace",
  "buffers",
  "workload",
  "admission",
] in
let ProfileSemanticC = std.contract.from_predicate (fun value =>
  value.trace.slot_count == 1
  && value.trace.command_queue_id == 0
  && (!value.physically_admitted || value.identities.candidate_receipt_blake3 != "not-measured")
  && value.buffers.maximum_sequence_length >= value.workload.generated_tokens
  && value.buffers.kv_pool_blocks * value.buffers.kv_block_size > value.buffers.maximum_sequence_length
  && value.admission.minimum_median_decode_speedup_ratio > 1
  && value.admission.maximum_ttft_regression_ratio >= 1
  && value.admission.maximum_tail_latency_regression_ratio >= 1
) in
{
  Profile = ProfileC,
  ProfileClosed = ProfileClosedC,
  ProfileSemantic = ProfileSemanticC,
}
