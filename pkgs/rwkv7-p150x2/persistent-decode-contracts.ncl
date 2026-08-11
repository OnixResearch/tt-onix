# Typed, fail-closed contracts for the bounded RWKV-7 persistent decode candidate.
let PositiveIntC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && std.number.is_integer value
) in
let NonEmptyStringC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.length value > 0
) in
let Blake3C = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
) in
let ExactC = fun expected => std.contract.from_predicate (fun value =>
  value == expected
) in
let admitted_maximum_decode_steps = 2 in
let admitted_maximum_elapsed_milliseconds = 60000 in
let admitted_queue_depth = 4 in
let admitted_cancellation_poll_interval_steps = 2 in
let admitted_output_capacity_tokens = 2 in
let ClosedRecordC = fun allowed => std.contract.from_predicate (fun value =>
  std.is_record value
  && std.array.length (std.record.fields value) == std.array.length allowed
  && std.array.all (fun field =>
    std.array.elem field allowed
  ) (std.record.fields value)
) in

let LimitsC = {
  maximum_decode_steps | PositiveIntC | ExactC admitted_maximum_decode_steps,
  maximum_elapsed_milliseconds | PositiveIntC | ExactC admitted_maximum_elapsed_milliseconds,
  queue_depth | PositiveIntC | ExactC admitted_queue_depth,
  cancellation_poll_interval_steps | PositiveIntC | ExactC admitted_cancellation_poll_interval_steps,
  output_capacity_tokens | PositiveIntC | ExactC admitted_output_capacity_tokens,
} in
let LimitsClosedC = ClosedRecordC [
  "maximum_decode_steps",
  "maximum_elapsed_milliseconds",
  "queue_depth",
  "cancellation_poll_interval_steps",
  "output_capacity_tokens",
] in
let AdmissionC = {
  require_exact_parity | ExactC true,
  require_simulator_smoke | ExactC true,
  require_zero_per_token_host_submissions | ExactC true,
  require_stable_physical_gain | ExactC true,
} in
let AdmissionClosedC = ClosedRecordC [
  "require_exact_parity",
  "require_simulator_smoke",
  "require_zero_per_token_host_submissions",
  "require_stable_physical_gain",
] in
let ProfileC = {
  schema_version | ExactC 1,
  profile_id | NonEmptyStringC,
  model_profile_blake3 | Blake3C,
  mechanism | ExactC "bounded_metal_trace_chunk",
  physically_admitted | Bool,
  limits | LimitsClosedC | LimitsC,
  admission | AdmissionClosedC | AdmissionC,
} in
let ProfileClosedC = ClosedRecordC [
  "schema_version",
  "profile_id",
  "model_profile_blake3",
  "mechanism",
  "physically_admitted",
  "limits",
  "admission",
] in
let ProfileSemanticC = std.contract.from_predicate (fun value =>
  value.limits.cancellation_poll_interval_steps > 1
  && value.limits.cancellation_poll_interval_steps <= value.limits.maximum_decode_steps
  && value.limits.output_capacity_tokens >= value.limits.maximum_decode_steps
) in
{
  Profile = ProfileC,
  ProfileClosed = ProfileClosedC,
  ProfileSemantic = ProfileSemanticC,
  MaximumDecodeSteps = admitted_maximum_decode_steps,
  MaximumElapsedMilliseconds = admitted_maximum_elapsed_milliseconds,
  QueueDepth = admitted_queue_depth,
  CancellationPollIntervalSteps = admitted_cancellation_poll_interval_steps,
  OutputCapacityTokens = admitted_output_capacity_tokens,
}
