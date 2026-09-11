# Typed, fail-closed contracts for the corrected window-16 candidate.
let PositiveIntC =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0 && std.number.is_integer value
  )
in
let Blake3C =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
  )
in
let ExactC = fun expected =>
  std.contract.from_predicate (fun value =>
    value == expected
  )
in
let evaluation_decode_window_values = [2, 4, 8, 16] in
let evaluation_maximum_decode_steps = 16 in
let evaluation_maximum_elapsed_milliseconds = 60000 in
let evaluation_queue_depth = 4 in
let evaluation_cancellation_poll_interval_steps = 2 in
let evaluation_output_capacity_tokens = 16 in
let ClosedRecordC = fun allowed =>
  std.contract.from_predicate (fun value =>
    std.is_record value
    && std.array.length (std.record.fields value) == std.array.length allowed
    && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
  )
in
let LimitsC = {
  maximum_decode_steps | PositiveIntC | ExactC evaluation_maximum_decode_steps,
  maximum_elapsed_milliseconds | PositiveIntC | ExactC evaluation_maximum_elapsed_milliseconds,
  queue_depth | PositiveIntC | ExactC evaluation_queue_depth,
  cancellation_poll_interval_steps | PositiveIntC | ExactC evaluation_cancellation_poll_interval_steps,
  output_capacity_tokens | PositiveIntC | ExactC evaluation_output_capacity_tokens,
  admitted_decode_windows | ExactC evaluation_decode_window_values,
}
in
let LimitsClosedC =
  ClosedRecordC [
    "maximum_decode_steps",
    "maximum_elapsed_milliseconds",
    "queue_depth",
    "cancellation_poll_interval_steps",
    "output_capacity_tokens",
    "admitted_decode_windows",
  ]
in
let AdmissionC = {
  require_exact_parity | ExactC true,
  require_simulator_smoke | ExactC true,
  require_zero_per_token_host_submissions | ExactC true,
  require_stable_physical_gain | ExactC true,
}
in
let AdmissionClosedC =
  ClosedRecordC [
    "require_exact_parity",
    "require_simulator_smoke",
    "require_zero_per_token_host_submissions",
    "require_stable_physical_gain",
  ]
in
let PrecisionC = {
  fp32_recurrence_vectors | ExactC true,
  bf16_readout_boundary | ExactC true,
  recurrence_math_fidelity | ExactC "HiFi4",
  readout_math_fidelity | ExactC "HiFi2",
  correction_owner | ExactC "caller_once_after_row_reshape",
}
in
let PrecisionClosedC =
  ClosedRecordC [
    "fp32_recurrence_vectors",
    "bf16_readout_boundary",
    "recurrence_math_fidelity",
    "readout_math_fidelity",
    "correction_owner",
  ]
in
let ProfileC = {
  schema_version | ExactC 1,
  profile_id | ExactC "rwkv7-p150x2-persistent-decode-window16-candidate-v2",
  predecessor_profile_blake3 | ExactC "0fd161f7172276d1efbe51623b1d110947e68de40024109618547cf7a42bb01a",
  diagnostic_receipt_blake3 | ExactC "f53b9364e95295188ce1c48eec27014e8fa90d37c1c0bc813b83d578eaca7264",
  model_profile_blake3 | Blake3C,
  mechanism | ExactC "bounded_metal_trace_chunk",
  physically_admitted | ExactC false,
  limits | LimitsClosedC | LimitsC,
  admission | AdmissionClosedC | AdmissionC,
  precision | PrecisionClosedC | PrecisionC,
}
in
let ProfileClosedC =
  ClosedRecordC [
    "schema_version",
    "profile_id",
    "predecessor_profile_blake3",
    "diagnostic_receipt_blake3",
    "model_profile_blake3",
    "mechanism",
    "physically_admitted",
    "limits",
    "admission",
    "precision",
  ]
in
let ProfileSemanticC =
  std.contract.from_predicate (fun value =>
    value.predecessor_profile_blake3 != value.model_profile_blake3
    && value.limits.cancellation_poll_interval_steps > 1
    && value.limits.cancellation_poll_interval_steps <= value.limits.maximum_decode_steps
    && value.limits.output_capacity_tokens >= value.limits.maximum_decode_steps
    && std.array.length value.limits.admitted_decode_windows == 4
  )
in
{
  Profile = ProfileC,
  ProfileClosed = ProfileClosedC,
  ProfileSemantic = ProfileSemanticC,
  MaximumDecodeSteps = evaluation_maximum_decode_steps,
  MaximumElapsedMilliseconds = evaluation_maximum_elapsed_milliseconds,
  QueueDepth = evaluation_queue_depth,
  CancellationPollIntervalSteps = evaluation_cancellation_poll_interval_steps,
  OutputCapacityTokens = evaluation_output_capacity_tokens,
  AdmittedDecodeWindows = evaluation_decode_window_values,
}
