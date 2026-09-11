# Typed bounded observation policy contract for corrected window-16 admission.
let ExactC = fun expected => std.contract.from_predicate (fun value => value == expected) in
let PositiveIntC =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0 && std.number.is_integer value
  )
in
let ClosedRecordC = fun allowed =>
  std.contract.from_predicate (fun value =>
    std.is_record value
    && std.array.length (std.record.fields value) == std.array.length allowed
    && std.array.all (fun field => std.array.elem field allowed) (std.record.fields value)
  )
in
let ObservationPolicyC = {
  schema_version | ExactC 1,
  policy_id | ExactC "rwkv7-p150x2-persistent-decode-window16-v2-observation",
  source_candidate_profile_id | ExactC "rwkv7-p150x2-persistent-decode-window16-candidate-v2",
  source_candidate_profile_blake3 | ExactC "acd6d609397beee32b6dbe83bd00d042661ce1961c2353dbfd5794870839fee6",
  expected_production_profile_id | ExactC "rwkv7-p150x2-persistent-decode-v4",
  monitoring_policy_id | ExactC "rwkv7-p150x2-persistent-decode-monitoring-window16-v2",
  admitted_windows | ExactC [2, 4, 8, 16],
  physical_device_ids | ExactC [0, 1],
  selected_decode_windows | ExactC [2, 4, 8, 16],
  sample_count_per_window | PositiveIntC,
  requests_per_sample | PositiveIntC,
  maximum_request_executions | PositiveIntC,
  maximum_duration_seconds | PositiveIntC,
  require_exact_parity | ExactC true,
  require_cleanup_complete | ExactC true,
  require_zero_fallbacks | ExactC true,
  require_zero_alerts | ExactC true,
  require_retain_recommendation | ExactC true,
  require_service_state_restoration | ExactC true,
  require_devices_unowned_after | ExactC true,
  aggregate_only_receipt | ExactC true,
}
in
let ObservationPolicyClosedC =
  ClosedRecordC [
    "schema_version",
    "policy_id",
    "source_candidate_profile_id",
    "source_candidate_profile_blake3",
    "expected_production_profile_id",
    "monitoring_policy_id",
    "admitted_windows",
    "physical_device_ids",
    "selected_decode_windows",
    "sample_count_per_window",
    "requests_per_sample",
    "maximum_request_executions",
    "maximum_duration_seconds",
    "require_exact_parity",
    "require_cleanup_complete",
    "require_zero_fallbacks",
    "require_zero_alerts",
    "require_retain_recommendation",
    "require_service_state_restoration",
    "require_devices_unowned_after",
    "aggregate_only_receipt",
  ]
in
let ObservationPolicySemanticC =
  std.contract.from_predicate (fun value =>
    value.maximum_request_executions == std.array.length value.selected_decode_windows
    * value.sample_count_per_window
    * value.requests_per_sample
  )
in
{
  ObservationPolicy = ObservationPolicyC,
  ObservationPolicyClosed = ObservationPolicyClosedC,
  ObservationPolicySemantic = ObservationPolicySemanticC,
}
