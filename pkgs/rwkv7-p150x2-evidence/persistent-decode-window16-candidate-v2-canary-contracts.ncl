# Typed policy contract for the corrected window-16 canary.
let ExactC = fun expected => std.contract.from_predicate (fun value => value == expected) in
let PositiveC =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0
  )
in
let PositiveIntC =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0 && std.number.is_integer value
  )
in
let RateC =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0 && value <= 1
  )
in
let CanaryPolicyC = {
  schema_version | ExactC 1,
  policy_id | ExactC "rwkv7-p150x2-persistent-decode-canary-window16-v2",
  persistent_profile_id | ExactC "rwkv7-p150x2-persistent-decode-window16-candidate-v2",
  persistent_profile_blake3 | ExactC "acd6d609397beee32b6dbe83bd00d042661ce1961c2353dbfd5794870839fee6",
  monitoring_policy_id | ExactC "rwkv7-p150x2-persistent-decode-monitoring-window16-v2",
  admitted_windows | ExactC [2, 4, 8, 16],
  physical_device_ids | ExactC [0, 1],
  machine_count | ExactC 1,
  sample_count | PositiveIntC,
  requests_per_worker | PositiveIntC,
  profiler_run_count | PositiveIntC,
  maximum_request_executions | PositiveIntC,
  required_candidate_telemetry_events | PositiveIntC,
  maximum_duration_seconds | PositiveIntC,
  candidate_traffic_fraction | RateC,
  minimum_paired_speedup | PositiveC,
  primary_thread_cpu_affinities | ExactC [2, 6],
  primary_thread_fifo_priority | ExactC 1,
  clean_action | ExactC "retain",
  warning_action | ExactC "hold",
  critical_action | ExactC "disable",
}
in
{
  CanaryPolicy = CanaryPolicyC,
}
