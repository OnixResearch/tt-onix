# Typed policy contract for the bounded persistent decode canary.

let ExactC = fun expected => std.contract.from_predicate (fun value => value == expected) in
let PositiveC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0
) in
let PositiveIntC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && std.number.is_integer value
) in
let RateC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && value <= 1
) in
let HashHexC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
) in
let CanaryPolicyC = {
  schema_version | ExactC 1,
  policy_id | ExactC "rwkv7-p150x2-persistent-decode-canary-v2",
  persistent_profile_id | ExactC "rwkv7-p150x2-persistent-decode-v3",
  persistent_profile_blake3 | HashHexC,
  monitoring_policy_id | ExactC "rwkv7-p150x2-persistent-decode-monitoring-v2",
  admitted_windows | ExactC [2, 4, 8],
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
} in
{
  CanaryPolicy = CanaryPolicyC,
}
