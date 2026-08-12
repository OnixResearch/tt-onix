# Typed policy contract for bounded persistent decode monitoring.

let ExactC = fun expected => std.contract.from_predicate (fun value => value == expected) in
let PositiveIntC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && std.number.is_integer value
) in
let RateC = std.contract.from_predicate (fun value =>
  std.is_number value && value >= 0 && value <= 1
) in
let MonitoringPolicyC = {
  schema_version | ExactC 1,
  policy_id | ExactC "rwkv7-p150x2-persistent-decode-monitoring-v2",
  maximum_batch_events | PositiveIntC,
  maximum_event_bytes | PositiveIntC,
  maximum_timing_seconds | PositiveIntC,
  minimum_fallback_samples | PositiveIntC,
  maximum_fallback_rate | RateC,
  admitted_windows | ExactC [2, 4, 8],
  critical_action | ExactC "disable_persistent_admission",
  warning_action | ExactC "investigate_fallback_rate",
} in
{
  MonitoringPolicy = MonitoringPolicyC,
}
