# Evidence policy for bounded window-16 monitoring.
let Contracts = import "persistent-decode-window16-monitoring-contracts.ncl" in
let policy_maximum_batch_events = 256 in
let policy_maximum_event_bytes = 1048576 in
let policy_maximum_timing_seconds = 60 in
let policy_minimum_fallback_samples = 8 in
let policy_maximum_fallback_rate = 0.25 in
let policy | Contracts.MonitoringPolicy = {
  schema_version = 1,
  policy_id = "rwkv7-p150x2-persistent-decode-monitoring-window16-v1",
  maximum_batch_events = policy_maximum_batch_events,
  maximum_event_bytes = policy_maximum_event_bytes,
  maximum_timing_seconds = policy_maximum_timing_seconds,
  minimum_fallback_samples = policy_minimum_fallback_samples,
  maximum_fallback_rate = policy_maximum_fallback_rate,
  admitted_windows = [2, 4, 8, 16],
  critical_action = "disable_persistent_admission",
  warning_action = "investigate_fallback_rate",
}
in
policy
