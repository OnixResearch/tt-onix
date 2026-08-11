# Production policy for bounded persistent decode monitoring.

let Contracts = import "persistent-decode-monitoring-contracts.ncl" in
let policy | Contracts.MonitoringPolicy = {
  schema_version = 1,
  policy_id = "rwkv7-p150x2-persistent-decode-monitoring-v1",
  maximum_batch_events = 256,
  maximum_event_bytes = 1048576,
  maximum_timing_seconds = 60,
  minimum_fallback_samples = 8,
  maximum_fallback_rate = 0.25,
  admitted_windows = [2, 4],
  critical_action = "disable_persistent_admission",
  warning_action = "investigate_fallback_rate",
} in
policy
