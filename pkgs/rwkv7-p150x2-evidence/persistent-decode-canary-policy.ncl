# Production policy for one bounded persistent decode canary.

let Contracts = import "persistent-decode-canary-contracts.ncl" in
let policy | Contracts.CanaryPolicy = {
  schema_version = 1,
  policy_id = "rwkv7-p150x2-persistent-decode-canary-v2",
  persistent_profile_id = "rwkv7-p150x2-persistent-decode-v3",
  persistent_profile_blake3 = "c5bfa37d83c026bfdb9255da4810bc90af71c7a0e949f05c9c5bf3e709654568",
  monitoring_policy_id = "rwkv7-p150x2-persistent-decode-monitoring-v2",
  admitted_windows = [2, 4, 8],
  physical_device_ids = [0, 1],
  machine_count = 1,
  sample_count = 4,
  requests_per_worker = 4,
  profiler_run_count = 2,
  maximum_request_executions = 104,
  required_candidate_telemetry_events = 12,
  maximum_duration_seconds = 600,
  candidate_traffic_fraction = 0.5,
  minimum_paired_speedup = 1.0,
  primary_thread_cpu_affinities = [2, 6],
  primary_thread_fifo_priority = 1,
  clean_action = "retain",
  warning_action = "hold",
  critical_action = "disable",
} in
policy
