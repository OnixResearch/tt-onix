# Evidence policy for one bounded window-8 canary.

let Contracts = import "persistent-decode-window8-canary-contracts.ncl" in
let canary_sample_count = 4 in
let canary_requests_per_worker = 4 in
let canary_profiler_run_count = 2 in
let canary_maximum_request_executions = 104 in
let canary_required_candidate_telemetry_events = 12 in
let canary_maximum_duration_seconds = 600 in
let canary_candidate_traffic_fraction = 0.5 in
let canary_minimum_paired_speedup = 1.0 in
let canary_primary_thread_fifo_priority = 1 in
let policy | Contracts.CanaryPolicy = {
  schema_version = 1,
  policy_id = "rwkv7-p150x2-persistent-decode-canary-window8-v1",
  persistent_profile_id = "rwkv7-p150x2-persistent-decode-window8-candidate-v1",
  persistent_profile_blake3 = "24c07fd0fee91e5e1f290c052f52492680e7edcde30230f3ed91bd5a711102f7",
  monitoring_policy_id = "rwkv7-p150x2-persistent-decode-monitoring-window8-v1",
  admitted_windows = [2, 4, 8],
  physical_device_ids = [0, 1],
  machine_count = 1,
  sample_count = canary_sample_count,
  requests_per_worker = canary_requests_per_worker,
  profiler_run_count = canary_profiler_run_count,
  maximum_request_executions = canary_maximum_request_executions,
  required_candidate_telemetry_events = canary_required_candidate_telemetry_events,
  maximum_duration_seconds = canary_maximum_duration_seconds,
  candidate_traffic_fraction = canary_candidate_traffic_fraction,
  minimum_paired_speedup = canary_minimum_paired_speedup,
  primary_thread_cpu_affinities = [2, 6],
  primary_thread_fifo_priority = canary_primary_thread_fifo_priority,
  clean_action = "retain",
  warning_action = "hold",
  critical_action = "disable",
} in
policy
