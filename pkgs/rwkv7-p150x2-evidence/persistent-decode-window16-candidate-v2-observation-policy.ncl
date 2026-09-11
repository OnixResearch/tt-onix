# Bounded aggregate-only observation policy after corrected window-16 admission.
let Contracts = import "persistent-decode-window16-candidate-v2-observation-contracts.ncl" in
let observation_sample_count_per_window = 1 in
let observation_requests_per_sample = 4 in
let observation_maximum_request_executions = 16 in
let observation_maximum_duration_seconds = 600 in
let policy
  | Contracts.ObservationPolicyClosed
  | Contracts.ObservationPolicy
  | Contracts.ObservationPolicySemantic
  = {
    schema_version = 1,
    policy_id = "rwkv7-p150x2-persistent-decode-window16-v2-observation",
    source_candidate_profile_id = "rwkv7-p150x2-persistent-decode-window16-candidate-v2",
    source_candidate_profile_blake3 = "acd6d609397beee32b6dbe83bd00d042661ce1961c2353dbfd5794870839fee6",
    expected_production_profile_id = "rwkv7-p150x2-persistent-decode-v4",
    monitoring_policy_id = "rwkv7-p150x2-persistent-decode-monitoring-window16-v2",
    admitted_windows = [2, 4, 8, 16],
    physical_device_ids = [0, 1],
    selected_decode_windows = [2, 4, 8, 16],
    sample_count_per_window = observation_sample_count_per_window,
    requests_per_sample = observation_requests_per_sample,
    maximum_request_executions = observation_maximum_request_executions,
    maximum_duration_seconds = observation_maximum_duration_seconds,
    require_exact_parity = true,
    require_cleanup_complete = true,
    require_zero_fallbacks = true,
    require_zero_alerts = true,
    require_retain_recommendation = true,
    require_service_state_restoration = true,
    require_devices_unowned_after = true,
    aggregate_only_receipt = true,
  }
in
policy
