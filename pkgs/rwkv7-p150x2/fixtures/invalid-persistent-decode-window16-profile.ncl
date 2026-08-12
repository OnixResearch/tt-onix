let contracts = import "../persistent-decode-window16-contracts.ncl" in
let invalid_candidate | contracts.Profile = {
  schema_version = 1,
  profile_id = "rwkv7-p150x2-persistent-decode-window16-candidate-v1",
  model_profile_blake3 = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88",
  mechanism = "bounded_metal_trace_chunk",
  physically_admitted = true,
  limits = {
    maximum_decode_steps = contracts.MaximumDecodeSteps,
    maximum_elapsed_milliseconds = contracts.MaximumElapsedMilliseconds,
    queue_depth = contracts.QueueDepth,
    cancellation_poll_interval_steps = contracts.CancellationPollIntervalSteps,
    output_capacity_tokens = contracts.OutputCapacityTokens,
    admitted_decode_windows = contracts.AdmittedDecodeWindows,
  },
  admission = {
    require_exact_parity = true,
    require_simulator_smoke = true,
    require_zero_per_token_host_submissions = true,
    require_stable_physical_gain = true,
  },
} in
invalid_candidate
