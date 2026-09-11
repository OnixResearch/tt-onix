let contracts = import "../persistent-decode-window16-candidate-v2-contracts.ncl" in
let invalid_candidate | contracts.Profile = {
  schema_version = 1,
  profile_id = "rwkv7-p150x2-persistent-decode-window16-candidate-v2",
  predecessor_profile_blake3 = "0fd161f7172276d1efbe51623b1d110947e68de40024109618547cf7a42bb01a",
  diagnostic_receipt_blake3 = "f53b9364e95295188ce1c48eec27014e8fa90d37c1c0bc813b83d578eaca7264",
  model_profile_blake3 = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88",
  mechanism = "bounded_metal_trace_chunk",
  physically_admitted = true,
  limits = {
    maximum_decode_steps = 16,
    maximum_elapsed_milliseconds = 60000,
    queue_depth = 4,
    cancellation_poll_interval_steps = 2,
    output_capacity_tokens = 16,
    admitted_decode_windows = [2, 4, 8, 16],
  },
  admission = {
    require_exact_parity = true,
    require_simulator_smoke = true,
    require_zero_per_token_host_submissions = true,
    require_stable_physical_gain = true,
  },
  precision = {
    fp32_recurrence_vectors = false,
    bf16_readout_boundary = true,
    recurrence_math_fidelity = "HiFi4",
    readout_math_fidelity = "HiFi2",
    correction_owner = "caller_once_after_row_reshape",
  },
}
in
invalid_candidate
