# Physical profile for homogeneous bounded persistent-decode A/B evidence.
let contracts = import "throughput-contracts.ncl" in
let admitted_model_profile_blake3 = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88" in
let admitted_repeat_count = 5 in
let admitted_request_count = 4 in
let admitted_prompt_token_count = 8 in
let admitted_generated_token_count = 8 in
let admitted_per_card_slot_count = 4 in
let admitted_warmup_token_count = 4 in
let admitted_minimum_dual_card_speedup = 1.50 in
let admitted_maximum_p99_latency_regression_fraction = 0.10 in
{
  schema_version = 1,
  profile_id = "rwkv7-p150x2-throughput-v1",
  model_profile_blake3 = admitted_model_profile_blake3,
  repeat_count = admitted_repeat_count,
  workload = {
    request_count = admitted_request_count,
    short_prompt_token_count = admitted_prompt_token_count,
    long_prompt_token_count = admitted_prompt_token_count,
    generated_token_count = admitted_generated_token_count,
    homogeneous_prompts = true,
  },
  scheduler = {
    mode = "deterministic_continuous_batching",
    tie_break = "least_loaded_card_then_lowest_card_then_lowest_slot",
    per_card_slot_count = admitted_per_card_slot_count,
  },
  runtime = {
    replica_mode = "independent_card_replicas",
    recurrent_state_residency = "device_fp32",
  },
  timing = {
    timed_region = "decode_only",
    warmup_token_count = admitted_warmup_token_count,
    exclude_state_readback = true,
    exclude_profiler_flush = true,
    exclude_control_replay = true,
    exclude_artifact_writes = true,
  },
  admission = {
    minimum_dual_card_speedup = admitted_minimum_dual_card_speedup,
    maximum_p99_latency_regression_fraction = admitted_maximum_p99_latency_regression_fraction,
    require_physical_hardware = true,
    require_profiler_attribution = true,
    machine_fact_source = "tt-smi -s --snapshot_no_tty",
    one_card_physical_device_id = 0,
    two_card_physical_device_ids = [0, 1],
  },
} | contracts.ThroughputProfile
