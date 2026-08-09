# Typed contracts for P150x2 inference-throughput evidence.
# The Rust admission core re-checks these rules before it admits a claim.

let PositiveC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0
) in
let NonNegativeC = std.contract.from_predicate (fun value =>
  std.is_number value && value >= 0
) in
let PositiveIntC = std.contract.from_predicate (fun value =>
  std.is_number value && value > 0 && std.number.is_integer value
) in
let HashHexC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
) in
let ExactC = fun expected => std.contract.from_predicate (fun value => value == expected) in

let WorkloadC = {
  request_count | PositiveIntC,
  short_prompt_token_count | PositiveIntC,
  long_prompt_token_count | PositiveIntC,
  generated_token_count | PositiveIntC,
} in
let SchedulerC = {
  mode | ExactC "deterministic_continuous_batching",
  tie_break | ExactC "least_loaded_card_then_lowest_card_then_lowest_slot",
  per_card_slot_count | PositiveIntC,
} in
let RuntimeC = {
  replica_mode | ExactC "independent_card_replicas",
  recurrent_state_residency | ExactC "device_fp32",
} in
let TimingC = {
  timed_region | ExactC "decode_only",
  warmup_token_count | PositiveIntC,
  exclude_state_readback | ExactC true,
  exclude_profiler_flush | ExactC true,
  exclude_control_replay | ExactC true,
  exclude_artifact_writes | ExactC true,
} in
let AdmissionC = {
  minimum_dual_card_speedup | PositiveC,
  maximum_p99_latency_regression_fraction | NonNegativeC,
  require_physical_hardware | ExactC true,
  require_profiler_attribution | ExactC true,
  machine_fact_source | ExactC "tt-smi -s --snapshot_no_tty",
  one_card_physical_device_id | ExactC 0,
  two_card_physical_device_ids | ExactC [0, 1],
} in
let ThroughputProfileC = {
  schema_version | ExactC 1,
  profile_id | ExactC "rwkv7-p150x2-throughput-v1",
  model_profile_blake3 | HashHexC,
  repeat_count | PositiveIntC,
  workload | WorkloadC,
  scheduler | SchedulerC,
  runtime | RuntimeC,
  timing | TimingC,
  admission | AdmissionC,
} in
let NonEmptyStringC = std.contract.from_predicate (fun value =>
  std.is_string value && std.string.length value > 0
) in
let CapacityCandidateC = {
  id | NonEmptyStringC,
  per_card_slot_count | PositiveIntC,
  maximum_wave_count | PositiveIntC,
  minimum_ab_throughput_gain_fraction | NonNegativeC,
  maximum_p99_latency_regression_fraction | NonNegativeC,
} in
let MemoryBoundsC = {
  fp32_bytes_per_element | ExactC 4,
  bf16_bytes_per_element | ExactC 2,
  temporary_tensor_bytes_per_slot | PositiveIntC,
  device_memory_budget_bytes | PositiveIntC,
  safety_reserve_bytes | PositiveIntC,
} in
let CapacityProfileC = {
  schema_version | ExactC 1,
  profile_id | NonEmptyStringC,
  model_profile_blake3 | HashHexC,
  candidates | Array CapacityCandidateC,
  memory | MemoryBoundsC,
  admission | {
    ab_repeat_count | PositiveIntC,
    minimum_candidate_gain_fraction | NonNegativeC,
  },
} in
{
  ThroughputProfile = ThroughputProfileC,
  CapacityProfile = CapacityProfileC,
  MemoryBounds = MemoryBoundsC,
}
