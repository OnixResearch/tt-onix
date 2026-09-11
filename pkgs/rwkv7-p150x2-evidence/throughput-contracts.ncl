# Typed contracts for P150x2 inference-throughput evidence.
# The Rust admission core re-checks these rules before it admits a claim.
let PositiveC =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0
  )
in
let NonNegativeC =
  std.contract.from_predicate (fun value =>
    std.is_number value && value >= 0
  )
in
let PositiveIntC =
  std.contract.from_predicate (fun value =>
    std.is_number value && value > 0 && std.number.is_integer value
  )
in
let HashHexC =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.is_match "^[0-9a-f]{64}$" value
  )
in
let ExactC = fun expected => std.contract.from_predicate (fun value => value == expected) in
let NonEmptyStringC =
  std.contract.from_predicate (fun value =>
    std.is_string value && std.string.length value > 0
  )
in
let throughput_profile_ids = [
  "rwkv7-p150x2-throughput-v1",
  "rwkv7-p150x2-window16-candidate-v2-throughput-v1",
  "rwkv7-p150x2-window16-candidate-v2-canary-throughput-v1",
]
in
let ThroughputProfileIdC =
  std.contract.from_predicate (fun value =>
    std.array.elem value throughput_profile_ids
  )
in

let WorkloadC = {
  request_count | PositiveIntC,
  short_prompt_token_count | PositiveIntC,
  long_prompt_token_count | PositiveIntC,
  generated_token_count | PositiveIntC,
  homogeneous_prompts | Bool,
}
in
let SchedulerC = {
  mode | ExactC "deterministic_continuous_batching",
  tie_break | ExactC "least_loaded_card_then_lowest_card_then_lowest_slot",
  per_card_slot_count | PositiveIntC,
}
in
let RuntimeC = {
  replica_mode | ExactC "independent_card_replicas",
  recurrent_state_residency | ExactC "device_fp32",
}
in
let TimingC = {
  timed_region | ExactC "decode_only",
  warmup_token_count | PositiveIntC,
  exclude_state_readback | ExactC true,
  exclude_profiler_flush | ExactC true,
  exclude_control_replay | ExactC true,
  exclude_artifact_writes | ExactC true,
}
in
let AdmissionC = {
  minimum_dual_card_speedup | PositiveC,
  maximum_p99_latency_regression_fraction | NonNegativeC,
  require_physical_hardware | ExactC true,
  require_profiler_attribution | ExactC true,
  machine_fact_source | ExactC "tt-smi -s --snapshot_no_tty",
  one_card_physical_device_id | ExactC 0,
  two_card_physical_device_ids | ExactC [0, 1],
}
in
let ThroughputProfileC = {
  schema_version | ExactC 1,
  profile_id | ThroughputProfileIdC,
  model_profile_blake3 | HashHexC,
  persistent_profile_id | NonEmptyStringC | optional,
  persistent_profile_blake3 | HashHexC | optional,
  repeat_count | PositiveIntC,
  workload | WorkloadC,
  scheduler | SchedulerC,
  runtime | RuntimeC,
  timing | TimingC,
  admission | AdmissionC,
}
in
let window_sixteen_v2_profile_ids = [
  "rwkv7-p150x2-window16-candidate-v2-throughput-v1",
  "rwkv7-p150x2-window16-candidate-v2-canary-throughput-v1",
]
in
let WindowSixteenV2BindingC =
  std.contract.from_predicate (fun value =>
    if std.array.elem value.profile_id window_sixteen_v2_profile_ids then
      value.persistent_profile_id == "rwkv7-p150x2-persistent-decode-window16-candidate-v2"
      && value.persistent_profile_blake3 == "acd6d609397beee32b6dbe83bd00d042661ce1961c2353dbfd5794870839fee6"
      && value.workload.generated_token_count == 16
      && value.timing.warmup_token_count == 4
      && value.admission.minimum_dual_card_speedup == 1.0
    else
      true
  )
in
let CapacityCandidateC = {
  id | NonEmptyStringC,
  per_card_slot_count | PositiveIntC,
  maximum_wave_count | PositiveIntC,
  minimum_ab_throughput_gain_fraction | NonNegativeC,
  maximum_p99_latency_regression_fraction | NonNegativeC,
}
in
let MemoryBoundsC = {
  fp32_bytes_per_element | ExactC 4,
  bf16_bytes_per_element | ExactC 2,
  temporary_tensor_bytes_per_slot | PositiveIntC,
  device_memory_budget_bytes | PositiveIntC,
  safety_reserve_bytes | PositiveIntC,
}
in
let CapacityProfileC = {
  schema_version | ExactC 1,
  profile_id | NonEmptyStringC,
  model_profile_blake3 | HashHexC,
  candidates | Array CapacityCandidateC,
  memory | MemoryBoundsC,
  admission
    | {
      ab_repeat_count | PositiveIntC,
      minimum_candidate_gain_fraction | NonNegativeC,
    },
}
in
{
  ThroughputProfile = ThroughputProfileC,
  WindowSixteenV2Binding = WindowSixteenV2BindingC,
  CapacityProfile = CapacityProfileC,
  MemoryBounds = MemoryBoundsC,
}
