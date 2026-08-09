# Typed card-local capacity candidates for RWKV-7 P150x2 decode.
# Nickel owns candidate capacities, memory reserves, and evidence bounds.
# The pure Rust planner consumes the exported values and rejects unsafe plans
# before any device opens. This file does not claim a throughput rate.
let contracts = import "throughput-contracts.ncl" in
let admitted_model_profile_blake3 = "d33ed1051c7595327368cce7b42e16f025e8fc78d96addb0353fba2e86314a88" in
{
  schema_version = 1,
  profile_id = "rwkv7-p150x2-grid-capacity-v1",
  model_profile_blake3 = admitted_model_profile_blake3,
  candidates = [
    {
      id = "B4",
      per_card_slot_count = 4,
      maximum_wave_count = 2,
      minimum_ab_throughput_gain_fraction = 0.0,
      maximum_p99_latency_regression_fraction = 0.10,
    },
    {
      id = "B16",
      per_card_slot_count = 16,
      maximum_wave_count = 2,
      minimum_ab_throughput_gain_fraction = 0.0,
      maximum_p99_latency_regression_fraction = 0.10,
    },
    {
      id = "B32",
      per_card_slot_count = 32,
      maximum_wave_count = 2,
      minimum_ab_throughput_gain_fraction = 0.0,
      maximum_p99_latency_regression_fraction = 0.10,
    },
    {
      id = "B64",
      per_card_slot_count = 64,
      maximum_wave_count = 2,
      minimum_ab_throughput_gain_fraction = 0.0,
      maximum_p99_latency_regression_fraction = 0.10,
    },
  ],
  memory = {
    fp32_bytes_per_element = 4,
    bf16_bytes_per_element = 2,
    temporary_tensor_bytes_per_slot = 5242880,      # 5 MiB per-slot temporary estimate
    device_memory_budget_bytes = 3221225472,        # 3 GiB card-local budget bound
    safety_reserve_bytes = 268435456,               # 256 MiB host-side reserve guard
  },
  admission = {
    ab_repeat_count = 3,
    minimum_candidate_gain_fraction = 0.0,
  },
} | contracts.CapacityProfile