# The capacity profile typed contract must reject a zero-slot candidate.
let contracts = import "../throughput-contracts.ncl" in
let profile = import "../throughput-capacity-profile.ncl" in
(profile & {
  candidates | force = [
    {
      id = "B0",
      per_card_slot_count = 0,
      maximum_wave_count = 2,
      minimum_ab_throughput_gain_fraction = 0.0,
      maximum_p99_latency_regression_fraction = 0.10,
    },
  ],
}) | contracts.CapacityProfile