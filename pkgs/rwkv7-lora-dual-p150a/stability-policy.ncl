# r[impl tenstorrent.harness.long_run]
# r[impl tenstorrent.harness.thermal]
# r[impl tenstorrent.harness.heldout_eval]
let contracts = import "contracts.ncl" in
{
  schema_version = 1,
  profile_id = "rwkv7-lora-dual-p150a-stability",
  parity_profile_id = "rwkv7-lora-dual-p150a",
  step_budget = 256,
  trajectory_cadence_steps = 16,
  thermal_cadence_steps = 16,
  maximum_loss = 100.0,
  maximum_gradient_norm = 1000.0,
  maximum_temperature_celsius = 90.0,
  maximum_power_watts = 250.0,
  maximum_eval_loss_ratio = 0.99,
  thermal_source = "tt-smi-jsonl",
  generalization_non_claim = "held-out loss does not prove generalization beyond the bound split",
  hash_algorithm = "blake3",
} | contracts.StabilityPolicy
