# r[impl tenstorrent.harness.regression_gate]
# r[impl tenstorrent.harness.determinism]
# Named regression and determinism limits for the dual-P150A measurement profile.
let contracts = import "contracts.ncl" in
{
  schema_version = 1,
  profile_id = "rwkv7-lora-dual-p150a-measurement",
  metric_bounds = [
    {
      metric = "time_seconds.mean",
      direction = "lower_is_better",
      maximum_regression_fraction = 0.12,
    },
    {
      metric = "time_seconds.p99",
      direction = "lower_is_better",
      maximum_regression_fraction = 0.18,
    },
    {
      metric = "loss.mean",
      direction = "lower_is_better",
      maximum_regression_fraction = 0.02,
    },
  ],
  determinism_tolerances = {
    time_seconds = 0.08,
    loss = 0.01,
  },
  hash_algorithm = "blake3",
} | contracts.RegressionPolicy
