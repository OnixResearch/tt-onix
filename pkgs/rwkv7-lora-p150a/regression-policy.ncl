# r[impl tenstorrent.harness.regression_gate]
# r[impl tenstorrent.harness.determinism]
# Named regression and determinism limits for the single-P150A measurement profile.
let contracts = import "contracts.ncl" in
{
  schema_version = 1,
  profile_id = "rwkv7-lora-p150a-single-measurement",
  metric_bounds = [
    {
      metric = "time_seconds.mean",
      direction = "lower_is_better",
      maximum_regression_fraction = 0.10,
    },
    {
      metric = "time_seconds.p99",
      direction = "lower_is_better",
      maximum_regression_fraction = 0.15,
    },
    {
      metric = "loss.mean",
      direction = "lower_is_better",
      maximum_regression_fraction = 0.02,
    },
  ],
  determinism_tolerances = {
    time_seconds = 0.05,
    loss = 0.01,
  },
  hash_algorithm = "blake3",
} | contracts.RegressionPolicy
