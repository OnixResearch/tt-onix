# Negative fixture: an admission that does not require exact parity.
#
# Otherwise identical to prefix-cache-admission-policy.ncl: four roots per workload in
# ABBA order, with the same gate bounds.
let contracts = import "prefix-cache-admission-contracts.ncl" in
{
  schema_version = 2,
  profile_id = "qwen38-p150x2-prefix-cache-admission-v2",
  prefix_cache_profile_id = "qwen38-p150x2-prefix-cache-v1",
  root_order = ["cold", "cached", "cached", "cold"],
  workloads = [
    { conversations = 3, turns = 4 },
    { conversations = 5, turns = 2 },
  ],
  gates = {
    require_matched_identity = true,
    # Every turn decodes exactly the same tokens in all four roots.
    require_exact_parity = false,
    require_resumed_hits = true,
    maximum_hit_median_ttft_ratio = 0.7,
    maximum_hit_median_wall_ratio = 0.85,
    maximum_miss_median_wall_ratio = 1.10,
    maximum_miss_wall_ratio = 1.25,
    maximum_startup_dram_drop_bytes = 536870912,
  },
}
| contracts.Closed
| contracts.Fields
| { gates | contracts.GatesClosed, .. }
