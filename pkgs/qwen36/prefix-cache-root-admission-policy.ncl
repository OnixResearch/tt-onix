# Predeclared ABBA gate for the opt-in shared-system prefix root.
# Five interleaved conversations exceed the four saved states. After the first
# donation, later chats must resume the root; a saved leaf must be evicted and
# its conversation must still resume at the root on the next turn.
let contracts = import "prefix-cache-admission-contracts.ncl" in
{
  schema_version = 3,
  profile_id = "qwen38-p150x2-prefix-cache-root-admission-v3",
  prefix_cache_profile_id = "qwen38-p150x2-prefix-cache-root-v2",
  root_order = ["cold", "cached", "cached", "cold"],
  workloads = [{ conversations = 5, turns = 3 }],
  shared_root_minimum_tokens = 2048,
  state_slots = 4,
  gates = {
    require_matched_identity = true,
    require_exact_parity = true,
    require_resumed_hits = true,
    maximum_hit_median_ttft_ratio = 0.7,
    maximum_hit_median_wall_ratio = 0.85,
    maximum_miss_median_wall_ratio = 1.10,
    maximum_miss_wall_ratio = 1.25,
    maximum_startup_dram_drop_bytes = 536870912,
  },
}
| contracts.SharedClosed
| contracts.SharedFields
| { gates | contracts.GatesClosed, .. }
