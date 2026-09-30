# Opt-in P150x2 tool-header screen: five interleaved chats and one changed-schema probe
# per fresh ABBA process. The changed tool keeps its name but alters the description
# inside the saved 2,048-token header, so no cached root may match that probe.
let contracts = import "prefix-cache-admission-contracts.ncl" in
{
  schema_version = 4,
  profile_id = "qwen38-p150x2-prefix-cache-tool-admission-v4",
  prefix_cache_profile_id = "qwen38-p150x2-prefix-cache-root-v2",
  root_order = ["cold", "cached", "cached", "cold"],
  workloads = [{ conversations = 5, turns = 3 }],
  shared_root_minimum_tokens = 2048,
  state_slots = 4,
  workload_id = "qwen38-agent-shared-tools-v1",
  require_changed_tool_miss = true,
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
| contracts.ToolClosed
| contracts.ToolFields
| { gates | contracts.GatesClosed, .. }
