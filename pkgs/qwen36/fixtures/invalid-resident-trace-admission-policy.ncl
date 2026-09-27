# Negative fixture: a resident-trace admission that tolerates a capture on every turn.
#
# Otherwise identical to resident-trace-admission-policy.ncl.
let contracts = import "resident-trace-admission-contracts.ncl" in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-resident-slot-trace-admission-v1",
  prefix_cache_profile_id = "qwen38-p150x2-prefix-cache-v1",
  root_order = ["baseline", "candidate", "candidate", "baseline"],
  workloads = [
    { workload_id = "qwen38-agent-conversations-v1", conversations = 3, turns = 4 },
    { workload_id = "qwen38-agent-conversations-v1", conversations = 5, turns = 2 },
    { workload_id = "qwen38-agent-conversations-sampled-v1", conversations = 3, turns = 4 },
  ],
  gates = {
    require_matched_identity = true,
    require_exact_parity = true,
    require_matching_prefix_cache = true,
    maximum_candidate_capture_share = 1.0,
    maximum_median_wall_ratio = 0.95,
    maximum_wall_ratio = 1.10,
    maximum_median_ttft_ratio = 1.05,
    maximum_startup_dram_drop_bytes = 67108864,
  },
}
| contracts.Closed
| contracts.Fields
| { gates | contracts.GatesClosed, workloads | Array contracts.WorkloadClosed, .. }
