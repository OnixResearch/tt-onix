# Predeclared gates for admitting the resident slot decode trace
# (change qwen38-resident-slot-trace).
#
# r[impl tenstorrent.qwen38_p150x2.resident_slot_trace]
#
# The baseline is the deployed package, which captures a new slot decode trace for every
# continuous-slot group. An agent turn that arrives alone is its own group, so it pays a
# capture of about 0.7-1.3 s between its first and second token. The candidate keeps one
# stable trace across groups of the same width and route. Both serve with the
# continuous-slot scheduler and the prefix cache.
#
# Four roots per workload, each from a fresh process, run in ABBA order: baseline,
# candidate, candidate, baseline. Each timing gate compares, turn by turn, the mean of
# the two candidate roots with the mean of the two baseline roots, which cancels load
# that drifts linearly across the roots. The sampled workload uses fixed seeds, so its
# tokens are comparable too. Fixed before the physical measurement.
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
    # One runtime, device pair, workload and cache profile; one package per role; roots
    # measured in root_order.
    require_matched_identity = true,
    # Every turn decodes exactly the same tokens in all four roots.
    require_exact_parity = true,
    # Every turn resumes and saves the same prefix in all four roots.
    require_matching_prefix_cache = true,
    # At most a quarter of a candidate root's turns capture a trace: the first turn of a
    # process or of a route, not every turn.
    maximum_candidate_capture_share = 0.25,
    # Whole requests must be clearly faster, and no turn much slower.
    maximum_median_wall_ratio = 0.95,
    maximum_wall_ratio = 1.10,
    # A refill beside a resident trace must not slow prefill. On TT-Metal 0.74, eager
    # prefill under a parked trace cost more than the capture it saved.
    maximum_median_ttft_ratio = 1.05,
    # Keeping a trace resident adds no startup allocation.
    maximum_startup_dram_drop_bytes = 67108864,
  },
}
| contracts.Closed
| contracts.Fields
| { gates | contracts.GatesClosed, workloads | Array contracts.WorkloadClosed, .. }
