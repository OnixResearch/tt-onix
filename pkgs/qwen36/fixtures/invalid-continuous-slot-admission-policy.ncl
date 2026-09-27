# Negative fixture: an admission that does not require exact candidate parity.
let contracts = import "continuous-slot-admission-contracts.ncl" in
{
  schema_version = 1,
  profile_id = "qwen38-p150x2-continuous-slot-admission-invalid",
  scheduler_profile_id = "qwen38-p150x2-continuous-slot-b4-v1",
  workload = {
    workload_id = "qwen38-staggered-mixed-length-v1",
    repetitions = 2,
    short_class = "short",
    long_class = "long",
  },
  gates = {
    require_matched_identity = true,
    require_all_completed = true,
    require_exact_candidate_parity = false,
    maximum_short_median_wall_ratio = 0.5,
    maximum_short_p95_wall_ratio = 1.0,
    maximum_long_p95_wall_ratio = 1.3,
    maximum_slowdown_ratio = 1.0,
    maximum_inter_token_gap_seconds = 6.0,
    maximum_median_inter_token_gap_ratio = 1.25,
    minimum_delivered_throughput_ratio = 0.95,
    minimum_startup_dram_free_ratio = 0.99,
  },
}
| contracts.Closed
| contracts.Fields
| { workload | contracts.WorkloadClosed | contracts.WorkloadCoherent, gates | contracts.GatesClosed, .. }
